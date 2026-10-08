"""Hardware-matched flat-ground velocity task for the Triton legs; 39-D actor.

The policy stands still at zero command and walks slowly on a
[v_right, v_forward, yaw_rate] command. The robot base frame is +X right,
+Y forward, +Z up, so mjlab's base-frame command x is lateral and y is forward;
the forward-only and heading command modes (which assume +X forward) are off.

Two inputs define the hardware:
  velocity_contract.json  joint order, stand pose, action scale, target slew,
                          clips, gains, torque caps, observation layout. Shared
                          with embedded/ctrl_scripts/run_velocity_policy.py.
  sysid_params.json       per-joint armature, friction, damping, kp scale and
                          suggested randomisation ranges, fitted to the
                          one-joint system-ID trials (embedded/utils/sysid_fit.py).
                          Optional: without it the codex-branch estimates are used.

Command path in training matches the robot: action * scale + stand pose ->
clip -> target slew -> torque guard (spring term |kp*(target - pos)| capped at
80/40/11 Nm motor side) -> PD actuator with 0-1 physics step delay.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from hardware_tracking_task import (
  JOINTS,
  SlewLimitedJointPositionAction,
  SlewLimitedJointPositionActionCfg,
  hardware_robot_spec,
  measured_gravity,
  measured_gyro,
  randomize_imu_calibration,
  speed_dependent_torque_limit,
  stalled_torque_penalty,
  torque_margin_penalty,
)

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "velocity_contract.json"
JOINT_TYPES = ("hip1", "hip2", "thigh", "knee", "ankle")
# Codex-branch estimates per joint type, used when no fit is supplied, or when the
# fit flags a value as stuck on its bound (the gantry data could not pin it).
FALLBACK_FIT = {
  "hip1": {"armature": 0.01, "frictionloss": 0.6, "damping": 0.0, "kp_scale": 1.0, "kd_scale": 1.0},
  "hip2": {"armature": 0.01, "frictionloss": 0.5, "damping": 0.0, "kp_scale": 1.0, "kd_scale": 1.0},
  "thigh": {"armature": 0.01, "frictionloss": 0.5, "damping": 0.0, "kp_scale": 1.0, "kd_scale": 1.0},
  "knee": {"armature": 0.01, "frictionloss": 0.6, "damping": 0.0, "kp_scale": 1.0, "kd_scale": 1.0},
  "ankle": {"armature": 0.01, "frictionloss": 0.25, "damping": 0.0, "kp_scale": 1.0, "kd_scale": 1.0},
}
FALLBACK_RANGES = {
  t: {"armature": [0.005, 0.03], "frictionloss": [0.1, 1.0], "kd_scale": [0.5, 1.5]} for t in JOINT_TYPES
}
# Joint damping on top of the actuator's velocity gain; the fit puts the whole
# viscous term into kd_scale, so this stays small.
DAMPING_RANGE = (0.0, 0.1)


def joint_type(joint: str) -> str:
  return joint.split("_")[1]


def load_contract(path: Path = CONTRACT_PATH) -> dict:
  contract = json.loads(Path(path).read_text())
  if tuple(contract["joint_order"]) != JOINTS:
    raise ValueError("velocity_contract.json joint_order differs from the model joint order")
  if sum(term["dim"] for term in contract["actor_obs"]) != contract["actor_obs_dim"]:
    raise ValueError("velocity_contract.json actor_obs dims do not add up")
  return contract


def load_fit(path: Path | None) -> dict | None:
  if path is None:
    return None
  data = json.loads(Path(path).read_text())
  return {
    "joints": {j["joint"]: j for j in data["joints"]},
    "ranges": data.get("suggested_randomisation", {}),
    "source": str(path),
  }


@dataclass(frozen=True)
class HardwareVelocityConfig:
  fit_path: Path | None = None
  # Optional soft gait style: a reference clip (npz with joint_pos [N, 10], sim
  # convention) and the weight of a phase-free pose-similarity reward. 0 = plain PPO.
  style_clip: Path | None = None
  style_weight: float = 0.0
  contract_path: Path = CONTRACT_PATH
  observation_delay_steps: tuple[int, int] = (1, 1)
  # Fitted command delays reach ~8 ms on one knee: 0-2 physics steps (0-10 ms).
  action_delay_physics_steps: tuple[int, int] = (0, 2)
  # The motors' torque replies put the applied kp at ~0.8-0.93x the commanded value.
  gain_scale: tuple[float, float] = (1.0, 1.0)
  randomize: bool = True
  standing_steps: int = 1500 * 24


def forward_command_observation(env, command_name):
  """Public command order [forward, lateral, yaw]; plant uses X right, Y forward."""
  return env.command_manager.get_command(command_name)[:, [1, 0, 2]]


@dataclass(kw_only=True)
class ForwardStartStopCommandCfg(UniformVelocityCommandCfg):
  standing_steps: int = 1500 * 24
  command_slew_m_s2: float = 0.5

  def build(self, env):
    return ForwardStartStopCommand(self, env)


class ForwardStartStopCommand(UniformVelocityCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._desired = torch.zeros_like(self.vel_command_b)
    self._slewed = torch.zeros_like(self.vel_command_b)
    self._moving = torch.arange(self.num_envs, device=self.device) % 2 == 0

  def _resample_command(self, env_ids):
    if hasattr(self, '_external_command'):
      self._desired[env_ids] = self._external_command
      return
    self._moving[env_ids] = ~self._moving[env_ids]
    moving = self._moving[env_ids] & (self._env.common_step_counter >= self.cfg.standing_steps)
    self._desired[env_ids] = 0
    max_speed = 0.1 if self._env.common_step_counter < 2*self.cfg.standing_steps else 0.2
    self._desired[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(0.05, max_speed)*moving
    self.is_standing_env[env_ids] = ~moving

  def compute(self, dt):
    super().compute(dt)
    delta = self._desired-self._slewed
    self._slewed += delta.clamp(-self.cfg.command_slew_m_s2*dt, self.cfg.command_slew_m_s2*dt)
    self.vel_command_b[:] = self._slewed

  def reset(self, env_ids):
    self._slewed[env_ids] = 0
    return super().reset(env_ids)

  def set_external_command(self, forward):
    if not 0 <= forward <= 0.2:
      raise ValueError('Evaluation speed is outside the trained command contract')
    self._external_command = torch.tensor([0., forward, 0.], device=self.device)
    self._desired[:] = self._external_command


def joint_params(fit: dict | None, joint: str, model: str) -> dict:
  """Nominal values for one joint: the fit, except values the fit flagged for
  this joint type as stuck on a bound or untrusted (gantry-confounded hips),
  which keep the fallback."""
  jtype = joint_type(joint)
  params = dict(FALLBACK_FIT[jtype])
  if fit is not None and joint in fit["joints"]:
    fitted = fit["joints"][joint]["fitted"]
    flags = fit["ranges"].get(jtype, {})
    stuck = set(flags.get("at_bound", [])) | set(flags.get("untrusted", []))
    params.update({k: v for k, v in fitted.items() if k not in stuck})
  return params


def guard_span_rad(contract: dict) -> tuple[float, ...]:
  """Joint-space target span the deployed torque guard allows around the
  current position: cap / kp in motor space, divided by the linkage ratio."""
  spans = []
  for joint, model, kp in zip(
    contract["joint_order"], contract["motor_model"], contract["kp_motor"], strict=True
  ):
    ratio = contract["ankle_linkage_ratio"] if "ankle" in joint else 1.0
    spans.append(contract["torque_cap_motor_nm"][model] / kp / ratio)
  return tuple(spans)


def _set_scalar(obj, name: str, value: float) -> None:
  """Joint damping is a float in older MuJoCo and an array (linear term first) in newer."""
  current = getattr(obj, name)
  if hasattr(current, "__len__"):
    current[0] = value
    setattr(obj, name, current)
  else:
    setattr(obj, name, value)


def make_robot_spec_fn(contract: dict, fit: dict | None):
  def spec_fn():
    spec = hardware_robot_spec()
    ratio2 = contract["ankle_linkage_ratio"] ** 2
    for i, (joint, model) in enumerate(zip(JOINTS, contract["motor_model"], strict=True)):
      params = joint_params(fit, joint, model)
      j = spec.joint(joint)
      j.armature = params["armature"]
      j.frictionloss = params["frictionloss"]
      _set_scalar(j, "damping", params.get("damping", 0.0))
      scale = ratio2 if "ankle" in joint else 1.0
      kp = contract["kp_motor"][i] * scale * params.get("kp_scale", 1.0)
      kv = contract["kd_motor"][i] * scale * params.get("kd_scale", 1.0)
      act = spec.actuator(joint.replace("_joint", "_act"))
      act.gainprm[0] = kp
      act.biasprm[1] = -kp
      act.biasprm[2] = -kv
    return spec

  return spec_fn


def stand_base_height(spec_fn, stand_pose: dict) -> float:
  """Base height that puts the feet just on the floor in the stand pose."""
  model = spec_fn().compile()
  data = mujoco.MjData(model)
  for joint, value in stand_pose.items():
    data.qpos[model.jnt_qposadr[model.joint(joint).id]] = value
  mujoco.mj_forward(model, data)
  lowest = min(
    data.geom_xpos[model.geom(name).id][2] - model.geom_size[model.geom(name).id][2]
    for name in ("left_foot_collision_box", "right_foot_collision_box")
  )
  return float(data.qpos[2] - lowest + 0.005)


@dataclass(kw_only=True)
class DeployedJointPositionActionCfg(SlewLimitedJointPositionActionCfg):
  guard_span_rad: tuple[float, ...] = ()

  def build(self, env):
    return DeployedJointPositionAction(self, env)


class DeployedJointPositionAction(SlewLimitedJointPositionAction):
  """clip -> target slew -> torque guard, in the order the robot runner applies them."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._span = torch.tensor(cfg.guard_span_rad, device=self.device, dtype=torch.float32)

  def process_actions(self, actions):
    super().process_actions(actions)  # scale, offset, clip, slew
    if self.cfg.clip is not None:
      self._commanded = torch.clamp(self._commanded, self._clip[:, :, 0], self._clip[:, :, 1])
    pos = self._entity.data.joint_pos[:, self._target_ids]
    sent = pos + (self._commanded - pos).clamp(-self._span, self._span)
    # The runner restarts its rate limiter from what was actually sent.
    self._commanded = sent.clone()
    self._processed_actions = sent


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def pd_gains_per_joint(env, env_ids, kp_range, kd_lo, kd_hi):
  """Scale each joint's kp by U(kp_range) and its kd by U(kd_lo[i], kd_hi[i]),
  relative to the model defaults. mjlab's pd_gains takes one kd range for every
  joint; the fit gives a different one per joint type."""
  robot = env.scene["robot"]
  ctrl_ids = robot.actuators[0].global_ctrl_ids
  ids = (torch.arange(env.num_envs, device=env.device, dtype=torch.int)
         if env_ids is None else env_ids.to(device=env.device, dtype=torch.int))
  n = len(ctrl_ids)
  kp = kp_range[0] + (kp_range[1] - kp_range[0]) * torch.rand(len(ids), n, device=env.device)
  lo = torch.tensor(kd_lo, device=env.device, dtype=torch.float32)
  hi = torch.tensor(kd_hi, device=env.device, dtype=torch.float32)
  kd = lo + (hi - lo) * torch.rand(len(ids), n, device=env.device)
  gain0 = env.sim.get_default_field("actuator_gainprm")
  bias0 = env.sim.get_default_field("actuator_biasprm")
  env.sim.model.actuator_gainprm[ids[:, None], ctrl_ids, 0] = gain0[ctrl_ids, 0] * kp
  env.sim.model.actuator_biasprm[ids[:, None], ctrl_ids, 1] = bias0[ctrl_ids, 1] * kp
  env.sim.model.actuator_biasprm[ids[:, None], ctrl_ids, 2] = bias0[ctrl_ids, 2] * kd


STYLE_JOINTS = (0, 3, 4, 5, 8, 9)   # hip1, knee, ankle on both legs (sagittal plane)


def style_pose_reward(env, clip_path: str, std: float, command_name: str, command_threshold: float):
  """exp(-d^2 / std^2), d = distance from the sagittal joint pose to the nearest
  frame of the reference clip; only while the robot is commanded to move. No
  phase, no discriminator: it shapes the gait without forcing the clip's timing."""
  if not hasattr(env, "_style_clip"):
    import numpy as np

    clip = np.load(clip_path)["joint_pos"][:, STYLE_JOINTS]
    env._style_clip = torch.tensor(clip, device=env.device, dtype=torch.float32)
  q = env.scene["robot"].data.joint_pos[:, STYLE_JOINTS]
  d2 = torch.cdist(q, env._style_clip).min(dim=1).values.square()
  cmd = env.command_manager.get_command(command_name)
  moving = (torch.linalg.vector_norm(cmd[:, :2], dim=1) > command_threshold).float()
  return torch.exp(-d2 / (std * std)) * moving


def make_triton_velocity_cfg(
  hardware: HardwareVelocityConfig | None = None, play: bool = False
) -> ManagerBasedRlEnvCfg:
  hardware = hardware or HardwareVelocityConfig()
  contract = load_contract(hardware.contract_path)
  fit = load_fit(hardware.fit_path)
  stand = dict(contract["stand_pose_rad"])
  spec_fn = make_robot_spec_fn(contract, fit)

  cfg = make_velocity_env_cfg()
  cfg.sim.njmax = 300
  cfg.sim.nconmax = None
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.entities = {
    "robot": EntityCfg(
      spec_fn=spec_fn,
      init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, stand_base_height(spec_fn, stand)),
        joint_pos=stand,
      ),
      articulation=EntityArticulationInfoCfg(
        actuators=(
          XmlActuatorCfg(
            target_names_expr=(".*_joint",),
            delay_min_lag=hardware.action_delay_physics_steps[0],
            delay_max_lag=hardware.action_delay_physics_steps[1],
          ),
        ),
        soft_joint_pos_limit_factor=0.95,
      ),
    )
  }

  # Sensors: no terrain scan on flat ground; feet and self-collision contacts.
  sites = ("left_foot", "right_foot")
  sensors = []
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      continue
    if sensor.name == "foot_height_scan":
      sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in sites)
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)
    sensors.append(sensor)
  sensors.append(
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(mode="subtree", pattern=r"^(left_foot|right_foot)$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=True,
    )
  )
  sensors.append(
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="floating_base", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="floating_base", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    )
  )
  cfg.scene.sensors = tuple(sensors)

  # Action: the deployed command path.
  cfg.actions["joint_pos"] = DeployedJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=contract["action_scale"],
    clip={
      name: (lo, hi)
      for name, lo, hi in zip(
        JOINTS, contract["target_clip_lo"], contract["target_clip_hi"], strict=True
      )
    },
    use_default_offset=True,
    target_speed_rad_s=contract["target_slew_rad_s"],
    guard_span_rad=guard_span_rad(contract),
  )

  # Actor: exactly the 39 values the robot can measure, in contract order.
  lo, hi = hardware.observation_delay_steps
  actor = {
    "base_ang_vel": ObservationTermCfg(
      func=measured_gyro, noise=Unoise(n_min=-0.01, n_max=0.01), delay_min_lag=lo, delay_max_lag=hi
    ),
    "projected_gravity": ObservationTermCfg(
      func=measured_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), delay_min_lag=lo, delay_max_lag=hi
    ),
    "joint_pos_rel": ObservationTermCfg(
      func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), delay_min_lag=lo, delay_max_lag=hi
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.05, n_max=0.05), delay_min_lag=lo, delay_max_lag=hi
    ),
    "last_action": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(func=forward_command_observation, params={"command_name": "twist"}),
  }
  names = [term["name"] for term in contract["actor_obs"]]
  if list(actor) != names:
    raise ValueError(f"actor terms {list(actor)} differ from the contract {names}")
  critic = {
    **{k: ObservationTermCfg(func=v.func, params=dict(v.params)) for k, v in actor.items()},
    "base_lin_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}),
    "foot_height": ObservationTermCfg(func=mdp.foot_height, params={"sensor_name": "foot_height_scan"}),
    "foot_air_time": ObservationTermCfg(func=mdp.foot_air_time, params={"sensor_name": "feet_ground_contact"}),
    "foot_contact": ObservationTermCfg(func=mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces, params={"sensor_name": "feet_ground_contact"}
    ),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(terms=actor, concatenate_terms=True, enable_corruption=not play),
    "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False),
  }

  # Command: base-frame x is lateral, y is forward.
  limits = contract["command_limits"]
  cfg.commands["twist"] = ForwardStartStopCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0, 3.0),
    standing_steps=hardware.standing_steps,
    rel_standing_envs=0.25,
    rel_heading_envs=0.0,
    rel_forward_envs=0.0,
    heading_command=False,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 0.0),
      lin_vel_y=(0.0, 0.2),
      ang_vel_z=(0.0, 0.0),
      heading=None,
    ),
  )
  cfg.commands["twist"].viz.z_offset = 0.8
  cfg.curriculum = {}  # ForwardStartStopCommand owns the standing/forward curriculum.

  # Rewards, sized for commands of a few tenths of a m/s on a 16 kg robot.
  rewards = cfg.rewards
  rewards.pop("angular_momentum")  # needs a subtree momentum sensor this model lacks
  rewards["track_linear_velocity"].params["std"] = 0.2
  rewards["track_angular_velocity"].params["std"] = 0.3
  rewards["upright"].params["asset_cfg"].body_names = ("hip",)
  rewards["body_ang_vel"].params["asset_cfg"].body_names = ("hip",)
  rewards["body_ang_vel"].weight = -0.05
  rewards["pose"].params["std_standing"] = {".*": 0.05}
  rewards["pose"].params["std_walking"] = {
    r".*hip1.*": 0.3,
    r".*hip2.*": 0.1,
    r".*thigh.*": 0.1,
    r".*knee.*": 0.35,
    r".*ankle.*": 0.25,
  }
  rewards["pose"].params["std_running"] = rewards["pose"].params["std_walking"]
  rewards["air_time"].weight = 0.25
  rewards["air_time"].params.update(threshold_min=0.05, threshold_max=0.4, command_threshold=0.05)
  for name in ("foot_clearance", "foot_swing_height"):
    rewards[name].params["target_height"] = 0.05
  for name in ("foot_clearance", "foot_slip"):
    rewards[name].params["asset_cfg"].site_names = sites
  rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": "self_collision", "force_threshold": 10.0},
  )
  if hardware.style_clip is not None and hardware.style_weight > 0.0:
    rewards["style_pose"] = RewardTermCfg(
      func=style_pose_reward,
      weight=hardware.style_weight,
      params={"clip_path": str(hardware.style_clip), "std": 0.3, "command_name": "twist",
              "command_threshold": 0.05},
    )
  rewards["torque_margin"] = RewardTermCfg(func=torque_margin_penalty, weight=-0.02)
  rewards["stalled_torque"] = RewardTermCfg(func=stalled_torque_penalty, weight=-0.02)

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
    ),
    "base_too_low": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum, params={"minimum_height": 0.35}
    ),
  }

  # Events: resets and pushes, then the hardware randomisation.
  events = cfg.events
  events["reset_base"].params["pose_range"] = {
    "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.02), "yaw": (-math.pi, math.pi)
  }
  events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)
  events["push_robot"].interval_range_s = (3.0, 6.0)
  events["push_robot"].params["velocity_range"] = {
    "x": (-0.25, 0.25), "y": (-0.25, 0.25), "z": (0.0, 0.0),
    "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.3, 0.3),
  }
  events["foot_friction"].params["asset_cfg"].geom_names = (
    "left_foot_collision_box", "right_foot_collision_box"
  )
  events["encoder_bias"].params["bias_range"] = (-0.03, 0.03)
  events["base_com"].params["asset_cfg"].body_names = ("hip",)
  events["torque_speed"] = EventTermCfg(mode="step", func=speed_dependent_torque_limit)
  events["torque_speed_reset"] = EventTermCfg(mode="reset", func=speed_dependent_torque_limit)
  if hardware.randomize and not play:
    events["imu_calibration"] = EventTermCfg(mode="reset", func=randomize_imu_calibration)
    ranges = {t: dict(v) for t, v in FALLBACK_RANGES.items()}
    if fit is not None:
      for jtype, values in fit["ranges"].items():
        if jtype in ranges:
          ranges[jtype].update({k: v for k, v in values.items() if k in ("armature", "frictionloss", "kd_scale")})
    # kd: each joint's nominal actuator kv already carries its fitted kd scale, so
    # the range is taken relative to that; kp around the commanded value.
    kd_lo, kd_hi = [], []
    for joint in JOINTS:
      nominal = joint_params(fit, joint, "")["kd_scale"]
      lo_k, hi_k = ranges[joint_type(joint)]["kd_scale"]
      kd_lo.append(lo_k / nominal)
      kd_hi.append(max(hi_k, lo_k + 1e-6) / nominal)
    events["pd_gains"] = EventTermCfg(
      mode="reset",
      func=pd_gains_per_joint,
      params={"kp_range": hardware.gain_scale, "kd_lo": kd_lo, "kd_hi": kd_hi},
    )
    for jtype in JOINT_TYPES:
      pattern = f".*{jtype}_joint"
      for field, func, rng in (
        ("frictionloss", dr.joint_friction, ranges[jtype]["frictionloss"]),
        ("armature", dr.joint_armature, ranges[jtype]["armature"]),
        ("damping", dr.joint_damping, DAMPING_RANGE),
      ):
        lo_v, hi_v = rng
        events[f"{field}_{jtype}"] = EventTermCfg(
          mode="reset",
          func=func,
          params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=pattern),
            "ranges": (float(lo_v), float(max(hi_v, lo_v + 1e-6))),
            "operation": "abs",
          },
        )
    events["mass_inertia"] = EventTermCfg(
      mode="reset",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names="hip|.*leg[1-4]|.*foot"),
        "alpha_range": (-0.0527, 0.0477),
      },
    )
  if play:
    cfg.episode_length_s = int(1e9)
    events.pop("push_robot", None)
    cfg.curriculum = {}

  # Only fitted actuator uncertainty in this first experiment. Contact, COM,
  # sensor noise and hip parameters have not been independently identified.
  for name in ("push_robot", "foot_friction", "encoder_bias", "base_com", "mass_inertia", "imu_calibration"):
    events.pop(name, None)
  for name in list(events):
    if any(name == f"{field}_{kind}" for field in ("armature", "frictionloss", "damping") for kind in ("hip1", "hip2")):
        events.pop(name, None)
    if name.startswith("damping_"):
      events.pop(name, None)
  if "pd_gains" in events:
    for i, joint in enumerate(JOINTS):
      if joint_type(joint) in ("hip1", "hip2"):
        events["pd_gains"].params["kd_lo"][i] = 1.0
        events["pd_gains"].params["kd_hi"][i] = 1.0
  for term in actor.values():
    term.noise = None  # Noise estimates await connected sensor validation.
  cfg.viewer.body_name = "hip"
  return cfg
