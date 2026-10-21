"""Velocity-command locomotion for the Triton lower body.

This is the joystick/controller-trained counterpart to the reference-tracking
task.  It keeps the hardware contract of ``hardware_tracking_task`` (same MJCF,
same position actuators, same slew cap, torque-speed envelope, observation
delay and sensor noise) but replaces the motion reference with a
``(vx, vy, yaw_rate)`` command, so the policy can be driven by a gamepad or a
teleoperation loop.

Observations are deliberately *deployable*: the actor group never sees base
linear velocity (the robot has no state estimator), only IMU angular velocity,
projected gravity, joint state, last action and the command.  The critic keeps
privileged terms.
"""

import math

import torch

from hardware_tracking_task import (
  JOINTS,
  TARGET_CLIP_HI,
  TARGET_CLIP_LO,
  HardwareTrackingConfig,
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
from mjlab.envs.mdp import dr, push_by_setting_velocity
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
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
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# Neutral stance: slight hip pitch, knee flexion and ankle compensation so the
# sole is horizontal. Computed by forward kinematics on triton_lower_body_actuated.
# (base height 0.698970 m puts both collision boxes on the ground).
STANCE_JOINT_POS = {
  r"(left|right)_hip1_joint": 0.05,
  r"(left|right)_hip2_joint": 0.0,
  r"(left|right)_thigh_joint": 0.0,
  r"(left|right)_knee_joint": -0.35,
  r"(left|right)_ankle_joint": 0.30,
}
STANCE_BASE_HEIGHT = 0.699

# The Triton lower-body MJCF uses **x lateral and y forward**: the hip roll
# axes are (0,1,0), the knee/ankle pitch axes are (1,0,0), and the feet sit at
# x = +-0.15 m, y ~ 0.  The deployed walking reference confirms it: the feet
# swing 0.47 m along y and the root travels 3.35 m along y while moving only
# 0.15 m in x.  Command forward velocity on the body **y** axis, and keep the
# body x range small; commanding x asks the robot to sidestep across its 31 cm
# stance, which PPO correctly refuses to do.
FORWARD_AXIS = "y"

FOOT_SITES = ("left_foot", "right_foot")
FOOT_GEOMS = ("left_foot_collision_box", "right_foot_collision_box")

# How much a joint may deviate from the neutral stance in each speed regime.
STD_STANDING = {".*": 0.05}
STD_WALKING = {
  r".*hip1.*": 0.30,
  r".*hip2.*": 0.15,
  r".*thigh.*": 0.15,
  r".*knee.*": 0.35,
  r".*ankle.*": 0.25,
}
STD_RUNNING = {
  r".*hip1.*": 0.45,
  r".*hip2.*": 0.20,
  r".*thigh.*": 0.20,
  r".*knee.*": 0.55,
  r".*ankle.*": 0.35,
}


##
# Gait phase clock.
#
# Standing still is a strong local optimum for a 10-DoF leg-only robot: it
# collects upright, posture and survival reward with zero risk of falling, and
# velocity tracking alone rarely pays for the first step.  A phase clock with a
# foot-height reference (the ``feet_phase`` trick from the old brax joystick
# task) gives a dense, achievable reward for stepping, which is a proto-motion
# style prior rather than a fixed reference clip.  The deployed controller must
# run the same clock; at 1.4 Hz the required joint speed matches the 3 rad/s
# slew cap this task is trained with.
##

GAIT_FREQUENCY = 1.4


def swing_profile(phase: torch.Tensor, height: float) -> torch.Tensor:
  """Smooth lift-and-land foot height reference, 0..height, per phase."""

  def bezier(start: torch.Tensor, end: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t = t.clamp(0.0, 1.0)
    return start + (end - start) * (t**3 + 3.0 * (t**2) * (1.0 - t))

  x = torch.remainder(phase + math.pi, 2.0 * math.pi) / (2.0 * math.pi)
  stance = bezier(torch.zeros_like(x), torch.full_like(x, height), 2.0 * x)
  swing = bezier(torch.full_like(x, height), torch.zeros_like(x), 2.0 * x - 1.0)
  return torch.where(x <= 0.5, stance, swing)


class PhaseClock:
  """Per-environment gait phase shared by the velocity and soccer tasks."""

  def __init__(self, num_envs: int, device, frequency: float, jitter: float):
    self.phase = torch.zeros(num_envs, device=device)
    self.frequency = torch.full((num_envs,), float(frequency), device=device)
    self.base_frequency = float(frequency)
    self.jitter = float(jitter)

  @property
  def phase_obs(self) -> torch.Tensor:
    return torch.stack((torch.cos(self.phase), torch.sin(self.phase)), dim=-1)

  def reset(self, env_ids) -> None:
    if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
      self.phase[env_ids] = torch.rand(len(env_ids), device=self.phase.device) * 2.0 * math.pi

  def resample_frequency(self, env_ids) -> None:
    if not isinstance(env_ids, torch.Tensor) or len(env_ids) == 0:
      return
    count = len(env_ids)
    jitter = (torch.rand(count, device=self.phase.device) * 2.0 - 1.0) * self.jitter
    self.frequency[env_ids] = self.base_frequency + jitter

  def advance(self, dt: float) -> None:
    self.phase = torch.remainder(
      self.phase + 2.0 * math.pi * self.frequency * dt, 2.0 * math.pi
    )


class PhaseVelocityCommandCfg(UniformVelocityCommandCfg):
  gait_frequency: float = GAIT_FREQUENCY
  frequency_jitter: float = 0.1

  def build(self, env):
    return PhaseVelocityCommand(self, env)


class PhaseVelocityCommand(UniformVelocityCommand):
  cfg: PhaseVelocityCommandCfg

  def __init__(self, cfg: PhaseVelocityCommandCfg, env):
    super().__init__(cfg, env)
    self.clock = PhaseClock(
      self.num_envs, self.device, cfg.gait_frequency, cfg.frequency_jitter
    )

  @property
  def phase(self) -> torch.Tensor:
    return self.clock.phase

  @property
  def phase_obs(self) -> torch.Tensor:
    return self.clock.phase_obs

  def reset(self, env_ids=None):
    self.clock.reset(env_ids)
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self.clock.resample_frequency(env_ids)

  def compute(self, dt: float) -> None:
    super().compute(dt)
    self.clock.advance(dt)


def phase_observation(env, command_name: str = "twist") -> torch.Tensor:
  term = env.command_manager.get_term(command_name)
  return term.phase_obs


def feet_phase_reward(
  env,
  command_name: str = "twist",
  swing_height: float = 0.08,
  sigma: float = 0.01,
  command_threshold: float = 0.15,
) -> torch.Tensor:
  """Reward feet following the clock's lift-and-land profile."""
  term = env.command_manager.get_term(command_name)
  robot = env.scene["robot"]
  site_ids = [robot.site_names.index(name) for name in FOOT_SITES]
  foot_z = robot.data.site_pos_w[:, site_ids, 2]
  target = torch.stack(
    (swing_profile(term.phase, swing_height), swing_profile(term.phase + math.pi, swing_height)),
    dim=-1,
  )
  reward = torch.exp(-torch.mean(torch.square(foot_z - target), dim=-1) / sigma)
  command = term.command
  moving = (torch.norm(command[:, :2], dim=-1) + command[:, 2].abs()) > command_threshold
  return reward * moving.float()


def make_triton_velocity_cfg(
  hardware: HardwareTrackingConfig | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Flat-ground velocity task with the deployed hardware model."""
  if hardware is None:
    hardware = HardwareTrackingConfig()
  cfg = make_velocity_env_cfg()

  ##
  # Robot and scene.
  ##
  cfg.scene.entities = {
    "robot": EntityCfg(
      spec_fn=hardware_robot_spec,
      init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, STANCE_BASE_HEIGHT),
        joint_pos=dict(STANCE_JOINT_POS),
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

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # No height scan: the task is flat ground, and the deploy path has no
  # exteroception either.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  for sensor in cfg.scene.sensors:
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree", pattern=r"^(left_foot|right_foot)$", entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="hip", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="hip", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, self_collision_cfg)

  ##
  # Actions: identical to the deployed contract, including the target slew cap.
  ##
  cfg.actions["joint_pos"] = SlewLimitedJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=0.2,
    clip={name: (low, high) for name, low, high in zip(JOINTS, TARGET_CLIP_LO, TARGET_CLIP_HI)},
    use_default_offset=True,
    target_speed_rad_s=hardware.target_speed_rad_s,
    software_torque_caps=hardware.software_torque_caps,
  )

  ##
  # Observations.
  ##
  actor = {
    "base_ang_vel": ObservationTermCfg(func=measured_gyro),
    "gravity": ObservationTermCfg(func=measured_gravity, noise=Unoise(-0.05, 0.05)),
    "joint_pos": ObservationTermCfg(
      func=velocity_mdp.joint_pos_rel, noise=Unoise(-0.015, 0.015)
    ),
    "joint_vel": ObservationTermCfg(
      func=velocity_mdp.joint_vel_rel, noise=Unoise(-1.5, 1.5)
    ),
    "actions": ObservationTermCfg(func=velocity_mdp.last_action),
    "command": ObservationTermCfg(
      func=velocity_mdp.generated_commands, params={"command_name": "twist"}
    ),
    "gait_phase": ObservationTermCfg(
      func=phase_observation, params={"command_name": "twist"}
    ),
  }
  for name in ("base_ang_vel", "gravity", "joint_pos", "joint_vel"):
    actor[name].delay_min_lag = hardware.observation_delay_steps[0]
    actor[name].delay_max_lag = hardware.observation_delay_steps[1]

  critic = dict(actor)
  critic["base_lin_vel"] = ObservationTermCfg(
    func=velocity_mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_vel"},
  )
  critic["foot_air_time"] = ObservationTermCfg(
    func=velocity_mdp.foot_air_time,
    params={"sensor_name": feet_ground_cfg.name},
  )
  critic["foot_contact"] = ObservationTermCfg(
    func=velocity_mdp.foot_contact,
    params={"sensor_name": feet_ground_cfg.name},
  )
  critic["foot_contact_forces"] = ObservationTermCfg(
    func=velocity_mdp.foot_contact_forces,
    params={"sensor_name": feet_ground_cfg.name},
  )

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=critic, concatenate_terms=True, enable_corruption=False
    ),
  }

  ##
  # Command.
  ##
  cfg.commands["twist"] = PhaseVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rel_standing_envs=0.1,
    # `heading` is defined on the body x axis in mjlab; that axis is lateral
    # here, so a heading controller would fight the walking direction.  Use
    # direct yaw-rate commands instead.
    rel_heading_envs=0.0,
    rel_forward_envs=0.0,
    heading_command=False,
    debug_vis=True,
    viz=PhaseVelocityCommandCfg.VizCfg(z_offset=0.9),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.25, 0.25),  # lateral
      lin_vel_y=(-0.6, 0.9),  # forward
      ang_vel_z=(-0.6, 0.6),
      heading=None,
    ),
  )

  ##
  # Events: hardware domain randomisation from the tracking task plus velocity
  # task standards.
  ##
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "z": (0.0, 0.01),
    "yaw": (-3.14, 3.14),
  }
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("hip",)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "z": (0.0, 0.0),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2, 0.2),
    "yaw": (-0.3, 0.3),
  }
  cfg.events["push_robot"].interval_range_s = (2.0, 5.0)

  cfg.events["torque_speed"] = EventTermCfg(
    mode="step",
    func=speed_dependent_torque_limit,
    params={"software_torque_caps": hardware.software_torque_caps},
  )
  cfg.events["torque_speed_reset"] = EventTermCfg(
    mode="reset",
    func=speed_dependent_torque_limit,
    params={"software_torque_caps": hardware.software_torque_caps},
  )
  if hardware.randomize:
    cfg.events["imu_calibration"] = EventTermCfg(
      mode="reset", func=randomize_imu_calibration
    )
    cfg.events["pd_gains"] = EventTermCfg(
      mode="reset",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
        "kp_range": hardware.gain_scale,
        "kd_range": hardware.damping_gain_scale,
      },
    )
    for part, pattern, friction in (
      ("rs04", r".*(hip1|knee)_joint", (0.3, 1.0)),
      ("rs03", r".*(hip2|thigh)_joint", (0.2, 0.8)),
      ("rs02", r".*ankle_joint", (0.1, 0.4)),
    ):
      cfg.events[f"friction_{part}"] = EventTermCfg(
        mode="reset",
        func=dr.joint_friction,
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=pattern),
          "ranges": friction,
          "operation": "abs",
        },
      )
    cfg.events["armature"] = EventTermCfg(
      mode="reset",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=r".*_joint"),
        "ranges": (0.5, 2.0),
        "operation": "scale",
      },
    )
    cfg.events["mass_inertia"] = EventTermCfg(
      mode="reset",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=r"hip|.*leg[1-4]|.*foot"
        ),
        "alpha_range": (-0.0527, 0.0477),
      },
    )
    cfg.events["base_com_offset"] = EventTermCfg(
      mode="reset",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names="hip"),
        "operation": "add",
        "ranges": {0: (-0.02, 0.02), 1: (-0.02, 0.02), 2: (-0.02, 0.02)},
      },
    )

  ##
  # Rewards.
  ##
  rewards = cfg.rewards
  rewards["pose"].params.update(
    std_standing=STD_STANDING,
    std_walking=STD_WALKING,
    std_running=STD_RUNNING,
    walking_threshold=0.15,
    running_threshold=0.9,
  )
  rewards["pose"].params["asset_cfg"].joint_names = (r".*_joint",)
  rewards["upright"].params["asset_cfg"].body_names = ("hip",)
  rewards["body_ang_vel"].params["asset_cfg"].body_names = ("hip",)
  rewards["body_ang_vel"].weight = -0.05
  rewards["angular_momentum"].weight = 0.0  # No root angmom sensor on this robot.
  # Stepping incentives have to outweigh the comfort of standing: standing still
  # is the dominant local optimum for a 10-DoF leg-only robot.  A measured
  # reward probe (zero action vs replaying the reference clip for 150 steps at a
  # 0.55 m/s command) gives 0.0007/step standing against 0.0210/step walking, so
  # the ordering is right; the weights below widen that margin so PPO does not
  # drift back into standing after a warm start.
  rewards["pose"].weight = 0.4
  rewards["upright"].weight = 0.6
  rewards["track_linear_velocity"].weight = 3.0
  rewards["air_time"].weight = 1.0
  rewards["air_time"].params["threshold_min"] = 0.1
  rewards["air_time"].params["threshold_max"] = 0.45
  rewards["air_time"].params["command_threshold"] = 0.1
  rewards["foot_clearance"].weight = -1.0
  rewards["foot_clearance"].params["asset_cfg"].site_names = FOOT_SITES
  rewards["foot_clearance"].params["target_height"] = 0.08
  rewards["foot_swing_height"].weight = -0.1
  rewards["foot_swing_height"].params["target_height"] = 0.08
  rewards["foot_slip"].params["asset_cfg"].site_names = FOOT_SITES
  rewards["track_linear_velocity"].params["std"] = math.sqrt(0.25)
  rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)
  rewards["feet_phase"] = RewardTermCfg(
    func=feet_phase_reward,
    weight=4.0,
    params={
      "command_name": "twist",
      "swing_height": 0.08,
      "sigma": 0.01,
      "command_threshold": 0.15,
    },
  )
  rewards["self_collisions"] = RewardTermCfg(
    func=velocity_mdp.self_collision_cost,
    weight=-0.5,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  # Hardware-specific effort shaping, identical to the tracking task.
  rewards["torque_margin"] = RewardTermCfg(
    func=torque_margin_penalty,
    weight=-0.02,
    params={"limits": hardware.software_torque_caps},
  )
  rewards["stalled_torque"] = RewardTermCfg(
    func=stalled_torque_penalty, weight=-0.02
  )

  ##
  # Terminations.
  ##
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(55.0)
  cfg.terminations["low_base"] = TerminationTermCfg(
    func=velocity_mdp.root_height_below_minimum,
    params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("robot")},
  )

  ##
  # Curriculum: widen the command range as the policy survives.
  ##
  # Start wide: a small command range makes "stand still" almost free
  # (exp(-0.3^2 / 0.25) = 0.70 of the tracking reward), so PPO never leaves the
  # standing optimum.  ±0.6-0.8 m/s costs standing most of its tracking reward.
  cfg.curriculum["command_vel"].params = {
    "command_name": "twist",
    "velocity_stages": [
      {"step": 0, "lin_vel_x": (-0.25, 0.25), "lin_vel_y": (-0.6, 0.9), "ang_vel_z": (-0.6, 0.6)},
      {"step": 600 * 24, "lin_vel_x": (-0.35, 0.35), "lin_vel_y": (-0.9, 1.2), "ang_vel_z": (-0.8, 0.8)},
      {"step": 1800 * 24, "lin_vel_x": (-0.45, 0.45), "lin_vel_y": (-1.2, 1.5), "ang_vel_z": (-1.0, 1.0)},
    ],
  }

  cfg.metrics = {"mean_action_acc": MetricsTermCfg(func=velocity_mdp.mean_action_acc)}

  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="hip",
    distance=2.5,
    elevation=-10.0,
    azimuth=90.0,
  )

  cfg.decimation = 4  # 50 Hz control at a 5 ms physics step, like the robot.
  cfg.episode_length_s = 20.0
  cfg.scene.num_envs = 1

  if play:
    cfg.episode_length_s = 1e9
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist = cfg.commands["twist"]
    assert isinstance(twist, PhaseVelocityCommandCfg)
    twist.ranges.lin_vel_x = (-0.45, 0.45)
    twist.ranges.lin_vel_y = (-1.2, 1.5)
    twist.ranges.ang_vel_z = (-1.0, 1.0)

  return cfg
