"""Hardware-constrained leg tracking task; keeps the 56-D policy interface."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch
from tracking_task import make_leg_tracking_cfg

from mjlab.actuator import XmlActuatorCfg
from mjlab.envs.mdp import dr, push_by_setting_velocity
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.envs.mdp.observations import projected_gravity
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking import mdp
from mjlab.utils.noise import UniformNoiseCfg

JOINTS = tuple(
  f"{side}_{part}_joint"
  for side in ("left", "right")
  for part in ("hip1", "hip2", "thigh", "knee", "ankle")
)
PEAK_TORQUE = (120.0, 60.0, 60.0, 120.0, 16.0) * 2
STALLED_TORQUE = (28.5, 15.0, 15.0, 28.5, 6.0) * 2
TARGET_CLIP_LO = (-1.4915, -0.3861737, -0.7461281, -2.04204, -0.57) * 2
TARGET_CLIP_HI = (1.4915, 0.5, 0.7461281, -0.05236, 0.57) * 2
TORQUE_SPEED_CURVES = {
  "rs04": ((0, 95, 110, 130, 150, 175, 190), (120, 120, 110, 90, 80, 50, 0)),
  "rs03": ((0, 145, 160, 170, 185, 195, 200), (60, 60, 52, 44, 30, 20, 0)),
  "rs02_joint": ((0, 165, 255, 345, 420), (16, 16, 11.3, 5.7, 0)),
}


@dataclass(frozen=True)
class HardwareTrackingConfig:
  # This is the deployed policy_bridge target slew limit, not motor no-load speed.
  target_speed_rad_s: float = 1.0
  observation_delay_steps: tuple[int, int] = (0, 1)
  action_delay_physics_steps: tuple[int, int] = (0, 1)
  gain_scale: tuple[float, float] = (0.8, 1.2)
  damping_gain_scale: tuple[float, float] = (0.7, 1.5)
  randomize: bool = True


@dataclass(kw_only=True)
class SlewLimitedJointPositionActionCfg(JointPositionActionCfg):
  target_speed_rad_s: float = 1.0

  def build(self, env):
    return SlewLimitedJointPositionAction(self, env)


class SlewLimitedJointPositionAction(JointPositionAction):
  """Replicate the deployed position-target slew cap after action scaling."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._step_dt = env.step_dt
    self._commanded = self._entity.data.default_joint_pos[:, self._target_ids].clone()
    self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  def process_actions(self, actions):
    super().process_actions(actions)
    if not self._initialized.all():
      ids = ~self._initialized
      self._commanded[ids] = self._entity.data.joint_pos[ids][:, self._target_ids]
      self._initialized[ids] = True
    delta = self._processed_actions - self._commanded
    max_step = self.cfg.target_speed_rad_s * self._step_dt
    self._commanded += delta.clamp(-max_step, max_step)
    self._processed_actions = self._commanded.clone()

  def reset(self, env_ids=None):
    super().reset(env_ids)
    self._initialized[env_ids] = False


def hardware_robot_spec():
  spec = mujoco.MjSpec.from_file(
    str(Path(__file__).with_name("chrobot_hardware_actuated.xml"))
  )
  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
  )
  spec.add_sensor(
    name="imu_lin_vel",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
  )
  return spec


def torque_margin_penalty(env):
  effort = env.scene["robot"].data.actuator_force.abs()
  limits = torch.tensor(PEAK_TORQUE, device=effort.device)
  return torch.square(torch.relu(effort / limits - 0.8)).sum(dim=1)


def stalled_torque_penalty(env):
  effort = env.scene["robot"].data.actuator_force.abs()
  limits = torch.tensor(STALLED_TORQUE, device=effort.device)
  return torch.square(torch.relu(effort / limits - 1.0)).sum(dim=1)


def randomize_imu_calibration(env, env_ids):
  if not hasattr(env, "hardware_imu_tilt"):
    env.hardware_imu_tilt = torch.zeros(env.num_envs, 3, device=env.device)
    env.hardware_gyro_bias = torch.zeros_like(env.hardware_imu_tilt)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  count = len(env_ids)
  tilt = (torch.rand(count, 2, device=env.device) * 2 - 1) * 0.0349066
  env.hardware_imu_tilt[env_ids, :2] = tilt
  env.hardware_gyro_bias[env_ids] = (
    torch.rand(count, 3, device=env.device) * 2 - 1
  ) * 0.05


def measured_gyro(env):
  gyro = mdp.builtin_sensor(env, sensor_name="robot/imu_ang_vel")
  if hasattr(env, "hardware_imu_tilt"):
    gyro = gyro + torch.cross(env.hardware_imu_tilt, gyro, dim=1)
    gyro = gyro + env.hardware_gyro_bias
  return gyro


def measured_gravity(env):
  gravity = projected_gravity(env)
  if hasattr(env, "hardware_imu_tilt"):
    gravity = gravity + torch.cross(env.hardware_imu_tilt, gravity, dim=1)
  return gravity


def _linear_curve(speed_rpm, knots, values):
  x = torch.tensor(knots, device=speed_rpm.device, dtype=speed_rpm.dtype)
  y = torch.tensor(values, device=speed_rpm.device, dtype=speed_rpm.dtype)
  idx = torch.searchsorted(x, speed_rpm.contiguous()).clamp(1, len(knots) - 1)
  left = idx - 1
  ratio = ((speed_rpm - x[left]) / (x[idx] - x[left])).clamp(0, 1)
  return (y[left] + ratio * (y[idx] - y[left])).clamp_min(0)


@requires_model_fields("actuator_forcerange")
def speed_dependent_torque_limit(env, env_ids, voltage_speed_scale=0.9):
  """Apply a 42 V approximation to the provided 48 V torque-speed curves."""
  robot = env.scene["robot"]
  speed_rpm = robot.data.joint_vel.abs() * (60.0 / (2.0 * torch.pi))
  speed_at_48v = speed_rpm / voltage_speed_scale
  limits = torch.empty_like(speed_at_48v)
  for joint_idx, motor in enumerate(("rs04", "rs03", "rs03", "rs04", "rs02_joint") * 2):
    motor_speed = speed_at_48v[:, joint_idx]
    if motor == "rs02_joint":
      motor_speed = motor_speed * 0.95
    limits[:, joint_idx] = _linear_curve(motor_speed, *TORQUE_SPEED_CURVES[motor])
  ctrl_ids = robot.actuators[0].global_ctrl_ids
  ids = (
    torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    if env_ids is None
    else env_ids.to(device=env.device, dtype=torch.int)
  )
  env.sim.model.actuator_forcerange[ids[:, None], ctrl_ids, 0] = -limits[ids]
  env.sim.model.actuator_forcerange[ids[:, None], ctrl_ids, 1] = limits[ids]


def make_hardware_leg_tracking_cfg(motion_file, hardware=None):
  if hardware is None:
    hardware = HardwareTrackingConfig()
  cfg = make_leg_tracking_cfg(motion_file)
  cfg.scene.entities["robot"].spec_fn = hardware_robot_spec
  cfg.scene.entities["robot"].articulation.actuators = (
    XmlActuatorCfg(
      target_names_expr=(".*_joint",),
      delay_min_lag=hardware.action_delay_physics_steps[0],
      delay_max_lag=hardware.action_delay_physics_steps[1],
    ),
  )
  cfg.actions["joint_pos"] = SlewLimitedJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=0.2,
    clip={
      name: (low, high)
      for name, low, high in zip(JOINTS, TARGET_CLIP_LO, TARGET_CLIP_HI, strict=True)
    },
    use_default_offset=True,
    target_speed_rad_s=hardware.target_speed_rad_s,
  )
  actor = cfg.observations["actor"].terms
  actor["base_ang_vel"].func = measured_gyro
  actor["base_ang_vel"].params = {}
  actor["gravity"].func = measured_gravity
  for name in ("base_ang_vel", "joint_pos", "joint_vel", "gravity"):
    actor[name].delay_min_lag = hardware.observation_delay_steps[0]
    actor[name].delay_max_lag = hardware.observation_delay_steps[1]
  actor["gravity"].noise = UniformNoiseCfg(n_min=-0.05, n_max=0.05)
  cfg.rewards["torque_margin"] = RewardTermCfg(func=torque_margin_penalty, weight=-0.02)
  cfg.rewards["stalled_torque"] = RewardTermCfg(
    func=stalled_torque_penalty, weight=-0.02
  )
  cfg.events["torque_speed"] = EventTermCfg(
    mode="step",
    func=speed_dependent_torque_limit,
  )
  cfg.events["torque_speed_reset"] = EventTermCfg(
    mode="reset",
    func=speed_dependent_torque_limit,
  )
  if hardware.randomize:
    cfg.events["imu_calibration"] = EventTermCfg(
      mode="reset",
      func=randomize_imu_calibration,
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
      ("rs04", ".*(hip1|knee)_joint", (0.3, 1.0)),
      ("rs03", ".*(hip2|thigh)_joint", (0.2, 0.8)),
      ("rs02", ".*ankle_joint", (0.1, 0.4)),
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
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
        "ranges": (0.5, 2.0),
        "operation": "scale",
      },
    )
    cfg.events["mass_inertia"] = EventTermCfg(
      mode="reset",
      func=dr.pseudo_inertia,
      # floating_base is massless; pseudo-inertia there would create NaNs.
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names="hip|.*leg[1-4]|.*foot"),
        "alpha_range": (-0.0527, 0.0477),
      },
    )
    cfg.events["base_com"] = EventTermCfg(
      mode="reset",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names="hip"),
        "operation": "add",
        "ranges": {0: (-0.02, 0.02), 1: (-0.02, 0.02), 2: (-0.02, 0.02)},
      },
    )
    cfg.events["push_robot"] = EventTermCfg(
      mode="interval",
      func=push_by_setting_velocity,
      interval_range_s=(3.0, 6.0),
      params={
        "velocity_range": {
          "x": (-0.1, 0.1),
          "y": (-0.1, 0.1),
          "z": (0.0, 0.0),
          "roll": (-0.05, 0.05),
          "pitch": (-0.05, 0.05),
          "yaw": (-0.1, 0.1),
        }
      },
    )
  return cfg
