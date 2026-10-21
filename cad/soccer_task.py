"""Walk-to-ball and kick task for the Triton lower body.

The robot has to approach a stationary ball, line up behind it and kick it
toward a goal line at ``+x``.  The locomotion machinery (actuators, torque-speed
envelope, slew cap, domain randomisation) is inherited unchanged from the
velocity task; only the command, the ball and the task rewards are new.

Command layout is ``(vx, vy, yaw_rate)`` exactly like the velocity task, but the
values are produced by ``BallApproachCommand`` from the ball geometry, so the
same 3-D command observation keeps working and a velocity policy can drive the
robot to a target point.
"""

from dataclasses import dataclass

import mujoco
import torch
from hardware_tracking_task import HardwareTrackingConfig
from velocity_task import (
  FOOT_SITES,
  GAIT_FREQUENCY,
  PhaseClock,
  feet_phase_reward,
  make_triton_velocity_cfg,
)

from mjlab.entity import EntityCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

BALL_RADIUS = 0.105
BALL_MASS = 0.43
# Goal line lives 4 m beyond the ball along the robot's forward axis.  The
# The Triton lower-body model uses y as forward (x is lateral), so task geometry
# is expressed on the y axis of the environment-local frame.
GOAL_OFFSET = 4.0
GOAL_DIR_W = (0.0, 1.0)
STANDOFF = 0.16  # Distance behind the ball centre the robot should line up at.
APPROACH_SPEED = 0.45


def ball_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball")
  body.add_freejoint(name="ball_joint")
  body.add_geom(
    name="ball_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(BALL_RADIUS,) * 3,
    mass=BALL_MASS,
    rgba=(0.95, 0.95, 0.95, 1.0),
    friction=(0.8, 0.02, 0.01),
    condim=3,
    priority=1,
  )
  return spec


@dataclass(kw_only=True)
class BallApproachCommandCfg(CommandTermCfg):
  """Velocity command derived from where the ball is."""

  ball_entity: str = "ball"
  goal_offset: float = GOAL_OFFSET
  desired_speed: float = APPROACH_SPEED
  stop_distance: float = 0.06
  standoff: float = STANDOFF
  max_yaw_rate: float = 0.6
  yaw_gain: float = 1.5
  gait_frequency: float = GAIT_FREQUENCY
  frequency_jitter: float = 0.1

  def build(self, env) -> "BallApproachCommand":
    return BallApproachCommand(self, env)


class BallApproachCommand(CommandTerm):
  cfg: BallApproachCommandCfg

  def __init__(self, cfg: BallApproachCommandCfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene["robot"]
    self.ball = env.scene[cfg.ball_entity]
    self.clock = PhaseClock(
      self.num_envs, self.device, cfg.gait_frequency, cfg.frequency_jitter
    )
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    # Goal position in world coordinates: fixed env-local x goal line at the
    # ball's lateral offset.  Lives next to env_origins so parallel envs do not
    # interfere.
    self.goal_xy = torch.zeros(self.num_envs, 2, device=self.device)
    self._steps = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["ball_distance"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  @property
  def phase(self) -> torch.Tensor:
    return self.clock.phase

  @property
  def phase_obs(self) -> torch.Tensor:
    return self.clock.phase_obs

  def compute(self, dt: float) -> None:
    super().compute(dt)
    self.clock.advance(dt)

  def _ball_xy(self):
    return self.ball.data.root_link_pos_w[:, :2]

  def reset(self, env_ids=None):
    if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
      ball_xy = self._ball_xy()[env_ids]
      self.goal_xy[env_ids, 0] = ball_xy[:, 0]
      self.goal_xy[env_ids, 1] = ball_xy[:, 1] + self.cfg.goal_offset
      self._steps[env_ids] = 0.0
      self.clock.reset(env_ids)
      self.clock.resample_frequency(env_ids)
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # The command is recomputed every step from the ball; nothing to sample.
    del env_ids

  def _update_metrics(self) -> None:
    self._steps += 1.0
    self.metrics["error_vel_xy"] += torch.norm(
      self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
    ) / self._steps
    self.metrics["error_vel_yaw"] += torch.abs(
      self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
    ) / self._steps
    self.metrics["ball_distance"] += (
      torch.norm(self._ball_xy() - self.robot.data.root_link_pos_w[:, :2], dim=-1)
      / self._steps
    )

  def _update_command(self) -> None:
    ball = self._ball_xy()
    base = self.robot.data.root_link_pos_w[:, :2]
    goal_dir = torch.tensor(GOAL_DIR_W, device=self.device).expand_as(ball)
    standoff_point = ball - goal_dir * self.cfg.standoff
    delta = standoff_point - base
    distance = torch.norm(delta, dim=-1, keepdim=True).clamp_min(1e-4)
    speed = torch.where(
      distance > self.cfg.stop_distance,
      torch.full_like(distance, self.cfg.desired_speed),
      torch.zeros_like(distance),
    )
    desired_world = delta / distance * speed
    # Rotate the desired world velocity into the base yaw frame.
    heading = self.robot.data.heading_w
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    self.vel_command_b[:, 0] = cos_h * desired_world[:, 0] + sin_h * desired_world[:, 1]
    self.vel_command_b[:, 1] = -sin_h * desired_world[:, 0] + cos_h * desired_world[:, 1]
    to_ball = ball - base
    bearing = torch.atan2(to_ball[:, 1], to_ball[:, 0])
    yaw_error = torch.atan2(torch.sin(bearing - heading), torch.cos(bearing - heading))
    self.vel_command_b[:, 2] = torch.clamp(
      self.cfg.yaw_gain * yaw_error, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate
    )


##
# Reward terms.
##


def _ball(env, name="ball"):
  return env.scene[name]


def ball_distance_to_goal(env, ball_name: str = "ball"):
  ball = _ball(env, ball_name).data.root_link_pos_w[:, :2]
  goal = env.command_manager.get_term("twist").goal_xy
  return torch.norm(ball - goal, dim=-1)


def _goal_dir(env) -> torch.Tensor:
  return torch.tensor(GOAL_DIR_W, device=env.device).expand(env.num_envs, 2)


def ball_goal_progress(env, ball_name: str = "ball"):
  """Potential difference of the ball's distance to the goal point (toward only)."""
  distance = ball_distance_to_goal(env, ball_name)
  if not hasattr(env, "_ball_prev_distance"):
    env._ball_prev_distance = distance.clone()
  delta = env._ball_prev_distance - distance
  env._ball_prev_distance = distance.clone()
  # Ignore resets and teleports; a genuine kick moves the ball < 15 cm per step.
  valid = (delta.abs() < 0.15).float()
  env.extras["log"]["Metrics/ball_distance_to_goal"] = distance.mean()
  return torch.clamp(delta, min=0.0, max=0.05) * valid


def ball_speed_toward_goal(env, ball_name: str = "ball", max_speed: float = 6.0):
  """Reward ball velocity along the goal direction."""
  ball = _ball(env, ball_name)
  toward = ball.data.root_link_lin_vel_w[:, :2] * _goal_dir(env)
  speed = toward.sum(dim=-1).clamp(0.0, max_speed)
  env.extras["log"]["Metrics/ball_speed"] = speed.mean()
  return speed


def foot_kick_speed(env, sensor_name: str):
  """Reward the foot moving forward while it touches the ball."""
  sensor = env.scene[sensor_name]
  found = sensor.data.found
  if found is None:
    return torch.zeros(env.num_envs, device=env.device)
  robot = env.scene["robot"]
  site_ids = torch.tensor(
    [robot.site_names.index(name) for name in FOOT_SITES], device=env.device
  )
  foot_vel = (robot.data.site_lin_vel_w[:, site_ids, :2] * _goal_dir(env)[:, None]).sum(-1)
  active = (found > 0).float()
  return torch.sum(foot_vel.clamp(min=0.0) * active, dim=-1) * (found.sum(dim=-1) > 0)


def approach_alignment(env, ball_name: str = "ball", standoff: float = STANDOFF):
  """Reward standing behind the ball at kick range, facing the goal."""
  ball = _ball(env, ball_name).data.root_link_pos_w[:, :2]
  robot = env.scene["robot"]
  base = robot.data.root_link_pos_w[:, :2]
  target = ball - torch.tensor(GOAL_DIR_W, device=env.device) * standoff
  error = torch.sum(torch.square(base - target), dim=-1)
  return torch.exp(-error / 0.05)


def ball_relative_xy(env, ball_name: str = "ball") -> torch.Tensor:
  """Ball position in the base frame (2-D)."""
  ball = _ball(env, ball_name).data.root_link_pos_w[:, :2]
  robot = env.scene["robot"]
  delta_w = ball - robot.data.root_link_pos_w[:, :2]
  heading = robot.data.heading_w
  cos_h, sin_h = torch.cos(heading), torch.sin(heading)
  return torch.stack(
    (cos_h * delta_w[:, 0] + sin_h * delta_w[:, 1], -sin_h * delta_w[:, 0] + cos_h * delta_w[:, 1]),
    dim=-1,
  )


def ball_velocity(env, ball_name: str = "ball") -> torch.Tensor:
  return _ball(env, ball_name).data.root_link_lin_vel_w


def ball_reached_goal(env, ball_name: str = "ball"):
  """Success: the ball has crossed the goal line for its env."""
  ball_y = _ball(env, ball_name).data.root_link_pos_w[:, 1]
  goal_y = env.command_manager.get_term("twist").goal_xy[:, 1]
  return ball_y > goal_y


def make_soccer_cfg(
  hardware: HardwareTrackingConfig | None = None, play: bool = False
):
  if hardware is None:
    hardware = HardwareTrackingConfig()
  cfg = make_triton_velocity_cfg(hardware, play=play)

  cfg.scene.entities["ball"] = EntityCfg(
    spec_fn=ball_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, BALL_RADIUS)),
  )

  feet_ball_cfg = ContactSensorCfg(
    name="feet_ball_contact",
    primary=ContactMatch(
      mode="subtree", pattern=r"^(left_foot|right_foot)$", entity="robot"
    ),
    secondary=ContactMatch(mode="subtree", pattern="ball", entity="ball"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ball_cfg,)

  # Ball is re-placed behind the goal direction every episode.
  cfg.events["reset_ball"] = EventTermCfg(
    mode="reset",
    func=velocity_mdp.reset_root_state_uniform,
    params={
      "pose_range": {
        "x": (-0.4, 0.4),
        "y": (0.7, 1.4),
        "z": (0.0, 0.0),
      },
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("ball"),
    },
  )
  cfg.events["reset_base"].params["pose_range"]["x"] = (-0.2, 0.2)
  cfg.events["reset_base"].params["pose_range"]["y"] = (-0.2, 0.2)

  cfg.commands["twist"] = BallApproachCommandCfg(
    resampling_time_range=(1e9, 1e9),
    ball_entity="ball",
    debug_vis=False,
  )

  actor = cfg.observations["actor"].terms
  actor["ball_position"] = ObservationTermCfg(
    func=ball_relative_xy, noise=Unoise(-0.02, 0.02)
  )
  actor["ball_position"].delay_min_lag = hardware.observation_delay_steps[0]
  actor["ball_position"].delay_max_lag = hardware.observation_delay_steps[1]
  cfg.observations["critic"].terms["ball_position"] = ObservationTermCfg(
    func=ball_relative_xy
  )
  cfg.observations["critic"].terms["ball_velocity"] = ObservationTermCfg(
    func=ball_velocity
  )

  rewards = cfg.rewards
  rewards["track_linear_velocity"].weight = 1.0
  rewards["track_angular_velocity"].weight = 0.5
  rewards["pose"].weight = 0.5
  rewards["air_time"].weight = 0.3
  rewards["ball_goal_progress"] = RewardTermCfg(
    func=ball_goal_progress, weight=40.0
  )
  rewards["ball_speed"] = RewardTermCfg(func=ball_speed_toward_goal, weight=1.5)
  rewards["foot_kick"] = RewardTermCfg(
    func=foot_kick_speed, weight=0.5, params={"sensor_name": feet_ball_cfg.name}
  )
  rewards["approach_alignment"] = RewardTermCfg(func=approach_alignment, weight=0.6)
  rewards["feet_phase"] = RewardTermCfg(
    func=feet_phase_reward,
    weight=3.0,
    params={
      "command_name": "twist",
      "swing_height": 0.08,
      "sigma": 0.01,
      "command_threshold": 0.15,
    },
  )

  cfg.terminations["reached_goal"] = TerminationTermCfg(
    func=ball_reached_goal, time_out=True
  )
  cfg.curriculum = {}

  cfg.scene.num_envs = 1
  cfg.episode_length_s = 15.0
  if play:
    cfg.episode_length_s = 30.0
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg
