"""EngineAI T800 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  T800_ACTION_SCALE,
  get_t800_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import (
  MotionCommandCfg,
  MotionReferenceJointPositionActionCfg,
)
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def engineai_t800_flat_tracking_env_cfg(
  play: bool = False,
  deploy: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create EngineAI T800 flat terrain tracking configuration.

  deploy=True restricts the actor observations to the six terms the EngineAI
  Native SDK runner can provide on hardware (see submission_tools/
  SDK_FINDINGS.md): it removes motion_anchor_pos_b (no world-position estimate
  on the robot) and base_lin_vel. Policies intended for SDK deployment must be
  trained with this variant (actor input 134 dims instead of 140).
  """
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_t800_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = T800_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "LINK_WAIST_YAW"
  motion_cmd.body_names = (
    "LINK_BASE",
    "LINK_HIP_ROLL_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_HIP_ROLL_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_WAIST_YAW",
    "LINK_SHOULDER_ROLL_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
    "LINK_SHOULDER_ROLL_R",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_YAW_R",
  )

  # T800 collision geoms are unnamed (class-based), so remove foot friction randomization.
  cfg.events.pop("foot_friction", None)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("LINK_WAIST_YAW",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R",
    "LINK_ELBOW_YAW_L",
    "LINK_ELBOW_YAW_R",
  )

  cfg.viewer.body_name = "LINK_WAIST_YAW"
  cfg.viewer.azimuth = 300.00

  if deploy:
    # SDK-deployable observation set; the critic keeps full observability.
    cfg.observations["actor"].terms.pop("motion_anchor_pos_b")
    cfg.observations["actor"].terms.pop("base_lin_vel")

    # Match the SDK runner's action semantics: q_des = ref(t) + action*scale
    # (the stock task uses a constant default-pose offset instead).
    cfg.actions["joint_pos"] = MotionReferenceJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=T800_ACTION_SCALE,
      command_name="motion",
    )

    # The runner yaw-aligns the reference anchor quat against the IMU,
    # which sits on LINK_BASE — anchor there so the actor's
    # motion_anchor_ori_b matches deployment exactly.
    motion_cmd.anchor_body_name = "LINK_BASE"

  # Fix sensor names for T800 (different from G1 naming convention).
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    if "base_lin_vel" in terms:
      terms["base_lin_vel"].params["sensor_name"] = "robot/base_link_linear_velocity"
    if "base_ang_vel" in terms:
      terms["base_ang_vel"].params["sensor_name"] = "robot/base_link_angular_velocity"

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
