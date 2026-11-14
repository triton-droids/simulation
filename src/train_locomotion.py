"""Train and evaluate a first flat-ground PPO walking policy in Isaac Lab.

Run in an Antioch Isaac Lab service:
    python src/train_locomotion.py --smoke
    python src/train_locomotion.py --num-envs 512 --iterations 500

This is a simulation baseline. The actuator and ankle transmission parameters
still need measurements before treating a policy as hardware ready.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import antioch

REWARD_PROFILES = ("baseline", "stepping", "stepping_straight", "stepping_wide")
TEACHER_ACTOR = None

DRIVEN_JOINTS = [
    "left_hip1_joint", "right_hip1_joint",
    "left_hip2_joint", "right_hip2_joint",
    "left_thigh_joint", "right_thigh_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_joint", "right_ankle_joint",
]


def commanded_velocity_progress(env, command_name: str = "base_velocity"):
    """Reward movement along the requested heading to give PPO a useful signal at rest."""
    import torch

    command = env.command_manager.get_command(command_name)[:, :2]
    velocity = env.scene["robot"].data.root_lin_vel_b[:, :2]
    speed = torch.linalg.vector_norm(command, dim=1).clamp_min(0.1)
    return (torch.sum(command * velocity, dim=1) / speed).clamp(-1.0, 1.0)


def commanded_velocity_stall_penalty(env, min_forward_speed: float = 0.08, command_name: str = "base_velocity"):
    """Flag commanded walking that produces less than a minimum projected speed."""
    import torch

    command = env.command_manager.get_command(command_name)[:, :2]
    velocity = env.scene["robot"].data.root_lin_vel_b[:, :2]
    command_speed = torch.linalg.vector_norm(command, dim=1)
    projected_speed = torch.sum(command * velocity, dim=1) / command_speed.clamp_min(1e-6)
    return ((command_speed > 0.1) & (projected_speed < min_forward_speed)).float()


def teacher_action_deviation(env):
    """Keep PPO actions near a warm-start gait while other rewards improve it."""
    import torch

    if TEACHER_ACTOR is None:
        raise RuntimeError("teacher_action_deviation requires --teacher-action-weight > 0")
    observation = env.observation_manager.compute_group("policy")
    with torch.no_grad():
        teacher_action = TEACHER_ACTOR({"policy": observation}).clamp(-1.0, 1.0)
    current_action = env.action_manager.action
    return torch.mean((current_action - teacher_action).square(), dim=-1)


def gait_reference_tracking(
    env,
    period: float = 1.0 / 0.35,
    hip_amplitude: float = 0.10,
    knee_amplitude: float = 0.08,
    action_scale: float = 0.22,
    balance_blend: float = 0.515,
    std: float = 0.20,
):
    """Penalize deviation from the successful balance-policy/gait-action blend."""
    import torch

    if TEACHER_ACTOR is None:
        raise RuntimeError("gait_reference_tracking requires a loaded warm-start teacher actor")
    if period <= 0 or action_scale <= 0 or std <= 0 or not 0.0 <= balance_blend <= 1.0:
        raise ValueError("gait reference period/scale/std must be positive and balance_blend in [0, 1]")
    robot = env.scene["robot"]
    names = robot.joint_names
    indices = [names.index(name) for name in ("left_hip1_joint", "right_hip1_joint", "left_knee_joint", "right_knee_joint")]
    observation = env.observation_manager.compute_group("policy")
    with torch.no_grad():
        teacher_action = TEACHER_ACTOR({"policy": observation}).clamp(-1.0, 1.0)
    phase = 2.0 * math.pi * (env.episode_length_buf * env.step_dt / period)
    swing = torch.sin(phase)
    ref_delta = torch.stack((
        hip_amplitude * swing,
        -hip_amplitude * swing,
        -knee_amplitude * torch.relu(swing),
        -knee_amplitude * torch.relu(-swing),
    ), dim=-1)
    target = teacher_action.clone()
    reference_actions = ref_delta / action_scale
    target[:, indices] = balance_blend * teacher_action[:, indices] + (1.0 - balance_blend) * reference_actions
    target.clamp_(-1.0, 1.0)
    error = env.action_manager.action - target
    absolute = error.abs()
    huber = torch.where(absolute < std, 0.5 * error.square() / std, absolute - 0.5 * std)
    return -torch.mean(huber, dim=-1)


def action_magnitude_l2(env):
    """Penalize saturated joint targets while preserving smooth gait actions."""
    import torch

    action = env.action_manager.action
    return torch.sum(action.square(), dim=-1)


def base_tilted_too_far(env, upright_cosine: float = 0.5):
    """Terminate on a fall using the base-frame gravity vector, not hip contact."""
    robot = env.scene["robot"]
    return -robot.data.projected_gravity_b[:, 2] < upright_cosine


def spawn_robot_with_foot_contacts(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Enable reporting on nested feet before contact sensors initialize."""
    import isaaclab.sim as sim_utils
    from pxr import PhysxSchema, Usd, UsdPhysics

    result = sim_utils.spawn_from_usd(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs
    )
    for root in sim_utils.find_matching_prims(prim_path):
        for prim in Usd.PrimRange(root):
            if prim.GetName() in {"left_foot", "right_foot"} and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                reporter = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                reporter.CreateThresholdAttr(0.0)
    return result


def build_env_cfg(
    num_envs: int,
    standing_pose: str = "neutral",
    reward_profile: str = "baseline",
    tracking_std: float = 0.60,
    progress_weight: float = 1.0,
    stall_penalty_weight: float = 0.0,
    stall_speed_threshold: float = 0.08,
    flat_orientation_weight: float = 2.0,
    termination_penalty_weight: float = 20.0,
    teacher_action_weight: float = 0.0,
    gait_reference_weight: float = 0.0,
    gait_reference_std: float = 0.20,
    gait_contact_weight: float = 1.0,
    differential_gait_reward: bool = False,
    stance_width_weight: float = 0.5,
    stance_width_std: float = 0.05,
    stance_width_min: float = 0.29,
    stance_width_max: float = 0.32,
    action_scale: float = 0.40,
    actor_base_velocity: bool = False,
    forward_speed_min: float = 0.20,
    forward_speed_max: float = 0.50,
    gait_period: float = 0.80,
    standing_envs: float = 0.10,
    reset_noise_scale: float = 1.0,
):
    """Start from Isaac Lab's standard velocity task and replace H1 specifics."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg
    from isaaclab_tasks.manager_based.locomotion.velocity import mdp

    cfg = H1RoughEnvCfg()
    if standing_pose not in {"neutral", "crouch", "legacy_gait"}:
        raise ValueError(f"Unknown standing pose: {standing_pose}")
    if reward_profile not in REWARD_PROFILES:
        raise ValueError(f"Unknown reward profile: {reward_profile}")
    usd = Path("robots/humanoid_legs_5/usd_import/humanoid_legs_5/humanoid_legs_5.usda").resolve()
    if not usd.is_file():
        raise FileNotFoundError(f"Packaged humanoid USD not found: {usd}")

    cfg.sim.dt = 1.0 / 200.0
    cfg.sim.render_interval = 4
    cfg.seed = 42
    cfg.decimation = 4  # 50 Hz policy, matching the historical control path.
    cfg.episode_length_s = 12.0
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 2.0
    cfg.scene.replicate_physics = True
    cfg.scene.height_scanner = None
    # Imported links are nested; select each physical foot explicitly.
    cfg.scene.contact_forces = None
    for side in ("left", "right"):
        foot_path = (
            "{ENV_REGEX_NS}/Robot/Geometry/hip/"
            f"{side}_leg1/{side}_leg2/{side}_leg3/{side}_leg4/{side}_foot"
        )
        setattr(cfg.scene, f"{side}_foot_contact", ContactSensorCfg(
            prim_path=foot_path, update_period=0.0, history_length=3,
            track_air_time=True,
        ))
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.38, 0.42)),
    )
    cfg.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=1500.0)

    # Use the validated neutral standing pose as the initial baseline.
    cfg.scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd), activate_contact_sensors=True,
            func=spawn_robot_with_foot_contacts,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.02),
            joint_pos={"left_knee_joint": 0.0, "right_knee_joint": 0.0},
        ),
        actuators={
            "hip1": ImplicitActuatorCfg(joint_names_expr=["left_hip1_joint", "right_hip1_joint"], stiffness=100.0, damping=2.0, effort_limit_sim=120.0, velocity_limit_sim=8.0),
            "hip2": ImplicitActuatorCfg(joint_names_expr=["left_hip2_joint", "right_hip2_joint"], stiffness=100.0, damping=2.0, effort_limit_sim=60.0, velocity_limit_sim=8.0),
            "thigh": ImplicitActuatorCfg(joint_names_expr=["left_thigh_joint", "right_thigh_joint"], stiffness=100.0, damping=2.0, effort_limit_sim=60.0, velocity_limit_sim=8.0),
            "knee": ImplicitActuatorCfg(joint_names_expr=["left_knee_joint", "right_knee_joint"], stiffness=80.0, damping=2.0, effort_limit_sim=120.0, velocity_limit_sim=8.0),
            # Simulation standing check: 20/2 let the ankles yield and tip over.
            "ankle": ImplicitActuatorCfg(joint_names_expr=["left_ankle_joint", "right_ankle_joint"], stiffness=100.0, damping=5.0, effort_limit_sim=16.0, velocity_limit_sim=8.0),
        },
    )
    if standing_pose == "crouch":
        from standing_pose import crouch_joint_positions
        cfg.scene.robot.init_state.joint_pos = crouch_joint_positions()
        cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.01)
    elif standing_pose == "legacy_gait":
        # Match the earlier balanced-gait checkpoints' reset and ankle drive.
        cfg.scene.robot.init_state.joint_pos = {
            "left_hip2_joint": -0.08, "right_hip2_joint": 0.08,
            "left_knee_joint": -0.30, "right_knee_joint": -0.30,
        }
        cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.02)
        cfg.scene.robot.actuators["ankle"].stiffness = 20.0
        cfg.scene.robot.actuators["ankle"].damping = 2.0

    # Hardware-observable actor inputs: angular rate, gravity direction,
    # commanded velocity, motor-joint state, and previous motor action.
    policy = cfg.observations.policy
    if not actor_base_velocity:
        policy.base_lin_vel = None
    policy.height_scan = None
    policy.base_ang_vel.noise = None
    policy.projected_gravity.noise = None
    policy.joint_pos.noise = None
    policy.joint_vel.noise = None
    policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=DRIVEN_JOINTS, preserve_order=True)}
    policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=DRIVEN_JOINTS, preserve_order=True)}
    policy.enable_corruption = False
    cfg.actions.joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=DRIVEN_JOINTS, scale=action_scale,
        use_default_offset=True, preserve_order=True,
    )
    cfg.commands.base_velocity.resampling_time_range = (5.0, 8.0)
    if not 0.0 <= standing_envs < 1.0:
        raise ValueError("standing_envs must be in [0, 1)")
    cfg.commands.base_velocity.rel_standing_envs = standing_envs
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.debug_vis = False
    # The foot sole is long along body Y, and the knee/ankle hinge axes place
    # the sagittal gait in the Y-Z plane. Start with forward-only requests.
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.20, 0.50)
    if forward_speed_min <= 0 or forward_speed_max < forward_speed_min:
        raise ValueError("forward speed range must satisfy 0 < min <= max")
    cfg.commands.base_velocity.ranges.lin_vel_y = (forward_speed_min, forward_speed_max)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.vel_xy_success_threshold = 0.20
    cfg.commands.base_velocity.vel_yaw_success_threshold = 0.20
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    # Drop H1 torso/terrain terms and start without pushes or domain randomization.
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.events.physics_material = None
    if reset_noise_scale < 0:
        raise ValueError("reset_noise_scale must be nonnegative")
    cfg.events.reset_base.params["pose_range"] = {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.05, 0.05)}
    cfg.events.reset_base.params["velocity_range"] = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02), "roll": (-0.02, 0.02), "pitch": (-0.02, 0.02), "yaw": (-0.02, 0.02)}
    for ranges in (cfg.events.reset_base.params["pose_range"], cfg.events.reset_base.params["velocity_range"]):
        for key, bounds in ranges.items():
            ranges[key] = tuple(value * reset_noise_scale for value in bounds)
    cfg.events.reset_robot_joints.params["position_range"] = (0.98, 1.02)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    if standing_pose in {"crouch", "legacy_gait"}:
        # Independent scaling would break the passive linkage's reset closure.
        cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

    rewards = cfg.rewards
    # Clear H1-specific terms, then define a compact tracking/stability objective.
    for name in vars(rewards):
        if not name.startswith("_"):
            setattr(rewards, name, None)
    if tracking_std <= 0 or progress_weight < 0 or stall_penalty_weight < 0 or stall_speed_threshold < 0 or flat_orientation_weight < 0 or termination_penalty_weight < 0 or teacher_action_weight < 0 or gait_reference_weight < 0 or gait_reference_std <= 0:
        raise ValueError("tracking_std must be positive and reward weights/thresholds nonnegative")
    if gait_contact_weight < 0 or stance_width_weight < 0 or stance_width_std <= 0:
        raise ValueError("reward weights must be nonnegative and stance_width_std positive")
    if not 0 < stance_width_min <= stance_width_max:
        raise ValueError("stance width must satisfy 0 < min <= max")
    rewards.track_lin_vel_xy_exp = RewardTermCfg(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=4.0, params={"command_name": "base_velocity", "std": tracking_std})
    rewards.track_ang_vel_z_exp = None
    rewards.command_progress = RewardTermCfg(func=commanded_velocity_progress, weight=progress_weight)
    if stall_penalty_weight:
        rewards.commanded_velocity_stall = RewardTermCfg(
            func=commanded_velocity_stall_penalty,
            weight=-stall_penalty_weight,
            params={"min_forward_speed": stall_speed_threshold},
        )
    if teacher_action_weight:
        rewards.teacher_action_deviation = RewardTermCfg(func=teacher_action_deviation, weight=-teacher_action_weight)
    if gait_reference_weight:
        rewards.gait_reference = RewardTermCfg(
            func=gait_reference_tracking,
            weight=gait_reference_weight,
            params={
                "period": gait_period,
                "action_scale": action_scale,
                "std": gait_reference_std,
                "balance_blend": 0.515,
            },
        )
    rewards.ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    rewards.lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0)
    rewards.joint_acc_l2 = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=DRIVEN_JOINTS, preserve_order=True)})
    rewards.action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    rewards.action_magnitude_l2 = RewardTermCfg(func=action_magnitude_l2, weight=-0.01)
    rewards.flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-flat_orientation_weight)
    rewards.joint_deviation = RewardTermCfg(func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=DRIVEN_JOINTS, preserve_order=True)})
    rewards.joint_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=DRIVEN_JOINTS, preserve_order=True)})
    rewards.termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-termination_penalty_weight)
    if reward_profile in {"stepping", "stepping_straight", "stepping_wide"}:
        from locomotion_rewards import alternating_contact, alternating_contact_differential, feet_slide, gait_phase_observation, POSTURE_JOINTS
        policy.gait_phase = ObservationTermCfg(func=gait_phase_observation, params={"period": gait_period})
        contact_reward = alternating_contact_differential if differential_gait_reward else alternating_contact
        rewards.gait_contact = RewardTermCfg(func=contact_reward, weight=gait_contact_weight, params={"period": gait_period})
        rewards.feet_slide = RewardTermCfg(func=feet_slide, weight=-0.2)
        # Body Y is forward: hip2 and thigh are lateral/yaw joints here.
        # Let sagittal hip1, knee and ankle joints leave the reference to step.
        rewards.joint_deviation.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(POSTURE_JOINTS), preserve_order=True
        )
    if reward_profile in {"stepping_straight", "stepping_wide"}:
        from locomotion_rewards import heading_error_l2, heading_error_observation, heading_tracking, yaw_rate_error_l2, yaw_rate_tracking
        rewards.track_ang_vel_z_exp = RewardTermCfg(func=yaw_rate_tracking, weight=2.0, params={"std": 0.5})
        rewards.yaw_rate_error = RewardTermCfg(func=yaw_rate_error_l2, weight=-0.5)
        # Fixed world heading: body Y points along world +Y at yaw zero.
        # Two additional observations give these profiles 43 policy inputs.
        policy.heading_error = ObservationTermCfg(
            func=heading_error_observation, params={"desired_yaw": 0.0},
        )
        rewards.heading_alignment = RewardTermCfg(
            func=heading_tracking, weight=1.0, params={"desired_yaw": 0.0, "std": 0.25},
        )
        rewards.heading_error = RewardTermCfg(
            func=heading_error_l2, weight=-0.5, params={"desired_yaw": 0.0},
        )
    if reward_profile == "stepping_wide":
        from locomotion_rewards import stance_width_tracking
        rewards.stance_width = RewardTermCfg(
            func=stance_width_tracking, weight=stance_width_weight,
            params={"minimum": stance_width_min, "maximum": stance_width_max, "std": stance_width_std},
        )
    cfg.terminations.base_contact = None
    from isaaclab.managers import TerminationTermCfg
    cfg.terminations.base_tilt = TerminationTermCfg(func=base_tilted_too_far, params={"upright_cosine": 0.5})
    cfg.curriculum.terrain_levels = None
    return cfg



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--standing-pose", choices=["neutral", "crouch", "legacy_gait"], default="neutral")
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default="baseline")
    parser.add_argument("--tracking-std", type=float, default=0.60, help="Velocity tracking reward width; smaller makes standing still less rewarding")
    parser.add_argument("--progress-weight", type=float, default=1.0, help="Weight for progress along the commanded direction")
    parser.add_argument("--stall-penalty-weight", type=float, default=0.0, help="Penalty for commanded forward speed below the stall threshold")
    parser.add_argument("--stall-speed-threshold", type=float, default=0.08)
    parser.add_argument("--flat-orientation-weight", type=float, default=2.0)
    parser.add_argument("--termination-penalty-weight", type=float, default=20.0)
    parser.add_argument("--teacher-action-weight", type=float, default=0.0, help="Keep policy actions near the resumed warm-start actor")
    parser.add_argument("--gait-reference-weight", type=float, default=0.0, help="Reward matching the alternating hip/knee reference")
    parser.add_argument("--gait-reference-std", type=float, default=0.20)
    parser.add_argument("--gait-contact-weight", type=float, default=1.0)
    parser.add_argument("--differential-gait-reward", action="store_true", help="Subtract the stationary 0.5 contact-match baseline")
    parser.add_argument("--stance-width-weight", type=float, default=0.5)
    parser.add_argument("--stance-width-std", type=float, default=0.05)
    parser.add_argument("--stance-width-min", type=float, default=0.29)
    parser.add_argument("--stance-width-max", type=float, default=0.32)
    parser.add_argument("--actor-base-velocity", action="store_true", help="Use body linear velocity for 44-input warm-start checkpoints")
    parser.add_argument("--action-scale", type=float, default=0.40)
    parser.add_argument("--forward-speed-min", type=float, default=0.20)
    parser.add_argument("--forward-speed-max", type=float, default=0.50)
    parser.add_argument("--speed-curriculum", help="Comma-separated fixed forward speeds; split iterations evenly and advance through them")
    parser.add_argument("--gait-period", type=float, default=0.80)
    parser.add_argument("--standing-envs", type=float, default=0.10)
    parser.add_argument("--reset-noise-scale", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--output", default="outputs/locomotion")
    parser.add_argument("--resume", help="Resume PPO state from an existing .pt checkpoint")
    parser.add_argument(
        "--resume-expand-phase", action="store_true",
        help="Resume a 39-input checkpoint for a 41-input stepping policy, adding zero-initialized phase inputs and resetting the optimizer",
    )
    parser.add_argument("--reset-optimizer", action="store_true", help="Restore actor/critic weights and iteration, but start fresh Adam moments")
    parser.add_argument("--smoke", action="store_true", help="Initialize and step 8 environments without PPO training")
    parser.add_argument("--pose-check", action="store_true", help="Hold the reset posture under gravity for 5 seconds")
    parser.add_argument("--pose-check-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.smoke:
        args.num_envs, args.iterations, args.steps_per_env = 8, 0, 2
    if args.pose_check:
        args.num_envs, args.iterations = 8, 0

    antioch.start_simulation()
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    cfg = build_env_cfg(
        args.num_envs, standing_pose=args.standing_pose,
        reward_profile=args.reward_profile,
        tracking_std=args.tracking_std,
        progress_weight=args.progress_weight,
        stall_penalty_weight=args.stall_penalty_weight,
        stall_speed_threshold=args.stall_speed_threshold,
        flat_orientation_weight=args.flat_orientation_weight,
        termination_penalty_weight=args.termination_penalty_weight,
        teacher_action_weight=args.teacher_action_weight,
        gait_reference_weight=args.gait_reference_weight,
        gait_reference_std=args.gait_reference_std,
        gait_contact_weight=args.gait_contact_weight,
        differential_gait_reward=args.differential_gait_reward,
        stance_width_weight=args.stance_width_weight,
        stance_width_std=args.stance_width_std,
        stance_width_min=args.stance_width_min,
        stance_width_max=args.stance_width_max,
        action_scale=args.action_scale,
        actor_base_velocity=args.actor_base_velocity,
        forward_speed_min=args.forward_speed_min,
        forward_speed_max=args.forward_speed_max,
        gait_period=args.gait_period,
        standing_envs=args.standing_envs,
        reset_noise_scale=args.reset_noise_scale,
    )
    if args.pose_check:
        if args.pose_check_seconds <= 0:
            parser.error("pose-check-seconds must be positive")
        cfg.episode_length_s = args.pose_check_seconds + 1.0
    print("Initializing humanoid_legs_5 velocity task", flush=True)
    env = ManagerBasedRLEnv(cfg=cfg)
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.GetName() in {"left_foot", "right_foot"}:
            print(
                "FOOT:",
                prim.GetPath(),
                "rigid_body:",
                prim.HasAPI(UsdPhysics.RigidBodyAPI),
                flush=True,
            )
    print("Articulation joints:", env.scene["robot"].joint_names, flush=True)
    print("Actions:", env.action_manager.total_action_dim, "policy obs:", env.observation_manager.group_obs_dim, flush=True)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    obs = wrapped.get_observations()
    if args.smoke:
        for i in range(args.steps_per_env):
            actions = torch.zeros((args.num_envs, len(DRIVEN_JOINTS)), device=env.device)
            obs, reward, done, extras = wrapped.step(actions)
            print(json.dumps({"step": i, "reward_mean": float(reward.mean()), "done_count": int(done.sum()), "obs_shape": list(obs["policy"].shape)}), flush=True)
        env.close()
        return
    if args.pose_check:
        robot = env.scene["robot"]
        foot_sensors = {side: env.scene[f"{side}_foot_contact"] for side in ("left", "right")}
        for side, sensor in foot_sensors.items():
            print(f"CONTACT SENSOR {side}: {sensor.body_names}", flush=True)
            if list(sensor.body_names) != [f"{side}_foot"]:
                raise RuntimeError(f"Unexpected {side} contact sensor bodies: {sensor.body_names}")
        actions = torch.zeros((args.num_envs, len(DRIVEN_JOINTS)), device=env.device)
        done_total = 0
        min_upright_cos = 1.0
        for i in range(round(args.pose_check_seconds / env.step_dt)):
            obs, reward, done, extras = wrapped.step(actions)
            done_total += int(done.sum())
            min_upright_cos = min(min_upright_cos, float(-robot.data.projected_gravity_b[:, 2].max()))
            if not torch.isfinite(robot.data.joint_pos).all():
                raise RuntimeError("Non-finite joint positions in standing check")
            if i % 10 == 9:
                projected = robot.data.projected_gravity_b
                feet = {}
                for side, sensor in foot_sensors.items():
                    force = sensor.data.net_forces_w[0, 0]
                    feet[side] = {
                        "force_world_N": force.tolist(),
                        "contact_above_1N": bool(torch.linalg.vector_norm(force) > 1.0),
                    }
                print(json.dumps({"t_s": round((i + 1) * 0.02, 2), "root_z_mean_m": float(robot.data.root_pos_w[:, 2].mean()), "upright_cos_mean": float(-projected[:, 2].mean()), "done_total": done_total, "env0_reset_this_step": bool(done[0]), "env0_feet": feet}), flush=True)
        print("POSE CHECK SUMMARY:", json.dumps({"seconds": args.pose_check_seconds, "num_envs": args.num_envs, "resets": done_total, "max_tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, min_upright_cos))))}), flush=True)
        env.close()
        if done_total:
            raise RuntimeError(f"Standing check failed: {done_total} resets")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = {
        "seed": 42,
        "device": str(env.device),
        "num_steps_per_env": args.steps_per_env,
        "max_iterations": args.iterations,
        "save_interval": 25,
        "empirical_normalization": True,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [256, 128, 128], "activation": "elu", "obs_normalization": True, "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.8, "std_type": "scalar"}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [256, 128, 128], "activation": "elu", "obs_normalization": True, "distribution_cfg": None},
        "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2, "entropy_coef": args.entropy_coef, "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": args.learning_rate, "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0},
        "logger": "tensorboard",
        "experiment_name": "humanoid_legs_5_flat_velocity",
        "run_name": time.strftime("%Y%m%d_%H%M%S"),
        "log_dir": str(out_dir.resolve()),
        "save_obs_groups": ["policy"],
    }
    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=str(out_dir.resolve()), device=str(env.device))
    if args.resume:
        print(f"Resuming PPO state from {args.resume}", flush=True)
        if args.resume_expand_phase:
            loaded = torch.load(args.resume, map_location=str(env.device), weights_only=False)
            expected_dim = env.observation_manager.group_obs_dim["policy"][0]
            for key in ("actor_state_dict", "critic_state_dict"):
                state = loaded[key]
                weight = state["mlp.0.weight"]
                old_dim = weight.shape[1]
                if expected_dim != old_dim + 2:
                    raise ValueError(
                        f"Phase expansion expects exactly two new inputs; checkpoint has {old_dim}, policy has {expected_dim}"
                    )
                state["mlp.0.weight"] = torch.nn.functional.pad(weight, (0, 2))
                for stat in ("_mean", "_var", "_std"):
                    name = f"obs_normalizer.{stat}"
                    value = state[name]
                    fill = 1.0 if stat in {"_var", "_std"} else 0.0
                    state[name] = torch.nn.functional.pad(value, (0, 2), value=fill)
            # New phase inputs start with zero influence; use a fresh optimizer
            # because Adam's first-layer moments still have the old input width.
            runner.alg.actor.load_state_dict(loaded["actor_state_dict"])
            runner.alg.critic.load_state_dict(loaded["critic_state_dict"])
            runner.current_learning_iteration = int(loaded.get("iter", 0))
            print(
                f"Expanded checkpoint observation width to {expected_dim}; "
                "actor/critic and iteration restored, optimizer reset",
                flush=True,
            )
        elif args.reset_optimizer:
            loaded = torch.load(args.resume, map_location=str(env.device), weights_only=False)
            runner.alg.actor.load_state_dict(loaded["actor_state_dict"])
            runner.alg.critic.load_state_dict(loaded["critic_state_dict"])
            runner.current_learning_iteration = int(loaded.get("iter", 0))
            print("Restored actor/critic and iteration; optimizer reset", flush=True)
        else:
            runner.load(args.resume, map_location=str(env.device))
    if args.teacher_action_weight or args.gait_reference_weight:
        import copy
        global TEACHER_ACTOR
        TEACHER_ACTOR = copy.deepcopy(runner.alg.actor).eval()
        for parameter in TEACHER_ACTOR.parameters():
            parameter.requires_grad_(False)
        print(
            "Frozen warm-start actor enabled as gait teacher "
            f"(action penalty weight {args.teacher_action_weight})",
            flush=True,
        )
    print(f"Training PPO: {args.num_envs} envs, {args.iterations} iterations; outputs at {out_dir.resolve()}", flush=True)
    if args.speed_curriculum:
        try:
            speeds = [float(value.strip()) for value in args.speed_curriculum.split(",")]
        except ValueError as exc:
            parser.error(f"invalid --speed-curriculum: {exc}")
        if not speeds or any(speed <= 0 for speed in speeds):
            parser.error("speed-curriculum must contain positive speeds")
        base_vel_term = env.command_manager.get_term("base_velocity")
        stage_iters = [args.iterations // len(speeds)] * len(speeds)
        for i in range(args.iterations % len(speeds)):
            stage_iters[i] += 1
        for stage, (speed, count) in enumerate(zip(speeds, stage_iters), start=1):
            if count == 0:
                continue
            base_vel_term.cfg.ranges.lin_vel_y = (speed, speed)
            env.command_manager.get_command("base_velocity")[:, 1] = speed
            print(
                f"Speed curriculum stage {stage}/{len(speeds)}: "
                f"{speed:.3f} m/s for {count} PPO iterations",
                flush=True,
            )
            runner.learn(
                num_learning_iterations=count,
                init_at_random_ep_len=not bool(args.gait_reference_weight),
            )
    else:
        runner.learn(
            num_learning_iterations=args.iterations,
            init_at_random_ep_len=not bool(args.gait_reference_weight),
        )
    runner.save(str(out_dir / "humanoid_legs_5_latest.pt"))
    report = {
        "model": "humanoid_legs_5",
        "task": "flat_ground_commanded_velocity",
        "iterations": args.iterations,
        "resumed_from": args.resume,
        "num_envs": args.num_envs,
        "steps_per_env": args.steps_per_env,
        "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef,
        "policy_rate_hz": 50,
        "driven_joints": DRIVEN_JOINTS,
        "standing_pose": args.standing_pose,
        "reward_profile": args.reward_profile,
        "heading_direction_terms": ({"desired_world_yaw_rad": 0.0, "std_rad": 0.25, "reward_weight": 1.0, "penalty_weight": -0.5, "disabled_for_commanded_turns": True, "observation": ["sin_heading_error", "cos_heading_error"]} if args.reward_profile in {"stepping_straight", "stepping_wide"} else None),
        "tracking_std": args.tracking_std,
        "progress_weight": args.progress_weight,
        "stall_penalty_weight": args.stall_penalty_weight,
        "stall_speed_threshold": args.stall_speed_threshold,
        "flat_orientation_weight": args.flat_orientation_weight,
        "termination_penalty_weight": args.termination_penalty_weight,
        "teacher_action_weight": args.teacher_action_weight,
        "gait_reference_weight": args.gait_reference_weight,
        "gait_reference_std": args.gait_reference_std,
        "random_phase_start": not bool(args.gait_reference_weight),
        "gait_contact_weight": args.gait_contact_weight,
        "differential_gait_reward": args.differential_gait_reward,
        "stance_width_weight": args.stance_width_weight,
        "stance_width_std": args.stance_width_std,
        "actor_base_velocity": args.actor_base_velocity,
        "action_scale": args.action_scale,
        "gait_period_s": args.gait_period,
        "standing_env_fraction": args.standing_envs,
        "reset_noise_scale": args.reset_noise_scale,
        "stance_width_reward": ({"band_m": [args.stance_width_min, args.stance_width_max], "std_m": args.stance_width_std, "weight": args.stance_width_weight} if args.reward_profile == "stepping_wide" else None),
        "policy_observation_dim": env.observation_manager.group_obs_dim["policy"][0],
        "reset_knee_position_rad": -0.4 if args.standing_pose == "crouch" else 0.0,
        "training_command_ranges": {"forward_body_y_mps": [args.forward_speed_min, args.forward_speed_max], "body_x_mps": [0.0, 0.0], "yaw_rate_radps": [0.0, 0.0]},
        "speed_curriculum_mps": ([float(value.strip()) for value in args.speed_curriculum.split(",")] if args.speed_curriculum else None),
        "checkpoint": str(out_dir / "humanoid_legs_5_latest.pt"),
        "hardware_ready": False,
        "limitations": ["actuator/linkage system identification incomplete", "evaluate standing, command tracking, falls, and contact before hardware use"],
    }
    (out_dir / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    env.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
