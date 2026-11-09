"""Train and evaluate a first flat-ground PPO walking policy in Isaac Lab.

Run in an Antioch Isaac Lab service:
    python src/train_locomotion.py --smoke
    python src/train_locomotion.py --num-envs 512 --iterations 500

This is a simulation baseline. The actuator and ankle transmission parameters
still need measurements before treating a policy as hardware ready.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import antioch


DRIVEN_JOINTS = [
    "left_hip1_joint", "right_hip1_joint",
    "left_hip2_joint", "right_hip2_joint",
    "left_thigh_joint", "right_thigh_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_joint", "right_ankle_joint",
]


def spawn_humanoid_with_foot_contact(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the USD and add PhysX contact reporting to both foot links."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from pxr import PhysxSchema
    import omni.usd

    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = omni.usd.get_context().get_stage()
    for side in ("left", "right"):
        foot_path = (
            f"{prim_path}/Geometry/hip/{side}_leg1/{side}_leg2/"
            f"{side}_leg3/{side}_leg4/{side}_foot"
        )
        foot_prim = stage.GetPrimAtPath(foot_path)
        if not foot_prim.IsValid():
            raise RuntimeError(f"USD foot link not found for contact reporting: {foot_path}")
        PhysxSchema.PhysxContactReportAPI.Apply(foot_prim)
    return prim


def commanded_velocity_progress(env, command_name: str = "base_velocity"):
    """Reward center-of-mass travel along the requested heading."""
    import torch

    command = env.command_manager.get_command(command_name)[:, :2]
    robot = env.scene["robot"]
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    velocity_w = robot.data.body_com_lin_vel_w.torch[:, 0]
    velocity = quat_apply_inverse(yaw_quat(robot.data.root_quat_w.torch), velocity_w)[:, :2]
    speed = torch.linalg.vector_norm(command, dim=1).clamp_min(0.1)
    return (torch.sum(command * velocity, dim=1) / speed).clamp(-1.0, 1.0)


def center_of_mass_velocity_b(env):
    """Expose COM velocity because this MJCF's free-root origin is near the soles."""
    robot = env.scene["robot"]
    from isaaclab.utils.math import quat_apply_inverse

    return quat_apply_inverse(robot.data.root_quat_w.torch, robot.data.body_com_lin_vel_w.torch[:, 0])


def track_com_lin_vel_xy_yaw_exp(env, command_name: str = "base_velocity", std: float = 0.5):
    """Track commanded planar velocity at the articulation COM in the yaw frame."""
    import torch
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)[:, :2]
    velocity_w = robot.data.body_com_lin_vel_w.torch[:, 0]
    velocity_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w.torch), velocity_w)[:, :2]
    error = torch.sum(torch.square(command - velocity_yaw), dim=1)
    return torch.exp(-error / (std * std))


def base_tilted_too_far(env, upright_cosine: float = 0.5):
    """Terminate on a fall using the base-frame gravity vector, not hip contact."""
    robot = env.scene["robot"]
    return -robot.data.projected_gravity_b[:, 2] < upright_cosine


def alternating_foot_lift_velocity(env, asset_cfg):
    """Give a dense gait signal when one foot rises faster than the other."""
    import torch

    robot = env.scene[asset_cfg.name]
    foot_vertical_speed = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    upward_speed = torch.clamp(foot_vertical_speed, min=0.0)
    single_swing = upward_speed.max(dim=1).values - 0.5 * upward_speed.min(dim=1).values
    return torch.clamp(single_swing - 0.03, min=0.0, max=0.20)


def alternating_joint_gait_reward(
    env,
    cycle_hz: float = 0.5,
    hip_amplitude: float = 0.15,
    knee_swing_amplitude: float = 0.12,
):
    """Track a slow alternating hip/knee cycle to seed a bipedal step."""
    import torch

    robot = env.scene["robot"]
    names = ("left_hip1_joint", "right_hip1_joint", "left_knee_joint", "right_knee_joint")
    ids = [robot.joint_names.index(name) for name in names]
    phase = 2.0 * math.pi * cycle_hz * env.episode_length_buf * env.step_dt
    swing = torch.sin(phase)
    left_swing = torch.clamp(swing, min=0.0)
    right_swing = torch.clamp(-swing, min=0.0)
    target = robot.data.default_joint_pos[:, ids].clone()
    target[:, 0] += hip_amplitude * swing
    target[:, 1] -= hip_amplitude * swing
    target[:, 2] -= knee_swing_amplitude * left_swing
    target[:, 3] -= knee_swing_amplitude * right_swing
    error = robot.data.joint_pos[:, ids] - target
    return torch.exp(-torch.sum(error.square(), dim=1) / 0.08)


def gait_phase_observation(env, cycle_hz: float = 0.5):
    """Expose the internal gait oscillator so a feed-forward policy can time steps."""
    import torch

    phase = 2.0 * math.pi * cycle_hz * env.episode_length_buf * env.step_dt
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


def build_env_cfg(
    num_envs: int,
    *,
    action_scale: float = 0.40,
    command_min: float = 0.20,
    command_max: float = 0.50,
    tracking_std: float = 0.60,
    progress_weight: float = 5.0,
    privileged_critic: bool = False,
    actor_base_velocity: bool = False,
    standing_envs: float = 0.10,
    upright_weight: float = 2.0,
    termination_weight: float = 20.0,
    phase_observation: bool = False,
    gait_cycle_hz: float = 0.5,
    gait_hip_amplitude: float = 0.15,
    gait_knee_amplitude: float = 0.12,
    gait_reward_weight: float = 10.0,
):
    """Start from Isaac Lab's standard velocity task and replace H1 specifics."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg
    from isaaclab_tasks.manager_based.locomotion.velocity import mdp

    cfg = H1RoughEnvCfg()
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
    # The imported USD nests feet below Geometry/hip/leg1/leg2/leg3/leg4.
    # The stock H1 sensor pattern only matches shallow children, so point it
    # at the two foot bodies explicitly for swing/contact rewards.
    if cfg.scene.contact_forces is not None:
        cfg.scene.contact_forces.default.prim_path = (
            "{ENV_REGEX_NS}/Robot/Geometry/hip/.*_foot"
        )
        cfg.scene.contact_forces.physx.prim_path = (
            "{ENV_REGEX_NS}/Robot/Geometry/hip/.*_foot"
        )
        cfg.scene.contact_forces.physx.track_air_time = True
        cfg.scene.contact_forces.physx.history_length = 3

    # Use a modest crouch so the legs have range to absorb impacts and lift a
    # foot during stance transitions.
    cfg.scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd), activate_contact_sensors=True,
            func=spawn_humanoid_with_foot_contact,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.02),
            # Add a small symmetric hip abduction to widen the reset stance.
            # The imported foot geometry stays tied to the measured CAD.
            joint_pos={
                "left_hip2_joint": -0.08,
                "right_hip2_joint": 0.08,
                "left_knee_joint": -0.30,
                "right_knee_joint": -0.30,
            },
        ),
        actuators={
            "hip1": ImplicitActuatorCfg(joint_names_expr=["left_hip1_joint", "right_hip1_joint"], stiffness=100.0, damping=2.0, effort_limit_sim=120.0, velocity_limit_sim=8.0),
            "hip2": ImplicitActuatorCfg(joint_names_expr=["left_hip2_joint", "right_hip2_joint"], stiffness=100.0, damping=2.0, effort_limit_sim=60.0, velocity_limit_sim=8.0),
            "thigh": ImplicitActuatorCfg(joint_names_expr=["left_thigh_joint", "right_thigh_joint"], stiffness=100.0, damping=2.0, effort_limit_sim=60.0, velocity_limit_sim=8.0),
            "knee": ImplicitActuatorCfg(joint_names_expr=["left_knee_joint", "right_knee_joint"], stiffness=80.0, damping=2.0, effort_limit_sim=120.0, velocity_limit_sim=8.0),
            "ankle": ImplicitActuatorCfg(joint_names_expr=["left_ankle_joint", "right_ankle_joint"], stiffness=20.0, damping=2.0, effort_limit_sim=16.0, velocity_limit_sim=8.0),
        },
    )

    # Hardware-observable actor inputs: angular rate, gravity direction,
    # commanded velocity, motor-joint state, and previous motor action.
    policy = cfg.observations.policy
    if phase_observation:
        policy.gait_phase = ObservationTermCfg(
            func=gait_phase_observation, params={"cycle_hz": gait_cycle_hz}
        )
    if privileged_critic:
        critic = copy.deepcopy(policy)
        critic.enable_corruption = False
        critic.height_scan = None
        for term_name in ("base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
            getattr(critic, term_name).noise = None
        critic.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=DRIVEN_JOINTS, preserve_order=True)}
        critic.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=DRIVEN_JOINTS, preserve_order=True)}
        cfg.observations.critic = critic
    if not actor_base_velocity:
        policy.base_lin_vel = None
    else:
        policy.base_lin_vel.noise = None
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
    cfg.commands.base_velocity.rel_standing_envs = standing_envs
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.debug_vis = False
    # The foot sole is long along body Y, and the knee/ankle hinge axes place
    # the sagittal gait in the Y-Z plane. Start with forward-only requests.
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (command_min, command_max)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.vel_xy_success_threshold = 0.10
    cfg.commands.base_velocity.vel_yaw_success_threshold = 0.20
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    # Drop H1 torso/terrain terms and start without pushes or domain randomization.
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.events.physics_material = None
    cfg.events.reset_base.params["pose_range"] = {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.05, 0.05)}
    cfg.events.reset_base.params["velocity_range"] = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02), "roll": (-0.02, 0.02), "pitch": (-0.02, 0.02), "yaw": (-0.02, 0.02)}
    cfg.events.reset_robot_joints.params["position_range"] = (0.98, 1.02)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    rewards = cfg.rewards
    # Clear H1-specific terms, then define a compact tracking/stability objective.
    for name in vars(rewards):
        if not name.startswith("_"):
            setattr(rewards, name, None)
    rewards.track_lin_vel_xy_exp = RewardTermCfg(func=track_com_lin_vel_xy_yaw_exp, weight=4.0, params={"command_name": "base_velocity", "std": tracking_std})
    rewards.track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    rewards.command_progress = RewardTermCfg(func=commanded_velocity_progress, weight=progress_weight)
    foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_foot")
    foot_body_cfg = SceneEntityCfg("robot", body_names=".*_foot")
    rewards.feet_air_time = RewardTermCfg(
        func=mdp.feet_air_time_positive_biped,
        weight=10.0,
        params={"command_name": "base_velocity", "sensor_cfg": foot_sensor_cfg, "threshold": 0.20},
    )
    rewards.alternating_foot_lift = RewardTermCfg(
        func=alternating_foot_lift_velocity,
        weight=5.0,
        params={"asset_cfg": foot_body_cfg},
    )
    rewards.reference_gait = RewardTermCfg(
        func=alternating_joint_gait_reward,
        weight=gait_reward_weight,
        params={"cycle_hz": gait_cycle_hz, "hip_amplitude": gait_hip_amplitude, "knee_swing_amplitude": gait_knee_amplitude},
    )
    rewards.feet_slide = RewardTermCfg(
        func=mdp.feet_slide,
        weight=-0.25,
        params={"sensor_cfg": foot_sensor_cfg, "asset_cfg": foot_body_cfg},
    )
    rewards.ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    rewards.lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0)
    rewards.joint_acc_l2 = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=DRIVEN_JOINTS, preserve_order=True)})
    rewards.action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    rewards.action_l2 = RewardTermCfg(func=mdp.action_l2, weight=-0.001)
    rewards.flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-upright_weight)
    rewards.joint_deviation = RewardTermCfg(func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=DRIVEN_JOINTS, preserve_order=True)})
    rewards.joint_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=DRIVEN_JOINTS, preserve_order=True)})
    rewards.termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-termination_weight)
    cfg.terminations.base_contact = None
    from isaaclab.managers import TerminationTermCfg
    cfg.terminations.base_tilt = TerminationTermCfg(func=base_tilted_too_far, params={"upright_cosine": 0.5})
    cfg.curriculum.terrain_levels = None
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--output", default="outputs/locomotion")
    parser.add_argument("--resume", help="Resume PPO state from an existing .pt checkpoint")
    parser.add_argument("--action-scale", type=float, default=0.40)
    parser.add_argument("--command-min", type=float, default=0.20)
    parser.add_argument("--command-max", type=float, default=0.50)
    parser.add_argument("--curriculum-start-min", type=float, help="Start commanded speed here and ramp to --command-min")
    parser.add_argument("--curriculum-start-max", type=float, help="Start commanded speed here and ramp to --command-max")
    parser.add_argument("--curriculum-iterations", type=int, default=0, help="Iterations over which to ramp forward command speed")
    parser.add_argument("--tracking-std", type=float, default=0.60)
    parser.add_argument("--progress-weight", type=float, default=5.0)
    parser.add_argument("--standing-envs", type=float, default=0.10, help="Fraction of environments commanded to stand still")
    parser.add_argument("--upright-weight", type=float, default=2.0)
    parser.add_argument("--termination-weight", type=float, default=20.0)
    parser.add_argument("--gait-cycle-hz", type=float, default=0.5)
    parser.add_argument("--gait-hip-amplitude", type=float, default=0.15)
    parser.add_argument("--gait-knee-amplitude", type=float, default=0.12)
    parser.add_argument("--gait-reward-weight", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--initial-std", type=float, help="Set policy exploration std after loading a checkpoint")
    parser.add_argument("--privileged-critic", action="store_true", help="Give the critic simulated base velocity while keeping it out of actor observations")
    parser.add_argument("--actor-base-velocity", action="store_true", help="Also give actor simulator base velocity as a teacher input")
    parser.add_argument("--phase-observation", action="store_true", help="Give the policy an internal gait clock")
    parser.add_argument("--resume-full-state", action="store_true", help="Continue a checkpoint trained with the same privileged-critic architecture")
    parser.add_argument("--resume-critic-only", action="store_true", help="Load critic weights but initialize a fresh actor, for actor-observation changes")
    parser.add_argument("--resume-expand-actor-velocity", action="store_true", help="Warm-start an old actor by adding base velocity inputs ahead of its existing observation features")
    parser.add_argument("--resume-expand-phase", action="store_true", help="Warm-start actor and critic while adding gait phase observations")
    parser.add_argument("--smoke", action="store_true", help="Initialize and step 8 environments without PPO training")
    parser.add_argument("--pose-check", action="store_true", help="Hold the reset posture under gravity for 5 seconds")
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
        args.num_envs,
        action_scale=args.action_scale,
        command_min=args.command_min,
        command_max=args.command_max,
        tracking_std=args.tracking_std,
        progress_weight=args.progress_weight,
        privileged_critic=args.privileged_critic,
        actor_base_velocity=args.actor_base_velocity,
        standing_envs=args.standing_envs,
        upright_weight=args.upright_weight,
        termination_weight=args.termination_weight,
        phase_observation=args.phase_observation,
        gait_cycle_hz=args.gait_cycle_hz,
        gait_hip_amplitude=args.gait_hip_amplitude,
        gait_knee_amplitude=args.gait_knee_amplitude,
        gait_reward_weight=args.gait_reward_weight,
    )
    print("Initializing humanoid_legs_5 velocity task", flush=True)
    env = ManagerBasedRLEnv(cfg=cfg)
    print("Articulation joints:", env.scene["robot"].joint_names, flush=True)
    if "contact_forces" in env.scene.sensors:
        sensor = env.scene.sensors["contact_forces"]
        print("Contact sensor bodies:", sensor.body_names, flush=True)
    print("Actions:", env.action_manager.total_action_dim, "policy obs:", env.observation_manager.group_obs_dim, flush=True)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    obs = wrapped.get_observations()
    if args.smoke:
        for i in range(args.steps_per_env):
            actions = torch.zeros((args.num_envs, len(DRIVEN_JOINTS)), device=env.device)
            obs, reward, done, extras = wrapped.step(actions)
            print(json.dumps({"step": i, "reward_mean": float(reward.mean()), "done_count": int(done.sum()), "obs_shape": list(obs["policy"].shape)}), flush=True)
            if i == args.steps_per_env - 1 and "contact_forces" in env.scene.sensors:
                contact = env.scene.sensors["contact_forces"].data
                robot = env.scene["robot"]
                foot_ids = [robot.body_names.index(name) for name in ("left_foot", "right_foot")]
                print(json.dumps({
                    "contact_bodies": env.scene.sensors["contact_forces"].body_names,
                    "current_contact_time_s": contact.current_contact_time.torch[0].tolist(),
                    "current_air_time_s": contact.current_air_time.torch[0].tolist(),
                    "max_contact_force_n": contact.net_forces_w_history.torch[0, :, :, :].norm(dim=-1).max(dim=0).values.tolist(),
                    "root_position_w_m": robot.data.root_pos_w[0].tolist(),
                    "foot_body_positions_w_m": robot.data.body_pos_w[0, foot_ids].tolist(),
                }), flush=True)
        env.close()
        return
    if args.pose_check:
        robot = env.scene["robot"]
        actions = torch.zeros((args.num_envs, len(DRIVEN_JOINTS)), device=env.device)
        done_total = 0
        for i in range(250):
            obs, reward, done, extras = wrapped.step(actions)
            done_total += int(done.sum())
            if i % 50 == 49:
                projected = robot.data.projected_gravity_b
                print(json.dumps({"t_s": round((i + 1) * 0.02, 2), "root_z_mean_m": float(robot.data.root_pos_w[:, 2].mean()), "upright_cos_mean": float(-projected[:, 2].mean()), "done_total": done_total}), flush=True)
        if "contact_forces" in env.scene.sensors:
            contact = env.scene.sensors["contact_forces"].data
            print(json.dumps({
                "contact_bodies": env.scene.sensors["contact_forces"].body_names,
                "current_contact_time_s": contact.current_contact_time.torch.mean(dim=0).tolist(),
                "current_air_time_s": contact.current_air_time.torch.mean(dim=0).tolist(),
                "max_contact_force_n": contact.net_forces_w_history.torch.norm(dim=-1).amax(dim=(0, 1)).tolist(),
                "foot_body_positions_w_mean_m": robot.data.body_pos_w[:, [robot.body_names.index("left_foot"), robot.body_names.index("right_foot")], :].mean(dim=0).tolist(),
            }), flush=True)
        env.close()
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
        "obs_groups": {"actor": ["policy"], "critic": ["critic"] if args.privileged_critic else ["policy"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [256, 128, 128], "activation": "elu", "obs_normalization": True, "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": args.initial_std or 0.8, "std_type": "scalar"}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [256, 128, 128], "activation": "elu", "obs_normalization": True, "distribution_cfg": None},
        "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2, "entropy_coef": 0.01, "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": args.learning_rate, "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0},
        "logger": "tensorboard",
        "experiment_name": "humanoid_legs_5_flat_velocity",
        "run_name": time.strftime("%Y%m%d_%H%M%S"),
        "log_dir": str(out_dir.resolve()),
        "save_obs_groups": ["policy"],
    }
    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=str(out_dir.resolve()), device=str(env.device))
    if args.resume:
        print(f"Resuming PPO state from {args.resume}", flush=True)
        load_cfg = None
        if args.resume_expand_phase:
            load_cfg = {"actor": False, "critic": False, "optimizer": False, "iteration": True}
        elif args.resume_critic_only or args.resume_expand_actor_velocity:
            load_cfg = {"actor": False, "critic": True, "optimizer": False, "iteration": True}
        elif args.privileged_critic and not args.resume_full_state:
            # The actor dimensions match the old checkpoint; start a fresh
            # privileged critic and optimizer for the expanded critic input.
            load_cfg = {"actor": True, "critic": False, "optimizer": False, "iteration": True}
        runner.load(args.resume, load_cfg=load_cfg, map_location=str(env.device))
        runner.alg.learning_rate = args.learning_rate
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = args.learning_rate
        print(f"Set resumed PPO learning rate to {args.learning_rate:g}", flush=True)
        if args.resume_expand_phase:
            checkpoint_data = torch.load(args.resume, map_location=env.device, weights_only=False)

            def expand_tail(module, old_state):
                new_state = module.state_dict()
                for key, old_value in old_state.items():
                    if key == "mlp.0.weight":
                        value = new_state[key].clone()
                        value[:, : old_value.shape[1]] = old_value
                        value[:, old_value.shape[1] :] = 0.0
                        new_state[key] = value
                    elif key.startswith("obs_normalizer.") and key in new_state and old_value.ndim == 2:
                        value = new_state[key].clone()
                        value[:, : old_value.shape[1]] = old_value
                        new_state[key] = value
                    elif key in new_state and new_state[key].shape == old_value.shape:
                        new_state[key] = old_value
                module.load_state_dict(new_state)

            expand_tail(runner.alg.actor, checkpoint_data["actor_state_dict"])
            expand_tail(runner.alg.critic, checkpoint_data["critic_state_dict"])
            print("Warm-started actor and critic with zero-initialized gait-phase inputs", flush=True)
        if args.resume_expand_actor_velocity:
            checkpoint_data = torch.load(args.resume, map_location=env.device, weights_only=False)
            old_actor = checkpoint_data["actor_state_dict"]
            new_actor = runner.alg.actor.state_dict()
            for key, old_value in old_actor.items():
                if key.startswith("obs_normalizer.") and key in new_actor and old_value.ndim == 2:
                    value = new_actor[key].clone()
                    value[:, 3:] = old_value
                    new_actor[key] = value
                elif key == "mlp.0.weight":
                    value = new_actor[key].clone()
                    value[:, 3:] = old_value
                    value[:, :3] = 0.0
                    new_actor[key] = value
                elif key in new_actor and new_actor[key].shape == old_value.shape:
                    new_actor[key] = old_value
            runner.alg.actor.load_state_dict(new_actor)
            print("Warm-started actor: preserved prior weights and added zero-initialized base-velocity inputs", flush=True)
    if args.initial_std is not None:
        with torch.no_grad():
            runner.alg.actor.distribution.std_param.fill_(args.initial_std)
        print(f"Set policy exploration std to {args.initial_std}", flush=True)
    print(f"Training PPO: {args.num_envs} envs, {args.iterations} iterations; outputs at {out_dir.resolve()}", flush=True)
    if args.curriculum_iterations > 0:
        term = env.command_manager.get_term("base_velocity")
        start_min = args.command_min if args.curriculum_start_min is None else args.curriculum_start_min
        start_max = args.command_max if args.curriculum_start_max is None else args.curriculum_start_max
        block_size = 25
        completed = 0
        while completed < args.iterations:
            block = min(block_size, args.iterations - completed)
            alpha = min(1.0, (completed + block) / args.curriculum_iterations)
            low = start_min + alpha * (args.command_min - start_min)
            high = start_max + alpha * (args.command_max - start_max)
            term.cfg.ranges.lin_vel_y = (low, high)
            env.command_manager.reset(torch.arange(args.num_envs, device=env.device))
            print(json.dumps({
                "curriculum_iteration": completed,
                "lin_vel_y_command_range_mps": [round(low, 4), round(high, 4)],
            }), flush=True)
            runner.learn(num_learning_iterations=block, init_at_random_ep_len=False)
            completed += block
    else:
        # A phase-conditioned policy must start at the same phase as the reset
        # joint pose; randomizing only episode time creates a mismatched gait.
        runner.learn(
            num_learning_iterations=args.iterations,
            init_at_random_ep_len=not args.phase_observation,
        )
    runner.save(str(out_dir / "humanoid_legs_5_latest.pt"))
    report = {
        "model": "humanoid_legs_5",
        "task": "flat_ground_commanded_velocity",
        "iterations": args.iterations,
        "num_envs": args.num_envs,
        "steps_per_env": args.steps_per_env,
        "policy_rate_hz": 50,
        "driven_joints": DRIVEN_JOINTS,
        "reset_knee_position_rad": -0.30,
        "training_command_ranges": {"forward_body_y_mps": [args.command_min, args.command_max], "body_x_mps": [0.0, 0.0], "yaw_rate_radps": [0.0, 0.0]},
        "standing_env_fraction": args.standing_envs,
        "upright_reward_weight": args.upright_weight,
        "termination_reward_weight": args.termination_weight,
        "joint_action_scale_rad": args.action_scale,
        "velocity_tracking_std_mps": args.tracking_std,
        "command_progress_weight": args.progress_weight,
        "privileged_critic": args.privileged_critic,
        "actor_base_velocity": args.actor_base_velocity,
        "phase_observation": args.phase_observation,
        "gait_cycle_hz": args.gait_cycle_hz,
        "gait_hip_amplitude_rad": args.gait_hip_amplitude,
        "gait_knee_amplitude_rad": args.gait_knee_amplitude,
        "gait_reward_weight": args.gait_reward_weight,
        "resume_full_state": args.resume_full_state,
        "learning_rate": args.learning_rate,
        "checkpoint": str(out_dir / "humanoid_legs_5_latest.pt"),
        "hardware_ready": False,
        "limitations": ["actuator/linkage system identification incomplete", "evaluate standing, command tracking, falls, and contact before hardware use"],
    }
    (out_dir / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    env.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
