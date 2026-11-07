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


def build_env_cfg(num_envs: int, standing_pose: str = "neutral", reward_profile: str = "baseline"):
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
    if standing_pose not in {"neutral", "crouch"}:
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

    # Hardware-observable actor inputs: angular rate, gravity direction,
    # commanded velocity, motor-joint state, and previous motor action.
    policy = cfg.observations.policy
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
        asset_name="robot", joint_names=DRIVEN_JOINTS, scale=0.40,
        use_default_offset=True, preserve_order=True,
    )
    cfg.commands.base_velocity.resampling_time_range = (5.0, 8.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.10
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.debug_vis = False
    # The foot sole is long along body Y, and the knee/ankle hinge axes place
    # the sagittal gait in the Y-Z plane. Start with forward-only requests.
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.20, 0.50)
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
    cfg.events.reset_base.params["pose_range"] = {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.05, 0.05)}
    cfg.events.reset_base.params["velocity_range"] = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02), "roll": (-0.02, 0.02), "pitch": (-0.02, 0.02), "yaw": (-0.02, 0.02)}
    cfg.events.reset_robot_joints.params["position_range"] = (0.98, 1.02)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    if standing_pose == "crouch":
        # Independent scaling would break the passive linkage's reset closure.
        cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

    rewards = cfg.rewards
    # Clear H1-specific terms, then define a compact tracking/stability objective.
    for name in vars(rewards):
        if not name.startswith("_"):
            setattr(rewards, name, None)
    rewards.track_lin_vel_xy_exp = RewardTermCfg(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=4.0, params={"command_name": "base_velocity", "std": 0.60})
    rewards.track_ang_vel_z_exp = None
    rewards.command_progress = RewardTermCfg(func=commanded_velocity_progress, weight=1.0)
    rewards.ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    rewards.lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0)
    rewards.joint_acc_l2 = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=DRIVEN_JOINTS, preserve_order=True)})
    rewards.action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    rewards.flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-2.0)
    rewards.joint_deviation = RewardTermCfg(func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=DRIVEN_JOINTS, preserve_order=True)})
    rewards.joint_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=DRIVEN_JOINTS, preserve_order=True)})
    rewards.termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-20.0)
    if reward_profile in {"stepping", "stepping_straight", "stepping_wide"}:
        from locomotion_rewards import alternating_contact, feet_slide, gait_phase_observation, POSTURE_JOINTS
        policy.gait_phase = ObservationTermCfg(func=gait_phase_observation)
        rewards.gait_contact = RewardTermCfg(func=alternating_contact, weight=1.0)
        rewards.feet_slide = RewardTermCfg(func=feet_slide, weight=-0.2)
        # Body Y is forward: hip2 and thigh are lateral/yaw joints here.
        # Let sagittal hip1, knee and ankle joints leave the reference to step.
        rewards.joint_deviation.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(POSTURE_JOINTS), preserve_order=True
        )
    if reward_profile in {"stepping_straight", "stepping_wide"}:
        from locomotion_rewards import yaw_rate_error_l2, yaw_rate_tracking
        rewards.track_ang_vel_z_exp = RewardTermCfg(func=yaw_rate_tracking, weight=2.0, params={"std": 0.5})
        rewards.yaw_rate_error = RewardTermCfg(func=yaw_rate_error_l2, weight=-0.5)
    if reward_profile == "stepping_wide":
        from locomotion_rewards import stance_width_tracking
        rewards.stance_width = RewardTermCfg(
            func=stance_width_tracking, weight=0.5,
            params={"minimum": 0.16, "maximum": 0.20, "std": 0.05},
        )
    cfg.terminations.base_contact = None
    from isaaclab.managers import TerminationTermCfg
    cfg.terminations.base_tilt = TerminationTermCfg(func=base_tilted_too_far, params={"upright_cosine": 0.5})
    cfg.curriculum.terrain_levels = None
    return cfg



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--standing-pose", choices=["neutral", "crouch"], default="neutral")
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default="baseline")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--output", default="outputs/locomotion")
    parser.add_argument("--resume", help="Resume PPO state from an existing .pt checkpoint")
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

    cfg = build_env_cfg(args.num_envs, standing_pose=args.standing_pose, reward_profile=args.reward_profile)
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
        "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2, "entropy_coef": 0.01, "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 3.0e-4, "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0},
        "logger": "tensorboard",
        "experiment_name": "humanoid_legs_5_flat_velocity",
        "run_name": time.strftime("%Y%m%d_%H%M%S"),
        "log_dir": str(out_dir.resolve()),
        "save_obs_groups": ["policy"],
    }
    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=str(out_dir.resolve()), device=str(env.device))
    if args.resume:
        print(f"Resuming PPO state from {args.resume}", flush=True)
        runner.load(args.resume, map_location=str(env.device))
    print(f"Training PPO: {args.num_envs} envs, {args.iterations} iterations; outputs at {out_dir.resolve()}", flush=True)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    runner.save(str(out_dir / "humanoid_legs_5_latest.pt"))
    report = {
        "model": "humanoid_legs_5",
        "task": "flat_ground_commanded_velocity",
        "iterations": args.iterations,
        "resumed_from": args.resume,
        "num_envs": args.num_envs,
        "steps_per_env": args.steps_per_env,
        "policy_rate_hz": 50,
        "driven_joints": DRIVEN_JOINTS,
        "standing_pose": args.standing_pose,
        "reward_profile": args.reward_profile,
        "stance_width_reward": ({"band_m": [0.16, 0.20], "std_m": 0.05, "weight": 0.5} if args.reward_profile == "stepping_wide" else None),
        "policy_observation_dim": env.observation_manager.group_obs_dim["policy"][0],
        "reset_knee_position_rad": -0.4 if args.standing_pose == "crouch" else 0.0,
        "training_command_ranges": {"forward_body_y_mps": [0.20, 0.50], "body_x_mps": [0.0, 0.0], "yaw_rate_radps": [0.0, 0.0]},
        "checkpoint": str(out_dir / "humanoid_legs_5_latest.pt"),
        "hardware_ready": False,
        "limitations": ["actuator/linkage system identification incomplete", "evaluate standing, command tracking, falls, and contact before hardware use"],
    }
    (out_dir / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    env.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
