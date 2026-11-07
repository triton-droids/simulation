"""Fixed-target PhysX standing check; no training or automatic resets."""

from __future__ import annotations

import json
import argparse
import math
import time
from pathlib import Path

import antioch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Simulation standing baseline, not identified hardware gains.
    parser.add_argument("--ankle-stiffness", type=float, default=100.0)
    parser.add_argument("--ankle-damping", type=float, default=5.0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--hold-seconds", type=float, default=10.0)
    parser.add_argument("--knee-angle", type=float, default=0.0)
    parser.add_argument("--hip-angle", type=float, default=0.0)
    parser.add_argument("--ankle-angle", type=float, default=0.0)
    parser.add_argument("--ramp-seconds", type=float, default=2.0)
    parser.add_argument("--video", help="Optional MP4 path; run with Antioch --stream")
    parser.add_argument("--record-start", type=float, default=2.0)
    parser.add_argument("--report", default="logs/lower_body_physics_check.json")
    args = parser.parse_args()
    if args.seconds <= 0 or args.hold_seconds < 0:
        parser.error("seconds must be positive and hold-seconds nonnegative")
    antioch.start_simulation()

    import isaaclab.sim as sim_utils
    import torch
    import omni.usd
    from pxr import Gf, UsdGeom
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from train_locomotion import spawn_robot_with_foot_contacts
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils import configclass
    from isaacsim.core.utils.viewports import set_camera_view

    usd = Path("robots/humanoid_legs_5/usd_import/humanoid_legs_5/humanoid_legs_5.usda").resolve()
    if not usd.is_file():
        raise FileNotFoundError(usd)

    actuated = [
        "left_hip1_joint", "right_hip1_joint", "left_hip2_joint", "right_hip2_joint",
        "left_thigh_joint", "right_thigh_joint", "left_knee_joint", "right_knee_joint",
        "left_ankle_joint", "right_ankle_joint",
    ]

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1800.0))
        robot: ArticulationCfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(usd), activate_contact_sensors=True, func=spawn_robot_with_foot_contacts),
            # The hip is now the physical root; at joint zero the soles sit
            # about 1 cm above the floor when the hip root is at z ~= 0.01 m.
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.02),
                joint_pos={".*_hip1_joint": args.hip_angle, ".*_knee_joint": args.knee_angle, ".*_ankle_joint": args.ankle_angle},
            ),
            actuators={
                "hip1": ImplicitActuatorCfg(
                    joint_names_expr=["left_hip1_joint", "right_hip1_joint"],
                    stiffness=100.0, damping=2.0, effort_limit_sim=120.0, velocity_limit_sim=8.0,
                ),
                "hip2": ImplicitActuatorCfg(
                    joint_names_expr=["left_hip2_joint", "right_hip2_joint"],
                    stiffness=100.0, damping=2.0, effort_limit_sim=60.0, velocity_limit_sim=8.0,
                ),
                "thigh": ImplicitActuatorCfg(
                    joint_names_expr=["left_thigh_joint", "right_thigh_joint"],
                    stiffness=100.0, damping=2.0, effort_limit_sim=60.0, velocity_limit_sim=8.0,
                ),
                "knee": ImplicitActuatorCfg(
                    joint_names_expr=["left_knee_joint", "right_knee_joint"],
                    stiffness=80.0, damping=2.0, effort_limit_sim=120.0, velocity_limit_sim=8.0,
                ),
                "ankle": ImplicitActuatorCfg(
                    joint_names_expr=["left_ankle_joint", "right_ankle_joint"],
                    stiffness=args.ankle_stiffness, damping=args.ankle_damping, effort_limit_sim=16.0, velocity_limit_sim=8.0,
                ),
            },
        )

    simulation = SimulationContext(SimulationCfg(dt=1.0 / 200.0, render_interval=4))
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    for side in ("left", "right"):
        setattr(scene_cfg, f"{side}_foot_contact", ContactSensorCfg(
            prim_path=f"/World/Robot/Geometry/hip/{side}_leg1/{side}_leg2/{side}_leg3/{side}_leg4/{side}_foot",
            update_period=0.0,
        ))
    scene = InteractiveScene(scene_cfg)
    simulation.reset()
    robot: Articulation = scene["robot"]
    set_camera_view(eye=[1.7, 1.7, 1.15], target=[0.0, 0.0, 0.55], camera_prim_path="/OmniverseKit_Persp")
    print("USD articulation joints:", robot.joint_names, flush=True)
    print("USD articulation bodies:", robot.body_names, flush=True)
    root_prim = robot.root_physx_view.prim_paths[0]
    print("USD articulation root:", root_prim, flush=True)
    print("Default joint pose:", robot.data.default_joint_pos[0].tolist(), flush=True)

    q0 = robot.data.default_joint_pos.clone()
    start_q = robot.data.joint_pos.clone()
    actuator_ids = [robot.joint_names.index(name) for name in actuated]
    records = []
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache()
    sole_points = {}
    foot_ids = {}
    for side in ("left", "right"):
        foot_ids[side] = robot.body_names.index(f"{side}_foot")
        foot_path = robot.cfg.prim_path + f"/Geometry/hip/{side}_leg1/{side}_leg2/{side}_leg3/{side}_leg4/{side}_foot"
        collider = stage.GetPrimAtPath(foot_path + f"/{side}_foot_collision_box")
        transform = UsdGeom.Xformable(collider).GetLocalTransformation()
        half_size = UsdGeom.Cube(collider).GetSizeAttr().Get() / 2
        sole_points[side] = torch.tensor([
            list(transform.Transform(Gf.Vec3d(x * half_size, y * half_size, -half_size)))
            for x in (-1, 1) for y in (-1, 1)
        ], device=robot.device, dtype=torch.float32)
    pin_paths = {
        side: (
            robot.cfg.prim_path + f"/Geometry/hip/{side}_leg1/{side}_leg2/"
            f"{side}_leg3/{side}_leg4/{side}_foot/{side}_heel_linkage/{side}_heel_bar_pin",
            robot.cfg.prim_path + f"/Geometry/hip/{side}_leg1/{side}_leg2/"
            f"{side}_leg3/{side}_leg4/{side}_ankle_crank/{side}_crank_pin",
        )
        for side in ("left", "right")
    }
    steps = round(args.seconds / simulation.get_physics_dt())
    max_tilt_deg = 0.0
    max_ankle_torque = 0.0
    ankle_ids = [robot.joint_names.index(f"{side}_ankle_joint") for side in ("left", "right")]
    writer = None
    if args.video:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image, ImageDraw
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        antioch.capture_viewport()
        writer = imageio.get_writer(args.video, fps=10, codec="libx264", quality=8)
    for i in range(steps):
        alpha = min(1.0, (i + 1) * simulation.get_physics_dt() / max(args.ramp_seconds, simulation.get_physics_dt()))
        target = start_q + alpha * (q0 - start_q)
        robot.set_joint_position_target_index(
            target=target[:, actuator_ids], joint_ids=actuator_ids
        )
        scene.write_data_to_sim()
        simulation.step(render=(i % 4 == 0))
        scene.update(simulation.get_physics_dt())
        upright = float(-robot.data.projected_gravity_b[0, 2])
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, upright))))
        max_tilt_deg = max(max_tilt_deg, tilt_deg)
        max_ankle_torque = max(max_ankle_torque, float(robot.data.applied_torque[0, ankle_ids].abs().max()))
        if i % 100 == 0 or i == steps - 1:
            root = robot.data.root_pos_w[0]
            quat = robot.data.root_quat_w[0]
            q = robot.data.joint_pos[0]
            cache.Clear()
            closure = {}
            for side, paths in pin_paths.items():
                a, b = [cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation() for path in paths]
                closure[side] = float((a - b).GetLength())
            feet = {side: scene[f"{side}_foot_contact"].data.net_forces_w[0, 0].tolist() for side in ("left", "right")}
            sole_heights = {}
            sole_world = []
            for side, points in sole_points.items():
                # This installed Isaac Lab 3 runtime uses XYZW quaternions.
                rotation = robot.data.body_quat_w[0, foot_ids[side]]
                xyz = rotation[:3].expand_as(points)
                uv = 2 * torch.cross(xyz, points, dim=-1)
                world_points = points + rotation[3] * uv + torch.cross(xyz, uv, dim=-1)
                world_points += robot.data.body_pos_w[0, foot_ids[side]]
                sole_heights[side] = world_points[:, 2].tolist()
                sole_world.extend(world_points.tolist())
            masses = robot.data.default_mass[0]
            com = (robot.data.body_com_pos_w[0] * masses[:, None]).sum(dim=0) / masses.sum()
            rec = {"t_s": round((i + 1) * simulation.get_physics_dt(), 3), "tilt_deg": tilt_deg, "com_world_m": com.tolist(), "sole_corners_world_m": sole_world, "sole_corner_z_m": sole_heights, "foot_forces_world_N": feet, "root_xyz_m": root.tolist(), "root_quat_xyzw": quat.tolist(), "joint_pos": q.tolist(), "pin_gap_m": closure}
            records.append(rec)
            print(json.dumps(rec), flush=True)
        if writer is not None and i % 20 == 0 and i * simulation.get_physics_dt() >= args.record_start:
            frame = antioch.capture_viewport()
            if frame is not None:
                image = Image.fromarray(np.asarray(frame)[:, :, :3].astype(np.uint8))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, image.width, 44), fill=(12, 18, 28))
                draw.text((14, 12), "Fixed-target crouched standing | no RL | no resets", fill=(255, 255, 255))
                writer.append_data(np.asarray(image))

    if writer is not None:
        writer.close()
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"max_tilt_deg": max_tilt_deg, "max_ankle_torque_Nm": max_ankle_torque, "settings": vars(args)}
    print("STANDING SUMMARY:", json.dumps(summary), flush=True)
    out.write_text(json.dumps({"usd": str(usd), "joint_names": robot.joint_names, "steps": steps, "summary": summary, "records": records}, indent=2) + "\n")
    print(f"Wrote {out}", flush=True)
    deadline = time.monotonic() + args.hold_seconds
    while time.monotonic() < deadline:
        simulation.render()


if __name__ == "__main__":
    main()
