"""Run small, one-joint-at-a-time drive checks with the hip root fixed.

Run inside an active Antioch Isaac Lab service with:
    python src/check_joint_motion.py

The temporary fixed-base USD is generated from the bundled MJCF at runtime.
This isolates joint and linkage behavior; it is not a balance or walking test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import antioch


def main() -> None:
    antioch.start_simulation()

    import isaaclab.sim as sim_utils
    import torch
    import omni.usd
    from pxr import UsdGeom
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils import configclass
    from isaacsim.core.utils.viewports import set_camera_view
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    asset_dir = Path("robots/humanoid_legs_5").resolve()
    xml = asset_dir / "humanoid_legs_5.xml"
    if not xml.is_file():
        raise FileNotFoundError(xml)
    converter = MjcfConverter(MjcfConverterCfg(
        asset_path=str(xml),
        usd_dir=str(asset_dir / "usd_import" / "fixed_base_check"),
        usd_file_name="humanoid_legs_5_fixed.usd",
        fix_base=True,
        force_usd_conversion=True,
        make_instanceable=False,
    ))
    usd = Path(converter.usd_path)
    if not usd.is_file():
        raise FileNotFoundError(usd)
    print("Fixed-base MJCF import:", usd, flush=True)

    actuated = [
        "left_hip1_joint", "right_hip1_joint", "left_hip2_joint", "right_hip2_joint",
        "left_thigh_joint", "right_thigh_joint", "left_knee_joint", "right_knee_joint",
        "left_ankle_joint", "right_ankle_joint",
    ]

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(usd), activate_contact_sensors=False),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
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
                    stiffness=20.0, damping=2.0, effort_limit_sim=16.0, velocity_limit_sim=8.0,
                ),
            },
        )

    dt = 1.0 / 120.0
    simulation = SimulationContext(SimulationCfg(dt=dt, render_interval=1, gravity=(0.0, 0.0, 0.0)))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    simulation.reset()
    robot: Articulation = scene["robot"]
    set_camera_view(eye=[1.7, 1.7, 1.15], target=[0.0, 0.0, 0.55], camera_prim_path="/OmniverseKit_Persp")

    print("USD:", usd, flush=True)
    print("Articulation root:", robot.root_view.prim_paths[0], flush=True)
    print("Joints:", robot.joint_names, flush=True)
    if len(robot.joint_names) != 14:
        raise RuntimeError(f"Expected 14 joints (10 driven, 4 passive); found {len(robot.joint_names)}")

    q0 = robot.data.default_joint_pos.clone()
    qvel0 = torch.zeros_like(q0)
    joint_ids = {name: robot.joint_names.index(name) for name in actuated}
    actuator_ids = [joint_ids[name] for name in actuated]
    root_prim = robot.root_view.prim_paths[0]
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache()

    pin_paths = {
        side: (
            root_prim + f"/Geometry/hip/{side}_leg1/{side}_leg2/{side}_leg3/{side}_leg4/{side}_foot/{side}_heel_linkage/{side}_heel_bar_pin",
            root_prim + f"/Geometry/hip/{side}_leg1/{side}_leg2/{side}_leg3/{side}_leg4/{side}_ankle_crank/{side}_crank_pin",
        )
        for side in ("left", "right")
    }
    foot_paths = {
        side: root_prim + f"/Geometry/hip/{side}_leg1/{side}_leg2/{side}_leg3/{side}_leg4/{side}_foot/{side}_foot"
        for side in ("left", "right")
    }

    def observe() -> dict:
        cache.Clear()
        gaps = {}
        for side, paths in pin_paths.items():
            a, b = [
                cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation()
                for path in paths
            ]
            gaps[side] = float((a - b).GetLength())
        feet = {
            side: cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation()
            for side, path in foot_paths.items()
        }
        result = {
            "root_pos": robot.data.root_pos_w[0].tolist(),
            "root_quat_xyzw": robot.data.root_quat_w[0].tolist(),
            "joint_pos": robot.data.joint_pos[0].tolist(),
            "foot_site_z": {side: float(pos[2]) for side, pos in feet.items()},
            "pin_gap_m": gaps,
        }
        for field in ("joint_pos_target", "applied_torque", "computed_torque"):
            value = getattr(robot.data, field, None)
            if value is not None:
                result[field] = value[0].tolist()
        return result

    def step_to(target: torch.Tensor, n_steps: int, sample: bool = False) -> list[dict]:
        samples = []
        for i in range(n_steps):
            robot.set_joint_position_target_index(
                target=target[:, actuator_ids], joint_ids=actuator_ids
            )
            scene.write_data_to_sim()
            simulation.step()
            scene.update(simulation.get_physics_dt())
            if sample and i % 6 == 0:
                record = observe()
                record["joint_pos_target_cmd"] = target[0].tolist()
                samples.append(record)
        return samples

    results = []
    all_samples = []
    test_baselines = []

    # A 0.035 rad excursion is deliberately small. Knees move negative because
    # their authored zero is the straight-leg limit. Each joint starts from a
    # fresh default state and gets a half-second hold.
    ramp_steps, hold_steps, settle_steps = 60, 60, 60
    for name in actuated:
        robot.reset()
        robot.write_joint_state_to_sim(q0, qvel0)
        scene.write_data_to_sim()
        simulation.step()
        scene.update(simulation.get_physics_dt())
        step_to(q0, 1)
        baseline = observe()
        baseline_joints = robot.data.joint_pos.clone()
        baseline_root = baseline["root_pos"]
        baseline_tilt = baseline["root_quat_wxyz"]
        test_baselines.append({
            "joint": name,
            "root_pos_m": baseline_root,
            "root_quat_xyzw": baseline_tilt,
            "joint_pos_rad": baseline_joints[0].tolist(),
            "foot_site_z_m": baseline["foot_site_z"],
        })

        index = joint_ids[name]
        delta = -0.035 if "knee" in name else 0.035
        target = baseline_joints.clone()
        target[0, index] += delta
        samples = []

        for i in range(ramp_steps):
            alpha = (i + 1) / ramp_steps
            phase = baseline_joints.clone()
            phase[0, index] += alpha * delta
            samples.extend(step_to(phase, 1, sample=(i % 6 == 0)))
        hold_samples = step_to(target, hold_steps, sample=True)
        samples.extend(hold_samples)
        for i in range(ramp_steps):
            alpha = 1.0 - (i + 1) / ramp_steps
            phase = baseline_joints.clone()
            phase[0, index] += alpha * delta
            samples.extend(step_to(phase, 1, sample=(i % 6 == 0)))
        samples.extend(step_to(baseline_joints, settle_steps, sample=True))

        baseline_joint = float(baseline_joints[0, index])
        measured = [s["joint_pos"][index] - baseline_joint for s in samples]
        hold_errors = [
            abs(s["joint_pos"][index] - (baseline_joint + delta))
            for s in hold_samples
        ]
        max_root_drift = max(
            ((sum((s["root_pos"][k] - baseline_root[k]) ** 2 for k in range(3))) ** 0.5)
            for s in samples
        )
        max_root_tilt = max(
            2.0 * math.acos(min(1.0, abs(sum(a * b for a, b in zip(baseline_tilt, s["root_quat_wxyz"])))) )
            for s in samples
        )
        max_pin_gap = max(max(s["pin_gap_m"].values()) for s in samples)
        result = {
            "joint": name,
            "target_delta_rad": delta,
            "baseline_joint_pos_rad": baseline_joint,
            "joint_index": index,
            "measured_min_rad": min(measured),
            "measured_max_rad": max(measured),
            "max_abs_tracking_error_rad": max(hold_errors, default=None),
            "max_root_drift_m": max_root_drift,
            "max_root_tilt_delta_rad": max_root_tilt,
            "max_pin_gap_m": max_pin_gap,
            "min_foot_site_z_m": min(s["foot_site_z"][side] for s in samples for side in ("left", "right")),
        }
        results.append(result)
        all_samples.extend({"joint_test": name, **s} for s in samples)
        print(json.dumps(result), flush=True)

    final = observe()
    summary = {
        "usd": str(usd),
        "physics_engine": "PhysX",
        "dt_s": dt,
        "baseline_per_joint_test": test_baselines,
        "steps_per_joint_test": 361,
        "joint_results": results,
        "final_root_pos_m": final["root_pos"],
        "final_joint_pos_rad": final["joint_pos"],
        "final_pin_gap_m": final["pin_gap_m"],
        "samples": all_samples,
    }
    out = Path("logs/lower_body_joint_motion_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
