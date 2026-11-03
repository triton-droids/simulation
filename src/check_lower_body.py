"""Short PhysX ground-contact test using the MJCF-authored actuator gains."""

from __future__ import annotations

import json
import time
from pathlib import Path

import antioch


def main() -> None:
    antioch.start_simulation()

    import isaaclab.sim as sim_utils
    import torch
    import omni.usd
    from pxr import UsdGeom
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
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
            spawn=sim_utils.UsdFileCfg(usd_path=str(usd), activate_contact_sensors=True),
            # The hip is now the physical root; at joint zero the soles sit
            # about 1 cm above the floor when the hip root is at z ~= 0.01 m.
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.02)),
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

    simulation = SimulationContext(SimulationCfg(dt=1.0 / 120.0, render_interval=1))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    simulation.reset()
    robot: Articulation = scene["robot"]
    set_camera_view(eye=[1.7, 1.7, 1.15], target=[0.0, 0.0, 0.55], camera_prim_path="/OmniverseKit_Persp")
    print("USD articulation joints:", robot.joint_names, flush=True)
    print("USD articulation bodies:", robot.body_names, flush=True)
    root_prim = robot.root_physx_view.prim_paths[0]
    print("USD articulation root:", root_prim, flush=True)
    print("Default joint pose:", robot.data.default_joint_pos[0].tolist(), flush=True)

    q0 = robot.data.default_joint_pos.clone()
    actuator_ids = [robot.joint_names.index(name) for name in actuated]
    records = []
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache()
    pin_paths = {
        side: (
            root_prim + f"/Geometry/hip/{side}_leg1/{side}_leg2/"
            f"{side}_leg3/{side}_leg4/{side}_foot/{side}_heel_linkage/{side}_heel_bar_pin",
            root_prim + f"/Geometry/hip/{side}_leg1/{side}_leg2/"
            f"{side}_leg3/{side}_leg4/{side}_ankle_crank/{side}_crank_pin",
        )
        for side in ("left", "right")
    }
    steps = 600
    for i in range(steps):
        robot.set_joint_position_target_index(
            target=q0[:, actuator_ids], joint_ids=actuator_ids
        )
        scene.write_data_to_sim()
        simulation.step()
        scene.update(simulation.get_physics_dt())
        if i % 60 == 0 or i == steps - 1:
            root = robot.data.root_pos_w[0]
            quat = robot.data.root_quat_w[0]
            q = robot.data.joint_pos[0]
            cache.Clear()
            closure = {}
            for side, paths in pin_paths.items():
                a, b = [cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation() for path in paths]
                closure[side] = float((a - b).GetLength())
            rec = {"t_s": round(i * simulation.get_physics_dt(), 3), "root_xyz_m": root.tolist(), "root_quat_xyzw": quat.tolist(), "joint_pos": q.tolist(), "pin_gap_m": closure}
            records.append(rec)
            print(json.dumps(rec), flush=True)

    out = Path("logs/lower_body_physics_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"usd": str(usd), "steps": steps, "records": records}, indent=2) + "\n")
    print(f"Wrote {out}", flush=True)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        simulation.render()


if __name__ == "__main__":
    main()
