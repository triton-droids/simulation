"""
Drive a cartpole under a sine effort in Isaac Lab.

    antioch session new
    antioch service exec -- python src/main.py
    antioch service exec --stream -- python src/main.py
    antioch service exec -- python src/main.py --seconds 5
    antioch service exec -- python src/main.py --seconds 600

The argument makes it easy to change the run without editing the file.
"""

from __future__ import annotations

import argparse
import math
import time

import antioch


def main() -> None:
    """
    Step a cartpole scene with rendering on until the deadline.

    It runs headless unless started with ``--stream``, which shows it in
    Mission Control.
    """

    parser = argparse.ArgumentParser(description="Drive a cartpole under a sine effort")
    parser.add_argument("--seconds", type=float, default=30.0, help="How long to simulate before exiting")
    arguments = parser.parse_args()

    antioch.start_simulation()

    import isaaclab.sim as sim_utils
    import torch  # pyright: ignore[reportMissingImports] — the Isaac Lab engine ships torch
    from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils import configclass
    from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
    from isaacsim.core.utils.viewports import set_camera_view

    @configclass
    class CartpoleSceneCfg(InteractiveSceneCfg):
        """
        One cartpole on a ground plane under a dome light.
        """

        ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=200.0))
        key_light = AssetBaseCfg(prim_path="/World/KeyLight", spawn=sim_utils.DistantLightCfg(intensity=400.0))
        cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cartpole")

    simulation = SimulationContext(SimulationCfg(dt=1.0 / 60.0))
    scene = InteractiveScene(CartpoleSceneCfg(num_envs=1, env_spacing=2.0))
    simulation.reset()
    set_camera_view(eye=[3.4, 3.4, 2.4], target=[0.0, 0.0, 1.2], camera_prim_path="/OmniverseKit_Persp")
    robot: Articulation = scene["cartpole"]

    # Render every step so a GUI stream, when available, shows the motion.
    deadline = time.monotonic() + arguments.seconds
    step = 0
    while time.monotonic() < deadline:
        effort = 2.0 * math.sin(step / 30.0)
        robot.set_joint_effort_target_index(target=torch.full_like(robot.data.joint_pos[:, [0]], effort), joint_ids=[0])
        scene.write_data_to_sim()
        simulation.step()
        scene.update(simulation.get_physics_dt())
        step += 1
    print(f"drove the cartpole for {step} steps over {arguments.seconds:g}s")


if __name__ == "__main__":
    main()
