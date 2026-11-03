"""
Example scenarios for Isaac Lab: one fast check, and one parameter sweep.

    antioch scenario run cartpole_balance    one run
    antioch suite run smoke                  the fast check
    antioch suite run sweep                  every case in authored order

The sweep shows how one scenario declaration expands into independent cases.
Antioch runs each scenario with its saved source and service images on a
session it starts or reuses. Use `--session SESSION_ID` to run one at a time
on an existing session with the same images. Each run installs its saved source.

`cartpole_balance` publishes review frames; `cartpole_drive` is deliberately
scalar-only, because six sweep cases want one comparable chart rather than six
sets of frames.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import antioch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

logger = antioch.Logger("cartpole")


def build_cartpole() -> tuple[SimulationContext, InteractiveScene, Articulation]:
    """
    Bring up one cartpole on a ground plane, ready to step.

    Isaac Lab imports live inside the body, not at module scope: that is the
    rule that lets this file be discovered and type-checked on a laptop with
    no simulator installed.

    :return: The simulation context, its scene, and the cartpole articulation.
    """

    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
    # Without this the active camera keeps its default vantage looking at the
    # world origin from several metres out, which renders a cartpole as a
    # speck. A side three-quarter view keeps the rail and the swinging pole
    # in frame for both an available GUI stream and the captured review frames.
    #
    # The target sits above the cart, not at it: aimed at 1.2 the pivot landed
    # a third of the way down the frame and an upright pole ran off the top
    # edge, which is exactly the pose worth seeing.
    set_camera_view(eye=[4.0, 4.0, 2.8], target=[0.0, 0.0, 1.7], camera_prim_path="/OmniverseKit_Persp")
    return simulation, scene, scene["cartpole"]


def drive(simulation: SimulationContext, scene: InteractiveScene, robot: Articulation, step: int, amplitude: float, period: float) -> None:
    """
    Advance one physics step under a sinusoidal cart effort.

    :param simulation: The running simulation context.
    :param scene: The interactive scene holding the cartpole.
    :param robot: The cartpole articulation.
    :param step: Index of this step, which sets the phase.
    :param amplitude: Peak effort applied to the cart joint.
    :param period: Steps per full effort cycle.
    """

    import torch  # pyright: ignore[reportMissingImports] — the Isaac Lab engine ships torch

    effort = amplitude * math.sin(2.0 * math.pi * step / period)
    robot.set_joint_effort_target_index(target=torch.full_like(robot.data.joint_pos[:, [0]], effort), joint_ids=[0])
    scene.write_data_to_sim()
    simulation.step()
    scene.update(simulation.get_physics_dt())


# The pictures below are authored: aimed at the rail by `build_cartpole`, and
# published only when the frame is worth reviewing.
@antioch.scenario(tags=["smoke"])
def cartpole_balance(run: antioch.ScenarioRun, steps: int = antioch.param(300, ge=1, description="Physics steps to simulate")) -> None:
    """
    Push an Isaac Lab cartpole and verify that its state stays finite.
    """

    import numpy as np

    simulation, scene, robot = build_cartpole()
    # The first capture of a run hands back a frame from before the scene was
    # drawn — measured as the bare Kit grid with no cartpole in it. It follows
    # the first call, not any particular step, so no cadence avoids it, and it
    # sits well inside the exposure band, so no check below can tell it from a
    # real picture. Spend it here and the review set never contains it.
    antioch.capture_viewport()

    attempted = 0
    review_frames = 0
    for step in range(steps):
        drive(simulation, scene, robot, step, amplitude=2.0, period=180.0)
        logger.scalar("pole_angle", float(robot.data.joint_pos[0, 1]))
        # The effort cycle is 180 steps, so sampling every twentieth step
        # photographs the swing rather than one phase of it
        if step % 20 == 0:
            attempted += 1
            frame = antioch.capture_viewport()
            if frame is not None:
                rgb = np.asarray(frame)[:, :, :3]
                # A black or blown-out picture is worse evidence than none
                if 10.0 <= float(rgb.mean()) <= 220.0:
                    logger.image("camera/rgb", rgb)
                    review_frames += 1

    rail_x = float(robot.data.root_pos_w[0, 0])
    final_angle = float(robot.data.joint_pos[0, 1])
    run.add_result("final_pole_angle", round(final_angle, 4))
    # Both counts, so a run that published nothing says so instead of looking
    # like a run that never tried
    run.add_result("review_frames", review_frames)
    run.add_result("review_frames_attempted", attempted)
    # Every criterion is recorded even when an earlier one fails, so a run that
    # went non-finite still reports where the cart ended up
    run.check("the pole angle stayed finite", math.isfinite(final_angle), detail=f"final angle {final_angle}")
    run.check("the cart stayed within a metre of the origin", abs(rail_x) < 1.0, detail=f"cart at x={rail_x:.3f} m")
    run.check("the review camera published usable frames", review_frames > 0, detail=f"{review_frames} of {attempted} attempts passed the exposure gate")


@antioch.scenario(tags=["sweep"], cases=[antioch.case(grid={"amplitude": [0.5, 2.0, 6.0], "period": [30.0, 120.0]}, id="a{amplitude}-p{period}")])
def cartpole_drive(
    run: antioch.ScenarioRun,
    amplitude: float = antioch.param(2.0, ge=0.1, le=20.0, description="Peak effort applied to the cart joint"),
    period: float = antioch.param(60.0, ge=2.0, description="Steps per full effort cycle"),
    steps: int = antioch.param(300, ge=1, description="Physics steps to simulate"),
) -> None:
    """
    Drive the cart at varying amplitude and frequency, and bound the swing.

    Six children from one declaration: three efforts against two frequencies.
    """

    simulation, scene, robot = build_cartpole()
    swing = 0.0
    for step in range(steps):
        drive(simulation, scene, robot, step, amplitude=amplitude, period=period)
        angle = float(robot.data.joint_pos[0, 1])
        logger.scalar("pole_angle", angle)
        swing = max(swing, abs(angle))

    run.add_result("peak_pole_angle", round(swing, 4))
    run.check("the pole angle stayed finite", math.isfinite(swing), detail=f"peak swing {swing} under amplitude {amplitude}")
