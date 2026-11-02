"""Show the imported lower body in Antioch's Isaac Sim viewer.

Run from the project root with:
    uv run antioch service exec --stream -- python src/view_legs.py
    uv run antioch service exec --stream -- python src/view_legs.py --xml

This is a visual inspection. It does not validate the linkage physics.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import antioch


def main() -> None:
    parser = argparse.ArgumentParser(description="View the imported Triton lower body")
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--xml", action="store_true", help="Import the bundled MJCF before viewing")
    args = parser.parse_args()

    asset_dir = Path(__file__).resolve().parents[1] / "robots" / "humanoid_legs_5"
    usd_path = asset_dir / "humanoid_legs_5.usd"
    xml_path = asset_dir / "humanoid_legs_5.xml"
    if args.xml and not xml_path.is_file():
        raise FileNotFoundError(f"The MJCF is missing: {xml_path}")
    if not args.xml and not usd_path.is_file():
        raise FileNotFoundError(f"The packaged USD is missing; expected {usd_path}")

    antioch.start_simulation()

    import omni.usd
    import omni.kit.app
    from isaacsim.core.utils.viewports import set_camera_view
    from pxr import UsdGeom, UsdLux

    if args.xml:
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

        print(f"Importing {xml_path}", flush=True)
        converter = MjcfConverter(MjcfConverterCfg(
            asset_path=str(xml_path),
            usd_dir=str(asset_dir / "usd_import"),
            usd_file_name="humanoid_legs_5.usd",
            fix_base=False,
            make_instanceable=False,
        ))
        usd_path = Path(converter.usd_path)
        if not usd_path.is_file():
            raise FileNotFoundError(f"Importer did not produce a USD: {usd_path}")
        print(f"Imported USD: {usd_path}", flush=True)

    stage = omni.usd.get_context().get_stage()
    robot = UsdGeom.Xform.Define(stage, "/World/HumanoidLegs5")
    robot.GetPrim().GetReferences().AddReference(str(usd_path))
    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.CreateIntensityAttr(2500.0)
    set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.5], camera_prim_path="/OmniverseKit_Persp")

    print(f"Viewing {usd_path} for {args.seconds:g} seconds")
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        # Pump Kit directly: this static viewer has no Isaac Lab scene/visualizer.
        omni.kit.app.get_app().update()
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
