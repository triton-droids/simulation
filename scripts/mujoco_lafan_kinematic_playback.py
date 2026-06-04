#!/usr/bin/env python3
"""Kinematic MuJoCo playback for LAFAN policy/reference traces."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "tritonhumanoid"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

MUJOCO_LAFAN_PATH = SOURCE_ROOT / "tritonhumanoid" / "eval" / "mujoco_lafan.py"
spec = importlib.util.spec_from_file_location("mujoco_lafan_eval", MUJOCO_LAFAN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load MuJoCo LAFAN eval module from {MUJOCO_LAFAN_PATH}")
mujoco_lafan = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mujoco_lafan
spec.loader.exec_module(mujoco_lafan)

CONTROL_DT = mujoco_lafan.CONTROL_DT
DEFAULT_ACTIVE_URDF = mujoco_lafan.DEFAULT_ACTIVE_URDF
DEFAULT_MODEL_CACHE = mujoco_lafan.DEFAULT_MODEL_CACHE
ISAAC_ROOT_HEIGHT = mujoco_lafan.ISAAC_ROOT_HEIGHT
ensure_isaac_trained_mjcf = mujoco_lafan.ensure_isaac_trained_mjcf
sleep_for_realtime = mujoco_lafan.sleep_for_realtime
validate_mjcf_against_urdf = mujoco_lafan.validate_mjcf_against_urdf
FALLBACK_FLOATING_XML = DEFAULT_MODEL_CACHE / "human_offset_corrected_floating_visual.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="Playback .npz containing qpos and optional qvel.")
    parser.add_argument("--model-xml", type=Path, default=None, help="Override generated Isaac-trained MJCF path.")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means play the whole trace.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--refresh-model", action="store_true", help="Regenerate the cached Isaac-trained MJCF.")
    parser.add_argument(
        "--lafan-reference",
        action="store_true",
        help="Treat qpos as raw LAFAN reference qpos and add the Isaac root-height offset.",
    )
    parser.add_argument(
        "--enable-contact",
        action="store_true",
        help="Keep MuJoCo contact detection enabled during kinematic playback.",
    )
    parser.add_argument(
        "--enable-gravity",
        action="store_true",
        help="Keep MuJoCo gravity enabled during kinematic playback.",
    )
    parser.add_argument("--hide-floor", action="store_true", help="Hide floor/ground geoms during playback.")
    parser.add_argument("--output", type=Path, default=None, help="Optional copied playback .npz path.")
    return parser.parse_args()


def require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("numpy is required for MuJoCo kinematic playback.") from exc
    return np


def require_mujoco():
    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("mujoco is required for MuJoCo kinematic playback.") from exc
    return mujoco


def validate_trace(trace) -> None:
    if "qpos" not in trace:
        raise RuntimeError("Trace is missing required key 'qpos'.")
    if trace["qpos"].ndim != 2:
        raise RuntimeError("Trace qpos must be a rank-2 array.")
    if "qvel" in trace and trace["qvel"].ndim != 2:
        raise RuntimeError("Trace qvel must be a rank-2 array when present.")


def resolve_model_xml(args: argparse.Namespace) -> Path:
    if args.model_xml is not None:
        validate_mjcf_against_urdf(args.model_xml, DEFAULT_ACTIVE_URDF)
        return args.model_xml
    try:
        return ensure_isaac_trained_mjcf(
            model_dir=DEFAULT_MODEL_CACHE,
            urdf_path=DEFAULT_ACTIVE_URDF,
            refresh=args.refresh_model,
        )
    except Exception as exc:
        print(f"[WARN] Could not generate canonical Isaac-trained MJCF: {exc}")
        print("[WARN] Falling back to a floating visual MJCF generated from the local URDF.")
        return ensure_floating_visual_mjcf(refresh=args.refresh_model)


def ensure_mujoco_loadable_xml(xml_path: Path) -> Path:
    """Return an XML path that current MuJoCo can load.

    Some source models use a floating child body named ``world``. MuJoCo's
    built-in root body is also named world, so newer bindings reject that as a
    duplicate. Renaming the child does not change qpos/qvel ordering.
    """

    if xml_path == FALLBACK_FLOATING_XML:
        return xml_path
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return xml_path
    world_body = None
    for child in worldbody.findall("body"):
        if child.attrib.get("name") == "world":
            world_body = child
            break
    if world_body is None:
        return xml_path

    playback_xml = xml_path.with_name(xml_path.stem + "_playback.xml")
    if playback_xml.exists() and playback_xml.stat().st_mtime >= xml_path.stat().st_mtime:
        return playback_xml
    world_body.attrib["name"] = "floating_root"
    ET.indent(root, space="  ")
    playback_xml.write_text(ET.tostring(root, encoding="unicode"))
    return playback_xml


def ensure_floating_visual_mjcf(*, refresh: bool = False) -> Path:
    if FALLBACK_FLOATING_XML.exists() and not refresh:
        return FALLBACK_FLOATING_XML

    np = require_numpy()
    mujoco = require_mujoco()
    DEFAULT_MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    fixed_model = mujoco.MjModel.from_xml_path(str(DEFAULT_ACTIVE_URDF))
    tmp_xml = FALLBACK_FLOATING_XML.with_suffix(".fixed.xml")
    mujoco.mj_saveLastXML(str(tmp_xml), fixed_model)
    root = ET.parse(tmp_xml).getroot()
    tmp_xml.unlink(missing_ok=True)

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib["timestep"] = str(mujoco_lafan.PHYSICS_DT)
    option.attrib["gravity"] = "0 0 0"

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Generated URDF MJCF has no worldbody.")
    children = list(worldbody)
    for child in children:
        worldbody.remove(child)
    floating = ET.Element("body", {"name": "floating_root", "pos": f"0 0 {ISAAC_ROOT_HEIGHT}"})
    floating.append(
        ET.Element(
            "inertial",
            {
                "pos": "0 0 0",
                "mass": "0.001",
                "diaginertia": "0.000001 0.000001 0.000001",
            },
        )
    )
    floating.append(ET.Element("freejoint"))
    for child in children:
        floating.append(child)
    worldbody.append(floating)
    ensure_floor(root)

    ET.indent(root, space="  ")
    FALLBACK_FLOATING_XML.write_text(ET.tostring(root, encoding="unicode"))

    model = mujoco.MjModel.from_xml_path(str(FALLBACK_FLOATING_XML))
    if model.nq != 17 or model.nv != 16:
        raise RuntimeError(f"Fallback MJCF dimensions mismatch: nq={model.nq}, nv={model.nv}; expected 17/16.")
    joint_names = [model.joint(i).name for i in range(model.njnt) if model.joint(i).name]
    missing = [name for name in mujoco_lafan.CH_MUJOCO_JOINT_NAMES if name not in joint_names]
    if missing:
        raise RuntimeError(f"Fallback MJCF missing joints: {missing}")
    return FALLBACK_FLOATING_XML


def ensure_floor(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return
    if worldbody.find("./geom[@name='ground']") is None:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "ground",
                "type": "plane",
                "size": "10 10 0.1",
                "pos": "0 0 0",
                "rgba": "0.3 0.3 0.3 1",
            },
        )


def disable_contact(mujoco, model) -> None:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)


def disable_gravity(np, model) -> None:
    model.opt.gravity[:] = np.asarray([0.0, 0.0, 0.0], dtype=model.opt.gravity.dtype)


def hide_geoms(model, names: tuple[str, ...]) -> None:
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        if name in names:
            model.geom_rgba[geom_id, 3] = 0.0


def set_state_from_trace(np, mujoco, model, data, trace, step: int, *, lafan_reference: bool) -> None:
    qpos = np.asarray(trace["qpos"][step], dtype=np.float64)
    if qpos.shape[0] != model.nq:
        raise RuntimeError(f"qpos width mismatch: trace has {qpos.shape[0]}, model expects {model.nq}.")

    data.qpos[:] = qpos
    if lafan_reference:
        data.qpos[2] += ISAAC_ROOT_HEIGHT

    data.qvel[:] = 0.0
    if "qvel" in trace:
        qvel = np.asarray(trace["qvel"][step], dtype=np.float64)
        if qvel.shape[0] != model.nv:
            raise RuntimeError(f"qvel width mismatch: trace has {qvel.shape[0]}, model expects {model.nv}.")
        data.qvel[:] = qvel

    if model.nu > 0:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def run_validate_only(args: argparse.Namespace) -> None:
    np = require_numpy()
    trace = np.load(args.trace)
    validate_trace(trace)
    xml_path = resolve_model_xml(args)
    if xml_path != FALLBACK_FLOATING_XML:
        validate_mjcf_against_urdf(xml_path, DEFAULT_ACTIVE_URDF)
    print("MuJoCo LAFAN kinematic playback validation OK")
    print(f"  trace: {args.trace}")
    print(f"  patched MJCF: {xml_path}")
    print(f"  frames: {trace['qpos'].shape[0]}")
    print(f"  qpos width: {trace['qpos'].shape[1]}")
    if "qvel" in trace:
        print(f"  qvel width: {trace['qvel'].shape[1]}")


def main() -> None:
    args = parse_args()
    if args.validate_only:
        run_validate_only(args)
        return

    np = require_numpy()
    mujoco = require_mujoco()
    trace = np.load(args.trace)
    validate_trace(trace)

    source_model_xml = resolve_model_xml(args)
    model_xml = ensure_mujoco_loadable_xml(source_model_xml)
    if source_model_xml != FALLBACK_FLOATING_XML:
        validate_mjcf_against_urdf(source_model_xml, DEFAULT_ACTIVE_URDF, require_mujoco=False)
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)

    if not args.enable_contact:
        disable_contact(mujoco, model)
    if not args.enable_gravity:
        disable_gravity(np, model)
    if args.hide_floor:
        hide_geoms(model, ("floor", "ground"))

    frames = int(trace["qpos"].shape[0])
    if args.max_steps > 0:
        frames = min(frames, int(args.max_steps))

    viewer = None
    logs = {"time_s": [], "qpos": [], "qvel": []}
    try:
        if args.render:
            viewer = mujoco.viewer.launch_passive(model, data)
        for step in range(frames):
            start = time.time()
            set_state_from_trace(
                np,
                mujoco,
                model,
                data,
                trace,
                step,
                lafan_reference=args.lafan_reference,
            )
            logs["time_s"].append(step * CONTROL_DT)
            logs["qpos"].append(data.qpos.copy())
            logs["qvel"].append(data.qvel.copy())
            if viewer is not None:
                viewer.sync()
            if args.real_time:
                sleep_for_realtime(start, CONTROL_DT)
    finally:
        if viewer is not None:
            viewer.close()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **{key: np.asarray(value) for key, value in logs.items()})
    print(f"played_frames={frames}")


if __name__ == "__main__":
    main()
