"""Resolve and view the repository-pinned Unitree G1 MuJoCo model.

This beginner-facing script is intentionally thin: the same resolver used by
training lives in :mod:`source.robots.unitree_g1.model`.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.robots.unitree_g1.model import (
    DEFAULT_SCENE,
    MENAGERIE_COMMIT,
    MENAGERIE_ROOT_ENV,
    MJX_SCENE,
    ROBOT_NAME,
    UNITREE_G1_MODEL_ENV,
    default_cache_root,
    resolve_unitree_g1_model,
)


def import_mujoco() -> ModuleType:
    """Import MuJoCo with an actionable beginner-facing failure message."""

    try:
        return importlib.import_module("mujoco")
    except ModuleNotFoundError as error:
        raise SystemExit(
            "MuJoCo is not installed in this Python environment.\n"
            "Install dependencies with:\n  python -m pip install -r requirements.txt"
        ) from error


def choose_keyframe(mujoco_module: ModuleType, model: Any, requested: str) -> int | None:
    """Return the requested/first conventional standing keyframe id."""

    if requested == "none":
        return None
    names = [requested] if requested != "auto" else ["stand", "home", "knees_bent"]
    for name in names:
        key_id = mujoco_module.mj_name2id(
            model, mujoco_module.mjtObj.mjOBJ_KEY, name
        )
        if key_id >= 0:
            return key_id
    if requested != "auto":
        available = [
            mujoco_module.mj_id2name(model, mujoco_module.mjtObj.mjOBJ_KEY, index)
            for index in range(model.nkey)
        ]
        raise SystemExit(
            f"Keyframe {requested!r} was not found. Available keyframes: {available}"
        )
    return None


def load_model(
    mujoco_module: ModuleType, scene_path: Path, keyframe: str
) -> tuple[Any, Any]:
    """Load and initialize one resolved scene."""

    try:
        model = mujoco_module.MjModel.from_xml_path(str(scene_path))
    except Exception as error:
        raise SystemExit(
            f"MuJoCo failed to load:\n  {scene_path}\n\nError: {error}\n\n"
            "Ensure the complete unitree_g1 assets directory is beside the scene."
        ) from error
    data = mujoco_module.MjData(model)
    key_id = choose_keyframe(mujoco_module, model, keyframe)
    if key_id is not None:
        mujoco_module.mj_resetDataKeyframe(model, data, key_id)
    mujoco_module.mj_forward(model, data)
    return model, data


def print_model_summary(scene_path: Path, model: Any, revision: str, pinned: bool) -> None:
    """Print model dimensions and provenance without serializing assets."""

    print(f"Resolved Unitree G1 model path: {scene_path}")
    print(f"Model revision: {revision} (pinned={str(pinned).lower()})")
    print(
        "Model stats: "
        f"nq={model.nq}, nv={model.nv}, nu={model.nu}, "
        f"ngeom={model.ngeom}, nmesh={model.nmesh}, nkey={model.nkey}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the public Unitree G1 robot model in MuJoCo."
    )
    parser.add_argument("--robot", default=ROBOT_NAME, choices=[ROBOT_NAME])
    parser.add_argument(
        "--menagerie-root",
        help=f"Local Menagerie checkout (or set {MENAGERIE_ROOT_ENV}).",
    )
    parser.add_argument(
        "--model",
        help=f"Explicit local scene override (or set {UNITREE_G1_MODEL_ENV}).",
    )
    parser.add_argument(
        "--mjx-scene",
        action="store_true",
        help=f"Use {MJX_SCENE}; the viewer default is {DEFAULT_SCENE}.",
    )
    parser.add_argument("--keyframe", default="auto")
    parser.add_argument("--cache-root", default=str(default_cache_root()))
    parser.add_argument(
        "--no-fetch-model", action="store_false", dest="fetch_model"
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.set_defaults(fetch_model=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        resolved = resolve_unitree_g1_model(
            mjx_scene=args.mjx_scene,
            model_path=args.model,
            menagerie_root=args.menagerie_root,
            cache_root=args.cache_root,
            fetch=args.fetch_model,
        )
    except (FileNotFoundError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    mujoco_module = import_mujoco()
    model, data = load_model(mujoco_module, resolved.scene_path, args.keyframe)
    print_model_summary(
        resolved.scene_path, model, resolved.revision, resolved.pinned
    )
    if args.no_viewer:
        return
    print(
        f"Opening MuJoCo viewer at Menagerie revision {MENAGERIE_COMMIT}. "
        "Close the window to exit."
    )
    viewer = importlib.import_module("mujoco.viewer")
    viewer.launch(model, data)


if __name__ == "__main__":
    main()
