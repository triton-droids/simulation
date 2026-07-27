"""Load the public Unitree G1 robot in MuJoCo.

This is the beginner-facing entry point for the RoboCup simulator workspace.
It resolves the pinned Unitree G1 model from MuJoCo Menagerie, downloads the
model into a hidden local cache when needed, loads the MJCF with MuJoCo's
Python API, and opens the MuJoCo viewer by default.

The loader intentionally stays independent from the existing MJX/Brax training
path. Loading the robot is the first reproducibility milestone; connecting
Unitree G1 to locomotion environments and policies is a later integration task.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_COMMIT = "71f066ad0be9cd271f7ed58c030243ef157af9f4"
MENAGERIE_REPO_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
ROBOT_NAME = "unitree_g1"
UNITREE_G1_MODEL_ENV = "UNITREE_G1_MODEL_PATH"
MENAGERIE_ROOT_ENV = "MUJOCO_MENAGERIE_PATH"
DEFAULT_SCENE = "scene.xml"
MJX_SCENE = "scene_mjx.xml"


def repo_root() -> Path:
    """Return the repository root based on this script's location.

    Returns:
        Absolute path to the project root.

    Failure cases:
        This assumes the script remains at `scripts/load_unitree_g1.py`.
    """

    return PROJECT_ROOT


def default_cache_root() -> Path:
    """Return the hidden MuJoCo Menagerie checkout used by default.

    Returns:
        A path under `.cache/`, which is ignored by git and excluded from export
        zips so downloaded meshes are not accidentally committed.
    """

    return repo_root() / ".cache" / "mujoco_menagerie"


def unique_paths(paths: list[Path]) -> list[Path]:
    """Remove duplicate paths while preserving search priority.

    Args:
        paths: Candidate filesystem paths.

    Returns:
        De-duplicated paths in their original order.
    """

    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        expanded_path = path.expanduser()
        key = str(expanded_path.resolve() if expanded_path.exists() else expanded_path)
        if key not in seen:
            unique.append(expanded_path)
            seen.add(key)
    return unique


def selected_scene_name(use_mjx_scene: bool) -> str:
    """Choose the Menagerie scene file to load.

    Args:
        use_mjx_scene: Whether to select the MJX-oriented scene.

    Returns:
        Scene filename inside the `unitree_g1/` model folder.
    """

    return MJX_SCENE if use_mjx_scene else DEFAULT_SCENE


def candidate_menagerie_roots(cli_root: str | None, cache_root: str | None) -> list[Path]:
    """Build the ordered list of Menagerie checkout roots to search.

    Args:
        cli_root: Optional checkout root passed with `--menagerie-root`.
        cache_root: Hidden checkout path used by the auto-download flow.

    Returns:
        Candidate directories that may contain `unitree_g1/scene.xml`.
    """

    candidates: list[Path] = []
    if cli_root:
        candidates.append(Path(cli_root))

    env_root = os.environ.get(MENAGERIE_ROOT_ENV)
    if env_root:
        candidates.append(Path(env_root))

    if cache_root:
        candidates.append(Path(cache_root))

    candidates.extend(
        [
            repo_root().parent / "mujoco_menagerie",
            repo_root() / "mujoco_menagerie",
        ]
    )
    return unique_paths(candidates)


def candidate_scene_paths(args: argparse.Namespace) -> list[Path]:
    """Build the ordered list of direct model scene paths to try.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Candidate scene files from explicit paths, environment variables, and
        common Menagerie checkout locations.
    """

    scene_paths: list[Path] = []
    scene_name = selected_scene_name(args.mjx_scene)

    # Explicit model paths win because they are the clearest user intent.
    if args.model:
        scene_paths.append(Path(args.model))

    env_model_path = os.environ.get(UNITREE_G1_MODEL_ENV)
    if env_model_path:
        scene_paths.append(Path(env_model_path))

    for menagerie_root in candidate_menagerie_roots(args.menagerie_root, args.cache_root):
        scene_paths.append(menagerie_root / ROBOT_NAME / scene_name)

    return unique_paths(scene_paths)


def run_git_command(command: list[str]) -> None:
    """Run a git command and convert failures into plain-English messages.

    Args:
        command: Complete git command as a list of arguments.

    Side effects:
        Runs a subprocess that may create or update the hidden model cache.

    Failure cases:
        Exits when git returns a non-zero status.
    """

    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        command_text = " ".join(command)
        details = (error.stderr or error.stdout or "").strip()
        raise SystemExit(f"Git command failed:\n  {command_text}\n\n{details}") from error


def fetch_unitree_g1_model(cache_root: Path) -> Path:
    """Fetch the pinned Unitree G1 model into the hidden local cache.

    Args:
        cache_root: Local MuJoCo Menagerie checkout root to create or update.

    Returns:
        Path to the downloaded `unitree_g1/scene.xml` file.

    Side effects:
        Creates or updates `.cache/mujoco_menagerie` with a sparse checkout.

    Failure cases:
        Exits if git is unavailable, the cache path is invalid, or the expected
        scene file is still missing after checkout.
    """

    if shutil.which("git") is None:
        raise SystemExit(
            "The Unitree G1 model is not cached yet, and git is not available.\n"
            "Install git, pass --model path/to/unitree_g1/scene.xml, or set "
            f"{UNITREE_G1_MODEL_ENV}."
        )

    cache_root = cache_root.expanduser()
    cache_root.parent.mkdir(parents=True, exist_ok=True)

    # Refuse to overwrite a user-created folder that is not a git checkout.
    if cache_root.exists() and not (cache_root / ".git").exists():
        raise SystemExit(
            f"Cannot download into {cache_root} because it exists but is not a git checkout.\n"
            "Move that folder or pass --model with a direct scene.xml path."
        )

    if not cache_root.exists():
        print(f"Downloading pinned Unitree G1 model into hidden cache: {cache_root}")
        run_git_command(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                MENAGERIE_REPO_URL,
                str(cache_root),
            ]
        )
    else:
        print(f"Using existing hidden model cache: {cache_root}")

    # Pin the source revision so teammates all load the same model files.
    run_git_command(["git", "-C", str(cache_root), "fetch", "--filter=blob:none", "origin", MENAGERIE_COMMIT])
    run_git_command(["git", "-C", str(cache_root), "checkout", MENAGERIE_COMMIT])
    run_git_command(["git", "-C", str(cache_root), "sparse-checkout", "set", ROBOT_NAME])

    scene_path = cache_root / ROBOT_NAME / DEFAULT_SCENE
    if not scene_path.exists():
        raise SystemExit(f"Download finished, but the expected model file is missing: {scene_path}")
    return scene_path


def missing_model_message(tried_paths: list[Path]) -> str:
    """Build actionable instructions for missing model files.

    Args:
        tried_paths: Candidate paths that did not exist.

    Returns:
        A multi-line error message suitable for `SystemExit`.
    """

    tried_text = "\n  ".join(str(path) for path in tried_paths)
    return (
        "Could not find the Unitree G1 scene file.\n\n"
        "Try the default auto-download again without --no-fetch-model:\n"
        "  python scripts/load_unitree_g1.py\n\n"
        "Or provide a model path manually:\n"
        "  python scripts/load_unitree_g1.py --model path/to/unitree_g1/scene.xml\n\n"
        "Environment variable overrides are also supported:\n"
        f"  {UNITREE_G1_MODEL_ENV}=path/to/unitree_g1/scene.xml\n"
        f"  {MENAGERIE_ROOT_ENV}=path/to/mujoco_menagerie\n\n"
        f"Paths checked:\n  {tried_text}"
    )


def resolve_scene_path(args: argparse.Namespace) -> Path:
    """Resolve or download the Unitree G1 scene file.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Path to the selected MJCF scene file.

    Side effects:
        May create `.cache/mujoco_menagerie` when the model is missing.

    Failure cases:
        Exits with setup instructions when no scene file can be found.
    """

    tried_paths = candidate_scene_paths(args)
    for scene_path in tried_paths:
        if scene_path.exists():
            return scene_path

    if args.fetch_model and not args.model:
        fetched_scene = fetch_unitree_g1_model(Path(args.cache_root))
        scene_path = fetched_scene.with_name(selected_scene_name(args.mjx_scene))
        if scene_path.exists():
            return scene_path

    raise SystemExit(missing_model_message(tried_paths))


def import_mujoco() -> ModuleType:
    """Import MuJoCo after model resolution so dependency errors are clear.

    Returns:
        Imported `mujoco` module.

    Failure cases:
        Exits with the exact beginner install command when MuJoCo is missing.
    """

    try:
        return importlib.import_module("mujoco")
    except ModuleNotFoundError as error:
        raise SystemExit(
            "MuJoCo is not installed in this Python environment.\n"
            "Install the beginner dependency set with:\n"
            "  python -m pip install -r requirements.txt"
        ) from error


def choose_keyframe(mujoco_module: ModuleType, model: Any, requested: str) -> int | None:
    """Find the keyframe to apply before opening the viewer.

    Args:
        mujoco_module: Imported MuJoCo module.
        model: Loaded MuJoCo model.
        requested: Keyframe name, `auto`, or `none`.

    Returns:
        MuJoCo keyframe id, or `None` when no keyframe should be applied.

    Failure cases:
        Exits when the user requested a specific keyframe that does not exist.
    """

    if requested == "none":
        return None

    keyframe_names = [requested]
    if requested == "auto":
        keyframe_names = ["stand", "home", "knees_bent"]

    for keyframe_name in keyframe_names:
        key_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_KEY, keyframe_name)
        if key_id >= 0:
            return key_id

    if requested != "auto":
        available_keyframes = [
            mujoco_module.mj_id2name(model, mujoco_module.mjtObj.mjOBJ_KEY, index)
            for index in range(model.nkey)
        ]
        raise SystemExit(f"Keyframe '{requested}' was not found. Available keyframes: {available_keyframes}")
    return None


def load_model(mujoco_module: ModuleType, scene_path: Path, keyframe: str) -> tuple[Any, Any]:
    """Load the selected MJCF scene and initialize simulation data.

    Args:
        mujoco_module: Imported MuJoCo module.
        scene_path: Path to the scene XML file.
        keyframe: Keyframe name, `auto`, or `none`.

    Returns:
        A `(model, data)` pair ready for MuJoCo viewing or validation.

    Failure cases:
        Exits with the MuJoCo load error if XML includes or mesh assets fail.
    """

    try:
        model = mujoco_module.MjModel.from_xml_path(str(scene_path))
    except Exception as error:
        raise SystemExit(
            f"MuJoCo failed to load:\n  {scene_path}\n\n"
            f"Error: {error}\n\n"
            "If this mentions a missing mesh, make sure the whole "
            "`unitree_g1/assets/` folder exists next to the scene file."
        ) from error

    data = mujoco_module.MjData(model)

    # Start from a known pose so the viewer opens with the robot standing.
    key_id = choose_keyframe(mujoco_module, model, keyframe)
    if key_id is not None:
        mujoco_module.mj_resetDataKeyframe(model, data, key_id)
    mujoco_module.mj_forward(model, data)

    return model, data


def print_model_summary(scene_path: Path, model: Any) -> None:
    """Print a concise confirmation that the robot loaded.

    Args:
        scene_path: Path that was loaded.
        model: Loaded MuJoCo model.

    Side effects:
        Writes model statistics to stdout.
    """

    print(f"Resolved Unitree G1 model path: {scene_path}")
    print(
        "Model stats: "
        f"nq={model.nq}, nv={model.nv}, nu={model.nu}, "
        f"ngeom={model.ngeom}, nmesh={model.nmesh}, nkey={model.nkey}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for IDE and terminal usage.

    Returns:
        Parsed arguments. Defaults are chosen so
        `python scripts/load_unitree_g1.py` downloads the pinned Unitree G1
        model if needed and opens the viewer.
    """

    parser = argparse.ArgumentParser(
        description="Load the public Unitree G1 robot model in MuJoCo."
    )
    parser.add_argument(
        "--robot",
        default=ROBOT_NAME,
        choices=[ROBOT_NAME],
        help="Robot to load. Only unitree_g1 is supported for this milestone.",
    )
    parser.add_argument(
        "--menagerie-root",
        help="Path to a local google-deepmind/mujoco_menagerie checkout.",
    )
    parser.add_argument(
        "--model",
        help=(
            "Direct path to a Unitree G1 MJCF scene file. "
            f"Also available through {UNITREE_G1_MODEL_ENV}."
        ),
    )
    parser.add_argument(
        "--mjx-scene",
        action="store_true",
        help="Load unitree_g1/scene_mjx.xml instead of unitree_g1/scene.xml.",
    )
    parser.add_argument(
        "--keyframe",
        default="auto",
        help="Keyframe to apply before viewing. Use 'none' to keep MuJoCo defaults.",
    )
    parser.add_argument(
        "--cache-root",
        default=str(default_cache_root()),
        help="Hidden Menagerie checkout path used by auto-download.",
    )
    parser.add_argument(
        "--no-fetch-model",
        action="store_false",
        dest="fetch_model",
        help="Do not auto-download the model; print setup instructions instead.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Load and print model stats without opening the interactive viewer.",
    )
    parser.set_defaults(fetch_model=True)
    return parser.parse_args()


def main() -> None:
    """Run the Unitree G1 loading workflow.

    Workflow:
        1. Resolve or download the model scene path.
        2. Import MuJoCo and load the MJCF.
        3. Print the resolved path and model statistics.
        4. Open the interactive viewer unless `--no-viewer` was passed.

    Side effects:
        May create `.cache/mujoco_menagerie` and opens a MuJoCo viewer window
        by default.
    """

    args = parse_args()
    scene_path = resolve_scene_path(args)
    mujoco_module = import_mujoco()
    model, data = load_model(mujoco_module, scene_path, args.keyframe)
    print_model_summary(scene_path, model)

    if args.no_viewer:
        return

    # Import lazily so non-GUI validation works in terminals without graphics.
    print("Opening MuJoCo viewer. Close the viewer window to exit.")
    viewer = importlib.import_module("mujoco.viewer")
    viewer.launch(model, data)


if __name__ == "__main__":
    main()
