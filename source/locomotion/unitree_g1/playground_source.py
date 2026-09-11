"""Resolve and load the exact MuJoCo Playground G1 implementation.

Only source code is loaded from this checkout.  Robot meshes and model assets
continue to come from the repository's existing pinned Menagerie resolver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import importlib.machinery
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAYGROUND_COMMIT = "8a4b4642d8eba8a80ac99ed125cb62c16e1457ad"
PLAYGROUND_REPO_URL = "https://github.com/google-deepmind/mujoco_playground.git"
PLAYGROUND_ROOT_ENV = "MUJOCO_PLAYGROUND_PATH"

SPARSE_PATHS = (
    "/pyproject.toml",
    "/LICENSE",
    "/mujoco_playground/_src/mjx_env.py",
    "/mujoco_playground/_src/gait.py",
    "/mujoco_playground/_src/wrapper.py",
    "/mujoco_playground/_src/locomotion/g1/",
)
REQUIRED_PATHS = (
    Path("LICENSE"),
    Path("mujoco_playground/_src/mjx_env.py"),
    Path("mujoco_playground/_src/gait.py"),
    Path("mujoco_playground/_src/wrapper.py"),
    Path("mujoco_playground/_src/locomotion/g1/base.py"),
    Path("mujoco_playground/_src/locomotion/g1/g1_constants.py"),
    Path("mujoco_playground/_src/locomotion/g1/joystick.py"),
    Path("mujoco_playground/_src/locomotion/g1/xmls/scene_mjx_feetonly.xml"),
)


@dataclass(frozen=True)
class ResolvedPlaygroundSource:
    root: Path
    revision: str
    pinned: bool

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["root"] = str(self.root)
        record.update(
            {
                "source": "mujoco-playground",
                "expected_revision": PLAYGROUND_COMMIT,
                "repository": PLAYGROUND_REPO_URL,
            }
        )
        return record


def default_playground_cache_root() -> Path:
    return PROJECT_ROOT / ".cache" / "mujoco_playground"


def _run_git(command: list[str], *, capture: bool = False) -> str:
    try:
        result = subprocess.run(
            command, check=True, text=True, capture_output=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        details = ""
        if isinstance(error, subprocess.CalledProcessError):
            details = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(f"Git command failed: {' '.join(command)}\n{details}") from error
    return result.stdout.strip() if capture else ""


def _checkout_revision(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        return _run_git(["git", "-C", str(root), "rev-parse", "HEAD"], capture=True)
    except RuntimeError:
        return None


def _missing_paths(root: Path) -> tuple[Path, ...]:
    return tuple(path for path in REQUIRED_PATHS if not (root / path).is_file())


def ensure_pinned_playground_checkout(cache_root: Path) -> Path:
    if shutil.which("git") is None:
        raise RuntimeError(
            "Pinned MuJoCo Playground source is not cached and git is unavailable."
        )
    cache_root = cache_root.expanduser()
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    if cache_root.exists() and not (cache_root / ".git").exists():
        raise RuntimeError(
            f"Refusing to overwrite non-git Playground cache: {cache_root}"
        )
    if not cache_root.exists():
        _run_git(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                PLAYGROUND_REPO_URL,
                str(cache_root),
            ]
        )
    _run_git(
        [
            "git",
            "-C",
            str(cache_root),
            "fetch",
            "--filter=blob:none",
            "origin",
            PLAYGROUND_COMMIT,
        ]
    )
    _run_git(
        ["git", "-C", str(cache_root), "checkout", "--detach", PLAYGROUND_COMMIT]
    )
    _run_git(
        [
            "git",
            "-C",
            str(cache_root),
            "sparse-checkout",
            "set",
            "--no-cone",
            *SPARSE_PATHS,
        ]
    )
    revision = _checkout_revision(cache_root)
    missing = _missing_paths(cache_root)
    if revision != PLAYGROUND_COMMIT or missing:
        raise RuntimeError(
            f"Invalid Playground cache revision={revision!r}, missing={missing!r}."
        )
    return cache_root


def resolve_playground_source(
    *,
    source_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    fetch: bool = True,
) -> ResolvedPlaygroundSource:
    cache = Path(cache_root).expanduser() if cache_root else default_playground_cache_root()
    candidates = []
    if source_root:
        candidates.append(Path(source_root).expanduser())
    if os.environ.get(PLAYGROUND_ROOT_ENV):
        candidates.append(Path(os.environ[PLAYGROUND_ROOT_ENV]).expanduser())
    candidates.append(cache)

    seen: set[str] = set()
    mismatches: list[str] = []
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        revision = _checkout_revision(candidate)
        missing = _missing_paths(candidate) if candidate.exists() else REQUIRED_PATHS
        if revision == PLAYGROUND_COMMIT and not missing:
            return ResolvedPlaygroundSource(candidate.resolve(), revision, True)
        if candidate.exists():
            mismatches.append(
                f"{candidate} (revision {revision or 'unknown'}, missing {len(missing)})"
            )

    if fetch:
        root = ensure_pinned_playground_checkout(cache)
        return ResolvedPlaygroundSource(root.resolve(), PLAYGROUND_COMMIT, True)

    detail = "\n  ".join(mismatches) if mismatches else "none found"
    raise FileNotFoundError(
        f"No complete MuJoCo Playground checkout at {PLAYGROUND_COMMIT}.\n"
        f"Mismatched candidates:\n  {detail}\n"
        f"Set {PLAYGROUND_ROOT_ENV}, pass a source root, or allow fetching."
    )


def _install_namespace(name: str, path: Path) -> None:
    existing = sys.modules.get(name)
    if existing is not None:
        existing_paths = tuple(Path(value).resolve() for value in getattr(existing, "__path__", ()))
        if path.resolve() not in existing_paths:
            raise RuntimeError(
                f"Module {name!r} is already loaded from a different Playground source."
            )
        return

    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    module.__spec__ = spec
    sys.modules[name] = module
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        setattr(sys.modules[parent_name], child_name, module)


def load_playground_g1_modules(source: ResolvedPlaygroundSource):
    """Load only G1's required modules, bypassing Playground's global registry."""

    package_root = source.root / "mujoco_playground"
    namespaces = (
        ("mujoco_playground", package_root),
        ("mujoco_playground._src", package_root / "_src"),
        (
            "mujoco_playground._src.locomotion",
            package_root / "_src" / "locomotion",
        ),
        (
            "mujoco_playground._src.locomotion.g1",
            package_root / "_src" / "locomotion" / "g1",
        ),
    )
    for name, path in namespaces:
        _install_namespace(name, path)
    importlib.invalidate_caches()
    mjx_env = importlib.import_module("mujoco_playground._src.mjx_env")
    wrapper = importlib.import_module("mujoco_playground._src.wrapper")
    joystick = importlib.import_module(
        "mujoco_playground._src.locomotion.g1.joystick"
    )
    return mjx_env, wrapper, joystick

