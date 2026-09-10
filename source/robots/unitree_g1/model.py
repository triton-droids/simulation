"""Resolve the repository-pinned MuJoCo Menagerie Unitree G1 model.

The downloaded model remains in an ignored sparse checkout.  An explicit
``--model``/environment override is supported for offline development, but the
ordinary viewer and training paths both verify the exact repository pin.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .metadata import G1Metadata


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MENAGERIE_COMMIT = "71f066ad0be9cd271f7ed58c030243ef157af9f4"
MENAGERIE_REPO_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
ROBOT_NAME = "unitree_g1"
UNITREE_G1_MODEL_ENV = "UNITREE_G1_MODEL_PATH"
MENAGERIE_ROOT_ENV = "MUJOCO_MENAGERIE_PATH"
DEFAULT_SCENE = "scene.xml"
MJX_SCENE = "scene_mjx.xml"


@dataclass(frozen=True)
class ResolvedG1Model:
    """Provenance for one resolved G1 scene."""

    scene_path: Path
    scene: str
    source: str
    revision: str
    pinned: bool

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["scene_path"] = str(self.scene_path)
        record["expected_revision"] = MENAGERIE_COMMIT
        record["repository"] = MENAGERIE_REPO_URL
        return record


def default_cache_root() -> Path:
    """Return the shared hidden Menagerie checkout."""

    return PROJECT_ROOT / ".cache" / "mujoco_menagerie"


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        key = str(expanded.resolve()) if expanded.exists() else str(expanded)
        if key not in seen:
            result.append(expanded)
            seen.add(key)
    return result


def _run_git(command: list[str], *, capture: bool = False) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        details = ""
        if isinstance(error, subprocess.CalledProcessError):
            details = (error.stderr or error.stdout or "").strip()
        command_text = " ".join(command)
        raise RuntimeError(f"Git command failed: {command_text}\n{details}") from error
    return result.stdout.strip() if capture else ""


def checkout_revision(root: Path) -> str | None:
    """Return a checkout's current revision, or ``None`` for a non-checkout."""

    if not (root / ".git").exists():
        return None
    try:
        return _run_git(["git", "-C", str(root), "rev-parse", "HEAD"], capture=True)
    except RuntimeError:
        return None


def ensure_pinned_checkout(cache_root: Path) -> Path:
    """Create/update the sparse cache and check out the required revision."""

    if shutil.which("git") is None:
        raise RuntimeError(
            "The G1 model is not cached and git is unavailable. Pass an explicit "
            f"model path or set {UNITREE_G1_MODEL_ENV}."
        )

    cache_root = cache_root.expanduser()
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    if cache_root.exists() and not (cache_root / ".git").exists():
        raise RuntimeError(
            f"Refusing to overwrite non-git directory used as model cache: {cache_root}"
        )
    if not cache_root.exists():
        _run_git(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                MENAGERIE_REPO_URL,
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
            MENAGERIE_COMMIT,
        ]
    )
    _run_git(["git", "-C", str(cache_root), "checkout", "--detach", MENAGERIE_COMMIT])
    _run_git(["git", "-C", str(cache_root), "sparse-checkout", "set", ROBOT_NAME])
    revision = checkout_revision(cache_root)
    if revision != MENAGERIE_COMMIT:
        raise RuntimeError(
            f"Menagerie cache revision is {revision!r}; expected {MENAGERIE_COMMIT}."
        )
    return cache_root


def _candidate_roots(menagerie_root: str | Path | None, cache_root: Path) -> list[Path]:
    roots: list[Path] = []
    if menagerie_root:
        roots.append(Path(menagerie_root))
    if os.environ.get(MENAGERIE_ROOT_ENV):
        roots.append(Path(os.environ[MENAGERIE_ROOT_ENV]))
    roots.extend(
        [
            cache_root,
            PROJECT_ROOT.parent / "mujoco_menagerie",
            PROJECT_ROOT / "mujoco_menagerie",
        ]
    )
    return _unique_paths(roots)


def resolve_unitree_g1_model(
    *,
    mjx_scene: bool = True,
    model_path: str | Path | None = None,
    menagerie_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    fetch: bool = True,
) -> ResolvedG1Model:
    """Resolve a direct override or the exact pinned Menagerie scene.

    Direct model overrides are intentionally marked unpinned. All ordinary
    Menagerie-root candidates must be Git checkouts at the expected revision.
    """

    scene = MJX_SCENE if mjx_scene else DEFAULT_SCENE
    direct = model_path or os.environ.get(UNITREE_G1_MODEL_ENV)
    if direct:
        path = Path(direct).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Unitree G1 model override does not exist: {path}")
        revision = checkout_revision(path.parent.parent) or "local-override"
        return ResolvedG1Model(path, path.name, "local-override", revision, False)

    cache = Path(cache_root).expanduser() if cache_root else default_cache_root()
    mismatches: list[str] = []
    for root in _candidate_roots(menagerie_root, cache):
        candidate = root / ROBOT_NAME / scene
        if not candidate.is_file():
            continue
        revision = checkout_revision(root)
        if revision == MENAGERIE_COMMIT:
            return ResolvedG1Model(
                candidate.resolve(), scene, "mujoco-menagerie", revision, True
            )
        mismatches.append(f"{root} (revision {revision or 'unknown'})")

    if fetch:
        root = ensure_pinned_checkout(cache)
        candidate = root / ROBOT_NAME / scene
        if not candidate.is_file():
            raise FileNotFoundError(f"Pinned checkout is missing expected scene: {candidate}")
        return ResolvedG1Model(
            candidate.resolve(), scene, "mujoco-menagerie", MENAGERIE_COMMIT, True
        )

    detail = "\n  ".join(mismatches) if mismatches else "none found"
    raise FileNotFoundError(
        f"No Unitree G1 {scene} at pinned Menagerie revision {MENAGERIE_COMMIT}.\n"
        f"Mismatched candidates:\n  {detail}\n"
        "Allow fetching, pass a pinned --menagerie-root, or use an explicit local model override."
    )


class UnitreeG1Model:
    """G1 robot adapter backed by the shared resolver and live introspection."""

    name = ROBOT_NAME

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        menagerie_root: str | Path | None = None,
        cache_root: str | Path | None = None,
        fetch: bool = True,
    ) -> None:
        self.resolution = resolve_unitree_g1_model(
            mjx_scene=True,
            model_path=model_path,
            menagerie_root=menagerie_root,
            cache_root=cache_root,
            fetch=fetch,
        )
        self.scene_path = self.resolution.scene_path
        self.xml_path = str(self.scene_path)
        self.xml = self.scene_path.read_text(encoding="utf-8")
        import mujoco

        from .metadata import introspect_g1_model

        self.mj_model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.metadata: G1Metadata = introspect_g1_model(self.mj_model)
        self.model_config = self.metadata.to_dict()
        self.source_record = self.resolution.to_dict()
