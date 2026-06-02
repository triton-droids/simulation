"""MuJoCo eval environment for the LAFAN walk-tracking policy.

The runtime part of this module needs ``mujoco`` and ``numpy``. Importing the
module itself only uses the standard library so XML validation stays available
in lightweight shells.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET


CH_MUJOCO_JOINT_NAMES: tuple[str, ...] = (
    "left_hip1_joint",
    "left_hip2_joint",
    "left_thigh_joint",
    "left_knee_joint",
    "left_ankle_joint",
    "right_hip1_joint",
    "right_hip2_joint",
    "right_thigh_joint",
    "right_knee_joint",
    "right_ankle_joint",
)

DEFAULT_MUJOCO_REPO = Path("/workspace/mujoco/cse-145-237d-humanoid-teleop")
DEFAULT_ACTIVE_URDF = Path(
    "/workspace/simulation/source/tritonhumanoid/tritonhumanoid/assets/human_offset_corrected.urdf"
)
DEFAULT_MODEL_CACHE = (
    DEFAULT_MUJOCO_REPO / "wearable_imu" / ".cache" / "ch_robot_isaac_trained_model"
)
DEFAULT_MJCF_PATH = DEFAULT_MODEL_CACHE / "ch_robot_10dof_isaac_trained.xml"
DEFAULT_CANONICAL_CACHE = DEFAULT_MUJOCO_REPO / "wearable_imu" / ".cache" / "ch_robot_model"
CH_ROBOT_BRANCH_REF = "origin/retargeting_holosoma"
CH_ROBOT_BRANCH_PATH = (
    "holosoma-main/src/holosoma_retargeting/holosoma_retargeting/models/ch_robot"
)

ISAAC_ROOT_HEIGHT = 0.6846
ISAAC_WORLD_TO_TORSO_POS = (0.1505, 0.008, -0.6996)
CONTROL_DT = 1.0 / 30.0
PHYSICS_DT = 1.0 / 240.0
DECIMATION = 8
OBS_SINGLE_DIM = 106
OBS_STACK_FRAMES = 3
OBS_DIM = OBS_SINGLE_DIM * OBS_STACK_FRAMES
ACTION_DIM = len(CH_MUJOCO_JOINT_NAMES)
CLIP_OBSERVATIONS = 5.0
CLIP_ACTIONS = 1.0
ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.1
RESIDUAL_ACTION_SCALE = 0.15
FUTURE_REF_OFFSETS = (1, 2, 4, 6)
MOTION_REFERENCE_POS_ERROR_SCALE = 1.0
MOTION_REFERENCE_VEL_SCALE = 0.1
SOFT_JOINT_LIMIT_FACTOR = 0.95
TERMINATION_HEIGHT = 0.4

RESIDUAL_ACTION_SCALE_BY_JOINT: dict[str, float] = {
    "left_hip1_joint": 1.0,
    "left_hip2_joint": 0.8,
    "left_thigh_joint": 1.0,
    "left_knee_joint": 1.0,
    "left_ankle_joint": 0.7,
    "right_hip1_joint": 1.0,
    "right_hip2_joint": 0.8,
    "right_thigh_joint": 1.0,
    "right_knee_joint": 1.0,
    "right_ankle_joint": 0.7,
}


@dataclass(frozen=True)
class JointSpec:
    name: str
    axis: tuple[float, float, float]
    lower: float
    upper: float
    origin: tuple[float, float, float]


@dataclass(frozen=True)
class PDGroupSpec:
    joint_names: tuple[str, ...]
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    effort_limit: float
    min_delay: int
    max_delay: int


TRAINED_PD_GROUPS: tuple[PDGroupSpec, ...] = (
    PDGroupSpec(("left_hip1_joint", "right_hip1_joint"), (250.0, 250.0), (5.0, 5.0), 120.0, 1, 1),
    PDGroupSpec(("left_hip2_joint", "right_hip2_joint"), (250.0, 250.0), (5.0, 5.0), 120.0, 0, 1),
    PDGroupSpec(("left_thigh_joint", "right_thigh_joint"), (100.0, 100.0), (2.0, 2.0), 120.0, 0, 1),
    PDGroupSpec(("left_knee_joint", "right_knee_joint"), (150.0, 150.0), (5.0, 5.0), 120.0, 1, 2),
    PDGroupSpec(("left_ankle_joint", "right_ankle_joint"), (120.0, 120.0), (0.8, 1.0), 120.0, 0, 1),
)


class ContractValidationError(RuntimeError):
    """Raised when the generated MuJoCo model drifts from the Isaac-trained contract."""


def _require_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name or name.startswith(f"{exc.name}."):
            raise ModuleNotFoundError(
                f"{exc.name!r} is required for MuJoCo LAFAN evaluation. "
                "Run this under the project/Isaac Python environment with sim dependencies installed."
            ) from exc
        raise


def _fmt_float(value: float) -> str:
    return f"{float(value):.12g}"


def _fmt_vec(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(_fmt_float(v) for v in values)


def _parse_vec(raw: str | None, expected: int) -> tuple[float, ...]:
    if raw is None:
        raise ContractValidationError("missing vector attribute")
    values = tuple(float(x) for x in raw.split())
    if len(values) != expected:
        raise ContractValidationError(f"expected {expected} values, got {raw!r}")
    return values


def _allclose(a: tuple[float, ...], b: tuple[float, ...], atol: float = 1e-6) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= atol for x, y in zip(a, b))


def parse_urdf_joint_specs(urdf_path: Path = DEFAULT_ACTIVE_URDF) -> list[JointSpec]:
    """Return the active Isaac URDF's 10 actuated joint specs in file order."""

    root = ET.parse(urdf_path).getroot()
    specs: list[JointSpec] = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        name = joint.attrib["name"]
        if name not in CH_MUJOCO_JOINT_NAMES:
            continue
        axis = _parse_vec(joint.find("axis").attrib.get("xyz"), 3)
        limit = joint.find("limit")
        origin = joint.find("origin")
        if limit is None or origin is None:
            raise ContractValidationError(f"URDF joint {name!r} is missing limit or origin")
        specs.append(
            JointSpec(
                name=name,
                axis=(float(axis[0]), float(axis[1]), float(axis[2])),
                lower=float(limit.attrib["lower"]),
                upper=float(limit.attrib["upper"]),
                origin=tuple(float(x) for x in origin.attrib.get("xyz", "0 0 0").split()),
            )
        )
    names = [spec.name for spec in specs]
    if names != list(CH_MUJOCO_JOINT_NAMES):
        raise ContractValidationError(
            f"URDF joint order mismatch: got {names}, expected {list(CH_MUJOCO_JOINT_NAMES)}"
        )
    return specs


def parse_urdf_world_to_torso(urdf_path: Path = DEFAULT_ACTIVE_URDF) -> tuple[float, float, float]:
    root = ET.parse(urdf_path).getroot()
    joint = root.find("./joint[@name='world_to_base']")
    if joint is None:
        return ISAAC_WORLD_TO_TORSO_POS
    origin = joint.find("origin")
    if origin is None:
        raise ContractValidationError("URDF world_to_base joint is missing origin")
    return tuple(float(x) for x in origin.attrib.get("xyz", "0 0 0").split())


def soft_limits_from_specs(specs: list[JointSpec], factor: float = SOFT_JOINT_LIMIT_FACTOR):
    np = _require_module("numpy")
    lower = np.asarray([spec.lower for spec in specs], dtype=np.float64)
    upper = np.asarray([spec.upper for spec in specs], dtype=np.float64)
    center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * float(factor)
    return center - half, center + half


def _find_direct_body(parent: ET.Element, name: str) -> ET.Element | None:
    for child in parent.findall("body"):
        if child.attrib.get("name") == name:
            return child
    return None


def _remove_children_by_tag_and_name(parent: ET.Element, tag: str, name: str | None = None) -> None:
    for child in list(parent):
        if child.tag != tag:
            continue
        if name is not None and child.attrib.get("name") != name:
            continue
        parent.remove(child)


def _ensure_floor_assets(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    if asset.find("./texture[@name='floor_grid_texture']") is None:
        ET.SubElement(
            asset,
            "texture",
            {
                "name": "floor_grid_texture",
                "type": "2d",
                "builtin": "checker",
                "rgb1": "0.18 0.20 0.22",
                "rgb2": "0.32 0.34 0.36",
                "width": "512",
                "height": "512",
            },
        )
    material = asset.find("./material[@name='floor_grid']")
    if material is None:
        material = ET.SubElement(asset, "material", {"name": "floor_grid"})
    material.attrib.update({"texture": "floor_grid_texture", "texrepeat": "8 8", "reflectance": "0.12"})

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    ground = worldbody.find("./geom[@name='ground']")
    if ground is None:
        ground = ET.SubElement(worldbody, "geom", {"name": "ground"})
    ground.attrib.update({"type": "plane", "size": "10 10 0.1", "pos": "0 0 0", "material": "floor_grid"})


def _patch_root_to_isaac_world(root: ET.Element, world_to_torso: tuple[float, float, float]) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ContractValidationError("MJCF is missing worldbody")

    world_body = _find_direct_body(worldbody, "world")
    if world_body is not None:
        _remove_children_by_tag_and_name(world_body, "freejoint")
        world_body.insert(0, ET.Element("freejoint"))
        if world_body.find("./site[@name='imu']") is None:
            world_body.insert(1, ET.Element("site", {"name": "imu"}))
        torso = _find_direct_body(world_body, "torso")
        if torso is None:
            raise ContractValidationError("Generated MJCF has world body but no torso child")
        torso.attrib["pos"] = _fmt_vec(world_to_torso)
        _remove_children_by_tag_and_name(torso, "freejoint")
        _remove_children_by_tag_and_name(torso, "site", "imu")
        return

    torso = _find_direct_body(worldbody, "torso")
    if torso is None:
        raise ContractValidationError("Canonical MJCF is missing root torso body")
    torso_index = list(worldbody).index(torso)
    worldbody.remove(torso)

    _remove_children_by_tag_and_name(torso, "freejoint")
    _remove_children_by_tag_and_name(torso, "site", "imu")
    torso.attrib["pos"] = _fmt_vec(world_to_torso)

    world_body = ET.Element("body", {"name": "world", "pos": f"0 0 {_fmt_float(ISAAC_ROOT_HEIGHT)}"})
    world_body.append(ET.Element("freejoint"))
    world_body.append(ET.Element("site", {"name": "imu"}))
    world_body.append(torso)
    worldbody.insert(torso_index, world_body)


def patch_mjcf_text_to_isaac_urdf(
    xml_text: str,
    urdf_path: Path = DEFAULT_ACTIVE_URDF,
) -> str:
    """Return MJCF text patched to the active Isaac-trained URDF contract."""

    specs = parse_urdf_joint_specs(urdf_path)
    world_to_torso = parse_urdf_world_to_torso(urdf_path)
    root = ET.fromstring(xml_text)

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.attrib["angle"] = "radian"
    compiler.attrib["coordinate"] = "local"
    compiler.attrib["meshdir"] = "meshes"

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib["timestep"] = _fmt_float(PHYSICS_DT)
    option.attrib["gravity"] = "0 0 -9.81"
    option.attrib.setdefault("integrator", "implicitfast")

    _patch_root_to_isaac_world(root, world_to_torso)
    _ensure_floor_assets(root)

    joint_by_name = {joint.attrib.get("name"): joint for joint in root.findall(".//joint") if "name" in joint.attrib}
    for spec in specs:
        joint = joint_by_name.get(spec.name)
        if joint is None:
            raise ContractValidationError(f"MJCF is missing joint {spec.name!r}")
        joint.attrib["type"] = "hinge"
        joint.attrib["axis"] = _fmt_vec(spec.axis)
        joint.attrib["range"] = f"{_fmt_float(spec.lower)} {_fmt_float(spec.upper)}"
        joint.attrib["limited"] = "true"

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    actuator.clear()
    for joint_name in CH_MUJOCO_JOINT_NAMES:
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": joint_name.replace("_joint", "_motor"),
                "joint": joint_name,
                "gear": "1",
                "ctrllimited": "true",
                "ctrlrange": "-120 120",
            },
        )

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def _copy_canonical_model_into(model_dir: Path, mujoco_repo: Path = DEFAULT_MUJOCO_REPO) -> Path:
    if DEFAULT_CANONICAL_CACHE.exists() and (DEFAULT_CANONICAL_CACHE / "ch_robot_10dof.xml").exists():
        shutil.copytree(DEFAULT_CANONICAL_CACHE, model_dir)
        return model_dir / "ch_robot_10dof.xml"

    if not mujoco_repo.exists():
        raise FileNotFoundError(f"MuJoCo repo does not exist: {mujoco_repo}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / "ch_robot.tar"
        with archive_path.open("wb") as archive_file:
            subprocess.run(
                ["git", "archive", CH_ROBOT_BRANCH_REF, CH_ROBOT_BRANCH_PATH],
                cwd=mujoco_repo,
                stdout=archive_file,
                check=True,
            )
        with tarfile.open(archive_path) as archive:
            archive.extractall(tmp_path)
        extracted = tmp_path / CH_ROBOT_BRANCH_PATH
        if not extracted.exists():
            raise RuntimeError(f"failed to extract {CH_ROBOT_BRANCH_PATH} from {CH_ROBOT_BRANCH_REF}")
        shutil.copytree(extracted, model_dir)

    return model_dir / "ch_robot_10dof.xml"


def ensure_isaac_trained_mjcf(
    *,
    model_dir: Path = DEFAULT_MODEL_CACHE,
    urdf_path: Path = DEFAULT_ACTIVE_URDF,
    mujoco_repo: Path = DEFAULT_MUJOCO_REPO,
    refresh: bool = False,
) -> Path:
    """Generate/cache the Isaac-trained MuJoCo model and return its XML path."""

    xml_path = model_dir / "ch_robot_10dof_isaac_trained.xml"
    if not refresh and xml_path.exists() and (model_dir / "meshes").exists():
        validate_mjcf_against_urdf(xml_path, urdf_path)
        return xml_path

    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    canonical_xml = _copy_canonical_model_into(model_dir, mujoco_repo=mujoco_repo)
    patched = patch_mjcf_text_to_isaac_urdf(canonical_xml.read_text(), urdf_path=urdf_path)
    xml_path.write_text(patched)
    validate_mjcf_against_urdf(xml_path, urdf_path)
    return xml_path


def _mjcf_joint_specs(root: ET.Element) -> list[JointSpec]:
    specs: list[JointSpec] = []
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name")
        if name not in CH_MUJOCO_JOINT_NAMES:
            continue
        lower, upper = _parse_vec(joint.attrib.get("range"), 2)
        specs.append(
            JointSpec(
                name=name,
                axis=tuple(float(x) for x in joint.attrib.get("axis", "0 0 0").split()),
                lower=float(lower),
                upper=float(upper),
                origin=tuple(float(x) for x in joint.attrib.get("pos", "0 0 0").split()),
            )
        )
    return specs


def validate_mjcf_text_against_urdf(xml_text: str, urdf_path: Path = DEFAULT_ACTIVE_URDF) -> None:
    """Validate MJCF text against the active Isaac URDF without loading MuJoCo."""

    root = ET.fromstring(xml_text)
    urdf_specs = parse_urdf_joint_specs(urdf_path)
    mjcf_specs = _mjcf_joint_specs(root)
    if [spec.name for spec in mjcf_specs] != [spec.name for spec in urdf_specs]:
        raise ContractValidationError(
            f"MJCF joint order mismatch: got {[s.name for s in mjcf_specs]}, "
            f"expected {[s.name for s in urdf_specs]}"
        )
    for actual, expected in zip(mjcf_specs, urdf_specs):
        if not _allclose(actual.axis, expected.axis):
            raise ContractValidationError(
                f"joint {actual.name} axis mismatch: MJCF {actual.axis}, URDF {expected.axis}"
            )
        if abs(actual.lower - expected.lower) > 1e-6 or abs(actual.upper - expected.upper) > 1e-6:
            raise ContractValidationError(
                f"joint {actual.name} range mismatch: MJCF {(actual.lower, actual.upper)}, "
                f"URDF {(expected.lower, expected.upper)}"
            )

    actuator = root.find("actuator")
    if actuator is None:
        raise ContractValidationError("MJCF is missing actuator block")
    actuator_joints = [child.attrib.get("joint") for child in list(actuator)]
    if actuator_joints != list(CH_MUJOCO_JOINT_NAMES):
        raise ContractValidationError(
            f"actuator joint order mismatch: got {actuator_joints}, expected {list(CH_MUJOCO_JOINT_NAMES)}"
        )
    if any(child.tag != "motor" for child in list(actuator)):
        raise ContractValidationError("generated MJCF must use torque motor actuators")

    worldbody = root.find("worldbody")
    world = _find_direct_body(worldbody, "world") if worldbody is not None else None
    if world is None or world.find("freejoint") is None:
        raise ContractValidationError("generated MJCF must have a root body named 'world' with a freejoint")
    torso = _find_direct_body(world, "torso")
    if torso is None:
        raise ContractValidationError("generated MJCF root body 'world' must contain child body 'torso'")
    expected_torso_pos = parse_urdf_world_to_torso(urdf_path)
    torso_pos = _parse_vec(torso.attrib.get("pos"), 3)
    if not _allclose(torso_pos, expected_torso_pos):
        raise ContractValidationError(
            f"world->torso offset mismatch: MJCF {torso_pos}, URDF {expected_torso_pos}"
        )


def validate_mjcf_against_urdf(
    xml_path: Path,
    urdf_path: Path = DEFAULT_ACTIVE_URDF,
    *,
    require_mujoco: bool = False,
) -> None:
    validate_mjcf_text_against_urdf(xml_path.read_text(), urdf_path=urdf_path)
    if not require_mujoco:
        return
    mujoco = _require_module("mujoco")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.nq != 17 or model.nv != 16 or model.nu != ACTION_DIM:
        raise ContractValidationError(f"MuJoCo dimensions mismatch: nq={model.nq} nv={model.nv} nu={model.nu}")


class ObservationStack:
    """Isaac-style observation stack with shape ``[obs_dim, stack_frames]``."""

    def __init__(self, obs_dim: int = OBS_SINGLE_DIM, stack_frames: int = OBS_STACK_FRAMES):
        self.np = _require_module("numpy")
        self.obs_dim = int(obs_dim)
        self.stack_frames = int(stack_frames)
        self.buffer = self.np.zeros((self.obs_dim, self.stack_frames), dtype=self.np.float32)

    def reset(self, obs: object) -> object:
        obs = self.np.asarray(obs, dtype=self.np.float32)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"single observation shape mismatch: got {obs.shape}, expected {(self.obs_dim,)}")
        self.buffer[:] = obs[:, None]
        return self.flatten()

    def append(self, obs: object) -> object:
        obs = self.np.asarray(obs, dtype=self.np.float32)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"single observation shape mismatch: got {obs.shape}, expected {(self.obs_dim,)}")
        self.buffer = self.np.roll(self.buffer, shift=-1, axis=1)
        self.buffer[:, -1] = obs
        return self.flatten()

    def flatten(self) -> object:
        return self.buffer.reshape(-1).astype(self.np.float32, copy=True)


class NumpyLafanMotionLibrary:
    REQUIRED_KEYS = {
        "qpos",
        "qvel",
        "joint_vel",
        "root_lin_vel_w",
        "root_lin_vel_b",
        "root_ang_vel_w",
        "root_ang_vel_b",
        "root_yaw",
        "yaw_rate_ref",
        "fps",
    }

    def __init__(
        self,
        motion_reference_dir: str | Path,
        *,
        motion_manifest_file: str | Path = "",
        motion_fps: int = 30,
        motion_min_length_s: float = 1.0,
        rng: object | None = None,
    ) -> None:
        self.np = _require_module("numpy")
        self.motion_fps = int(motion_fps)
        self.rng = rng if rng is not None else self.np.random.default_rng()
        paths = self._resolve_motion_paths(Path(motion_reference_dir), Path(motion_manifest_file) if motion_manifest_file else None)
        min_len = max(1, int(self.np.ceil(float(motion_min_length_s) * self.motion_fps)))
        motions = [self._load_motion(path) for path in paths]
        motions = [motion for motion in motions if motion["length"] >= min_len]
        if not motions:
            raise RuntimeError(f"No valid LAFAN motions loaded from {motion_reference_dir!r}")
        self.motions = motions
        self.lengths = self.np.asarray([motion["length"] for motion in motions], dtype=self.np.int64)
        self.num_motions = len(motions)

    def _resolve_motion_paths(self, motion_dir: Path, manifest_path: Path | None) -> list[Path]:
        if manifest_path is not None and str(manifest_path):
            if not manifest_path.exists():
                raise FileNotFoundError(f"LAFAN motion manifest does not exist: {manifest_path}")
            paths: list[Path] = []
            for raw_line in manifest_path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                path = Path(line).expanduser()
                if not path.is_absolute():
                    path = manifest_path.parent / path
                paths.append(path)
            return sorted(paths)
        if not motion_dir.exists():
            raise FileNotFoundError(f"LAFAN motion directory does not exist: {motion_dir}")
        return sorted(motion_dir.rglob("*.npz"))

    def _array(
        self,
        data: object,
        key: str,
        shape_tail: tuple[int, ...] | None = None,
        expected_shape: tuple[int, ...] | None = None,
    ):
        array = self.np.asarray(data[key], dtype=self.np.float32)
        if expected_shape is not None and array.shape != expected_shape:
            raise RuntimeError(f"LAFAN field {key!r} expected shape {expected_shape}, got {array.shape}")
        if shape_tail is not None and array.shape[1:] != shape_tail:
            raise RuntimeError(f"LAFAN field {key!r} expected trailing shape {shape_tail}, got {array.shape}")
        if not self.np.isfinite(array).all():
            raise RuntimeError(f"LAFAN field {key!r} contains non-finite values")
        return array

    def _load_motion(self, path: Path) -> dict[str, object]:
        if not path.exists():
            raise FileNotFoundError(f"LAFAN motion file listed but not found: {path}")
        with self.np.load(str(path), allow_pickle=True) as data:
            missing = sorted(self.REQUIRED_KEYS.difference(data.files))
            if missing:
                raise RuntimeError(f"LAFAN motion {path} is missing required keys: {missing}")
            fps = float(self.np.asarray(data["fps"]).reshape(-1)[0])
            if abs(fps - float(self.motion_fps)) > 1e-6:
                raise RuntimeError(f"LAFAN motion {path} has fps={fps:g}; expected {self.motion_fps}.")
            qpos = self._array(data, "qpos")
            qvel = self._array(data, "qvel")
            joint_vel = self._array(data, "joint_vel")
            length = int(qpos.shape[0])
            if qpos.shape != (length, 17):
                raise RuntimeError(f"LAFAN motion {path} expected qpos shape (T, 17), got {qpos.shape}")
            if qvel.shape != (length, 16):
                raise RuntimeError(f"LAFAN motion {path} expected qvel shape (T, 16), got {qvel.shape}")
            if joint_vel.shape != (length, ACTION_DIM):
                raise RuntimeError(f"LAFAN motion {path} expected joint_vel shape (T, 10), got {joint_vel.shape}")
            return {
                "path": path,
                "length": length,
                "root_pos": qpos[:, 0:3],
                "root_quat": qpos[:, 3:7],
                "joint_pos": qpos[:, 7:17],
                "joint_vel": joint_vel,
                "root_lin_vel_w": self._array(data, "root_lin_vel_w", expected_shape=(length, 3)),
                "root_lin_vel_b": self._array(data, "root_lin_vel_b", expected_shape=(length, 3)),
                "root_ang_vel_w": self._array(data, "root_ang_vel_w", expected_shape=(length, 3)),
                "root_ang_vel_b": self._array(data, "root_ang_vel_b", expected_shape=(length, 3)),
                "root_yaw": self._array(data, "root_yaw", expected_shape=(length,)),
                "yaw_rate_ref": self._array(data, "yaw_rate_ref", expected_shape=(length,)),
            }

    def sample(self, random_start: bool = True) -> tuple[int, int, int]:
        motion_id = int(self.rng.integers(0, self.num_motions))
        length = int(self.lengths[motion_id])
        if random_start:
            high = max(length - 1, 1)
            start = int(self.np.floor(float(self.rng.random()) * float(high)))
        else:
            start = 0
        return motion_id, start, length

    def frame(self, motion_id: int, frame: int) -> dict[str, object]:
        motion = self.motions[int(motion_id)]
        idx = min(max(int(frame), 0), int(motion["length"]) - 1)
        return {
            "root_pos": motion["root_pos"][idx],
            "root_quat": motion["root_quat"][idx],
            "joint_pos": motion["joint_pos"][idx],
            "joint_vel": motion["joint_vel"][idx],
            "root_lin_vel_w": motion["root_lin_vel_w"][idx],
            "root_lin_vel_b": motion["root_lin_vel_b"][idx],
            "root_ang_vel_w": motion["root_ang_vel_w"][idx],
            "root_ang_vel_b": motion["root_ang_vel_b"][idx],
            "root_yaw": float(motion["root_yaw"][idx]),
            "yaw_rate_ref": float(motion["yaw_rate_ref"][idx]),
        }

    def future_joint_pos(self, motion_id: int, frame: int, future_offsets: tuple[int, ...] = FUTURE_REF_OFFSETS):
        motion = self.motions[int(motion_id)]
        max_frame = int(motion["length"]) - 1
        frames = [min(int(frame) + int(offset), max_frame) for offset in future_offsets]
        return self.np.asarray([motion["joint_pos"][idx] for idx in frames], dtype=self.np.float32)


class DelayedPDController:
    def __init__(self, joint_names: tuple[str, ...], rng: object):
        self.np = _require_module("numpy")
        self.joint_names = tuple(joint_names)
        self.rng = rng
        self.kp = self.np.zeros(len(joint_names), dtype=self.np.float64)
        self.kd = self.np.zeros(len(joint_names), dtype=self.np.float64)
        self.effort = self.np.zeros(len(joint_names), dtype=self.np.float64)
        self._group_bounds: list[tuple[list[int], int, int]] = []
        index = {name: i for i, name in enumerate(joint_names)}
        for group in TRAINED_PD_GROUPS:
            ids = [index[name] for name in group.joint_names]
            for local, joint_id in enumerate(ids):
                self.kp[joint_id] = group.stiffness[local]
                self.kd[joint_id] = group.damping[local]
                self.effort[joint_id] = group.effort_limit
            self._group_bounds.append((ids, group.min_delay, group.max_delay))
        self.delays = self.np.zeros(len(joint_names), dtype=self.np.int64)
        self.history: deque[object] = deque(maxlen=max(group.max_delay for group in TRAINED_PD_GROUPS) + 1)

    def reset(self) -> None:
        self.history.clear()
        for ids, min_delay, max_delay in self._group_bounds:
            delay = int(self.rng.integers(min_delay, max_delay + 1))
            for joint_id in ids:
                self.delays[joint_id] = delay

    def delayed_position_target(self, q_des: object) -> object:
        q_des = self.np.asarray(q_des, dtype=self.np.float64)
        self.history.append(q_des.copy())
        delayed = q_des.copy()
        for joint_id, delay in enumerate(self.delays):
            delay = int(delay)
            if len(self.history) > delay:
                delayed[joint_id] = self.history[-(delay + 1)][joint_id]
        return delayed

    def torque(self, q_des: object, q: object, qd: object) -> object:
        q_delayed = self.delayed_position_target(q_des)
        q = self.np.asarray(q, dtype=self.np.float64)
        qd = self.np.asarray(qd, dtype=self.np.float64)
        tau = self.kp * (q_delayed - q) - self.kd * qd
        return self.np.clip(tau, -self.effort, self.effort)


def _normalize_quat(np, quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quat_conjugate(np, quat):
    quat = np.asarray(quat, dtype=np.float64)
    return np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def _quat_multiply(np, a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_rotate_inverse(np, quat, vec):
    quat = _normalize_quat(np, quat)
    vq = np.asarray([0.0, *np.asarray(vec, dtype=np.float64)], dtype=np.float64)
    return _quat_multiply(np, _quat_multiply(np, _quat_conjugate(np, quat), vq), quat)[1:4]


def yaw_from_quat(np, quat) -> float:
    w, x, y, z = _normalize_quat(np, quat)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def wrap_angle(np, angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class MujocoLafanWalkTrackingEnv:
    """Single-environment MuJoCo evaluator matching the Isaac LAFAN policy contract."""

    def __init__(
        self,
        *,
        motion_dir: str | Path,
        motion_manifest: str | Path = "",
        model_xml: str | Path | None = None,
        policy_dt: float = CONTROL_DT,
        seed: int | None = None,
        render: bool = False,
        refresh_model: bool = False,
    ) -> None:
        self.np = _require_module("numpy")
        self.mujoco = _require_module("mujoco")
        self.rng = self.np.random.default_rng(seed)
        self.policy_dt = float(policy_dt)
        self.decimation = int(round(self.policy_dt / PHYSICS_DT))
        if self.decimation != DECIMATION:
            raise ValueError(f"expected decimation={DECIMATION}, got {self.decimation}")

        if model_xml is None:
            self.xml_path = ensure_isaac_trained_mjcf(refresh=refresh_model)
        else:
            self.xml_path = Path(model_xml)
            validate_mjcf_against_urdf(self.xml_path)
        validate_mjcf_against_urdf(self.xml_path, require_mujoco=True)
        self.model = self.mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = self.mujoco.MjData(self.model)

        self.joint_names = CH_MUJOCO_JOINT_NAMES
        self.joint_ids = [
            self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]
        if any(joint_id < 0 for joint_id in self.joint_ids):
            raise ContractValidationError(f"missing MuJoCo joints for {self.joint_names}")
        self.qpos_addr = self.np.asarray([self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids], dtype=self.np.int64)
        self.qvel_addr = self.np.asarray([self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids], dtype=self.np.int64)

        specs = parse_urdf_joint_specs()
        self.hard_lower = self.np.asarray([spec.lower for spec in specs], dtype=self.np.float64)
        self.hard_upper = self.np.asarray([spec.upper for spec in specs], dtype=self.np.float64)
        center = 0.5 * (self.hard_lower + self.hard_upper)
        half = 0.5 * (self.hard_upper - self.hard_lower) * SOFT_JOINT_LIMIT_FACTOR
        self.soft_lower = center - half
        self.soft_upper = center + half
        self.action_scale_per_joint = self.np.asarray(
            [RESIDUAL_ACTION_SCALE_BY_JOINT[name] for name in self.joint_names], dtype=self.np.float64
        )

        self.motion_lib = NumpyLafanMotionLibrary(
            motion_dir,
            motion_manifest_file=motion_manifest,
            motion_fps=30,
            motion_min_length_s=1.0,
            rng=self.rng,
        )
        self.pd = DelayedPDController(self.joint_names, self.rng)
        self.obs_stack = ObservationStack()
        self.actions = self.np.zeros(ACTION_DIM, dtype=self.np.float64)
        self.prev_actions = self.np.zeros(ACTION_DIM, dtype=self.np.float64)
        self.motion_id = 0
        self.motion_start_frame = 0
        self.motion_end_frame = 1
        self.step_count = 0
        self.last_obs = self.np.zeros(OBS_DIM, dtype=self.np.float32)
        self._viewer = None
        self._render_enabled = bool(render)
        if self._render_enabled:
            self.render()

    def reset(self, *, motion_id: int | None = None, start_frame: int | None = None, random_start: bool = True):
        if motion_id is None or start_frame is None:
            sampled_motion_id, sampled_start, sampled_end = self.motion_lib.sample(random_start=random_start)
            self.motion_id = sampled_motion_id if motion_id is None else int(motion_id)
            self.motion_start_frame = sampled_start if start_frame is None else int(start_frame)
            self.motion_end_frame = sampled_end
        else:
            self.motion_id = int(motion_id)
            self.motion_start_frame = int(start_frame)
            self.motion_end_frame = int(self.motion_lib.lengths[self.motion_id])

        self.step_count = 0
        self.actions[:] = 0.0
        self.prev_actions[:] = 0.0
        self.pd.reset()
        ref = self.motion_lib.frame(self.motion_id, self.motion_start_frame)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[0:3] = self.np.asarray(ref["root_pos"], dtype=self.np.float64) + self.np.asarray(
            [0.0, 0.0, ISAAC_ROOT_HEIGHT], dtype=self.np.float64
        )
        self.data.qpos[3:7] = _normalize_quat(self.np, ref["root_quat"])
        self.data.qpos[self.qpos_addr] = self.np.asarray(ref["joint_pos"], dtype=self.np.float64)
        self.data.qvel[0:3] = self.np.asarray(ref["root_lin_vel_w"], dtype=self.np.float64)
        self.data.qvel[3:6] = self.np.asarray(ref["root_ang_vel_w"], dtype=self.np.float64)
        self.data.qvel[self.qvel_addr] = self.np.asarray(ref["joint_vel"], dtype=self.np.float64)
        self.data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

        obs0 = self._compute_single_observation()
        self.last_obs = self.obs_stack.reset(obs0)
        return self.last_obs

    def step(self, action):
        action = self.np.asarray(action, dtype=self.np.float64).reshape(-1)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action shape mismatch: got {action.shape}, expected {(ACTION_DIM,)}")

        old_action = self.actions.copy()
        action = self.np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS)
        self.prev_actions = old_action
        self.actions = action

        control_ref = self._target_for_frame(self.motion_start_frame + self.step_count)
        q_des = self.np.asarray(control_ref["joint_pos"], dtype=self.np.float64) + (
            RESIDUAL_ACTION_SCALE * self.action_scale_per_joint * action
        )
        q_des = self.np.clip(q_des, self.soft_lower, self.soft_upper)

        for _ in range(self.decimation):
            q = self.data.qpos[self.qpos_addr].copy()
            qd = self.data.qvel[self.qvel_addr].copy()
            self.data.ctrl[:] = self.pd.torque(q_des, q, qd)
            self.mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs0 = self._compute_single_observation()
        self.last_obs = self.obs_stack.append(obs0)
        done, done_reason = self._done()
        info = self._info(done_reason)
        return self.last_obs, 0.0, done, info

    def render(self) -> None:
        if self._viewer is None:
            viewer = _require_module("mujoco.viewer")
            self._viewer = viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    @property
    def frame(self) -> int:
        return self.motion_start_frame + self.step_count

    def _target_for_frame(self, frame: int) -> dict[str, object]:
        ref = self.motion_lib.frame(self.motion_id, frame)
        ref["root_pos_offset"] = self.np.asarray(ref["root_pos"], dtype=self.np.float64) + self.np.asarray(
            [0.0, 0.0, ISAAC_ROOT_HEIGHT], dtype=self.np.float64
        )
        ref["future_joint_pos"] = self.motion_lib.future_joint_pos(self.motion_id, frame, FUTURE_REF_OFFSETS)
        return ref

    def _compute_single_observation(self):
        target = self._target_for_frame(self.frame)
        root_quat = _normalize_quat(self.np, self.data.qpos[3:7])
        root_lin_vel_w = self.data.qvel[0:3].copy()
        root_ang_vel_w = self.data.qvel[3:6].copy()
        root_lin_vel_b = quat_rotate_inverse(self.np, root_quat, root_lin_vel_w)
        root_ang_vel_b = quat_rotate_inverse(self.np, root_quat, root_ang_vel_w)
        up_b = quat_rotate_inverse(self.np, root_quat, self.np.asarray([0.0, 0.0, 1.0], dtype=self.np.float64))

        act_pos = self.data.qpos[self.qpos_addr].copy()
        act_vel = self.data.qvel[self.qvel_addr].copy()
        act_pos_scaled = 2.0 * (act_pos - self.soft_lower) / (self.soft_upper - self.soft_lower + 1e-6) - 1.0

        target_joint_pos = self.np.asarray(target["joint_pos"], dtype=self.np.float64)
        target_joint_vel = self.np.asarray(target["joint_vel"], dtype=self.np.float64)
        future_joint_errors = self.np.asarray(target["future_joint_pos"], dtype=self.np.float64) - act_pos[None, :]
        root_height_error = float(target["root_pos_offset"][2] - self.data.qpos[2])
        root_yaw_error = wrap_angle(self.np, float(target["root_yaw"]) - yaw_from_quat(self.np, root_quat))
        yaw_rate_ref = float(target["yaw_rate_ref"])
        yaw_rate_error = float(root_ang_vel_b[2] - yaw_rate_ref)

        obs = self.np.concatenate(
            [
                up_b,
                root_lin_vel_b,
                root_ang_vel_b * ANG_VEL_SCALE,
                act_pos_scaled,
                act_vel * DOF_VEL_SCALE,
                self.prev_actions,
                (target_joint_pos - act_pos) * MOTION_REFERENCE_POS_ERROR_SCALE,
                target_joint_vel * MOTION_REFERENCE_VEL_SCALE,
                self.np.asarray([root_height_error], dtype=self.np.float64),
                self.np.asarray([root_yaw_error], dtype=self.np.float64),
                self.np.asarray(target["root_lin_vel_b"], dtype=self.np.float64),
                self.np.asarray([yaw_rate_ref], dtype=self.np.float64),
                self.np.asarray([yaw_rate_error], dtype=self.np.float64),
                future_joint_errors.reshape(-1) * MOTION_REFERENCE_POS_ERROR_SCALE,
            ]
        ).astype(self.np.float32)
        if obs.shape != (OBS_SINGLE_DIM,):
            raise RuntimeError(f"observation shape mismatch: got {obs.shape}, expected {(OBS_SINGLE_DIM,)}")
        return obs

    def _done(self) -> tuple[bool, str]:
        if self.frame >= self.motion_end_frame:
            return True, "motion_end"
        if float(self.data.qpos[2]) < TERMINATION_HEIGHT:
            return True, "height"
        return False, ""

    def _info(self, done_reason: str) -> dict[str, object]:
        target = self._target_for_frame(self.frame)
        joint_error = self.data.qpos[self.qpos_addr] - self.np.asarray(target["joint_pos"], dtype=self.np.float64)
        root_height_error = float(self.data.qpos[2] - target["root_pos_offset"][2])
        root_yaw_error = wrap_angle(self.np, yaw_from_quat(self.np, self.data.qpos[3:7]) - float(target["root_yaw"]))
        return {
            "motion_id": int(self.motion_id),
            "frame": int(self.frame),
            "step": int(self.step_count),
            "done_reason": done_reason,
            "joint_pos_rmse": float(self.np.sqrt(self.np.mean(joint_error * joint_error))),
            "root_height_error": root_height_error,
            "root_yaw_error": root_yaw_error,
            "root_z": float(self.data.qpos[2]),
        }


def sleep_for_realtime(start_time: float, dt: float) -> None:
    remaining = float(dt) - (time.time() - start_time)
    if remaining > 0.0:
        time.sleep(remaining)
