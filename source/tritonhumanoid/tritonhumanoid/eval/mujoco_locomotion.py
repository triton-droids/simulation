"""MuJoCo utilities for Isaac locomotion sim2sim evaluation.

The XML validation helpers import with only the Python standard library. Runtime
rollouts require ``mujoco`` and ``numpy`` and load those modules lazily.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import shutil
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

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
ASSET_ROOT = MODULE_ROOT / "assets"
DEFAULT_ACTIVE_URDF = ASSET_ROOT / "human_offset_corrected.urdf"
DEFAULT_SOURCE_MJCF = ASSET_ROOT / "mujoco" / "ch_robot_10dof.xml"
DEFAULT_MODEL_CACHE = REPO_ROOT / "logs" / "mujoco" / "model"
DEFAULT_PATCHED_MJCF = DEFAULT_MODEL_CACHE / "ch_robot_10dof_isaac_locomotion.xml"

CONTROL_DT = 1.0 / 50.0
PHYSICS_DT = 1.0 / 200.0
DECIMATION = 4
OBS_SINGLE_DIM = 42
OBS_STACK_FRAMES = 3
OBS_DIM = OBS_SINGLE_DIM * OBS_STACK_FRAMES
ACTION_DIM = len(CH_MUJOCO_JOINT_NAMES)
CLIP_OBSERVATIONS = 5.0
CLIP_ACTIONS = 1.0
ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.1
ACTION_SCALE = 0.8
SOFT_JOINT_LIMIT_FACTOR = 0.95
DEFAULT_JOINT_VELOCITY_LIMIT = 15.0
DEFAULT_ARMATURE = 0.01
TERMINATION_HEIGHT = 0.4
UPRIGHT_THRESHOLD = 0.5
COMMAND_YAW_OFFSET = -1.5707963267948966

ACTION_SCALE_BY_JOINT: dict[str, float] = {
    "left_hip2_joint": 0.50,
    "right_hip2_joint": 0.50,
    "left_thigh_joint": 0.30,
    "right_thigh_joint": 0.30,
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
    velocity_limit: float
    armature: float
    min_delay: int
    max_delay: int


TRAINED_PD_GROUPS: tuple[PDGroupSpec, ...] = (
    PDGroupSpec(("left_hip1_joint", "right_hip1_joint"), (250.0, 250.0), (5.0, 5.0), 120.0, 15.0, 0.01, 1, 1),
    PDGroupSpec(("left_hip2_joint", "right_hip2_joint"), (250.0, 250.0), (5.0, 5.0), 120.0, 15.0, 0.01, 0, 1),
    PDGroupSpec(("left_thigh_joint", "right_thigh_joint"), (100.0, 100.0), (2.0, 2.0), 120.0, 15.0, 0.01, 0, 1),
    PDGroupSpec(("left_knee_joint", "right_knee_joint"), (150.0, 150.0), (5.0, 5.0), 120.0, 15.0, 0.01, 1, 2),
    PDGroupSpec(("left_ankle_joint", "right_ankle_joint"), (120.0, 120.0), (0.8, 1.0), 120.0, 15.0, 0.01, 0, 1),
)


class ContractValidationError(RuntimeError):
    """Raised when the generated MuJoCo model drifts from the Isaac contract."""


class PlaceholderMJCFError(ContractValidationError):
    """Raised when the local MJCF file is still the install-script placeholder."""


def _require_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name or name.startswith(f"{exc.name}."):
            raise ModuleNotFoundError(
                f"{exc.name!r} is required for MuJoCo locomotion evaluation. "
                "Run scripts/install_mujoco.sh or install the runtime dependencies."
            ) from exc
        raise


def resolve_joint_velocity_limit(value: float | None = None) -> float:
    if value is not None:
        return float(value)
    raw = os.environ.get("MUJOCO_JOINT_VELOCITY_LIMIT")
    if raw is None or raw == "":
        return DEFAULT_JOINT_VELOCITY_LIMIT
    return float(raw)


def clip_joint_velocity_array(np, qvel: object, qvel_addr: object, velocity_limit: float) -> None:
    limit = float(velocity_limit)
    if limit <= 0.0:
        return
    qvel[qvel_addr] = np.clip(qvel[qvel_addr], -limit, limit)


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


def create_dummy_mjcf(path: Path = DEFAULT_SOURCE_MJCF) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    path.write_text(
        """<!-- PLACEHOLDER_MJCF: replace with the canonical ch_robot_10dof.xml before MuJoCo eval. -->
<mujoco model="placeholder_ch_robot">
  <worldbody>
    <body name="placeholder"/>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return path


def ensure_source_mjcf(path: Path = DEFAULT_SOURCE_MJCF) -> Path:
    if not path.exists():
        return create_dummy_mjcf(path)
    return path


def _reject_placeholder_xml(xml_text: str, path: Path | None = None) -> None:
    if "PLACEHOLDER_MJCF" in xml_text or "placeholder_ch_robot" in xml_text:
        location = f": {path}" if path is not None else ""
        raise PlaceholderMJCFError(
            "Local MuJoCo MJCF is still a placeholder"
            f"{location}. Replace it with the canonical ch_robot_10dof.xml and rerun validation."
        )


def parse_urdf_joint_specs(urdf_path: Path = DEFAULT_ACTIVE_URDF) -> list[JointSpec]:
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
        return (0.1505, 0.008, -0.6996)
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


def _find_direct_body(parent: ET.Element | None, name: str) -> ET.Element | None:
    if parent is None:
        return None
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

    world_body = ET.Element("body", {"name": "world", "pos": "0 0 0.6846"})
    world_body.append(ET.Element("freejoint"))
    world_body.append(ET.Element("site", {"name": "imu"}))
    world_body.append(torso)
    worldbody.insert(torso_index, world_body)


def patch_mjcf_text_to_isaac_urdf(xml_text: str, urdf_path: Path = DEFAULT_ACTIVE_URDF) -> str:
    _reject_placeholder_xml(xml_text)
    specs = parse_urdf_joint_specs(urdf_path)
    world_to_torso = parse_urdf_world_to_torso(urdf_path)
    root = ET.fromstring(xml_text)

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.attrib["angle"] = "radian"
    compiler.attrib["coordinate"] = "local"
    compiler.attrib["meshdir"] = str(ASSET_ROOT / "robot_meshes")

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
        joint.attrib["armature"] = _fmt_float(DEFAULT_ARMATURE)

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


def ensure_isaac_locomotion_mjcf(
    *,
    source_xml: Path = DEFAULT_SOURCE_MJCF,
    model_dir: Path = DEFAULT_MODEL_CACHE,
    urdf_path: Path = DEFAULT_ACTIVE_URDF,
    refresh: bool = False,
) -> Path:
    source_xml = ensure_source_mjcf(source_xml)
    xml_path = model_dir / "ch_robot_10dof_isaac_locomotion.xml"
    if not refresh and xml_path.exists():
        validate_mjcf_against_urdf(xml_path, urdf_path)
        return xml_path

    source_text = source_xml.read_text(encoding="utf-8")
    _reject_placeholder_xml(source_text, source_xml)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    patched = patch_mjcf_text_to_isaac_urdf(source_text, urdf_path=urdf_path)
    xml_path.write_text(patched, encoding="utf-8")
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
    _reject_placeholder_xml(xml_text)
    root = ET.fromstring(xml_text)
    option = root.find("option")
    if option is None or abs(float(option.attrib.get("timestep", "nan")) - PHYSICS_DT) > 1e-12:
        raise ContractValidationError("generated MJCF must set option timestep to 0.005")

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
    world = _find_direct_body(worldbody, "world")
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
    validate_mjcf_text_against_urdf(xml_path.read_text(encoding="utf-8"), urdf_path=urdf_path)
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
        return self.buffer.reshape(-1).astype(self.np.float32)


class DelayedPDController:
    def __init__(self, joint_names: tuple[str, ...], rng: object):
        self.np = _require_module("numpy")
        self.joint_names = tuple(joint_names)
        self.rng = rng
        self.kp = self.np.zeros(len(joint_names), dtype=self.np.float64)
        self.kd = self.np.zeros(len(joint_names), dtype=self.np.float64)
        self.effort = self.np.zeros(len(joint_names), dtype=self.np.float64)
        self.velocity_limit = self.np.zeros(len(joint_names), dtype=self.np.float64)
        self.armature = self.np.zeros(len(joint_names), dtype=self.np.float64)
        self._group_bounds: list[tuple[list[int], int, int]] = []
        index = {name: i for i, name in enumerate(joint_names)}
        for group in TRAINED_PD_GROUPS:
            ids = [index[name] for name in group.joint_names]
            for local, joint_id in enumerate(ids):
                self.kp[joint_id] = group.stiffness[local]
                self.kd[joint_id] = group.damping[local]
                self.effort[joint_id] = group.effort_limit
                self.velocity_limit[joint_id] = group.velocity_limit
                self.armature[joint_id] = group.armature
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


def _quat_conjugate(np, q):
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


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


def _quat_rotate_inverse(np, q, v):
    q = _normalize_quat(np, q)
    vq = np.asarray([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return _quat_multiply(np, _quat_multiply(np, _quat_conjugate(np, q), vq), q)[1:4]


def _rotate_xy(np, vec, yaw_offset: float = COMMAND_YAW_OFFSET):
    c = float(np.cos(yaw_offset))
    s = float(np.sin(yaw_offset))
    out = np.asarray(vec, dtype=np.float64).copy()
    x = out[0]
    y = out[1]
    out[0] = c * x - s * y
    out[1] = s * x + c * y
    return out


def _as_trace_row(data, key: str, index: int):
    if key not in data:
        raise KeyError(f"Trace is missing required key {key!r}")
    return data[key][index]


class MujocoLocomotionEnv:
    """Single-environment MuJoCo evaluator matching the Isaac locomotion policy contract."""

    def __init__(
        self,
        *,
        model_xml: str | Path | None = None,
        policy_dt: float = CONTROL_DT,
        seed: int | None = None,
        render: bool = False,
        refresh_model: bool = False,
        joint_velocity_limit: float | None = None,
    ) -> None:
        self.np = _require_module("numpy")
        self.mujoco = _require_module("mujoco")
        self.rng = self.np.random.default_rng(seed)
        self.policy_dt = float(policy_dt)
        self.joint_velocity_limit = resolve_joint_velocity_limit(joint_velocity_limit)
        self.decimation = int(round(self.policy_dt / PHYSICS_DT))
        if self.decimation != DECIMATION:
            raise ValueError(f"expected decimation={DECIMATION}, got {self.decimation}")

        if model_xml is None:
            self.xml_path = ensure_isaac_locomotion_mjcf(refresh=refresh_model)
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
        self.soft_lower, self.soft_upper = soft_limits_from_specs(specs)
        self.default_joint_pos = self.np.asarray([0.4, 0.0, 0.0, -0.8, 0.4, 0.4, 0.0, 0.0, -0.8, 0.4], dtype=self.np.float64)
        self.action_scale_per_joint = self.np.asarray(
            [ACTION_SCALE_BY_JOINT.get(name, 1.0) for name in self.joint_names], dtype=self.np.float64
        )

        self.pd = DelayedPDController(self.joint_names, self.rng)
        self.obs_stack = ObservationStack()
        self.actions = self.np.zeros(ACTION_DIM, dtype=self.np.float64)
        self.prev_actions = self.np.zeros(ACTION_DIM, dtype=self.np.float64)
        self.commands = self.np.zeros(3, dtype=self.np.float64)
        self.step_count = 0
        self.last_obs = self.np.zeros(OBS_DIM, dtype=self.np.float32)
        self.last_q_des = self.default_joint_pos.copy()
        self._viewer = None
        self._render_enabled = bool(render)
        if self._render_enabled:
            self.render()

    def reset(self, *, qpos: object | None = None, qvel: object | None = None, command: object | None = None):
        self.step_count = 0
        self.actions[:] = 0.0
        self.prev_actions[:] = 0.0
        self.commands[:] = 0.0 if command is None else self.np.asarray(command, dtype=self.np.float64)
        self.pd.reset()
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[0:3] = self.np.asarray([0.0, 0.0, 0.6846], dtype=self.np.float64)
        self.data.qpos[3:7] = self.np.asarray([1.0, 0.0, 0.0, 0.0], dtype=self.np.float64)
        self.data.qpos[self.qpos_addr] = self.default_joint_pos
        if qpos is not None:
            qpos = self.np.asarray(qpos, dtype=self.np.float64)
            self.data.qpos[: min(qpos.shape[0], self.model.nq)] = qpos[: self.model.nq]
        if qvel is not None:
            qvel = self.np.asarray(qvel, dtype=self.np.float64)
            self.data.qvel[: min(qvel.shape[0], self.model.nv)] = qvel[: self.model.nv]
        self.data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        obs0 = self._compute_single_observation()
        self.last_obs = self.obs_stack.reset(obs0)
        return self.last_obs

    def set_state_from_trace(self, trace, index: int):
        self.data.qpos[:] = self.np.asarray(_as_trace_row(trace, "qpos", index), dtype=self.np.float64)
        self.data.qvel[:] = self.np.asarray(_as_trace_row(trace, "qvel", index), dtype=self.np.float64)
        if "commands" in trace:
            self.commands[:] = self.np.asarray(trace["commands"][index], dtype=self.np.float64)
        self.mujoco.mj_forward(self.model, self.data)

    def step(self, action, command: object | None = None):
        action = self.np.asarray(action, dtype=self.np.float64).reshape(-1)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action shape mismatch: got {action.shape}, expected {(ACTION_DIM,)}")
        if command is not None:
            self.commands[:] = self.np.asarray(command, dtype=self.np.float64)

        old_action = self.actions.copy()
        self.actions = self.np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS)
        self.prev_actions = old_action

        q_des = self.default_joint_pos + ACTION_SCALE * self.action_scale_per_joint * self.actions
        q_des = self.np.clip(q_des, self.hard_lower, self.hard_upper)
        self.last_q_des = q_des.copy()
        for _ in range(self.decimation):
            q = self.data.qpos[self.qpos_addr].copy()
            qd = self.data.qvel[self.qvel_addr].copy()
            self.data.ctrl[:] = self.pd.torque(q_des, q, qd)
            self.mujoco.mj_step(self.model, self.data)
            clip_joint_velocity_array(self.np, self.data.qvel, self.qvel_addr, self.joint_velocity_limit)

        self.step_count += 1
        obs0 = self._compute_single_observation()
        self.last_obs = self.obs_stack.append(obs0)
        done, done_reason = self._done()
        info = self._info(done_reason)
        return self.last_obs, 0.0, done, info

    def _compute_single_observation(self):
        root_quat = _normalize_quat(self.np, self.data.qpos[3:7])
        root_lin_vel_w = self.data.qvel[0:3].copy()
        root_ang_vel_w = self.data.qvel[3:6].copy()
        lin_vel_b = _quat_rotate_inverse(self.np, root_quat, root_lin_vel_w)
        ang_vel_b = _quat_rotate_inverse(self.np, root_quat, root_ang_vel_w)
        lin_vel_cmd = _rotate_xy(self.np, lin_vel_b)
        ang_vel_cmd = _rotate_xy(self.np, ang_vel_b)
        up_b = _quat_rotate_inverse(self.np, root_quat, self.np.asarray([0.0, 0.0, 1.0], dtype=self.np.float64))
        up_cmd = _rotate_xy(self.np, up_b)
        q = self.data.qpos[self.qpos_addr].copy()
        qd = self.data.qvel[self.qvel_addr].copy()
        act_pos_scaled = 2.0 * (q - self.soft_lower) / (self.soft_upper - self.soft_lower + 1e-6) - 1.0
        obs = self.np.concatenate(
            [
                lin_vel_cmd,
                ang_vel_cmd * ANG_VEL_SCALE,
                up_cmd,
                self.commands,
                act_pos_scaled,
                qd * DOF_VEL_SCALE,
                self.prev_actions,
            ]
        ).astype(self.np.float32)
        if obs.shape != (OBS_SINGLE_DIM,):
            raise RuntimeError(f"observation shape mismatch: got {obs.shape}, expected {(OBS_SINGLE_DIM,)}")
        return self.np.clip(obs, -CLIP_OBSERVATIONS, CLIP_OBSERVATIONS)

    def _done(self) -> tuple[bool, str]:
        root_z = float(self.data.qpos[2])
        up_z = float(_quat_rotate_inverse(self.np, self.data.qpos[3:7], self.np.asarray([0.0, 0.0, 1.0]))[2])
        if root_z < TERMINATION_HEIGHT:
            return True, "height"
        if up_z < UPRIGHT_THRESHOLD:
            return True, "tilt"
        return False, ""

    def _info(self, done_reason: str) -> dict[str, object]:
        q = self.data.qpos[self.qpos_addr].copy()
        qd = self.data.qvel[self.qvel_addr].copy()
        return {
            "step": int(self.step_count),
            "done_reason": done_reason,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "joint_pos": q,
            "joint_vel": qd,
            "q_des": self.last_q_des.copy(),
            "actions": self.actions.copy(),
            "commands": self.commands.copy(),
            "root_pos_w": self.data.qpos[0:3].copy(),
            "root_quat_w": self.data.qpos[3:7].copy(),
            "root_lin_vel_w": self.data.qvel[0:3].copy(),
            "root_ang_vel_w": self.data.qvel[3:6].copy(),
            "obs_latest": self.last_obs.copy(),
            "torque": self.data.ctrl.copy(),
        }

    def render(self) -> None:
        viewer_mod = _require_module("mujoco.viewer")
        if self._viewer is None:
            self._viewer = viewer_mod.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


def command_profile(step: int, dt: float, profile: str = "stand_forward_yaw") -> tuple[float, float, float]:
    if profile == "none":
        return (0.0, 0.0, 0.0)
    if profile != "stand_forward_yaw":
        raise ValueError(f"unsupported command profile: {profile}")
    t = float(step) * float(dt)
    cycle = 10.0
    t_mod = t % cycle
    if t_mod < 2.0:
        return (0.0, 0.0, 0.0)
    if t_mod < 6.0:
        return (0.6, 0.0, 0.0)
    return (0.0, 0.0, 0.6)


def validate_policy_metadata(policy_path: Path) -> dict:
    metadata_path = policy_path.with_name("ppo_metadata.pt")
    if not metadata_path.exists():
        return {"metadata_found": False}
    torch = _require_module("torch")
    metadata = torch.load(metadata_path, map_location="cpu")
    errors = []
    if int(metadata.get("num_observations", -1)) != OBS_DIM:
        errors.append(f"num_observations={metadata.get('num_observations')} expected {OBS_DIM}")
    if int(metadata.get("num_actions", -1)) != ACTION_DIM:
        errors.append(f"num_actions={metadata.get('num_actions')} expected {ACTION_DIM}")
    if bool(metadata.get("normalize_input", True)):
        errors.append("normalize_input must be false for this checkpoint")
    if errors:
        raise RuntimeError("Policy metadata mismatch: " + "; ".join(errors))
    return {"metadata_found": True, "metadata": metadata}


def load_torchscript_policy(policy_path: Path):
    if not policy_path.exists():
        raise FileNotFoundError(f"Exported policy not found: {policy_path}")
    validate_policy_metadata(policy_path)
    torch = _require_module("torch")
    return torch, torch.jit.load(str(policy_path), map_location="cpu").eval()


def sleep_for_realtime(start_time: float, dt: float = CONTROL_DT) -> None:
    sleep_time = float(dt) - (time.time() - start_time)
    if sleep_time > 0.0:
        time.sleep(sleep_time)
