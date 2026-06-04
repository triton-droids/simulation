from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "tritonhumanoid"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tritonhumanoid.eval import mujoco_locomotion as ml


def _canonical_mjcf_text() -> str:
    joints = "\n".join(
        f'      <joint name="{name}" type="hinge" axis="1 0 0" range="-1 1"/>'
        for name in ml.CH_MUJOCO_JOINT_NAMES
    )
    return f"""
<mujoco model="test_ch_robot">
  <compiler angle="radian"/>
  <worldbody>
    <body name="torso">
{joints}
    </body>
  </worldbody>
</mujoco>
"""


def test_patches_canonical_mjcf_to_active_isaac_urdf_contract() -> None:
    patched = ml.patch_mjcf_text_to_isaac_urdf(_canonical_mjcf_text(), ml.DEFAULT_ACTIVE_URDF)

    ml.validate_mjcf_text_against_urdf(patched, ml.DEFAULT_ACTIVE_URDF)
    root = ET.fromstring(patched)
    option = root.find("option")
    assert option is not None
    assert float(option.attrib["timestep"]) == pytest.approx(ml.PHYSICS_DT)

    left_thigh = root.find(".//joint[@name='left_thigh_joint']")
    left_hip2 = root.find(".//joint[@name='left_hip2_joint']")
    right_hip2 = root.find(".//joint[@name='right_hip2_joint']")
    assert left_thigh is not None
    assert left_hip2 is not None
    assert right_hip2 is not None
    assert left_thigh.attrib["axis"] == "0 0 -1"
    assert left_thigh.attrib["range"] == "-0.785398 0.785398"
    assert left_hip2.attrib["range"] == "-1.57 0.436332"
    assert right_hip2.attrib["range"] == "-0.436332 1.57"
    assert left_thigh.attrib["armature"] == "0.01"

    root_body = root.find(f"./worldbody/body[@name='{ml.MUJOCO_ROOT_BODY_NAME}']")
    assert root_body is not None
    assert root_body.find("freejoint") is not None
    torso = root_body.find("./body[@name='torso']")
    assert torso is not None
    assert torso.attrib["pos"] == "0.1505 0.008 -0.6996"

    actuator = root.find("actuator")
    assert actuator is not None
    assert [child.tag for child in actuator] == ["motor"] * len(ml.CH_MUJOCO_JOINT_NAMES)
    assert [child.attrib["joint"] for child in actuator] == list(ml.CH_MUJOCO_JOINT_NAMES)
    assert [child.attrib["ctrlrange"] for child in actuator] == ["-120 120"] * len(ml.CH_MUJOCO_JOINT_NAMES)


def test_observation_stack_matches_isaac_flattening_order() -> None:
    np = pytest.importorskip("numpy")
    stack = ml.ObservationStack(obs_dim=2, stack_frames=3)

    reset = stack.reset(np.asarray([1.0, 2.0], dtype=np.float32))
    appended = stack.append(np.asarray([3.0, 4.0], dtype=np.float32))

    assert reset.tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    assert appended.tolist() == [1.0, 1.0, 3.0, 2.0, 2.0, 4.0]


def test_joint_order_permutation_maps_isaac_policy_to_mujoco_order() -> None:
    assert ml.ISAAC_POLICY_JOINT_NAMES != ml.CH_MUJOCO_JOINT_NAMES
    perm = ml.joint_order_permutation(ml.ISAAC_POLICY_JOINT_NAMES, ml.CH_MUJOCO_JOINT_NAMES)

    assert perm == (0, 2, 4, 6, 8, 1, 3, 5, 7, 9)
    assert tuple(ml.ISAAC_POLICY_JOINT_NAMES[i] for i in perm) == ml.CH_MUJOCO_JOINT_NAMES


def test_joint_order_permutation_rejects_missing_or_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        ml.joint_order_permutation(["a", "a"], ["a", "b"])
    with pytest.raises(ValueError, match="missing"):
        ml.joint_order_permutation(["a", "b"], ["a", "c"])


def test_delayed_pd_controller_uses_trained_joint_contract() -> None:
    np = pytest.importorskip("numpy")
    controller = ml.DelayedPDController(ml.CH_MUJOCO_JOINT_NAMES, np.random.default_rng(0))
    controller.reset()

    assert controller.kp.shape == (10,)
    assert controller.kd.shape == (10,)
    assert controller.effort.tolist() == [120.0] * 10
    assert controller.velocity_limit.tolist() == [15.0] * 10
    assert controller.armature.tolist() == [0.01] * 10
    assert controller.kp[ml.CH_MUJOCO_JOINT_NAMES.index("left_knee_joint")] == pytest.approx(150.0)
    assert controller.kd[ml.CH_MUJOCO_JOINT_NAMES.index("right_ankle_joint")] == pytest.approx(1.0)

    tau = controller.torque(np.ones(10), np.zeros(10), np.zeros(10))
    assert tau.shape == (10,)
    assert np.all(tau <= 120.0)


def test_joint_velocity_limit_resolves_default_env_and_override(monkeypatch) -> None:
    monkeypatch.delenv("MUJOCO_JOINT_VELOCITY_LIMIT", raising=False)
    assert ml.resolve_joint_velocity_limit() == pytest.approx(15.0)

    monkeypatch.setenv("MUJOCO_JOINT_VELOCITY_LIMIT", "12.5")
    assert ml.resolve_joint_velocity_limit() == pytest.approx(12.5)
    assert ml.resolve_joint_velocity_limit(9.0) == pytest.approx(9.0)


def test_joint_velocity_clipping_can_be_disabled() -> None:
    np = pytest.importorskip("numpy")
    qvel = np.asarray([20.0, -18.0, 3.0], dtype=np.float64)
    qvel_addr = np.asarray([0, 1], dtype=np.int64)

    ml.clip_joint_velocity_array(np, qvel, qvel_addr, 15.0)
    assert qvel.tolist() == [15.0, -15.0, 3.0]

    qvel = np.asarray([20.0, -18.0, 3.0], dtype=np.float64)
    ml.clip_joint_velocity_array(np, qvel, qvel_addr, 0.0)
    assert qvel.tolist() == [20.0, -18.0, 3.0]


def test_dummy_mjcf_fails_with_clear_error(tmp_path: Path) -> None:
    placeholder = tmp_path / "ch_robot_10dof.xml"
    ml.create_dummy_mjcf(placeholder)
    with pytest.raises(ml.PlaceholderMJCFError, match="placeholder"):
        ml.ensure_isaac_locomotion_mjcf(source_xml=placeholder, model_dir=tmp_path / "cache")


def test_exported_policy_metadata_contract_when_available(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    policy = tmp_path / "ppo_policy.pt"
    policy.write_bytes(b"placeholder")
    torch.save(
        {
            "num_observations": ml.OBS_DIM,
            "num_actions": ml.ACTION_DIM,
            "normalize_input": False,
        },
        policy.with_name("ppo_metadata.pt"),
    )

    result = ml.validate_policy_metadata(policy)
    assert result["metadata_found"] is True


def test_eval_module_imports_without_runtime_sim_dependencies() -> None:
    assert importlib.util.find_spec("tritonhumanoid.eval.mujoco_locomotion") is not None
