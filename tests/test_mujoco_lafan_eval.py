from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import pytest

from tritonhumanoid.eval import mujoco_lafan as ml


def _canonical_mjcf_text() -> str:
    if not ml.DEFAULT_MUJOCO_REPO.exists():
        pytest.skip(f"MuJoCo repo not found: {ml.DEFAULT_MUJOCO_REPO}")
    try:
        return subprocess.check_output(
            [
                "git",
                "show",
                f"{ml.CH_ROBOT_BRANCH_REF}:{ml.CH_ROBOT_BRANCH_PATH}/ch_robot_10dof.xml",
            ],
            cwd=ml.DEFAULT_MUJOCO_REPO,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"canonical ch_robot MJCF is unavailable from git: {exc}")


def test_patches_canonical_mjcf_to_active_isaac_urdf_contract() -> None:
    patched = ml.patch_mjcf_text_to_isaac_urdf(_canonical_mjcf_text(), ml.DEFAULT_ACTIVE_URDF)

    ml.validate_mjcf_text_against_urdf(patched, ml.DEFAULT_ACTIVE_URDF)

    root = ET.fromstring(patched)
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

    world = root.find("./worldbody/body[@name='world']")
    assert world is not None
    assert world.find("freejoint") is not None
    torso = world.find("./body[@name='torso']")
    assert torso is not None
    assert torso.attrib["pos"] == "0.1505 0.008 -0.6996"

    actuator = root.find("actuator")
    assert actuator is not None
    assert [child.tag for child in actuator] == ["motor"] * len(ml.CH_MUJOCO_JOINT_NAMES)
    assert [child.attrib["joint"] for child in actuator] == list(ml.CH_MUJOCO_JOINT_NAMES)


def test_observation_stack_matches_isaac_flattening_order() -> None:
    np = pytest.importorskip("numpy")

    stack = ml.ObservationStack(obs_dim=2, stack_frames=3)
    reset = stack.reset(np.asarray([1.0, 2.0], dtype=np.float32))
    appended = stack.append(np.asarray([3.0, 4.0], dtype=np.float32))

    assert reset.tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    assert appended.tolist() == [1.0, 1.0, 3.0, 2.0, 2.0, 4.0]


def test_delayed_pd_controller_uses_trained_joint_contract() -> None:
    np = pytest.importorskip("numpy")

    controller = ml.DelayedPDController(ml.CH_MUJOCO_JOINT_NAMES, np.random.default_rng(0))
    controller.reset()

    assert controller.kp.shape == (10,)
    assert controller.kd.shape == (10,)
    assert controller.effort.tolist() == [120.0] * 10
    assert controller.kp[ml.CH_MUJOCO_JOINT_NAMES.index("left_knee_joint")] == pytest.approx(150.0)
    assert controller.kd[ml.CH_MUJOCO_JOINT_NAMES.index("right_ankle_joint")] == pytest.approx(1.0)

    tau = controller.torque(np.ones(10), np.zeros(10), np.zeros(10))
    assert tau.shape == (10,)
    assert np.all(tau <= 120.0)


def test_exported_policy_metadata_contract_when_available() -> None:
    metadata_path = Path(
        "logs/rl_games/humanoid_flat_direct/lafan_walk_tracking/exported_policy/ppo_metadata.pt"
    )
    if not metadata_path.exists():
        pytest.skip("exported policy metadata has not been generated")
    torch = pytest.importorskip("torch")

    metadata = torch.load(metadata_path, map_location="cpu")
    assert metadata["num_observations"] == ml.OBS_DIM
    assert metadata["num_actions"] == ml.ACTION_DIM
    assert metadata["normalize_input"] is False


def test_eval_module_imports_without_runtime_sim_dependencies() -> None:
    assert importlib.util.find_spec("tritonhumanoid.eval.mujoco_lafan") is not None
