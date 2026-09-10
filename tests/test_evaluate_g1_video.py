import os
from datetime import datetime, timedelta
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import jax.numpy as jp
import numpy as np

if sys.platform == "win32":
    os.environ["MUJOCO_GL"] = "glfw"

from source.scripts import evaluate_g1


def test_evaluation_timestamp_is_explicit_utc() -> None:
    timestamp = datetime.fromisoformat(evaluate_g1._utc_timestamp())

    assert timestamp.utcoffset() == timedelta(0)


def _synthetic_trace(*, terminal_done: bool) -> dict[str, np.ndarray]:
    steps = 4
    scalar_zero = np.zeros(steps, dtype=np.float32)
    trace = {
        "valid": np.ones(steps, dtype=bool),
        "done": scalar_zero.copy(),
        "linear_error": np.zeros((steps, 2), dtype=np.float32),
        "yaw_error": scalar_zero.copy(),
        "reward": scalar_zero.copy(),
        "torso_up_z": np.ones(steps, dtype=np.float32),
        "pelvis_height": np.full(steps, 0.75, dtype=np.float32),
    }
    for name in (
        "effort",
        "power",
        "action_rate",
        "action_clipping_fraction",
        "action_saturation_fraction",
        "target_saturation_fraction",
        "maximum_abs_applied_action",
        "joint_acceleration",
        "foot_slip",
        "undesired_contact",
        "collision_contact",
        "nonfoot_ground_contact",
        "self_collision_contact",
        "joint_limit_violation",
        "left_contact",
        "right_contact",
        "left_contact_transition",
        "right_contact_transition",
        "cause_low",
        "cause_high",
        "cause_inverted",
        "cause_undesired",
        "cause_invalid",
    ):
        trace[name] = scalar_zero.copy()
    trace["done"][-1] = float(terminal_done)
    return trace


def test_terminal_on_final_step_is_not_labeled_successful() -> None:
    summary = evaluate_g1._summarize_trace(
        _synthetic_trace(terminal_done=True),
        controller="synthetic",
        command_name="stand",
        command=(0.0, 0.0, 0.0),
        seed=0,
        dt=0.02,
        requested_steps=4,
    )

    assert summary["fall"] is True
    assert summary["survived_full_horizon"] is False
    assert summary["episode_success"] is False


def test_action_diagnostics_use_applied_action_and_separate_target_limits() -> None:
    class FakeEnvironment:
        ctrl_lower = jp.array([-0.25, -0.5])
        ctrl_upper = jp.array([0.25, 0.5])

        @staticmethod
        def action_to_targets(action):
            applied = jp.clip(action, -1.0, 1.0)
            targets = jp.clip(
                0.5 * applied,
                FakeEnvironment.ctrl_lower,
                FakeEnvironment.ctrl_upper,
            )
            return applied, targets

    applied, targets, diagnostics = evaluate_g1._action_diagnostics(
        FakeEnvironment(), jp.array([2.0, -0.5])
    )

    np.testing.assert_allclose(applied, [1.0, -0.5])
    np.testing.assert_allclose(targets, [0.25, -0.25])
    assert float(diagnostics["action_clipping_fraction"]) == 0.5
    assert float(diagnostics["action_saturation_fraction"]) == 0.5
    assert float(diagnostics["target_saturation_fraction"]) == 0.5
    assert float(diagnostics["maximum_abs_applied_action"]) == 1.0


def test_action_rate_uses_clipped_actions_not_raw_policy_outputs() -> None:
    # Raw [2, -3] would have cost 13 from the prior zero action.  The evaluator
    # must agree with the environment, which applies [1, -1] and has cost 2.
    applied = jp.clip(jp.array([2.0, -3.0]), -1.0, 1.0)

    cost = evaluate_g1._applied_action_rate(applied, jp.zeros(2))

    assert float(cost) == 2.0


def test_contact_breakdown_separates_nonfoot_ground_and_self_collision() -> None:
    env = SimpleNamespace(
        floor_geom_id=0,
        foot_geom_ids=(jp.array([10]), jp.array([11])),
    )
    pipeline_state = SimpleNamespace(
        contact=SimpleNamespace(
            # foot--floor support, torso--floor contact, and hand--thigh contact
            geom=jp.array([[0, 10], [0, 20], [30, 40]]),
            dist=jp.array([-0.01, -0.02, -0.03]),
        )
    )

    nonfoot_ground, self_collision = evaluate_g1._contact_breakdown(
        env, pipeline_state
    )

    assert bool(nonfoot_ground)
    assert bool(self_collision)


def test_contact_breakdown_does_not_label_clean_foot_support_as_collision() -> None:
    env = SimpleNamespace(
        floor_geom_id=0,
        foot_geom_ids=(jp.array([10]), jp.array([11])),
    )
    pipeline_state = SimpleNamespace(
        contact=SimpleNamespace(
            geom=jp.array([[0, 10], [11, 0], [30, 40]]),
            # The padded/self pair is inactive.
            dist=jp.array([-0.01, -0.02, 1.0]),
        )
    )

    nonfoot_ground, self_collision = evaluate_g1._contact_breakdown(
        env, pipeline_state
    )

    assert not bool(nonfoot_ground)
    assert not bool(self_collision)


def test_gait_summary_reports_support_and_contact_transitions() -> None:
    trace = _synthetic_trace(terminal_done=False)
    trace["left_contact"] = np.array([1, 1, 0, 0], dtype=np.float32)
    trace["right_contact"] = np.array([1, 0, 1, 0], dtype=np.float32)
    trace["left_contact_transition"] = np.array([0, 0, 1, 0], dtype=np.float32)
    trace["right_contact_transition"] = np.array([0, 1, 0, 1], dtype=np.float32)

    summary = evaluate_g1._summarize_trace(
        trace,
        controller="synthetic",
        command_name="forward",
        command=(0.5, 0.0, 0.0),
        seed=0,
        dt=0.02,
        requested_steps=4,
    )

    assert summary["double_support_fraction"] == 0.25
    assert summary["single_support_fraction"] == 0.5
    assert summary["flight_fraction"] == 0.25
    assert summary["left_only_support_fraction"] == 0.25
    assert summary["right_only_support_fraction"] == 0.25
    assert summary["left_contact_transition_count"] == 1
    assert summary["right_contact_transition_count"] == 2
    assert summary["contact_transitions_per_second"] == 37.5
    assert summary["both_feet_transitioned"] is True


def test_git_record_is_explicit_when_export_has_no_git(monkeypatch) -> None:
    def no_repository(*_args, **_kwargs):
        raise evaluate_g1.subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(evaluate_g1.subprocess, "run", no_repository)

    assert evaluate_g1._git_record() == {
        "available": False,
        "commit": None,
        "branch": None,
        "dirty": None,
    }


def test_video_falls_back_to_opencv_when_ffmpeg_is_missing(
    monkeypatch, tmp_path: Path
) -> None:
    frames = [
        np.full((48, 64, 3), fill_value, dtype=np.uint8)
        for fill_value in (0, 80, 160)
    ]

    def missing_ffmpeg(*_args, **_kwargs) -> None:
        raise RuntimeError("Program 'ffmpeg' is not found")

    monkeypatch.setattr(evaluate_g1.media, "write_video", missing_ffmpeg)
    output = tmp_path / "fallback.mp4"

    class FakeEnvironment:
        dt = 0.02

        @staticmethod
        def render(*_args, **_kwargs):
            return frames

    backend = evaluate_g1._write_video(
        FakeEnvironment(), object(), {"pipeline_state": []}, 0, output, 2
    )

    assert backend == "opencv_mp4v"
    assert output.stat().st_size > 0
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames)
    finally:
        capture.release()
