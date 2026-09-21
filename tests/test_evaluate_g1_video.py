import os
import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import jax
import jax.numpy as jp
import numpy as np
import pytest

if sys.platform == "win32":
    os.environ["MUJOCO_GL"] = "glfw"

from source.scripts import evaluate_g1


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_numeric_batches_preserve_episode_order_and_exclude_padding(batch_size):
    def episode(params, use_policy, seed, command):
        signal = seed + jp.sum(command) + jp.where(use_policy, params, 0.)
        return jp.zeros(1), {"reward": jp.arange(3) + signal,
                             "valid": jp.array([True, True, False])}
    rollout = jax.jit(episode if batch_size == 1 else
                      jax.vmap(episode, in_axes=(None, None, 0, 0)))
    commands = [("forward", (.5, 0., 0.)), ("left", (0., .3, 0.))]
    seeds = [11, 12, 13]
    rows = list(evaluate_g1._numeric_episode_traces(
        rollout, jp.asarray(2.), True, commands, seeds, batch_size))
    assert len(rows) == 6
    for row, (values, seed) in zip(rows, [(v, s) for _, v in commands for s in seeds]):
        np.testing.assert_allclose(row["reward"], np.arange(3) + seed + sum(values) + 2.)
        np.testing.assert_array_equal(row["valid"], [True, True, False])


def test_completed_air_intervals_exclude_trace_boundaries_and_stance():
    # Initial and final air fragments are censored; the middle swing is 60 ms.
    contact = np.array([False, False, True, False, False, False, True, False])
    assert evaluate_g1._completed_air_intervals(contact, .02) == pytest.approx([.06])
    assert evaluate_g1._completed_air_intervals(np.ones(8, dtype=bool), .02) == []
    assert evaluate_g1._completed_air_intervals(np.zeros(8, dtype=bool), .02) == []
    assert evaluate_g1._completed_air_intervals(np.array([True, False, True, False, False, True]), .02) == pytest.approx([.02, .04])


def test_explicit_command_grid_values_order_and_video_validation(tmp_path, monkeypatch):
    path = tmp_path / "commands.json"
    path.write_text(json.dumps({"left": [0, .25, 0], "forward": [.45, 0, 0]}))
    base = ["evaluate_g1", "--run-dir", "unused", "--command-grid", str(path)]
    monkeypatch.setattr(sys, "argv", base)
    args = evaluate_g1.parse_args()
    assert args.command_grid_entries == (("forward", (.45, 0., 0.)), ("left", (0., .25, 0.)))
    monkeypatch.setattr(sys, "argv", base + ["--video", "--video-command", "forward"])
    assert evaluate_g1.parse_args().video_command == "forward"
    monkeypatch.setattr(sys, "argv", base + ["--video"])
    with pytest.raises(SystemExit):
        evaluate_g1.parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--commands", "forward"])
    with pytest.raises(SystemExit):
        evaluate_g1.parse_args()


@pytest.mark.parametrize("raw", [{}, {"unknown": [0,0,0]}, {"forward": [1,2]},
                                {"forward": [True,0,0]}, {"forward": [float("nan"),0,0]}])
def test_explicit_command_grid_rejects_ambiguous_or_nonfinite_values(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        evaluate_g1._load_command_grid(path)


def test_video_command_selector(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate_g1", "--run-dir", "unused", "--video", "--video-command", "forward"])
    assert evaluate_g1.parse_args().video_command == "forward"
    monkeypatch.setattr(sys, "argv", ["evaluate_g1", "--run-dir", "unused"])
    assert evaluate_g1.parse_args().video_command == "combined"


def test_diagnostic_command_subset_and_video_must_agree(monkeypatch):
    base = ["evaluate_g1", "--run-dir", "unused", "--commands", "forward"]
    monkeypatch.setattr(sys, "argv", base)
    assert evaluate_g1.parse_args().commands == ["forward"]
    monkeypatch.setattr(sys, "argv", base + ["--video", "--video-command", "forward"])
    assert evaluate_g1.parse_args().video_command == "forward"
    monkeypatch.setattr(sys, "argv", base + ["--video"])
    with pytest.raises(SystemExit):
        evaluate_g1.parse_args()


@dataclass(frozen=True)
class _CommandState:
    info: dict[str, jp.ndarray]
    obs: dict[str, jp.ndarray]
    reward: jp.ndarray

    def replace(self, **updates):
        return replace(self, **updates)


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


def test_evaluation_network_reuses_training_observation_normalization() -> None:
    calls = []

    def factory(observation_size, action_size, **kwargs):
        calls.append((observation_size, action_size, kwargs))
        return "network"

    network = evaluate_g1._make_evaluation_network(
        factory,
        {"state": 103, "privileged_state": 216},
        29,
        normalize_observations=True,
    )

    assert network == "network"
    assert calls[0][0] == {"state": 103, "privileged_state": 216}
    assert calls[0][1] == 29
    assert (
        calls[0][2]["preprocess_observations_fn"]
        is evaluate_g1.running_statistics.normalize
    )


def test_evaluation_network_leaves_preprocessing_disabled_when_training_did() -> None:
    captured = {}

    def factory(_observation_size, _action_size, **kwargs):
        captured.update(kwargs)
        return "network"

    evaluate_g1._make_evaluation_network(
        factory, 103, 29, normalize_observations=False
    )

    assert "preprocess_observations_fn" not in captured


def test_held_command_is_used_for_reward_and_restored_after_resampling() -> None:
    class ResamplingEnvironment:
        @staticmethod
        def step(state, _action):
            seen_command = state.info["command"]
            info = dict(state.info)
            info["command"] = jp.full(3, 9.0)
            obs = {
                name: value.at[9:12].set(info["command"])
                for name, value in state.obs.items()
            }
            return state.replace(info=info, obs=obs, reward=seen_command[0])

    state = _CommandState(
        info={"command": jp.zeros(3)},
        obs={"state": jp.zeros(103), "privileged_state": jp.zeros(216)},
        reward=jp.zeros(()),
    )
    held = jp.array([0.4, -0.2, 0.3])

    next_state = evaluate_g1._step_with_held_command(
        ResamplingEnvironment(), state, jp.zeros(1), held
    )

    assert float(next_state.reward) == float(held[0])
    np.testing.assert_allclose(next_state.info["command"], held)
    np.testing.assert_allclose(next_state.obs["state"][9:12], held)
    np.testing.assert_allclose(next_state.obs["privileged_state"][9:12], held)


def test_playground_backend_maps_state_and_bounds_without_native_aliases() -> None:
    data = SimpleNamespace(qpos=jp.arange(4.0), qvel=jp.arange(3.0))
    state = SimpleNamespace(data=data)
    environment = SimpleNamespace(
        playground_source=object(),
        mj_model=SimpleNamespace(
            actuator_ctrlrange=np.array([[-1.0, 2.0], [-3.0, 4.0]])
        ),
        _env=SimpleNamespace(
            _soft_lowers=jp.array([-0.5, -0.25]),
            _soft_uppers=jp.array([0.5, 0.25]),
        ),
    )

    assert evaluate_g1._state_data(environment, state) is data
    np.testing.assert_allclose(evaluate_g1._qpos(environment, data), data.qpos)
    np.testing.assert_allclose(evaluate_g1._qvel(environment, data), data.qvel)
    lower, upper = evaluate_g1._control_bounds(environment)
    np.testing.assert_allclose(lower, [-1.0, -3.0])
    np.testing.assert_allclose(upper, [2.0, 4.0])
    soft_lower, soft_upper = evaluate_g1._soft_joint_bounds(environment)
    np.testing.assert_allclose(soft_lower, [-0.5, -0.25])
    np.testing.assert_allclose(soft_upper, [0.5, 0.25])
    instrumentation = evaluate_g1._instrumentation_metadata(environment)
    assert instrumentation["nonfoot_ground_contact_available"] is False


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


def test_c16_recovery_queue_uses_actual_evaluator_cli(monkeypatch):
    plan_path = Path(__file__).resolve().parents[1] / "research/queues/c16_evaluation_recovery.json"
    job = json.loads(plan_path.read_text())["jobs"][0]
    assert len(job["stages"]) == 1
    monkeypatch.setattr(sys, "argv", job["stages"][0]["argv"])
    args = evaluate_g1.parse_args()
    assert args.commands == job["gate"]["commands"] == ["stand", "forward"]
    assert args.checkpoint == 3368960
    assert args.steps == job["gate"]["horizon"] == 500
    assert args.seeds == "3000,3001"
    assert args.nominal_reset and args.video and args.video_command == "forward"


@pytest.mark.parametrize("filename,checkpoint", [("c17_gait.json",2007040), ("c18_yaw.json",1003520)])
def test_gait_queues_use_actual_evaluator_cli(monkeypatch, filename, checkpoint):
    path = Path(__file__).resolve().parents[1] / "research/queues" / filename
    job = json.loads(path.read_text())["jobs"][0]
    stage = next(s for s in job["stages"] if s["id"] == "evaluate")
    monkeypatch.setattr(sys, "argv", stage["argv"])
    args = evaluate_g1.parse_args()
    assert args.commands == job["gate"]["commands"] == ["forward"]
    assert args.checkpoint == checkpoint and args.reference_label == "initial"
    assert args.seeds == "3000,3001" and args.steps == 500
    assert args.nominal_reset and args.video


@pytest.mark.parametrize("filename,checkpoint,first_stage", [("c19_commands.json",2007040,"prepare"), ("c20_linear.json",1003520,"train")])
def test_command_queues_use_actual_evaluator_cli(monkeypatch, filename, checkpoint, first_stage):
    path = Path(__file__).resolve().parents[1] / "research/queues" / filename
    job = json.loads(path.read_text())["jobs"][0]
    stage = next(s for s in job["stages"] if s["id"] == "evaluate")
    monkeypatch.setattr(sys, "argv", stage["argv"])
    args = evaluate_g1.parse_args()
    assert args.commands == job["gate"]["commands"] == [name for name, _ in evaluate_g1.COMMANDS]
    assert args.checkpoint == checkpoint and args.reference_label == "initial"
    assert args.seeds == "4000" and args.steps == 500
    assert args.nominal_reset and args.video_command == "combined"
    assert job["stages"][0]["id"] == first_stage


def test_c20_randomized_queue_cli(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "research/queues/c20_randomized.json"
    job = json.loads(path.read_text())["jobs"][0]
    assert len(job["stages"]) == 1
    monkeypatch.setattr(sys, "argv", job["stages"][0]["argv"])
    args = evaluate_g1.parse_args()
    assert args.commands == job["gate"]["commands"]
    assert args.checkpoint == 1003520 and args.reference_label == "initial"
    assert args.seeds == "5000,5001" and args.steps == 500
    assert args.randomized_reset and not args.nominal_reset
    assert args.video and args.video_command == "combined"


def test_c21_evaluation_clis(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "research/queues/c21_recovery.json"
    jobs = json.loads(path.read_text())["jobs"]
    assert len(jobs) == 2
    for job, seeds, randomized in zip(jobs, ["4000", "5000,5001"], [False, True]):
        stage = next(s for s in job["stages"] if s["id"] == "evaluate")
        monkeypatch.setattr(sys, "argv", stage["argv"])
        args = evaluate_g1.parse_args()
        assert args.commands == job["gate"]["commands"]
        assert args.checkpoint == 1003520 and args.reference_label == "initial"
        assert args.seeds == seeds and args.randomized_reset == randomized
        assert args.nominal_reset != randomized
        assert args.video and args.video_command == "combined"
