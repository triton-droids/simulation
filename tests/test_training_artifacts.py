import json

from robocub.utils.training_artifacts import TrainingArtifactMonitor


def test_training_monitor_writes_summary_progress_and_plot(tmp_path):
    monitor = TrainingArtifactMonitor(tmp_path)
    monitor.record(0, {"eval/episode_reward": 1.0, "training/kl_mean": 0.01})
    monitor.record(100, {"eval/episode_reward": 2.0, "training/kl_mean": 0.02})

    summary = monitor.finalize(
        status="completed",
        best_checkpoint_step=100,
        best_episode_reward=2.0,
        wall_time_seconds=3.5,
    )

    assert summary["final_step"] == 100
    assert (tmp_path / "plots" / "final_progress.png").is_file()
    assert json.loads((tmp_path / "summary.json").read_text())["best_episode_reward"] == 2.0
    assert len(json.loads((tmp_path / "progress.json").read_text())) == 2
