"""Protect the cluster dashboard's parsing, settings, and safety boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from robocub.user_cli.cluster_dashboard import (
    ClusterService,
    parse_env,
    parse_jobs,
    update_env,
    windows_path,
)


def test_parse_jobs_reads_return_script_rows() -> None:
    """Dashboard rows should preserve remote status and completed trial counts."""

    output = """
Running:
  Sweep_12 | status=running | task=Unitree G1 PPO sweep | policy=ppo_g1 | submitted=2026-11-07T20:25:09Z | completed_trials=3 | exit_code=unknown
Finished:
  Sweep_11 | status=succeeded | task=Unitree G1 PPO sweep | policy=ppo_g1 | submitted=2026-11-07T19:00:00Z | completed_trials=4 | exit_code=0
"""
    jobs = parse_jobs(output)
    assert [(job.label, job.status, job.completed_trials) for job in jobs] == [
        ("Sweep_12", "running", 3),
        ("Sweep_11", "succeeded", 4),
    ]
    assert jobs[0].active is True
    assert jobs[1].active is False


def test_update_env_changes_only_resource_values(tmp_path: Path) -> None:
    """Saving GUI resources must leave branch, host, and key settings untouched."""

    env_path = tmp_path / ".env"
    env_path.write_text(
        'BRANCH_NAME="feature"\nSSH_KEY_PATH="/private/key"\nSLURM_CPUS_PER_TASK="8"\n',
        encoding="utf-8",
    )
    update_env(
        {
            "SLURM_CPUS_PER_TASK": "4",
            "SLURM_GPU_COUNT": "2",
            "SLURM_MEMORY": "64G",
        },
        env_path,
    )
    settings = parse_env(env_path)
    assert settings["BRANCH_NAME"] == "feature"
    assert settings["SSH_KEY_PATH"] == "/private/key"
    assert settings["SLURM_CPUS_PER_TASK"] == "4"
    assert settings["SLURM_GPU_COUNT"] == "2"
    assert settings["SLURM_MEMORY"] == "64G"


def test_update_env_rejects_non_resource_keys(tmp_path: Path) -> None:
    """The GUI must never rewrite SSH or repository identity settings."""

    env_path = tmp_path / ".env"
    env_path.write_text('ATLANTIS_HOST="example"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        update_env({"ATLANTIS_HOST": "attacker"}, env_path)


def test_windows_path_converts_git_bash_drive_paths() -> None:
    """Native SSH should receive a Windows path rather than Git Bash syntax."""

    assert windows_path("/c/Users/brand/.ssh/key").lower().startswith("c:")
    assert windows_path("relative/key") == "relative/key"


def test_cluster_service_rejects_unsafe_remote_labels() -> None:
    """Cancellation and downloads must remain inside numbered sweep folders."""

    for label in ("Sweep_2", "Sweep_200"):
        ClusterService._validate_label(label)
    for label in ("../Sweep_2", "Sweep_x", "Sweep_2/other"):
        with pytest.raises(ValueError, match="Invalid sweep label"):
            ClusterService._validate_label(label)
