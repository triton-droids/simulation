"""Tests for credential-free metrics logging."""

from __future__ import annotations

import builtins
import json

from source.utils.experiment_logging import create_metric_logger


def test_local_logger_writes_jsonl_without_importing_wandb(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "wandb" or name.startswith("wandb."):
            raise AssertionError("local logging must not import W&B")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    logger = create_metric_logger(
        mode="local",
        output_dir=tmp_path,
        run_name="offline-test",
        config={"seed": 7},
    )
    logger.log({"reward": 1.25}, step=3)
    logger.close()

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == ["run_start", "metrics", "run_end"]
    assert records[1]["step"] == 3
    assert records[1]["metrics"]["reward"] == 1.25


def test_disabled_logger_needs_no_files_or_cloud_package(tmp_path):
    logger = create_metric_logger(
        mode="none",
        output_dir=tmp_path,
        run_name="disabled-test",
        config={},
    )
    logger.log({"ignored": 1}, step=0)
    logger.close()
    assert list(tmp_path.iterdir()) == []
