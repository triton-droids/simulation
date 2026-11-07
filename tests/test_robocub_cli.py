"""Protect the beginner CLI paths and shared PyCharm run buttons."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from robocub.user_cli import cli


def test_parser_exposes_beginner_commands() -> None:
    """Every shared command name should resolve to an executable handler."""

    parser = cli.build_parser()
    for command in ("watch", "train", "evaluate", "status", "doctor"):
        args = parser.parse_args([command])
        assert callable(args.handler)


def test_status_reports_checkpoint_path(capsys) -> None:
    """Status should identify the robot and canonical T02 checkpoint."""

    result = cli.command_status(argparse.Namespace(json=False))
    output = capsys.readouterr().out
    assert "Unitree G1" in output
    assert "1003520" in output.replace(",", "")
    assert str(cli.T02_POLICY) in output
    assert result in (0, 2)


def test_watch_selects_field_launcher(monkeypatch, tmp_path: Path) -> None:
    """Field mode should delegate to the field-aware MuJoCo viewer."""

    policy = tmp_path / "policy"
    policy.write_bytes(b"weights")
    calls: list[tuple[Path, list[str]]] = []
    monkeypatch.setattr(cli, "T02_POLICY", policy)
    monkeypatch.setattr(
        cli,
        "_run",
        lambda script, arguments=(): calls.append((script, list(arguments))) or 0,
    )
    args = argparse.Namespace(scene="field", record=None, extra=[])
    assert cli.command_watch(args) == 0
    assert calls[0][0].name == "watch_g1_soccer.py"
    assert calls[0][0].parent.name == "tools"


def test_shared_pycharm_configurations_are_valid_xml() -> None:
    """All documented push-button workflows should have parseable XML."""

    run_directory = cli.ROOT / ".run"
    configurations = sorted(run_directory.glob("*.run.xml"))
    assert [path.stem for path in configurations] == [
        "1 Watch Robot.run",
        "2 Train Smoke.run",
        "3 Show Status.run",
        "4 Evaluate T02.run",
        "5 Sweep Smoke.run",
        "6 Cluster Dashboard.run",
    ]
    for configuration in configurations:
        ET.parse(configuration)
