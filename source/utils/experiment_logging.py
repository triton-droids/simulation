"""Credential-free local metrics logging with optional W&B adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol


class MetricLogger(Protocol):
    """Minimal interface used by the training loop."""

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        """Record one metrics event."""

    def close(self) -> None:
        """Flush and release logger resources."""


def _json_value(value: Any) -> Any:
    """Convert scalar array-like values to JSON-compatible Python values."""

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class NullLogger:
    """No-op logger used by tests and explicitly logging-disabled runs."""

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        del metrics, step

    def close(self) -> None:
        return None


class JsonlLogger:
    """Append metrics to a local JSON Lines file without network access."""

    def __init__(self, output_dir: Path, run_name: str, config: Mapping[str, Any]):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / "metrics.jsonl"
        self._stream = self.path.open("a", encoding="utf-8")
        self._write({"event": "run_start", "run_name": run_name, "config": config})

    def _write(self, payload: Mapping[str, Any]) -> None:
        self._stream.write(json.dumps(_json_value(payload), sort_keys=True) + "\n")
        self._stream.flush()

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        self._write({"event": "metrics", "step": int(step), "metrics": metrics})

    def close(self) -> None:
        if not self._stream.closed:
            self._write({"event": "run_end"})
            self._stream.close()


class WandbLogger:
    """Opt-in W&B adapter; importing W&B is deferred until requested."""

    def __init__(
        self,
        mode: str,
        run_name: str,
        project: str | None,
        config: Mapping[str, Any],
        output_dir: Path,
    ):
        try:
            import wandb
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "W&B logging was requested but wandb is not installed. "
                "Install the optional dependency or use --logger local/none."
            ) from error

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=run_name,
            config=_json_value(config),
            mode="offline" if mode == "wandb-offline" else "online",
            dir=str(output_dir),
        )

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        self._wandb.log(dict(metrics), step=step)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


def create_metric_logger(
    mode: str,
    output_dir: Path,
    run_name: str,
    config: Mapping[str, Any],
    project: str | None = None,
) -> MetricLogger:
    """Create a local, disabled, offline-W&B, or online-W&B logger."""

    if mode == "none":
        return NullLogger()
    if mode == "local":
        return JsonlLogger(output_dir, run_name, config)
    if mode in {"wandb-offline", "wandb"}:
        return WandbLogger(mode, run_name, project, config, output_dir)
    raise ValueError(f"Unknown logger mode: {mode}")
