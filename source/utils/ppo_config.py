"""Validated mappings from repository PPO config to Brax runtime controls."""

from __future__ import annotations

from typing import Any


def trainer_runtime_controls(train_cfg: Any) -> dict[str, object]:
    """Return safety-relevant Brax arguments that must not use library defaults.

    These values already existed in the repository configuration, but Brax has
    materially different defaults. Keeping the conversion here makes the
    forwarding behavior independently testable without importing the Hydra
    training entry point.
    """

    action_repeat_value = float(train_cfg.action_repeat)
    action_repeat = int(action_repeat_value)
    if action_repeat <= 0 or action_repeat != action_repeat_value:
        raise ValueError("PPO action_repeat must be a positive integer.")

    max_grad_norm = train_cfg.max_grad_norm
    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)
        if max_grad_norm <= 0.0:
            raise ValueError("PPO max_grad_norm must be positive or None.")

    return {
        "action_repeat": action_repeat,
        "normalize_observations": bool(train_cfg.normalize_observations),
        "max_grad_norm": max_grad_norm,
    }
