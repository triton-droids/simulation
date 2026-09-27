"""Validated mappings from repository PPO config to Brax runtime controls."""

from __future__ import annotations

from typing import Any
import math


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

    controls = {
        "action_repeat": action_repeat,
        "normalize_observations": bool(train_cfg.normalize_observations),
        "max_grad_norm": max_grad_norm,
    }
    schedule = getattr(train_cfg, "learning_rate_schedule", "NONE")
    if schedule not in ("NONE", "ADAPTIVE_KL"):
        raise ValueError("Unsupported PPO learning-rate schedule")
    if schedule == "ADAPTIVE_KL":
        low = float(train_cfg.learning_rate_schedule_min_lr)
        high = float(train_cfg.learning_rate_schedule_max_lr)
        target = float(train_cfg.desired_kl)
        initial = float(train_cfg.learning_rate)
        if not all(math.isfinite(v) and v > 0 for v in (low, high, target, initial)) or not low <= initial <= high:
            raise ValueError("Adaptive KL requires positive finite target and ordered learning-rate bounds")
        controls.update(learning_rate_schedule=schedule,
                        learning_rate_schedule_min_lr=low,
                        learning_rate_schedule_max_lr=high,
                        desired_kl=target)
    return controls
