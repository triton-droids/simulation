"""Helpers for saving inference-only policy parameters across Brax layouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def inference_params_from_training_params(params: Any) -> tuple[Any, Any]:
    """Return ``(normalizer, policy)`` from supported PPO parameter trees.

    Brax 0.14.2 exposes training parameters as
    ``(normalizer, policy, value)``.  Some older releases instead nest policy
    and value parameters in the second element.  The compact checkpoint used
    for inference must contain neither optimizer state nor value parameters.
    """

    if not isinstance(params, Sequence) or isinstance(params, (str, bytes)):
        raise TypeError("PPO training parameters must be a sequence")
    if len(params) < 2:
        raise ValueError("PPO training parameters must contain at least two items")

    normalizer_params = params[0]
    network_params = params[1]

    if len(params) >= 3:
        # Current Brax layout: normalizer, policy, value.
        policy_params = network_params
    elif isinstance(network_params, Mapping) and "policy" in network_params:
        # Historical layout: normalizer, {policy, value}.
        policy_params = network_params["policy"]
    elif hasattr(network_params, "policy"):
        # Historical named container with a policy attribute.
        policy_params = network_params.policy
    else:
        # Already inference-shaped: normalizer, policy.
        policy_params = network_params

    return normalizer_params, policy_params
