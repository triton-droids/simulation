"""Helpers for saving inference-only policy parameters across Brax layouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def set_unseen_g1_command_std(params: Any, lateral_std: float, yaw_std: float) -> tuple:
    """Give previously constant-zero G1 command channels a declared variance prior.

    Only lateral/yaw indices 10/11 in actor and critic observations change.
    Means, counts, weights and all other statistics stay unchanged. Updating
    both std and Welford summed variance keeps subsequent updates consistent.
    This is an explicit parameter preparation, never an automatic restore edit.
    """
    import jax.numpy as jp
    import numpy as np

    if len(params) != 3:
        raise ValueError("Expected full normalizer/actor/critic checkpoint")
    targets = np.asarray([lateral_std, yaw_std], dtype=float)
    if not np.isfinite(targets).all() or np.any(targets <= 1e-5):
        raise ValueError("Command std priors must be finite and above 1e-5")
    stats = params[0]
    if int(stats.mode) != 0:
        raise ValueError("Command prior requires Welford statistics")
    expected_sizes = {"state": 103, "privileged_state": 216}
    if set(stats.mean) != set(expected_sizes):
        raise ValueError("Expected pinned G1 actor/critic observation keys")
    count = stats.count
    count_value = (float(count.hi) * 2**32 + float(count.lo)
                   if hasattr(count, "hi") else float(count))
    if count_value <= 0:
        raise ValueError("Command prior requires populated running statistics")
    std = dict(stats.std)
    summed = dict(stats.summed_variance)
    for key, size in expected_sizes.items():
        if np.shape(stats.mean[key]) != (size,):
            raise ValueError("Unexpected G1 observation layout")
        if (np.any(np.abs(np.asarray(stats.mean[key])[10:12]) > 1e-8)
                or np.any(np.asarray(stats.std[key])[10:12] > 1e-5)):
            raise ValueError("Refusing to replace statistics of already-varying commands")
        std[key] = jp.asarray(stats.std[key]).at[10:12].set(targets)
        variance = targets**2 - float(stats.std_eps)
        if np.any(variance <= 0):
            raise ValueError("Prior variance must exceed normalization epsilon")
        summed[key] = jp.asarray(stats.summed_variance[key]).at[10:12].set(variance * count_value)
    return (stats.replace(std=std, summed_variance=summed), params[1], params[2])


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
