"""Diagnostic-only intervention on raw policy phase inputs, never simulator state."""
import jax.numpy as jp


def replace_policy_phase(observation, angle, double_stance=False):
    if angle is None and not double_stance:
        return observation
    phases = jp.ones(2) * jp.pi if double_stance else jp.array([angle, angle + jp.pi])
    encoded = jp.concatenate([jp.cos(phases), jp.sin(phases)])
    return {key: value.at[99:103].set(encoded) for key, value in observation.items()}
