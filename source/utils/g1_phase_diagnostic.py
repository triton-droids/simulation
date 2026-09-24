"""Diagnostic-only intervention on raw policy phase inputs, never simulator state."""
import jax.numpy as jp


def replace_policy_phase(observation, angle):
    if angle is None:
        return observation
    phases = jp.array([angle, angle + jp.pi])
    encoded = jp.concatenate([jp.cos(phases), jp.sin(phases)])
    return {key: value.at[99:103].set(encoded) for key, value in observation.items()}
