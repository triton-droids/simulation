# Copyright 2026 Triton Droids
# Tracking kernels and reward family adapted from MuJoCo Playground commit
# 8a4b4642d8eba8a80ac99ed125cb62c16e1457ad (Apache-2.0).
"""Pure reward/cost functions for the standard Unitree G1 velocity task."""

from __future__ import annotations

import jax
import jax.numpy as jp

from source.tools.gait import get_rz


def aggregate_weighted_terms(
    weighted_terms: dict[str, jax.Array], control_dt: float
) -> jax.Array:
    """Sum signed terms and integrate them over one control interval.

    The G1 behavioral baseline does not clamp negative totals. In particular,
    retaining the signed termination cost gives PPO a survival signal.
    """

    return sum(weighted_terms.values(), start=jp.zeros(())) * control_dt


def tracking_linear_velocity(
    command: jax.Array, local_velocity: jax.Array, error_scale: float
) -> jax.Array:
    """Exponential planar velocity reward ``exp(-squared_error / scale)``."""

    squared_error = jp.sum(jp.square(command[:2] - local_velocity[:2]))
    return jp.exp(-squared_error / error_scale)


def tracking_yaw_rate(
    command: jax.Array, local_angular_velocity: jax.Array, error_scale: float
) -> jax.Array:
    """Exponential yaw-rate reward ``exp(-squared_error / scale)``."""

    squared_error = jp.square(command[2] - local_angular_velocity[2])
    return jp.exp(-squared_error / error_scale)


def vertical_velocity_cost(
    pelvis_global_velocity: jax.Array, torso_global_velocity: jax.Array
) -> jax.Array:
    return jp.square(pelvis_global_velocity[2]) + jp.square(torso_global_velocity[2])


def roll_pitch_angular_velocity_cost(global_angular_velocity: jax.Array) -> jax.Array:
    return jp.sum(jp.square(global_angular_velocity[:2]))


def upright_orientation_cost(
    torso_up_axis: jax.Array,
    target: jax.Array = jp.array([0.073, 0.0, 1.0]),
) -> jax.Array:
    return jp.sum(jp.square(torso_up_axis - target))


def actuator_effort_cost(actuator_force: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(actuator_force))


def mechanical_power_cost(joint_velocity: jax.Array, actuator_force: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(joint_velocity * actuator_force))


def action_rate_cost(action: jax.Array, previous_action: jax.Array) -> jax.Array:
    return jp.sum(jp.square(action - previous_action))


def joint_acceleration_cost(joint_acceleration: jax.Array) -> jax.Array:
    return jp.sum(jp.square(joint_acceleration))


def joint_limit_cost(
    joint_position: jax.Array, soft_lower: jax.Array, soft_upper: jax.Array
) -> jax.Array:
    below = -jp.clip(joint_position - soft_lower, max=0.0)
    above = jp.clip(joint_position - soft_upper, min=0.0)
    return jp.sum(below + above)


def foot_slip_cost(foot_xy_velocity: jax.Array, contact: jax.Array) -> jax.Array:
    """Squared world-frame foot speed while the corresponding foot contacts."""

    return jp.sum(jp.sum(jp.square(foot_xy_velocity), axis=-1) * contact)


def undesired_contact_cost(undesired_contact: jax.Array) -> jax.Array:
    return undesired_contact.astype(jp.float32)


def termination_cost(done: jax.Array) -> jax.Array:
    return done.astype(jp.float32)


def alive_reward() -> jax.Array:
    """Return the constant per-step survival reward."""

    return jp.array(1.0)


def foot_air_time_reward(
    air_time: jax.Array,
    first_contact: jax.Array,
    command: jax.Array,
    threshold_min: float = 0.2,
    threshold_max: float = 0.5,
) -> jax.Array:
    useful = jp.clip(air_time - threshold_min, max=threshold_max - threshold_min)
    return jp.sum(useful * first_contact) * (jp.linalg.norm(command) > 0.01)


def foot_phase_reward(
    foot_height: jax.Array,
    phase: jax.Array,
    swing_height: float,
    command: jax.Array,
) -> jax.Array:
    target = get_rz(phase, swing_height=swing_height)
    squared_error = jp.sum(jp.square(foot_height - target))
    return jp.exp(-squared_error / 0.01) * (jp.linalg.norm(command) > 0.01)


def stand_still_cost(
    command: jax.Array, joint_position: jax.Array, default_pose: jax.Array
) -> jax.Array:
    return jp.sum(jp.abs(joint_position - default_pose)) * (
        jp.linalg.norm(command) < 0.01
    )


def pose_cost(joint_position: jax.Array, default_pose: jax.Array) -> jax.Array:
    return jp.sum(jp.square(joint_position - default_pose))
