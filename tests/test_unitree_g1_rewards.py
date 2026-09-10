"""Sign and monotonic-direction tests for every G1 Phase 4 reward term."""

from __future__ import annotations

import jax.numpy as jp
import numpy as np

from source.config.g1 import G1MJXConfig
from source.locomotion.unitree_g1 import rewards


def test_signed_aggregation_does_not_erase_termination_penalty():
    total = rewards.aggregate_weighted_terms(
        {"tracking": jp.array(1.0), "termination": jp.array(-100.0)},
        control_dt=0.02,
    )

    assert np.isclose(float(total), -1.98)


def test_tracking_rewards_increase_as_explicit_scaled_error_decreases():
    command = jp.array([1.0, -0.25, 0.5])
    sigma = 0.25
    linear_match = rewards.tracking_linear_velocity(command, command, sigma)
    linear_error = rewards.tracking_linear_velocity(command, jp.zeros(3), sigma)
    yaw_match = rewards.tracking_yaw_rate(command, command, sigma)
    yaw_error = rewards.tracking_yaw_rate(command, jp.zeros(3), sigma)
    assert linear_match == 1.0
    assert yaw_match == 1.0
    assert 0.0 < linear_error < linear_match
    assert 0.0 < yaw_error < yaw_match
    assert np.isclose(
        float(rewards.tracking_linear_velocity(jp.array([1.0, 0.0, 0.0]), jp.zeros(3), sigma)),
        np.exp(-1.0 / sigma),
    )


def test_vertical_velocity_cost_is_nonnegative_and_monotonic():
    zero = rewards.vertical_velocity_cost(jp.zeros(3), jp.zeros(3))
    moving = rewards.vertical_velocity_cost(jp.array([0.0, 0.0, 1.0]), jp.array([0.0, 0.0, -2.0]))
    assert zero == 0.0
    assert moving == 5.0


def test_roll_pitch_angular_velocity_cost_is_nonnegative_and_monotonic():
    assert rewards.roll_pitch_angular_velocity_cost(jp.zeros(3)) == 0.0
    assert rewards.roll_pitch_angular_velocity_cost(jp.array([1.0, -2.0, 9.0])) == 5.0


def test_orientation_cost_is_nonnegative_and_monotonic():
    target = jp.array([0.073, 0.0, 1.0])
    assert rewards.upright_orientation_cost(target) == 0.0
    assert rewards.upright_orientation_cost(jp.array([0.5, 0.2, 0.5])) > 0.0


def test_effort_and_power_costs_are_nonnegative_and_monotonic():
    zero = jp.zeros(3)
    force = jp.array([1.0, -2.0, 3.0])
    velocity = jp.array([2.0, 0.5, -1.0])
    assert rewards.actuator_effort_cost(zero) == 0.0
    assert rewards.actuator_effort_cost(force) == 6.0
    assert rewards.mechanical_power_cost(zero, force) == 0.0
    assert rewards.mechanical_power_cost(velocity, force) == 6.0


def test_action_rate_and_acceleration_costs_are_nonnegative_and_monotonic():
    zero = jp.zeros(3)
    values = jp.array([1.0, -2.0, 3.0])
    assert rewards.action_rate_cost(zero, zero) == 0.0
    assert rewards.action_rate_cost(values, zero) == 14.0
    assert rewards.joint_acceleration_cost(zero) == 0.0
    assert rewards.joint_acceleration_cost(values) == 14.0


def test_joint_limit_cost_is_zero_inside_and_grows_outside():
    lower = jp.array([-1.0, -2.0])
    upper = jp.array([1.0, 2.0])
    assert rewards.joint_limit_cost(jp.zeros(2), lower, upper) == 0.0
    assert rewards.joint_limit_cost(jp.array([-1.5, 2.25]), lower, upper) == 0.75


def test_foot_slip_cost_requires_contact_and_grows_with_speed():
    velocity = jp.array([[1.0, 0.0], [0.0, 2.0]])
    assert rewards.foot_slip_cost(velocity, jp.array([False, False])) == 0.0
    one_foot = rewards.foot_slip_cost(velocity, jp.array([True, False]))
    both_feet = rewards.foot_slip_cost(velocity, jp.array([True, True]))
    assert one_foot == 1.0
    assert both_feet == 5.0


def test_contact_and_termination_costs_have_expected_boolean_direction():
    assert rewards.undesired_contact_cost(jp.array(False)) == 0.0
    assert rewards.undesired_contact_cost(jp.array(True)) == 1.0
    assert rewards.termination_cost(jp.array(False)) == 0.0
    assert rewards.termination_cost(jp.array(True)) == 1.0


def test_alive_reward_is_positive_and_constant():
    assert rewards.alive_reward() == 1.0


def test_air_time_reward_increases_for_useful_first_contact():
    command = jp.array([0.5, 0.0, 0.0])
    none = rewards.foot_air_time_reward(jp.array([0.1, 0.1]), jp.array([False, False]), command)
    useful = rewards.foot_air_time_reward(jp.array([0.4, 0.1]), jp.array([True, False]), command)
    standing = rewards.foot_air_time_reward(jp.array([0.4, 0.4]), jp.array([True, True]), jp.zeros(3))
    assert none == 0.0
    assert useful > none
    assert standing == 0.0


def test_phase_reward_is_nonnegative_and_decreases_with_height_error():
    command = jp.array([0.5, 0.0, 0.0])
    phase = jp.array([-jp.pi, 0.0])
    target_height = jp.array([0.0, 0.15])
    matched = rewards.foot_phase_reward(target_height, phase, 0.15, command)
    wrong = rewards.foot_phase_reward(jp.array([0.15, 0.0]), phase, 0.15, command)
    standing = rewards.foot_phase_reward(target_height, phase, 0.15, jp.zeros(3))
    assert matched > wrong >= 0.0
    assert standing == 0.0


def test_standstill_and_pose_costs_are_zero_at_default_and_grow_with_deviation():
    default = jp.array([0.1, -0.2])
    deviated = jp.array([0.4, -0.7])
    assert rewards.stand_still_cost(jp.zeros(3), default, default) == 0.0
    assert rewards.stand_still_cost(jp.zeros(3), deviated, default) > 0.0
    assert rewards.stand_still_cost(jp.array([0.2, 0.0, 0.0]), deviated, default) == 0.0
    assert rewards.pose_cost(default, default) == 0.0
    assert rewards.pose_cost(deviated, default) > 0.0


def test_configured_reward_scales_have_reward_or_cost_signs():
    scales = G1MJXConfig.RewardScales()
    positive = {
        "tracking_lin_vel",
        "tracking_ang_vel",
        "feet_air_time",
        "feet_phase",
        "alive",
    }
    negative = {
        "lin_vel_z",
        "ang_vel_xy",
        "orientation",
        "torques",
        "energy",
        "action_rate",
        "dof_acc",
        "dof_pos_limits",
        "feet_slip",
        "collision",
        "termination",
        "stand_still",
        "pose",
    }
    assert set(scales.__dict__) == positive | negative
    assert all(getattr(scales, name) > 0.0 for name in positive)
    assert all(getattr(scales, name) < 0.0 for name in negative)
