"""Biped stepping rewards for humanoid_legs_5 (Isaac Lab 3, XYZW).

Contact-phase matching is adapted from Unitree's feet_gait, Apache-2.0.
Reference: unitreerobotics/unitree_rl_lab @
4960b84732b0c2ec593dccbfe963fda1bcd7b1e3; see NOTICE.
Changes: separate named foot sensors, normalized matching, upright gating,
and a shared policy clock. Slip is measured at our sole centers.
"""

import math
import torch

GAIT_PERIOD = 0.8  # Initial experimental setting, not a measured optimal cadence.
STANCE_FRACTION = 0.55
COMMAND_THRESHOLD = 0.1
FOOT_NAMES = ("left_foot", "right_foot")
SOLE_CENTERS = ((-0.00042, -0.005, -0.01213), (-0.310851, -0.005, -0.01213))
POSTURE_JOINTS = ("left_hip2_joint", "right_hip2_joint", "left_thigh_joint", "right_thigh_joint")


def _tensor(value):
    return getattr(value, "torch", value)


def moving(env, command_name="base_velocity"):
    return torch.linalg.vector_norm(env.command_manager.get_command(command_name), dim=-1) > COMMAND_THRESHOLD


def phase(env, period=GAIT_PERIOD):
    if period <= 0:
        raise ValueError("Gait period must be positive")
    return (env.episode_length_buf * env.step_dt / period).remainder(1.0)


def gait_phase_observation(env, period=GAIT_PERIOD, command_name="base_velocity"):
    """Same clock as the gait reward; zeros distinguish a standing command."""
    angle = phase(env, period) * (2 * math.pi)
    return torch.stack((angle.sin(), angle.cos()), dim=-1) * moving(env, command_name)[:, None]


def foot_contacts(env):
    """Return explicit [left, right] ordering; fail on a wrongly selected body."""
    contacts = []
    for body_name in FOOT_NAMES:
        sensor = env.scene[f"{body_name}_contact"]
        if list(sensor.body_names) != [body_name]:
            raise ValueError(f"Expected only {body_name}, found {sensor.body_names}")
        force = _tensor(sensor.data.net_forces_w)[:, 0, :]
        contacts.append(torch.linalg.vector_norm(force, dim=-1) > 1.0)
    return torch.stack(contacts, dim=-1)


def alternating_contact(env, period=GAIT_PERIOD, stance_fraction=STANCE_FRACTION, command_name="base_velocity"):
    """Match alternating stance/swing, with short double-support intervals."""
    if not 0.5 <= stance_fraction < 1.0:
        raise ValueError("stance_fraction must be in [0.5, 1)")
    clock = phase(env, period)
    expected = torch.stack((clock < stance_fraction, (clock + 0.5).remainder(1) < stance_fraction), dim=-1)
    match = (foot_contacts(env) == expected).float().mean(dim=-1)
    upright = (-_tensor(env.scene["robot"].data.projected_gravity_b)[:, 2]).clamp(0, 1)
    return match * moving(env, command_name) * upright


def alternating_contact_differential(env, period=GAIT_PERIOD, stance_fraction=STANCE_FRACTION, command_name="base_velocity"):
    """Score gait-specific contact improvement over the 0.5 standing baseline."""
    return 2.0 * alternating_contact(env, period, stance_fraction, command_name) - moving(env, command_name).float()


def feet_slide(env):
    """Nonnegative stance-foot sole-center horizontal speed; use a negative weight.

    Body origins in this CAD asset are offset from the soles. Account for
    angular velocity rather than treating body-origin velocity as sole slip.
    Keep the penalty active when commanded to stand as well as walk.
    """
    robot = env.scene["robot"]
    ids = [robot.body_names.index(name) for name in FOOT_NAMES]
    quat = _tensor(robot.data.body_link_quat_w)[:, ids]
    offsets = quat.new_tensor(SOLE_CENTERS).expand(quat.shape[0], -1, -1)
    uv = 2 * torch.cross(quat[..., :3], offsets, dim=-1)
    world_offsets = offsets + quat[..., 3:] * uv + torch.cross(quat[..., :3], uv, dim=-1)
    velocity = _tensor(robot.data.body_link_lin_vel_w)[:, ids] + torch.cross(
        _tensor(robot.data.body_link_ang_vel_w)[:, ids], world_offsets, dim=-1
    )
    return (torch.linalg.vector_norm(velocity[..., :2], dim=-1) * foot_contacts(env)).sum(dim=-1)


def lateral_sole_spacing(env):
    """Signed left-minus-right sole spacing along heading-frame X (Y forward).

    Use the actual sole centers, not the offset CAD foot-body origins. Signed
    spacing prevents crossed feet from earning the same score as open feet.
    """
    from trajectory_metrics import yaw_xyzw

    robot = env.scene["robot"]
    ids = [robot.body_names.index(name) for name in FOOT_NAMES]
    quat = _tensor(robot.data.body_link_quat_w)[:, ids]
    offsets = quat.new_tensor(SOLE_CENTERS).expand(quat.shape[0], -1, -1)
    uv = 2 * torch.cross(quat[..., :3], offsets, dim=-1)
    centers = _tensor(robot.data.body_link_pos_w)[:, ids] + offsets + quat[..., 3:] * uv + torch.cross(quat[..., :3], uv, dim=-1)
    delta = centers[:, 0] - centers[:, 1]
    yaw = yaw_xyzw(_tensor(robot.data.root_quat_w))
    return yaw.cos() * delta[:, 0] + yaw.sin() * delta[:, 1]


def stance_width_tracking(env, minimum=0.16, maximum=0.20, std=0.05, command_name="base_velocity"):
    """Prefer a moderate spacing band through stance and swing while moving."""
    if not 0 < minimum <= maximum or std <= 0:
        raise ValueError("Expected 0 < minimum <= maximum and std > 0")
    width = lateral_sole_spacing(env)
    error = (minimum - width).clamp_min(0) + (width - maximum).clamp_min(0)
    upright = (-_tensor(env.scene["robot"].data.projected_gravity_b)[:, 2]).clamp(0, 1)
    return torch.exp(-(error / std).square()) * moving(env, command_name) * upright


def yaw_rate_error_l2(env, command_name="base_velocity"):
    """Squared world-vertical turning-rate error, including standing commands."""
    measured = _tensor(env.scene["robot"].data.root_ang_vel_w)[:, 2]
    commanded = env.command_manager.get_command(command_name)[:, 2]
    return (measured - commanded).square()


def _heading_error(env, desired_yaw):
    """Shortest signed yaw error in radians; yaw zero faces body Y along world Y."""
    from trajectory_metrics import yaw_xyzw

    yaw = yaw_xyzw(_tensor(env.scene["robot"].data.root_quat_w))
    delta = desired_yaw - yaw
    return torch.atan2(delta.sin(), delta.cos())


def heading_error_l2(env, desired_yaw=0.0, command_name="base_velocity"):
    """Squared fixed-heading error, including standing; use a negative weight.

    Disable fixed-heading tracking when a nonzero yaw rate is commanded.
    This target is world yaw, not an episode's initial heading.
    """
    straight = env.command_manager.get_command(command_name)[:, 2].abs() < 1e-3
    return _heading_error(env, desired_yaw).square() * straight


def heading_error_observation(env, desired_yaw=0.0, command_name="base_velocity"):
    """[sin(error), cos(error)] for the same fixed yaw used by heading rewards.

    Includes standing commands. Return zeros during commanded turns, when the
    fixed-heading rewards are disabled. The circular encoding avoids a jump
    in observations as the shortest heading error crosses +/-pi.
    """
    error = _heading_error(env, desired_yaw)
    straight = env.command_manager.get_command(command_name)[:, 2].abs() < 1e-3
    return torch.stack((error.sin(), error.cos()), dim=-1) * straight[:, None]


def heading_tracking(env, desired_yaw=0.0, std=0.25, command_name="base_velocity"):
    """Reward upright alignment with a fixed world yaw; std is in radians."""
    if std <= 0:
        raise ValueError("Heading tracking standard deviation must be positive")
    straight = env.command_manager.get_command(command_name)[:, 2].abs() < 1e-3
    upright = (-_tensor(env.scene["robot"].data.projected_gravity_b)[:, 2]).clamp(0, 1)
    return torch.exp(-(_heading_error(env, desired_yaw) / std).square()) * straight * upright


def yaw_rate_tracking(env, std=0.5, command_name="base_velocity"):
    """Bounded tracking reward plus a separate L2 term for large turning errors."""
    if std <= 0:
        raise ValueError("Yaw tracking standard deviation must be positive")
    return torch.exp(-yaw_rate_error_l2(env, command_name) / std**2)
