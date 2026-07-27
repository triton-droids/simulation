"""Reward terms used by the default humanoid locomotion task."""

from .registry import register_reward
import jax
import jax.numpy as jp
from source.utils.math_utils import rotate_vec, quat_inv
from brax import base
from typing import Any
from source.tools import gait


@register_reward("lin_vel")
def _lin_vel(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    """Reward tracking of commanded planar linear velocity.

    Returns:
        Exponential tracking reward for x/y local linear velocity.
    """

    local_lin_vel = self.get_local_linvel(pipeline_state)
    lin_vel_error = jp.sum(jp.square(info["command"][:2] - local_lin_vel[:2]))
    reward = jp.exp(-self.tracking_sigma * lin_vel_error)
    return reward

@register_reward("ang_vel")
def _ang_vel(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    """Reward tracking of commanded yaw angular velocity.

    Returns:
        Exponential tracking reward for local yaw rate.
    """

    local_ang_vel = self.get_gyro(pipeline_state)
    ang_vel_error = jp.square(info["command"][2] - local_ang_vel[-1])
    reward = jp.exp(-self.tracking_sigma * ang_vel_error)
    return reward

# Base-related rewards.

@register_reward("lin_vel_z")
def _lin_vel_z(self, pipeline_state: base.State, action: jax.Array, info: dict[str, Any]) -> jax.Array:
    """Penalize vertical base velocity.

    Returns:
        Squared z-axis global linear velocity.
    """

    global_linvel = self.get_global_linvel(pipeline_state)
    return jp.square(global_linvel[2])

@register_reward("ang_vel_xy")
def _ang_vel_xy(self, pipeline_state: base.State, action: jax.Array, info: dict[str, Any]) -> jax.Array:
    """Penalize roll and pitch angular velocity.

    Returns:
        Sum of squared x/y global angular velocity.
    """

    global_angvel = self.get_global_angvel(pipeline_state)
    return jp.sum(jp.square(global_angvel[:2]))

@register_reward("orientation")
def _orientation(self, pipeline_state: base.State, action: jax.Array, info: dict[str, Any]) -> jax.Array:
    """Penalize torso tilt away from upright.

    Returns:
        Sum of squared horizontal gravity-vector components.
    """

    gravity = self.get_gravity(pipeline_state)
    return jp.sum(jp.square(gravity[:2]))

@register_reward("base_height")
def _base_height(self, pipeline_state: base.State, action: jax.Array, info: dict[str, Any]) -> jax.Array:
    """Penalize deviation from the target base height.

    Returns:
        Squared base-height error.
    """

    base_height = pipeline_state.qpos[2]
    return jp.square(
        base_height - self.base_height_target
    )

# Energy related rewards.
@register_reward("torques")
def _torques(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    """Penalize actuator torque magnitude.

    Returns:
        Sum of absolute actuator forces.
    """

    return jp.sum(jp.abs(pipeline_state.actuator_force))

@register_reward("energy")
def _energy(
    self, 
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    """Penalize mechanical energy proxy.

    Returns:
        Sum of absolute joint velocity times actuator force.
    """

    return jp.sum(jp.abs(pipeline_state.qvel) * jp.abs(pipeline_state.qfrc_actuator))

@register_reward("action_rate")
def _action_rate(
    self, 
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    """Penalize abrupt action changes.

    Returns:
        Squared difference between current and previous actions.
    """

    c1 = jp.sum(jp.square(action - info["last_act"]))
    return c1



# Feet related rewards.
@register_reward("feet_slip")
def _feet_slip(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    """Penalize foot motion while feet are in contact.

    Returns:
        Contact-weighted foot slip cost.
    """

    body_vel = self.get_global_linvel(pipeline_state)
    reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * info["last_contact"])
    return reward

@register_reward("feet_clearance")
def _feet_clearance(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    """Penalize foot height error during swing.

    Returns:
        Velocity-weighted foot clearance cost.
    """

    feet_vel = pipeline_state.cvel[self.feet_body_ids,3:-1]
    vel_norm = jp.sqrt(jp.linalg.norm(feet_vel, axis=-1))
    foot_pos = pipeline_state.site_xpos[self.feet_body_ids]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self.max_foot_height)
    return jp.sum(delta * vel_norm)

@register_reward("feet_height")
def _feet_height(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    """Penalize swing peaks that miss the desired foot height.

    Returns:
        First-contact foot-height error.
    """

    error = (info["swing_peak"] / self.max_foot_height) - 1.0
    return jp.sum(jp.square(error) * info["first_contact"])


@register_reward("feet_phase")
def _feet_phase(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
  ) -> jax.Array:
    """Reward matching the phase-based foot height target.

    Returns:
        Exponential foot-height tracking reward, disabled for near-zero
        commands.
    """

    # Reward for tracking the desired foot height.
    foot_pos = pipeline_state.xpos[self.feet_body_ids]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(info["phase"], swing_height=self.max_foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    cmd_norm = jp.linalg.norm(info["command"])
    reward *= cmd_norm > 0.1  # No reward for zero commands.
    return reward

@register_reward("feet_air_time")
def _feet_air_time(
    self,
    pipeline_state: base.State,
    info: dict[str, Any],
    action: jax.Array,
) -> jax.Array:
    """Reward useful swing air time when walking is commanded.

    Returns:
        Clipped foot air-time reward, disabled for near-zero commands.
    """

    cmd_norm = jp.linalg.norm(info["command"])
    air_time = (info["feet_air_time"] - 0.2) * info["first_contact"]
    air_time = jp.clip(air_time, max=0.5 - 0.2)
    reward = jp.sum(air_time)
    reward *= cmd_norm > 0.1  # No reward for zero commands.
    return reward

# Pose-related rewards.

@register_reward("joint_deviation_hip")
def _joint_deviation_hip(
    self, 
    pipeline_state: base.State, 
    info: dict[str, Any], 
    action: jax.Array
) -> jax.Array:
    """Penalize hip joint deviation during lateral commands.

    Returns:
        Hip deviation cost gated by lateral command magnitude.
    """

    cost = jp.sum(
        jp.abs(pipeline_state.qpos[self.hip_indices] - self.default_pose[self.hip_indices])
    )
    cost *= jp.abs(info["command"][1]) > 0.1
    return cost

@register_reward("joint_deviation_knee")
def _joint_deviation_knee(
    self, 
    pipeline_state: base.State, 
    info: dict[str, Any], 
    action: jax.Array
) -> jax.Array:
    """Penalize knee deviation from the default pose.

    Returns:
        Sum of absolute knee joint position errors.
    """

    return jp.sum(
        jp.abs(
            pipeline_state.qpos[self.knee_indices] - self.default_pose[self.knee_indices]
        )
    )

@register_reward("pose")
def _pose(
    self, 
    pipeline_state: base.State, 
    info, 
    action) -> jax.Array:
    """Penalize weighted full-joint deviation from the default pose.

    Returns:
        Weighted squared joint-position error.
    """

    qpos = pipeline_state.qpos[7:]
    return jp.sum(jp.square(qpos - self.default_pose) * self._weights)

# Other rewards.

@register_reward("stand_still")
def _stand_still(
      self,
      pipeline_state: base.State,
      info: dict[str, Any],
    action: jax.Array,
    ) -> jax.Array:
    """Penalize moving away from the initial pose while commanded to stand.

    Returns:
        Joint-position deviation cost when the command norm is near zero.
    """

    cmd_norm = jp.linalg.norm(info["command"])
    return jp.sum(jp.abs(pipeline_state.qpos - self.init_q)) * (cmd_norm < 0.1)

@register_reward("survival")
def _survival(
    self, 
    pipeline_state: base.State, 
    info: dict[str, Any], 
    action: jax.Array
) -> jax.Array:
    """Calculates a survival reward based on the pipeline state and action taken.

    The reward is negative if the episode is marked as done before reaching the
    specified number of reset steps, encouraging survival until the reset threshold.

    Args:
        pipeline_state (base.State): The current state of the pipeline.
        info (dict[str, Any]): A dictionary containing episode information, including
            whether the episode is done and the current step count.
        action (jax.Array): The action taken at the current step.

    Returns:
        jax.Array: A float32 array representing the survival reward.
    """
    return (info["done"] & (info["step"] < self.reset_steps)).astype(jp.float32)



