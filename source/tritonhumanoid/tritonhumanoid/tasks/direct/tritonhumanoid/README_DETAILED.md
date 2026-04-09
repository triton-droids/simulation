# Triton Humanoid Locomotion Detailed Walkthrough

This document explains how the local locomotion environment is wired together, focusing on:

- `tritonhumanoid_env.py`
- `tritonhumanoid_env_cfg.py`

It also notes a few important relationships outside those files when they materially affect how these two files behave.

## Scope and Boundaries

- `tritonhumanoid_env.py` defines the local `LocomotionEnv` class, quaternion helpers, and the local ADR helper class.
- `tritonhumanoid_env_cfg.py` defines `EventCfg` and `HumanoidEnvCfg`.
- The outer RL loop itself is owned by Isaac Lab's `DirectRLEnv`. This file provides the callback pieces that `DirectRLEnv` calls.
- The robot articulation details such as actual PD gains, motor latency groups, and default joint angles are not defined in these two files. They come from `HUMANOID_LOCOMOTION_DELAYED_PD_CFG`.

## Actual Registration Path

One easy-to-miss detail:

- `__init__.py` registers the gym environment with:
  - entry point: `tritonhumanoid_env:LocomotionEnv`
  - env config entry point: `tritonhumanoid_env_cfg:HumanoidEnvCfg`
- `tritonhumanoid_env_cfg.py` imports `LocomotionEnv` from `isaaclab_tasks.direct.locomotion.locomotion_env`, but that import is not used by the local registration.

So the active pair in this package is:

- local env class: `tritonhumanoid_env.py`
- local config class: `tritonhumanoid_env_cfg.py`

## High-Level Runtime Flow

Under the standard `DirectRLEnv` callback contract, the local environment behaves like this per control step:

1. `_pre_physics_step(actions)` runs once per policy/control step.
2. `_apply_action()` provides the joint targets used during the decimated physics stepping.
3. After physics advances, `_get_rewards()`, `_get_dones()`, and `_get_observations()` are queried.
4. `_reset_idx(env_ids)` is called for environments that terminated or timed out.

Important derived timing:

- Physics timestep: `sim.dt = 1/200 = 0.005 s`
- Decimation: `4`
- Control timestep: `0.005 * 4 = 0.02 s`
- Physics rate: `200 Hz`
- Control/policy rate: `50 Hz`

This environment now caches robot/sensor state once per control step with `_state_valid`, so rewards, dones, and observations reuse the same state extraction instead of re-reading everything multiple times in the same step.

## `tritonhumanoid_env_cfg.py`

### 1. Base Environment Settings

| Setting | Value | Notes |
| --- | --- | --- |
| `episode_length_s` | `20.0` | Framework-level max episode duration |
| `min_episode_length_s` | `5.0` | Used for randomized episode length |
| `randomize_episode_length` | `True` | Active in local env |
| `decimation` | `4` | 4 physics steps per control action |
| `action_space` | `10` | 10 actuated leg joints |
| `state_space` | `0` | No privileged state vector |

Derived control-step counts:

- Max episode length: `20.0 / 0.02 = 1000` control steps
- Minimum randomized episode length: `5.0 / 0.02 = 250` control steps

### 2. Simulation, Terrain, Scene, Robot, Sensors

| Setting | Value |
| --- | --- |
| `sim.dt` | `1/200 = 0.005 s` |
| `sim.render_interval` | `4` |
| `sim.physics_material.static_friction` | `1.0` |
| `sim.physics_material.dynamic_friction` | `1.0` |
| `terrain.prim_path` | `"/World/ground"` |
| `terrain.terrain_type` | `"plane"` |
| `terrain.collision_group` | `-1` |
| `terrain.physics_material.friction_combine_mode` | `"average"` |
| `terrain.physics_material.restitution_combine_mode` | `"average"` |
| `terrain.physics_material.static_friction` | `1.0` |
| `terrain.physics_material.dynamic_friction` | `1.0` |
| `terrain.physics_material.restitution` | `0.0` |
| `terrain.debug_vis` | `False` |
| `scene.num_envs` | `4096` |
| `scene.env_spacing` | `4.0` |
| `scene.replicate_physics` | `True` |
| `robot` | `HUMANOID_LOCOMOTION_DELAYED_PD_CFG.replace(...)` |
| `contact_sensor.prim_path` | `"/World/envs/env_.*/Robot/.*"` |
| `contact_sensor.history_length` | `3` |
| `contact_sensor.update_period` | `0.005` |
| `contact_sensor.track_air_time` | `True` |

The active terrain is a flat plane. There is also a commented-out terrain generator block for mixed flat/rough heightfields, but it is inactive.

### 3. Frame Transformer / Tracking Site

The `ee_site` frame transformer is used to create a single target frame named `"top"`:

- Source frame prim path: `"/World/envs/env_.*/Robot/world"`
- Target parent prim path: `"/World/envs/env_.*/Robot/torso"`
- Target name: `"top"`
- Offset position: `(-0.155, -0.016, 0.765)`
- Offset rotation: `(1.0, 0.0, 0.0, 0.0)` in `(w, x, y, z)`

This matters because the local env uses `track_pos_w` from this site:

- reward/done height logic is based on the `"top"` site height, not directly on the root or COM height

### 4. Actions

Base action config:

| Setting | Value |
| --- | --- |
| `action_scale` | `0.8` |
| `action_space` | `10` |

Per-joint multipliers:

| Joint | Multiplier |
| --- | --- |
| `left_hip2_joint` | `0.50` |
| `right_hip2_joint` | `0.50` |
| `left_thigh_joint` | `0.3` |
| `right_thigh_joint` | `0.3` |
| All other actuated joints | `1.0` |

The effective commanded joint target is:

```text
q_des = default_actuated_pos
      + action_scale
      * per_joint_scale
      * action
      * motor_strength_mult
```

Then `q_des` is clamped to the soft joint limits.

Ignoring ADR motor-strength scaling, the nominal max offset magnitudes from `action_scale * per_joint_scale` are:

| Joint group | Effective max offset |
| --- | --- |
| `hip1`, `knee`, `ankle` | `0.8 rad` |
| `hip2` | `0.4 rad` |
| `thigh` | `0.24 rad` |

The cfg file also carries a comment block with the documented actuated joint limits:

| Joint | Documented limit range |
| --- | --- |
| `left_hip1_joint` | `[-1.57, 1.57]` |
| `left_hip2_joint` | `[-1.57, 0.436332]` |
| `left_thigh_joint` | `[-0.785398, 0.785398]` |
| `left_knee_joint` | `[-2.0944, 0]` |
| `left_ankle_joint` | `[-0.6, 0.6]` |
| `right_hip1_joint` | `[-1.57, 1.57]` |
| `right_hip2_joint` | `[-0.436332, 1.57]` |
| `right_thigh_joint` | `[-0.785398, 0.785398]` |
| `right_knee_joint` | `[-2.0944, 0]` |
| `right_ankle_joint` | `[-0.6, 0.6]` |

Note: the inline comment says "increase to 0.5 after initial learning", but the current value is already `0.8`, so that comment is stale/inconsistent.

### 5. Observation Configuration

Scales and fixed sensor noise:

| Setting | Value |
| --- | --- |
| `ang_vel_scale` | `0.25` |
| `dof_vel_scale` | `0.1` |
| `joint_pos_obs_noise_std_rad` | `0.005716356937792565` |
| `joint_vel_obs_noise_std_rad_s` | `0.13718300792805946` |
| `command_yaw_offset` | `-pi/2` |

Observation dimensions:

- Base single-frame observation:
  - `3` command-frame linear velocity
  - `3` command-frame angular velocity
  - `3` command-frame up vector
  - `3` commands
  - `10` scaled actuated joint positions
  - `10` scaled actuated joint velocities
  - `10` previous actions
  - total: `42`
- If `use_phase_obs=True`, append `sin(phase)` and `cos(phase)`:
  - total: `44`
- `obs_stack_frames = 3`
- Default actual observation size at runtime:
  - `use_phase_obs=False` by default
  - `42 * 3 = 126`

The cfg comment block also records the intended single-frame slice layout:

| Slice | Index range |
| --- | --- |
| `lin_vel_cmd` | `[0, 3)` |
| `ang_vel_cmd_scaled` | `[3, 6)` |
| `up_cmd` | `[6, 9)` |
| `commands` | `[9, 12)` |
| `act_pos_scaled` | `[12, 22)` |
| `act_vel_scaled` | `[22, 32)` |
| `prev_actions` | `[32, 42)` |

The file also defines:

```python
obs_term_cfg = {
    "projected_gravity": {"noise": 0.01, "scale": 1.0, "clip": 1.0},
    "base_ang_vel": {"noise": 0.05, "scale": 0.25, "clip": 5.0},
    "base_lin_vel": {"noise": 0.05, "scale": 1.0, "clip": 5.0},
    "dof_pos_delta": {"noise": 0.01, "scale": 1.0, "clip": 2.0},
    "dof_vel": {"noise": 0.1, "scale": 0.1, "clip": 5.0},
    "prev_actions": {"noise": 0.0, "scale": 1.0, "clip": 1.0},
    "commands": {"noise": 0.0, "scale": 1.0, "clip": 1.0},
    "phase": {"noise": 0.0, "scale": 1.0, "clip": 1.0},
}
```

But the local env does not actually use `obs_term_cfg`; observations are built manually in `tritonhumanoid_env.py`.

### 6. Termination Thresholds

| Setting | Value |
| --- | --- |
| `termination_height` | `0.4` |
| `upright_threshold` | `0.5` |

In the local env:

- the height check is `track_pos_w[:, 2] < 0.4`
- the tilt check is `up_b[:, 2] < 0.5`

### 7. Reward Configuration

Active scalar weights:

| Setting | Value |
| --- | --- |
| `lin_vel_reward_scale` | `3.0` |
| `yaw_rate_reward_scale` | `0.6` |
| `upright_reward_scale` | `0.6` |
| `alive_reward` | `0.05` |
| `action_cost_scale` | `0.005` |
| `joint_limit_cost_scale` | `0.2` |
| `death_cost` | `-1.0` |
| `lin_vel_sigma` | `0.10` |
| `yaw_rate_sigma` | `0.5` |
| `lin_vel_z_cost_scale` | `0.08` |
| `ang_vel_xy_cost_scale` | `0.03` |
| `flat_ori_cost_scale` | `0.25` |
| `command_speed_threshold` | `0.2` |
| `standstill_speed_threshold` | `0.15` |
| `standstill_penalty_scale` | `0.8` |
| `speed_shortfall_cost_scale` | `1.0` |
| `air_time_command_speed_threshold` | `0.1` |
| `action_rate_cost_scale` | `0.01` |
| `dof_vel_cost_scale` | `0.0001` |
| `dof_vel_delta_cost_scale` | `0.01` |
| `energy_cost_scale` | `0.002` |
| `stand_cmd_lin_thresh` | `0.08` |
| `stand_cmd_yaw_thresh` | `0.10` |
| `stand_pose_reward_scale` | `0.8` |
| `stand_upright_reward_scale` | `0.4` |
| `stand_pose_sigma` | `0.02` |
| `stand_vel_cost_scale` | `0.8` |
| `stand_action_cost_scale` | `0.02` |
| `stand_hip2_cost_scale` | `1.0` |
| `freeze_phase_when_standing` | `True` |
| `stand_phase_lin_threshold` | `0.08` |
| `stand_phase_yaw_threshold` | `0.10` |
| `stand_phase_value` | `pi` |
| `pose_return_scale` | `0.3` |
| `pose_return_upright_threshold` | `0.7` |
| `symmetry_cost_scale` | `0.1` |
| `thigh_pose_cost_scale` | `0.1` |
| `anti_phase_reward_scale` | `0.15` |
| `anti_phase_sigma` | `0.25` |
| `anti_phase_pos_gain` | `2.5` |
| `anti_phase_min_speed` | `0.15` |
| `gait_actual_speed_thresh` | `0.08` |
| `gait_upright_thresh` | `0.70` |
| `yaw_cmd_reward_thresh` | `0.15` |
| `walk_cmd_speed_thresh` | `0.15` |
| `walk_hip2_cost_scale` | `0.12` |
| `contact_phase_reward_scale` | `0.08` |
| `contact_phase_sigma` | `0.35` |
| `contact_phase_min_speed` | `0.15` |
| `gate_smoothness_to_swing` | `False` |
| `swing_gate_alpha` | `1.0` |
| `foot_body_regex` | `"left_foot|right_foot"` |
| `foot_contact_force_thresh` | `30.0 N` |
| `min_air_time` | `0.2 s` |
| `feet_air_time_reward_scale` | `0.2` |
| `air_time_symmetry_cost_scale` | `0.35` |
| `foot_slip_cost_scale` | `0.02` |
| `undesired_contact_force_thresh` | `80.0 N` |
| `undesired_contact_cost_scale` | `0.1` |
| `no_fly_cost_scale` | `0.20` |
| `touchdown_cost_scale` | `0.16` |
| `touchdown_vel_ref` | `0.6` |
| `touchdown_min_cmd_speed` | `0.15` |
| `touchdown_force_cost_scale` | `0.0` |
| `touchdown_force_thresh` | `120.0 N` |

Many of these are defined but not all of them are currently active in the final reward equation. The reward section below explains exactly which ones are actually used.

### 8. Phase Settings

| Setting | Value |
| --- | --- |
| `use_phase_obs` | `False` |
| `gait_period_s` | `1.0` |
| `gait_period_randomization_width` | `0.0` |
| `randomize_phase` | `True` |
| `phase_offset_default` | `(0.0, pi)` |

Important details:

- even with `use_phase_obs=False`, phase is still used by the anti-phase and contact-phase reward terms
- `phase_offset_default` is defined but not used in the local env
- `randomize_phase=True` means reset uses a uniform random phase in `[0, 2*pi]`
- `gait_period_randomization_width` is defined but unused

### 9. Command Curriculum and Command Sampling

| Setting | Value |
| --- | --- |
| `use_curriculum` | `True` |
| `curriculum_stage1_steps_per_env` | `2500` |
| `curriculum_stage2_steps_per_env` | `5000` |
| `curriculum_stage0_vx_min` | `0.3` |
| `curriculum_stage0_vx_max` | `1.0` |
| `zero_command_probability` | `0.0` |
| `turn_in_place_probability` | `0.02` |
| `turn_in_place_yaw_min` | `0.3` |
| `turn_in_place_yaw_max` | `1.0` |
| `turn_in_place_min_stage` | `1` |
| `command_resample_interval_s` | `10.0` |
| `command_resample_interval_steps` | `0` |
| `lin_vel_x_range` | `(-1.0, 1.0)` |
| `lin_vel_y_range` | `(-0.5, 0.5)` |
| `ang_vel_yaw_range` | `(-1.0, 1.0)` |
| `stand_prob` | `0.2` |

Derived values:

- `command_resample_interval_steps` is `0`, so the env uses seconds
- `10.0 / 0.02 = 500` control steps between resamples
- stage 1 begins after `2500` control steps per env, which is `50 s` simulated time
- stage 2 begins after `5000` control steps per env, which is `100 s` simulated time

Important details:

- `lin_vel_x_range`, `lin_vel_y_range`, `ang_vel_yaw_range`, and `stand_prob` are defined but not used by the local `_sample_commands()` implementation
- actual command sampling is hard-coded inside `_sample_commands()`

### 10. Reset Noise and Disturbance Settings

Config fields:

| Setting | Value |
| --- | --- |
| `reset_joint_pos_noise` | `0.05` |
| `reset_joint_vel_noise` | `0.2` |
| `push_force_range` | `(0.2, 0.8)` |
| `push_strong` | `False` |
| `strong_min_push_interval_s` | `1.0` |
| `strong_max_push_interval_s` | `3.0` |
| `min_push_interval_s` | `5.0` |
| `max_push_interval_s` | `10.0` |
| `push_z_fraction` | `0.25` |
| `push_angvel_scale` | `0.8` |

Derived values at the current `0.02 s` control timestep:

- push interval min: `5.0 / 0.02 = 250` control steps
- push interval max: `10.0 / 0.02 = 500` control steps

Important details:

- `reset_joint_pos_noise` and `reset_joint_vel_noise` are defined but not used by the local reset code
- reset noise is actually driven by ADR `robot_spawn`
- `push_strong`, `strong_min_push_interval_s`, and `strong_max_push_interval_s` are defined but unused

### 11. Base Reset-Time Event Randomization (`EventCfg`)

All of these event terms run in mode `"reset"` with `min_step_count_between_reset=0`.

| Event | Target | Base Range / Behavior |
| --- | --- | --- |
| `robot_physics_material` | All rigid bodies | static friction `(0.6, 1.2)`, dynamic friction `(0.5, 1.1)`, restitution `(0.0, 0.2)`, `num_buckets=128` |
| `robot_joint_stiffness_and_damping` | All joints | uniform scale, stiffness `(1.0, 1.0)`, damping `(1.0, 1.0)` |
| `gravity` | Physics scene | gravity scale `(0.9, 1.1)` |
| `robot_scale_mass` | All rigid bodies | mass scale `(1.0, 1.2)` |
| `robot_com_offset` | All rigid bodies | add COM offset `(0.0, 0.005)` if function exists |
| `robot_inertia_scale` | All rigid bodies | inertia scale `(0.85, 1.15)` if function exists |
| `robot_joint_friction_armature` | All joints | friction scale `(0.5, 1.5)`, armature scale `(0.5, 1.5)` |

### 12. ADR Global Settings

| Setting | Value |
| --- | --- |
| `enable_adr` | `True` |
| `num_adr_increments` | `100` |
| `starting_adr_increments` | `0` |
| `adr_update_interval_steps` | `500` |
| `adr_success_rate_to_increase` | `0.85` |
| `adr_success_rate_to_decrease` | `0.2` |
| `adr_min_steps_before_decrease` | `10000` |
| `adr_ema_factor` | `0.05` |
| `adr_warmup_steps` | `20000` |
| `adr_min_stage` | `2` |
| `adr_track_err_lin_increase_threshold` | `0.25` |
| `adr_track_err_lin_decrease_threshold` | `0.45` |
| `adr_track_err_yaw_increase_threshold` | `0.25` |
| `adr_track_err_yaw_decrease_threshold` | `0.6` |
| `adr_command_scale_min_stage` | `2` |
| `adr_push_start_difficulty` | `0.3` |
| `adr_push_ramp_difficulty` | `0.3` |
| `adr_micro_wrench_start_difficulty` | `0.4` |
| `adr_micro_wrench_ramp_difficulty` | `0.3` |
| `adr_print_every_update` | `True` |
| `adr_debug_print` | `True` |
| `adr_debug_print_every_steps` | `2000` |

Derived details:

- ADR difficulty is `current_increments / 100`
- push scaling is `0` below difficulty `0.3`, ramps linearly to `1` by `0.6`
- micro-wrench scaling is `0` below difficulty `0.4`, ramps linearly to `1` by `0.7`
- stage requirement means ADR will not update until curriculum reaches stage 2
- with `control_dt = 0.02`, `adr_warmup_steps = 20000` is `400 s` simulated time

### 13. ADR Event-Term Max Ranges

These are the "max difficulty" target values that the local ADR code lerps toward.

| Event | Base | ADR Max | Effect |
| --- | --- | --- | --- |
| `robot_physics_material.static_friction_range` | `(0.6, 1.2)` | `(0.4, 1.5)` | Widens |
| `robot_physics_material.dynamic_friction_range` | `(0.5, 1.1)` | `(0.3, 1.4)` | Widens |
| `robot_physics_material.restitution_range` | `(0.0, 0.2)` | `(0.0, 0.4)` | Widens |
| `robot_joint_stiffness_and_damping.stiffness_distribution_params` | `(1.0, 1.0)` | `(1.0, 1.0)` | No actual ADR ramp |
| `robot_joint_stiffness_and_damping.damping_distribution_params` | `(1.0, 1.0)` | `(1.0, 1.0)` | No actual ADR ramp |
| `gravity.gravity_distribution_params` | `(0.9, 1.1)` | `(0.8, 1.2)` | Widens |
| `robot_scale_mass.mass_distribution_params` | `(1.0, 1.2)` | `(0.7, 1.3)` | Widens |
| `robot_joint_friction_armature.friction_distribution_params` | `(0.5, 1.5)` | `(0.5, 1.5)` | No actual ADR ramp |
| `robot_joint_friction_armature.armature_distribution_params` | `(0.5, 1.5)` | `(0.5, 1.5)` | No actual ADR ramp |
| `robot_com_offset.com_distribution_params` | `(0.0, 0.005)` | `(0.0, 0.005)` | Optional, but no actual ADR ramp |
| `robot_inertia_scale.inertia_distribution_params` | `(0.85, 1.15)` | `(0.85, 1.15)` | Optional, but no actual ADR ramp |

### 14. ADR Custom Randomization Ranges

| Group | Key | Range |
| --- | --- | --- |
| `push` | `push_force_range` | `((0.2, 0.8), (0.5, 2.0))` |
| `robot_spawn` | `joint_pos_noise` | `(0.0, 0.06)` |
| `robot_spawn` | `joint_vel_noise` | `(0.0, 0.20)` |
| `sensor_extrinsics` | `imu_mount_deg` | `(0.0, 5.0)` |
| `action_noise` | `std` | `(0.0, 0.03)` |
| `obs_noise` | `gravity_std` | `(0.0, 0.03)` |
| `obs_noise` | `gyro_std` | `(0.0, 0.05)` |
| `obs_noise` | `joint_pos_std` | `(0.0, 0.0)` |
| `obs_noise` | `joint_vel_std` | `(0.0, 0.0)` |
| `obs_noise` | `joint_torque_std` | `(0.0, 0.20)` |
| `motor_strength` | `per_joint_mult_range` | `(0.0, 0.3)` |
| `imu_bias` | `gravity_bias_range` | `(0.0, 0.1)` |
| `imu_bias` | `gyro_bias_range` | `(0.0, 0.2)` |
| `micro_wrench` | `lin_acc_std` | `(0.0, 0.01)` |
| `micro_wrench` | `ang_acc_std` | `(0.0, 0.02)` |
| `micro_wrench` | `rho` | `(0.0, 0.6)` |
| `micro_wrench` | `max_lin_acc` | `(0.0, 0.03)` |
| `micro_wrench` | `max_ang_acc` | `(0.0, 0.05)` |
| `command_scale` | `scale` | `(1.0, 1.5)` |

Important details:

- `joint_pos_std` and `joint_vel_std` ADR ranges are both zero, so they currently do nothing
- `joint_torque_std` is defined, but the local observation vector does not include torques, so it is also unused
- there is no ADR `latency` group in this file, even though `LocomotionADR.print_params()` has a code path for it

### 15. Debug / Visualization Settings

| Setting | Value |
| --- | --- |
| `obs_max_latency` | `0` |
| `debug_vel_vis` | `False` |
| `debug_vel_vis_all_envs` | `True` |
| `debug_vel_vis_all_envs_max_envs` | `64` |
| `debug_env_id` | `0` |
| `vel_vis_scale` | `0.5` |
| `vel_vis_height` | `0.25` |
| `vel_vis_every_n` | `2` |
| `debug_obs_print` | `False` |
| `debug_obs_print_steps` | `5` |
| `debug_obs_print_every` | `1` |
| `debug_obs_print_env` | `0` |
| `debug_print_orderings` | `False` |
| `enable_reward_logging` | `False` |

Important details:

- `obs_max_latency=0` means latency is disabled by default
- `debug_vel_vis_all_envs=True` is irrelevant unless `debug_vel_vis` is also enabled
- `vel_vis_scale` and `vel_vis_every_n` are defined but unused by the local marker code

## `tritonhumanoid_env.py`

### 1. Quaternion Helpers

The file defines its own quaternion helpers:

- `quat_rotate_inverse(q, v)`
- `quat_rotate(q, v)`

Both assume quaternion order `(w, x, y, z)` and batched shapes:

- `q`: `[N, 4]`
- `v`: `[N, 3]`

`quat_rotate_inverse()` negates the vector part of the quaternion and uses the standard cross-product formulation for inverse rotation.

### 2. ADR Helper Class: `LocomotionADR`

`LocomotionADR` is a thin controller around Isaac Lab's event manager.

It does three things:

1. Snapshots the event-manager term parameters at difficulty 0.
2. Maintains a scalar difficulty:
   - `difficulty = current_increments / max_increments`
3. Provides lerped event and custom values as difficulty changes.

Important implementation details:

- `set_num_increments()` clamps increments to `[0, max]`
- `apply_event_ranges()` linearly interpolates each configured event-term parameter from the snapshot to the configured ADR max
- `get_custom()` supports:
  - scalar pairs like `(min, max)`
  - nested endpoint pairs like `((lo0, hi0), (lo1, hi1))`
- `print_params()` prints selected event/custom values for debugging

### 3. Constructor: What Gets Built

The constructor does a lot of environment-specific setup.

#### 3.1 Observation metadata

It recomputes observation dimensions from the incoming cfg:

- single-frame base dim: `3 + 3 + 3 + 3 + action_space * 3`
- with `action_space = 10`, that is `42`
- if phase obs is enabled, add `2`
- then multiply by `obs_stack_frames`

#### 3.2 State cache

The local env uses:

- `_state_valid = False`

This is invalidated at the start of each control step and after resets, then refreshed lazily by `_update_state()`.

#### 3.3 Actuated joint discovery

Actuated joint regex:

```text
left_hip1_joint|left_hip2_joint|left_thigh_joint|left_knee_joint|left_ankle_joint|
right_hip1_joint|right_hip2_joint|right_thigh_joint|right_knee_joint|right_ankle_joint
```

The file also prints:

- the discovered joint indices
- the discovered default actuated pose vector

The comment block at the bottom of `tritonhumanoid_env_cfg.py` records the expected action/joint order as:

1. `left_hip1_joint`
2. `right_hip1_joint`
3. `left_hip2_joint`
4. `right_hip2_joint`
5. `left_thigh_joint`
6. `right_thigh_joint`
7. `left_knee_joint`
8. `right_knee_joint`
9. `left_ankle_joint`
10. `right_ankle_joint`

#### 3.4 Joint-group indices used by rewards

The constructor precomputes:

- `_sym_left_action_ids` / `_sym_right_action_ids` for left-right symmetry
- `_thigh_action_ids` for thigh neutral-pose penalty
- `_hip1_left_action_id` / `_hip1_right_action_id` for anti-phase gait reward
- `_hip2_action_ids` for stand/walk hip2 posture terms

#### 3.5 IMU body and tracking site

- `imu_ids = self.robot.find_bodies("world")`
- `_imu_body_idx` is the first `"world"` body found
- `_top_frame_idx` is the index of the frame-transformer target named `"top"`

#### 3.6 Buffers and cached constants

The constructor creates:

- `_world_up`: `[N, 3]` repeated world-up vector
- `default_actuated_pos`
- `actuated_lower`
- `actuated_upper`
- `actions`
- `prev_actions`
- `q_des`
- `prev_act_vel`
- `commands`
- `phase_offset`

Derived control timing:

- `_control_dt = self.cfg.sim.dt * self.cfg.decimation = 0.02`

Command resampling:

- if `command_resample_interval_steps > 0`, use that
- else compute from seconds
- with the default config, `_cmd_resample_interval_steps = 500`

#### 3.7 Command-frame alignment

The env supports a yaw offset between the robot body frame and the command frame:

- `_command_yaw_offset = -pi/2`
- precomputes `_cmd_yaw_cos`, `_cmd_yaw_sin`
- also precomputes inverse rotation cos/sin
- `_use_cmd_yaw_offset = True` because `abs(-pi/2) > 1e-6`

Practical meaning:

- the env rotates measured body-frame vectors into a command-aligned frame
- commands are sampled and compared in this command frame
- this is how the code reconciles a robot model whose forward axis is effectively `+Y` with commands expressed as `+X` forward

#### 3.8 COM tracking

The env reads:

- `_track_com_linear = getattr(self.cfg, "track_com_linear_velocity", True)`
- `_refresh_runtime_masses_on_reset = getattr(self.cfg, "refresh_runtime_masses_on_reset", False)`

These fields are not declared in `HumanoidEnvCfg`, so the defaults are:

- `_track_com_linear = True`
- `_refresh_runtime_masses_on_reset = False`

The env immediately calls `_cache_body_masses()`.

#### 3.9 Curriculum, debug, stacking, ADR, and randomization buffers

The constructor also initializes:

- command curriculum counters and stage names
- marker visualization state if enabled
- observation stack buffer of shape `[num_envs, obs_single_dim, obs_stack_frames]` if stacking is enabled
- observation-debug slice metadata
- randomized episode lengths buffer
- ADR counters and EMAs
- optional observation-latency buffer
- `motor_strength_mult`
- IMU bias and mount-misalignment buffers
- micro-disturbance buffers
- current ADR custom params
- push timers
- `extras["log"]`

#### 3.10 Push-timer derived values

The constructor computes:

- `self.dt = self.cfg.sim.dt * self.cfg.decimation = 0.02`
- `_push_interval_steps_min = 250`
- `_push_interval_steps_max = 500`

Then it samples `next_push_steps` per environment uniformly in `[250, 500]`.

#### 3.11 ADR startup

Because `enable_adr=True`, the constructor creates `self.adr`, sets:

- `starting_adr_increments = 0`

So startup difficulty is:

- `0 / 100 = 0.0`

At startup the effective ADR custom parameters are therefore:

- push force range = `(0.2, 0.8)` but then scaled by push ramp
- because push ramp starts at difficulty `0.3`, startup push scale is `0.0`
- startup action noise = `0.0`
- startup command scale = `1.0`
- startup micro-wrench scale = `0.0`

That means:

- base push range exists in config, but pushes are effectively off at ADR difficulty 0 because of the push ramp
- micro disturbances are also off at startup

### 4. Scene Setup: `_setup_scene()`

This method:

1. Creates the robot articulation from `self.cfg.robot`.
2. Creates:
   - `ContactSensor(self.cfg.contact_sensor)`
   - `FrameTransformer(self.cfg.ee_site)`
3. Adds them to `self.scene`.
4. Instantiates the terrain from `self.cfg.terrain`.
5. Clones environments with `copy_from_source=False`.
6. On CPU only, explicitly filters collisions against the terrain prim.
7. Adds a dome light:
   - intensity: `2000.0`
   - color: `(0.75, 0.75, 0.75)`

### 5. Mass Caching: `_cache_body_masses()`

Mass caching works like this:

- Prefer `self.robot.data.default_mass` if available
- Else call `self.robot.root_physx_view.get_masses()`
- Expand to `[num_envs, num_bodies]` if necessary
- Move to sim device
- Cache:
  - `_body_mass`
  - `_body_mass_sum`

Important caveat:

- reset-time mass randomization is active through `EventCfg`
- but `_refresh_runtime_masses_on_reset` defaults to `False`
- so COM weighting may use stale startup masses unless the underlying runtime mass tensor already reflects the randomization or the flag is provided externally

### 6. ADR Custom Params: `_update_adr_custom_params()`

If ADR is disabled:

- push range comes directly from `cfg.push_force_range`
- `_action_noise_std = 0`
- all `_obs_noise[*] = 0`
- `_command_scale = 1`
- `_push_scale = 1`
- `_micro_wrench_scale = 1`

If ADR is enabled:

1. Pull `push_force_range` from ADR custom config.
2. Compute difficulty.
3. Apply push ramp:
   - start at difficulty `0.3`
   - linearly ramp for `0.3`
   - full strength by difficulty `0.6`
4. Pull `action_noise.std`.
5. Pull configured observation-noise values.
6. Pull command scale if present.
7. Apply micro-wrench ramp:
   - start at difficulty `0.4`
   - linearly ramp for `0.3`
   - full strength by difficulty `0.7`

### 7. Per-Episode ADR Sampling: `_sample_custom_dr_for_resets()`

For each reset environment:

- if ADR is off:
  - `motor_strength_mult = 1`
  - `obs_latency_steps = 0`
  - `imu_bias_gravity = 0`
  - `imu_bias_gyro = 0`
  - `imu_mount_axis = [1, 0, 0]`
  - `imu_mount_ang = 0`

- if ADR is on:
  - sample per-joint motor strength multiplier from `[1-r, 1+r]`
  - set `obs_latency_steps = 0`
  - sample gravity bias uniformly in `[-b_grav, b_grav]`
  - sample gyro bias uniformly in `[-b_gyro, b_gyro]`
  - sample IMU mount axis from normalized Gaussian vectors
  - sample IMU mount angle uniformly in `[-max_rad, max_rad]`

At max ADR difficulty:

- `motor_strength_mult` can reach `[0.7, 1.3]`
- gravity bias can reach `[-0.1, 0.1]`
- gyro bias can reach `[-0.2, 0.2] rad/s`
- mount angle can reach `[-5 deg, 5 deg]`

Important detail:

- `obs_latency_steps` is maintained, but `obs_max_latency=0` by default, so latency is effectively disabled in the current config

### 8. Push Disturbances: `_maybe_apply_pushes()`

This callback treats pushes as direct velocity kicks, not forces.

Logic:

1. If current max push delta-v is `<= 0`, return.
2. Increment all `push_counters`.
3. Select environments where `push_counters >= next_push_steps`.
4. For selected environments:
   - sample random direction `dirs ~ N(0,1)`
   - scale the `z` component by `push_z_fraction = 0.25`
   - renormalize direction
   - sample magnitude uniformly in `[current_push_min, current_push_max]`
   - add `delta_v` to root linear velocity
   - sample angular kick direction
   - scale angular kick by `push_angvel_scale * magnitude`
   - write root velocity directly back to sim
5. Reset push counter and resample the next push step in `[250, 500]`

Important detail:

- because push scale ramps with ADR, pushes are disabled at low ADR difficulty even though the base cfg push range is non-zero

### 9. Micro Disturbances: `_apply_micro_disturbance()`

This implements an OU-like correlated disturbance:

```text
micro = rho * micro + (1 - rho) * randn * std
micro = clamp(micro, -max, max)
root_vel += micro * dt
```

Separate processes are maintained for:

- linear acceleration
- angular acceleration

Important details:

- only active if ADR exists
- only active if `_micro_wrench_scale > 0`
- with current config this does not start ramping until difficulty `0.4`

### 10. ADR Update Logic: `_maybe_update_adr()`

ADR updates use:

- `success_rate_ema`
- `lin_err_ema`
- `yaw_err_ema`

Update gates:

- do nothing before `adr_warmup_steps = 20000`
- do nothing before curriculum stage `2`
- do nothing until `adr_update_interval_steps = 500` has elapsed since last update

Increase condition:

- `success_rate_ema >= 0.85`
- `lin_err_ema <= 0.25`
- `yaw_err_ema <= 0.25`

Decrease condition:

- `success_rate_ema <= 0.2`, or
- `lin_err_ema >= 0.45`, or
- `yaw_err_ema >= 0.6`

Additional decrease gate:

- at least `10000` control steps since last decrease

Each update changes ADR by exactly one increment.

### 11. Action Preprocessing: `_pre_physics_step(actions)`

This is the first local callback on each control step.

It does the following in order:

1. `self._global_policy_step += 1`
2. invalidate the cached state
3. clamp incoming actions to `[-1, 1]`
4. if `self._action_noise_std > 0`, add Gaussian noise and clamp again
5. copy current `self.actions` into `self.prev_actions`
6. store the new action into `self.actions`
7. increment `_global_env_steps` by `num_envs`
8. update command curriculum
9. resample commands for envs whose `episode_length_buf % _cmd_resample_interval_steps == 0`
10. maybe apply pushes
11. apply micro disturbances
12. optionally call `_visualize_markers()`

Important details:

- there is no action latency buffer here
- there is no first-order action filter here
- those features exist in `standing_env.py`, but not in this locomotion env
- `_visualize_markers()` does not refresh state itself; when called here it will use the last available cached state, not a freshly updated current-state snapshot

### 12. Action Application: `_apply_action()`

This converts the processed action into joint position targets:

```text
pos_offsets =
    action_scale
  * per_joint_scale
  * actions
  * motor_strength_mult

q_des = default_actuated_pos + pos_offsets
q_des = clamp(q_des, actuated_lower, actuated_upper)
```

Then it writes:

- position target = `q_des`
- velocity target = zero
- effort target = zero

Important details:

- actions are position offsets around the robot's default actuated pose
- `motor_strength_mult` scales the command itself, not the actuator model
- the file keeps `q_des` as a buffer but does not otherwise reuse it

### 13. State Cache: `_update_state()`

This function refreshes the main sensor/state tensors exactly once per control step.

If `_state_valid` is already true, it returns immediately.

Otherwise it computes:

#### 13.1 IMU state from body `"world"`

- `imu_pos_w`
- `imu_quat_w`
- `imu_lin_vel_w`
- `imu_ang_vel_w`
- `imu_lin_vel_b = quat_rotate_inverse(imu_quat_w, imu_lin_vel_w)`
- `imu_ang_vel_b = quat_rotate_inverse(imu_quat_w, imu_ang_vel_w)`

If the command yaw offset is enabled:

- rotate the XY components of IMU linear/angular velocity into the command frame
- store:
  - `imu_lin_vel_cmd`
  - `imu_ang_vel_cmd`

#### 13.2 Up vector

- `up_b = quat_rotate_inverse(imu_quat_w, world_up)`
- optionally rotate `up_b` into command frame to get `up_cmd`

#### 13.3 Tracking site

- `track_pos_w = self.scene["ee_site"].data.target_pos_w[:, top_frame_idx]`

#### 13.4 COM state

The env prefers:

- `body_com_pos_w` if available, else `body_pos_w`
- `body_com_lin_vel_w` if available, else `body_lin_vel_w`

If `_track_com_linear` is true:

- compute mass-weighted COM position and COM linear velocity
- rotate COM linear velocity into body frame
- optionally rotate into command frame
- store:
  - `com_pos_w`
  - `com_lin_vel_w`
  - `com_lin_vel_b`
  - `com_lin_vel_cmd`

Else:

- `com_lin_vel_cmd = imu_lin_vel_cmd`

Angular velocity for reward tracking always uses:

- `com_ang_vel_cmd = imu_ang_vel_cmd`

#### 13.5 Joint state

- `dof_pos = self.robot.data.joint_pos`
- `dof_vel = self.robot.data.joint_vel`
- `act_pos = dof_pos[:, joint_dof_idx]`
- `act_vel = dof_vel[:, joint_dof_idx]`
- `act_pos_scaled = 2 * (act_pos - lower) / (upper - lower + 1e-6) - 1`

Then `_state_valid = True`.

Important observation/reward mismatch:

- observation uses `imu_lin_vel_cmd`
- reward tracking uses `com_lin_vel_cmd` when `_track_com_linear=True`

### 14. Observation Builder: `_compute_single_observation()`

The observation is built from cached state plus simulated sensor corruption.

Starting tensors:

- `up_cmd`
- `ang_vel_cmd`
- `act_pos`
- `act_vel`

Then it applies:

#### 14.1 IMU bias

- `up_cmd += imu_bias_gravity`
- `ang_vel_cmd += imu_bias_gyro`

#### 14.2 IMU mount misalignment

Small-angle approximation:

```text
theta = imu_mount_axis * imu_mount_ang
x' = x + cross(theta, x)
```

Applied to:

- `up_cmd`
- `ang_vel_cmd`

#### 14.3 ADR observation noise

If ADR exists:

- gravity noise if `gravity_std > 0`
- gyro noise if `gyro_std > 0`
- joint position noise if `joint_pos_std > 0`
- joint velocity noise if `joint_vel_std > 0`

#### 14.4 Always-on observation noise

Regardless of ADR:

- joint position noise with std `0.005716356937792565 rad`
- joint velocity noise with std `0.13718300792805946 rad/s`

#### 14.5 Final observation layout

The final single-frame observation concatenates:

1. `imu_lin_vel_cmd` -> 3
2. `ang_vel_cmd * ang_vel_scale` -> 3
3. `up_cmd` -> 3
4. `commands` -> 3
5. `act_pos_scaled` -> 10
6. `act_vel * dof_vel_scale` -> 10
7. `prev_actions` -> 10

Default total: `42`

If phase observation is enabled:

- compute `phase = 2*pi*(episode_length_buf * 0.02 / gait_period_s) + phase_offset`
- if standing and `freeze_phase_when_standing=True`, replace phase with `pi`
- append:
  - `sin(phase)`
  - `cos(phase)`

Important details:

- the local observation vector does not include joint torques
- therefore ADR `joint_torque_std` is unused here
- reward uses clean state, while observation is intentionally noisy and biased

### 15. Observation Post-Processing: `_get_observations()`

Order:

1. `self._update_state()`
2. `obs = self._compute_single_observation()`
3. If latency buffer exists:
   - roll buffer left
   - append newest obs at last index
   - gather delayed observation based on `obs_latency_steps`
4. If stack buffer exists:
   - roll stack left
   - append newest obs at last index
   - reshape from `[N, obs_dim, stack]` to `[N, obs_dim * stack]`
5. Optional debug printing of slices
6. Return `{"policy": obs}`

Default runtime behavior:

- `obs_max_latency = 0`, so latency path is inactive
- `obs_stack_frames = 3`, so stacking path is active
- default returned observation shape is `[num_envs, 126]`

Warm-start behavior on reset:

- if observation history/stacking buffers exist, reset fills all frames with the same initial observation

### 16. Reward Builder: `_get_rewards()`

This method first refreshes cached state, then computes many candidate terms. Not all of them end up in the final reward.

#### 16.1 Core tracking terms

```text
vel_err = com_lin_vel_cmd[:, :2] - commands[:, :2]
yaw_err = com_ang_vel_cmd[:, 2] - commands[:, 2]

r_lin = exp(-||vel_err||^2 / lin_vel_sigma)
r_yaw = exp(-(yaw_err^2) / yaw_rate_sigma)
upright = clamp(up_b[:, 2], 0, 1)
```

With current config:

- `r_lin = exp(-||vel_err||^2 / 0.10)`
- `r_yaw = exp(-(yaw_err^2) / 0.5)`

The code also accumulates episode statistics for ADR:

- `_ep_lin_err_sum`
- `_ep_yaw_err_sum`
- `_ep_len`

#### 16.2 Gating terms

Computed every step:

- `cmd_speed = ||commands_xy||`
- `act_speed = ||com_lin_vel_b_xy||`
- `phase = 2*pi*(episode_length_buf * 0.02 / gait_period_s) + phase_offset`
- `air_time_gate = 1(cmd_speed > 0.1)`
- `gait_gate = 1(cmd_speed > 0.15 and act_speed > 0.08 and up_b.z > 0.70)`
- `yaw_gate = 1(|command_yaw| > 0.15)`

Then:

- `r_yaw = r_yaw * yaw_gate`

So yaw tracking reward is zero whenever the command yaw magnitude is small.

#### 16.3 Stand-related terms

The code computes:

- `stand_cmd`
- `stand_pose_err`
- `stand_pose_rew`
- `stand_upright_rew`
- `stand_vel_cost`
- `stand_action_cost`
- `stand_hip2_pen`
- `walk_hip2_pen`

But these are currently commented out of the final reward equation.

#### 16.4 Anti-phase hip reward

If both hip1 joints exist:

1. `target = sin(phase)`
2. compute hip1 offsets from default pose
3. normalize with `tanh(anti_phase_pos_gain * offset)` where gain is `2.5`
4. compare left to `+target`, right to `-target`
5. use Gaussian match with sigma `0.25`
6. average left/right matches
7. gate by `gait_gate`

This reward is active in the final reward.

#### 16.5 Pose, limits, symmetry, and stability terms

Computed terms:

- `act_cost = sum(actions^2)`
- `at_limit = count(abs(act_pos_scaled) > 0.98)`
- `pose_return_penalty = 0.3 * mean((act_pos - default)^2)` if `up_b.z > 0.7`, else `0`
- `sym_pen = mean((|left_off| - |right_off|)^2)`
- `thigh_pose_pen`
- `lin_vel_z_cost = imu_lin_vel_b[:, 2]^2`
- `ang_vel_xy_cost = sum(imu_ang_vel_b[:, :2]^2)`
- `flat_ori_cost = sum(up_b[:, :2]^2)`

Only some of these are active in the final reward.

#### 16.6 Smoothness and energy terms

Computed terms:

- `action_rate_cost = sum((actions - prev_actions)^2)`
- `dof_vel_cost = sum(act_vel^2)`
- `dof_vel_delta_cost = sum((act_vel - prev_act_vel)^2)`
- `energy_cost = sum(abs(applied_torque * act_vel))`

Then:

- `prev_act_vel[:] = act_vel`

If `gate_smoothness_to_swing=True`, these costs would be scaled by swing fraction. The default is `False`, so this gating path is inactive.

#### 16.7 Contact-based terms

Feet are lazily initialized the first time rewards need them.

Contact detection:

- use latest contact sensor history slice: `forces_hist[:, 0, ...]`
- positive vertical force only: `clamp(Fz, min=0)`
- contact if `Fz > 30 N`

Active contact-related computations:

- `air_rew`:
  - touchdown-only reward
  - `max(current_air_time - 0.2, 0)`
  - gated by `cmd_speed > 0.1`
- `air_time_sym_pen = (air_time_left - air_time_right)^2`
- `slip_cost`:
  - horizontal foot speed squared during contact
- `touchdown_vel_cost`:
  - downward foot speed at touchdown
  - normalized by `touchdown_vel_ref = 0.6`
  - gated by `cmd_speed > 0.15`
- `touchdown_force_cost`:
  - only computed if `touchdown_force_cost_scale > 0`
  - default scale is `0.0`, so current training ignores it
- `undesired`:
  - any non-foot body contact force norm exceeding `80 N`
- `contact_phase_rew`:
  - contact signal is `left_contact - right_contact`, so values are in `{-1, 0, +1}`
  - compare against `sin(phase)` using Gaussian width `0.35`
  - gate by `gait_gate`
- `no_fly`:
  - 1 if both feet are off the ground
  - currently not used in final reward

#### 16.8 Final active reward equation

The current active reward is:

```text
reward =
    3.0 * r_lin
  + 0.6 * r_yaw
  + 0.6 * upright
  + 0.05
  - 0.2 * at_limit
  - pose_return_penalty
  - 0.08 * lin_vel_z_cost
  - 0.01 * action_rate_cost
  - 0.002 * energy_cost
  - 0.1 * sym_pen
  + 0.2 * air_rew
  + 0.15 * anti_phase_rew
  + 0.08 * contact_phase_rew
  - 0.35 * air_time_sym_pen
  - 0.02 * slip_cost
  - 0.1 * undesired
  - 0.16 * touchdown_vel_cost
  - touchdown_force_scale * touchdown_force_cost
```

Then the reward is overridden on terminated environments:

```text
reward = death_cost = -1.0
```

#### 16.9 Terms computed but not currently active in the final reward

The code computes these but comments them out of the final equation:

- `action_cost_scale * act_cost`
- `ang_vel_xy_cost_scale * ang_vel_xy_cost`
- `flat_ori_cost_scale * flat_ori_cost`
- `dof_vel_cost_scale * dof_vel_cost`
- `dof_vel_delta_cost_scale * dof_vel_delta_cost`
- `standstill_penalty_scale * standstill`
- `speed_shortfall_cost_scale * speed_shortfall`
- `thigh_pose_cost_scale * thigh_pose_pen`
- `walk_hip2_cost_scale * walk_hip2_pen`
- `no_fly_cost_scale * no_fly`
- the entire stand-command reward block:
  - `stand_pose_reward_scale * stand_pose_rew`
  - `stand_upright_reward_scale * stand_upright_rew`
  - `stand_vel_cost_scale * stand_vel_cost`
  - `stand_action_cost_scale * stand_action_cost`
  - `stand_hip2_cost_scale * stand_hip2_pen`

Also note:

- `contact_phase_min_speed` exists in cfg but is not used

#### 16.10 Reward logging

If:

- `enable_reward_logging=True`
- `common_step_counter` exists
- `common_step_counter % 200 == 0`

then the env writes many scalar summaries into `self.extras`, including:

- unscaled tracking and penalty terms
- scaled reward contributions
- total reward stats
- command tracking diagnostics

By default `enable_reward_logging=False`.

### 17. Done Logic: `_get_dones()`

This method:

1. refreshes cached state
2. computes timeout:
   - `episode_length_buf >= randomized_episode_lengths - 1`
3. computes failure:
   - `fell = track_pos_w[:, 2] < 0.4`
   - `too_tilted = up_b[:, 2] < 0.5`
4. returns `(died, time_out)`

Important detail:

- the height termination uses the `"top"` tracking frame height, not root height or COM height

### 18. Reset Logic: `_reset_idx(env_ids)`

Reset sequence:

1. Normalize `env_ids` to a tensor.
2. If ADR exists and this is not the very first step:
   - determine batch success from timeout rate
   - compute mean episode linear and yaw tracking errors
   - call `_maybe_update_adr()`
3. `self.robot.reset(env_ids)`
4. `super()._reset_idx(env_ids)`
5. resample episode lengths
6. pull default joint positions and velocities
7. if ADR exists:
   - sample joint position noise from `robot_spawn.joint_pos_noise`
   - add uniform noise
   - clamp all joints to soft limits
   - sample joint velocity noise from `robot_spawn.joint_vel_noise`
8. pull default root state and offset the root position by `scene.env_origins`
9. write root pose, root velocity, and joint state to sim
10. invalidate cached state
11. optionally refresh runtime mass cache if that external flag is enabled
12. zero:
   - `actions`
   - `prev_actions`
   - `prev_act_vel`
   - `_ep_lin_err_sum`
   - `_ep_yaw_err_sum`
   - `_ep_len`
13. reset push timers and resample next push steps
14. sample custom ADR per-episode variables
15. sample commands
16. randomize phase if enabled, else set phase offset to zero
17. if visualization or observation buffers are needed, refresh cached state
18. optionally draw markers
19. if history/stack buffers exist, warm-start them with the current observation

With the default config:

- episode length is sampled uniformly from `[250, 1000]` control steps
- reset joint noise comes from ADR:
  - position noise in `[0.0, 0.06]`
  - velocity noise in `[0.0, 0.20]`
- `phase_offset` is uniform in `[0, 2*pi]`

Important details:

- `reset_joint_pos_noise` and `reset_joint_vel_noise` config fields are not used here
- reset noise is completely driven by ADR `robot_spawn`
- because `randomize_phase=True`, `phase_offset_default` is ignored

### 19. Curriculum and Command Sampling

#### 19.1 Stage progression

The env keeps:

- `_global_env_steps += num_envs` each control step
- `per_env_steps = _global_env_steps / num_envs`

So `per_env_steps` is effectively just the number of elapsed control steps.

Stages:

- stage 0: before `2500`
- stage 1: `2500` to `4999`
- stage 2: `5000+`

The env prints a banner whenever the stage changes.

#### 19.2 Command sampling behavior

If curriculum is disabled:

- `vx ~ U(-1.0, 1.0)`
- `vy ~ U(-0.5, 0.5)`
- `yaw ~ U(-1.0, 1.0)`

If curriculum is enabled:

- stage 0:
  - `vx ~ U(0.3, 1.0)`
  - `vy = 0`
  - `yaw = 0`
- stage 1:
  - `vx ~ U(-1.0, 1.0)`
  - `vy = 0`
  - `yaw ~ U(-1.0, 1.0)`
- stage 2:
  - `vx ~ U(-1.0, 1.0)`
  - `vy ~ U(-0.5, 0.5)`
  - `yaw ~ U(-1.0, 1.0)`

Optional masks:

- `zero_mask` with probability `zero_command_probability = 0.0`
- `turn_mask` with probability `turn_in_place_probability = 0.02`, only from stage `>= 1`

Turn-in-place behavior:

- set `vx = 0`
- set `vy = 0`
- sample `yaw ~ U(-1.0, 1.0)`
- enforce minimum magnitude `0.3`

ADR command scaling:

- apply `command_scale` only if current stage `>= adr_command_scale_min_stage`
- with current cfg, that means stage `2` only

Important details:

- `lin_vel_x_range`, `lin_vel_y_range`, `ang_vel_yaw_range`, and `stand_prob` are not used by this implementation

### 20. Feet Initialization: `_init_feet()`

This lazily discovers feet only once:

1. `robot.find_bodies(self.cfg.foot_body_regex)`
2. require exactly 2 matches
3. match those body names against contact-sensor body names by:
   - `endswith`
   - or substring inclusion
4. create:
   - `_feet_body_ids`
   - `_feet_sensor_ids`
   - `prev_foot_contact`

### 21. Episode Length Randomization: `_resample_episode_lengths()`

If randomization is disabled:

- set all selected envs to `max_episode_length`

If enabled:

- sample uniformly from `[min_steps, max_episode_length]`

With current config:

- `min_steps = 250`
- `max_steps = 1000`

### 22. Marker Visualization

`define_markers()` creates two USD arrow markers:

- marker 0: red command arrow
- marker 1: green actual-velocity arrow

Both use:

- USD: `Props/UIElements/arrow_x.usd`
- scale: `(0.25, 0.25, 0.5)`

`_visualize_markers()`:

1. chooses one env or all envs depending on debug flags
2. uses `track_pos_w + marker_offset` as marker origin
3. converts command-frame XY command back into body/world orientation
4. computes yaw-only arrow orientations for:
   - command direction
   - actual velocity direction
5. visualizes red/green pairs

Important details:

- the arrows encode direction only, not magnitude
- `vel_vis_scale` is not used
- `vel_vis_every_n` is not used
- the docstring says single-env, but the code supports all-env visualization
- when called from `_pre_physics_step`, it uses whatever state was last cached; it does not call `_update_state()` itself

## Config Fields and Code Paths That Are Currently Dormant or Unused

These fields exist in `HumanoidEnvCfg` but are not referenced by the local `tritonhumanoid_env.py` implementation:

- `obs_term_cfg`
- `contact_phase_min_speed`
- `penalty_curriculum_enabled`
- `penalty_curriculum_mode`
- `penalty_curriculum_min_scale`
- `penalty_curriculum_max_scale`
- `penalty_curriculum_use_ema`
- `penalty_curriculum_ema_alpha`
- `penalty_curriculum_window`
- `penalty_curriculum_exclude_timeouts`
- `penalty_curriculum_degree`
- `penalty_curriculum_low_len_frac`
- `penalty_curriculum_high_len_frac`
- `penalty_curriculum_target_len_frac`
- `penalty_curriculum_power`
- `gait_period_randomization_width`
- `phase_offset_default`
- `lin_vel_x_range`
- `lin_vel_y_range`
- `ang_vel_yaw_range`
- `stand_prob`
- `reset_joint_pos_noise`
- `reset_joint_vel_noise`
- `push_strong`
- `strong_min_push_interval_s`
- `strong_max_push_interval_s`
- `vel_vis_scale`
- `vel_vis_every_n`

Some config fields are not referenced directly in `tritonhumanoid_env.py` because they are framework-level fields used by Isaac Lab / `DirectRLEnv`, for example:

- `episode_length_s`
- `scene`
- `events`
- `state_space`

## Important Design Mismatches and Caveats

- The config file imports an external `LocomotionEnv`, but local gym registration uses the local `tritonhumanoid_env.py` class.
- Observation uses IMU linear velocity, while reward tracking uses COM linear velocity by default.
- Observation uses noisy/biased IMU and joint features, while rewards and terminations use clean internal state.
- The local env defines torque noise in ADR config, but the observation vector contains no torques.
- Many reward terms are computed for diagnostics or experimentation but are commented out of the active reward equation.
- `push_strong` and its strong-interval settings are present in cfg but have no effect.
- `vel_vis_scale` and `vel_vis_every_n` are present in cfg but have no effect.
- Mass randomization is active, but mass-cache refresh on reset is off by default unless provided externally.
- `LocomotionADR.print_params()` has a `latency` branch, but this config does not define any ADR latency group.

## Bottom Line

This locomotion environment is a 50 Hz position-target locomotion task with:

- 4096 parallel flat-ground environments
- 10 actuated leg joints
- a manually built 126-D stacked observation by default
- command-frame velocity tracking with a `-pi/2` yaw remap
- a reward dominated by linear tracking, uprightness, air-time, anti-phase timing, contact-phase timing, and several active penalties
- reset-time event randomization plus an ADR layer that ramps disturbance and sensor corruption over training
- a number of additional config fields and reward terms that currently exist as dormant or experimental code paths rather than active behavior
