# Motion Reference Integration Handoff

## Summary

We added the first integration point between the Holosoma retargeting pipeline and the IsaacLab locomotion environment.

The retargeting side now produces a converted RL tracking motion file:

```text
source/tritonhumanoid/tritonhumanoid/data/motions/sub10_largebox_049_clip120_mj_fps50.npz
```

This file is already in the converted format, not the raw retargeting format. It contains:

```text
fps
joint_pos
joint_vel
body_pos_w
body_quat_w
body_lin_vel_w
body_ang_vel_w
joint_names
body_names
```

The file has 199 frames at 50 FPS, so it represents about 4 seconds of motion.

## Why We Made These Changes

The current IsaacLab locomotion environment was command-based: the policy observes commanded velocity, robot state, joint state, and previous actions. To train a whole-body or motion-tracking policy, the policy also needs to observe a target reference motion.

The Holosoma converter output is the right boundary between the retargeting repo and the simulation repo. Instead of making IsaacLab depend directly on Holosoma code, we now let IsaacLab consume a stable `.npz` motion contract.

This keeps the integration clean:

```text
Holosoma retargeting -> converted motion .npz -> IsaacLab motion reference loader
```

That is better than merging the full Holosoma repository into the simulation repo immediately, because Holosoma includes extra MuJoCo, visualization, dataset, Docker, and training code that we do not need inside the IsaacLab runtime yet.

## Files Changed

### `tritonhumanoid_env_cfg.py`

Added motion reference config options:

```python
use_motion_reference = True
motion_reference_file = "data/motions/sub10_largebox_049_clip120_mj_fps50.npz"
motion_reference_observation = True
motion_reference_random_start = True
motion_reference_pos_error_scale = 1.0
motion_reference_vel_scale = 0.1
motion_reference_debug_print = False
```

The observation dimension is updated to include the motion reference observation terms.

### `tritonhumanoid_env.py`

Added a motion reference loader that:

1. Loads the converted `.npz` file.
2. Validates required keys and array shapes.
3. Reads `joint_names` from the `.npz`.
4. Reads IsaacLab action joint order from `self.robot.data.joint_names`.
5. Builds a name-based remap from converter order to IsaacLab action order.
6. Samples a per-env starting reference frame on reset.
7. Advances the reference frame based on `episode_length_buf`, environment control timestep, and motion FPS.
8. Adds target joint position error and target joint velocity to the policy observation.

## Important Joint Order Issue

The converted motion file stores joints in this order:

```text
left_hip1_joint
left_hip2_joint
left_thigh_joint
left_knee_joint
left_ankle_joint
right_hip1_joint
right_hip2_joint
right_thigh_joint
right_knee_joint
right_ankle_joint
```

IsaacLab action order is:

```text
left_hip1_joint
right_hip1_joint
left_hip2_joint
right_hip2_joint
left_thigh_joint
right_thigh_joint
left_knee_joint
right_knee_joint
left_ankle_joint
right_ankle_joint
```

So we do not directly use `joint_pos[:, 7:]`. We remap by name.

The remap is:

```text
[0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
```

This prevents target values from being sent to the wrong joint.

## Current Scope

This change only adds the data interface and reference motion observation.

It does not yet add a motion-tracking reward. That is intentional. We should first verify that:

1. The motion file loads correctly.
2. Joint names remap correctly.
3. Target frames advance over time.
4. Reset samples valid reference frames.
5. Observation shape matches the policy config.

Once those are confirmed, the next step is adding a simple joint-position tracking reward.

## Verification Done

Static Python compile check:

```bash
python -m py_compile \
  source/tritonhumanoid/tritonhumanoid/tasks/direct/tritonhumanoid/tritonhumanoid_env.py \
  source/tritonhumanoid/tritonhumanoid/tasks/direct/tritonhumanoid/tritonhumanoid_env_cfg.py
```

Motion file contract check:

```text
fps: 50
joint_pos: (199, 17)
joint_vel: (199, 16)
remap: [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
```

We have not yet run a full IsaacSim rollout after this change.

## Recommended Runtime Check

Run a small debug job with:

```python
motion_reference_debug_print = True
debug_obs_print = True
debug_obs_print_steps = 3
```

Expected debug output should show:

```text
[MotionReference] loaded ...
[MotionReference] reference joint_names=...
[MotionReference] action joint_names=...
[MotionReference] remap reference->action=[0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
[MotionReference] first target ...
```

## Next Step

After runtime verification, add a simple reward term:

```text
exp(-mean((current_action_order_joint_pos - target_action_order_joint_pos)^2) / sigma)
```

Start with joint tracking only. Add body/keypoint tracking later using `body_pos_w` and `body_names`.
