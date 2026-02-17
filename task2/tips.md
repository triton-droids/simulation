# Task 2 Tips

## Fast Debug Checklist
- Confirm observation and action dimensions before training.
- Verify planner output shape matches environment action shape.
- Run `5-10` episodes first and inspect one saved trajectory.
- Check dataset is non-empty before creating a dataloader.
- During training, log loss and evaluation success every few epochs.

## Environment Implementation Hints
- Start with observations as `[qpos, qvel]`, then add object pose terms if needed.
- Keep `step()` API strict: `(obs, reward, terminated, truncated, info)`.
- Use `TimeLimit` for truncation; reserve `terminated` for task success/failure logic.
- In `reset_model`, randomize only small ranges first so planner feasibility stays high.

## Planner Hints
- `target_pose` should be in world frame with shape `(7,)` as `[x, y, z, qw, qx, qy, qz]`.
- `qpos` passed to the planner should match robot joint ordering in the URDF/SRDF setup.
- Execute planned joint positions with gripper command appended as the final action value.
- If planning fails often, reduce randomization range and increase approach height above cube.

## Dataset Hints
- Flatten trajectories into one list of `(obs, action)` pairs.
- Convert to `float32` arrays/tensors once in `__init__`.
- Return tensors in `__getitem__` to avoid repeated conversion overhead.
- Shuffle in `DataLoader` for training.

## Network And Loss Hints
- Use one MLP backbone and two output heads: `mean` and `log_std`.
- Keep `log_std` bounded for stability (for example clamp to `[-5, 2]`).
- Train with Gaussian NLL:
  `loss = -Normal(mean, exp(log_std)).log_prob(action).sum(-1).mean()`
- Evaluate with deterministic action (`mean`) first.

## Common Failure Modes
- Planner demonstrations are inconsistent, causing noisy supervision.
- Action scales differ between planner and environment control limits.
- Dataset contains many near-duplicate failed states.
- Policy drifts due to covariate shift during long-horizon execution.

## Useful References
- MuJoCo Python API: https://mujoco.readthedocs.io/en/stable/python.html
- MuJoCo named access (IDs, bodies, joints): https://mujoco.readthedocs.io/en/stable/python.html#named-access
- Gymnasium docs: https://gymnasium.farama.org/
- Behavior cloning overview (conceptual): https://underactuated.mit.edu/imitation.html
- PyTorch tutorial index: https://pytorch.org/tutorials/
