# Evaluation and MuJoCo Sim2Sim

This guide covers two evaluation workflows:

1. Kinematic MuJoCo replay of an IsaacLab rollout.
2. IsaacLab-vs-MuJoCo deployment parity logging.

The examples below use the latest saved episode checkpoint from:

```text
logs/rl_games/humanoid_flat_direct/2026-06-04_00-46-13/nn/last_humanoid_flat_direct_ep_2250_rew_2828.0017.pth
```

The MuJoCo source model is:

```text
mjcf/robot_description/scene.xml
```

Use the Isaac Python runtime with conda environment variables unset:

```bash
ISAAC_PY="env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL /workspace/isaaclab/_isaac_sim/python.sh"
```

## Command Profile

The forward-only command profile sends:

```text
(vx, vy, wz) = (0.6, 0.0, 0.0)
```

Override the forward speed with `--forward_vx`.

The older default profile is `stand_forward_yaw`: 2 seconds standing, 4 seconds forward at `vx=0.6`, then 4 seconds yaw at `wz=0.6`, repeating.

## 1. Kinematic MuJoCo Replay

This records a rollout in IsaacLab and replays the recorded root pose, root velocity, joint positions, and joint velocities in MuJoCo.

This is not a MuJoCo physics deployment. The playback script writes `qpos/qvel` directly each frame and calls `mj_forward()`, so it is useful for checking the MJCF body/joint mapping and visually inspecting the IsaacLab motion without MuJoCo stepping, contacts, gravity, floor interaction, or policy dynamics.

Playback does not load a policy checkpoint. It only replays the `.npz` trace you pass with `--trace`, so regenerate the trace after changing checkpoints and use a checkpoint-specific trace filename.

Record the IsaacLab trace:

```bash
$ISAAC_PY scripts/rl_games/parity_log.py \
  --task=Isaac-Humanoid-Locomotion-Flat-Direct-v0 \
  --num_envs=1 \
  --num_steps=1000 \
  --headless \
  --action_source=policy \
  --checkpoint=logs/rl_games/humanoid_flat_direct/2026-06-04_00-46-13/nn/last_humanoid_flat_direct_ep_2250_rew_2828.0017.pth \
  --command_profile=forward \
  --forward_vx=0.6 \
  --sim2sim-log \
  --output=logs/parity/isaac_forward_ep2250_kinematic.npz
```

Validate the trace and MuJoCo model:

```bash
$ISAAC_PY scripts/mujoco_playback_locomotion.py \
  --trace logs/parity/isaac_forward_ep2250_kinematic.npz \
  --source-xml mjcf/robot_description/scene.xml \
  --refresh-model \
  --validate-only
```

Replay in MuJoCo without physics/contact/gravity/floor interaction:

```bash
$ISAAC_PY scripts/mujoco_playback_locomotion.py \
  --trace logs/parity/isaac_forward_ep2250_kinematic.npz \
  --source-xml mjcf/robot_description/scene.xml \
  --refresh-model \
  --render \
  --real-time \
  --hide-floor
```

Kinematic playback disables MuJoCo contact detection and gravity by default. `--hide-floor` removes the floor/ground visual geoms so the viewer shows only the recorded robot motion. Use `--enable-contact` only when you intentionally want MuJoCo contact detection/contact visualization during kinematic replay. Use `--enable-gravity` only when you intentionally want MuJoCo gravity enabled while forwarding the recorded state.

The trace must contain `qpos` and `qvel`; that is why the record command uses `--sim2sim-log`.

`mujoco_playback_locomotion.py` reads the sidecar metadata file next to the trace, for example `logs/parity/isaac_forward_ep2250_kinematic.json`, and uses its `joint_order` field to reorder IsaacLab's interleaved joint state into MuJoCo XML joint order before writing `qpos/qvel`.

### Check Whether the Trace Actually Moved

Before debugging MuJoCo visualization, inspect the IsaacLab trace itself:

```bash
$ISAAC_PY - <<'PY'
import numpy as np

d = np.load("logs/parity/isaac_forward_ep2250_kinematic.npz")
root = d["root_pos_w"]
cmd = d["commands"]
done = np.where(d["done"].reshape(-1) > 0.5)[0]
end = int(done[0]) if len(done) else len(root) - 1
start = min(100, end)
disp = root[end] - root[start]
print("checkpoint trace command mean:", cmd[start:end].mean(axis=0))
print("root displacement:", disp)
for key in ("base_lin_vel", "lin_vel_cmd", "root_lin_vel_w"):
    vel = d[key][start:end]
    print(key, "mean:", vel.mean(axis=0), "max_abs_x:", abs(vel[:, 0]).max())
print("done indices:", done[:10].tolist())
PY
```

For the `2026-06-04_00-46-13` ep2250 checkpoint, the policy may visibly jiggle/rock while producing little net forward displacement. If `base_lin_vel` and `root displacement` are near zero, MuJoCo kinematic playback is faithfully showing a mostly-stationary IsaacLab rollout.

## 2. IsaacLab vs MuJoCo Deployment Parity

This deploys the policy in both simulators and compares logged channels.

Important format distinction:

- IsaacLab/RL-Games uses the checkpoint `.pth`.
- MuJoCo deployment uses an exported TorchScript policy `.pt`.

### Export TorchScript Policy

Export the policy from the RL-Games checkpoint:

```bash
$ISAAC_PY scripts/rl_games/play.py \
  --task=Isaac-Humanoid-Locomotion-Flat-Direct-v0 \
  --num_envs=1 \
  --headless \
  --video \
  --video_length=1 \
  --checkpoint=logs/rl_games/humanoid_flat_direct/2026-06-04_00-46-13/nn/last_humanoid_flat_direct_ep_2250_rew_2828.0017.pth
```

Expected output:

```text
logs/rl_games/humanoid_flat_direct/2026-06-04_00-46-13/exported_policy/ppo_policy.pt
logs/rl_games/humanoid_flat_direct/2026-06-04_00-46-13/exported_policy/ppo_metadata.pt
```

### Record IsaacLab Deployment

```bash
$ISAAC_PY scripts/rl_games/parity_log.py \
  --task=Isaac-Humanoid-Locomotion-Flat-Direct-v0 \
  --num_envs=1 \
  --num_steps=1000 \
  --headless \
  --action_source=policy \
  --checkpoint=logs/rl_games/humanoid_flat_direct/2026-06-04_00-46-13/nn/last_humanoid_flat_direct_ep_2250_rew_2828.0017.pth \
  --command_profile=forward \
  --forward_vx=0.6 \
  --sim2sim-log \
  --output=logs/parity/isaac_forward_ep2250_policy.npz
```

### Run MuJoCo Deployment

```bash
$ISAAC_PY scripts/mujoco_eval_locomotion.py \
  --policy=logs/rl_games/humanoid_flat_direct/2026-06-04_00-46-13/exported_policy/ppo_policy.pt \
  --source-xml=mjcf/robot_description/scene.xml \
  --refresh-model \
  --command-profile=forward \
  --max-steps=1000 \
  --output=logs/parity/mujoco_forward_ep2250_policy.npz \
  --metrics-json=logs/parity/mujoco_forward_ep2250_policy_metrics.json \
  --render \
  --real-time
```

For a headless non-rendering run, omit `--render --real-time`.

### Compare Logs

Use `parity_log.py` with `--compare_with` to generate comparison metrics:

```bash
$ISAAC_PY scripts/rl_games/parity_log.py \
  --task=Isaac-Humanoid-Locomotion-Flat-Direct-v0 \
  --num_envs=1 \
  --num_steps=1000 \
  --headless \
  --action_source=policy \
  --checkpoint=logs/rl_games/humanoid_flat_direct/2026-06-04_00-46-13/nn/last_humanoid_flat_direct_ep_2250_rew_2828.0017.pth \
  --command_profile=forward \
  --forward_vx=0.6 \
  --sim2sim-log \
  --output=logs/parity/isaac_forward_ep2250_policy_compare.npz \
  --compare_with=logs/parity/mujoco_forward_ep2250_policy.npz
```

Expected comparison output:

```text
logs/parity/isaac_forward_ep2250_policy_compare_compare.json
```

Inspect these channels first:

- `commands`
- `obs_latest`
- `actions_input` / `actions`
- `act_pos`
- `act_vel`
- `base_lin_vel`
- `base_ang_vel`
- `up_b`

The comparison JSON reports per-channel correlation, best lag, raw RMSE, and affine-corrected RMSE.

## Useful Validation Commands

Validate the edited MuJoCo model contract:

```bash
$ISAAC_PY - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "source/tritonhumanoid")
from tritonhumanoid.eval.mujoco_locomotion import validate_mjcf_against_urdf

validate_mjcf_against_urdf(Path("mjcf/robot_description/scene.xml"), require_mujoco=True)
print("MuJoCo scene validation OK")
PY
```

Run focused MuJoCo tests:

```bash
$ISAAC_PY -m pytest tests/test_mujoco_locomotion_eval.py
```

## Notes

- `scripts/mujoco_playback_locomotion.py` is for kinematic replay and does not step MuJoCo physics. It disables MuJoCo contact detection and gravity by default.
- `scripts/mujoco_eval_locomotion.py` deploys the TorchScript policy in MuJoCo physics.
- `scripts/rl_games/parity_log.py --sim2sim-log` records the extra channels required for MuJoCo replay and sim2sim comparison.
- `scene.xml` is patched/cached under `logs/mujoco/model/ch_robot_10dof_isaac_locomotion.xml` when using `--source-xml ... --refresh-model`.
- IsaacLab records policy joints in interleaved order, while the MuJoCo model stores joints as left-chain then right-chain. The MuJoCo helpers keep XML/physics state in MuJoCo order and policy/trace observations/actions in Isaac policy order.
