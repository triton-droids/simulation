# LAFAN Walk Tracking Training

The default walk-only training set is the 12 enriched CH clips in:

```text
/cephfs/holosoma/data/lafan/retargeted/ch_robot_stance_flatfoot_locomotion_full_floor_norm_with_vel
```

That directory currently also contains `run2_*` clips. The helper scripts generate a temporary manifest containing only the expected `walk*` clips so run motions do not enter walk training by accident.

## Reference Playback

Run reference playback before PPO:

```bash
./run_lafan_walk_reference_playback.sh
```

Useful overrides:

```bash
HEADLESS=0 NUM_ENVS=4 NUM_STEPS=900 ./run_lafan_walk_reference_playback.sh --debug_print
```

## Train

Start rl-games PPO training:

```bash
./run_train_lafan_walk_tracking.sh
```

Useful overrides:

```bash
NUM_ENVS=8192 MAX_ITERATIONS=2000 ./run_train_lafan_walk_tracking.sh
```

Resume from a specific checkpoint:

```bash
CHECKPOINT=logs/rl_games/humanoid_flat_direct/my_run/nn/last_humanoid_flat_direct_ep_500_rew_123.0.pth \
  ./run_train_lafan_walk_tracking.sh
```

Resume from the newest checkpoint under `logs/rl_games/humanoid_flat_direct`:

```bash
RESUME_LAST=1 ./run_train_lafan_walk_tracking.sh
```

Equivalent shorthand:

```bash
CHECKPOINT=latest ./run_train_lafan_walk_tracking.sh
```

If your checkpoints live elsewhere, override the search root:

```bash
RESUME_LAST=1 LOG_ROOT=/path/to/logs/rl_games/humanoid_flat_direct ./run_train_lafan_walk_tracking.sh
```

Preview the exact training command without launching IsaacSim:

```bash
DRY_RUN=1 RESUME_LAST=1 ./run_train_lafan_walk_tracking.sh
```

Extra Hydra overrides can be appended:

```bash
./run_train_lafan_walk_tracking.sh env.motion_random_start=false env.motion_reference_debug_print=true
```

## Play A Checkpoint

Play the latest checkpoint using the same walk-only manifest:

```bash
./run_play_lafan_walk_tracking.sh
```

Useful overrides:

```bash
HEADLESS=1 NUM_ENVS=32 ./run_play_lafan_walk_tracking.sh --use_last_checkpoint
```

To point at a copied dataset, set:

```bash
MOTION_DIR=/path/to/ch_robot_stance_flatfoot_locomotion_full_floor_norm_with_vel ./run_train_lafan_walk_tracking.sh
```
