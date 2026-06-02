# LAFAN Walk Tracking Training

The default walk-only training set is the 12 enriched CH clips in:

```text
/cephfs/holosoma/data/lafan/retargeted/ch_robot_stance_flatfoot_locomotion_full_floor_norm_with_vel
```

That directory currently also contains `run2_*` clips. The helper scripts generate a stable manifest containing only the expected `walk*` clips so run motions do not enter walk training by accident.

By default the manifest is written to:

```text
manifests/lafan_walk_tracking_manifest.txt
```

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

This writes checkpoints under:

```text
logs/rl_games/humanoid_flat_direct/lafan_walk_tracking
```

Useful overrides:

```bash
NUM_ENVS=8192 MAX_ITERATIONS=2000 SEED=11 ./run_train_lafan_walk_tracking.sh
```

Resume from a specific checkpoint:

```bash
CHECKPOINT=logs/rl_games/humanoid_flat_direct/my_run/nn/last_humanoid_flat_direct_ep_500_rew_123.0.pth \
  ./run_train_lafan_walk_tracking.sh
```

Resume from the newest checkpoint under the current experiment directory:

```bash
RESUME_LAST=1 ./run_train_lafan_walk_tracking.sh
```

Equivalent shorthand:

```bash
CHECKPOINT=latest ./run_train_lafan_walk_tracking.sh
```

If your checkpoints live elsewhere, override the search root:

```bash
RESUME_LAST=1 LOG_ROOT=/path/to/logs/rl_games/humanoid_flat_direct/lafan_walk_tracking ./run_train_lafan_walk_tracking.sh
```

Preview the exact training command without launching IsaacSim:

```bash
DRY_RUN=1 RESUME_LAST=1 ./run_train_lafan_walk_tracking.sh
```

Extra Hydra overrides can be appended:

```bash
./run_train_lafan_walk_tracking.sh env.motion_random_start=false env.motion_reference_debug_print=true
```

Name a separate run with:

```bash
EXPERIMENT_NAME=lafan_walk_resid010 ./run_train_lafan_walk_tracking.sh env.residual_action_scale=0.10
```

## Multi-GPU

For one distributed rl-games run on five visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 NUM_GPUS=5 NUM_ENVS=2048 EXPERIMENT_NAME=lafan_walk_ddp5 \
  ./run_train_lafan_walk_tracking.sh
```

`NUM_ENVS` is per training process when `NUM_GPUS > 1`, so the command above creates about `5 * 2048` total environments.

Early on, five independent single-GPU runs are often more informative than one distributed run:

```bash
CUDA_VISIBLE_DEVICES=0 EXPERIMENT_NAME=lafan_walk_base SEED=1 ./run_train_lafan_walk_tracking.sh
CUDA_VISIBLE_DEVICES=1 EXPERIMENT_NAME=lafan_walk_resid010 SEED=2 ./run_train_lafan_walk_tracking.sh env.residual_action_scale=0.10
CUDA_VISIBLE_DEVICES=2 EXPERIMENT_NAME=lafan_walk_resid020 SEED=3 ./run_train_lafan_walk_tracking.sh env.residual_action_scale=0.20
CUDA_VISIBLE_DEVICES=3 EXPERIMENT_NAME=lafan_walk_no_rand_start SEED=4 ./run_train_lafan_walk_tracking.sh env.motion_random_start=false
CUDA_VISIBLE_DEVICES=4 EXPERIMENT_NAME=lafan_walk_lr5e4 SEED=5 ./run_train_lafan_walk_tracking.sh agent.params.config.learning_rate=5.0e-4
```

Recommended order:

1. Run reference playback first and check that tracking errors are near zero.
2. Overfit a short run on the 12 walk clips with one GPU.
3. Launch the five independent variants above.
4. Use `NUM_GPUS=5` distributed training once reward curves and resets look sane.

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
