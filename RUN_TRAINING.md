# Training Runbook

Commands assume the repo root:

```bash
cd /workspace/simulation
```

## Default Locomotion Training

The launcher defaults to 16,384 environments and uses Isaac Python directly when it can:

```bash
./run_train_locomotion.sh
```

Current PPO batch geometry:

```text
num_envs:        16384
horizon_length:  24
samples/update: 393216
minibatch_size: 98304
minibatches:    4
mini_epochs:    3
optimizer steps/update: 12
```

These are set in:

```text
source/tritonhumanoid/tritonhumanoid/tasks/direct/tritonhumanoid/agents/rl_games_ppo_cfg.yaml
```

## Domain Randomization Modes

Adaptive ADR is the config default. This ramps randomization based on training success:

```bash
DOMAIN_RANDOMIZATION_MODE=adaptive ./run_train_locomotion.sh
```

Fixed ADR starts at the maximum configured randomization ranges and holds them there:

```bash
DOMAIN_RANDOMIZATION_MODE=fixed ./run_train_locomotion.sh
```

Disable ADR while keeping the normal reset/event configuration:

```bash
DOMAIN_RANDOMIZATION_MODE=off ./run_train_locomotion.sh
```

Fast iteration mode disables ADR, events, contact sensors, reward logging, pushes, reset noise, and latency:

```bash
FAST_TRAIN=1 ./run_train_locomotion.sh
```

Do not combine `FAST_TRAIN=1` with `DOMAIN_RANDOMIZATION_MODE=adaptive` or `fixed`; the launcher exits early because fast mode disables domain randomization.

## Environment Count

Override the number of environments:

```bash
NUM_ENVS=8192 ./run_train_locomotion.sh
NUM_ENVS=16384 ./run_train_locomotion.sh
```

When changing `NUM_ENVS`, consider whether the PPO batch still has enough minibatches. The current `minibatch_size=98304` is tuned for 16,384 envs:

```text
16384 envs * 24 horizon = 393216 samples/update = 4 minibatches
8192 envs  * 24 horizon = 196608 samples/update = 2 minibatches
4096 envs  * 24 horizon = 98304 samples/update  = 1 minibatch
```

For smaller env counts, pass temporary PPO overrides or edit the YAML:

```bash
NUM_ENVS=8192 ./run_train_locomotion.sh \
  agent.params.config.minibatch_size=49152
```

```bash
NUM_ENVS=4096 ./run_train_locomotion.sh \
  agent.params.config.minibatch_size=24576
```

Those keep roughly 4 minibatches/update.

## Solver Iterations

The robot asset defaults to transfer-oriented solver settings:

```text
solver_position_iteration_count: 12
solver_velocity_iteration_count: 2
```

Run with the defaults:

```bash
./run_train_locomotion.sh
```

Try a faster but still reasonable compromise:

```bash
SOLVER_POSITION_ITERATIONS=8 SOLVER_VELOCITY_ITERATIONS=1 ./run_train_locomotion.sh
```

Avoid `4/0` for final sim2sim or sim2real training unless you are intentionally prioritizing throughput over contact/velocity fidelity.

## Direct Training Command

Equivalent direct command:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/rl_games/train.py \
  --task=Isaac-Humanoid-Locomotion-Flat-Direct-v0 \
  --num_envs=16384 \
  --headless
```

Direct command with fixed ADR:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/rl_games/train.py \
  --task=Isaac-Humanoid-Locomotion-Flat-Direct-v0 \
  --num_envs=16384 \
  --headless \
  env.enable_adr=true \
  env.domain_randomization_mode=fixed
```

Direct command without ADR:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/rl_games/train.py \
  --task=Isaac-Humanoid-Locomotion-Flat-Direct-v0 \
  --num_envs=16384 \
  --headless \
  env.enable_adr=false
```

## TensorBoard Logging

RL-Games writes TensorBoard event files under:

```text
logs/rl_games/humanoid_flat_direct/<run_id>/summaries/
```

Start TensorBoard:

```bash
tensorboard --logdir /workspace/simulation/logs/rl_games --host 0.0.0.0 --port 6006
```

If `tensorboard` is missing in the shell you are using for viewing:

```bash
python -m pip install tensorboard
```

Useful scalar names:

```text
losses/a_loss          actor/policy loss
losses/c_loss          critic/value loss
losses/entropy         policy entropy
losses/bounds_loss     action bounds loss
info/kl                PPO KL
info/last_lr           learning rate
performance/step_fps   simulation/training throughput
rewards/step           raw episode reward
shaped_rewards/step    shaped episode reward
episode_lengths/step   episode length
```

To log explained variance, enable RL-Games diagnostics for the run:

```bash
./run_train_locomotion.sh agent.params.config.use_diagnostics=True
```

Then look for:

```text
diagnostics/exp_var
diagnostics/clip_frac/*
diagnostics/rms_value/mean
diagnostics/rms_value/var
```

## W&B Sync

The training script can sync TensorBoard summaries to Weights & Biases:

```bash
DOMAIN_RANDOMIZATION_MODE=fixed ./run_train_locomotion.sh \
  --wandb \
  --wandb_project tritonhumanoid \
  --wandb_name locomotion_16k_fixed_dr
```

For offline W&B logging:

```bash
./run_train_locomotion.sh \
  --wandb \
  --wandb_project tritonhumanoid \
  --wandb_name locomotion_16k_debug \
  --wandb_mode offline
```

## Common Runs

Fast smoke test:

```bash
FAST_TRAIN=1 NUM_ENVS=4096 ./run_train_locomotion.sh --max_iterations 20
```

16k adaptive ADR with diagnostics:

```bash
DOMAIN_RANDOMIZATION_MODE=adaptive ./run_train_locomotion.sh \
  agent.params.config.use_diagnostics=True
```

16k fixed max DR with diagnostics:

```bash
DOMAIN_RANDOMIZATION_MODE=fixed ./run_train_locomotion.sh \
  agent.params.config.use_diagnostics=True
```

16k no ADR:

```bash
DOMAIN_RANDOMIZATION_MODE=off ./run_train_locomotion.sh
```
