# Training

Every public training run now creates one self-contained folder under `Artifacts/Runs/`.
It automatically records metrics, produces a final graph, attempts a final
rollout video, and keeps its checkpoints and resolved configuration together.

## Verify the pipeline first

```powershell
.venv\Scripts\python.exe Scripts\train.py --smoke
```

This runs 1,024 environment steps with a tiny network and writes a checkpoint.
It answers “can reset, PPO, logging, and checkpointing run?” It does not produce
a useful walking policy.

The output looks like:

```text
Artifacts/Runs/run_YYYYMMDD_HHMMSS/
├── summary.json
├── progress.json
├── resolved_config.json
├── plots/final_progress.png
├── videos/final_walk-eval.mp4
└── logs/
    ├── metrics.jsonl
    └── checkpoints/
```

One final best-checkpoint video is automatic by default. If the machine cannot render video,
training remains valid, prints a warning, and still writes metrics and plots.
Use `--no-video` only when intentionally running headless or minimizing sweep
cost.

## Full historical profile

```powershell
.venv\Scripts\python.exe Scripts\train.py
```

This selects the preserved `ppo_g1` five-million-step profile. The repository
does not claim that it is a validated three-seed locomotion recipe. Do not start
it merely to test installation; use `--smoke` instead.

## What training uses

- Robot/task: native Unitree G1 velocity task
- Simulator: MuJoCo XLA (MJX)
- Trainer: Brax PPO
- Default logger: local
- Default seed: `0`
- Camera: none during the vectorized training loop

Training prints metrics, not a live robot window. Many parallel environments
are stepped for throughput. Watch a saved checkpoint separately with the viewer
or evaluator.

## Hyperparameter and environment sweeps

The beginner sweep performs the same nested loop used by Pupper:

1. Select one trainer/PPO override bundle.
2. Pair it with every environment override bundle.
3. Train each combination in an isolated numbered folder.
4. Rank completed trials by best evaluation reward.

First verify the complete sweep pipeline with tiny runs:

```powershell
.venv\Scripts\python.exe Scripts\sweep.py --smoke
```

Edit these two files:

- `App/Configs/sweeps/training_hyperparameters.json` — PPO/trainer values.
- `App/Configs/sweeps/environment_hyperparameters.json` — MJX task/world values.

The supplied two rows in each file produce four trials. A completed sweep contains:

```text
Artifacts/Sweeps/sweep_YYYYMMDD_HHMMSS/
├── sweep_config.json
├── results.json
├── leaderboard.json
├── best_trial.json
├── trial_001/
├── trial_002/
└── ...
```

Run a custom file with:

```powershell
.venv\Scripts\python.exe Scripts\sweep.py `
  --training-hyperparameters configs\sweeps\my_training.json `
  --environment-hyperparameters configs\sweeps\my_environment.json
```

Use `--no-video` for a large screening sweep, then rerun finalists with video.
The leaderboard ranks reward; it does not prove walking quality, gait quality,
or RoboCup readiness. Videos and fixed evaluations still require review.

## Evidence boundary

Gate 4 has not passed. The corrected C06 evaluation in the research record
balanced for ten seconds but stood still under a `0.5 m/s` forward command.
Older normalization-enabled evaluations that omitted observation preprocessing
are invalid and must not be cited. Read `ASTRA_HANDOFF.md` before spending a
non-smoke training budget.
