# Architecture for Humans

## What this project does

Robocub trains and evaluates Unitree G1 locomotion policies for Triton Droids
RoboCup work. The working locomotion path uses MuJoCo/MJX and Brax PPO; the
Isaac Lab Field01 soccer environment is a separate integration track.

## Entry points

Normal users run only files in `App/Scripts/`:

| File | Responsibility |
|---|---|
| `visualize.py` | Restore T02 and open the MuJoCo field camera |
| `train.py` | Start one isolated PPO training run |
| `sweep.py` | Run every trainer JSON row against every environment JSON row |
| `evaluate.py` | Evaluate the canonical T02 checkpoint |
| `status.py` | Print the selected simulator, checkpoint, and weights path |

Each file is intentionally tiny. It delegates command parsing and path handling
to `robocub.user_cli/`, which then invokes the implementation under `robocub/`.

## Single-run data flow

```text
App/Scripts/train.py
  -> robocub.user_cli/cli.py
     -> creates Artifacts/Runs/run_TIMESTAMP
     -> robocub/scripts/train.py
        -> reads Hydra configs under robocub/config
        -> builds the Unitree G1 MJX environment
        -> runs Brax PPO
        -> writes checkpoints and JSONL metrics
        -> robocub/utils/training_artifacts.py
           -> writes progress.json, summary.json, graph, and final video
```

## Sweep data flow

```text
training_hyperparameters.json ----\
                                  -> robocub.user_cli/sweep.py -> numbered trials
environment_hyperparameters.json -/                         -> leaderboard
```

The JSON files contain values. Python contains only validation, the nested
loop, subprocess isolation, and ranking. Every trial uses the exact same
single-run pipeline, so sweep and manual results have the same folder shape.

## Visualization data flow

`App/Scripts/visualize.py` selects the field viewer. The viewer restores
`App/Models/T02/logs/checkpoints/1003520/policy`, constructs the MuJoCo scene, and
sends a fixed sequence of velocity commands to the policy. Field geometry is
visible, but the policy has no verified ball interaction or soccer strategy.

## Configuration boundaries

- `robocub/config/` controls the working MJX locomotion environment and PPO.
- `App/Configs/sweeps/` lists MJX values to compare in repeated runs.
- `App/Configs/field01.yaml` controls the separate Isaac Lab Field01 runtime.
- `App/Models/` contains selected policies; `Artifacts/Runs/` and `Artifacts/Sweeps/` contain outputs.

Keeping these boundaries explicit prevents a YAML file from appearing to
control a simulator that does not actually read it.
