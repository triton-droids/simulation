# Start Here

> Full plain-English manual: [`Documentation/Guides/README.md`](Documentation/Guides/README.md)

This repository follows the same public-script pattern as Pupper. Normal users
only need the five files in `App/Scripts/`.

## I want to see the robot walk

```powershell
.venv\Scripts\python.exe Scripts\visualize.py
```

The first launch can spend roughly a minute compiling. A separate window named
`MuJoCo : g1 scene` then opens. If the console prints commands but no camera is
visible, use Alt-Tab and select that MuJoCo window.

This uses the saved T02 walking policy. The field, goals, and ball are visual in
this demo; soccer control and kicking are not implemented.

## I want to verify training works

```powershell
.venv\Scripts\python.exe Scripts\train.py --smoke
```

This is a 1,024-step integration run, not a useful trained policy.

Each run automatically writes its checkpoints, metrics, final graph, summary,
and attempted rollout video into a new folder under `Artifacts/Runs/`.

## I want to compare settings automatically

```powershell
.venv\Scripts\python.exe Scripts\sweep.py --smoke
```

Edit `App/Configs/sweeps/training_hyperparameters.json` and
`App/Configs/sweeps/environment_hyperparameters.json`. Results, a leaderboard, and
the best-trial record are written under `Artifacts/Sweeps/`.

Full historical-profile training is explicit:

```powershell
.venv\Scripts\python.exe Scripts\train.py
```

That profile is not a validated three-seed locomotion recipe.

## I want to know what policy is active

```powershell
.venv\Scripts\python.exe Scripts\status.py
```

The selected weights are stored at:

```text
App/Models/T02/logs/checkpoints/1003520/policy
```

The matching resolved configuration is:

```text
App/Models/T02/resolved_config.json
```

## PyCharm

Use the checked-in run configurations in the top toolbar:

1. `1 Watch Robot`
2. `2 Train Smoke`
3. `3 Show Status`
4. `4 Evaluate T02`
5. `5 Sweep Smoke`

Internal implementation remains under `robocub/`, `robocub/tools/`, and
`robocub.soccer/`. Do not run files in `robocub/tools/` unless a development document
specifically asks for one.

For setup, training values, rewards, field limitations, weights, or common
failures, use the linked manual instead of guessing from internal scripts.
