# Triton Droids RoboCup Simulator

This project trains, evaluates, and visualizes Unitree G1 locomotion policies
for Triton Droids RoboCup work. The working locomotion path uses MuJoCo/MJX and
Brax PPO; the Isaac Lab Field01 soccer environment is a separate integration
track.

## Start Here

Read [`START_HERE.md`](START_HERE.md) for the shortest possible instructions.
The complete beginner manual is
[`Documentation/Guides/README.md`](Documentation/Guides/README.md).

```powershell
# Open the saved T02 walker on the visible RoboCup field.
.venv\Scripts\python.exe App\Scripts\visualize.py

# Verify PPO training, checkpointing, graphs, and run outputs.
.venv\Scripts\python.exe App\Scripts\train.py --smoke

# Run the example trainer × environment parameter matrix.
.venv\Scripts\python.exe App\Scripts\sweep.py --smoke

# Print the selected model and exact weights path.
.venv\Scripts\python.exe App\Scripts\status.py

# Evaluate the canonical T02 checkpoint.
.venv\Scripts\python.exe App\Scripts\evaluate.py
```

PyCharm exposes the same five operations as numbered shared run configurations.

## Project Layout

```text
App/             Things normal users run or edit
Artifacts/       Generated runs, sweeps, results, outputs, and exports
Documentation/   Beginner guides and preserved research records
robocub/         Production Python package and internal developer tools
tests/           Automated behavior and regression tests
examples/        Small example programs
```

Hidden folders such as `.cache/`, `.venv/`, `.run/`, and `.idea/` are local
infrastructure. See
[`Documentation/Guides/PROJECT_STRUCTURE.md`](Documentation/Guides/PROJECT_STRUCTURE.md)
for a detailed map.

## Models and Outputs

The selected model is:

```text
App/Models/T02/logs/checkpoints/1003520/policy
```

Single training runs are written under `Artifacts/Runs/`. Parameter sweeps are
written under `Artifacts/Sweeps/`. Historical research evidence is preserved
under `Artifacts/Results/` and `Documentation/Research/`.

The live field demo shows the learned walker inside visible field geometry. It
does not prove ball interaction, kicking, soccer strategy, Isaac transfer, or
physical-robot transfer.

## Sweep Configuration

Edit only these files to choose values:

```text
App/Configs/sweeps/training_hyperparameters.json
App/Configs/sweeps/environment_hyperparameters.json
```

The sweep runs every training row against every environment row and writes a
leaderboard plus a best-trial record.

## Setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe App\robocub.py doctor
```

The Unitree model is resolved from the pinned MuJoCo Menagerie checkout in
`.cache/mujoco_menagerie/`. Long useful PPO runs require a suitable
CUDA-enabled JAX environment; the validated native Windows path uses CPU.

## Testing

```powershell
.venv\Scripts\python.exe -m pytest -q
```

`pytest.ini` limits discovery to the active tests so copied research snapshots
under `Artifacts/Results/` are never imported as duplicate tests.

## Research Boundary

Gate 4 has not passed. A corrected historical evaluation balanced for ten
seconds but stood still under a `0.5 m/s` command. Older normalization-enabled
evaluations that omitted preprocessing are invalid. Before spending a non-smoke
training budget, read
[`Documentation/Research/handoffs/ASTRA_HANDOFF.md`](Documentation/Research/handoffs/ASTRA_HANDOFF.md).

## Attribution

This repository includes Triton Droids simulator work and code derived from
`toddlerbot`. Unitree G1 assets come from MuJoCo Menagerie at the pinned
revision recorded by the model resolver. License and attribution details remain
in `LICENSE` and `NOTICE`.
