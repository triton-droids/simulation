# RoboCup Documentation

This folder is the operator manual for the Triton Droids RoboCup simulator.
Start with the row matching what you want to do; you do not need to understand
the files under `robocub/` or `robocub/tools/` to use the normal workflows.

| Goal | Read this | Run this |
|---|---|---|
| See the saved robot walk on a field | [Visualization](VISUALIZATION.md) | `.venv\Scripts\python.exe Scripts\visualize.py` |
| Check whether the installation is usable | [Setup](SETUP.md) | `.venv\Scripts\python.exe robocub.py doctor` |
| Find the active weights | [Models and outputs](OUTPUTS_AND_MODELS.md) | `.venv\Scripts\python.exe Scripts\status.py` |
| Verify the training pipeline | [Training](TRAINING.md) | `.venv\Scripts\python.exe Scripts\train.py --smoke` |
| Sweep trainer and environment values | [Training](TRAINING.md#hyperparameter-and-environment-sweeps) | `.venv\Scripts\python.exe Scripts\sweep.py --smoke` |
| Understand a reward value | [Rewards reference](REWARDS_REFERENCE.md) | No command required |
| Understand a PPO value | [PPO reference](PPO_REFERENCE.md) | No command required |
| Understand the field simulator | [Field01](FIELD01.md) | See the capability table first |
| Fix a failure | [Troubleshooting](TROUBLESHOOTING.md) | Start with `robocub.py doctor` |
| Understand the folders | [Project structure](PROJECT_STRUCTURE.md) | No command required |
| Understand the architecture | [Architecture for humans](ARCHITECTURE_FOR_HUMANS.md) | No command required |

## The public scripts

Only these files are intended as the normal push-button entry points:

- `App/Scripts/visualize.py` opens the saved T02 policy in a live MuJoCo field view.
- `App/Scripts/train.py` starts PPO training; `--smoke` is the safe pipeline check.
- `App/Scripts/status.py` prints the active model and exact weights path.
- `App/Scripts/evaluate.py` evaluates the saved T02 checkpoint.
- `App/Scripts/sweep.py` runs a trainer × environment matrix and ranks the trials.

Everything in `robocub/tools/` is an internal utility or preserved specialist workflow.
Everything in `robocub/` is implementation code. Everything in `Documentation/Research/` is
experimental evidence and history, not a beginner run menu.

## Important truth boundary

The live field viewer places the learned MuJoCo walker inside visible field
geometry. The field, goals, and ball are visual. It does not prove kicking,
ball interaction, soccer strategy, hardware transfer, or transfer into the
Isaac Lab Field01 environment.
