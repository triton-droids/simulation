# Triton Droids RoboCup

**Open the control panel:** run **App/Scripts/dashboard.py** in this project.
In PyCharm select **6 RoboCup Dashboard**, then Run.

```powershell
.venv\Scripts\python.exe App\Scripts\dashboard.py
```

The panel provides local/Atlantis compute checks, saved progress, results and the
saved robot viewer. A compute selection affects a future launch; it does not move
active work. Starting training requires a registered approved plan.

## What do you want to do?

| Goal | Entry point | Guide |
|---|---|---|
| Open controls and inspect work | `App/Scripts/dashboard.py` | [Control panel](Documentation/DASHBOARD.md) |
| Watch the saved G1 on the field | `App/Scripts/visualize.py` | [Viewer](Documentation/Guides/VISUALIZATION.md) |
| Find the selected weights | `App/Scripts/status.py` | [Models and outputs](Documentation/Guides/OUTPUTS_AND_MODELS.md) |
| Check installation | `App/robocub.py doctor` | [Setup](Documentation/Guides/SETUP.md) |
| Understand training/settings | Read before launching | [Training](Documentation/Guides/TRAINING.md) |

Start with [START_HERE.md](START_HERE.md). The [manual index](Documentation/Guides/README.md)
separates operator instructions from research evidence.

## Where things live

| Location | Purpose |
|---|---|
| [App](App/README.md) | Public buttons, example configs, selected models |
| `robocub/` | Simulator, trainer and developer utilities |
| `Artifacts/` | Generated checkpoints, metrics, videos and research results |
| `Documentation/Guides/` | Operator manual and parameter explanations |
| `Documentation/Research/` | Contracts, frozen plans and historical decisions |
| [Scripts](Scripts/README.md) | Legacy remote shell workflows, not the control panel |
| `tests/`, `examples/` | Regression checks and small developer examples |

See [Project structure](Documentation/Guides/PROJECT_STRUCTURE.md).

## What is demonstrated

T02 displays a learned G1 walker in a MuJoCo soccer-field scene. It does not
demonstrate kicking, soccer strategy, physical-robot or Isaac Lab transfer.
[Isaac Lab Field01](Documentation/Guides/FIELD01.md) is a separate integration track.
Three-seed certification requires all original numeric, control and visual gates;
a demo or reward leaderboard is not that certification.

Native Windows Python on this machine is not the CUDA training environment.
Do not launch example training/sweeps alongside the active local GPU batch.

Browse [named policies](App/Policies/README.md) and [recordings](Artifacts/Videos/README.md).
