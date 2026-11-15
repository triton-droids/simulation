# Project structure

| Directory | Purpose |
|---|---|
| `Scripts/` | Public launchers; dashboard.py is the GUI |
| `Scripts/examples/` | Small API demonstrations |
| `Scripts/legacy/` | Preserved remote utilities, not the active campaign |
| `Configs/` | Example world/trainer settings and approved launch catalog |
| `Policies/` | Named policy folders, provenance and validation status |
| `Artifacts/Runs/`, `Artifacts/Results/` | Original experiment outputs |
| `Artifacts/Videos/` | Video index; originals stay with their runs |
| `documentation/guides/` | Operator explanations |
| `documentation/research/` | CURRENT, CONTRACT, queues and source references |
| `documentation/research/archive/` | Historical reports, decisions and handoffs |
| `robocub/`, `tests/` | Implementation and regression checks |

T02 weights are physically in Policies/T02. Other named policy folders point to
their original experiment checkpoints to preserve evidence. Add a new policy using
the [policy template](../../Policies/TEMPLATE.md), never by overwriting an old one.

`.cache/` remains at the project root because its WSL environment/assets have path
dependencies. `.run/` remains where PyCharm discovers shared run configurations.
Both are excluded in the local IDE configuration; hide excluded files to unclutter
the view. Pytest now uses `.cache/pytest/`; earlier scratch files are preserved.
