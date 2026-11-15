# Public controls

Run **App/Scripts/dashboard.py**. This is a Python GUI script, not an installed desktop application.

| Location | Role |
|---|---|
| `App/Scripts/dashboard.py` | Control panel: compute, status, results and viewer |
| `App/Scripts/visualize.py` | Saved T02 walker on the field |
| `App/Scripts/status.py` | Selected model and exact checkpoint path |
| `App/Scripts/evaluate.py` | Evaluate the selected T02 model |
| `App/Scripts/train.py` | Historical/example trainer; `--smoke` checks the pipeline |
| `App/Scripts/sweep.py` | Example parameter matrix, not the research campaign |
| `App/Scripts/cluster_dashboard.py` | Compatibility alias for the control panel |
| `App/robocub.py` | CLI, including `doctor` and `dashboard` |
| `Configs/sweeps/` | Example trainer and task/world bundles |
| `Configs/dashboard_plans.json` | Approved launch catalog; empty is valid |
| `Models/T02/` | Named demo policy and resolved configuration |

Read [Training](../Documentation/Guides/TRAINING.md) before launching experiments.
Changing example settings does not change an active frozen research plan.

Browse [named policies](../App/Policies/README.md) and [recordings](../Artifacts/Videos/README.md).
