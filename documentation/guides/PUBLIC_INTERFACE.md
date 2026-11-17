# Public controls

Run **Scripts/dashboard.py**. This is a Python GUI script, not an installed desktop application.

| Location | Role |
|---|---|
| `Scripts/dashboard.py` | Control panel: compute, status, results and viewer |
| `Scripts/visualize.py` | Saved T02 walker on the field |
| `Scripts/status.py` | Selected model and exact checkpoint path |
| `Scripts/evaluate.py` | Evaluate the selected T02 model |
| `Scripts/train.py` | Historical/example trainer; `--smoke` checks the pipeline |
| `Scripts/sweep.py` | Example parameter matrix, not the research campaign |
| `Scripts/cluster_dashboard.py` | Compatibility alias for the control panel |
| `Scripts/robocub_cli.py` | CLI, including `doctor` and `dashboard` |
| `Configs/sweeps/` | Example trainer and task/world bundles |
| `Configs/dashboard_plans.json` | Approved launch catalog; empty is valid |
| `Policies/g1-walking-demo/` | Named demo policy and resolved configuration |

Read [Training](TRAINING.md) before launching experiments.
Changing example settings does not change an active frozen research plan.

Browse [named policies](../../Policies/README.md) and [recordings](../../Artifacts/Videos/README.md).
