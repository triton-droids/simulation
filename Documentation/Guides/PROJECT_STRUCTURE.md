# Project structure

Public controls live in `App/`, implementation in `robocub/`, generated evidence
in `Artifacts/`. Operator instructions are separate from research records.

```text
App/Scripts/dashboard.py  Main GUI launcher
START_HERE.md              One-page operator instructions
App/
  Scripts/                Dashboard, viewer, status, evaluation, example training
  Configs/                Example parameter bundles and approved launch catalog
  Policies/               Named policy records and provenance
  Models/                 Original demo weights (preserved compatibility path)
robocub/                  Implementation and internal developer utilities
Artifacts/
  Runs/                   Ordinary training outputs
  Sweeps/                 Example matrix outputs
  Results/                Frozen research runs, reviews and preserved failures
  Videos/                 Browsable recording index; originals stay with runs
  Exports/                Generated exports
Documentation/
  DASHBOARD.md            Control panel guide
  Guides/                 Operator manual and parameter explanations
  Research/               Contracts, frozen plans and experiment history
Scripts/                  Legacy remote shell workflows; see its README
tests/                    Regression tests
examples/                 Small developer examples
```

## Which settings should I edit?

- Example trainer bundles: `App/Configs/sweeps/training_hyperparameters.json`.
- Example world bundles: `App/Configs/sweeps/environment_hyperparameters.json`.
- Approved launch catalog: `App/Configs/dashboard_plans.json`.
- Research plans: `Documentation/Research/queues/`; never edit active plans or
  their dependencies. Example sweep settings do not override research queues.

Read [Training](TRAINING.md), [PPO](PPO_REFERENCE.md) and [Rewards](REWARDS_REFERENCE.md).
`.cache/`, `.venv/`, `.idea/` and `.run/` are local infrastructure. PyCharm's
Show Excluded Files can expose caches/results normally hidden from its project view.
Hiding them does not delete evidence. `pytest.ini` excludes old source snapshots
from ordinary test collection. Legacy files remain in place to preserve old paths.
