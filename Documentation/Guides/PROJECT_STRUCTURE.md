# Project Structure

A normal user does not need to understand the entire repository. The project
now has six meaningful top-level locations:

```text
App/             Things normal users run or edit
Artifacts/       Generated runs, sweeps, results, outputs, and exports
Documentation/   Beginner guides and preserved research records
robocub/         Production Python package and internal tools
tests/           Automated behavior and regression tests
examples/        Small example programs
```

## Public Interface

```text
App/Scripts/          Watch, train, sweep, evaluate, and status buttons
App/Configs/sweeps/   JSON lists of parameter values to test
App/Models/           Named policies selected for normal use
Artifacts/Runs/       Output from individual training runs
Artifacts/Sweeps/     Output and leaderboards from parameter sweeps
Documentation/Guides/ Human instructions
```

## Internal and Historical Locations

| Folder | Why it exists | Normal users edit it? |
|---|---|---|
| `robocub/` | Simulator, PPO, robots, rewards, soccer code, and tools | No |
| `robocub/user_cli/` | Command orchestration behind `App/Scripts/` | No |
| `robocub/tools/` | Specialist and legacy developer utilities | No |
| `robocub/soccer/` | Soccer and Field01 implementation | Only its developers |
| `Documentation/Research/` | Experiment evidence, decisions, and audits | No |
| `tests/` | Automated regression checks | No |
| `.cache/` | Downloaded assets and compilation caches | No |
| `.venv/` | Installed Python environment | No |
| `.run/` | Shared PyCharm button definitions | Usually no |

PyCharm excludes `Artifacts/` and cache folders from indexing and its normal
Project view. If they remain visible, turn off **Show Excluded Files** in the
Project panel. The files remain on disk; no evidence is deleted.

## Sweep Values

```text
App/Configs/sweeps/
|-- training_hyperparameters.json
`-- environment_hyperparameters.json
```

Each JSON object is one value bundle. The sweep runs every training bundle
against every environment bundle. Python contains only validation, the nested
loop, isolated process launching, and leaderboard generation.

`pytest.ini` prevents copied source snapshots under `Artifacts/Results/` from
being mistaken for active tests. Without that boundary, Pytest imports duplicate
modules with identical names and fails during collection.
