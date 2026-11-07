# Models and Outputs

## Canonical Saved Model

```text
App/Models/
`-- T02/
    |-- resolved_config.json
    `-- logs/
        `-- checkpoints/
            `-- 1003520/
                `-- policy
```

`policy` is the exact file loaded by the public viewer and evaluator.
`resolved_config.json` records the matching training configuration. Run
`App/Scripts/status.py` instead of guessing which model is active.

`App/Models/T02/` is local and ignored by Git. A fresh clone does not
automatically contain the weights.

## Folder Meanings

| Folder | Meaning | Safe to delete? |
|---|---|---|
| `App/Models/` | Named models selected for normal use | No, unless backed up |
| `Artifacts/Runs/` | Individual runs with checkpoints, plots, and videos | Only unwanted runs |
| `Artifacts/Sweeps/` | Ranked parameter sweeps and trial folders | Only unwanted sweeps |
| `Artifacts/Results/` | Preserved historical experiment evidence | Only if no evidence is needed |
| `Artifacts/Outputs/` | Older Hydra-generated output | Usually generated; inspect first |
| `Artifacts/Exports/` | Generated upload archives | Yes |
| `Documentation/Research/` | Decisions, plans, reports, and audits | No |
| `.cache/` | Downloaded model/source and compilation caches | Yes; downloads may repeat |
| `.venv/` | Local Python environment | Yes; reinstall afterward |

## Checkpoints Are Not Videos

A checkpoint contains learned policy state, not a recording. Use
`App/Scripts/evaluate.py --video` for an evaluation video or
`App/Scripts/visualize.py` for interactive viewing.
