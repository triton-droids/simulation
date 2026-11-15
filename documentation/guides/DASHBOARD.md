# RoboCup control panel

Run **Scripts/dashboard.py**, or choose **6 RoboCup Dashboard** in PyCharm.

```powershell
.venv\Scripts\python.exe Scripts\dashboard.py
```

`Scripts/cluster_dashboard.py` is a compatibility alias for the same panel.

| Control | Behavior |
|---|---|
| Compute selector | Chooses a future launch source; never moves active work |
| Check compute | Timestamped local NVIDIA/WSL JAX or remote scheduler check; scheduler availability is not GPU health |
| Refresh saved progress | Reads saved statuses; local display also refreshes every 30 seconds |
| Check selected Slurm job | Explicit scheduler query; saved remote status may be stale |
| Inspect selected / Open results | Show saved status/receipt or open output artifacts |
| Watch robot on field | Saved T02 demo, not the latest selected training checkpoint |
| Start approved plan | Registered reviewed plan into fresh output paths |
| Stop selected | Confirmation then local queue STOP or Slurm cancellation |

Saved checkpoints survive stopping; unsaved progress may not. Stop is not exact
pause/resume. Serial batches also have a supervisor; consult the active contract.
The panel does not manage Codex scheduled reviews, migrate checkpoints, or resize
running allocations. Unknown/stale compute evidence must not be treated as ready.

## Disabled launch button

`Configs/dashboard_plans.json` may be empty: no new plan is registered.
Active/completed plans belong in status/history, never in the catalog for replay.
Maintainers register an approved unlaunched committed plan with `id`, `source`
(`local` or `atlantis`), `enabled`, project-relative `plan` and exact `sha256`.
Atlantis also needs `resources`: `gpu_type`, `cpus`, `memory_gb`, `hours`.
Changed hashes, existing evidence or prior attempts block replay. Reconcile
failures and use fresh outputs, not button retries.

Local launches use existing Ubuntu WSL, its GPU runtime, queue lock and private
cache. Additional providers require explicit health/launch/status/stop backends;
arbitrary provisioning and cloud purchasing are not implemented. This dashboard
never invokes the legacy `Scripts/legacy/sweep.sh`.
