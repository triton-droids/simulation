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

## Policies and saved checkpoints

Run `Scripts/dashboard.py` in PyCharm. The first tab is **Policies & saved checkpoints**.

1. Choose a friendly policy name. Stable research IDs and original folders are preserved.
2. Select a checkpoint row. Saved step is the step in that run; total trained steps adds a recorded parent where available.
3. **Load checkpoint** selects and checks its files. Weights are deserialized when you click **Watch loaded checkpoint**, which opens that exact checkpoint on the field using local CPU inference. The compute selector applies to future training, not this viewer.
4. **Open checkpoint files** opens the selected checkpoint folder.
5. **Prepare training selection** writes a new draft under `Artifacts/Results/dashboard_drafts`. It does not train, resume, alter a policy, or change the schedule. A reviewed recipe and budget and explicit resume authorization are still required.

Times are UTC. “Checkpoint commit” comes from saved Orbax metadata; “File modified” is only a filesystem timestamp and may reflect a transfer. Missing files are shown explicitly. The latest row is highlighted for convenience, not promoted as a scientific winner.

Loading or viewing does not count as validation. These saved weights do not contain a complete optimizer/PRNG/environment resume state. Training remains paused; the dashboard rejects launches while the user pause marker exists.

## Policy-first workspace

`Scripts/dashboard.py` now opens an expandable policy/checkpoint tree. Policy rows show their latest checkpoint step and UTC timestamp; children expose older checkpoints. Selecting a row selects that exact checkpoint for viewing and recipe preparation. Selection is not scientific promotion.

Choose compute, select a policy/checkpoint, then choose **New training** (fresh weights) or **Continue weights** (parameter warm start, not exact optimizer/PRNG resume). The basic form edits environment count, seed, additional steps, learning rate and wall-time budget. Advanced settings expose batch size, minibatches, rollout length, checkpoint/evaluation count, discount, entropy, CPU and RAM. **Load checkpoint settings** restores the saved hyperparameters while resetting the requested new step budget to 1,003,520. **Import recipe JSON** can load a full resolved configuration or exported recipe. Continuing weights requires compatible architecture and normalization. Video recording is optional.

**Save training recipe** creates immutable new recipe/plan/registration JSON files in `Configs/training_recipes/<unique-id>/`. No run is started. **Register reviewed recipe** adds its exact plan hash to the launch catalog; commit the reviewed recipe and source before submission, as required by the existing isolated-run backend. **Launch saved recipe** uses those frozen bytes (not unsaved form edits), confirms the destination, and uses the existing local WSL or Atlantis launcher. The training pause marker blocks launch and is never removed by this UI. Saved jobs have explicit time limits and fresh output paths. These plans perform development training only; they do not claim or replace the full campaign validation gates.

For Atlantis continuation, the submitter stages only the selected, hash-bound checkpoint files inside the new isolated snapshot. It does not mutate shared source or checkpoints. An incomplete submission remains subject to existing reconciliation/no-duplicate rules. Remote launch was not exercised during this UI change.

**Open results** shows local saved outputs of the selected policy, or the selected historical run when opened from Run history. JSON/CSV/JSONL/logs display inside the panel; large text is limited to the last 150 KB. Images preview inside the panel. Videos support play/pause and scrubbing using OpenCV/Pillow. **Recordings** opens the existing library index inside the same panel, with original experiment paths clearly shown. These global videos are not attributed to the currently selected policy. Remote-only recordings need downloading before preview; no automatic SSH transfers occur on selection. The live MuJoCo viewer still opens its own interactive window. **Show checkpoint files** is the optional File Explorer action.
