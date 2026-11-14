# RoboCup control center

Run `App/Scripts/dashboard.py` with the project interpreter, select **6 RoboCup
Dashboard** in PyCharm, or use `.venv/Scripts/python.exe App/robocub.py dashboard`.
The older `App/Scripts/cluster_dashboard.py` button opens this interface too.
The old sweep dashboard implementation remains available for historical work;
the new interface never invokes `Scripts/sweep.sh`.

Select the laptop or Atlantis for the **next** launch. Selection never migrates
an active job. **Check compute** verifies local NVIDIA visibility and WSL JAX,
or queries Atlantis's scheduler. Scheduler availability is not GPU runtime health.
Every check is timestamped. Unknown, failed and stale states are explicit.
Saved local progress refreshes every 30 seconds without SSH calls. Use
**Check selected Slurm job** for current scheduler evidence, shown in Advanced.

**Watch robot on field** uses the existing T02 viewer. **Open results** opens
the selected run's artifacts. **Inspect selected** shows its saved status or
receipt. **Stop selected** asks for confirmation, then requests local queue STOP
or Slurm cancellation. It retains saved checkpoints, not unsaved progress or an
exact resumable optimizer state. A stale local run must be reconciled first.

## Frozen launch catalog

`App/Configs/dashboard_plans.json` is deliberately empty initially: completed
LP01 must not be rerun, and the full recovery plans have not yet been registered.
The disabled launch button explains this. Register a reviewed, committed plan:

```json
{"version":1,"plans":[{"id":"EXAMPLE","source":"local","enabled":true,
 "plan":"Documentation/Research/queues/EXAMPLE.json","sha256":"EXACT_SHA256"}]}
```

Atlantis entries also require `resources` with `gpu_type` (`rtx2080ti` or `p100`),
`cpus`, `memory_gb`, and `hours`. Use only the existing authorized research budget.
The existing snapshot launcher performs schema/resource/clean-commit checks.
Local launch uses Ubuntu WSL and `.cache/g1_wsl_venv/bin/python`, the queue's lock,
private compiler cache and explicit reproducible profile. A changed hash, existing
output, or prior launch attempt blocks resubmission. Failure requires review and
a fresh path; changing the dropdown does not authorize replaying a finished run.

## Additional compute providers

Add a provider backend with explicit health, launch, status and stop operations,
then expose its ID in the selector and catalog validation. Do not map an unknown
provider to Atlantis or claim readiness from SSH connectivity alone. Credentials
stay outside the catalog in the existing ignored environment configuration.
Cloud purchasing, automatic GPU discovery/registration, checkpoint migration and
arbitrary host provisioning are not implemented by this dashboard.
