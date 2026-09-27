# Unattended G1 experiments

Active queue: `research/queues/post_f1_b08.json`. Read only
`results/post_f1_b08/queue/status.json` first. Current authorization:
`research/POST_F1_CONTINUATION.md`. Hourly minimal checks; continue bounded
research until one frozen recipe passes three fresh independent seeds.
Diagnostic-only jobs may use `review_artifact` instead of a CSV gate;
they explicitly report numeric_gate_evaluated=false and require review.

The local runner uses Python's standard library and makes no model or network
calls. One queue owns an OS file lock; jobs and their training/evaluation stages
run sequentially on the single GPU. The first frozen queue is
`research/queues/c16_balance.json`. Its compact state is
`results/gate4_corrective/C16_queue/status.json`.
The hourly same-task heartbeat is `g1-experiment-batch-review` (active).
Initial launcher preflight exposed WSL's default CRLF mismatch. The runner now
uses `git -c core.autocrlf=true diff --quiet HEAD` plus an untracked-file check;
this also avoids stat-cache-only dirty reports from `git status`.
an end-to-end CRLF fixture guards against recurrence. No training occurred in
that failed preflight. Its launcher log is preserved; launch uses a new log.

## Run a frozen queue

From WSL in the project root:

```bash
.cache/g1_wsl_venv/bin/python source/scripts/run_g1_queue.py --plan research/queues/c16_balance.json --validate-only
# Commit the runner, plan and research records before launching.
nohup .cache/g1_wsl_venv/bin/python -u source/scripts/run_g1_queue.py --plan research/queues/c16_balance.json > results/gate4_corrective/C16_queue_launcher.log 2>&1 < /dev/null &
```

The launcher and children survive the invoking shell. The computer/WSL must
remain running. The app must remain open for the scheduled model follow-up.
The runner itself does not depend on an active model turn.

## Frozen behavior

- Save exact plan bytes, SHA-256, clean Git revision, process IDs and timestamps.
- Refuse existing queue/run/evaluation destinations; never delete evidence.
- Stop on nonzero exit, missing expected output, wrong final training step,
  nonfinite training metrics, or exceeded stage/queue wall-time budget.
- C16 ignores KL only through the first positive callback (481,280 steps).
  Two consecutive subsequent reported KL values >=.2 stop the run. Initial
  nonfinite values still stop immediately. This operationalizes "sustained".
- Allow cold imports and compilation within a generous two-hour training cap,
  one-hour evaluation cap and three-hour overall cap. These are upper bounds,
  not estimates or permission for additional training steps.
- Check every declared matched control/command/reset row exactly once before
  scoring the gate. Report individual rules, mean survival and tracking errors.
- Numeric rejection advances only to another independently predeclared job in
  the same queue. Infrastructure/numerical errors stop the whole queue.
- No automatic reward-based winner selection, experiment invention, repeated
  retries, parameter changes, or claims of final research success.
- Visual review is mandatory before promoting a candidate or extending C16.
  Extension eligibility is reported separately and does not launch an extension.
- A queue-level `STOP` file (or job-level `STOP`) terminates the active child
  process group within the local polling interval, then records an error.
- Do not edit/commit research code or plans while a queue is running: revision
  or tracked changes stop progression before the next stage.

## Model wake protocol

Read the active `status.json` first. While status is running and the heartbeat
is fresh, end the wake quietly; do not read full logs or poll in an active turn.
If its heartbeat is older than five minutes, inspect the saved runner/child
PIDs and launcher log once: host suspend, lost runner or I/O failure can leave
stale state. A stale heartbeat alone is not permission to start a duplicate.

On error, inspect only the relevant stage log and metrics. Diagnose and fix the
cause, preserve failed artifacts, and use new destinations for any justified
replacement. Never rerun training simply to recover an evaluation failure;
evaluate its completed checkpoint in a new evaluation directory.

On needs_review, follow research/POST_F1_CONTINUATION.md. Review compact
numeric results and relevant gait videos, then freeze and launch the cheapest
justified next batch. Retain failures. Update the scheduled status path.
Pause only on validated three-seed success or user stop; report genuine blockers.

For C16 specifically, first inspect all four stand/forward episodes and the
trained forward video. The declared extension remains capped at 1,648,640 steps
and requires its numeric eligibility plus behavior review. A balance pass
permits designing the separate gait-shaping stage; it is not locomotion success.
The original independent multi-seed and fresh held-out Gate-4 requirements remain.

## Tests

`python -m unittest tests.test_g1_queue -v` is dependency-free. Linux/WSL runs
all process-group tests; other hosts skip those tests. Coverage includes actual
child failure, timeout/termination, missing output, no-overwrite, incomplete
metric writes, KL transients, nonfinite values, malformed evaluation sets,
strict height gates and separating extension eligibility from success.

## Staged development screening
Screening jobs declare screening_only=true. Dependent jobs declare
requires_screening with earlier screen IDs; only screening_passed unlocks them.
Rejected screens skip dependent evaluation, preserve evidence, and cannot count
as full validation. Missing/nonfinite evidence remains an execution error.
See research/PIPELINE_AUDIT_A01.md for the frozen screen and strategy.
