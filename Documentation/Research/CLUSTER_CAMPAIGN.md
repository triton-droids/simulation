# Atlantis Milestone 2 continuation â€” authorized 2026-11-07

The user explicitly authorized bounded background research until one frozen recipe
passes the original gates on three independent fresh training seeds. This supersedes
the September M1-only stop instructions. M1/T02 and all failed M2 evidence stay intact.
Soccer skills and hardware are not part of this campaign. Training uses MJX/Brax and
the pinned Playground research adapter; the Isaac field is a separate integration.

## Active baseline RB01r1

Plan: `Documentation/Research/queues/cluster_rb01.json`.
Launch receipt: `Artifacts/Results/cluster_research/RB01r1/submission.json`.
Monitor: `.venv/Scripts/python.exe robocub/tools/cluster_research.py status --run RB01r1`.
The receipt names the isolated remote snapshot, job ID and queue output directory.

Hypothesis: re-establish B01's demonstrated recovery on Atlantis after migration,
measure actual cost, and determine whether its standing/yaw limitation persists.
The recent four-trial cluster smoke used the legacy environment; it is not this
baseline and provides no evidence of a useful learned controller.

Frozen recipe is preserved B01: development seed 11, from scratch, 512 environments,
20,070,400 steps, original reward, learning rate .0003, gamma .97, original random
resets, no sensor noise/push/domain randomization. Source/config and final checkpoint
are fixed before launch. This is a reproduction/capacity baseline, not a new causal
comparison and not independent validation. Fetching pinned Playground is enabled
in the private snapshot because the shared cluster cache lacks it.

Resource bound: one RTX2080Ti, 8 CPU cores, 32GB, long partition, 8h Slurm limit.
Existing queue bound is 6h, training stage 4h, evaluation stages 2h each subject to
the overall cap. No automatic unchanged training extension. Record actual runtime,
throughput, failures and checkpoint artifacts; a parameter warm start is not an
optimizer/PRNG-exact resume. Completed training must never repeat for evaluation recovery.

The inherited 48-row nominal/randomized screen runs automatically with matched
controls and videos. Its unchanged fail-closed rules condition full evaluations.
Partial screens, reward ranking, completed jobs, and three seed numbers are never
a success verdict. Original control-superiority, survival, height, tracking,
standing, collision/termination and visual gait criteria remain mandatory.

## Decision sequence

1. Review RB01 learning curve, full screen rows and videos. Check parity and actual
   compute cost before any scaling. If execution fails, repair only its cause in a
   new snapshot/output path; preserve the failure and avoid duplicate live jobs.
2. Freeze at most two evidence-justified alternatives against this baseline. State
   the measured weakness, one intended change, training budget, final checkpoint,
   unchanged evaluation gates and rejection rule before launch. No blind sweeps.
   Initially at most two concurrent one-GPU jobs, each <=20,070,400 steps / 8 GPU-h;
   fit resource/time budgets using the measured baseline. Review after this batch.
3. If no gain, diagnose the failure and close unsuccessful hypotheses before a new
   bounded batch. Do not resume the old endless coefficient/mask tweaks. The user
   authorizes further justified batches, but every batch needs a recorded decision.
4. Only after genuine full development numeric/control/visual success freeze the
   complete training recipe, three predetermined unused training seeds and a fresh
   held-out command/reset grid before any final runs. Verify freshness against all
   historical records. Train every component independently from initialization;
   never share a trained ancestor or select/replace lucky seeds/checkpoints.
5. All three must pass full numeric/control gates and gait-video review. Report
   every seed and mean/sample standard deviation. On success package the recipe,
   weights, provenance, limits and report; notify once and pause the automation.
   Failed final validation becomes development data; choose a newly frozen test
   only after a separately justified development change, never just swap seeds.

## Low-token supervision

Every four hours, read the launch receipt and request compact status once. A queued
or healthy running job requires no logs, polling or active model waiting. Slurm and
the Python queue perform all training/evaluation between checks. A stale runner
heartbeat is not proof the job is dead: reconcile squeue/sacct first. Missing receipt
or ambiguous submission: inspect that isolated run directory and named Slurm jobs,
never submit blindly. On connection failure retain the job and retry next wake;
notify only a new persistent blocker. No training runs on the login node.

Use the isolated submitter, not Scripts/sweep.sh: the latter stages all changes,
pushes GitHub and resets a reusable checkout. The research submitter instead uploads
a clean committed source archive, rejects private/local archive members, initializes
an immutable per-job snapshot and reuses the existing Linux Python environment.
Record package versions per allocation. Only pinned model cache is linked; Playground
and compiler caches are private. Do not change a running snapshot or its Python
dependencies. No GitHub push, hardware access or credentials in uploads.

Keep notification quiet for unchanged state. Notify meaningful findings, a blocker
requiring attention, or validated completion. A timer cannot guarantee discovery;
scientific failures must be reported honestly without weakening acceptance criteria.

## Startup repair — 2026-11-08

RB01 job7501 failed in 3 seconds before the queue or training started: the uv-created
remote environment has no pip. GPU discovery passed. Preserve its isolated directory
and local launcher_failure.log. RB01r1 keeps the same frozen queue/recipe/budget,
but records installed packages using Python importlib.metadata instead of pip freeze.
This is an execution repair, not a scientific failure or training extension.
