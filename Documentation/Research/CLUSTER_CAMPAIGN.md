# Atlantis Milestone 2 continuation â€” authorized 2026-11-07

The user explicitly authorized bounded background research until one frozen recipe
passes the original gates on three independent fresh training seeds. This supersedes
the September M1-only stop instructions. M1/T02 and all failed M2 evidence stay intact.
Soccer skills and hardware are not part of this campaign. Training uses MJX/Brax and
the pinned Playground research adapter; the Isaac field is a separate integration.

## Active batch RC01 — matched late objective (2026-11-08)

Plans: `Documentation/Research/queues/cluster_rc01_control.json` and
`Documentation/Research/queues/cluster_rc01_rate.json`.
Receipts: `Artifacts/Results/cluster_research/RC01_control/submission.json` and
`Artifacts/Results/cluster_research/RC01_rate_r1/submission.json`.
Monitor each once with `cluster_research.py status --run RC01_control` / `RC01_rate`.
Review BOTH arms together; a finished arm does not authorize advancing while its
matched partner is still running. At most two concurrent one-GPU jobs.

## Completed baseline RB01r1

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

## RB01r1 review and RC01 decision — 2026-11-08

Job7522 completed in1h01m25s; training stage52m51s, final20,070,400steps.
Reported final training throughput7638.7steps/s (excludes compilation/evaluation).
Both screens rejected; all8 nominal plus8 random trained episodes survived500steps.
Nominal min pelvis.74589m, mean linear.13696, mean yaw.22661; worst standing
linear.14945 passes, yaw.28612 fails. Random min pelvis.69664m, mean linear.22083,
worst.32548 pass; mean yaw.26149 and worst.35851 fail. Full evaluations correctly
skipped. Twelve distributed frames across nominal/random forward videos show
upright alternating stepping without visible collapse; this is sampled review,
not full gait certification. Metrics and videos saved under RB01r1/review locally.
Learning progressed through all8 callbacks, ended at declared final checkpoint;
final reported KL.06466 is below the frozen sustained-KL stop threshold.
Verdict: recovery reproduced, standing/yaw limitation persists. Not a full pass.

Next cheapest discriminating experiment: isolate late action-rate shaping after
recovery acquisition. RC01_control and RC01_rate both warm-start the SAME verified
RB01r1 final checkpoint, seed11, lr1e-4, 1,003,520steps each; only action_rate0 versus
-.1 differs. Parent policy SHA256 cf6f488e37d49016dde7ece8dce8ec0b15ad96d1c6a95ee28a33c32305d3db23
is checked before training; actor/normalizer restoration is checked afterward.
These are parameter continuations with optimizer restart, NOT exact resumes or
independent training seeds. Matched control accounts for extra steps/LR restart.
No phase/air-time masks, inference filters, or further coefficient values are added.
T02 motivated this timing test but had different prior training history; closed
B08/B09 from-initialization failures are not repeated or relabeled as new evidence.
Original baseline checkpoint0 supplies true untrained control, not warm-started
candidate checkpoint0. Nominal video now shows standing, random video forward.
All original numeric screen/full gates remain unchanged. Each queue4h total,
training1h, same max8h one-GPU Slurm allocation; only2,007,040 new steps combined.
Decision: compare all rows of BOTH arms. Promote only if both screens and full
numeric/control/visual evaluation pass without recovery or moving-gait regression.
If neither passes, close this late-shaping comparison: no coefficient escalation,
unchanged extension or reward-based winner selection. Reassess a structural or
training-distribution hypothesis from measured failures before another bounded batch.
Final3-seed confirmation still requires a separately frozen from-initialization
recipe for every component; these two dependent development arms cannot count.
Plan validation and exact argv comparison passed: only intended coefficient differs
apart from paths. No runner logic was changed for this experiment.

RC01_rate job7526 failed before training with nvidia-smi reporting GPU0 device
handle Unknown Error / No devices found. Evidence retained. One fresh allocation
RC01_rate_r1 is authorized as infrastructure recovery with identical plan; no
training completed in the failed allocation. RC01_control job7524 is already running
and remains untouched. If allocation failures recur, investigate scheduler/device
health rather than cycling allocations indefinitely.

RC01_rate_r1 job7528 repeated the identical GPU PCI0000:B1:00.0 failure in1s.
Stop allocation retries. Rate arm is infrastructure-blocked pending a healthy GPU
assignment/admin repair; not scientific rejection. Control7524 was still RUNNING
on zixian, allocated GPU index3. Do not cancel it or promote it without its matched
arm. Next wake: compact checks only while control healthy; once complete preserve
its results and diagnose available healthy allocation without changing recipe or
masking out devices outside Slurm permissions. Notify user once about faulty GPU;
no repeated unchanged-blocker notifications.
