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

## Completed control / allocation blocker review - 2026-11-08 17:05 UTC

Control7524 completed in22m54s. Exact parent actor/normalizer restore passed.
All16 trained screen episodes survived500steps. Random screening now passes
(mean linear.19683/yaw.20679; worst.28223/.29954; min pelvis.71792m).
Nominal standing yaw.24764 remains above.15 despite standing linear.14372 passing;
full stages correctly skipped. Six frames each from standing and random forward
videos show upright in-place stepping and forward alternating gait, without visible
collapse. This is sampled review, not full validation. Saved all row-level metrics,
restore audit and videos under RC01_control/review. No winning comparison can be
made until the frozen rate arm actually trains. Do not repeat the completed control.

No account jobs remain active. Scheduler still advertises zixian as MIXED with
8 RTX2080Ti, no drain reason; BootTime2026-11-06T02:24:39, SlurmdStartTime
2026-11-06T02:25:05. Three GPUs are allocated to other work; these counts do not
identify healthy free devices. Neither failed allocation is evidence of a bad
recipe. Ordinary Slurm GPU-count requests do not identify a particular PCI device;
do not use GPU binding/visibility to access an unallocated GPU. No further blind
allocations submitted. Ask administrator to repair/exclude PCI0000:B1:00.0 from
zixian's allocatable GRES or provide an approved healthy allocation path.

Next scheduled wakes must not reread finished training logs/videos. Read local
cluster_recovery.json then a single scheduler health query. Unchanged boot/slurmd/
GRES/drain state with no user/admin repair evidence: quiet exit, no retries. Changed
health configuration or explicit repair confirmation permits one bounded GPU
preflight before a fresh identical rate-arm snapshot; keep all failed evidence.
Do not infer GPU repair merely from changing free/allocated counts. Continue once
usable allocation is established; preserve matched recipe and original gates.

## Reboot-triggered recovery - 2026-11-09

zixian BootTime changed to2026-11-08T18:54:29 and SlurmdStartTime to
18:54:56; GRES remains8 RTX2080Ti with no drain reason. One frozen preflight
job7532 (debug,oneGPU,2CPU,4GB,5min limit) completed in3seconds: nvidia-smi
and a JAX64x64 GPU matrix multiply passed on assigned PCI3D:00.0. This proves
a usable allocation, not repair of every GPU or specifically B1.
Authorize one fresh RC01_rate_r2 isolated submission with the unchanged
cluster_rc01_rate.json plan: same parent/hash,seed11,1,003,520steps,lr1e-4,
action_rate-.1,original screening/full gates and resource limits. Control7524
is already reviewed and must not repeat. Compare both arms after completion.
If the hardware fault recurs, preserve evidence and stop allocation retries.
No runner logic changed; existing verified plan and runner are reused.

## Compiler-cache recovery - 2026-11-09 09:08 UTC

RC01_rate_r2/job7534 failed after94seconds before training progress, during
initial evaluation reset compilation. XLA autotune cache insertion under the
shared source .cache directory raised Device or resource busy. This is distinct
from GPU discovery failure; preserve the remote stage log and local failure tail.
Small repair: cluster launcher creates a unique /tmp JAX cache per allocation
with mktemp, inherited by all stages. Recipe, parent, training budget and gates
remain identical. No shared cache deletion or dependency changes. Freeze one
fresh RC01_rate_r3 submission of the existing rate plan after shell/cache and
runner tests. Full training execution remains unverified until job results.
The completed control is untouched.

Validation: queue/archive tests14passed,5platform skips. Authorized remote
login-shell probe verified exact mktemp/export expression plus file creation and
atomic rename in /tmp. This is filesystem validation, not a GPU training pass.

## RC01 closed and RD01 diagnostic freeze - 2026-11-09
Rate job7536 completed22m44s (0.379 GPU-hours allocated). Cache repair worked;
actor/normalizer restoration exact. All16 screen episodes survive500. Nominal
standing yaw.19012 improves versus control.24764 but fails.15; linear.12497 passes.
Random screen passes:mean linear.20128/yaw.20059,worst.30185/.29807,height.72213.
Full evaluation skipped correctly. Six distributed frames each show upright
in-place stepping and alternating forward gait, no visible collapse. This is
sampled visual evidence, not full gait certification. Neither arm passes:
close action-rate comparison; no coefficient escalation or unchanged extension.

RD01 is zero training: four unmodified final-policy stand traces, BOTH matched
arms, nominal6000/random6002,500steps each. Existing diagnostic runner saves
actions,contacts,velocities and terminal causes. OneGPU,3h queue cap,40min/trace,
source fetch5min; unchanged Slurm8h outer limit. No new policy selection. Compare
whole/first100/last250 yaw and linear errors, support, height, action-difference
RMS and demeaned Hann yaw spectrum at50Hz. If BOTH nominal policies survive500,
height>.6,>50% settled yaw power above5Hz and action-difference RMS>.1, this
supports separately freezing a training-consistent temporal-control comparison
on this recovery-capable baseline; first reconcile historical T01 failures and
recovery-latency risks. Otherwise reject jitter justification and identify whether
settling or sustained stepping dominates before choosing a structural change.
Do not tune phase constants/reward coefficients or apply inference-only filters.
Short traces have no settled window. Partial diagnostics never constitute gates.
Command: cluster_research.py submit --run RD01 --plan
Documentation/Research/queues/cluster_rd01.json. No runner logic changed.

RD01/job7538 stopped before any trace: optional onnxruntime was imported even
for local-only policies. Move that import into the oracle-only branch; no package
installation or numerical changes. Preserve failure; fresh RD01r1 plan/output
keeps all four traces and budgets. No training repeated.

RD01r1/job7540 failed in24seconds before traces: local-only mode still hashed
an unused oracle ONNX asset. Complete isolation of oracle path lookup, checksum
and runtime initialization inside oracle mode. Local-only summary now reports
null oracle hash rather than implying verification. Preserve both failed traces.
Fresh RD01r2 uses unchanged four diagnostic trajectories and budgets; no training.
Verification:16 tests passed,5 platform skips; local-only skips oracle asset access and reference mode still rejects a wrong checksum. Fresh plan validation passed.

## RD01 evidence and RD02 freeze - 2026-11-09 21:08 UTC
RD01r2 completed5m14s (0.0872 GPU-h), all4 trajectories500steps, no terminal
causes. Saved summaries/traces and temporal_review.json locally. Nominal control
whole/first100/last250 yaw .20681/.20514/.20487; rate .16129/.16081/.16222.
Settled nominal support fractions .828/.852, heights .7516/.7548. Settled yaw
power above5Hz48.20%/55.51%, action differences .13376/.10012; dominant1.4Hz.
BOTH-arms prerequisite fails: no temporal-filter training authorized. Random
traces survive; late yaw .21767/.17930, linear .12270/.11032. Persistent stepping,
not mere settling. Historical T01 filter improved precision but lost recovery
(6/8 random survivors vs8/8 control); do not revive that failed filter branch.

RD02: zero-training command-discrimination diagnostic, same two final policies,
nominal original reset6000, commands(.05,0,0) and(.15,0,0),500steps each. Reuse
RD01 zero-command traces;4new trajectories,2000steps,3h queue cap,40min each.
No phase, observation, reward or action intervention. Compare mean signed forward
velocity and support/phase patterns, whole and last250, to corresponding zero.
Frozen rule: both policies must survive all tested commands,height>.6 and exhibit
late mean vx(.15)-vx(0)>=.05 with monotonic mean vx across0/.05/.15 to count as
clear low-speed command discrimination. Failure motivates a separate command
encoding/conditioning audit, not reward tuning. If met, command blindness is
unsupported; investigate why recognized zero commands retain periodic stepping
before freezing an architecture experiment. Correlation is diagnostic, not proof
of cause. No automatic training or additional speed sweep. Original full gates
and fresh three-seed criteria unchanged. No runner changes; validate frozen plan
and launch RD02 via cluster_research.py.

## RD02 reviewed; RD03 stopping diagnostic
RD02/job7544 completed5m24s (0.09 GPU-h). Both policies survive all commands
500steps,height>.749. Late vx at0/.05/.15: control-.03140/.02798/.14378;
rate .04827/.09981/.19635. Both monotonic, increments .17518/.14808 exceed.05.
Command-blindness hypothesis unsupported. Support stays .828-.860 (persistent
stepping). Saved reward audit shows standing phase rewards .6133/.6216 and
stand penalties-1.069/-1.103; association,not causal proof. Prior mask tests
remain closed; do not repeat reward tuning.
RD03 tests state dependence before choosing an architecture: each same final
policy nominal6000 starts at(.15,0,0), switches to zero BEFORE action250,
ends500. TWO trajectories,zero training,90min queue,40min each. Reuse RD02
constant.15 and RD01 zero traces. Saved command trace and command-relative RMSE
prevent mixing moving/standing scores. Synchronize state AND observation command.
If both survive,height>.6 and final150steps have yaw/linear<=.15 and support<.1,
standing acquisition is state-dependent and a separately frozen transition-training
hypothesis is justified. Otherwise close state-dependent stopping explanation;
structural stand/walk separation requires design review,not further speed/phase
sweeps. Neither outcome is a full gate pass. No inference filter or training here.
Validation: exact switch boundary and no-switch checks passed;13 regression tests passed,5 platform skips;plan validated. Existing command state/observation synchronization helper reused.

## User-directed parallel replications RP01 - Nov9 local time
User explicitly requested three independent runs after latest job and maximum
parallelization. RD03/job7548 completed3m04s;both survive500 but final150 stand
yaw .20915/.16050 and support .833/.847 fail frozen stopping rule. No transition
training justified. Persistent stepping remains; no development full pass.
RP01 is a user-directed bounded DEVELOPMENT replication,not final validation:
predetermined seeds101,202,303,each from initialization with exact RB01 original
20,070,400step stage then own-parent RC01-rate1,003,520step parameter continuation.
Total63,221,760 steps;3 independent oneGPU jobs submitted together (maximum ready
parallelism). Each7h queue/8h allocation limit; nominal expected~75min per seed
based on prior runs; Slurm availability determines actual concurrency. No shared
trained ancestor;each own baseline checkpoint0 controls. Original development
screens and conditional full suites unchanged. Seed11 failure remains evidence;
no lucky replacement,coefficient change,extension or early-checkpoint selection.
No claim of three-seed success from this batch; all results retained. Review all
three together and report distributions. If all full development gates pass,
still freeze fresh final seeds AND held-out tests before final validation.
Freshness: no101/202/303 matches found in prior frozen queue files before creating
these plans; now reserve them as development seeds.
Validation:all3 plans pass schema;asserted independent initialization,own parent
restore paths and own untrained reference controls. Runner logic unchanged.

Parallel scheduling policy now supersedes old two-alternative concurrency limit:
submit ALL ready independent jobs in each approved frozen bounded batch together,
oneGPU/job,never serialize separate seeds. Keep within-step train/eval dependencies
and matched reviews; no extra seeds or speculative experiments merely to fill GPUs.
Use Slurm allocation limits and existing per-job budgets;do not bypass assignments
or alter cluster configuration. Monitor all active receipts once per hourly wake.

RP01 jobs7559/7561/7563 submitted together. Account cap16CPU/96GB; another
user job reserves6CPU/40GB. Running7559 reserves8CPU/32GB. To fit all3 GPUs,
updated pending7561/7563 to1CPU/12GB each (Slurm accepted); recipe unchanged.
Prior baseline MaxRSS7472776K and late7472244K support12GB budget. Compilation
may slow with1CPU; retain limits and report performance confound. Original and
updated resources preserved in receipts. No shared configuration or running-job
changes. Initial memory suffix syntax rejected without submission; numeric MB
update accepted. Scheduler may take time to dispatch;do not duplicate.
Also lowered retained MinCPUsNode8 to1 on both pending jobs; accepted by Slurm, queue reason cleared to None. Dispatch remains scheduler-controlled.

## Seed303 infrastructure recovery - Nov9 local time
All3 allocations started concurrently. At03:15UTC seed101 completed base20,070,400
and entered late stage;202 healthy at11,468,800 base steps.303/job7563 failed13s
before training: mediapy import hit missing matplotlib stylelib/seaborn-v0_8-paper.
mplstyle. Failure tail preserved. Read-only follow-up finds asset present and
mediapy import succeeds (matplotlib3.11.1). Transient file availability suspected,
not proven; no packages modified. One identical fresh RP01_303r1 submission using
original seed303 plan/output paths inside NEW isolated snapshot. Preserve7563;
no seed substitution or scientific changes. Retain1CPU/12GB request for recovery
to fit concurrent account budget; other active jobs untouched. If repeats, stop
blind retries and diagnose shared environment stability. No runner changes needed.

## RP01 rejected; RE01 pre-continuation localization - Nov9 local time
All3 full recipes completed. Costs101:58m55s,202:87m17s,303:85m26s;
total3.8606 allocated GPU-hours plus13s failed303 allocation. Restore parity
passes all3. Nominal all survive;101/303 stand linear .00875/.00743,yaw
.02715/.02965 but forward error .44770/.44821 and support/airtime0 (no walking).
202 passes nominal incl stand .12770/.13659,forward error .15733,support .84.
All random screens fail:mean steps274.125/397/182,min40/67/41 respectively.
All full stages skipped. Six distributed frames per stand/random-forward clip
reviewed across all3:101/303 planted nominal posture and randomized collapse;
202 upright alternating forward gait on saved seed6000,not evidence all resets
pass. Sampled review only,not continuous final certification. No winner chosen.

RE01 freezes three PARALLEL evaluation-only jobs using EVERY saved base final
checkpoint20070400 (own untrained0 controls),original nominal/random screens
and videos. No training;do not evaluate/select intermediate checkpoints. Budget
2h queue each,oneGPU,3CPU,16GB each;all3 fit prior account headroom. Determine
whether base already lacks motion/recovery or late stage loses them. If base
fails the same checks,focus acquisition redesign;if base passes those checks
and late fails,focus retention during precision adaptation. Mixed outcomes
require per-seed accounting;no promoting a lucky seed or accepting base as final.
Closed recipe is not extended. Full original gates and fresh held-out3seed
validation remain mandatory. Source prep then2screens; frozen plans
cluster_re01_101/202/303.json. Added bounded CPU/memory CLI options so submissions
fit account caps immediately; defaults8/32 preserved. Invalid bounds rejected
before network. No active dependencies or cluster settings modified.

## RE01 rejected; RB02 matched acquisition batch-size hypothesis
RE01 base evaluations completed9m34/10m14/9m24 (0.4867 GPU-hours total).
101/303 already fail nominal walking at base:forward errors .44808/.44799,
support/airtimes0,standing yaw .03099/.03443.202 walks (error.14871,support.818)
but stand yaw.17471 fails. Random base survivors4/8,7/8,3/8 vs late4/8,6/8,2/8.
Thus acquisition failure precedes fine-tuning in101/303;202 also loses recovery
during late adaptation. No blanket retention-only explanation. Six sampled
frames from each new clip confirm planted nominal101/303 and randomized falls;
202 visible alternating gait. Saved all row metrics/videos;not full certification.

RB02 freezes ONE acquisition hypothesis across SAME development seeds101/202/303:
PPO batch_size128 instead of32, everything else exact original RB01 acquisition
recipe,including512env,16minibatches,4updates,lr3e-4,entropy.005,original rewards,
20,070,400steps from initialization. Larger on-policy batches may reduce noisy
updates and standing-only convergence;this is a hypothesis,not established cause.
It also reduces update frequency at fixed steps,so cannot attribute effects to
variance alone. No late fine-tune. Compare each to own saved RE01 base results;
no repeated controls/training.3 independent GPUs,3CPU/16GB each,4htraining/6hqueue
per job;60,211,200steps total. Final checkpoint only,unchanged screens and
conditional full suites/own untrained controls. Submit all3 together.
Decision: if101/303 fail to acquire moving gait or202 loses walking/recovery,
close larger-batch hypothesis,no batch-size sweep/unchanged extension. Progress
requires original nominal forward gait gates and no lower per-seed randomized
survival than4/7/3. This intermediate improvement is NOT full development pass.
Only full original numeric/control/visual development success allows freezing
new final seeds/held-out grid. Retain seed11 and all3 failures;no seed shopping.
Validation:argv differs ONLY seed/output/batch_size;128*16 divisible512 and
final step exact across7training evaluation intervals. No runner logic changed.
Monitoring now supports status --compact:preserves status/errors/stage/heartbeat/steps/Slurm,omits full gate arrays.17 tests passed,5 platform skips. Use compact checks to reduce routine tokens;full evidence stays on disk.

## RB02 rejected; RD04 shared disturbed-start failure diagnostic
RB02 jobs7603/7607/7605 completed61m58/50m35/50m45:2.7217 allocated
GPU-hours. All three nominal screens pass; randomized survivors6/8 each.
All failures are reset6001, stand/forward respectively:101=41/45 steps,
202=55/59,303=59/50. Other six trained random rows per seed survive500.
Full suites skipped. Seed202 regresses from7/8 to6/8, violating the frozen
no-survival-loss rule. Close larger-batch candidate; no unchanged extension
or batch-size sweep. Nominal acquisition improved but no full pass or winner.
Six sampled frames per nominal/random forward6000 video were inspected for
all3: upright stepping on these successful starts. Sampled review cannot
certify complete gait or explain reset6001 failure; evidence retained locally.

RD04 freezes three parallel ZERO-training jobs, one per saved RB02 final
checkpoint20070400, each original randomized reset6001 at stand and forward
0.45,500step caps. Six trajectories total; no interventions or baseline reruns.
Plans cluster_rd04_101/202/303.json contain executable argv; launch each with
cluster_research.py submit --run RD04_SEED --plan PLAN --cpus 3 --memory-gb 16.
OneGPU/job,3CPU/16GB,90min queue,40min/trace; launch all ready jobs together.
Compare terminal causes, earliest height/foot-separation/contact divergence,
first50 and preterminal20 samples. Short traces are not settled windows.
Decision: if scalar diagnostic reproduces all failures with a common physical
precursor, freeze one bounded mechanism-directed redesign; if causes differ,
retain per-seed accounting; if failures do not reproduce, reconcile scalar and
vector evaluation/reset contracts before training. No automatic retraining,
phase constants, filtering or reward coefficient sweeps. This does not relax
collision, termination or any final gates. No diagnostic counts as full pass.
Require full development success before new unused final seeds/held-out grid.
No runner logic changed; plans use existing terminal instrumentation.

RD04_202 upload timed out at60s before runner setup/submission; remote reconciliation
found only RD04_101/job7683 and RD04_303/job7681 active, no202 run script/job.
Preserve partial202 snapshot/receipt. Archive upload alone now allows180s;
all normal SSH/status timeouts unchanged. One fresh RD04_202r1 snapshot uses
identical frozen plan, no training repeated and active snapshots untouched.

## RD04 terminal evidence; RD05 causal disturbance diagnostic
All six original-reset traces terminate on foot-foot contact, with loss of height
BEFORE collision. Scalar steps stand/forward:101=41/45,202=55/69,303=60/56;
screen steps were41/45,55/59,59/50. All reproduce failure, not bitwise timing.
Pelvis below.6 first at33/37,45/58,48/45; foot horizontal separation below.1
first at41/45,55/68,59/55. Minimum heights.315/.313,.211/.167,.041/.120.
First20 mean forward velocity .55-.62 despite zero/.45 command. First50 mean
heights .652/.657,.692/.722,.686/.673; last20 .569/.573,.537/.505,.458/.499.
These shortened traces have no settled window. No nonfinite/inverted/shin
termination. Thus contact is terminal, not evidence that foot spacing caused
initial balance loss. RD04 costs2m44/2m54/2m44=0.1394 allocated GPU-hours.
Saved summaries/NPZ under each receipt/review; no new videos produced by this
numeric trace diagnostic. Prior RB02 video review remains sampled only.

RD05 freezes ONE zero-training causal ablation on all six matching cases:
remove ONLY original horizontal root velocity, retain sampled qpos, vertical/
angular/joint velocity and phase; existing assertions and recomputed physics/
observations verify intervention. Same fixed RB02 checkpoints,commands,6001,
500steps. Reuse saved RD04 full-reset baselines; no reruns or new seeds.
Historical C02/D05 showed horizontal sensitivity on6002; do NOT assume transfer
across these independently trained RB02 policies/reset6001. Do not repeat old
baseline/oracle audits. This six-case test checks transfer before any redesign.
Plans cluster_rd05_101/202/303.json; same submit command/resources asRD04,
3CPU/16GB/oneGPU each,90min queue,40min/trace,all three submitted together.
Frozen decision: only if ALL6 survive500 and keep pelvis>.6 does removal
support a shared horizontal-recovery explanation. Mixed outcomes retain every
case and reject a blanket explanation; no further velocity-component sweep.
Any redesign must reconcile prior failed C04-C06 pushes/short horizons and
P01 curriculum, not repeat them unchanged. No automatic training or inference
intervention. Original evaluation keeps full random velocity,scale1,candidates1;
modified-reset diagnostics NEVER count as development/final passes. All full
gates, independent fresh seeds/held-out criteria preserved. Existing runner
logic only; validate all plans and assert only ablation/output differ fromRD04.

## RD05 causal evidence; RB03 initialization-time return-credit test
RD05 all six horizontal-velocity ablations survive500 with minimum heights
stand/forward:101=.73057/.74123,202=.72572/.73333,303=.72427/.72077.
Linear RMSE:.12105/.11731,.13674/.11019,.13945/.12003; yaw RMSE
.13342/.18082,.15619/.15819,.18765/.21526. These are altered-reset diagnostics,
NOT full passes (including remaining tracking errors). All initial-state
assertions passed. Common horizontal-disturbance sensitivity is supported on
these six cases only, not proof of global mechanism or guaranteed redesign.
Cost3m54/3m44/3m44=0.1894 allocated GPU-hours. No gait videos generated by this
numeric diagnostic; saved summaries/traces retained under RD05 receipt/review.

Historical reconciliation: C04/C06 late push training and C05/T04 shortened
training horizons failed recovery, so no push/horizon repetition. P01 used
three short warm-start curriculum stages and failed; no curriculum extension.
T05 changed discount .97->.995 late in an already learned critic, retained all
survivors but failed nominal stand yaw .150995567; that TRANSFER test remains
closed. C10 .995 continuations also stalled; no late discount escalation.
RB03 instead freezes one from-INITIALIZATION return-credit acquisition test:
original RB02 recipe and batch128, but discount .995 throughout all20,070,400
steps, each of development101/202/303 independently initialized. No checkpoint
ancestor, optimizer restart or late adaptation. Hypothesis: learning the critic
and policy jointly with longer return credit can acquire braking/recovery before
the .8-1.4second collapse while retaining nominal gait. At50steps,.97^50=.218
versus.995^50=.778; this motivates but does not prove credit assignment. GAE,
finite unroll and critic accuracy still limit it; longer returns may destabilize.
This is NOT a novel parameter, nor permission for further discount variants.
Matched saved RB02 anchors avoid repeated controls/training; all3 seeds, no
selection. Full reset velocities/physics/observations/rewards/gates unchanged.
No phase/mask/filter/reward tuning. Total60,211,200steps,oneGPU/3CPU/16GB each,
all3 submitted together,4h training/6h queue each. Existing KL/nonfinite guards
and own checkpoint0 control retained. Final20070400 only; original nominal/random
screens and conditional full suites/videos unchanged. Plans cluster_rb03_SEED.json;
submit --run RB03_SEED --plan PLAN --cpus 3 --memory-gb 16.
Decision: require ALL3 retain nominal gates, at least6/8 random survival each,
and BOTH6001 commands survive500 per seed to establish the targeted recovery
gain. Any regression or missing recovery rejects this initialization hypothesis;
no gamma sweep or unchanged extension. Improvement is not full success: ALL
original full numeric/control/visual development gates required before unused
final independent seeds and held-out grid. No lucky checkpoint/seed promotion.
Validation: three argv comparisons assert discount/output only; all queue plans
validate. No runner logic changed. Estimated50-65min per run from RB02, so retain
hourly compact checks, no active model waiting for training.

## RB03 rejected; RD06 action-distribution diagnostic
RB03 jobs7697/7695/7693 completed50m35/51m25/49m04 =2.5178 GPU-hours.
Nominal survivors101/202/303=4/8,8/8,1/8;random=4/8,6/8,2/8.
Ordered stand6000/6001/6002/6014 then forward same resets:
nominal101 steps16,17,16,17,500,500,500,500;
202 all500;303=55,50,14,46,16,16,16,500.
Random101=48,90,13,98,500,500,500,500;
202=500,22,500,500,500,46,500,500;
303=81,43,62,54,48,500,53,500.
All screens rejected/full suites skipped. Nominal worst linear/yaw .2362/1.5941,
.2243/.3600,1.0270/3.8448. No development pass or seed winner. Initialization
credit hypothesis CLOSED;no further discount values or unchanged extension.
Six evenly spaced video frames per nominal/random forward6000 clip reviewed:
101/202 upright ordinary gait frames,303 short abrupt asymmetric gait/contact;
representative successful clips do not supersede failed rows or certify gait.

Saved late internal eval lengths RB02=475.44/473.06/474.75, RB03=413.81/500/334.31.
Final KL RB02=.06166/.05778/.06292;RB03=.03323/.03588/.03296. No KL-stop trigger.
Final entropy_loss RB02=.01685/.01261/.01710 versus RB03=-.04192/-.04032/-.04260.
Entropy cost unchanged .005, so learned distribution changed substantially;
this is a clue, not proof mean-action evaluation caused failure. Internal eval
is configured deterministic and passed through train.py; no defect asserted.

RD06 freezes ZERO-training nominal stand6000 at each RB03 final20070400,500step
cap, three predetermined policy RNG keys17/29/43 PER seed101/202/303. Nine total
trajectories, reuse saved deterministic baseline16/500/55 steps, no baseline
reruns. Existing stochastic-policy diagnostic code; no physics/command/phase
intervention. Source prep then3traces per job,2hqueue,40min/trace,oneGPU3CPU16GB,
submit all three together. Plans cluster_rd06_SEED.json specify all commands.
Historical A03 C10 difficult-start sampling0/6 remains failed; this probes a NEW
observed high-entropy nominal failure,not reopening or rerunning that audit.
Decision: only if all3 draws for EACH policy survive500 with height>.6 supports
further investigation of mean-versus-distribution failure; any failed draw
rejects robust sampling explanation and closes this branch without more keys.
Record linear/yaw/gait metrics too; sampled survival NEVER passes final gates.
No choosing lucky draws,seeds or checkpoints;no inference sampling deployment.
No automatic entropy tuning; justify/freeze any later training-consistent design
separately. Original deterministic full evaluation and held-out3seed contract
remain. No runner logic changed; all plans validated and fixed RNG grids checked.
