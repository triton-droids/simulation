# Atlantis Milestone 2 continuation — authorized 2026-11-07

The user explicitly authorized bounded background research until one frozen recipe
passes the original gates on three independent fresh training seeds. This supersedes
the September M1-only stop instructions. M1/T02 and all failed M2 evidence stay intact.
Soccer skills and hardware are not part of this campaign. Training uses MJX/Brax and
the pinned Playground research adapter; the Isaac field is a separate integration.

## Active batch RC01 � matched late objective (2026-11-08)

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

## Startup repair � 2026-11-08

RB01 job7501 failed in 3 seconds before the queue or training started: the uv-created
remote environment has no pip. GPU discovery passed. Preserve its isolated directory
and local launcher_failure.log. RB01r1 keeps the same frozen queue/recipe/budget,
but records installed packages using Python importlib.metadata instead of pip freeze.
This is an execution repair, not a scientific failure or training extension.

## RB01r1 review and RC01 decision � 2026-11-08

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

## RD06 rejected; RB04 fixed-transition temporal-fragment test
RD06 jobs7706/7708/7710 completed3m44/3m53/3m54=0.1919 allocated GPU-hours.
Keys17/29/43:101 steps56/414/354,202 all500,303 500/423/500. Four of nine
terminate foot-foot; survivors5/9 cannot satisfy frozen all9 criterion. Yaw RMSE
101=1.86194/1.54013/1.48122,202=1.43294/1.34053/1.31463,
303=1.63799/1.44094/1.48212. Linear RMSE .190-.217;single support .662-.729.
All sampled standing remains imprecise. Close distribution/sampling explanation;
no more draws, entropy tuning, stochastic deployment or lucky-result selection.
Numeric summaries saved per receipt/review; no new gait videos generated by this
trace-only diagnostic. Previous RB03 video review retained,not repeated.

RB04 freezes one different training-data structure:80-step contiguous unrolls
versus RB02's20, paired with batch_size32 versus128 to hold samples per optimizer
batch exactly40960 (length*batch*16minibatches),updates4 and total20,070,400steps
unchanged. Same development seeds101/202/303,from scratch, original RB02 gamma.97,
lambda.95,rewards,resets,networks,lr3e-4,512env. No RB03 ancestor or longer discount.
Measured RB02 recovery collapses41-60steps, beyond one20step/.4s fragment;80steps
covers1.6s. Installed Brax compute_gae inspected: backward recursion spans T and
uses bootstrap_value at its boundary. Longer fragments reduce dependence on
intermediate critic bootstrap for delayed failure. Gamma/lambda still attenuate
credit strongly; this is uncertain,not guaranteed recovery. Fewer independent
fragments also change within-batch correlation,so do not claim pure causal isolation.
Different from C05/T04 episode-horizon resets: episode stays500,full resets stay
unchanged,update counts/data budgets preserved. Prior discount/push/curriculum/
optimizer/reward/phase/filter branches remain closed. No local length sweep.
Three parallel oneGPU3CPU16GB jobs,total60,211,200steps;4htrain/6hqueue each.
Final20070400 only,own untrained0 control,unchanged48screen/conditional180full
numeric/control/gait gates. Reuse saved RB02 anchors,not retrain comparison.
Commands:cluster_research.py submit --run RB04_SEED --plan
Documentation/Research/queues/cluster_rb04_SEED.json --cpus 3 --memory-gb 16.
Frozen decision:ALL3 must retain nominal gates,at least6/8 random survivors and
both6001commands survive500 to support targeted gain. Otherwise close temporal
fragment hypothesis,no unchanged extension/additional lengths. Partial gain
never full pass;full development numeric/control/visual success required before
freezing unused independent final seeds AND held-out tests. All3 then train every
component independently in parallel. No altered-reset diagnostics as validation.
Validation:all3 argv equality except length/batch/output checked;40960 transitions
per batch and exact final-step divisibility verified;plans validate. No runner
logic changed. Keep hourly compact checks until measured runtimes justify change.

## RB04 seed202 execution repair
RB04_202/job7734 failed13seconds before training at unconditional rollouts.py
mediapy import: matplotlib stylelib/seaborn-v0_8-notebook.mplstyle unavailable.
Relevant stage log preserved locally RB04_202/failure_recovery_train.log.
This repeats the shared-file import fault class,not scientific failure. Current
read-only stat finds382byte file with Nov7 timestamps and mediapy import succeeds;
root shared-filesystem cause remains unproven. Do NOT blindly retry unchanged.
Small repair: rollouts.py loads mediapy/moviepy only within render_video; this
training plan has video disabled,so optional video data cannot abort its import.
Rendering behavior retained,not disabled. No package install/shared mutation.
Two regression tests passed: real non-video module import with media/moviepy
blocked, and rendering imports/calls its writer and compositor. Actual remote
repaired module also imported successfully with both video dependencies blocked.
This isolates training's unnecessary dependency; evaluation still needs video
packages and must be recovered separately if it fails. Never repeat completed
training for evaluation recovery. Fresh RB04_202r1 uses SAME frozen202 plan,
seed,budget,gates. Completed101/job7733 and303/job7730 remain untouched;review
all3 together after replacement. No winner or full-pass claim. Original failed
receipt/output preserved. No repeated blind allocation/package retries.

## RB04 rejected; RP02 independent-environment capacity prerequisite
Completed101/202r1/303 costs51m04/49m15/50m36=2.5153 GPU-hours,plus preserved
13second failed202 import. All trained full suites skipped. Nominal survivors8/8
all3 but only101 passes gates;202 forward error~.447 fails motion;303 yaw fails.
Nominal worst linear/yaw101=.1599/.2915,202=.4473/.0350,303=.0999/.2221.
Random survivors6/8,4/8,6/8. Ordered stand6000/6001/6002/6014 then forward same:
101 steps500,60,500,500,500,98,500,500;
202=56,40,500,500,58,41,500,500;
303=500,53,500,500,500,63,500,500.
Random worst linear/yaw1.7360/.7680,1.6001/.3330,1.9239/1.7089.
Reject temporal-fragment hypothesis:nominal regressions and6001 failures. No
longer unroll sweep/extension. Six sampled frames per nominal/random forward
clip inspected for all3:101/303 visible ordinary upright gait,202 nominal planted
and randomized fall. Samples not full visual certification. All CSV/videos saved.

Next separately bounded hypothesis is independent-environment diversity per
optimizer batch,not increasing batch transitions or repeating longer fragments.
RB02 uses512 environments*4 sequential20step unrolls;2048 environments*1 unroll
would retain40960 transitions,batch128*16,20step length,4updates,gamma.97 and
same total steps. More independent histories may help unstable recovery;not proven,
and age/correlation of collected trajectories changes. No reset-ID selection,
rare-seed oversampling,altered gates or larger nominal training-step budget.
Before a full campaign,RP02_capacity runs ONE2048env81920step seed11 capability
probe on oneGPU3CPU16GB,30min train/45minqueue. Original RB02 reward/config,
2 internal evaluations,checkpoint0 and81920 required;no external pass claims.
It exercises actual training/compilation/checkpoint saving at proposed shape.
No shared dependencies/settings changed. Existing post-first KL guard retained;
this two-update capability probe cannot judge convergence or policy quality.
Commands:cluster_research.py submit --run RP02_capacity --plan
Documentation/Research/queues/cluster_rp02_capacity.json --cpus 3 --memory-gb 16.
On completion inspect runtime,finite metrics,final checkpoint/manifest and memory
accounting where available. Successful completion proves only that this bounded
workload fits allocation;do not claim measured GPU headroom without telemetry.
If OOM/timeout,close2048-on-current-allocation path,no environment-count sweep or
blind retries. If successful,freeze one matched20,070,400step-per-seed2048env
study for existing development101/202/303 versus saved RB02 BEFORE launch;all3
ready jobs together. No warmstart from capacity probe. Retain original full
gates and full development prerequisite for unused final seeds/held-out tests.
Only one capacity task is ready now;do not duplicate it just to occupy GPUs.
Prior discount/fragment/sampling/push/curriculum/reward/phase branches closed.
Plan-only change,integer batch/env/timestep compatibility and queue validation
checked. No runner logic changed;lazy optional-video import repair preserved.

## RP02 capacity passed; RB05 fixed-batch environment-diversity freeze
RP02/job7738 completed9m24 (0.1567GPUhours),training manifest540.07seconds,
checkpoint0 and81920 exist,two metric records all scalar values finite.
Slurm MaxRSS6180332K (~5.9GiB host);GPU peak/headroom NOT measured.
Training/sps606.13 includes startup compilation in this tiny run,not a reliable
steady-state full-run estimate. Final early KL8.8666 and meanlength55.5 are NOT
learning success;this was only fit/checkpoint capability. Existing full-run
post-first two-consecutive KL>=.2 stop rule remains unchanged,not disabled.
No capacity checkpoint reused; full recipe is independent initialization.

RB05 freezes2048env instead of512, otherwise EXACT saved RB02:batch128,
unroll20,16minibatches,4updates,gamma.97,lambda.95,lr3e-4,original rewards,
full random starts,episode500,20,070,400steps and8evals each. Development seeds
101/202/303 all independent from scratch;60,211,200steps total.40960 transitions
per optimizer batch unchanged,now one20step fragment per2048 environments
instead of four sequential fragments per512. Hypothesis:more independent
histories reduce correlated recovery acquisition failure. It changes trajectory
age/correlation,not just reset counts; not a proven mechanism. No upstream exact
replication or final validation claim. No new reward/discount/length experiments.
Three oneGPU3CPU16GB jobs submitted together;4htraining/6hqueue perjob. Use
cluster_research.py submit --run RB05_SEED --plan
Documentation/Research/queues/cluster_rb05_SEED.json --cpus 3 --memory-gb 16.
All other stage arguments/gates copied unchanged;fixed final20070400,own0control,
original48screen/conditional180full suites and representative videos. Compare
ALL3 to saved RB02,not rerunning control training or promoting lucky seeds.
Decision:ALL3 retain nominal gates,randomsurvival>=6/8 and BOTH6001commands
survive500 to support targeted improvement. Any regression/missing recovery
rejects hypothesis;no environment-count sweep or unchanged extension. Even
improvement needs full original numeric/control/visual development passes before
unused fresh training seeds AND held-out grid;all3 then independently train every
component in parallel. Every failure retained. Partial screens never fullpass.
Validation:all3 training argv differ ONLY num_envs/output from RB02;batch integer
compatibility/final-step divisibility and queue schemas checked. No runner logic
changed. Hourly firstcheck;adjust cadence only using measured run progress/runtime,
not startup throughput or Slurm walltime. No active model waiting for training.

## RB05 partial gains, rejected recipe; RC02 paired precision test
RB05 costs43m35/57m47/43m38=2.4167GPUhours;MaxRSS7.6-7.8GiB host.
101 all16 screen episodes survive;nominal standyaw.2102-.2283 fails. Random
stand6001linear.3526>.35.202 passes both screens;full nominal24/24 survives
and numeric/control gates pass,full random24/24 survives but backward6001linear
.4230>.35 fails. Extra12forward resets all500,maxlinear.2873/yaw.2425.
303 nominal passes but random6/8,stand/forward6001 terminate62/63steps.
Thus frozen ALL3 rule fails. No environment-count sweep or unchanged extension;
no full development pass or promoting202 alone. Partial recovery gain in101/202
is preserved as evidence. Nine representative videos sampled6frames each show
ordinary upright moving gait;not continuous certification or evidence failed
reset/standing rows pass. All numeric failures retained;101/303 full suites skipped.
Final KL .05988/.06417/.06246,entropy_loss .01498/.01862/.01748,internal mean
length473.25/500/500. Precision/recovery failures persist despite finite updates.

RC02 freezes ONE paired late learning-rate test across ALL three parents:
lr1e-4 versus matched unchanged3e-4,each1,146,880steps (4*40960*7 exactly),
2048env,batch128,unroll20,original gamma.97/rewards/full resets. Six independent
jobs,6,881,280new steps total. This is not automatic unchanged extension: control
arms exist only to discriminate smaller updates from extra training/restarts.
Same seed and own fixed RB05 final20070400 parent within each pair;no shared
trained ancestor BETWEEN seed families. Parameter/normalizer warmstarts,NOT exact
optimizer/critic/PRNG resumes;both arms have identical restart semantics.
Hypothesis:smaller late updates improve precision while preserving acquired
recovery. Final KL around.06 motivates checking,not proof learning-rate cause.
Historical adaptive L01 from initialization failed;no adaptive target/floor sweep.
RC01 tested action-rate reward at fixedlr1e-4,not this matched learning-rate pair;
its failed recipe remains closed. No reward/mask/discount/phase/filter changes.
Parent SHA256 frozen in each plan;exact initial actor/normalizer parity required.
Untrained control remains each RB05 checkpoint0,NOT the warmstarted checkpoint0.
Fixed final1146880 only,original screens/conditionalfull gates and videos.
Six ready jobs submitted together,oneGPU2CPU12GB each (measured hostpeak<8GiB),
1htrain/4hqueue each;Slurm account limits determine simultaneous execution.
Commands:cluster_research.py submit --run RC02_SEED_ARM --plan
Documentation/Research/queues/cluster_rc02_SEED_ARM.json --cpus 2 --memory-gb 12.
Decision:compare BOTH arms for every seed,no mixing winners across seeds. LowLR
must retain parent survival8/8,8/8,6/8 and acquire both6001survivors for303,pass
nominal screening all3,and improve101 standingyaw versus matchedcontrol without
new gait regression to support hypothesis. Otherwise close,no LR sweep/extension.
Either complete recipe still needs ALL original full numeric/control/visual
DEVELOPMENT gates before new unused finalseeds/freshheldout;all3 trainedindependently.
No partial screen/single-seed or sampled-video success claims. All failures retained.
Plan-only changes:pair argv equivalence exceptLR/output,exact budget,unchanged
evaluation thresholds/referencecontrol paths checked;validate all6 before launch.

## RC02 submission recovery - Nov10 local/Nov11 UTC
Prior six simultaneous transfers encountered SSH resets/abort. Only202low/job7769
submitted and completed;five others stopped before sbatch (101control at run.sh
write,others during archive upload). Prior attempt to record this was blocked by
automatic-approval usage limit;no recovery launch occurred then. This wake read
all receipts/statuses and reconciled squeue/sacct sinceNov10:only7769 exists for
RC02,no active jobs. All five partial source directories exist but no run.sh or
exit_code. Reconciliation shell returned1 solely because ls found missing files;
SSH succeeded,output retained in session. Preserve all partial directories and
local per-run submission_failure.json;do not overwrite/delete or resubmit originals.
Five fresh run names appendr1,identical existing plans/seeds/parents/budgets.
Serialize snapshot transfers/setup to avoid simultaneous SSH bursts,submit each
ready GPU job immediately;do NOT wait for training to finish before submitting
next. GPU jobs remain independent/concurrent as scheduler permits.202low remains
untouched and no winner selected before all six complete. No code/dependency or
scientific recipe changes;source snapshots differ only documentation/status.
If connection fails again preserve and reconcile next wake,not blind retries.

## RC02 rejected; RD07 matched recovery-response diagnostic
All six finished successfully. Costs control101/low101/control202/low202/
control303/low303=33m03/32m59/33m00/44m16/33m12/30m45,3.4542 GPU-hours.
All nominal episodes survive. Control nominal stand worst yaw .18741/.16080/
.15196 fails all three; low .20972/.12938/.15472 passes only202.
Random survivors control8/7/8,low7/8/6.101low stand6001 falls98steps;
202control forward6001 falls132;303low stand/forward falls66/67.303control
now survives both but random standyaw.37146 fails.101low does not improve
standyaw versus control and loses recovery. Frozen hypothesis rejected;no
LR sweep/unchanged extension or mixing per-seed winners.
202low passes original full nominal/random numeric-control gates,24/24 each,
worst linear/yaw .12540/.22614 and .20572/.29745;extra12forward resets all500,
worstlinear.19716/yaw.22100. This is useful single-seed development evidence,
NOT a full three-seed or held-out success. All other full suites skipped.
All actor/normalizer parity stages pass. Final low KL .01827/.01833/.01859
versuscontrol .06044/.06349/.06149:small updates do not uniformly preserve
recovery. CSV,metrics,manifests,statuses and videos saved in each run/review.
Six random-forward clips sampled six frames each,plus202low full combined
nominal/random and extra-forward samples:ordinary upright stepping visible.
These are samples,not continuous certification;existing screen plans rendered
forward only,not standing videos. Numeric failures preclude promotion anyway.

Cheap saved-parameter check:normalizer mean max shift .0154-.0354 parent std,
RMS .0051-.0076;std ratios .9776-1.0146. No evidence of gross normalization
reset;small effects not excluded,does not justify normalization intervention.
Runtime Brax restore_value_fn default is TRUE,training forwards no override:
correct earlier prompt wording 'fresh critic'. Saved value parameters are restored;
optimizer/step/PRNG are not. Actor/normalizer parity alone did not verify critic.
No scientific result is relabeled an exact resume. Evidence:
Artifacts/Results/cluster_research/rc02_normalizer_drift.txt.

RD07 freezes twelve ZERO-training original-reset6001 traces,stand and forward.45
for BOTH RC02 final1146880 arms across ALL101/202/303. Six independent jobs,
oneGPU2CPU12GB each,two500step caps,90minqueue each;unchanged deterministic
physics/rewards/phase/normalization/reset,candidates1/scale1. No new screening
or baseline training. Unlike RD04's all-failing RB02 traces,this contrasts
successful/failed matched RC02 policies;do not repeat the already resolved
question whether initial horizontal velocity matters.
Hypothesis:failed recoveries show delayed horizontal-error braking relative to
matched successes BEFORE height loss. Verify equal initial qpos/qvel/phase;
record terminal causes,first50 and preterminal20 contact,height,foot separation,
actions/saturation. Trace velocity is pelvis-local;rotate velocity minus held
command into world coordinates using qpos quaternion before comparison. Project
onto initial world horizontal velocity error,divide by its norm;initial projected
normalized error is1. Freeze early metric as mean at steps11-20. Require initial
error norm>.1,all compared trajectories still alive and height>.6 through20.
Four prespecified screen-divergent pairs:101stand,202forward,303stand/forward.
Braking-lag explanation supported ONLY if all four scalar pairs reproduce the
success/failure split,successful members survive500 withheight>.6,and every
success early ratio<=.75 while its failed mate>=.90. Otherwise reject blanket
lag explanation,retain per-case evidence;scalar/vector disagreement must be
reconciled before training. Correlation is not proof of causation. Remaining
four trajectories are fixed context,not discarded. No extra reset/command sweep.
No automatic smoothing/filter/reward/phase intervention or training. Review
results before separately freezing a mechanistic redesign;closed branches stay
closed. Short traces are not settled windows. Diagnostic thresholds never replace
original validation gates. No seed/checkpoint selection or promoting202 alone.
Commands:cluster_research.py submit --run RD07_SEED_ARM --plan
Documentation/Research/queues/cluster_rd07_SEED_ARM.json --cpus 2 --memory-gb 12.
Pace SSH uploads/setup but submit every ready job without waiting for others'
training;Slurm controls useful concurrency. Validate all6 plans and frozen argv,
commit before launch. No runner changes;original full gates remain untouched.
Only full development success permits separately frozen unused finalseeds and
held-out grid,all3 training every component independently in parallel.

## RD07 rejects early-braking rule; evaluation-path discrepancy requires RD08
All12 traces completed; six jobs3m24/3m14/3m14/3m24/3m24/3m04 =.3289GPUh.
Initial qpos/qvel/phase_dt exactly equal across all12 saved initial states.
Scalar steps stand/forward:101control500/500,101low93/500,202control163/500,
202low500/500,303control500/500,303low64/64. All failures terminal foot-foot.
101low/303low lose height below.6 before terminal collision;202control stand
collides while height remains>.66,so not one shared collapse mechanism.
First50 raw action saturation0 all12; successful and failed traces overlap in
foot separation/support/action-rate. Preterminal failed foot separation minima
.0735/.0813/.0472/.0552m; no inference-only clipping/filter/reward conclusion.
Early normalized world horizontal error ratios stand/forward:
101control1.197/-.579,101low1.239/-.629;202control1.307/-.190,
202low1.118/-.262;303control1.318/-.500,303low1.462/-.373.
All first20 heights>.68. Successful standing does not meet<=.75 rule;failed303
forward already brakes/overshoots. Reject blanket early-braking-lag hypothesis.
No automatic smoothing or new training. Full per-window measurements/terminal
causes:Artifacts/Results/cluster_research/rd07_analysis.json;raw traces retained.

Material discrepancy:screen202control stand/forward500/132 versusRD07 163/500.
Other failures differ modestly in timing(10198vs93;30366/67vs64/64),but202 flips
both survival labels. Do NOT substitute the passing diagnostic for official
failures or select evaluation paths. Source inspection corrects terminology:
evaluate_g1 main uses scalar fused lax.scan,not its optional vmap helper;
compare_g1_gait_rewards uses separately jitted policy/physics in a Python loop.
Both split reset keys and hold commands;resample_time differs10vs10.02 but held
commands and500step cap should prevent resampling impact. Both use original
reset scale1/candidates1/no noise/push;source assets and actual numerical path
still need runtime reconciliation. Contact sensitivity/numerical fusion is a
hypothesis,not a confirmed defect. No pipeline rewrite or gate change yet.

RD08 freezes TWO independent identical official evaluator repetitions a/b,
RC02_202_controlr1 fixedfinal1146880,original random6001,stand/forward.45,
500steps,existing genuine RB05_202 checkpoint0 and standing controls. ZERO
training. Use unmodified evaluate_g1 --save-numeric-traces,which exports existing
host arrays without changing compiled rollout;no video/reward/physics changes.
Two oneGPU2CPU12GB jobs,40minstage/60minqueue caps,6episodes perjob. This repeat
is specifically a reproducibility test,not evaluation recovery or winner search.
No extra seeds/checkpoints. Reuse saved RC02 screen and RD07 traces;do not rerun
training or already reviewed videos. Compare both official repeats row-by-row
and numeric arrays (finite/allclose rtol1e-5,atol1e-6;also report exact equality),
including valid/done and height/local tracking-error/contact. Both must retain
original official stand500/forward132 to establish cross-run reproducibility.
If repeats agree with original but differ from RD07,localize first divergence
against saved stepwise arrays and review graph/config differences before proposing
a fix;do not claim root cause merely from endpoint disagreement. If they differ
from each other/original,investigate runtime/device/source provenance and stop
training,not additional blind repeats. A threshold-crossing difference is material
regardless of small initial floating-point differences. Neither repeat can count
as new validation. Original frozen gates remain;all historical failures preserved.
Commands:cluster_research.py submit --run RD08_a/b --plan
Documentation/Research/queues/cluster_rd08_a/b.json --cpus 2 --memory-gb 12.
Plan-only change;pair argv equality except output verified,queue validation and
diff checks required before commit/isolated launch. All ready jobs submitted with
paced transfers,without waiting for another job. Campaign awaits reconciliation.

### RD08 transport recovery
Nov11 09:17UTC wake:SSH available. RD08_a compact status has no queue/job;
squeue empty and queried sacct no rows. Isolated RD08_a/source exists but no
run.sh,exit_code,launcher.log;prior setup SSH banner abort occurred before sbatch.
RD08_ar1 and RD08_b remote directories absent;RD08_b has no local receipt.
Preserve original partial directory/receipt/submission_failure.json. Fresh
RD08_ar1 uses unchanged cluster_rd08_a.json;RD08_b uses unchanged b plan.
No code/recipe/evaluation changes or training. Pace submissions,never wait for
another GPU job to finish;if transport fails again preserve and reconcile later.

## RD08 fails cross-job numerical reproducibility; RD09 execution isolation
RD08_ar1/b completed2m14/2m24=.07722GPUh. Both stand500;forward141 versus132
(original132). Untrained/standing controls terminate40 in both. Neither trained
command trace is exact or allclose(rtol1e-5,atol1e-6) across jobs. Stand first
linear discrepancy outside tolerance atstep30;forward at33. Max stand height
.00399m,yaw.35838;forwardheight.39419m,yaw3.41803. Survival class agrees for
these two official repeats but termination timing and trajectory do not.
Against RD07 stepwise trace,official linear/yaw differences begin step1;contact
switch differences occur by11/17steps. No proven code defect or root cause yet.
Numeric traces and analysis retained under cluster_research/rd08_analysis.json.

Read-only provenance checks:all four relevant snapshots' evaluator/comparator/
source-loader hashes match;runtime_versions.txt hashes match. Recorded versions
Brax.14.2,JAX/JAXlib.11.0,MuJoCo3.10.0. RTX2080Ti/driver610.43.02 same;
physical GPU UUID was not recorded in old launcher heads,so device identity is
unknown. Playground revision8a4b4642d8eba8a80ac99ed125cb62c16e1457ad and shared
Menagerie revision71f066ad0be9cd271f7ed58c030243ef157af9f4 match current read-only
checks;revision equality alone is not an exhaustive historical asset hash audit.
Evidence saved rd08_provenance.json. No training,gate changes,or favorable-path
selection authorized by these discrepancies. Original failures remain binding.

RD09 freezes ONE zero-training same-executable test,not another blind job repeat.
Reconstruct official evaluator environment/network/config for same202control
final1146880,reset6001,stand/forward.45,500steps;explicitly lower/compile existing
_build_rollout ONCE,then call that executable in order stand,forward,stand,forward.
Same policy/seed/commands/device/process/executable. Save all4 numeric traces,
initialqpos/qvel,policyhash,effectiveconfig,assignedGPU UUID/PCI/driver and allowed
JAX environment settings. Do not change precision/autotuning/device assignments.
OneGPU2CPU12GB,40minstage/60minqueue;no model waiting for completion.
If both pairs exactly repeat and initial states match,execution repeatability is
supported for this test;investigate cross-compilation/device/config effects next,
not claim solver correctness or full root cause. If either differs,report first
field/time discrepancy and investigate execution-level determinism;no automatic
training or endless repeats. Same tolerances/exact report,NaNs fail closed.
No new validation or replacement of official failures even if this path survives.
Two tests passed for equality/divergence localization,boolean done/valid changes,
nonfinite and schema fail-closed behavior. Script syntax and queue validated.
Existing evaluator and its numerical rollout are unmodified;new diagnostic calls
its compiled function. Full original numeric/control/visual and fresh3seed/heldout
gates remain. Submit isolated committed source via cluster_research.py submit
--run RD09 --plan Documentation/Research/queues/cluster_rd09.json --cpus2 --memory-gb12
(spaced CLI options as usual). Only this bounded diagnostic is ready;no invented
work to fill GPUs. Preserve all evidence and avoid shared configuration changes.

## RD09 same-executable exact; RD10 isolates compilation from device
RD09/job7801 completed2m24=.040GPUh. Both stand traces500 and forward132,
all saved numeric fields bitwise identical and finite within each pair;initial
qpos/qvel exactly identical. Assigned RTX2080Ti PCI3D UUIDa0f9bb2a-c3bd-24f7-fe8b-
4bad488aa5b8,driver610.43.02. Policy SHA256c5fe29d6edb7513c3eaa1ade910b6254c2c89482672f1b453d3832c484fdb4c4.
XLA_FLAGS unset;job-local JAX cache. This supports execution repeatability for
this one executable/device,not general solver correctness or full root cause.
No numeric failure is replaced by a passing diagnostic;training remains deferred.

RD10 freezes ONE job with TWO sequential fresh Python processes/compilations on
its SAME assigned GPU. Each invokes unchanged RD09 diagnostic with a unique
job-local tempfile JAX cache created before importing JAX;all other settings
unchanged. Each process compiles once and executes stand/forward twice,8traces
total;same saved202controlfinal1146880,original6001reset,500steps,zero training.
OneGPU2CPU12GB,25minstage each/60minqueue. Sequential stages are necessary to
hold the allocated physical GPU fixed,not serialization of independent seeds.
Compare within-process exactness first,initial states/policyhash/effectiveconfig
and GPU UUID equality. Then compare a/b first traces fieldwise exact/allclose
rtol1e-5,atol1e-6. If within pairs exact but across processes differ with matched
provenance,variability exists across fresh-process/compilation paths on one GPU;
autotuning still unproven. If all exact,cross-job device/other-context explanation
remains unresolved,not ruled out;do not launch more blind repeats or training.
If within pairs differ,retain evidence and investigate execution-level behavior.
No precision/autotune flag change,shared cache mutation,Slurm bypass or new GPU
selection. Temporary caches stay private;all failures/trace evidence retained.
Plan-only change;validate queue and both wrapper argv before committed isolated
launch. Existing comparison tests cover exact/allclose/NaN/schema logic. No new
training,video reruns,gate changes or final-seed claims. No invented parallel work.
Command:cluster_research.py submit --run RD10 --plan
Documentation/Research/queues/cluster_rd10.json --cpus 2 --memory-gb 12.

## RD10 isolates fresh-process variability; RD11 compiler-artifact capture
RD10/job7816 completed4m37=.07694GPUh. In each fresh process all fields repeat
bitwise exactly;across a/b neither command is allclose. a stand500/forward141,
b stand500/forward132. Initial states,policyhash,effective config and physical
GPU UUID all match (same PCI3D as RD09). Cross-process deviations reproduce
RD08 patterns. This excludes different physical GPU identity as necessary for
this observed discrepancy. It does not prove autotuning is the cause.
Measurements:cluster_research/rd10_analysis.json and all raw traces preserved.
No training or acceptance-gate changes;no favorable evaluator selection.

Official references consulted Nov11:
https://openxla.org/xla/determinism describes compilation variability from GPU
autotuning and determinism options;these remain hypotheses/options,not applied.
https://docs.jax.dev/en/latest/jax.stages.html documents Lowered.as_text and
Compiled.as_text as debugging representations,NOT portable serialization or
proof of semantic identity. No precision/autotune flags are changed.

RD11 freezes ONE job with two fresh processes/private caches on same assigned GPU,
now COMPILATION ONLY. Same savedpolicy/reset/input shapes and official lowered
rollout graph. Export lowered.txt and compiled.txt plus hashes,effectiveconfig,
policyhash,GPU UUID/driver;execute ZERO episodes. No robot trajectory repeats.
OneGPU2CPU12GB,25minstage/60minqueue. New diagnostic flags capture compiler text
and suppress the execution loop;default RD09 behavior remains unchanged. Missing
compiler text fails rather than silently pretending coverage. 2 existing trace
comparison tests pass;script syntax and wrapper/queue schemas verified. Actual
GPU compiler-text capture remains to be verified by this bounded job.
Compare raw/debug-normalized text;separate path/module/source-location changes
from operation/layout/fusion/backend_config/kernel-choice differences. Equal
lowered IR with differing optimized IR supports compiler-path divergence;locate
specific differences before proposing numerical controls. Different lowered IR
requires source/Python graph-order investigation. If text identical,do not infer
binary equality or rerun blindly;investigate available metadata/artifacts. Text
hash differences alone are not proof of autotuning. No automatic training or
flag change;closed scientific branches/gates remain unchanged. Preserve captures
and report uncertainty. RD08-10 completed:do not rerun their trajectories.
Command:cluster_research.py submit --run RD11 --plan
Documentation/Research/queues/cluster_rd11.json --cpus 2 --memory-gb 12.

## RD11 substantive compiler difference; user-authorized RD12 repair verification
RD11/job7830 completed3m54=.065GPUh;runtime IR capture succeeded. Lowered IR
identical SHA4b248a454830906f431d408c99dca15eb65b1a33435e2177aa2431a1e10a2bda.
Optimized IR differs in exactly4lines:one fingerprint metadata and THREE
cuBLAS-Lt matmul selected_algorithm fields (1 versus0),in forward/fwd_position
physics calculations. Line counts78177 match. Policy/config/GPUUUID identical;
assigned PCIDA. Local review/summaries.json and compiler_diff.json preserve facts.
This is substantive compiler-choice divergence,not mere file metadata. RD11
had no rollouts,so causal attribution of prior trajectory differences needs test.

User now explicitly says FIX/VERIFY/get moving;authorizes this bounded job-local
numerical-runtime repair test, superseding prior RD11 prohibition on flag changes
for its observation-only scope. RD12 changes ONLY XLA_FLAGS to documented
--xla_gpu_autotune_level=0 before JAX import;no precision setting,physics,
termination,reward,policy,reset or gate changes. Ref:https://openxla.org/xla/determinism
(disabling live benchmarks chooses default configurations;may reduce performance).
Two fresh processes/private caches within oneGPU allocation;existing RD09 script
captures IR AND runs stand/forward twice each,8trajectories total. Same202control
final1146880,original6001,500stepcap. OneGPU2CPU12GB,25minstage/60minqueue.
No new training. Both wrapper environments/argv executed with runpy/tempfile
mocked to verify flag set before imports,capture enabled,compile-only absent.
Queue syntax/validation checked before commit. Unexpected preexisting XLA_FLAGS
fails closed rather than overwritten. Historical outputs untouched.
Success requires within AND across process numeric traces exact (also report
frozen allclose),finite,initialstates/policy/config/GPUUUID match;inspect optimized
algorithm selections for stability apart from harmless metadata. Survival itself
is NOT repair success/validation;failed policies remain failed. If stable,adopt
an explicit recorded runtime profile for future jobs and recheck original gates
without repeating completed training. If unstable,retain evidence,no flag sweep
or automatic training. Measure runtime. Do not choose a numerical path because
it passes;the decision rule is reproducibility,not favorable behavior.
Command:cluster_research.py submit --run RD12 --plan
Documentation/Research/queues/cluster_rd12.json --cpus 2 --memory-gb 12.


## RD12 verified repair; RE02 saved-policy evaluation restart (Nov11)
RD12/job7834 completed4m34=.07611GPUh. With ONLY job-local
XLA_FLAGS=--xla_gpu_autotune_level=0, both fresh processes/private caches have
IDENTICAL lowered AND compiled text hashes. All within/across-process stand and
forward numeric trace fields are finite and bitwise exact (also allclose).
Initial states, policy hash, effective configuration and GPU query match.
Stand500/forward132 in every repeat: failed walking remains failed, not a
favorable-path selection. Compiled SHA1bd834aca957cfd58b77bc666fc91a442ffe24fb07807d116819f4c0b57d1c8d.
Artifacts: cluster_research/RD12/review/cross_process.json and raw summaries/NPZ.
This verifies the repair for this graph/device/test; not universal determinism
or a three-seed policy. Cross-device behavior and other graphs remain unproven.

Launcher now exposes explicit --runtime-profile reproducible; legacy default
preserves old recipes. Profile/flags recorded in submission receipt, installed
before JAX preflight, inherited by evaluation subprocesses, conflicting inherited
flags fail closed. No shared environment changes. GPU UUID/PCI now recorded in
launcher output. Ten local tests pass; actual remote bash verifies unset/matching
flags succeed and conflicting flag exits2. Existing diagnostics isolated and
completed; no mutation of active source snapshots. No precision or gate change.

Freeze RE02_101_control,RE02_101_low,RE02_202_control,RE02_202_low,
RE02_303_control,RE02_303_low: ZERO training, all six saved RC02 final1146880
checkpoints, own genuine RB05 untrained0 controls. Original nominal/random
screens, conditional full suites, extra forward reset grid and videos unchanged.
All gate objects and screening dependencies equality-checked against RC02 plans;
all six queue schemas validated. Source-run policy hash recorded before eval.
OneGPU2CPU12GB each,4h queue limit,2h maximum per evaluation stage; submit all ready
jobs with paced SSH uploads, no waiting for another job to finish. No new seeds,
checkpoints, reward/optimizer changes or repeats of training. This one bounded
re-evaluation is justified by the verified runtime repair, not an RC02 extension.
Review all six together against preserved legacy outcomes; do not mix arm winners
or certify from partial screens. Runtime-profile-specific passes must retain all
original numeric/control AND visual gates. Even complete development success
requires a separately frozen complete recipe and unused fresh training seeds plus
fresh held-out tests, every component independently trained in parallel.
If profiles change outcomes, report all changes, including regressions. No attempt
to choose per-policy flags or favorable evaluator. Use reproducible profile for
this full batch consistently. Original failures remain in the record.
Commands for each TAG=101_control,101_low,202_control,202_low,303_control,303_low:
.venv/Scripts/python.exe robocub/tools/cluster_research.py submit --run RE02_TAG
--plan Documentation/Research/queues/cluster_re02_TAG.json --cpus 2 --memory-gb 12
--runtime-profile reproducible


## RE02 reviewed; RD13 bounded saved-critic rank audit
All six RE02 jobs completed, total84m27=1.4075GPUh. Original screen verdicts
unchanged under reproducible runtime: control random survivors8/7/8;low7/8/6.
Nominal standing yaw control .187407/.160804/.151962,low .209721/.129382/.154724;
only202low passes both screens. Failures101low stand6001 step98,202control
forward6001 step132,303low stand/forward6001 steps66/67. All original failures
remain;no three-seed pass. 303control random worstlinear .39054 also fails.
202low full nominal24/24 and random24/24 numeric/control pass,extra forward12/12
survive. Full nominal/random maxlinear .17734/.29093;extra .27883, versus prior
legacy .12540/.20572/.19716: numeric regression despite same gate outcomes.
Do not infer cross-device bitwise equality from aggregate verdict agreement.
All CSV/summary evidence in RE02*/review;compact re02_review_summary.json saved.
Sampled six evenly spaced frames from each nominal/random forward video for all
six policies, plus202low full combined/extra-forward clips and controls (21 clips
total). Trained samples remain upright with alternating foot placement, combined
turning visible;untrained/standing controls fall. Sampled frames are not continuous
visual gait certification;no standing clips were generated by this screen plan.
Single-seed numeric success is not a recipe promotion or final validation.

RD13 is a cheapest read-only structural diagnostic before another training choice:
six independent jobs,one per saved RC02 final1146880 critic,8 ORIGINAL random reset
states each(6000,6001,6002,6014 x stand/forward.45). ZERO trajectories/training.
Reconstruct same normalized216-dimensional privileged critic,load saved third
checkpoint component (fail if absent);never initialize a substitute. Preserve
policy/label hashes,initialqpos/qvel,predictions,original RE02 outcomes. Source
inspection confirms actor already observes local linear velocity;no missing-velocity
feature hypothesis. Historical O01 added height/contact and failed;not repeated.
Finite-unroll/discount changes have failed;this audit asks whether learned initial
value ordering even distinguishes the remaining failures before critic redesign.

Frozen rule: for EACH of four failed policy-command cases (101low stand,
202control forward,303low stand/forward), failed6001 value >= maximum value of
same-policy/same-command surviving resets is the prerequisite for a shared
optimistic-value-ranking explanation. Require all four;otherwise reject blanket
explanation,retain per-case values. Ties count as no discrimination. Successful
policies/commands retained as controls,not selected away. Critic predicts discounted
stochastic-policy REWARD,not survival or deterministic returns;rank inversion is
NOT calibration error or causal proof and does NOT automatically authorize training.
If prerequisite holds,measure actual discounted return/critic residual before any
critic intervention;if absent,do not sweep critic width or discounts on this basis.
No numerical gate or scientific failure changed. No rollout graph changes.
Six oneGPU2CPU12GB jobs,20minstage/30minqueue each,reproducible runtime profile.
All ready independent jobs submitted with paced uploads. Four synthetic ranking
cases passed (inversion,noninversion,all-survive,no same-command survivor),syntax
and six queue schemas validated. GPU runtime critic application remains pending;
installed Brax source confirms params=(normalizer,policy,value) and value apply
signature(processor_params,value_params,obs). No package changes.
Command per TAG=101_control,101_low,202_control,202_low,303_control,303_low:
cluster_research.py submit --run RD13_TAG --plan
Documentation/Research/queues/cluster_rd13_TAG.json --cpus 2 --memory-gb 12
--runtime-profile reproducible. Keep hourly diagnostic phase. Final independent
fresh-three-seed/held-out/full numeric/control/visual requirements unchanged.


## RD13 loader repair, no training repeated
All six original RD13 jobs7861/7863/7865/7867/7869/7871 failed in34seconds each
before predictions (.05667GPUh total). Root cause: audit opened inference-only
policy export, which intentionally has normalizer+actor only. The full Orbax
checkpoint directory DOES contain saved critic weights. This is an audit-loader
bug, not missing trained critics or scientific failure. Prior critic restore
statement remains true for full Brax parameter warmstarts.
Repair uses Brax checkpoint.load(full directory), requires3 components and finite
nonempty critic, and checks full actor/normalizer EXACTLY against compact export
before use. No initialized substitute, changed saved weights or new dependency.
All six failed predict logs and actual remote CPU verification saved together in
cluster_research/rd13_loader_repair.json. Repaired loader executed against every
actual checkpoint: all6 exact actor/normalizer parity and finite value-network
predictions at mean observation passed. Runtime reset-state audit still pending.
Syntax and all6 fresh queue schemas validated. Fresh RD13_TAGr1 plans retain
identical states,labels,decision rule,budget and reproducible profile; old outputs
untouched. Six independent2CPU12GB/oneGPU jobs,20minstage30minqueue,uploads paced.
No training or completed evaluations repeated. Hourly evaluation phase retained.


## RD13 rejected optimistic ranking; RN01 numerical-training consistency freeze
RD13 r1 all6 completed84sec each=.14GPUh;48 finite predictions,actor/normalizer
parity passes. All FOUR failed states rank BELOW every same-command survivor:
101low stand -1.976 vs[-.449,-.602,-.269];202control forward .01497 vs
[.94875,.43421,.62502];303low stand -2.093 vs[-.487,-.513,-.282];303low forward
-.40564 vs[1.00751,.43622,.71011]. Frozen prerequisite fails0/4. Blanket optimistic
initial-value explanation CLOSED;no critic-width/discount sweep or return-residual
followup on that basis. Critic is not a survival classifier;no claim of calibration
or causal proof. Full48rows,initialstates/config/hashes:rd13_results.json.

Separately justified RN01 tests the VERIFIED numerical-runtime repair during
TRAINING, not just evaluation. RD10-12 established cross-compilation matmul choices
can alter physical trajectories; RE02 fixed evaluation but existing RB05/RC02
weights were acquired without a pinned algorithm choice. Hypothesis: a consistent
compiler profile across acquisition/continuation/evaluation may improve learning
and robustness. This remains unproven; not a claim that compiler variability
explains every failure. User explicitly authorized fixing/verifying and resuming
research. This is one bounded new numerical-runtime experiment, NOT rerunning
training to recover evaluation, a learning-rate extension, or reopening critic
ranking. RE02 already completed;old outcomes preserved as matched comparisons.

RN01_101/202/303 all FROM INITIALIZATION, seed-matched development comparison to
RB05 + RC02_low (use RE02 low evaluation outcomes under repaired runtime). Only
intended recipe change is --runtime-profile reproducible through ALL stages.
Base original RB05 20,070,400steps,2048env,batch128,unroll20,4updates,lr3e-4,
gamma.97,original rewards/full resets;own-parent late1,146,880steps lr1e-4 as RC02.
Total21,217,280steps each,63,651,840total. No shared trained ancestor;fresh optimizer/
step/PRNG at late stage as before,full critic/actor/normalizer parameter warmstart,
NOT exact resume. Own basecheckpoint0 untrained controls;only fixed late1146880
counts for candidate assessment. No lucky intermediate checkpoint or seed swap.
Original screens,conditional full numeric/control suites,forward reset grid/videos
unchanged. Training argv normalized for paths exactly matches source plans for
all3;gate objects exactly match RC02;3schemas valid. No runner logic change.

Predeclared targeted gain requires ALL3 nominal screens pass AND8/8 randomized
survivors per seed (including both6001commands);otherwise reject this recipe-level
benefit, no flag sweep or unchanged extension. Even targeted gain is NOT a full
pass:all original numeric/control/visual gates must pass before unused finalseeds
and freshheldout grid. Report every seed vs7/8,8/8,6/8 RE02 low survivors,nominal
standing yaw.20972/.12938/.15472,all regressions and compute cost. Preserve default
runtime history,do not select a favorable per-seed runtime. No gate weakening.

Three ready independent oneGPU3CPU16GB jobs submitted with paced uploads;within
job base/late/eval dependencies stay sequential.4hbase,1hlate,7hqueue/8hSlurm caps.
Existing post-first two-consecutiveKL>=.2 stop retained. OOM/runtime errors require
relevant-stage diagnosis;never blind allocation/algorithm retries. Training under
new profile throughput/headroom not yet measured;prior similar complete pipeline
~1-1.5h,so initial hourly phase until measured >=2h justifies actual4h recurrence.
No active model waiting for training. Command per SEED=101,202,303:
cluster_research.py submit --run RN01_SEED --plan
Documentation/Research/queues/cluster_rn01_SEED.json --cpus 3 --memory-gb 16
--runtime-profile reproducible.


## RN01 rejected; RE03 acquisition/retention localization
RN01 all3 completed,51m16/71m16/51m56=2.90778GPUh total. Restore parity passes.
Nominal all8survive but101 stays nearly static(forwardlinear.44680,standyaw.02102),
303standyaw.16540 fails;202screenspass. Randomsurvivors4/8,8/8,8/8 versus RE02low
7/8,8/8,6/8:303gains,101regresses.101fallsstand6000/6001at57/42,forwardat59/42.
202fullnominal24survivors but turnleft/right yaw.28920/.26618>.25;fullrandom24
survivors butworstyaw.37829>.35. Extra12forward allsurvive butmaxlinear.43510.
All original gates remain,no full3seed pass. 15clips sampled at6frames each:
101nominal plantedfeet,random forwardcollapse;202/303samples alternating steps;
202combinedturning visible but fails quantitative yaw. Control clipsfall. Sampled
review NOT continuous visual certification. Metrics/csv/videos/learninglogs saved
RN01*/review and rn01_review_summary.json. Both fixed-final inference paths retained.
Frozen targeted-gain rule fails;training-benefit hypothesis CLOSED,no compilerflag
sweep or unchanged extension. Keep reproducible profile as correctness setting,
not choose old/new runtime by reward/pass. Stability repair != improved policy.

RE03 ZERO-training localization: evaluate EVERY RN01 base final20070400 before
late continuation,each against own basecheckpoint0,original nominal/random screens,
conditional full suites/extra forward grid/videos.Three independent oneGPU3CPU16GB
jobs,4hqueue/2hstage,allready submitted with paced uploads;reproducible profile.
Fixed base checkpoint only,not a search across saved intermediates. Record hashes.
Compare ALL3 against corresponding late results already reviewed. If101walking/
recovery failures already present atbase,focus acquisition rather than another
late tweak;ifbase passes andlate loses,focus retention with separate justification.
For202 quantify whether turn-yaw failures precede late stage;303 same forstandyaw.
Mixed findings require per-seed accounting. This diagnostic never promotes a base
checkpoint or mixes seed winners;no automatic extension or training. Prior RE01
used DIFFERENT policies;do not rerun it. Originalclosedbranchesstayclosed.
All3 schemas validated,gateobjects exactly equal saved RE02plans;no runner logic
changed. Full development and unused final3seed/freshheldout requirements unchanged.
Expected short evaluations:actualhourly cadence retained.
Command perSEED101/202/303:cluster_research.py submit --run RE03_SEED --plan
Documentation/Research/queues/cluster_re03_SEED.json --cpus 3 --memory-gb 16
--runtime-profile reproducible.


## RE03 localizes mixed failures; RB06 one PPO clipping comparison
RE03 completed9m14/23m45/10m04=.7175GPUh. All3 nominalsurvive8;101already static
(forwardlinear.44665,standyaw.01232),random4/8(falls58/42/59/42 forstand/forward
6000/6001). Thus101failure acquired BEFORE late stage,not caused by fine-tuning.
202base screenspass,fullnominalturnyaw.27956/.27414 fail;fullrandomnumericpass24/24,
extra forward11/12(reset6012fall88,linearworst.36775). Late202 fixed that fall but
worsened yaw/fullrandom:retain both effects.303base nominalstandyaw.14108 passes,
random6/8(6001stand64/forward81);late gains2survivors but standyaw.16540 fails.
Mixed acquisition/retention;no basecheckpointpromotion or mixedseedwinners.
All rows in re03_summary.json and RE03*/review.15clips sampled at6frames each:
101nominalstatic/randomcollapse;202/303 stepping;202combinedturning;controlsfall.
Not continuous visualcertification. No candidate has full3seed success.

Saved RN01 BASE learning curves: first reported training epoch(step2867200)
KL means101=.2418,202=.4222,303=.2971,then .032-.061. These are EPOCH means,
NOT proof of a first-update spike; highest earlyKL202 later walks,so no simple
correlation or causal claim. All show large early distribution changes while
acquisition is unreliable. Freeze ONE from-initialization PPO ratio clipping
comparison .1 versus saved .2 (RN01bases measured inRE03),not a sweep. Narrower
clipped surrogate may reduce harmful update incentives while retaining fixed LR;
clipping is NOT a hard KL bound and can slow useful learning. Measure both.
Historical L01 adaptiveLR reachednear-floor and failed;do NOT repeat its scheduler,
target/floor tuning. This differs:existingfixedlr3e-4,onlyclipping_epsilon changes.
No reward/entropy/discount/phase/filter/architecture or reset change.

RB06_101/202/303 FROM SCRATCH20,070,400steps EACH,60,211,200total,2048env,batch128,
unroll20,4updates,gamma.97,originalreward/fullresets,reproducible compiler profile.
No late stage or inheritedparameters. Fixedfinal20070400,own0controls,alloriginal
screens/conditionalfullgates/forwardgrid/videos. Existing train.py also appends
xla_gpu_triton_gemm_any=True for training (same as RN01/earlier);do not claim train
and evaluation flags are otherwise identical or change that flag in this trial.

Predeclared mechanism support: firstepochKL <=75% matched RN01base value for EACH
seed. Behavioral prerequisite:101acquiresoriginalforwardgait gates,202/303retain
nominal gates,all3 nominalscreenspass and randomsurvival no lowerthan4/8,8/8,6/8.
KL reduction ALONE never progress;failure of either prerequisite closes this
clipping hypothesis,no epsilon sweep/unchangedextension. Any improvement still
requiresALL originalfullnumeric/control/visual gates before freshfinalseeds and
freshheldout tests. Report all3,regressions,cost;no luckycheckpoint orseedselection.
Training post-firstconsecutiveKL>=.2 stop unchanged. Four-hourtraining/six-hourqueue
caps,oneGPU3CPU16GB perjob,all3ready submitted with paced uploads. Expected~1h based
RN01measuredcosts,actualhourlyinitialchecks;no activewaiting. Existing trainer
forwards configclipping_epsilon;structured override .1 verified;all3trainingargv
matchRN01base after paths except thatONE addition;gateobjects unchanged,3schemas
validated. No runnerlogic change. Command perSEED101/202/303:
cluster_research.py submit --run RB06_SEED --plan
Documentation/Research/queues/cluster_rb06_SEED.json --cpus 3 --memory-gb 16
--runtime-profile reproducible.


## RB06_202 startup import failure; RI01 bounded environment preflight
RB06_101/303 completed;retain them unchanged and defer jointscientific review.
RB06_202/job7907 failed13sec beforetraining at trimesh/base.py Path3D|None:
Path3D was ExceptionWrapper after a suppressed .path import exception. Original
trace did not expose that cause. Evidence in RB06_202/training_failure.txt.
Shared environment has prior intermittent import/file faults;do not blindlyretry.
Current CPU import succeeds,114trimesh .py files exactly match package RECORD
hashes. This does NOT prove rootcause or historicalfile integrity. Optionalshapely
metadata missing is not itself cause:successful imports tolerate this absence.

RI01 ONE zero-training Slurm allocation,oneGPU2CPU12GB,15minqueue,two sequential
freshprocess imports,5minstage each. No episodes,packageinstalls,flagchanges or
sharedconfig modifications. Trace original exceptions specifically in trimesh/path
before ExceptionWrapper masks them;save all,including normallytolerated missing
networkx/svg/lxml/shapely. Hash all114trimesh sources before/after,compare and capture
versions. Exact failing importentry robocub.locomotion thenMJX/Brax;no env creation.
Diagnostic code actuallyexecuted on remoteCPU:imports pass,hashesmatch,seven
expected optionalimportexceptions captured. Syntax/queueschema validated.
Both allocated-process imports must pass,sourcehashes unchanged acrossprocesses,
versionsmatch recorded environment before ONE identical fresh RB06_202 replacement
may be submitted. This is verification of a healthy executionpath,NOT a proven
repair of intermittentstorage. Keep source/recipe/budget/seed202 unchanged;failed
run had no training torepeat. Completed101/303 remain untouched. If preflight or
replacementrepeats fault,stop allocation retries and require stableisolatedruntime
or admin-supported repair;no automatic package modification/adminmessages.
Preserveall failureevidence. ActiveRI01,command cluster_research.py submit --run RI01
--plan Documentation/Research/queues/cluster_ri01.json --cpus 2 --memory-gb 12
--runtime-profile reproducible. Hourly shortpreflight;actual4h only if unchanged
infrastructure blocker. No three-seed success or partial winner claim.


## RI01 passed; one RB06_202r1 replacement
RI01/job7920 completed44sec=.01222GPUh. Bothfreshprocessimports pass;all114source
hashes identical before/after/across,versions match savedCPUverification,seven
normallytoleratedoptionalimportexceptions each. Rawreports RI01/review.json.
This satisfies frozen healthyexecutionpath prerequisite,NOT underlyingstorage
repair or proof intermittentfault cannot recur. No dependencies/settings changed.
Submit ONE identicalseed202 replacement RB06_202r1 with fresh isolated run/output;
oldjob7907hadno training. Original101/303completed remain untouched;reviewALL3
jointly whenreplacement completes. Ifimportfault recurs,stop allocationretries
and require stable isolatedruntime/adminsupportedrepair;no blindinstallation.
Plan equality verified after onlyoutputpath replacement;schema validated. Original
scientificbudget,commands,gates/runtimeprofile unchanged. Commands:
cluster_research.py submit --run RB06_202r1 --plan
Documentation/Research/queues/cluster_rb06_202r1.json --cpus 3 --memory-gb 16
--runtime-profile reproducible. Hourly shortbatch checks;no active trainingwaiting.


## RB06 rejected; RD14 normalization-only startup diagnostic (Nov12)
All three complete:7905/7922/7909,42m55/42m45/43m24 =2.15111GPUh.
First reported epoch KL .233642/.408622/.284707 versus RN01base
.2418/.4222/.2971:ratios .9663/.9679/.9583,all fail <=.75 mechanism rule.
Later KL .0145-.0283. Both mechanism and behavior prerequisites FAIL.
All nominal episodes survive but all nominal screens reject:stand yaw ranges
101 .2000-.2235,202 .2278-.2285,303 .2109-.2390,all above .15.
Forward yaw202 .3007-.3305 and303 up to .2683 also fail.101 acquires
forward tracking (linear .0968-.0998) and random survival improves4to8;
202 regresses8to6,303 remains6. Random6001 falls202 stand64/forward65,
303 stand51/forward65;101 randomstand6001yaw .37592 fails .35.
All full suites skipped. No three-seed solution, no epsilon sweep or extension.
All48 trained screen rows and learning curves reviewed;rawsummary
Artifacts/Results/cluster_research/rb06_review_summary.json. Six forwardclips
sampled at six evenly spaced frames each:all show alternating foot positions,
continued upright progression in nominal/random6000. Sampled frames are not
continuous visual certification;no stand videos were generated by these screens.

Cheapest distinct follow-up:source inspection verified fixed-LR Brax updates
normalizer BEFORE SGD, but collected old distribution parameters use prior
normalizer. Reported analytical KL therefore includes preprocessing changes,
not just optimizer movement. This is not a confirmed implementation defect or
cause of behavior failures. Remote source hashes:train32ed2754f3f9b545c7f381aba9953d1160ca82cc0d01deb59d1f5847872eea32,
lossesb6dfe0aefeb1e3d11252fc40c6ec4512757fc7fccc0b3e37ec9ccd349351007c.
Prior RC02 normalizer-drift audit concerned late warmstarts, not initialization.
L01 adaptive-LR failure remains closed;no schedule/floor/normalization change.

RD14:THREE zero-training/reset-only audits of saved RN01 initialcheckpoint0,
seeds101/202/303,2048 original training reset observations each. Match original
single-process reset-key derivation and random commands;no fixed test resets,
no episodes,optimizer updates,parameter changes or reward intervention. Compare
analytic old||new action-distribution KL from one running-statistics update
against unchanged-normalizer identity,actorweights fixed. Same KL reduction
as installed loss. Store observations/config/checkpoint hash/allmetrics.
This is NOT actual first20step rollout data or an epoch-KL decomposition.
Frozen rule:finite metrics,identity logits bitwise equal and identity KL matches
the installed formula floor(actiondim*log(1+1e-5)) within1e-5,normalization-only mean KL
>=.2 for EACH of three policies motivates ONLY separately frozen first-batch
instrumentation,never automatic training. Any failure rejects this blanket
reset-only justification;no observation subsets,seeds or threshold sweeps.
Even a positive result cannot establish normalization causes learning failures.
OneGPU2CPU12GB each,20minstage/30minqueue,all3ready submitted with paceduploads.
No training repeated;preserve all failures/M1. Hourly short diagnostic checks.
Commands:cluster_research.py submit --run RD14_SEED --plan
Documentation/Research/queues/cluster_rd14_SEED.json --cpus 2 --memory-gb 12
--runtime-profile reproducible. Original full gates unchanged.

Prelaunch unit test exposed the installed KL formula positive identity floor:
Brax new.kl_divergence(old) computes old||new with log(scale ratio+1e-5),
not new||old. Corrected direction and identity criterion BEFORE launch;original
PPO metrics unchanged. Initial failed unit test retained in development history.
Real Brax/JAX compiled synthetic-network test passes:identity floor,finite shift,
unchanged actorweights/normalizer verified. All3 queue schemas validated. Actual
G1 reset execution remains pending in the allocated jobs;no runtime pass claimed.


## RD14 prerequisite rejected; RD15 signed standing-yaw discrimination
RD14 all3 completed104/124/113seconds =.09472GPUh. Normalizer-only KL
101 .14235094,202 .13237850,303 .15526584,ALL below frozen .2 threshold.
Identity logits exactlyequal;KL .00028981/.00028983/.00028978 matches expected
.00029039 floor within1e-5;all metrics finite. Reports/observations saved under
Artifacts/Results/cluster_research/RD14_SEED/review/audit. This rejects blanket
reset-only normalization justification. No first-batch follow-up or normalization
training change;do not lower threshold/select observation subsets. Does not prove
normalization has no effect on real rollout batches. No behavior validation.

RD15 freezes a different measured question:RB06 all3 standing RMSE .20-.24 and
single-support .796-.826 persist, but RMSE cannot distinguish signed directional
bias from alternating yaw. Existing RB06 clips cover forward only,not standing.
Do NOT reopen phase/filter/coefficient/sampling hypotheses or repeat RD01 spectra.
Use UNCHANGED official fused evaluator,deterministic,original nominal6000,stand,
500steps,each RB06 fixedfinal20070400. Original untrained0 and standing controls
also captured by evaluator;9 trajectories total,three trained. New standing
videos/numeric traces,zero training/intervention. OneGPU2CPU12GB perjob,
40minstage/60minqueue,three independent ready jobs with paced uploads.
Before interpretation,trained500steps and yaw/linear RMSE must reproduce saved
screen6000 within rtol1e-5/atol1e-6;otherwise reconcile provenance before training.
Compute WHOLE and last250 signed yaw mean,RMS,abs(mean)/RMS,contact fractions,
linear error,height. Never use shortened episodes as settledwindows. Frozen
sustained-bias prerequisite:ALL3 survive500,height>.6,last250 abs(mean yaw)>=.1
AND abs(mean)/RMS>=.75. If met permits separately frozen left-right structural
symmetry audit,NOT automatic augmentation/training or inference correction.
Otherwise reject blanket directional-bias account;retain per-seed outcomes.
No threshold/seed/phase sweep. Full original gates unchanged;this is diagnostic,
not validation or a recipe promotion. Review all3 standing videos/metrics/cost.
All existing closed branches remainclosed. Commands cluster_research.py submit
--run RD15_SEED --plan Documentation/Research/queues/cluster_rd15_SEED.json
--cpus 2 --memory-gb 12 --runtime-profile reproducible. Hourly shortchecks.
Existing official evaluator unchanged;schema/path/arguments checked prelaunch.


## RD15 rejects sustained directional bias; RD16 gradient compatibility freeze
All3 jobs7930/7932/7934 complete284sec each=.23667GPUh. All reproduce RB06
nominal6000 trained500step,linear/yaw metrics within frozen tolerance (values
exact). Whole yaw mean101-.03111,202-.05255,303+.08367;last250 means
-.03503/-.05475/+.08313,RMS.20185/.22781/.23731,ratios.17353/.24034/.35028.
All fail both signed-bias thresholds. Minimum whole height>.752;last250
single-support.816/.800/.816. Oscillatory motion dominates;no symmetry followup
on directional-bias premise. No phase/filter/threshold sweep. Three newstanding
videos sampled6frames each:upright alternating foot placements,303visible net
turning;sampling is not continuousvisual certification. Raw rd15_summary.json.
Cost includes matched controls;no training or validation pass.

Read-only saved returns compare RE03 RN01base vsRB06 standing at all4resetseeds
for EACH policy family. Quieter RN01base has HIGHER original reward in all12
matched episodes;1016000return13.3744 versus10.0591,yaw.01060vs.20003.
2026000return11.0801vs10.4093,30311.9327vs8.9495. Artifact
rd15_saved_return_comparison.json. This comparison does not justify reward tuning
or prove global objective alignment;A13 and all closed reward branches remainclosed.

Distinct RD16 hypothesis:local standing and forward PPO policy-loss gradients
conflict in shared actor parameters. Observed standing/walking acquisition tradeoff
motivates measurement,NOT a proven shared-network defect. THREE zero-optimizer
jobs on ALL RB06final20070400,fullsavedcritic/actor/normalizer parity required.
Each collects64nominal environments forstand0 andforward.45,forBOTH predetermined
stochastic draws17/29. Percommand/draw100burnin+20collectedsteps,500episode wrapper,
coherent fullautoreset,original physics/rewards.92160total simulationtransitions
includingburnin across3jobs,15360lossbatchtransitions. This is command-stratified
stochastic diagnostic data,NOT original training distribution or validation.
Record burnin/batch terminalcounts;burnin is not assumed settled afterresets.
Frozen normalizer/critic/weights,no optimizer,entropyexcluded to isolate PPO policy
loss;originalgamma.97,GAE.95,clip.1,normalized advantages. Save fullcollectedbatches,
gradientvectors,losses,norms,cosines,sourcehash/config. Checkallbatchcommands.
Frozen prerequisite:finite nonzero gradients and cosine<=-.1 for BOTH draws of
EACH actor (all6pairs). Positive result permits separately frozen architecture
DESIGN review,not automatic splitcontroller/training. Anyfailure rejects blanket
shared-actor interference explanation;no extra draws/commands/thresholds.
No causalclaim from localcosines;critic error and sampling remain confounders.
OneGPU2CPU12GB each,50minstage/60minqueue,all3ready paceduploads. Originalgates
unchanged. Reusefullcheckpoint loader previously runtimeverified;2newlocaltests
pass actualcompiledBraxPPO gradients withoutparameterupdates and cosine/degenerate
controls. G1batchexecution pending;do not claim runtimepass. Plans validated before
commit/launch. Hourly shortchecks;no activewaiting. Commands cluster_research.py
submit --run RD16_SEED --plan Documentation/Research/queues/cluster_rd16_SEED.json
--cpus 2 --memory-gb 12 --runtime-profile reproducible.


## RD16 rejects shared-actor conflict; RD17 saved-batch gradient budget
All3 complete544/524/524seconds=.44222GPUh. Actor cosine pairs17/29:
101+.28152/+.35613,202+.46485/+.35884,303+.37122/+.34433;ALLpositive,
none meet<=-.1 frozen prerequisite. All gradients finite/nonzero,actor norms
2.821-3.566;all12 batches have ZERO burnin or collected terminations. Full/compact
checkpoint parity passes. No shared-actor separation on this premise,noextra
samples/commands/thresholds. Stochastic nominal survival here is not validation.
Raw batches/gradients preserved,RD16_SEED/review/audit;rd16_summary.json.

Distinct implementation mechanism:installed Brax trainsactor/critic in oneparameter
pytree and appliesoptax.clip_by_global_norm(1) beforeAdam to total PPOlossgradient.
RD16 excluded critic/entropy to isolate policy conflict,so did not measure global
clipping budget. RD17 replays exactly ALL12 saved RD16batches on samefullsaved
RB06finals,NO newrollouts oroptimizerupdates. Restore normalizer/critic/actor,
verifycheckpoint matchesbatchhash,reproduce saved policy_loss within1e-6/1e-5.
Original total PPOloss includesentropy.005,valuecoefficient.5,normalizedadvantages,
gamma.97,GAE.95,clip.1,rewardscale1. Save actor/critic/global gradientnorms,critic
actor ratio,and impliedsharedclipmultiplier. NoactualAdamupdateorhistoryclaim.
Frozen dominance prerequisite:EVERY12batch critic/actor norm>=10 AND sharedclip
multiplier<=.1,finite nonzeroactor andlossreplaypasses. Ifmet permits separately
frozen optimizer-coupling DESIGN review,notautomatictraining. Adam maycompensate
common scaling;gradientdominancealone cannotprove actorupdate starvation.
Anyfailure rejects blanket critic-dominance mechanism;no valuecoefficient/clip
threshold sweeps ornewbatches. Priorcritic-ranking branch remainsclosed.
Three independent oneGPU2CPU12GB jobs,15minstage20minqueue. Noassetfetch/simulator
required. Fullweights/batches remainread-only. RealBrax localtest checksfull-loss
gradient norms,globalnormcomposition,clipfactor,policy-lossreplay withoutupdates.
Plansvalidate beforecommit/launch;actualremoteexecutionpending. Hourly shortchecks.
Commands cluster_research.py submit --run RD17_SEED --plan
Documentation/Research/queues/cluster_rd17_SEED.json --cpus 2 --memory-gb 12
--runtime-profile reproducible. Alloriginalgates/final3freshseedcriteriaunchanged.
