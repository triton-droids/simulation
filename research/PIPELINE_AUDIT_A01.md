# Pipeline audit and staged screening, 2026-09-24

## Verdict and scope
C16 rejected: nominal survival24/24 but yaw/standing gates fail; randomized
12/24,mean281.667 steps,linear/yaw RMSE .65758/.54722; extra forward10/12.
Sampled nominal combined6000 and forward6010 frames show upright stepping,
not a full visual pass. No more local reward/entropy sweeps are justified now.
No new implementation defect has been established by this audit. Runtime
inference and screening checks below are pending in A01. No training authorized
by the A01 plan (zero training steps,4-hour cap).

## Evidence and distinctions
- Resets: adapter delegates random resets to pinned upstream Joystick.reset;
  upstream samples joint multipliers .5..1.5 and six root velocities -.5.. .5.
  Native reset range fields are NOT forwarded. This known configuration trap
  makes a config-only gradual reset curriculum ineffective. Candidate selection
  defaults1; evaluation forces1. Nominal reset explicitly rebuilds data/obs.
- Observations:103 actor/216 critic; transition command, previous action,phase
  and critic air time are synchronized. Six selected adapter tests passed on
  WSL, including physical-terminal versus timeout reset bookkeeping at lengths1/3.
- Normalization: train.py passes trainer_runtime_controls, which explicitly
  forwards normalize_observations=true. Evaluation builds the same normalizer
  preprocessing, actor key/state and critic key/privileged_state. A01 runs the
  existing saved-policy inference parity audit with64 inputs and a deliberately
  wrong preprocessing negative control. D06 did not find extreme reset scaling.
- Actions: C10 config scale .5,10 simulation frames, action_repeat1, matching
  upstream default scale. Adapter clips normalized actions before upstream PD
  targets. ONNX reference uses the same adapter; D09 raw actions remain within
  bounds. No evidence of action saturation explaining those failures.
- Termination: upstream ends on inverted torso OR specified foot-foot/foot-shin
  contacts OR invalid state; adapter also catches infinities. It does NOT end
  merely at pelvis height .6; .6 is a stricter evaluation gate. Thus a terminal
  at high pelvis height is not itself a bug. Existing D09 trace summaries lack
  which termination sensor fired. Do not call every termination a fall.
- Truncation: full reset capture/restore preserves steps,truncation,episode_done
  and accumulators; selected tests cover physical versus horizon termination.
  Pinned Brax PPO losses use termination=(1-discount)*(1-truncation), masking
  truncation in GAE. No local reversal of terminal/truncation found.
- Restore: C16 restore_parity.json reports exact17 actor/normalizer leaves.
  Linux checkpoint staging is already verified. Warm starts do NOT restore
  optimizer/step/PRNG; these experiments are parameter continuations, not exact
  interrupted-run resumes. This is a training-design difference, not silent
  corruption. Network layers512/256/128 match saved config/evaluation.
- Reference: ONNX same-environment success establishes feasibility of these
  starts, not equivalence of its training recipe/normalization/optimization to
  ours. Do not infer its training history from the exported policy or use it
  as imitation labels. Training samples commands; evaluation holds them fixed:
  intentional held-command task, not an observation mismatch.

## Screening contract
Before full development evaluation: stand and forward.45, seeds6000,6001
(ordinary) plus6002,6014 (difficult), nominal AND randomized,500 steps.
All trained/untrained/standing rows retained:48 episodes versus180 full rows.
Use existing applicable per-regime numeric rules including survival,height,
tracking and forward gait; remove only command-specific rules absent from the
screen. Missing/duplicate/nonfinite evidence is an error. Both screens must pass
before full180 rows execute; pass merely permits full evaluation, never promotion
or final success. This screen may conservatively reject candidates; never relax
it in response to a particular candidate. Full independent validation remains
unchanged. A01 applies this to C10 without retraining to verify rejection and
skip behavior on real data; full stages are frozen and conditionally reachable.
Runner subprocess tests verify rejected-screen dependents do not execute.

## Next decision and budget
First review A01 inference parity and actual screening/skip evidence. If parity
fails, repair only its cause and rerun the audit without training. If it passes,
cheapest next diagnostic is terminal-cause/contact/attitude timing at the two
known failures and an ordinary start (at most6 paired500-step trajectories,
zero training). Prioritize this over new reward coefficients: D09 seed6002
terminates32 steps with minimum pelvis .729, which warrants identifying the
specific contact versus inversion trigger.
Then assess an early gradual reset curriculum: it requires actual state-level
velocity/joint scaling with coherent recomputed sensors/observations, endpoint
parity tests and original evaluation resets. Start modest and increase towards
original reset range during initial learning, not only after nominal walking.
A binary randomized-from-scratch attempt already failed; do not repeat it as a
new idea. Freeze one matched pilot and control, each at most3,368,960 training
steps, only after diagnostics justify exact curriculum and budgets. Screen both;
if neither improves difficult-start survival without ordinary regression, stop
that hypothesis. Do not launch this proposed pilot automatically from A01.
All failed evidence, final control superiority, fresh held-out tests, visual review
and three predetermined fresh independent training seeds remain mandatory.

## A01 runtime evidence and A02 freeze
A01 inference parity: max action difference0 over64 observations; deliberately
wrong identity preprocessing differs1.9222, so the audit detects that fault.
Nominal screen passes; randomized fails (min31 steps,mean274.625). All three
full jobs are skipped_screening:180 additional evaluations avoided. No full pass.
A02 records torso gravity and each pinned termination contact sensor in every
trace, asserts their OR plus finite-state guard equals actual done at every step,
and reports cause/first trigger. No physics, policy, reward or terminal-rule edits.
Saved C10 checkpoint1003520, original randomized forward.45 starts6000,6002,
6014, paired with diagnostic reference,500 steps each: six trajectories maximum,
zero training,90-minute cap. Tests exercise independent sensor addresses,
inversion and infinities; runtime parity additionally tests actual environment.
No new gait videos generated by A01; prior C10 visual samples already reviewed.
Review causes before freezing a curriculum pilot; do not weaken contact rules.

## A02 results and P01 matched pilot freeze
C10 ordinary6000 survives500.6002 terminates31 via foot-foot contact while
minimum pelvis .724;6014 terminates57 via foot-foot contact after height drops
as low as .107. Neither terminal reports inversion/nonfinite. Reference survives
500 at all3. Do not conflate collision trigger with root cause or weaken it.
P01 tests earlier gradual reset learning against full-disturbance-from-start,
not another late reward tweak. Both from scratch developmentseed11,three
parameter-warm-start stages1003520 steps each (3010560 per arm;6021120 total).
Both identical PPO/reward settings:lr.0003,gamma.97,horizon500,original full
command ranges,linear3,yaw.75,phase3,contact-phase2,orientation-2,otherdefaults.
Curriculum joint deviations and root velocity scale .25,.5,1;control1,1,1.
Random world translation/yaw retained. Whole state/sensors/observations rebuilt
when scaled; scale1 preserves upstream path. Both arms incur the same optimizer
restart schedule. Actor/normalizer warm-start parity and cross-arm initial-policy
parity are asserted. No selection of training seeds or use of oracle labels.
All evaluation forces disturbance scale1 and candidates1. Frozen48-row screen
per arm; full180 rows only if both screens pass. Forward videos in screens.
Commands research/queues/post_f1_p01.json;10-hour cap. No automatic extension.
Compare difficult-start survival and ordinary-start regression on original
resets. If neither arm improves these jointly, abandon this curriculum pilot;
if improved but gates fail, record specific failures before any new hypothesis.
No claim this short pilot must solve full locomotion. Final recipe, all gates,
fresh held-out tests and three new independent training seeds remain required.
Validation: real MJX reset scales0/.5/1 passed qpos/qvel scaling, unchanged root pose, finite observations, recomputed observation coherence and exact upstream actor observation parity at scale1 (results/reset_curriculum_audit.log). Queue validation and diff checks passed.

## P01 rejected; P02 collision-precursor hypothesis
P01 curriculum/control randomized survivors0/8 each,mean171.75/170.125;
nominal1/8 versus0/8,mean367.75/92.25. Curriculum improves nominal duration
but not recovery without ordinary regression versus C10. Sampled forward6000
videos show crouching/unstable steps; no pass. Abandon this pilot, no extension.
Screening skipped all six full jobs (360 additional evaluation episodes).
Post-hoc A02 trace geometry, heading-frame signed left-minus-right foot width:
ordinary C10 min.173m; difficult6002 min.066, below.10 at step28 then collision31;
6014 min-.023, below.10 at step52 then collision57. Reference minima.128/.208/.146
and survives all. These selected traces motivate a hypothesis, not causal proof.
P02 adds bounded squared deficit clip((.16-width)/.16,0,1)^2 with weight-4,
control_dt scaling once; no cost at width>=.16. Measured in root yaw frame,
left/right pinned feet sites, global yaw/translation invariant. Applies both
standing and moving. Contact and termination rules unchanged. Risk: overly wide
or rigid gait; final command/gait gates remain. Reference only informs diagnosis,
never targets or imitation data. Default weight0 preserves existing behavior.
Single bounded C10-parent continuation,seed11,1003520steps; all C11 recipe
settings otherwise unchanged. Same-parent C11 supplies historical control.
48-row frozen screen,180 full rows only after both screens pass,5-hour cap.
No unchanged extension if difficult survival fails to improve without ordinary
regression. Tests verify bounded geometry and yaw/translation invariance; actual
MJX reset/step required before launch. No final independent seed success claimed.
P02 validation: geometry unit test, real MJX reset/step smoke (results/narrow_feet_smoke.log), queue validation and diff checks passed.

## P02 rejected and A03 action-sampling diagnostic
P02 nominal8/8 survive but standing yaw .15675 exceeds .15; randomized4/8
survive just as C10,mean276 versus274.625. Difficult forward6002/6014 lasts
41/55 versus31/54. No survival success; abandon separation-cost pilot, no
extension. Sampled ordinary forward videos show stepping, not a full gait pass.
Full180 evaluations skipped. Original terminal and evaluation rules unchanged.
A03 evaluates frozen C10 stochastically at reset6000/6002/6014 with independent
policy RNG keys17/29/43,forward.45,500steps. Nine local trajectories,zero training,
2-hour cap. Existing A02 deterministic/oracle data remain matched references;
do not rerun reference for each key. Record every trajectory; no lucky-draw
selection or promotion. Question: does the trained action distribution contain
recoveries that its deterministic action choice fails to reproduce? This is
not a claim of inference bug; stochastic training and deterministic deployment
are intentional different modes. Final evaluations remain deterministic.
If all difficult draws fail, close this explanation; if recoveries appear, report
frequency and ordinary regression, then design a bounded learning/distribution
experiment rather than changing the final evaluation mode. Freeze decisions only
after all9 results. CLI defaults preserve original behavior. Verify saved-policy
same-key repeatability/different-key sensitivity/deterministic invariance before
launch; per-transition terminal decomposition checks remain active.
A03 validation: saved-policy repeatability, RNG sensitivity and deterministic invariance passed (results/a03_sampler_check.log); Python syntax, frozen queue and diff checks passed.

## A03 closed; B01 uninterrupted baseline feasibility probe
A03 ordinary6000 survives3/3 but yawRMSE .424-.447; difficult6002 lasts31/31/63
and6014 lasts40/57/47,zero6 survivors. All report foot-foot contact;6002/key43
also inversion. No stochastic recovery evidence; close this explanation, retain
all draws. No new videos from this numeric diagnostic; prior C10 reviewed.
Budget audit: DECISIONS.md line444 and EXPERIMENT_PLAN.md line510 record pinned
upstream200M-step/8192-env profile. Recent P01 only3.01M per arm and optimizer
restarts. These are not comparable training budgets; no proof of impossibility.
B01 is one prospectively bounded uninterrupted20070400-step from-scratch
seed11 development probe,512envs,batch32*16,unroll20,8evals,lr3e-4,gamma.97,
entropy.005,original default reward scales (phase1,contact-phase0,narrow-feet0),
full randomized starts with scale1/candidate1,upstream command ranges vx[-1,1],
vy[-.5,.5],yaw[-1,1]. Noise,pushes,domain randomization remain disabled;
horizon500 rather than upstream profile. This is NOT exact upstream replication,
and changes versus P01 prevent attributing outcomes solely to budget.
No warm starts, optimizer resets, oracle labels or parameter sweeps. Final20070400
checkpoint fixed before run; ignoreKL only through first2867200 callback; existing
sustainedKL/nonfinite guards remain.6-hour queue cap,4-hour training cap.
Screen fixed48 original rows,own checkpoint0 untrained reference,then conditional
full180 with original thresholds. No checkpoint cherry-picking. If both screens
fail/no recovery gain, do not extend unchanged; review learning curve and stop
this baseline hypothesis or identify a specific justified redesign. A pass only
permits full development evaluation and later fresh independent validation.
This replaces short repeated fine-tuning with one bounded budget/recipe probe.

## B01 recovery gain and B02 matched yaw correction
B01 fixed-final checkpoint: nominal8/8 and randomized8/8 survive500; random
height .7033,mean/worst linear .20560/.29515 pass. Both screens still reject:
nominal mean yaw .26840,stand worst .30174; randomized mean yaw .28197.
Sampled ordinary forward videos show upright alternating steps,not full visual
validation. Full jobs correctly skipped. Recovery improved versus C10's4/8 on
identical screen. Do not call this full command-grid or independent-seed success.
Learning curve internal mean survival53->76->182->406->470->500->500->445;
last decline retained, no checkpoint selection. Budget/recipe changes mean
recovery cannot be attributed to training duration alone.
B02 freezes two matched continuations from B01 final20070400,seed11,1003520
steps each,lr1e-4 (both),gamma.97,original B01 command ranges/rewards except
prospective yaw weight3 versus.75 control. Actual matched argv equivalence
checked after output paths and yaw weight removal. Exact restore audits.
48 screening rows each,full180 each only if both screens pass. Reference remains
B01 checkpoint0 for both arms; original horizon500 and all thresholds unchanged.
8-hour cap,2,007,040 new training steps total. No unchanged automatic extension.
Select by all screen/full/control/gait evidence,not reward or yaw alone. If yaw
improves at cost of recovery, reject; if neither clears gates, record remaining
failure before another specific hypothesis. Three fresh training seeds and fresh
held-out grid remain required after development; all B01/B02 are development.
No runner/physics code changes. Plan validation,diff check and paired-command
assertion required before commit/launch.

## B02 reviewed; B03 stand-command reward conflict
B02 yaw3 random7/8 survive,mean446.125;nominal8/8 but stand yaw .1900>.15.
Reject yaw3 because recovery regresses. Control random8/8 and all random screen
numeric gates PASS (linear/yaw means.21949/.22617); nominal8/8 survival but
stand linear .15894 and yaw .22117 fail. Full jobs skipped; no full-gate pass.
Sampled ordinary random-forward videos show upright alternating support.
Nominal stand single-support .824-.834 and completed air .28-.34s indicate
continued stepping. Upstream _reward_feet_phase gates on commanded OR actual
motion>.1, so movement can sustain phase-height reward at zero command. This
is an intentional upstream objective, not a software bug; hypothesis: conflict
with strict standing precision. Do not weaken stand gates or contact rules.
B03 compares command-only phase gating vs unchanged mask from SAME B02control
final1003520,seed11,1003520steps each,lr1e-4,gamma.97,yawweight.75,all other
settings identical. Subtract only weighted phase term*dt at command norm<=.01;
moving reward,physics,terminal rules remain unchanged. Defaultfalse preserves
historical behavior. Both training/evaluation use the saved recipe; gate metrics
exclude shaped reward. Real-MJX stand/move parity smoke required before launch.
Nominal screen videos now show stand; random screen videos forward. Original48
screen rows and conditional180 full rows each,8-hour cap. No unchanged extension.
If standing improves with any recovery loss,reject; full validation and all3
fresh independent training seeds/held-out grid still mandatory. Reference remains
B01 true initialization checkpoint0. No winner selection from reward alone.
B03 validation: real-MJX stand/moving reward and physics/termination parity passed; three adapter tests including exact nonzero phase subtraction passed. Queue validation and diff checks passed.

## B03 review and B04 matched stand-pose test
B03 both arms survive all16 screen episodes and pass random numeric screen.
Command-only phase arm nominal worst stand linear/yaw .15602/.16691 versus
control .15544/.17848; both exceed .15. Random mean yaw .18217 versus .19778.
Nominal stand sampled video frames show continued stepping in both. Phase gating
modestly improves yaw without measured recovery loss,not a full-gate pass.
B04 compares existing stand_still weight-3 versus-1 from SAME B03phase_gate
final1003520,seed11,1003520steps each,lr1e-4,gamma.97,yaw.75,phase gatingtrue.
Only stand weight differs; normalized training argv equality asserted. Upstream
stand_still is sum(abs(jointpos-defaultpose)) only when command norm<.01.
It penalizes pose deviation,not velocity directly. Hypothesis: reduce residual
zero-command stepping. Risk: impeding recovery during zero-command disturbances;
reject any lost survival/random gate or moving gait/tracking regression.
Original fixed48-row screens and conditional180 full evaluations each;8-hour
cap,2,007,040 training steps total. Exact actor/normalizer restore audits; no
new runner/physics code. B01checkpoint0 reference. Do not weaken .15 stand
thresholds or call passing random screening final validation. If neither clears
nominal standing,close this coefficient test and reassess mechanism; no blind
weight escalation or automatic unchanged extension. Fresh independent3-seed
validation required after any full development pass.


## B04 rejected; A04 frozen standing mechanism diagnostic
Both arms survive all16 screening episodes. Stand3 nominal worst stand linear/yaw
.16223/.29223 and random mean/worst yaw .25906/.36578 fail. Control passes random
screen but nominal stand .15420/.18109 fails. All full evaluations skipped.
Sampled frames from both nominal stand and random forward videos show ongoing
zero-command stepping and upright forward stepping. This is not final visual
validation. Close this stand-pose coefficient hypothesis; no weight escalation.
A04 uses zero training steps: fixed B04control final1003520, stand command0,0,0,
seed6000 nominal and randomized original resets,500 steps each, deterministic.
Compare each to the verified reference in the same MJX task (4 trajectories total).
Plan: research/queues/post_f1_a04.json;90-minute queue cap;40-minute stage caps.
No physics/reward/inference changes, no fitting to reference actions. Record mean
velocity versus oscillatory variance, contacts/phase, action saturation and weighted
reward terms. Compare first100 and last250 steps to distinguish initial settling
from persistent motion; verify zero-command phase reward is actually zero.
Decision: an observed parity/implementation defect requires its smallest verified
repair; otherwise persistent stepping with phase reward zero motivates considering
one matched mixed-command standing-exposure test (not coefficient escalation).
If error is confined to initial settling, target recovery/transients instead. Do
not launch either training hypothesis until traces support it and a budget is
frozen. No claim that reference shares our training architecture. All original
screen/full/standing/control/gait gates remain unchanged; diagnostic windows do
not replace whole-episode scoring. Three fresh independent seeds/held-out grid
still required after a full development pass. Retain all failures.


## A04 reviewed; B05 mixed standing exposure frozen
All four diagnostic trajectories survive500. Local phase reward is exactly zero,
no action saturation. Nominal first100/last250 linear RMSE .13695/.13517,
yaw .20117/.14655; final mean lateral velocity-.06280, yaw mean.02132/std.14499,
single support.828. Random final250 is similar (.13414/.14488, support.828).
Persistent stepping/oscillation remains, while reset transients add error. Reference
also keeps stepping and fails standing precision (nominal final linear/yaw
.18990/.18762); it is not a standing target or label source. No newly found defect.
Do not replace whole-episode gates with favorable settled windows.
B05 compares30% zero commands versus original10%, same B04control final1003520,
seed11,1003520steps each,lr1e-4,gamma.97,phase gatingtrue,stand penalty-1.
Original moving command ranges, rewards, reset distribution unchanged. Uses pinned
upstream sampler's fourth random key with larger Bernoulli threshold; preserves
all retained moving samples and default10% behavior. Applies on reset and command
resampling through the upstream method. Held-command evaluation remains unchanged.
Hypothesis: more mixed standing exposure reduces persistent stepping without
forgetting movement. This is not the old zero-command-only training pilot.
Total2,007,040 new steps,8-hour queue cap,original48-row screens and conditional
180 full evaluations per arm; true untrained reference B01checkpoint0. Restore
parity mandatory. If neither clears nominal standing without recovery/moving
regression, close this exposure hypothesis; no automatic unchanged extension or
probability sweep. Both arms reviewed, never select reward alone. Full original
control/gait gates and three fresh training seeds/held-out grid remain mandatory.
B05 verification:16 adapter tests passed; additional real pinned upstream sampler test passed, confirming30% exposure and exact retained moving samples. Queue validation,paired argv assertion and diff checks passed.


## B05 exposure hypothesis closed; B06 targeted yaw damping frozen
Both arms survive16/16 screen episodes and pass randomized numeric screening.
Nominal stand linear passes:30% .13411 versus control .14526; yaw still fails:
.20471 versus .18867 (limit.15). All full evaluations skipped. Sampled stand
frames show continuing steps and forward frames upright stepping; no final visual
pass claimed. Close standing-exposure hypothesis: no probability sweep/extension.
A04 yaw variance dominates mean yaw; B05 leaves yaw as nominal screen failure.
B06 tests an explicit squared pelvis gyro-z cost ONLY at zero command(norm<.01),
weight-1 versus0. This differs from failed global yaw reward3: moving/turning reward
unchanged. Same B05control final1003520 parent,seed11,1003520steps each,lr1e-4,
gamma.97,phase gatingtrue,stand pose-1,standing exposure10%. Pre-transition command
and post-transition gyro; cost*control_dt added once. Default0 is historical parity;
reset initializes metric to preserve JAX tree; saved recipe used in evaluation.
No changes to physics,termination,commands,reset distribution or evaluation gates.
Budget2,007,040 training steps total,8-hour queue cap; fixed48 screening rows each,
conditional180 full rows each; B01checkpoint0 untrained reference; restore audit.
Reject yaw gains that lose standing linear precision,recovery or moving gait.
If neither arm passes both screens, close this damping hypothesis; do not escalate
coefficient or continue unchanged. Review remaining objective conflict before any
further training. Partial screens never equal full pass; original control/visual
and three fresh independent training seeds/held-out criteria remain mandatory.
B06 verification:3 focused adapter tests passed; real MJX reset/step metric structure and stand/moving reward/physics/termination parity passed (results/b06_stand_yaw_smoke.log). Paired argv assertion, queue validation and diff check passed.


## B06 closed; remaining objective conflict and B07 freeze
B06 both16/16 screen survival, nominal stand linear .14729/.14157 passes but
stand yaw .24195/.21583 fails. Damping random screen passes; control worst yaw
.44120 fails. All full evaluations skipped. Sampled stand/forward videos retain
stepping and upright walking. Close damping hypothesis; no coefficient escalation.
Objective review of existing A04 traces (not repeated simulation): settled stand
weighted feet_air_time reward+.01216 in both reset regimes despite phase reward0.
Pinned upstream _reward_feet_air_time explicitly deletes/ignores command. Other
positive terms are tracking rewards; pose penalty dominates costs at~-1.023.
This small air-time incentive is a real objective conflict, not an implementation
bug and not proof it causes the plateau. Removing it is a bounded causal test.
B07 command-gates feet_air_time at norm<=.01 versus unchanged upstream. Includes
both positive and negative air-time terms; moving/turning rewards unchanged.
Same B06control fixed-final1003520 parent,seed11,1003520steps each,lr1e-4,gamma.97,
phase gatingtrue,stand yaw0,stand pose-1,standing exposure10%. No cherry-picked
checkpoint. One paired budget2,007,040 steps,8-hour cap. Fixed48 screen rows and
conditional180 full rows each,original gates; B01checkpoint0 untrained reference.
Pre-transition command,weighted delta*dt once; defaultfalse restores old behavior.
No reset,physics,termination or observation changes. Restore audits mandatory.
If neither passes both screens, close this objective-mask hypothesis; do not run
another mask/coefficient sweep or extend unchanged. Before further training assess
whether the learned periodic gait representation, rather than more reward tuning,
requires a bounded structural diagnostic. Reject stand gains with recovery/moving
regression. Full original numeric/control/visual gates and three fresh independent
training seeds plus held-out tests remain required; screens are not full passes.
B07 verification:4 focused tests passed; real MJX smoke exercised nonzero air-time removal and unchanged moving reward/physics/termination (results/b07_airtime_smoke.log). Queue validation,paired argv equality and diff checks passed.


## B07 closed; A05 frozen phase-input structural diagnostic
B07 air-time gate survives16/16 and passes random numeric screen; nominal stand
linear.14940 passes but yaw.22611 fails. Control nominal linear/yaw.15394/.21222
fails and random7/8 survives (mean444.875,min59,height.35746). All full evaluations
skipped. Sampled stand/forward frames show residual stepping and upright ordinary
forward motion, not final visual approval. Close objective-mask hypothesis; no
further mask/coefficient sweep or unchanged extension. Retain failed control.
A05 uses ZERO training: fixed B07airtime_gate final1003520, deterministic stand,
nominal reset6000 and difficult random6002. Each gets unchanged input baseline,
phase clamp0 and phase clamp pi/2: six500-step trajectories,3-hour queue cap,
40-minute stage caps. No oracle rerun. Uses original physical resets/termination.
Clamp ONLY raw phase observation indices99:103 before saved normalizer/inference,
actor and privileged vectors consistently to cos/sin of antipodal phases. Actual
simulator phase,reward,history,physics and contacts remain unchanged. DefaultNone
is exact historical inference. This is an out-of-training-distribution diagnostic,
not a candidate deployment mode or legitimate full validation pass.
Record all six survival,height,linear/yaw errors,contacts and first100/last250
motion metrics. No choosing favorable phase or seed. Evidence supporting a future
standing-specific phase representation requires BOTH clamped angles on BOTH
resets to retain500-step survival/height>.6,lower yaw RMSE by at least20% versus
matched baseline, and linear RMSE no more than.02 worse. Even this only motivates
one prospectively frozen training/evaluation-consistent representation experiment.
If effects are angle-dependent or falls occur, do not promote an inference clamp;
record sensitivity/inconclusiveness and design the cheapest specific alternative
structural test. Failure cannot prove phase independence due distribution shift.
All original whole-episode gates,full command/control/gait evaluation and three
fresh independent training seeds/held-out tests remain unchanged and outstanding.
A05 verification: jitted observation-intervention test passed for both angles, exact no-op default and all nonphase indices; queue validation and diff checks passed.


## A05 rejected; A06 source-motivated double-stance diagnostic
A05 nominal baseline500 steps,linear/yaw.14134/.17784; random baseline500,
.19988/.19382. Nominal phase0 terminates foot-foot at102(height.27631),phase90
survives500 but yaw.35124. Random phase0/90 terminate foot-foot at64/55,
heights.52019/.42550,yaw.67297/.70400. All interventions fail frozen promotion.
Nominal baseline first100/last250 yaw.19579/.17497; random.23317/.17126.
Nominal phase90 first/last yaw.36951/.36226; failed trials' last250 windows
contain only102/64/55 available steps, not settled survival. No favorable clamp
selected; failure under distribution shift does not establish phase independence.
Specific new structural evidence: pinned joystick.py lines396-402 has a disabled
standing proposal setting BOTH phases to pi at zero command. A05 used antipodal
phases, always encoding one swinging leg; canonical double stance is distinct.
A06 freezes only two500-step zero-training trials, B07airtime_gate final1003520,
nominal6000/random6002,stand,deterministic,raw observation phase[-1,-1,0,0].
Reuse exact A05 matching baselines; no repeat baseline/oracle or phase sweep.
Actual simulator phase/history/reward/physics unchanged; intervention precedes
saved normalizer; observation indices99:103 only. Default paths unchanged.
Plan research/queues/post_f1_a06.json,90-minute queue cap,40-minute stage caps.
This tests the encoding, NOT full implementation of upstream's proposed state
update; it remains out-of-training-distribution and cannot count as a gate pass.
Promotion evidence for a future training-consistent structural experiment requires
BOTH cases500 steps,height>.6,yaw at least20% below matched baseline,linear no
more than.02 worse. Record all outcomes,first100/last250 and contacts. If either
fails,close inference-phase-intervention branch; do not try more phase constants.
Reassess a separately justified training architecture/objective design before
new training. All original full numeric/control/gait gates and three fresh
independent training seeds/held-out tests remain unchanged and outstanding.
A06 verification: both jitted diagnostic tests passed, including exact double-stance encoding/nonphase preservation and prior intervention/default parity. Queue validation, Python compile and diff checks passed.


## A06 rejected; S01 specialist architecture feasibility freeze
Both double-stance interventions terminate foot-foot: nominal98 steps,height
.26018,linear/yaw.75791/.47876; random57,height.36160,.92147/.88712. First100
and last250 summaries each contain entire shortened98/57 traces, not settled
windows. Matching A05 baselines survive500; neither promotion condition passes.
Close inference-phase branch; no more phase constants or inference-only changes.
Reassessment: B03-B07 corrections to the mature mixed-command walking policy
have not produced standing precision; phase intervention breaks its feedback
behavior. This does not prove an architectural limitation. Test prerequisite
capability for a possible separate standing branch before building a switch.
S01 is ONE from-scratch standing specialist feasibility probe, seed11,3010560
steps uninterrupted,512envs,lr3e-4,gamma.97,8evals,500-step horizon,full original
random resets,candidates1/scale1,zero command ranges. Original baseline rewards
with existing phase/air-time command masks true,stand pose-1,yaw damping0.
Cyclic phase observations remain normal: no new phase encoding or model change.
No warmstart,reference labels,optimizer restarts,checkpoint selection or retuning.
Contrast with old C07: it fine-tuned weak C02 with phase shaping active and
orientation-4; S01 learns standing from initialization with no zero-command gait
incentives. This is not causal isolation or a repeat mixed-command exposure sweep.
Budget3,010,560 training steps,2-hour train cap,6-hour queue cap; final checkpoint
fixed in advance. Guards ignore KL only through430080 and retain sustainedKL/
nonfinite checks. Fixed48 original screen rows include stand AND forward; own
checkpoint0 untrained reference. Full180 still conditional on BOTH original
screen passes; expected movement loss cannot be hidden or counted as success.
Record all stand rows,forward rows and videos. Specialist capability criterion:
all4 nominal stand rows survive500,height>.6,linear/yaw<=.15; all4 randomized
stand rows survive500,height>.6,linear/yaw<=.35. These are prerequisite criteria
ONLY, not substituted full gates. If met but movement fails, the only permitted
next direction is a prospectively frozen joint-controller/transition feasibility
study with original full evaluation; no deployment or automatic switch. If unmet,
close this specialist probe with no unchanged extension; use learning curve and
failure evidence to reassess before additional compute. No inevitable success
claim. Final recipe must independently train every component for three fresh
predetermined seeds and fresh held-out tests and pass all original numeric,
control-superiority,standing,gait and visual gates. Nothing is waived.
S01 verification: plan validation, fixed checkpoint/monitor/initialization assertions and output-video path checks passed; no runner/model logic changed.


## S01 prerequisite fails; A07 frozen temporal-control diagnostic
S01 all4 nominal stand rows survive500,height>=.75495,linear.02686-.02771,
single-support fraction0, but yaw.18969-.20972 fails. All4 randomized stand rows
terminate at34/24/44/41 (6000/6001/6002/6014),minheight.69081/.65808/.63081/.56666;
linear.492-.726,yaw.413-.585. Forward rows also fail; full jobs skipped. Sampled
nominal stand frames show planted feet,random forward frames show backwards fall.
No specialist promotion or controller switch. Close3.01M probe, no unchanged
extension. Internal evaluation mean lengths53,39,29,35,41,66,121,99 show incomplete
recovery,not convergence or proof standing cannot be learned. Final checkpoint
retained; do not pick earlier internal peak. S01 differs qualitatively from
walking-in-place B07, yet yaw precision remains unresolved.
Existing B07 A05 nominal last250 yaw spectrum(Hann,demeaned,50Hz) has54.8% power
above5Hz; B04 A04 has44.6%. Action per-channel delta RMS .20093/.15943. These
are correlations,not established causes. Need same trace for planted-foot S01.
A07 freezes TWO zero-training unmodified deterministic S01 final3010560 stand
trajectories: nominal6000/random6002,500steps cap,original resets and terminal
rules. Existing compare script saves actions,gyro,positions,contacts,reward and
terminal signals. No phase intervention,filter,oracle or completed training rerun.
Plan research/queues/post_f1_a07.json;90-minute queue cap,40-minute stage caps.
Review terminal causes,whole/first100/last250 errors; shortened traces are not
settled windows. Compare settled nominal yaw mean/variance,single-support,action
first-difference RMS,Hann spectral power above5Hz and dominant frequency against
saved B07/B04 traces (no reruns). High-frequency action/body motion while feet
remain planted motivates testing temporal actuator-target regularization with
matched training/evaluation; it does not justify an inference-only filter.
Prospective evidence rule: S01 nominal500steps,support fraction<.1,>50% demeaned
yaw spectral power above5Hz and action delta RMS>.1 supports ONE bounded matched
temporal-control experiment. If absent, do not add smoothing speculatively;
use measured failure cause to select a different specific diagnostic/design.
Randomized terminal evidence must inform recovery risks of any added latency.
Original full gates and three fresh independent seeds/held-out tests unchanged.


## A07 evidence met; T01 matched temporal-control freeze
A07 completed Sept24 at21:10UTC and awaited review; no training ran during that
review gap. Nominal S01 survives500,minimumheight.75495,support0,linear/yaw
.02746/.20577. First100 linear/yaw.02989/.19463;last250 .02717/.20933.
Settled Hann demeaned yaw power97.56% above5Hz,peak23.8Hz,action difference
RMS.49118 meets ALL frozen evidence conditions. Whole500 power97.66%,peak23.9Hz.
B07/B04 saved settled values54.79%/44.65%,action RMS.20093/.15943. This supports
a causal temporal-control experiment,not proof of actuator jitter causation.
Random S01 foot-foot termination43steps,height.64165,linear/yaw.47402/.59496;
short trace not a settled window. Added delay could worsen early recovery.
T01 compares alpha.5 EMA versus alpha1 unchanged, same B07airtime_gate final
1003520 parent (mixed-command recovery-capable policy, NOT failed S01 specialist),
seed11,1003520steps each,lr1e-4,gamma.97,phase/airtime gatingtrue,standing exposure
10%,standpose-1,standyaw0. Same moving command distribution/rewards in both arms.
Applied action = alpha*clip(raw,-1,1)+(1-alpha)*previous APPLIED action. Upstream
last_act already stores applied action and synchronized actor history exposes it;
reset zeros that history. No hidden filter state or observation-size changes.
Alpha1 bypass is exact old behavior. Saved config applies filter in both training
and all evaluation; no inference-only retrofit. Action remains bounded; alpha.5
adds approximately one control-step low-frequency delay (20ms),explicit recovery
risk. No termination/reset/contact/gate changes. Both actors warmstart exactly;
optimizer restarts in both arms; normalizer/actor restoration checked.
Budget2,007,040 training steps total,8-hour queue cap,original48-row screens and
conditional180 full evaluation each,B01checkpoint0 reference. Review BOTH arms,
stand and forward videos,all original numeric/control gates. Reject precision
improvement with recovery/moving regression. If neither clears both screens,
close temporal-control hypothesis; no alpha sweep or unchanged extension. Final
three fresh seeds/held-out full validation still mandatory; partial tests never
count as success. No assumption that this will resolve the remaining full grid.
T01 verification:6 focused adapter/reset tests passed; real MJX two-seed reset and multistep filtered-versus-manual target equivalence passed including exact physical states/done/history (results/t01_filter_smoke.log). Paired argv check,queue validation,diff check passed.


## T01 rejected overall despite nominal breakthrough; A08 freeze
Filtered T01 passes ALL nominal screen checks:8/8 survive500,stand linear/yaw
.14251/.09636,forward support.804,air.26/.28. Control nominal stand.15494/.24199
fails. Filtered random6/8 survive:reset6001 stand120steps and forward67; other
six survive500. Control random8/8 passes all numeric checks. Thus reject T01
as a full candidate; no alpha sweep or unchanged extension. Both full suites
correctly skipped. Sampled nominal stand/random ordinary forward videos remain
upright with stepping; not a complete visual certification or three-seed pass.
This is measured nominal precision improvement with recovery loss, not success.
A08 freezes FOUR zero-training traces: BOTH T01 fixed-final1003520 policies,
stand and forward.45,random reset6001,500steps cap,deterministic,original settings.
No disabling/changing filter,phase,reset or commands; no oracle/baseline reruns
outside this previously untraced matched failure.3-hour queue cap,40-minute
stage caps,plan research/queues/post_f1_a08.json. Preserve all four outcomes.
Inspect terminal causes and first divergence in pelvis height,foot clearance/
separation,contacts,local velocity,yaw and proposed actions. Reconstruct applied
EMA from clipped raw actions and initial zero history for filtered traces; raw
and applied action differences must not be conflated. Compare matched first50
steps and preterminal20 windows; short failures are not settled trajectories.
Frozen decision: if filtered failures share early foot collision after reduced
corrective motion while controls retain separation, evidence supports evaluating
smoothness learned without added actuator delay, not another filter coefficient.
If failure is different, choose a redesign tied to observed cause; no speculative
training launch. Correlation alone cannot establish added delay as the cause.
No inference-only rescue and no promoting a favorable command/reset. All original
numeric/control/gait/standing gates,three fresh training seeds and fresh held-out
validation remain unchanged. This diagnostic never counts as a full pass.
Automation persistence issue: tool reports update success but saved prompt remains
C22/30minutes. Until fixed, QUEUE_WORKFLOW active path is authoritative; do not
rerun or re-review C22. No duplicate schedule or queue was created.


## A08 reviewed; T02 learned action smoothness without delay
Both matched control traces survive500; filtered forward/stand terminate foot-foot
at63/83 with minheight.33918/.62424. Screening had67/120; diagnostic trajectory
lengths differ, so do not claim bitwise rollout parity or overwrite screen data.
Qualitative paired collision/recovery loss reproduces. Controllers have separately
trained weights, so the traces cannot isolate filter latency as sole cause.
First50 clipped raw/applied action-difference RMS:filtered forward.17450/.10533,
stand.19511/.11112;controls .25201/.25201 and.23597/.23597. Filter discrepancy
RMS.10810/.11399. Forward filtered foot-site horizontal distance drops to.05269m
preterminal,control first50 minimum.17330m;stand filtered last20 minimum.13630m
and collision sensor fires (site distance alone is not geometry collision).
Controls retain height>.71 through first50. These meet the qualitative frozen
criterion for testing learned smoothness without forced actuator delay; no alpha
sweep or inference filter rescue. Four outcomes retained; no final pass.
T02 is one matched continuation from SAME B07airtime_gate final1003520 as T01,
seed11,1003520steps each,lr1e-4,gamma.97;existing action_rate weight-.1 versus0,
alpha1 BOTH arms,all other rewards/commands/reset/phase settings same. Upstream
cost=sum((clipped_action-last_applied_action)^2),dt included once by upstream.
No new reward implementation or environment architecture. At observed control
RMS .236-.252 over29 joints the penalty is about.16-.18 beforedt, a bounded
starting magnitude relative to tracking rewards; no coefficient search authorized.
PPO can trade smoothness against recovery using immediate actions instead of
unconditionally delaying them. This remains a hypothesis,not guaranteed recovery.
Total2,007,040 new training steps,8-hour cap,exact restore audit,original48-row
screens and conditional180 full evaluations each;B01checkpoint0 untrained control.
Review both arms including videos and ALL original gates. Reject stand gains
with recovery/moving regressions. If neither clears both screens close this
learned-rate-cost hypothesis,no coefficient sweep or unchanged extension. If both
pass,full numeric/control/visual development evidence still required before one
frozen recipe and three fresh independent training seeds/fresh held-out tests.
A08 is not another general source audit; completed diagnostics must not repeat.


## T02 substantial development gain but full rejection; A09 freeze
Rate-cost arm passes BOTH original screens. Full nominal24/24 survive500 and
all numeric checks pass,mean linear/yaw.13060/.14076,stand worst.14027/.13149.
Full random23/24 survive500; sole failure backward-.25 reset6001 at109steps,
linear1.49275,yaw.90042. Full random mean linear.26254 also fails. Additional
forward resets6010-6021 all12/12 survive,mean linear.15733. All recorded full
suite failures are retained; no final development or three-seed pass.
Controls comparison: trained/untrained/standing mean linear nominal
.13060/1.13295/1.08649,random.26254/1.36374/1.35873,extra forward
.15733/1.27314/1.21280. Trained mean lengths500/483.708/500 exceed both
controls. These satisfy aggregate20% linear superiority but cannot override
survival/tracking failure. T02 control arm fails both screens,stand.15605/.18296
and one random failure198steps; its full stages skipped as frozen.
Sampled frames from all five rate-cost and both control clips show upright
ordinary standing/forward/combined stepping. Not continuous final visual approval;
no gait certification claimed. Both-arm results considered,not reward selection.
A09 freezes THREE zero-training backward-.25 traces: fixed T02 rate_cost final
1003520 at failing reset6001 and ordinary6000,plus matched T02control at6001.
Original random reset scale1/candidates1,deterministic500steps cap,no oracle,
filteralpha1 and saved rewards unchanged.2-hour queue cap,40-minute stage caps.
Inspect terminal decomposition,early height/velocity/foot geometry/action changes
and preterminal20; compare matching time windows to controls. Exact failures may
vary across diagnostic compilation; never overwrite original full-suite evidence.
Decision: if early foot collision/missed corrective response persists, identify
one specific correction tied to that mechanism; no action-rate coefficient sweep.
If control also fails, do not attribute all failure to rate cost. If diagnostic
survives, treat sensitivity as uncertainty and use a bounded deterministic
reproducibility check before altering recipe. No lucky checkpoint/reset selection,
no unconditional continuation. T02 earned continued investigation by passing both
screens,not eligibility for final independent validation until ALL development
numeric/control/visual gates pass. All three fresh independent training seeds and
fresh held-out criteria remain unchanged. Record a complete frozen recipe chain
before eventual fresh-seed study; inherited development checkpoints cannot count.


## A09 sensitivity; A10 exact-harness reproducibility freeze
A09 rate_cost backward6001 survives500,height.71889,linear.37605/yaw.29034,
so STILL fails original worst linear.35. Control6001 survives500 but
linear.53753/yaw.70411 fails. Rate_cost6000 survives500,.19374/.15081.
Original full-suite rate_cost6001 failed109steps. Do not overwrite original,
claim success or blame rate cost alone. Diagnostic scalar loop and evaluation
compiled rollout are distinct execution paths; survival discrepancy needs a
bounded reproducibility check before changing training. Not proof of an error.
A10 executes EXACT original T02 rate_cost full_randomized stage TWICE in fresh
processes/paths,unchanged saved checkpoint1003520 and full command/reset order,
B01checkpoint0 and standing controls,original500-step horizon,batching/default
settings and video command. Only output directory differs. No training or new
seeds; no choosing the best repeat. Original plus both repeats are retained.
Plan research/queues/post_f1_a10.json,3-hour queue cap,original stage timeouts.
Compare all per-row numeric values/survival and run metadata against original;
identify whether exact evaluation repeats reproduce109-step backward failure.
If repeats agree with original, treat scalar diagnostic as a separate numerical
path and target the reproducible original failure. If repeats vary, record
sensitivity and investigate execution consistency; do not treat any favorable
repeat as a development pass. Even two passing repeats cannot erase original
failure or allow premature three-seed promotion. Do not weaken gates or rerun
until lucky. Next training decision must account for this result. No need to
repeat completed preprocessing/source audits without new defect evidence.


## A10 stable failure; T03 command-envelope comparison freeze
Both exact-harness repeats reproduce original backward6001 at109steps with
identical height-.02933025,linear1.49274528,yaw.90041691. Trained row survival
and heights identical across all24 rows; max other trained linear/yaw variations
.0005738/.0022716. Some untrained/standing rows vary up to1step/.06yaw; preserve
all data,do not claim global bitwise determinism. Target failure is reproducible.
Scalar A09 results remain separate execution-path sensitivity,not proof of fixed
failure or grounds for accepting a lucky repeat. No more evaluation repeats.
T02 internal mean lengths500,477,500,471,443,500,457,471 do not show convergence;
no earlier checkpoint selected. Broad training commands span vx+-1,vy+-.5,yaw+-1
while original gate commands are moderate speeds. A single matched envelope test
can increase continuous low-speed training density without choosing failing seeds
or discrete evaluation commands. Causal mechanism is uncertain; no promised gain.
T03 compares uniform vx[-.6,.6],vy[-.35,.35],yaw[-.6,.6] versus unchanged broad
ranges,from SAME T02rate_cost final1003520,seed11,1003520steps each,lr1e-4,
gamma.97,action-rate-.1,filteralpha1,phase/air-time gatingtrue. All other settings
identical,including10% zero commands and original random resets. Moderate box
contains all original default and development command vectors; evaluation grid,
held commands and thresholds unchanged. This is a training-distribution change,
not restricting which commands are tested. No reset-seed oversampling.
Total2,007,040 new steps,8-hour queue cap,exact restore audits;original48-row
screens/conditional180 full rows per arm,B01checkpoint0 control. Both arms and
all failure evidence reviewed. If neither fully improves remaining recovery while
retaining standing/movement,close envelope hypothesis;no range sweep or automatic
unchanged extension. Existing screen/full skip logic unchanged. Before final
three-seed study require ALL full development numeric/control/visual gates,then
freeze complete recipe chain and fresh predetermined training seeds/held-out tests.
T03 plan-only change: paired argv equivalence except command ranges and output
paths asserted; validate plan and diff before launch. No new runner/model logic.

## T03 rejection; A11 exact-rollout trace capture freeze
Both restore audits exact (17 actor/normalizer leaves). Moderate envelope passes
nominal but randomized stand6001 fails98steps; random7/8 survive. Reject narrower
range,close envelope hypothesis; no range sweep or unchanged extension.
Control passes screens and full nominal24/24; extra forward12/12 survive.
Full random23/24: previous backward6001 now survives500 but linear.386746>.35;
right6001 terminates106steps with undesired contact,minimum height.430893,
linear.97467/yaw1.14383. Mean random linear.24350/yaw.23093 pass but cannot
override worst-row/survival failure. Not a full pass; no fresh-seed study yet.
Sampled frames from all seven trained videos show ordinary upright stepping and
standing; results/post_f1_t03/review.jpg. Sampling is not final continuous gait
certification. Failure episodes are not the representative saved videos.
A11 freezes TWO zero-training full-randomized evaluations: fixed T02 rate_cost
final1003520 and T03control final1003520,original command grid/order,seeds6000-6002,
controls,500steps and video settings unchanged. Sole functional addition writes
ALREADY computed host numeric arrays to NPZ after device synchronization. It does
not change compiled rollout outputs/physics/inputs. AST equality checked for
_build_rollout,_summarize_trace,_numeric_episode_traces against previous revision.
Budget zero training,2 full evaluations,3-hour queue cap. Fresh paths post_f1_a11.
This is acquisition of previously discarded temporal evidence,not another search
for favorable repeat results. Original failure records remain authoritative; if
survival differs,record sensitivity and do not promote a lucky result.
Review valid-only whole/first50/preterminal20 velocity and yaw errors,height,
torso tilt,contact asymmetry/slip,action-rate and saturation around failures;
compare T02 vs T03 on backward/right6001 and ordinary6000,using matched windows.
Classify collision preceding loss of balance versus loss of height/tilt preceding
collision; aggregate undesired-contact flags cannot establish exact foot pair.
Surviving backward traces must distinguish early recovery error from persistent
late tracking error. No full-pass claims from diagnostics and no automatic
training extension. Select one bounded mechanistic training/representation test
only after this evidence; do not resume reward coefficient/phase/range sweeps.
Original complete independent three-seed,held-out,numeric/control/visual gates
remain. No inherited development checkpoint counts as a fresh independent seed.
