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
