# Continued research after F1 failure

The user explicitly authorizes continued bounded research until a recipe passes
three fresh independent training seeds. This supersedes the F1 stop-after-failure
instruction; it does not permit selecting three lucky seeds or weakening gates.
F1 remains a failed, incomplete validation study. Seeds 11/22 and command grid
6000-6002 are now development evidence, not fresh final validation data.

D01 freezes a zero-training comparison: evaluate seed11 linear checkpoint
1003520 on the same randomized grid as the failed recovery checkpoint.
Budget: 72 episodes of 500 steps maximum including two controls; one gait
video per controller; three-hour queue cap. Exact commands and unchanged
numeric gates: queues/post_f1_d01.json. Outputs are fresh and evidence preserved.
If the parent materially outperforms recovery, investigate the recovery stage;
otherwise investigate the earlier learned behavior. Do not blindly add training.

At each completed batch, use numeric gates and video to choose the cheapest
discriminating next experiment, freeze hypothesis/budget/commands/gates, test
changed logic, commit and launch locally. Resolve the seed22 Orbax rename
PermissionError before further training; preserve interrupted stage artifacts.
Do not claim parameter warm-start restores optimizer/PRNG training state.
Once a development recipe passes all required tests, freeze a new validation
study with three predetermined fresh training seeds and fresh held-out tests.
All three must pass; failures return to development, never disappear.

Hourly model wakes read only compact status while healthy. No active waiting,
no full logs except relevant errors, no hardware or remote pushes. Notify only
meaningful findings, errors requiring attention, and verified final success.

## D01 outcome and C01 freeze
D01: pre-recovery randomized survival 3/24, mean steps118.67, linear RMSE1.2685.
Final F1 recovery: 8/24, mean steps219.21, linear RMSE0.9311. Recovery improves
this weak parent; the failure predates recovery. Sampled combined-video frames
are upright with changing leg poses in both; this successful episode does not
represent the failing majority and does not establish a gait pass.
C01 tests one additional 1,003,520-step parameter warm-start from seed11 final
recovery, same seed11, lr1e-4, weights, randomized resets and full commands.
Optimizer/step/PRNG restart is explicit, not an exact training continuation.
Both nominal and randomized development suites retain all gates and controls.
Budget one training stage and 144 assessment episodes; five-hour queue cap.
No automatic extension: assess survival/tracking changes before further work.

Checkpoint repair: complete Orbax directory transaction in Linux /tmp, write
policy there, then copy the complete tree into a fresh output checkpoint path.
This avoids the failing Windows-mounted directory rename. Never overwrite
existing evidence; incomplete copies fail the stage and remain for diagnosis.

## C01 outcome and D02 freeze
C01 nominal passes24/24, mean linear/yaw RMSE .06890/.09415. Randomized
survival7/24, mean steps203, linear/yaw RMSE .86408/.41165; F1 recovery
had8/24 and219.21 mean steps. Additional training does not improve survival.
Reviewed randomized combined6000 video frames show backward tipping/fall.
All eight commands fail at reset6002; forward fails32 steps. Do not extend
this unchanged training again without new evidence.
D02 compares C01 and C22lr1e4 seed7 on forward .45, randomized reset6002,
using the evaluator reset key and sampled gait frequency. Existing trace
script records initial state, policy/oracle actions, rewards and contacts.
Budget four trajectories total (two policies and matched oracle runs),500
steps each, no training,90-minute queue cap. Exact commands in post_f1_d02.json.
Use matched traces to localize failure before changing reward/reset curriculum.

## D02 outcome and C02 freeze
At identical randomized forward reset6002, C01 terminates32 steps, C22 at69,
while the reference survives500. C01 has no completed left-foot air interval;
C22 has short/shuffling intervals. Both drift opposite commanded forward
motion. Orientation penalties are -.525/-.575 vs reference -.032; actions
are not saturated. This is a shared learned recovery weakness, not an
impossible reset. Correlation does not prove orientation reward is causal.
C02 tests only orientation scale -4 vs existing -2 from the C01 checkpoint,
seed11, lr1e-4,1,003,520 steps with unchanged randomized resets and all other
weights. Exact warm-start audit and nominal/randomized tests run automatically.
Budget144 assessment episodes,5-hour queue cap, fresh destinations. Accept
only joint nominal/randomized gate and visual pass; otherwise inspect survival
and posture changes before choosing another bounded experiment. No final
three-seed claim uses these development seeds or commands.

## C02 outcome and C03 freeze
C02 nominal24/24 passes; randomized13/24 survives, mean steps304.875,
linear/yaw RMSE .64810/.27178. All reset6002 commands still fail.
Sampled combined6000 video shows upright alternating leg poses throughout,
previous C01 same episode fell; no overall robustness pass.
C03 adds matched -2 control and -8 orientation candidates to existing -4 C02.
Both start exactly C01 final checkpoint, seed11,1003520 steps each,lr1e-4,
identical other settings. This distinguishes extra training from reward change.
Total2,007,040 training steps and288 assessment episodes;10-hour queue cap.
Joint existing nominal/randomized numeric and visual criteria required; never
choose by reward alone. No training-budget extension or changed final gates.

## C03 outcome and D03 freeze
Matched orientation penalties -2/-4/-8 survive10/13/9 of24 randomized episodes;
all nominal gates pass. -4 remains a development candidate, not a final pass.
Sampled combined6000 videos show both C03 policies upright with alternating
leg poses; they do not resolve other failed episodes. No stronger-weight search.
D03 uses C02 checkpoint on forward .45, reset6002,500 steps: full reset,
zero initial base velocity, or nominal joint scales, changing only one component.
Record effective reset config and initial state for attribution. Compare shared
initial qpos/qvel components before interpreting results. Existing oracle runs
provide matched references. Six trajectories total, zero training,2-hour cap.
These altered-reset results are diagnostic only and cannot satisfy any final gate.

D03 invalid ablation: all three initial states and rollouts were identical.
Playground adapter only reads reset.randomize; native reset scale fields are
ignored. Preserve evidence, draw no component-sensitivity conclusion.
D03-fixed changes actual sampled qpos/qvel only in the diagnostic script,
recomputes MJX data/contacts/observations, and asserts intended component changed
and unrelated qpos/qvel components did not. Full-reset baseline is unchanged.
Repeat six trajectories in fresh paths; no training or final evaluator changes.

D03-fixed full baseline completed; no_velocity stopped on exact qpos equality
against a separately compiled reset. Recovery samples the original once and
passes that state into the ablation, avoiding separately fused random draws.
Unmodified components allow only 1e-6 absolute numerical difference (zero
relative tolerance); removed components still require actual change and the
specified zero velocity/nominal joints. This is diagnostic numerical matching,
not a relaxation of final gait gates. One-step integration checks cover both
ablations before repeating only unfinished traces in fresh recovery paths.

## Corrected D03 result and C04 freeze
C02 full reset6002 forward:65 steps. Zero initial velocity:500 steps,
linear/yaw RMSE .06962/.07197. Nominal joints with original velocity:80 steps.
Assertions verified removed component changed and other initial coordinates
matched within1e-6. This localizes sensitivity to initial velocity for this
one reset; it is not broad success. Trace evidence is numeric, not a gait pass.
C04 warm-starts C02,seed11,lr1e-4,orientation-4,1003520 steps. Enable existing
Playground pushes with interval1-3s,magnitude .1-.5; otherwise unchanged.
Hypothesis: repeated small velocity disturbances teach recovery more frequently
than episode-start perturbations alone. Full randomized resets remain in training.
Evaluations explicitly disable pushes as before and retain all original gates,
commands, initial velocities, references and videos. Budget144 test episodes,
five-hour cap. Compare to C02; no blind extension if robustness fails to improve.

## C04 outcome and C05 freeze
C04 nominal gates pass24/24; randomized11/24 survives, mean steps272.58,
linear/yaw RMSE .64508/.29431. C02 parent survives13/24; no robustness gain.
Sampled combined6000 video remains upright with changing support legs, but
other episodes fail. Do not extend push training on this evidence.
C05 starts the C02 checkpoint (not C04),seed11,lr1e-4,orientation-4,
1003520 steps, unchanged full randomized resets and pushes disabled.
Only training episode horizon changes500 to100 to increase startup recovery
experience. This may sacrifice sustained gait, so final evaluation stays500
steps with identical original gates, controls and videos. Budget144 assessment
episodes,5-hour cap. Accept only full original tests; training reward and
100-step training survival cannot establish success. No blind extension.

## C05 outcome and D04 freeze
C05 nominal gates pass24/24; randomized12/24 survives, mean steps290.08,
linear/yaw RMSE .71153/.28472. C02 remains13/24; do not extend short-horizon
training. Sampled combined6000 video is upright with alternating leg poses;
failed episodes still reject robustness. D04 uses C02 reset6002 forward.45,
zero linear velocities only vs zero angular velocities only. Each starts from
one shared sampled state with assertions on modified/unmodified coordinates.
Four trajectories including oracle controls,500 steps maximum each,zero
training,90-minute cap. Diagnostic only; no final evaluation gate changes.

## D04 result and D05 freeze
Removing linear velocity gives500 steps,linear/yaw RMSE .06910/.07504;
removing angular velocity still fails65,linear RMSE1.5241. Initial-state
assertions passed. This isolates linear velocity sensitivity at reset6002,
not general robustness. D05 removes horizontal velocity vs vertical velocity
separately on C02,forward.45,reset6002. Four trajectories including oracle,
500 steps maximum each,zero training,90-minute cap. Verify preserved initial
coordinates; no altered-reset diagnostic can count as final validation.

## D05 outcome and C06 freeze
Zero horizontal velocity:500 steps,linear/yaw RMSE .06817/.07327. Zero
vertical velocity:64 steps,linear RMSE1.62164. Horizontal reset speed .602m/s
exceeds C04 push cap .5m/s. These single-reset traces guide development only.
C06 repeats C04 from the SAME C02 parent,seed11,1003520 steps,lr1e-4,
orientation-4,500-step training horizon,push interval1-3s. Only magnitude
changes .1-.5 to .5-1.0. This is a bounded threshold hypothesis; larger pushes
may impair nominal gait. Evaluate original nominal/randomized500-step suites
with pushes disabled as before,144 episodes,5-hour cap. Require all original
gates and video; no automatic unchanged extension. Compare with C04 and C02.

## C06 outcome and C07 freeze
C06 nominal passes24/24; randomized12/24 survives,mean steps288.08,
linear/yaw RMSE .67099/.29125. Reviewed combined6000 frames show backward
fall. No stronger-push or unchanged extension is warranted.
C07 starts C02 checkpoint,seed11,lr1e-4,orientation-4,1003520 steps,
500-step horizon,full randomized resets,pushes off. Only training command
ranges become zero in all three axes: practice braking/stabilizing with
existing gait-phase shaping retained. Risk: forgetting commanded locomotion.
Evaluate all original eight commands,500 steps,nominal and full randomized
resets with true untrained and standing controls,144 episodes,5-hour cap.
Do not treat stationary recovery alone as success; require original gait and
tracking gates. Review recovery gains and retention before any next stage.

## C07 outcome and D06 freeze
C07 nominal22/24,randomized8/24; mean randomized steps211.29 and linear/yaw
RMSE .83056/.35968. Stand randomized6001/6002 fail48/62 steps. Reject this
curriculum; representative combined6000 frames remain upright but are not a pass.
D06 audits C02 actor observation mean/std and standardized input deviations
for nominal/randomized resets6000-6002 at forward.45. Six resets,zero training,
zero rollouts,one-hour cap. Report count leaves and largest standardized channels.
This is a read-only scaling diagnostic, not proof of causal normalization error.
Do not adjust scales or reset normalizer until evidence supports a specific issue.

## D06 result and C08 freeze
D06 max initial standardized actor deviation:nominal1.408,randomized1.997,
1.608,2.364 at seeds6000-6002; no channels above10. Normalizer count12400640.
This does not support a gross initial observation scaling fault. Do not reset
normalizer based on this audit; later trajectory scaling remains unaudited.
C08 starts C02,seed11,lr1e-4,orientation-4,1003520 steps,full commands and
randomized resets,500-step horizon,pushes disabled. Remove both clock-phase
shaping terms (feet_phase3->0,feet_contact_phase2->0) as one functional ablation.
Hypothesis: permit corrective foot placement unconstrained by gait phase.
Risk: shuffling/static or asymmetric gait. Retain original support/airtime,
tracking,height,survival and visual gates; no reward-based success claim.
Budget144 assessment episodes,5-hour cap. No automatic unchanged extension.

## C08 outcome and C09 freeze
C08 nominal passes24/24; randomized11/24,mean steps268.29,linear/yaw
RMSE .79719/.31791. Sampled combined6000 remains upright with alternating
leg poses but failures elsewhere reject robustness. Restore phase shaping.
C09 compares lr1e-4 control versus3e-4 from the identical C02 checkpoint,
seed11,1003520 steps EACH,orientation-4,feet_phase3,contact_phase2,full
commands/randomized resets,500-step horizon,pushes off. The unchanged branch
is a matched control, not an indefinite extension. Earlier C21 seed7 regression
at3e-4 cautions against assuming a gain; current parent and seed differ.
Total2,007,040 training steps,288 assessment episodes,10-hour cap. Keep finite
and sustained-KL safeguards. Judge original joint gates and videos, not reward.
If no clear improvement, do not extend this local sweep; reassess the training
recipe rather than repeatedly perturb this checkpoint.

## C09 outcome and R01 foundation freeze
C09 control1e-4 survives14/24 randomized,mean320.375; faster3e-4 survives8/24,
mean207.958. Both nominal gates pass; sampled combined6000 frames upright.
One additional survivor is insufficient to justify more local checkpoint sweeps.
R01 returns to the foundation: train FROM SCRATCH seed11,3,368,960 steps,
original F1 balance recipe (linear1,yaw.75,feet_phase1,contact_phase0,forward.5).
Only reset.randomize changes false->true; no development checkpoint warm start.
This remains development seed11, not an independent final validation seed.
Evaluate stand/forward.5,reset6000-6002,500 steps,nominal and randomized,
with true initialization and standing controls:36 episodes plus forward videos.
Five-hour cap. This is a foundation diagnostic, NOT a relaxed final walking gate.
Require full survival/upright finite behavior to advance foundation; inspect
tracking and contact traces for next gait stage. Preserve original full-command,
independent-three-seed final criteria. Compare to original F1 balance evidence.

## R01 outcome and C10 freeze
R01 randomized0/6 survives; nominal forward3/3 survives but high linear errors
.459-.470 and static-looking sampled poses. Nominal stand0/3. Randomized
forward video shows backward fall. Foundation gate fails; do not proceed to
gait stages or count this as evidence of robust walking.
C10 returns to C02 checkpoint with one temporal-credit change: discounting
.97->.995. At65 steps weights are .138 vs .722; this motivates, but does not
prove, improved anticipatory recovery. Same seed11,lr1e-4,orientation-4,
phase3/contact2,full commands/resets,500-step horizon,pushes off,1003520 steps.
C09 control is matched parent/seed/budget at gamma.97. No seed selection.
Evaluate original full nominal/randomized grids and controls,144 episodes,
five-hour cap; preserve KL safeguards. Compare all gates and videos, not reward.

## C10 outcome and C11 bounded continuation
C10 nominal24/24 passes; randomized16/24 survives,mean352.46,linear/yaw
RMSE .49246/.25864. All eight failures share reset6002 (31-71 steps); do not
omit it. Sampled combined6000 frames upright with alternating support poses.
C09 matched control14/24,mean320.38. This gain supports ONE bounded additional
1003520-step warm-start from C10,seed11,gamma.995,lr1e-4,all other settings
unchanged. Optimizer/PRNG restart remains explicit, not exact training resume.
Budget144 original test episodes,5-hour cap,all original gates/videos intact.
If improvement stalls or regresses, do not extend unchanged automatically.
No independent final seed success is claimed by this development result.

## C11 outcome and C12 freeze
C11 nominal passes24/24,randomized15/24,mean330.67,linear/yaw RMSE
.40830/.27025. Survival regressed vs C10 16/24; no unchanged extension.
Sampled combined6000 frames upright with alternating support,not overall pass.
C12 compares unroll20 control vs80 from SAME C10 checkpoint,seed11,gamma.995,
lr1e-4,all other settings unchanged. Equal1,146,880 steps each fits both PPO
step quanta (batch32*minibatches16*unroll,7 evaluation epochs). Final checkpoint
1146880; parent remains1003520. Ignore KL through first positive163840-step
callback; existing sustained .2 threshold stays. Longer unroll may improve
credit across~65-step failures, though GAE/critic quality still matter.
Total2,293,760 steps,288 original evaluation episodes,10-hour cap. Preserve
all gates; record possible memory/runtime changes rather than silently reducing
batch size. No final seed or robustness claims from development comparisons.

## C12 outcome and C13 freeze
C12 unroll20 control16/24,mean351.67,linearRMSE .44025; unroll80 14/24,
mean323.33,linearRMSE .54822. Both nominal pass; sampled combined6000
frames upright. All reset6002 cases fail. Reject longer-unroll extension.
C13 returns to C10 checkpoint,seed11,1003520 steps,unroll20,gamma.995,
lr1e-4,all other settings unchanged. Only termination scale changes-100 to-500.
Upstream step directly sums reward terms*dt without positive clipping, so
this changes fall cost (dt.02: -2 to-10). Hypothesis: stronger survival pressure
outweighs immediate tracking gains during recovery. Risk: static/cautious gait.
Compare to C11 matched parent/budget; retain full original tracking/gait and
survival gates,144 evaluations,5-hour cap. No reward-only success or blind extension.

## C13 outcome and D07 freeze
C13 nominal passes24/24; randomized16/24,mean350.04,linear/yaw RMSE
.38565/.30137. Same survival as C10, worse yaw average. Sampled combined6000
frames upright; no robust pass. Do not increase terminal penalty again blindly.
D07 audits the frozen C10 development policy,forward.45,randomized reset
seeds6010-6021 (12),500 steps,with untrained/standing references:36 episodes.
These are additional development resets, not fresh final validation. No training,
2.5-hour cap. Retain every result and forward6010 videos. Purpose: determine
breadth of failures before more narrow tuning on reset6002. No seed selection,
no changes to final full-grid or independent-training-seed requirements.
