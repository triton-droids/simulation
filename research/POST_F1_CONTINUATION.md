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
