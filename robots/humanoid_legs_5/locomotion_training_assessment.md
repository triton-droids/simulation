# First locomotion training assessment

This report describes a flat-ground Isaac Lab PPO experiment on the packaged
`humanoid_legs_5` USD. It ran 256 environments and a 500-iteration fine-tune
of a stability checkpoint. The policy controls 10 motor joints at 50 Hz; the
four linkage joints remain passive.

## Result

This experiment did **not** produce a walking policy.

- During the last 100 training iterations, average episode length was about
  504 of 600 policy steps (10.1 of 12 seconds).
- Command success averaged 8.9% in that window and varied substantially.
- A deterministic five-second evaluation requested 0.35 m/s along body Y and
  measured 0.0042 m/s mean body-Y velocity, with 0.3458 m/s horizontal
  velocity error.
- Mean absolute raw actor output was 3.58 before normalized action clipping.
  This suggests the policy is saturating its actions.

The preview video [`locomotion_preview.mp4`](locomotion_preview.mp4) shows this
candidate under that command. It is an inspection video, not evidence of
walking. Do not deploy the checkpoint on hardware.

## Next fixes

1. Resolve actor action saturation through the action distribution and
   regularization.
2. Make foot contact sensing resolve the nested USD link paths; the current
   stock contact sensor selected only the hip root.
3. Verify the free-root standing pose and both sole collision/contact shapes
   under gravity in PhysX.
4. Measure motor axes, usable travel, ankle transmission, and joint response;
   replace estimated drive and inertia values.
5. Re-evaluate standing, commanded forward motion, turns, and stops before
   describing a checkpoint as walking.

The reviewed checkpoint is committed as
`checkpoints/humanoid_legs_5_ppo_candidate_1200.pt`. The complete TensorBoard
event data and the other periodic checkpoints remain in the local ignored
directory `outputs/locomotion_upright_v1/`.

## Follow-up fine-tune, 2026-11-09

Resumed from the local `outputs/locomotion_gait_v2/humanoid_legs_5_latest.pt`
checkpoint for 500 additional PPO iterations on Antioch. The training reset
pose was widened slightly by setting the hip abduction joints to -0.05 and
+0.05 rad; the packaged CAD geometry was not changed. Periodic checkpoints,
evaluation files, and the full TensorBoard log are in the ignored local
directory `outputs/locomotion_gait_v2_wide_finetune_cont/`.

The run improved survival but did not produce useful walking. Its best tested
checkpoint was `model_1025.pt`: a deterministic 10-second evaluation with a
0.35 m/s body-Y command averaged 0.0066 m/s body-Y velocity, 0.3434 m/s
horizontal velocity error, 0.885 upright cosine, and 3.69 mean absolute raw
actor output before action clipping. The final checkpoint performed similarly.
The separate fixed-command result does not support treating this as a walking
policy, despite intermittent training success metrics. The video
[`locomotion_preview_wide_finetune.mp4`](locomotion_preview_wide_finetune.mp4)
shows the selected checkpoint under that command.

## Current walking candidate, 2026-11-09

Further PPO-only attempts on Antioch either kept the stable standing policy,
learned unstable lurching, or returned to standing during fine-tuning. A useful
step emerged by combining the standing-capable checkpoint with a slow gait
reference on the two hip swing and knee joints. The balance policy still
controls the other joints.

The current hybrid preview uses `humanoid_legs_5_balance_stage21.pt`, a 0.35 Hz
alternating hip/knee reference (0.10/0.08 rad), and a 0.515 blend between the
policy and reference actions. In a 10-second, one-environment Antioch preview
under a 0.12 m/s body-Y command, it averaged 0.129 m/s COM velocity during the
final half and had one fall/reset event. A 32-environment, six-second sweep of
the same setting averaged 0.081 m/s COM velocity, 0.974 upright cosine, and
five reset events total. The robot takes alternating steps and translates in
the saved video [`locomotion_hybrid_gait_012_stable.mp4`](locomotion_hybrid_gait_012_stable.mp4).

A distilled neural actor is also included at
`checkpoints/humanoid_legs_5_gait_imitation_012.pt`. In a fixed 0.12 m/s test
across 64 environments, it averaged 0.105 m/s COM velocity; the median
environment reached 0.137 m/s and 78% exceeded 0.05 m/s. Mean upright cosine
was 0.936, but environments averaged 0.86 fall/reset events over ten seconds.
It is therefore a gait candidate, not a robust walking policy. PPO refinement
currently erases the gait and returns to standing.

These results establish stepping and translation in Isaac Lab/PhysX, but not
reliable long-duration locomotion. Keep the task in simulation. The next work
is to reduce fall resets, test multiple command speeds and stopping, and
continue PPO from the distilled gait without losing the step pattern. Hardware
calibration and motor/linkage validation remain separate requirements.

The gait checkpoints, videos, training code, and evaluation files are included
in this branch.

## Follow-up checkpoint tuning, 2026-11-10

The most recent tuning runs changed the PPO reward and reset settings; they did
not change the USD, MJCF, meshes, or linkage constraints. The long gait
imitation checkpoint remained the only recent policy that translated
consistently, but it was fragile: in a 64-environment, ten-second test at a
0.12 m/s body-Y command, it averaged 0.106 m/s body-Y velocity, 0.945 upright
cosine, and 0.48 m forward displacement. Only 23 of 64 trials finished without
a fall/reset; average valid time was 7.73 seconds.

PPO fine-tuning exposed two reward failures. With weak progress reward, the
policy returned to standing: the latest ten-second preview had no resets but
moved only 0.089 m forward and averaged 0.003 m/s body-Y velocity in its final
half. Raising progress pressure produced faster lurches rather than a stable
gait: the tested checkpoint averaged 0.265 m/s body-Y velocity but all 64
trials fell within 3.33 seconds. A stronger upright/fall penalty restored
survival but also returned the actor to standing.

Action-scale checks showed the same tradeoff. At scales 0.16–0.20, all 32
trials survived but forward velocity was about 0.005–0.006 m/s. At 0.22, mean
body-Y velocity reached 0.119 m/s, but only 9 of 32 trials survived ten
seconds. At 0.24, only 3 of 32 survived. The 36-setting balance/gait overlay
sweep also found no setting with both useful forward speed and low falls.

The latest preview is
[`locomotion_latest_episode_teacher_stability_v3.mp4`](locomotion_latest_episode_teacher_stability_v3.mp4),
with trajectory and foot-contact telemetry beside it. It shows the latest
PPO checkpoint standing rather than walking. The measurement bundle is
[`locomotion_tuning_2026-11-10.json`](locomotion_tuning_2026-11-10.json).

The drop from stepping to standing came from PPO optimizing a fixed, low-speed
command whose tracking reward was already high while standing; stronger
progress shaping then destabilized the gait. It was not caused by repairing
the ankle linkage. Keep the long gait-imitation checkpoint as the walking
baseline, and tune rewards/actuation against fixed-command displacement,
survival, contact, and foot-swing metrics before extending training.
