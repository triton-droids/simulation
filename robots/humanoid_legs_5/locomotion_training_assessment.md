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
