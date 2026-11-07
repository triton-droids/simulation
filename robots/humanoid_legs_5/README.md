# humanoid_legs_5

Portable lower-body MJCF bundle with the two moving heel bars and closed-loop
pin constraints. The 24 referenced meshes are included beside the XML. Open
`humanoid_legs_5_full_sweep.mp4` for the entire lower body and
`humanoid_legs_5_ankle_sweep.mp4` for the pin motion.

## What has been checked

- Both links come from the Fusion export dated 2026-10-20. Each heel bar has
  166.2476 mm hole spacing; the inferred crank radius is 45 mm.
- The model has 10 named actuators, four additional passive joints, and a
  15.999 kg total mass after removing the artificial 1 g torso placeholder.
  The measured hip mass and inertia are now on the free-floating root body.
- Both linkages close continuously at 241 positions per side over the old
  simulator's ±0.6 rad ankle range. The maximum analytic pin gap is below
  1e-8 m. The 1,000-step zero-gravity drive check stays finite and its maximum
  pin gap is below 0.1 mm.
- Bar masses are separated from the old lumped foot bodies, and 1 g crank
  proxy masses are separated from the old shin bodies. The artificial 1 g
  torso placeholder was then removed, leaving 15.999 kg. The validation report
  records the mass adjustments and method.

Run the portable check from any directory:

```bash
python -m pip install mujoco numpy
python robots/humanoid_legs_5/validate_mjcf.py
```

## Validation still required before training

This is a **kinematic inspection asset**, not a calibrated locomotion model.
The motor axis was inferred from a cylindrical legacy mesh. The left Fusion
upper pin differed from its bar bore by 2.649 mm; the neutral linkage was
analytically closed without changing bar length. Hardware motor axes, travel,
zero offsets, and transmission mapping have not been measured.

The old foot inertia tensor cannot be decomposed into a physically valid foot
plus bar tensor. The remaining foot tensors are positive estimates made by
scaling the old ones. The crank is a 1 g capsule proxy, not identified rotor
inertia. The model still drives the foot hinges directly. Existing policies
cannot be loaded unchanged because the four passive joints change the state
layout.

The packaged USD was regenerated from this MJCF and imported in Antioch Isaac
Lab 3.0 with PhysX. A fixed-root, zero-gravity sweep moved all 10 actuated
joints in their requested directions by 0.035 rad; maximum hold tracking error
was 0.0062 rad, and the sampled ankle pin pairs stayed coincident. This
validates the imported joint drives and loop setup only; it does not test
ground contact or balance.

The free-root grounded zero pose is **not stable** with the original MJCF-authored
actuator gains: it tips past 90 degrees by 3 seconds. The knees start fully
straight at their limit, and a standing/crouched reference pose plus actuator
behavior still need validation. See `isaaclab_physx_check.json` for the test
conditions and results. The 10-vs-14 actuator notice is expected because the
four linkage joints are passive. Keep the first locomotion work on PhysX.

The motor axis, zero offsets, travel, transmission mapping, and some inertias
remain estimates. Direct ankle-hinge actuation and the policy input/output
contract still need calibration before hardware use. See
`../../notebooks/GETTING_STARTED.ipynb`.

## First locomotion task

`../../src/train_locomotion.py` builds a flat-ground Isaac Lab task from the
packaged USD. The actor sees 39 values: base angular velocity, projected
gravity, body velocity command, 10 motor-joint positions and velocities, and
the previous 10 actions. It outputs targets for the 10 motor joints at 50 Hz;
the four linkage joints remain passive. The knee reset is neutral (0 rad) following the standing check.
The initial command curriculum assumes forward is body Y, inferred from the
mesh and joint-axis layout; confirm the robot convention against CAD/hardware.
It commands 0.2–0.5 m/s along that axis, with 10% standing commands.

The current simulation baseline instead uses ankle gains 100/5 and a neutral
reset pose. A ten-second single-robot check stayed upright, and the task's
five-second eight-environment check had zero resets. See
`standing_validation.json` for the controlled comparison; these results do
not validate full sole contact or hardware gains.

The task is an experiment, not a successful walking setup. A 500-iteration
continuation on 256 environments reached about 504 of 600 policy steps per
episode on average in the final 100 updates, but command success averaged only
8.9%. A separate deterministic test requested 0.35 m/s along body Y and
measured about 0.004 m/s, so the saved actor mostly stood still. That experiment used contact
sensing that selected only the hip root. The current task explicitly senses
both feet; it still uses gravity-vector tilt termination and no foot-contact
reward. See the [full
assessment](locomotion_training_assessment.md).

[`locomotion_preview.mp4`](locomotion_preview.mp4) shows the selected PPO
candidate under a forward command. It is not a successful gait. The policy has
not been validated for hardware.

The reviewed candidate weights are in `checkpoints/`. Training and
fixed-command evaluation commands are in the repository
[README](../../README.md#locomotion-experiment). Full TensorBoard logs and the
other periodic checkpoints remain in the trainer's local ignored `outputs/`
directory.

The reproducible source builder and original Fusion export are on the
`codex/hardware-retrain-local` development branch under `cad/`. This portable
bundle is intended for a smaller deployment branch.
