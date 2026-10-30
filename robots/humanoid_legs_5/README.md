# humanoid_legs_5

Portable lower-body MJCF bundle with the two moving heel bars and closed-loop
pin constraints. The 24 referenced meshes are included beside the XML. Open
`humanoid_legs_5_full_sweep.mp4` for the entire lower body and
`humanoid_legs_5_ankle_sweep.mp4` for the pin motion.

## What has been checked

- Both links come from the Fusion export dated 2026-10-20. Each heel bar has
  166.2476 mm hole spacing; the inferred crank radius is 45 mm.
- The full model compiles in MuJoCo 3.8.1. It has 10 named actuators, four
  additional passive joints, and a 16 kg total mass.
- Both linkages close continuously at 241 positions per side over the old
  simulator's ±0.6 rad ankle range. The maximum analytic pin gap is below
  1e-8 m. The 1,000-step zero-gravity drive check stays finite and its maximum
  pin gap is below 0.1 mm.
- Bar masses are removed from the old lumped foot bodies, and 1 g crank proxy
  masses are removed from the old shin bodies. The original total mass is
  preserved. `humanoid_legs_5_validation.json` records the exact method and
  numbers.

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

The MJCF-to-USD import, PhysX closed-loop behavior, grounded motion, contact,
and policy contract must all be checked on the Antioch Isaac Sim/Isaac Lab
version. Use PhysX for the first import check; current Newton documentation
lists closed chains as unsupported. See `../../notebooks/ANTIOCH_HUMANOID_LEGS_5.ipynb`.

The reproducible source builder and original Fusion export are on the
`codex/hardware-retrain-local` development branch under `cad/`. This portable
bundle is intended for a smaller deployment branch.
