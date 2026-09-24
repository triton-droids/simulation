# Tracking baseline progress

GPU confirmed outside the sandbox: RTX 5060 Ti, 16 GB, driver 595.84.
PyTorch CUDA tensor execution succeeded. Sandbox GPU visibility was the
cause of the earlier misleading driver failure.

Reinstalled incomplete rsl-rl-lib 5.2.0. Repaired missing wandb files;
wandb and its dependencies were refreshed in the existing virtual environment.
The lockfile has not been changed; installed versions are not fully synchronized
to it. Preserve this distinction when reproducing the environment.

`tracking_task.py` adapts mjlab's existing BeyondMimic-style tracking factory
to the lower-body robot. It wraps the existing ten XML servos and uses the
hip as the motion anchor. The actor receives reference joint positions and
velocities, gyro angular velocity, joint positions/velocities, previous actions,
and projected gravity. It does not receive actual global position, yaw, or
linear velocity. A simulated velocimeter is available only to the privileged
critic. Projected gravity must be reproduced by the real single-IMU estimator;
we have not yet validated its mounting, calibration, timing, or noise model.

`standing_reference.npz` is 500 constant standing frames at 50 Hz, generated
by forward kinematics with the feet on a z=0 plane. It is a wiring reference,
not a walking clip and not an RL-trained controller. The body arrays exclude
the world body. `tracking_body_order.json` records the exact ordering required
when converting future retargeted clips; never copy T800/G1 arrays directly.

Validation: four GPU environments reset and ran eight zero-action steps.
Actor observations had shape (4, 56); critic observations (4, 164).
This verifies task initialization and stepping, not policy learning or balance.
The task is a factory with a standalone training entry point, not yet registered
with the mjlab training CLI.

## First learning check

`train_tracking.py` now runs local PPO with TensorBoard logging, model/reference
snapshots, and checkpoints. Default settings are deliberately only two iterations
and 32 environments. From the `simulation/mjlab` directory:

```sh
UV_CACHE_DIR=/tmp/droids-uv-cache MPLCONFIGDIR=/tmp/droids-mpl uv run --no-sync python ../cad/train_tracking.py --iterations 2 --num-envs 32
```

An initial run revealed that MotionCommand uses the first configured tracked body
to initialize the floating root. Including `floating_base` first fixed immediate
reference resets. The corrected two-iteration run finished on the 5060 Ti and
saved checkpoints under `logs/legs_tracking/20260914_114824`. Its mean episode
length was 23.67 steps; this is a short learning check, not successful learned
standing. Reference validity checks currently cover shapes and finite values,
not semantic ordering, quaternion norms, sample rate, or motion feasibility.

`team_reference_candidate.npz` was recovered from the Isaac v2 branch;
`team_reference_provenance.json` records its source commit. Its ten joint names
match, but it is a teleoperation replay with root z=0.0804 m, incompatible with
the current root frame without conversion. It has not been used for learning
or presented as a human walking clip. The saved walk2 manifest points to an
external `/cephfs` reference not included in that Git branch.

Next: choose/retarget a slow-walking clip for this ten-joint embodiment,
validate its ordering and foot contacts, add a reproducible training entry
point and PPO configuration, then run a short learning test. Actuator limits,
timing, reference tracking tolerances, friction/noise ranges, and updated
CAD COM/inertias remain baseline assumptions requiring calibration before
hardware deployment. The existing browser still runs the hand-written controller.
