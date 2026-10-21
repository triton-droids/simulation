# Triton lower-body sim-to-real guide

This is the working procedure for velocity-command locomotion on the Triton
Droids humanoid lower body. The first milestone does not require teleoperation
data. The existing retargeted walk is enough to train a tracking teacher and
bootstrap a velocity policy. Teleoperation becomes useful later for gait
variety, turns, sidesteps, stopping, and kicking.

## Canonical files

| Purpose | File |
| --- | --- |
| Hardware-aware robot structure | `cad/triton_lower_body.xml` |
| Hardware-aware robot plus actuators | `cad/triton_lower_body_actuated.xml` |
| Hardware tracking task | `cad/hardware_tracking_task.py` |
| Velocity-command task | `cad/velocity_task.py` |
| Velocity training and evaluation | `cad/train_velocity.py`, `cad/evaluate_velocity.py` |
| Hardware evidence | `hardware_data/embedded_2728fe1/REPORT.md`, `analysis.json` |
| Ankle mechanism candidate | `cad/ankle_linkage_inspection/triton_lower_body_linkage_candidate.xml` |

Files containing `legacy` preserve the older 16 kg model used by the original
reference-retargeting pipeline. They are not the model to use for new
hardware-aware policy training.

## Stage 1: establish the hardware contract

Freeze the quantities that deployment and simulation must share:

- joint order, sign, zero offset, and position limits;
- control rate, action scale, target clipping, and target slew limit;
- PD gains and motor-to-joint transmission ratio;
- IMU frame, orientation convention, filtering, and timestamp behavior;
- torque-speed envelope, software torque guard, and communication timeout;
- policy observation order and normalization.

The archived logs establish much of this contract at 50 Hz. They do not
uniquely identify friction, inertia, contact, or calibrated joint torque.
The reproducible analysis reports 0.075-0.102 rad tracking RMSE across usable
live trials and a 40-100 ms command-to-velocity response envelope. That response
includes controller, mechanism, contact, filtering, and transport dynamics; it
is not a direct estimate of network delay.

Gate: replaying an identical observation/action sequence through the simulator
and deployment adapter produces the same ordered targets, signs, clipping, and
slew-limited commands.

## Stage 2: targeted system identification

Run this stage in parallel with simulation policy development. It is required
before untethered deployment, but the existing logs are sufficient to train the
teacher and student while these more diagnostic trials are collected.

Use a support fixture so the tested joint cannot drop the robot. Test one joint
at a time near a neutral pose and stay inside conservative position, velocity,
current, and temperature limits.

For each actuator and mechanically distinct load case:

1. Record a stationary hold for sensor noise, encoder bias, and communication
   timing.
2. Apply small position steps in both directions at two or three PD gain sets.
3. Apply a low-amplitude logarithmic chirp across the bandwidth used for
   walking.
4. Repeat at several joint angles to expose gravity and configuration effects.
5. For the ankle, record both motor-side and foot-side angle so the linkage
   Jacobian, offset, and backlash can be fitted directly.

Log monotonic timestamps, requested and applied commands, motor and joint
position, velocity, current or torque feedback, bus voltage, temperature, IMU,
and reply freshness. Prefer the motor feedback rate; preserve the raw samples
before resampling to the 50 Hz policy rate.

Fit delay and command-path dynamics first, then friction and reflected inertia.
Split trials by gain set and reserve complete trials for validation. Do not fit
all walking logs as one dataset, and do not treat the motor's reported torque as
calibrated joint torque without a separate calibration.

Gate: parameters predict held-out step and chirp responses, including phase
delay and overshoot. Domain-randomization ranges cover the held-out residuals
without becoming arbitrarily wide.

## Stage 3: learn a stable walking teacher

Train a phase/reference-conditioned tracking policy from the existing
retargeted walk. Randomize only measured or defensible quantities. Evaluate
recovery from state perturbations instead of judging a single clean rollout.

Gate: at least 95 percent of held-out 30-second episodes finish without a fall,
with no joint-limit violation and acceptable torque, slew, and slip statistics.

## Stage 4: distill velocity control

Roll out the tracking teacher under commands and disturbances. Store the
velocity-task observation and the teacher's corrective action at every control
step. Train a velocity-conditioned student on this closed-loop dataset, then
fine-tune it with PPO.

Use a short observation history or recurrent policy so delay, contact phase,
and unobserved base velocity can be inferred. Evaluate forward, backward,
lateral, yaw, stopping, and command transitions separately.

Gate: held-out command sequences stay upright and track velocity without the
standing-policy collapse seen in the current experiments.

## Where AMP fits

AMP is optional style regularization, not the balance controller. Add it after
the tracking teacher is stable. Train a discriminator on short local-frame
windows containing joint state, projected gravity, base angular velocity, foot
kinematics, and contact state. Exclude absolute world position and yaw. Combine
the discriminator reward with velocity tracking and safety costs, and compare
it against the simpler phase/reference prior.

With only one walking clip, tracking and teacher distillation are usually more
predictable than a full adversarial prior. Add AMP when there are several clean
motions or teleoperation recordings and the goal is smoother transitions and
human-like style.

## Stage 5: real-world alignment

Deploy only standing, weight-shift, and short supported rollouts first. Compare
real and simulated next-state transitions from the same observations and
commands. A residual action model can then learn the repeatable mismatch, as in
ASAP, while randomization covers the remaining uncertainty.

Every hardware rollout must retain the exact policy version, model version,
configuration, raw observations, raw and final commands, safety events, and
timestamps.

## Stage 6: soccer

Treat approach, kick, and recovery as measurable skills. First train a kick
from a stable setup pose, including recovery after impact. Then use the velocity
policy to approach a stand-off pose behind the ball and switch to the kick
skill. Combine them only after both pass their own evaluations.

## References

- AMP: <https://arxiv.org/abs/2104.02180>
- Humanoid-Gym: <https://arxiv.org/abs/2404.05695>
- ASAP: <https://arxiv.org/abs/2502.01143>
- BeyondMimic: <https://arxiv.org/abs/2508.08241>
- ProtoMotions: <https://github.com/NVlabs/ProtoMotions>
- Asimov MJLab velocity policy: <https://github.com/menloresearch/asimov-mjlab>
- Asimov Cyclotron PPO + AMP: <https://github.com/menloresearch/cyclotron>
