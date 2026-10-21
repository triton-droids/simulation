# Velocity locomotion and soccer kick — plan, findings, and status

Status: pipeline built and validated end to end (env, training, evaluation,
ONNX export, video). The learned gait is **not** walking reliably yet; see
[Status](#status) for exactly what runs and what does not.

Everything here lives in `cad/` next to the hardware tracking task and reuses its
hardware model, so sim2real assumptions stay in one place.

## What was added

| File | Purpose |
| --- | --- |
| `cad/velocity_task.py` | `(vx, vy, yaw_rate)` velocity task on the hardware-faithful MJCF: same actuators, slew cap, torque–speed envelope, delays, IMU tilt/bias, gain/friction/mass DR as `hardware_tracking_task.py`. Includes the gait-phase clock and phase reward. |
| `cad/train_velocity.py` | PPO training entry point (rsl_rl, 50 Hz control, 4096 envs), budgeted, ONNX export. |
| `cad/evaluate_velocity.py` | Fixed-command-grid evaluation: fall rate, achieved body velocity, tracking reward, yaw error. |
| `cad/soccer_task.py` | Ball entity + approach command + kick rewards + success termination. |
| `cad/train_soccer.py`, `cad/evaluate_soccer.py` | Soccer training and goal/kick-speed evaluation. |
| `cad/budgeted_velocity_runner.py`, `cad/budgeted_soccer_runner.py` | Wall-clock budget + checkpoint selection. |
| `cad/bc_warmstart.py` | Behaviour-cloning warm start: imitates the retargeted reference into the velocity actor. |
| `cad/teleop_to_motion.py` | Converts a teleop joint log into a tracking reference (`reference.npz`). |
| `cad/render_policy_video.py` | Headless MP4 of a policy. |
| `cad/run_velocity_experiment.sh` | Launcher. |

Commands:

```bash
# train (from scratch or from a warm start)
cad/run_velocity_experiment.sh --num-envs 4096 --target-speed 3.0 --hours 1.0 \
    --run-dir logs/legs_velocity/run1 [--resume logs/legs_velocity/bc_init.pt]

# evaluate on the command grid
cd mjlab && PYTHONPATH=../cad .venv/bin/python ../cad/evaluate_velocity.py \
    --checkpoint ../logs/legs_velocity/run1/best.pt --num-envs 64 --steps 400

# warm start from your retargeted clip
cd mjlab && PYTHONPATH=../cad .venv/bin/python ../cad/bc_warmstart.py \
    --motion ../logs/legs_tracking/20260914_170827/reference.npz \
    --output ../logs/legs_velocity/bc_init.pt --epochs 3000

# soccer
cd mjlab && PYTHONPATH=../cad .venv/bin/python ../cad/train_soccer.py --num-envs 2048 --hours 1.0

# video
cd mjlab && PYTHONPATH=../cad .venv/bin/python ../cad/render_policy_video.py \
    --checkpoint ../logs/legs_velocity/run1/best.pt --out ../logs/legs_velocity/run1/policy.mp4
```

## Findings that change how you should train this robot

### 1. The model's forward axis is `+y`, not `+x`

In both `cad/triton_lower_body.xml` and
`mjlab/src/mjlab/asset_zoo/robots/triton_robot/xmls/triton_lower_body_original.xml`:

* hip roll/pitch/yaw axes are `(0,1,0)`, `(1,0,0)`, `(0,0,-1)`;
* foot collision boxes are `0.04 x 0.105 x 0.0076` half-extents, i.e. 8 cm wide and
  21 cm long **along y**;
* the feet sit at `x = +0.15 / -0.16` (lateral) and `y ~ 0`;
* the deployed reference (`logs/legs_tracking/20260914_170827/reference.npz`)
  swings the feet **0.47 m along y** and only 0.07 m along x, and its root
  travels **3.35 m along y** (0.57 m/s) versus 0.15 m along x.

So a gamepad must map *left stick forward* to `vy`. My first velocity task
commanded `vx`, which asked the robot to sidestep across a 31 cm-wide stance; the
policy correctly refused. `velocity_task.py` now commands forward velocity on
body `y` and keeps the `x` range at ±0.25 m/s.

This also matters for `heading_command` in mjlab, which is defined on the body
`x` axis — it is disabled here; yaw is commanded directly.

### 2. The 1 rad/s target slew is the main sim2real risk, and it is yours to change

`hardware_tracking_task.py` reproduces the deployed bridge's 1 rad/s
position-target slew (a *software* guard written by them, not a motor limit).
Measured on the deployed reference:

| quantity | value |
| --- | --- |
| peak joint speed in the reference | **5.13 rad/s** (ankle), 4.6–5.1 rad/s knee/thigh |
| mean joint speed in the reference | 1.05 rad/s |
| root speed of the reference | 0.57 m/s |
| deployed target slew cap | 1.00 rad/s |

The deployed policy is asked to track a reference that needs up to 5 rad/s while
the command path can only move 1 rad/s. That is the most likely reason the
tracking policy did not transfer, and it is a one-line change in the embedded
bridge.

The velocity task is therefore trained at `--target-speed 3.0` with a 1.4 Hz gait
clock, which is the joint-speed budget a normal walking gait needs. For
comparison, a 1 rad/s cap allows roughly a 0.5 Hz shuffle. **Deploy a policy
trained at the cap you will actually run** — `train_velocity.py
--target-speed 1.0` trains against the current guard, `3.0` trains against a
raised one.

### 3. Standing still is a strong local optimum

A measured reward probe at a 0.55 m/s command (`zero action` vs replaying the
reference clip for 150 steps, per-step reward summed over terms):

| policy | reward/step | falls |
| --- | --- | --- |
| stand still | 0.0007 | none |
| reference replay | 0.0210 | falls after ~8 s |

Walking is 30x better per step, and PPO *still* parks in standing, because the
first step costs a fall and the per-step rewards are small. What actually fixed
the bootstrap was a **behaviour-cloning warm start** (`bc_warmstart.py`): cloning
the reference joint targets into the velocity actor (action MSE 9e-5 rad², i.e.
~0.01 rad per joint) produced a policy that walks forward at 0.65 m/s
immediately — from zero to moving with no RL at all.

A gait-phase clock (cos/sin in the observation, 1.4 Hz, ±0.1 Hz jitter per
episode, plus a lift-and-land foot-height reward) is included as a proto-motion
prior, mirroring your old brax `feet_phase` reward. It is the second lever that
keeps PPO from collapsing back to standing.

### 4. Where AMP / proto-motions fit (and where they do not)

* **AMP / motion priors** are the right tool for *style* and for data efficiency
  when you have demonstrations — they replace the hand-tuned pose terms with a
  learned "does this look like a human" discriminator. They do **not** solve
  commanding; you still need a command-conditioned task like `velocity_task.py`.
* The practical combination used by most humanoid teams today is exactly what is
  in this repo now: **velocity-conditioned PPO + a motion prior** (AMP
  discriminator *or* a phase/foot-height reference, as here) + domain
  randomization matched to measured hardware limits.
* Your retargeted clip is already a usable prior. `bc_warmstart.py` consumes it
  directly; for a true AMP setup you would feed the same clip as the demo buffer.

### 5. Your teleop suite is the highest-value data source you have

Because the rig maps 1:1 into the robot's joint space, teleop logs are
**already retargeted** — no IK, no scaling, no morphology mismatch. That is
better data than any human mocap for this robot. `teleop_to_motion.py` turns a
log into the exact `reference.npz` format your tracking task consumes
(validated: converting your deployed reference reproduces its body positions to
4e-8 m).

Record, per frame at 50 Hz:

* the 10 commanded joint positions (required);
* base pose + velocity if the rig/robot can provide it (otherwise the converter
  produces a walk-in-place reference, which is fine for style and for BC);
* a `time` vector (fps is inferred).

Then use it for (a) BC warm starts, (b) tracking references, (c) an AMP demo
buffer, (d) initial-state distributions. Ten minutes of deliberate walking
(slow, normal, turns, sidesteps, stops) is worth more than hours of clip
retargeting.

### 6. mjlab vs Isaac Lab

Stay in mjlab for this robot. You already have the hardware-faithful MJCF,
torque–speed envelope, delay model and system-ID facts wired into it; Isaac Lab
would mean re-deriving all of that, and its PPO stack is the same rsl_rl. Move
to Isaac Lab only if you need something specific: many-robot asset zoo,
perceptive terrain, cameras/depth for the ball, or a maintained AMP
implementation. For the kick, adding a camera to the ball is the main reason you
might eventually want Isaac Lab (or just add a ball-position observation, as
`cad/soccer_task.py` already does).

## Soccer / kick design

`cad/soccer_task.py` adds:

* a 0.43 kg, 11 cm ball entity (friction 0.8, `condim=3`);
* `BallApproachCommand` — the same 3-D `(vx, vy, yaw)` command, but computed from
  the ball geometry: walk to a stand-off point 16 cm behind the ball, face it,
  and stop there. The actor also observes the ball position in the base frame
  (2 extra dims), so it can time the kick;
* rewards: ball progress toward a goal line 4 m beyond the ball, ball speed
  along the goal direction, foot speed while in contact with the ball, and
  alignment at the stand-off point, on top of the velocity task's locomotion
  terms;
* success termination when the ball crosses the line, plus
  `evaluate_soccer.py` which reports goals/episode, peak ball speed and ball
  travel per episode.

The intended curriculum is: velocity walking (this task) → approach + kick
(soccer task, warm started from the walking checkpoint). Because the soccer actor
has 2 extra observations, the velocity checkpoint cannot be loaded directly —
train the soccer task from scratch, or clip the first-layer input as described
under next steps.

## Status

Verified working:

* `velocity_task` / `soccer_task` build and step on CUDA (action/observation
  contracts asserted in the training scripts: 41-D velocity actor, 43-D soccer
  actor, 50 Hz control, 10 joints in hardware order).
* Budgeted training, checkpoint selection, ONNX export, evaluation, and headless
  video recording all run end to end.
* `bc_warmstart.py` clones the reference clip into the velocity actor (action
  MSE 9e-5 rad², ~0.01 rad per joint). The resulting policy **walks**: 0.66 m/s
  forward in `logs/legs_velocity/bc_walk.mp4`, up from 0.00 m/s for the PPO-only
  policies. It is open-loop imitation, so it falls after ~6 s (fall rate 1.0 at
  300 steps).
* `teleop_to_motion.py` reproduces the deployed reference body poses to 4e-8 m.

Measured failures (each is a concrete next experiment, not a mystery):

* PPO-only policies park in standing: 0.00 m/s achieved for every command, at a
  0.55 m/s command, across 300-450 iterations from scratch (three runs, with and
  without the `+y` forward-axis fix).
* PPO fine-tuning *from* the BC walking init collapses back to standing within
  ~100-200 iterations. Tried: reward rebalance (tracking 3.0, phase 4.0, pose
  0.4), 5 s and 8 s episodes, and `--init-std 0.2` to protect the gait from
  exploration noise. The BC gait is a *feed-forward* replay, so every walking
  episode ends in a fall while standing survives to the time limit; PPO is
  optimising exactly what the task says.
* The soccer policy has not been trained because it needs a walking base.

Bug fixed along the way: `evaluate_velocity.py` defaulted to the 1 rad/s slew
cap while training used 3 rad/s, which made a trained policy look like it could
not walk. Evaluation now defaults to 3.0 and takes `--target-speed`.

## Next steps, in order of expected value

1. **Distill the tracking policy, not the clip.** This is the fix for the
   collapse above. Your deployed tracking policy is a *closed-loop* walker: it
   reacts to balance errors. Roll it out in `cad/tracking_task.py`, log
   `(q, qd, gyro, gravity, last_action, command, phase)` at every control step,
   build the velocity-task observation from those same states
   (`bc_warmstart.py` already computes every term), and clone its actions. That
   gives a feedback-corrected gait as the PPO initialisation instead of the
   open-loop replay that falls at 6 s. Expect the walk to survive, and only then
   fine-tune.
2. **If PPO still sits down**, train with a *termination penalty that is smaller
   than the gait's per-step advantage*, or simply train the velocity task with
   the gait term (`feet_phase`) as the only reward source for the first
   iterations (reward-only bootstrap) before enabling velocity tracking. Both
   keep "fall over" from being cheaper than "walk".
3. **Collect 10 minutes of teleop walking** and repeat (1) and (2) with it.
   Filter to the segment where the robot is stable and the operator is walking
   deliberately.
4. **Raise the deployed slew guard** to ≥3 rad/s (or run a dedicated test) and
   retrain `--target-speed` to match. Without this, expect a shuffle at best.
5. **Kick**: once walking holds for 10 s, train `soccer_task.py`. If the learned
   kick stays weak, add a dedicated kick primitive (short episode, fixed
   stand-off pose) and chain it with the approach policy — a hierarchical
   approach is easier to tune than a monolithic walk-and-kick reward.
6. **Export path**: `best.onnx` is already produced by the runner; verify with
   `cad/verify_tracking_onnx.py`-style checks (named inputs, obs order, action
   scale 0.2, 50 Hz) before any hardware test.
