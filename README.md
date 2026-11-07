# humanoid_legs_5 for Antioch

This is the small deployment branch for Triton Droids' lower-body linkage
inspection asset. Start with [Getting Started](notebooks/GETTING_STARTED.ipynb) to install the tools, sign in to Antioch, open the packaged USD in Isaac Sim.
The portable [MJCF and videos](robots/humanoid_legs_5/) are included with all
24 meshes. `validate_mjcf.py` checks the MuJoCo linkage locally.

The torso placeholder has been removed: the hip assembly is now the physical
floating root and the packaged USD has been regenerated. Isaac Lab/PhysX
fixed-root joint and linkage checks pass. A first flat-ground PPO task and
checkpoint evaluation are in `src/train_locomotion.py` and
`src/evaluate_locomotion.py`; they use the packaged USD directly and control
the 10 motors while leaving four linkage joints passive.

## Standing check

Run a single robot with fixed joint targets, foot-force logging, and no automatic
resets or training:

```bash
uv run antioch service exec --stream -- python src/check_lower_body.py
```

The simulation baseline now uses ankle stiffness 100 and damping 5, retaining
the 16 Nm torque limit, and a neutral starting pose. It held for thirty seconds
with maximum tilt 0.26 degrees. The old 20/2 gains fell past 95 degrees at the
same 200 Hz physics timestep. The task's eight-environment check also completed
thirty seconds with zero resets and maximum tilt 0.28 degrees:

```bash
uv run antioch service exec -- python src/train_locomotion.py --pose-check --pose-check-seconds 30
```

Live rigid-body transforms placed all four corners of each sole collision box
within 0.002 mm of the ground in samples after the first second; each foot
reported 78–79 N vertically. This checks collision-box placement, not the
visual mesh or pressure distribution. A ten-update PPO run on 256 environments
and checkpoint reload/evaluation also completed successfully.

The previous bent-knee task reset still fell with the stronger ankle gains;
it is not the validated starting pose. These are simulation-only standing
results, not disturbance, hardware, or walking validation.
See `robots/humanoid_legs_5/standing_validation.json`. The historical checkpoint
below predates these gains and reset changes and needs re-evaluation.

## Locomotion experiment

### Experimental stepping rewards

For the circling-policy correction, `--reward-profile stepping_straight` adds
world-vertical yaw-rate tracking (weight 2.0, exponential width 0.5 rad/s) and
a squared yaw-rate error penalty (weight -0.5). It retains the stepping
profile's 41 inputs, so a stepping checkpoint can be fine-tuned with it.
The original `stepping` profile remains available for comparisons.

`--reward-profile stepping_wide` retains all `stepping_straight` terms and its
41 inputs, and adds a 0.5-weight spacing reward. It prefers left-to-right sole
center separation of 0.16–0.20 m along heading-frame X (body Y is forward),
with a 0.05 m exponential falloff outside that band. It is active for moving
commands and upright robots, including swing phases. Signed spacing discourages
crossing; forward stride length cannot substitute for sideways separation.
This is an experimental target, not a validated fix for support asymmetry.
Fine-tune a copy of the straight checkpoint with this profile; changing the
profile only during inference does not change an already-trained policy.

Evaluation now reports unwrapped heading drift, world path length, and
forward/lateral displacement of the hip center of mass relative to the initial
heading. Trajectory metrics stop at each robot's first termination and exclude
reset jumps; check the survivor count alongside them. Body-forward speed alone
does not establish straight walking.

`--reward-profile stepping` keeps velocity tracking and the baseline penalties,
adds alternating foot-contact matching (weight 1.0, 0.8 s cycle, 55% stance),
and penalizes stance-foot sole-center sliding (weight -0.2). The posture penalty
only selects hip2 and thigh (lateral/yaw) joints, freeing hip1/knee/ankle motion.
Gait matching switches off for standing commands and inverted poses; slip
remains penalized while standing. Constants are initial experimental settings.

The policy gets sine/cosine gait phase, so this profile has **41 observations**
instead of the baseline's 39. Train a fresh policy; old checkpoints are not
compatible. Evaluation and recording must use the same `--reward-profile`
and `--standing-pose` as training.

```bash
uv run antioch service exec -- python src/train_locomotion.py \
  --standing-pose crouch --reward-profile stepping \
  --num-envs 256 --iterations 1 --steps-per-env 24 \
  --output outputs/locomotion_crouch_stepping_smoke_20261107
```

This one-iteration smoke run completed successfully on 2026-11-07 and saved
`humanoid_legs_5_latest.pt`. It checks execution, not learned walking. Six
reward behavior tests cover alternating versus static/airborne contact,
wrong-leg phase, standing commands, slip, offset sole kinematics, and body
selection. Run them with:

```bash
uv run antioch service exec -- python -m unittest discover -s tests -v
```

`src/locomotion_rewards.py` adapts contact-phase matching from Unitree RL Lab
revision `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`; attribution is in `NOTICE`.
It does not depend on the local Unitree checkout.

### Bent-knee standing option

`--standing-pose crouch` selects coordinated hip/knee/ankle targets of
0.2/-0.4/0.2 radians. `src/standing_pose.py` solves the passive linkage angles
from the MJCF geometry so a crouched reset keeps the loop closed. Neutral
remains the default for compatibility with the recent neutral-pose checkpoint.

The crouch passed a 30-second fixed-target check (maximum tilt 0.81 degrees)
and a 30-second direct-reset task check across eight environments (zero resets,
maximum tilt 1.59 degrees). After settling, the projected center of mass stayed
at least 10.3 cm inside the convex hull of the sole corners and both feet
carried approximately 78 N. See `robots/humanoid_legs_5/standing_crouch.mp4`
and `robots/humanoid_legs_5/crouch_standing_validation.json`. This is a standing
baseline, not a new trained policy or a walking result.

```bash
uv run antioch service exec -- python src/train_locomotion.py --pose-check --pose-check-seconds 30 --standing-pose crouch
```

To reproduce the fixed-target crouch recording (no policy or training):

```bash
uv run antioch service exec --stream -- python src/check_lower_body.py \
  --hip-angle 0.2 --knee-angle -0.4 --ankle-angle 0.2 \
  --seconds 30 --hold-seconds 0 --record-start 22 \
  --video outputs/crouch_review/standing_crouch.mp4 \
  --report outputs/crouch_review/standing_crouch.json
```

For a future fresh crouch training run, add `--standing-pose crouch` to the
training command and use a new output directory. Pass the same option to
evaluation and recording for that checkpoint. Do not apply a new pose reference
to an old policy and expect its original behavior.

The original preview's policy is preserved in
`robots/humanoid_legs_5/checkpoints/humanoid_legs_5_ppo_candidate_1200.pt`
(saved iteration 1200). To replay its historical gains/reset, use
`src/record_locomotion.py --original-preview-settings --checkpoint <that path>`
with `antioch service exec --stream` and a separate `--output` path. It is a
locomotion candidate that mostly stands, not a separately validated standing
controller.

From the repository root in the Antioch project, run:

```bash
uv run antioch service exec -- python src/train_locomotion.py \
  --num-envs 256 --iterations 500 --steps-per-env 24 \
  --output outputs/locomotion_run
```

Evaluate a checkpoint on a fixed forward command (the current task assumes
body Y is forward based on mesh and joint-axis layout; verify this against the
CAD assembly before treating it as a robot convention):

```bash
uv run antioch service exec -- python src/evaluate_locomotion.py \
  --checkpoint outputs/locomotion_run/humanoid_legs_5_latest.pt \
  --num-envs 64 --duration 10 --vy 0.35 \
  --output outputs/locomotion_run/eval_forward.json
```

Record a streamed Isaac Sim preview from the checkpoint you trained:

```bash
uv run antioch service exec --stream -- python src/record_locomotion.py \
  --checkpoint outputs/locomotion_run/humanoid_legs_5_latest.pt \
  --seconds 8 --command-y 0.35 \
  --output robots/humanoid_legs_5/locomotion_preview.mp4
```

The branch includes the selected review candidate at
`robots/humanoid_legs_5/checkpoints/humanoid_legs_5_ppo_candidate_1200.pt`.
To replay its fixed-command evaluation, substitute that path for
`outputs/locomotion_run/humanoid_legs_5_latest.pt` in the evaluation command.
The included preview was recorded from that candidate.

For a gait diagnostic replay, add `--gait-diagnostics --follow-camera` and use
the pose/reward profile matching your checkpoint. Alongside the MP4 and usual
JSON, this writes `<video-name>.gait.json`: 50 Hz foot forces, sole corner
heights, sole center motion, raw/applied actions, and processed joint targets.
Its summary excludes the first two seconds and stops at the first reset.
Load shares are estimates from sampled net vertical foot forces; swing
statistics include only complete contact-to-contact intervals within the window.

The current checkpoint is **not a walking policy**. After a 500-iteration
fine-tune, a fixed 0.35 m/s body-Y request produced about 0.004 m/s mean
body-Y velocity over a five-second evaluation. Use it to reproduce the
experiment, not to claim locomotion success. See the [training
assessment](robots/humanoid_legs_5/locomotion_training_assessment.md), which
also lists the next model and contact fixes. The reviewed candidate checkpoint
and short preview are in the robot bundle; full training logs remain local.

See `robots/humanoid_legs_5/isaaclab_physx_check.json` for import, drive, and
linkage validation, plus remaining model calibration. Development history and
Fusion extraction tools remain on the `codex/hardware-tracking-retrain`
branch (source commit `1d01b69`).

Antioch is configured at this repository root. Run `uv sync --locked` here,
then follow the notebook. The robot meshes are in
`robots/humanoid_legs_5/meshes/`. Older CAD exports and scripts are preserved
locally in `local_archive/cad_exports/`. This archive is excluded from Git;
keep it out of Antioch uploads.
