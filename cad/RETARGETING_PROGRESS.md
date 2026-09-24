# Lower-body Holosoma adapter

Prepared a separate adapter for the existing URKL Holosoma checkout; no
T800 files were modified. `prepare_holosoma.py` creates seven massless
landmark bodies at anatomical points from the existing robot XML. This is
necessary because the original mesh-based link origins coincide and cannot
serve directly as human joint-position targets. Model compilation confirms
unchanged 17 qpos coordinates, 16 velocities, ten actuated joints and 16 kg.
Neutral landmarks have the expected left/right and hip/knee/ankle ordering.

`holosoma_leg_config.py` builds a custom ten-joint Kimodo-format config with
pelvis/hip/knee/ankle matching and no arm/head targets. Config construction
was validated, but the retargeting solver has not been run.

Remaining before solver use:

- Obtain one neutral, straight, slow human walking BVH. Kimodo generation
  is not installed/configured by this adapter; the existing prompt-generation
  script only creates text prompts, not motion. Existing local clips are
  fighting/stance references, not the desired slow walk.
- Validate skeleton units, axis convention and lower-body scaling. The
  adapter's 0.7 m robot-height setting is a placeholder, NOT a validated
  whole-human scale. Scaling full human height to legs-only height would
  shorten the reference legs incorrectly. Use matched hip-to-ankle lengths
  or independently preprocess the lower-body reference.
- Check solver dependencies and collision/foot-lock handling for the marker
  bodies. Foot landmarks are at ankle pivots, not sole contact points.
- Run retargeting and visually validate contacts, pose, limits and velocities.
- Generate training body states using the original non-marker model so the
  body-array order stays compatible with the tracking task.

Suggested motion: natural slow forward walking on flat ground, short steps,
upright pelvis, steady heading, no turns, crouching, kicks or jumping, with
brief standing segments before and after. This is a request for generated
motion, not evidence that a clip has been generated.

## Kimodo setup (2026-09-14)

Cloned the official nv-tlabs/kimodo repository into `droids/kimodo`, commit
`1aece8c124d73d255ceff5086d983b844c9f4e94`. Installed the base CLI and compiled
motion_correction in a separate Python 3.13 environment. Installed official
PyTorch 2.8.0+cu128 wheels; verified RTX 5060 Ti recognition outside sandbox.
`kimodo_environment.txt` records the installed packages.

`generate_slow_walk.sh` specifies one six-second SOMA-RP-v1.1 walking sample,
seed 42, NPZ and standard-T-pose BVH outputs. CPU text encoding is enabled.
Generation was attempted, but no motion was produced. A separate access
probe using existing Hugging Face authentication returned HTTP 403 for
`meta-llama/Meta-Llama-3-8B-Instruct/config.json`: account not authorized.
NVIDIA's installation documentation requires this gated text encoder.
The generation run completed its eleven-file Kimodo download, then failed
while instantiating LLM2VecEncoder with the same gated-repository HTTP 403.
No generation process remains running.

Next action: request access at
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct using the account
associated with the locally configured token. If changing accounts, log in
locally using `droids/kimodo/.venv/bin/hf auth login`; do not send tokens in
chat. Once access is granted, run `bash cad/generate_slow_walk.sh` from
simulation. Retargeting and walking-policy training remain unperformed.

## Access approved and first candidate generated

After the user received approval, generation was retried successfully.
The accelerated Hugging Face download stalled; stopped that process and
retried with `HF_HUB_DISABLE_XET=1`. All required weights downloaded and the
CLI completed on CUDA with CPU text encoding, producing
`motions/slow_walk_seed42.npz` and `.bvh`: 180 frames, six seconds at 30 Hz.
All numeric output arrays are finite. Human root horizontal speed averages
0.783 m/s, so the text prompt alone did not guarantee very slow motion.
`preview_kimodo.py` produces a root-following front/side stick-skeleton GIF
and numeric audit. This is a human reference, not robot simulation.

Important unresolved quality check: Transformers/PEFT loading printed
missing/unexpected LoRA weights followed by explicit adapter loading.
It is not yet verified whether the final text encoder loaded every intended
adapter correctly. Do not promote this candidate to RL training until that
is checked and motion/contact quality and robot-scale speed are reviewed.
No walking retargeting or walking RL was run in this retry.

## Robot reference fit

Created a separate `cad/.retarget-venv` for the existing Holosoma checkout.
Encoder audit found the final default adapter active, with 224 LoRA-B
matrices, all nonzero (`motions/encoder_audit.log`). This confirms a populated
final adapter, not a complete numerical equivalence test of both loading stages.

`retarget_slow_walk.py` adapts source segment directions to XML thigh/shin
lengths (0.271063 / 0.274979 m) and hip spacing, rather than scaling the full
human height to 0.7 m. Overall segment ratio is 0.638055. Exact 77-joint order
is saved in `motions/soma_skeleton.json`. Source sagittal direction had to be
reflected to agree with the XML's negative knee flexion; this mapping uses
point targets, not source joint rotations, and needs physical convention review.

An eight-frame and full 180-frame Holosoma trial completed, but the full fit
had large errors and was rejected for training. `fit_walk_ik.py` implements
a separate direct bounded position-IK diagnostic: corrected-direction fit
has mean ankle position errors 0.51 / 0.64 mm, maximum tracked-point error
3.31 mm. Knees now flex within limits. This is a fallback diagnostic, not a
successful Holosoma training reference.

`render_walk.py` renders the original articulated 16 kg model following the
direct fit. `motions/legs_walk_playback.gif` is kinematic playback: qpos is
set directly without physics. One constant -38.98 mm vertical offset aligns
the global lowest collision-foot point to the floor. Sole heights range
0–50 mm, but stance contact quality is not established; left sole minimum
is 5.36 mm. Ankles reach their +0.6 rad limit in 21.1% / 24.4% of frames.
Therefore do not train from this candidate yet. Next resolve foot orientation,
stance contacts and ankle saturation, review replay visually, then generate
50 Hz original-model body states. Walking RL and hardware execution remain
unperformed.

## Lower-sway replacement

User rejected the first replay's excessive sway. Generated a new seed-73
clip, then four cautious-walk alternatives (seed 91) with Kimodo. Seed 73
reduced tilt but increased lateral travel and speed, so it was not selected.
Selected `motions/cautious_walk/cautious_walk_00.npz`, fitted with the same
XML segment/hip-width mapping and direct IK. Outputs are isolated in
`motions/calm91`; original clip/replay retained. Pipeline scripts now accept
separate input/output directories; `--prepare-only` avoids repeating the
previously rejected Holosoma solver when preparing direct-IK targets.

Selected replay uses 180 poses at 20 Hz rather than source 30 Hz: nine seconds,
a uniform 1.5x time stretch, not additional generated motion. Robot pelvis
tilt peak-to-peak falls from 20.20 to 8.99 degrees (55.5% reduction). Detrended
lateral displacement decreases only slightly, 68.9 to 66.0 mm; mean pelvis
horizontal speed decreases 0.499 to 0.377 m/s. These are kinematic measures,
not evidence of dynamic stability. `calm91/comparison.json` records results.
`calm91/legs_walk_playback.gif` is the replacement preview. Ankles still
reach limits in 20.0% / 26.1% of frames; stance contacts and foot orientation
remain unresolved before RL. No walking training or hardware action ran.

## Full-speed contact refinement and walking RL smoke test

User approved the calmer clip at original speed. Created isolated
`motions/calm91_contacts`: source remains cautious_walk_00, 30 Hz, six-second
playback. Added optional contact-aware direct IK: source contact flags hold
stance target XY, stance-foot collision-box bottom corners are penalized
toward the floor, stronger foot-normal tracking, and ankles constrained
inside +/-0.55 rad (existing physical/XML limits remain +/-0.6).
The source contact flags are kinematic estimates, not force validation.

Before a constant 1.14 mm ground offset, mean absolute stance corner height
is 0.864 mm, maximum 3.154 mm; maximum penetration 1.139 mm. Refined replay
has zero frames at original ankle limits. Position tracking intentionally
trades millimeter-level source fidelity for foot/contact consistency.
`contact_audit.json` and `playback_audit.json` record these kinematic checks.
This does not establish dynamic support, torque feasibility or stability.

`convert_walk_reference.py` converts full-speed replay to 299 frames at 50 Hz,
interpolates root rotations with SLERP, differentiates generalized positions
with MuJoCo, and computes original non-marker 13-body FK/velocities. It asserts
the exact joint/body order. Maximum reference joint speed is 5.133 rad/s;
motor speed/delay/torque calibration and reference smoothness still need review.
Endpoint sampling spans 5.96 seconds rather than including an extra endpoint
beyond the last source frame; do not loop the clip as if it were seamless.

Ran 32 environments and two PPO iterations using this walking reference on
RTX 5060 Ti via mjlab `uv run --no-sync`. Completed 1536 steps; checkpoint
directory `logs/legs_tracking/20260914_155709`, final mean episode length
39.89 steps, mean reward 0.93. This verifies walking-reference RL wiring only,
not a learned walking policy. The actor still uses one-IMU-compatible inputs.
No hardware execution. Next: training/evaluation of a sufficiently learned
policy, including falls, contacts and actuator effort/speed, before sim2real.

## First longer walking training run

Completed 1000 PPO iterations / 6,144,000 environment steps with 256 simulated
robots on RTX 5060 Ti, approximately 3m54s. Checkpoints and agent configuration
are in `logs/legs_tracking/20260914_155930`; final model_999.pt. Training script
now supports checkpoint save interval and saves agent.json for reproducible
evaluation. No robot XML or hardware parameters changed.

`evaluate_tracking.py` ran deterministic policy inference for 500 steps
(10 seconds) across 64 environments, no actor observation corruption but
baseline friction/encoder bias retained. Completed episodes average 7.922s;
some reach the ten-second horizon. This mixes failure and time-limit resets
and is not a success rate. Average joint tracking error metric remains 0.716
and joint-velocity error metric 4.227, so reference execution remains poor.
The first recorded environment lasts 9.98s before its time-limit reset but
advances only 0.284m in forward Y, far below the reference pace. It learned
better balance/limited motion, not the intended human-like forward walk.

Saved actual physics rollout as policy_rollout.npz, .gif and .png in the run
directory. Rendering uses recorded poses without a ground-height correction,
and truncates before the first automatic reset; it is distinct from reference
replay. evaluation.json and rollout_audit.json contain results. Next investigate
why the baseline favors balance over reference execution (action parameterization,
tracking rewards, phase conditioning and feasibility), then train/evaluate an
adjusted baseline. No hardware execution or sim2real claim.

## User's completed 10,000-iteration continuation evaluated

Run `logs/legs_tracking/20260914_160628` resumed model_999.pt; final checkpoint
model_10998.pt. Evaluated using the same deterministic 64-environment,
500-step/10-second test as the prior baseline. Added separate failure/time-limit
counts to the evaluator to avoid confusing survival with locomotion.
Results: one failure reset, 63 time-limit resets; mean completed duration
9.896 seconds. Mean body-position error decreased 0.1167 -> 0.0591 m;
joint-position metric decreased 0.7157 -> 0.3234. This is clear balance and
pose-tracking improvement from longer training.

The recorded first environment advances only 0.151 m in forward Y over
9.98 seconds, compared with 0.284 m for the prior policy. It steps/moves
without following the intended forward pace; robust forward locomotion is
not established. Saved evaluation.json, rollout_audit.json and actual
policy_rollout.npz/.gif/.png in the continuation directory. No training or
hardware action started during this review. Next inspect action/reward and
reference-position alignment before assuming further training fixes pace.
