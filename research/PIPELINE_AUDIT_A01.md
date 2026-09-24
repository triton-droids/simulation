# Pipeline audit and staged screening, 2026-09-24

## Verdict and scope
C16 rejected: nominal survival24/24 but yaw/standing gates fail; randomized
12/24,mean281.667 steps,linear/yaw RMSE .65758/.54722; extra forward10/12.
Sampled nominal combined6000 and forward6010 frames show upright stepping,
not a full visual pass. No more local reward/entropy sweeps are justified now.
No new implementation defect has been established by this audit. Runtime
inference and screening checks below are pending in A01. No training authorized
by the A01 plan (zero training steps,4-hour cap).

## Evidence and distinctions
- Resets: adapter delegates random resets to pinned upstream Joystick.reset;
  upstream samples joint multipliers .5..1.5 and six root velocities -.5.. .5.
  Native reset range fields are NOT forwarded. This known configuration trap
  makes a config-only gradual reset curriculum ineffective. Candidate selection
  defaults1; evaluation forces1. Nominal reset explicitly rebuilds data/obs.
- Observations:103 actor/216 critic; transition command, previous action,phase
  and critic air time are synchronized. Six selected adapter tests passed on
  WSL, including physical-terminal versus timeout reset bookkeeping at lengths1/3.
- Normalization: train.py passes trainer_runtime_controls, which explicitly
  forwards normalize_observations=true. Evaluation builds the same normalizer
  preprocessing, actor key/state and critic key/privileged_state. A01 runs the
  existing saved-policy inference parity audit with64 inputs and a deliberately
  wrong preprocessing negative control. D06 did not find extreme reset scaling.
- Actions: C10 config scale .5,10 simulation frames, action_repeat1, matching
  upstream default scale. Adapter clips normalized actions before upstream PD
  targets. ONNX reference uses the same adapter; D09 raw actions remain within
  bounds. No evidence of action saturation explaining those failures.
- Termination: upstream ends on inverted torso OR specified foot-foot/foot-shin
  contacts OR invalid state; adapter also catches infinities. It does NOT end
  merely at pelvis height .6; .6 is a stricter evaluation gate. Thus a terminal
  at high pelvis height is not itself a bug. Existing D09 trace summaries lack
  which termination sensor fired. Do not call every termination a fall.
- Truncation: full reset capture/restore preserves steps,truncation,episode_done
  and accumulators; selected tests cover physical versus horizon termination.
  Pinned Brax PPO losses use termination=(1-discount)*(1-truncation), masking
  truncation in GAE. No local reversal of terminal/truncation found.
- Restore: C16 restore_parity.json reports exact17 actor/normalizer leaves.
  Linux checkpoint staging is already verified. Warm starts do NOT restore
  optimizer/step/PRNG; these experiments are parameter continuations, not exact
  interrupted-run resumes. This is a training-design difference, not silent
  corruption. Network layers512/256/128 match saved config/evaluation.
- Reference: ONNX same-environment success establishes feasibility of these
  starts, not equivalence of its training recipe/normalization/optimization to
  ours. Do not infer its training history from the exported policy or use it
  as imitation labels. Training samples commands; evaluation holds them fixed:
  intentional held-command task, not an observation mismatch.

## Screening contract
Before full development evaluation: stand and forward.45, seeds6000,6001
(ordinary) plus6002,6014 (difficult), nominal AND randomized,500 steps.
All trained/untrained/standing rows retained:48 episodes versus180 full rows.
Use existing applicable per-regime numeric rules including survival,height,
tracking and forward gait; remove only command-specific rules absent from the
screen. Missing/duplicate/nonfinite evidence is an error. Both screens must pass
before full180 rows execute; pass merely permits full evaluation, never promotion
or final success. This screen may conservatively reject candidates; never relax
it in response to a particular candidate. Full independent validation remains
unchanged. A01 applies this to C10 without retraining to verify rejection and
skip behavior on real data; full stages are frozen and conditionally reachable.
Runner subprocess tests verify rejected-screen dependents do not execute.

## Next decision and budget
First review A01 inference parity and actual screening/skip evidence. If parity
fails, repair only its cause and rerun the audit without training. If it passes,
cheapest next diagnostic is terminal-cause/contact/attitude timing at the two
known failures and an ordinary start (at most6 paired500-step trajectories,
zero training). Prioritize this over new reward coefficients: D09 seed6002
terminates32 steps with minimum pelvis .729, which warrants identifying the
specific contact versus inversion trigger.
Then assess an early gradual reset curriculum: it requires actual state-level
velocity/joint scaling with coherent recomputed sensors/observations, endpoint
parity tests and original evaluation resets. Start modest and increase towards
original reset range during initial learning, not only after nominal walking.
A binary randomized-from-scratch attempt already failed; do not repeat it as a
new idea. Freeze one matched pilot and control, each at most3,368,960 training
steps, only after diagnostics justify exact curriculum and budgets. Screen both;
if neither improves difficult-start survival without ordinary regression, stop
that hypothesis. Do not launch this proposed pilot automatically from A01.
All failed evidence, final control superiority, fresh held-out tests, visual review
and three predetermined fresh independent training seeds remain mandatory.
