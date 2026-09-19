# Gate 4 and Corrective Experiment Plan

## C12 predeclaration: broader commands after verified forward gait (2026-09-19)

C11 passes both nominal forward seeds, dense visual alternation, and fixed
1.5-Hz survival/air-duration checks. Broader commands may now be investigated.
First explicitly prepare a separate checkpoint with lateral/yaw command std
priors .16431676725154984/.27386127875258304: these are the standard deviations
of symmetric uniform ranges +/-.3 and +/-.5 with 10% zero commands. Only
previously constant-zero indices 10/11 in actor/critic normalizers change;
means, counts, weights and other channels remain unchanged. Welford summed
variance is updated consistently. Require exact old-subspace action parity
on 64 inputs and full checkpoint round-trip equality before any training.
The observed unprepared normalization of a .1 command is 100,000. This is a
declared curriculum preparation, not an exact untouched warm-start.

**Frozen stage:** seed 0 from prepared C11 final; vx=[-.3,.6], vy=[-.3,.3],
yaw=[-.5,.5], unchanged 10% zero. Reward weights: tracking linear 3, angular
2.25 (restores upstream 1:.75 ratio), feet_phase 3, feet_contact_phase 2;
other C11 task/PPO settings unchanged, including nominal/noiseless/no-push
reset, 512 envs, 16 eval envs, 500 steps. This is a task-curriculum transition,
not a single-factor causal ablation. First budget 2,007,040 steps, eight evals.
Normal parameter warm-start semantics: optimizer/step/PRNG restart.

**Assessment:** final checkpoint first, full eight-command grid on nominal
development reset seed 4000; use explicit --commands to label the output a
diagnostic. Compare checkpoint-0 `initial` and standing. Provisional broader
command success requires all eight >=400 steps and at least seven full 500;
mean linear/yaw RMSE <=.25 each; forward/backward/left/right/combined linear
RMSE <=.25 each; both pure turns yaw RMSE <=.25; stand linear/yaw <=.15.
Forward must retain 35-95% single support and >=.12 s median completed air
for each foot, and combined-command video must show alternating gait.

**Predeclared extension:** if the full success rule is not met at 2,007,040,
permit exactly 4,014,080 additional steps only if at least six commands
survive 500, mean linear and yaw errors each improve >=20% against the initial
policy, and the forward support/air-duration gate is retained. This second
parameter-warm-start segment resets optimizer/PRNG explicitly; total cap is
6,021,120 new steps. Assess its final checkpoint first. Otherwise stop this
configuration and diagnose; no discretionary unchanged extension beyond the
cap. No final success claim before independent final training seeds and new
held-out command/reset data, including appropriate reset variation.

## C11 predeclaration: sustained phase-aligned support (2026-09-16)

C10 passes its numerical forward checks on seeds 3000/3001 and fixed 1.5 Hz,
but dense video and foot-height traces show irregular brief lifts: median
air intervals .06 s versus oracle .20 s. This is a visual-quality hold, not a
retroactive change to C10's numerical thresholds. Do not broaden commands yet.

**Hypothesis and cheap test:** an optional local reward directly aligning
contact with the existing gait phase will favor sustained alternating support.
Raw reward is `mean(-(2*contact-1)*cos(pre_transition_phase))`, enabled only for
single support and a nonzero command. Standing and flight get exactly zero;
wrong-phase support is negative. This is an original conventional task
extension, not exact upstream reward reproduction, imitation or HOMIE.
On full saved matched 500-step C10/oracle traces, raw means are .157208/.525066.
Weight 2 changes reward totals before dt from 4.538207/4.794796 to
4.852622/5.844928, growing walking's margin from 5.7% to 20.5%. This is
counterfactual arithmetic. Prerequisites: helper invariants, pre-transition
phase/command and dt regression, and real MJX reset/step with weight 0 and 2.
The disabled default leaves old saved configurations/rewards unchanged.

**Frozen pilot:** from local C10 final checkpoint 1003520, all C10 settings
unchanged except `sim.reward_scales.feet_contact_phase=2.0`; seed 0, forward
commands only, 1,003,520 new steps, eight evaluations. Normalizer/actor/critic
warm-start; optimizer/step/PRNG restart. Verify checkpoint-0 actor/normalizer
equality. Stop for nonfinite values or sustained post-initial KL >=.2.

**Gate:** final checkpoint first. Retain C10's numerical/visual forward gate
on nominal seeds 3000/3001, plus 500-step fixed-1.5-Hz survival. To make the
visual requirement reproducible prospectively, each foot's median completed
air interval in that fixed-phase trace must be >=.12 s (excluding intervals
cut by either trace boundary). Initial C10 and standing remain controls;
record yaw drift explicitly. No unchanged extension or command broadening
if the gate fails. These are development diagnostics, not final held-out tests.

## C10 predeclaration: foot-height timing (2026-09-16)

C09 learned useful motion but failed the support gate (25.4% single support),
and its fixed-1.5-Hz seed-2000 trace eventually falls. Do not extend unchanged.
The zero-training comparison in `C10_gait_shaping_diagnostic/` uses identical
physics, command .5, phase frequency and reset for C09 and the oracle. Over
the first fixed 250 steps, C09 remains upright (min pelvis .738660) but has
25.6% single support and phase reward .488307 versus oracle 74.8%/.690843.
This five-second diagnostic window was selected after the full-trace failure
was observed; it is not held-out evidence. Whole traces remain saved.

**Hypothesis:** increasing existing feet_phase weight 1 -> 3 will encourage
alternating foot clearance in the already moving policy. Counterfactual
scoring of those fixed first-250-step traces changes total reward before dt
from 3.074706/3.406496 (C09/oracle) to 4.051320/4.788183: walking's relative
margin grows from 10.8% to 18.2%. This is arithmetic, not learned behavior.
No new reward function, imitation or HOMIE behavior is introduced.

**Frozen pilot:** warm-start local C09 final checkpoint 1003520; retain all
C09 forward-only task/PPO settings and tracking weight 3, change only phase
weight to 3; seed 0, exactly 1,003,520 additional steps, eight evaluations.
Restore normalizer/actor/critic, reset optimizer/step/PRNG as before. Check
checkpoint-0 inference parameter equality. Stop for nonfinite values or
sustained post-initial KL >=.2. Evaluate final checkpoint first.

**Gate:** command .5, nominal reset seeds 3000 and 3001, 500 requested steps
each: >=400 survival, min pelvis >.6, linear RMSE <=.3, >=10 transitions each
foot, 35-95% single support, visually alternating gait with no hopping, and
finite outputs. Compare checkpoint-0 C09 (`initial`) and standing. This test
targets gait quality while retaining tracking; it does not require a further
20% RMSE reduction over already-moving C09. Prior C06 and genuine untrained
controls remain separate evidence, not relabeled as C10 checkpoint 0.
If either seed fails, no unchanged extension or broader commands. Passing
these development tests still requires the fixed-1.5-Hz diagnostic to survive
and new held-out command/reset checks before any final three-seed claim.

## C09 predeclaration (2026-09-16)

C08 completed 2,007,040 steps and failed its final-checkpoint forward gate:
71/500 steps, RMSE .598791, 8.45% single support, 9/1 foot transitions,
minimum pelvis .381873 m. Video shows hopping and collapse. Do not extend
C08 unchanged or broaden its commands.

**Question:** does separating balance acquisition from gait acquisition make
the tracking-weight-3 objective learnable? C07 already provides the cheap
prerequisites: C06 balances for 500 steps, its saved inference is exact, and
the same-MJX walking oracle has a 57.4% reward advantage after reweighting.
C08 from scratch failed to retain balance. Test a forward-only parameter
warm-start from local C06 checkpoint 5007360, not the shipped oracle and not
an omnidirectional stage. This does not satisfy the final three-seed goal.

**Frozen pilot:** C08 task/reward/PPO settings unchanged, seed 0, except
initial normalizer/policy/value parameters come from C06 and the budget is
1,003,520 additional steps, eight evaluations (14 batches per epoch).
Optimizer, training step and PRNG restart: this is not an exact resume.
Restoring the old value function despite reweighting is explicitly part of
this diagnostic; monitor its adaptation. Before judging training, verify
checkpoint 0 retains the source policy and initial evaluation retains balance.
Stop for nonfinite output or sustained post-initial KL >=.2.

**Gate:** final checkpoint first; fixed forward .5, nominal reset seed 3000,
500 requested steps; finite, >=400 survival, pelvis >.6 m, linear RMSE <=.3
and >=20% below initial C06 and standing controls, >=10 transitions per foot,
35-95% single support and visually alternating gait. Reference checkpoint 0
must be labeled `initial`, not `untrained`. C08's genuine untrained control
remains separately available. No unchanged extension on failure. A successful
pilot still requires a new command curriculum and matched independent final
training seeds, with their own balance acquisition and fresh held-out data.

## Successor predeclaration: C08 (2026-09-16)

C07 verified exact saved-action parity and exposed the D-033 autoreset bug.
The same-MJX oracle walks, while C06 stance earns 87.4% of its reward at
`vx=.5`. Before further PPO, finish C07 checkpoint recovery, full regressions,
and a 1,024-step wrapper integration smoke. Preserve all artifacts.

**Hypothesis:** on the corrected wrapper, increasing only linear tracking
weight from 1 to 3 will help escape the low-speed static optimum. The cheapest
test was counterfactual scoring of the saved same-MJX stance and oracle traces:
the gait margin grows from 14.4% to 57.4%. This supports a bounded learning
test but does not prove acquisition. The wrapper correction is a separate
correctness change; a C06-versus-C08 comparison cannot isolate its causal
contribution from reward reweighting.

**Frozen pilot:** from-scratch seed 0, Playground adapter, C06 commands
`vx=[.2,.6]`, `vy=yaw=0` with unchanged 10% zero, nominal/noiseless/no-push/no
domain randomization; corrective PPO, 512 envs, 16 eval envs, 500-step episodes,
20-step unroll, batch 32, 16 minibatches, four updates, LR `.0003`, entropy
`.005`, discount `.97`, reward scale 1, 512/256/128 actor and critic. Only
reward override: `sim.reward_scales.tracking_lin_vel=3.0`. Budget exactly
2,007,040 steps (196 rollout batches), 15 evaluations (14 batches per epoch).
No warm-start. Keep final and training-best checkpoints; assess the final
checkpoint first, without selecting for a prettier rollout.

**Go/no-go:** all finite; post-initial KL below `.2`; diagnostic forward
`vx=.5`, nominal seed 3000, 500 requested steps: survive >=400, minimum pelvis
>.6 m, linear RMSE <=.3 and >=20% below both controls, >=10 transitions each
foot, single support 35-95%, and visually alternating support. Reject static
stance, hopping, falling transitions or numerical reward-only improvement.
Do not scale unchanged or broaden commands if this gate fails. Seed 3000 is
development data, not final held-out evidence. Freeze entirely new command
sequences/reset streams before any final matched three-seed campaign.

Status: handoff freeze on 2026-09-15. No new training is running. Gate 4 has
not passed, HOMIE Phases 5-7 remain out of scope, and the next authorized
research action should be the zero-training-step C07 audit below. The original
2026-09-08 frozen plan is retained afterward as historical provenance.

## Current evidence-validity correction

Commit `f642bc2fbdbc575dc3d4865cdd2fc72a941029e2` found that
`evaluate_g1.py` loaded a checkpoint's running statistics but omitted
`running_statistics.normalize` when reconstructing its PPO network. Every
pre-fix evaluation of a checkpoint actually trained with normalization enabled
is invalid inference evidence—including the original matched Gate 4 family and
C01-C04.
Their training-side Brax metrics and files remain valid. Gate 4 is therefore
not passed, but the old held-out numbers no longer establish a faithful policy
failure either.

This does **not** invalidate identity inference for the pre-D-012 pilots and
initial nominal family: the trainer failed to forward their serialized `true`
flag, so Brax actually trained them with normalization disabled. Those results
remain negative evidence for unintended/misconfigured profiles, not evidence
for the declared normalized experiment.

## C06 declaration and completed outcome

C06 was frozen before execution as a forward-only gait-acquisition diagnostic:

- Playground adapter, exact authoritative reward/dynamics/action semantics;
- seed 0; 5,000,000 requested (`5,007,360` actual) steps;
- commands `vx=[0.2,0.6]`, `vy=0`, `yaw=0`, plus the upstream 10% zero command;
- nominal reset; observation noise, push, and domain randomization disabled;
- 512 environments, 500-step episodes, unroll 20, batch 32, 16 minibatches,
  four updates, LR `3e-4`, entropy `.005`, discount `.97`, reward scale 1, and
  512/256/128 actor/critic networks.

The declared gate required finite values, final KL below `.2`, fixed `vx=.5`
survival of at least 400/500, forward vector RMSE at least 20% below both
checkpoint 0 and standing, minimum pelvis height above `.6 m`, transitions by
both feet, 35-95% single support, and an obviously alternating gait on video.
Only after every condition could it warm-start an omnidirectional stage.

C06 completed in 1,272.85 s. Training-side length rose `69 -> 500`, return
`-1.6033 -> 15.8929`, and final KL was `.0946`. Its first fixed evaluation and
videos omitted normalization and are invalid. The corrected evaluation at
`results/gate4_corrective/C06_playground_forward_curriculum_seed0_5000000/evaluation/checkpoint_5007360_normalized_seed2000_500/`
shows both stand and `vx=.5` surviving 500/500, but forward vector RMSE is
`.4999 m/s`, double support is 100%, single support is 0%, and each foot makes
zero transitions. It learned a static stance, not gait. C06 fails, must not be
rerun unchanged, and must not seed omnidirectional training.

## Next experiment: C07 (no PPO steps)

C07 is an evaluation/invariant audit, not a training run:

1. Add a saved-checkpoint action-parity check between Brax training-time
   network construction and `evaluate_g1.py`.
2. Extend the forced-terminal autoreset test to cover environment history and
   Brax EpisodeWrapper bookkeeping. This is a coverage gap, not a confirmed
   explanation of C06.
3. Evaluate C06 correctly on the evaluator's full eight-command grid for seeds
   2000-2002. Add a tested video-command selector and make one corrected forward
   video; the current hard-coded combined video is out of distribution for C06.
   Treat stand/forward as the only in-distribution focal commands. Because C06
   uses nominal reset, these seeds do not vary the initial pose/velocity and
   must not be presented as randomized-reset robustness. Do not tune on them.
4. Reevaluate the existing C02 and C04 selected checkpoints with normalized
   inference before spending more compute.
5. If inspecting the historical full native family, create a temporary
   worktree at `27de436`, the closest recoverable snapshot of its historical
   transition semantics, and backport only the normalization fix. The old runs
   recorded a dirty base HEAD rather than an exact source commit, and today's
   corrected transition timing is not checkpoint-compatible.

After C07, compare C06's per-step reward/action/contact trace with the C05b
oracle at the same `vx=.5` command. Freeze one bounded seed-0 experiment that
changes exactly one supported gait-acquisition factor. Do not launch a medium,
full, or three-seed run until alternating support is measured and visually
obvious.

## Historical frozen plan (2026-09-08 onward)

The sections below preserve what was declared and observed at the time.
Earlier runs are explicitly listed as integration/capacity/tuning pilots and
are not counted as successful baseline seeds.

## Runtime-plumbing correction

The first nominal three-seed execution of this plan is retained under
`results/gate4/baseline_seed{0,1,2}_2129920`, but is excluded from the baseline
conclusion. Source inspection prompted by measured KL spikes found that
`train.py` serialized `normalize_observations=true` and `max_grad_norm=1.0`
without forwarding either value to Brax; the actual library defaults were no
normalization and no gradient clipping. The configured `action_repeat` was also
implicit rather than forwarded.

Before the corrected baseline, a regression-tested mapping was added for these
three runtime controls. A 1,024-step CUDA integration run checkpointed, and a
491,520-step seed-0 probe improved evaluation return from `1.0062` to `2.0245`
and mean survival from `51.25` to `64.75` steps; after the cold-start update its
reported KL remained between `0.0033` and `0.0036`. The corrected baseline is
therefore re-frozen with the same seeds, environment-step budget, reward,
command distribution, and held-out decision rule below, plus explicitly active
observation normalization, gradient clipping at `1.0`, and action repeat `1`.
No held-out threshold or reward coefficient was changed for this correction.

## Reference-scaled Gate 4 attempt

The corrected 512-environment seed-0 run improved its internal shaped return.
Its then-reported fixed held-out failure was later invalidated by the evaluator
normalization audit; standing-only termination diagnostics still established
that the original `0.45 m` pelvis cutoff ended nominal rollouts prematurely.
After lowering that cutoff to `0.25 m`, a bounded 2,048-environment capacity
probe fit in GPU memory with more than 5 GiB headroom and reached about 7.7k
reported steps/s. Its selected 196,608-step checkpoint was historically
reported to improve held-out linear RMSE on reset seed 1000 (`1.214` versus
`1.228` untrained and `1.273` standing), but that PPO comparison is invalid
after `f642bc2` and the run was explicitly too short for a baseline claim.

The final attempt is frozen before training at seeds 0, 1, and 2; exactly
10,485,760 environment steps (160 transition batches), 2,048 training
environments, 32 evaluation environments, 11 evaluations, five PPO updates
per batch, 32 minibatches, batch size 64, and learning rate `2.5e-5`. Episode
length, unroll length, discount, GAE, reward scaling, network sizes, reward,
commands, observation normalization, gradient clipping, random reset
distribution, and the held-out decision rule remain as declared above. The
larger batch was a partial move toward the paper-v1 Table XX profile of 32,768
environments and 400M steps while remaining bounded for the 8 GiB laptop GPU.
The later pinned Playground source profile is 8,192 environments and 200M
steps; these are distinct upstream snapshots, not contradictory local settings.

## Research question

Can a compact PPO policy in the repository's MJX/Brax stack learn flat-ground Unitree G1 planar velocity tracking that measurably outperforms (1) an untrained random-initialized policy and (2) a standing-only zero-action controller?

This is a standard G1 velocity baseline, not a HOMIE reproduction.

## Corrective contact/reward probe after failed reference-scaled seed 0

The 10,485,760-step seed-0 run above is retained as an inconclusive training
experiment. Its historical fixed evaluation reported 100% falls, 0.97 s mean
duration, linear-vector RMSE 1.104, and yaw RMSE 1.971, but its trained-policy
inference was invalid. Valid training reward decomposition showed the
`alive=3.0` term contributed about 6.85 return at the selected checkpoint,
while the two tracking rewards together contributed about 1.32; the policy
learned a survival maneuver rather than commanded velocity control.

The same audit found that the pinned Menagerie scene uses three capsule geoms
per foot for floor support but separate `*_foot_box_collision` geoms for its
explicit cross-leg pairs. The old code and synthetic test incorrectly used a
support capsule for both roles. The correction uses the model-verified box
IDs for Playground-style cross-foot/cross-shin termination and treats
non-foot ground contact as a nonterminal penalized collision; orientation and
the calibrated height bound still terminate a fall.

Before another full run, freeze one seed-0 probe at exactly 2,097,152 steps
(32 transition batches), 2,048 environments, 32 evaluation environments,
five evaluations, unroll 32, 32 minibatches, batch size 64, five PPO updates,
and Playground's `1e-4` learning rate. Tracking weights are doubled to
`2.0/1.5`, alive is reduced to `0.5`, standstill/pose use Playground's
`-1.0/-0.1`, and every contract-required regularizer remains signed and
nonzero. Commands are narrowed before the probe to the declared evaluation
envelope: x `[-0.5, 0.5]`, y `[-0.3, 0.3]`, yaw `[-0.5, 0.5]`. All other
runtime, reset, noise, model, and network settings remain unchanged.

The probe is a go/no-go check using only its training-side evaluations: mean
episode length must improve over its step-0 value, return must improve without
an unstable post-warmup KL spike above 0.1, and both tracking accumulations
must rise rather than only alive accumulation. If it passes, freeze matched
seeds 0, 1, and 2 at 20,971,520 steps each with the same settings. Because
seeds 1000--1002 have now informed a correction, they are validation evidence,
not held out for the corrective family; the replacement held-out reset seeds
2000--2002 and the eight command vectors below are frozen here and will not be
inspected until all matched runs finish.

The probe completed in 916.5 s and passed that rule. At 1,572,864 steps its
mean episode length was 53.38 versus 50.34 at step 0, return was -0.516 versus
-1.470, linear/yaw tracking accumulations were 34.55/43.91 versus 28.63/27.43,
and post-warmup KL was 0.0131. The matched 20,971,520-step seeds 0, 1, and 2
were therefore declared the final frozen Gate 4 family. At the time of that
historical plan freeze, no evaluation on reset seeds 2000--2002 had been run.

The first full seed-0 launch used the probe's five-evaluation cadence and was
stopped after 25 minutes without a step-0 callback. Brax derives the number of
transition batches compiled into one epoch from total steps divided by
evaluation intervals: this made the full run compile 80 batches per epoch,
versus 16 in the completed reference-scaled run. The attempt remains documented
as a compile-efficiency failure, but the 2026-09-11 audit found no distinct
artifact directory. The final family was refrozen at 21
evaluations (20 training intervals), which restores 16 batches per compiled
epoch. This changes only evaluation/checkpoint cadence: total environment
steps, rollout batches, optimizer updates, learning rate, environments,
rewards, commands, training seeds, and held-out protocol are unchanged.

The first final-family seed-2 process was externally interrupted after its
durable 10,485,760-step callback. It is preserved with an
`interrupted_at_10485760` suffix and excluded from the matched family. The
pinned Brax restore path does not retain optimizer state, PRNG/environment
stream, or its step counter, so resuming its policy parameters would not be an
equivalent continuation. Seed 2 is therefore restarted from step zero with the
exact frozen profile. Reset seeds 2000--2002 remain uninspected.

## Training profiles

- **Integration smoke:** one seed, the minimum vectorized environments and timesteps that exercise reset, PPO updates, evaluation, and checkpoint serialization. Its result is pipeline evidence only.
- **Baseline exploratory:** seeds 0, 1, and 2; exactly 2,129,920 environment steps (130 transition batches), 512 training environments, 4 evaluation environments, 11 evaluations, episode length 1,000, unroll 32, 16 minibatches, batch size 32, 2 PPO updates per batch, learning rate `5e-5`, discount `0.98`, GAE `0.95`, and reward scaling `0.1`. Actor layers are `(512, 256, 64)` and critic layers `(256, 256, 256, 256)`. No domain randomization or pushes are enabled. Runs use the validated WSL2 CUDA device.
- **Fallback when compute is genuinely insufficient:** retain all smoke/partial checkpoints and report achieved timesteps and wall time without elevating the result to Gate 4.

The budget is intentionally much smaller than the paper's 400M-step, 32,768-environment research configuration because the available GPU is an 8 GiB laptop device and the assignment is a 1–3 week exploratory prototype. A 512-environment probe completed without OOM; the accepted 491,520-step tuning pilot reached roughly 5.3–6.2k steady-state steps/s. All three frozen seeds will use this same profile unless an actual hardware failure prevents completion.

## Held-out evaluation suite

Use fixed command segments and seeds not used for reset streams during training. Each controller receives the same initial states and command trace. Proposed planar commands in `[v_x, v_y, yaw_rate]` are:

1. stand `[0.0, 0.0, 0.0]`
2. forward `[0.5, 0.0, 0.0]`
3. backward `[-0.3, 0.0, 0.0]`
4. left `[0.0, 0.3, 0.0]`
5. right `[0.0, -0.3, 0.0]`
6. turn left `[0.0, 0.0, 0.5]`
7. turn right `[0.0, 0.0, -0.5]`
8. combined `[0.4, 0.2, 0.35]`

Evaluate deterministic trained policies, an untrained policy network with the matched architecture/seed, and a standing controller that always returns zero action. The checkpoint selection rule is highest mean evaluation reward reported during training, with all checkpoints retained outside Git.

## Metrics

Machine-readable per-seed/per-command output will include linear velocity RMSE and MAE, yaw-rate RMSE and MAE, fall flag/rate, episode duration, success rate under declared error thresholds, torso tilt, absolute mechanical-power proxy, actuator effort, action-rate cost, joint acceleration, foot-slip proxy, undesired-contact rate, soft joint-limit violations, and left/right gait timing asymmetry. Wall time, environment steps, software versions, Git dirty state, model pin, and hardware are recorded alongside metrics.

Representative videos will be generated from a fixed combined-command rollout for each controller class. Videos are diagnostic evidence, not a substitute for the fixed metrics.

Before held-out evaluation, fix the representative policy to training seed 0
and its training-side selected checkpoint. Render the combined command
`[0.4, 0.2, 0.35]` from reset seed 2000 for trained, matched untrained, and
standing controllers. Do not replace these videos with a visually better seed
after inspecting the evaluation.

## Decision rule

Gate 4 requires finite evaluation and the trained controller to beat both controls on aggregate tracking error while retaining nontrivial episode survival. If only a best seed succeeds, if controls are evaluated on different traces, or if the trained policy exploits termination/reward clipping, Gate 4 does not pass.

Before inspecting reset seeds 2000--2002, interpret that rule mechanically as
follows. Across the matched three-policy family, the trained controller's mean
linear-vector RMSE and mean yaw-rate RMSE must each be lower than both the
matched untrained-policy and standing-controller means. Every evaluated value
must be finite. Trained mean episode duration must exceed both controls and its
family fall rate must be below one, demonstrating at least one full-horizon
rollout. Finally, at least two of the three trained policy seeds must individually
beat both of their matched controls on both tracking RMSEs, exceed both on mean
duration, and have fall rate below one. Video review and clean-checkout command
verification remain separate mandatory Gate 4 checks.

## Post-evaluation outcome (appended 2026-09-09)

The frozen family and held-out suite completed without changing the rule above.
Each of policy seeds 0, 1, and 2 ran for exactly 20,971,520 environment steps.
Training-side selection chose checkpoints 17,825,792; 19,922,944; and
20,971,520 respectively. The evaluator then produced the complete Cartesian
grid of three controllers, eight commands, and reset seeds 2000-2002 for every
policy seed: 216 finite episodes in total.

The trained family mean linear-vector RMSE (`0.9254`) beat untrained (`0.9878`)
and standing (`0.9942`), and trained duration (`1.3742 s`) exceeded both
controls (`1.1364/1.1333 s`). The trained yaw RMSE (`1.0575`) was substantially
worse than both controls (`0.4539/0.3238`), however, and all three controllers
had fall rate `1.0`. No trained rollout survived the 500-step horizon and zero
of three policy seeds passed the individual matched rule. The frozen decision
therefore evaluates to **false**, and Gate 4 fails.

A source audit after the family was evaluated also found that `Joystick.step()`
constructs the returned observation before shifting `last_act`, advancing phase,
and resampling its command. This creates an extra-step action lag and a command
mismatch on resampling boundaries. Correcting it would change policy semantics,
so the completed artifacts remain untouched and strict expected-failure tests
record the required future behavior. Any corrected training must be declared as
a new family with a new pre-held-out freeze; these held-out results must not be
relabelled. HOMIE Phases 5-7 remain unstarted.

## Post-evaluation correction (2026-09-11)

The 2026-09-09 decision above is historical, not current. Commit `f642bc2`
proved the evaluator omitted observation normalization for PPO checkpoints.
The 216 raw rows and threshold calculation remain preserved, but they do not
describe the trained policies under the inference function used during
training. The observation/history bug was separately fixed for new families in
`4426499`; it remains another reason not to evaluate the old policies under the
current native environment.

Current Gate 4 status is **not passed / prior held-out verdict invalidated**.
C05b proves gait feasibility, and C06's only valid corrected PPO evaluation
shows static balance rather than walking. The current plan is the C07
zero-training-step audit at the top of this file. HOMIE Phases 5-7 remain
unstarted and unauthorized.
