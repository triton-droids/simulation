# Frozen Gate 4 Experiment Plan

Status: frozen on 2026-09-08 before baseline seed 0. Earlier runs are explicitly
listed as integration/capacity/tuning pilots and are not counted as baseline seeds.

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

The corrected 512-environment seed-0 run improved its internal shaped return
but failed the fixed held-out suite, and termination diagnostics established
that the original `0.45 m` pelvis cutoff ended nominal rollouts prematurely.
After lowering that cutoff to `0.25 m`, a bounded 2,048-environment capacity
probe fit in GPU memory with more than 5 GiB headroom and reached about 7.7k
reported steps/s. Its selected 196,608-step checkpoint modestly improved
held-out linear RMSE on reset seed 1000 (`1.214` versus `1.228` untrained and
`1.273` standing), but was explicitly too short for a baseline claim.

The final attempt is frozen before training at seeds 0, 1, and 2; exactly
10,485,760 environment steps (160 transition batches), 2,048 training
environments, 32 evaluation environments, 11 evaluations, five PPO updates
per batch, 32 minibatches, batch size 64, and learning rate `2.5e-5`. Episode
length, unroll length, discount, GAE, reward scaling, network sizes, reward,
commands, observation normalization, gradient clipping, random reset
distribution, and the held-out decision rule remain as declared above. The
larger batch is a partial move toward Playground's authoritative 32,768-env,
400M-step G1 profile while remaining bounded for the 8 GiB laptop GPU.

## Research question

Can a compact PPO policy in the repository's MJX/Brax stack learn flat-ground Unitree G1 planar velocity tracking that measurably outperforms (1) an untrained random-initialized policy and (2) a standing-only zero-action controller?

This is a standard G1 velocity baseline, not a HOMIE reproduction.

## Corrective contact/reward probe after failed reference-scaled seed 0

The 10,485,760-step seed-0 run above is retained as a failed experiment. Its
fixed evaluation on seeds 1000--1002 found 100% falls, 0.97 s mean duration,
linear-vector RMSE 1.104, and yaw RMSE 1.971. Reward decomposition showed the
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
are therefore the final frozen Gate 4 family. No evaluation on reset seeds
2000--2002 has been run at the time of this freeze.

The first full seed-0 launch used the probe's five-evaluation cadence and was
stopped after 25 minutes without a step-0 callback. Brax derives the number of
transition batches compiled into one epoch from total steps divided by
evaluation intervals: this made the full run compile 80 batches per epoch,
versus 16 in the completed reference-scaled run. The partial directory is
preserved as a compile-efficiency failure. The final family is refrozen at 21
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
