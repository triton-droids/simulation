# Decisions and Gate Log

## D-001 — Separate G1 environment rather than array-size generalization

**Decision:** Add `source/locomotion/unitree_g1` and G1-specific metadata/configuration. Keep `source/locomotion/default_humanoid_legs` recognizable and separately tested.

**Alternatives considered:** (a) replace every hard-coded 12 with `nu`; (b) replace the repository with MuJoCo Playground; (c) add a disconnected top-level project. All three either preserve invalid robot assumptions or expand the task unnecessarily. The selected seam is the smallest reversible option.

## D-002 — Native Brax `PipelineEnv` adapter for Gates 2–4

**Decision:** Load the pinned `scene_mjx.xml` through the existing Brax/MJX pipeline and expose the dictionary observations expected by the existing PPO code.

**Reason:** This preserves the repository's registry, training, checkpoint, and rendering surfaces. A current Playground runtime wrapper would require a new dependency and still need a model/contact adapter to honor the repository pin. Brax's maintenance warning is recorded; this is not a commitment to a broader platform direction.

## D-003 — Playground behavior, pinned Menagerie assets

**Decision:** Adapt G1 task semantics from MuJoCo Playground commit `8a4b4642d8eba8a80ac99ed125cb62c16e1457ad`, but resolve assets from Menagerie commit `71f066ad0be9cd271f7ed58c030243ef157af9f4` only.

**Consequence:** Contact and foot-velocity calculations cannot be copied literally because Playground's current G1 XML adds feet-only geoms and custom sensors that the pinned scene does not contain. Those values will be derived from the pinned model's sites, body velocities, and explicit contact pairs and covered by tests.

## D-004 — Include base linear velocity in the actor observation

**Decision:** Include local pelvis linear velocity in the Phase 4 actor observation.

**Reason:** It is part of the selected MuJoCo Playground G1 joystick actor input and gives the small prototype a direct velocity error signal. The actor observation is therefore 103 values: local linear velocity (3), gyro (3), projected gravity (3), command (3), relative joint position (29), joint velocity (29), previous action (29), and foot phase (4).

## D-005 — Local logging is the default

**Decision:** Training writes JSONL metrics locally by default. W&B is opt-in; offline W&B is an explicit mode and imports the package lazily.

**Reason:** Tests, smoke runs, and PPO integration must work with no account, network session, or credentials.

## D-006 — Pre-existing default-environment defects

The pre-change audit found:

1. Privileged observation configuration is 88 while construction produces 112.
2. The scheduled push creates a replacement velocity but discards the immutable `tree_replace` return value.
3. The manual smoke module prints nonexistent `State.q`/`State.qd` fields after a successful reset and step.
4. `pytest` was neither installed nor declared.

These are baseline defects, not G1 regressions. Gate 1 will add tests first and apply only narrow fixes needed for the contract checks.

## D-007 — Temporary Brax/JAX pmap compatibility adapter

**Decision:** Retain the validated JAX 0.11/MuJoCo 3.10 environment and install local drop-in implementations of the two removed `jax.device_put_*` helpers before entering pinned Brax 0.14.2 training.

**Evidence:** The first 1,024-step PPO integration attempt failed before training because Brax called removed `jax.device_put_replicated`. JAX's official migration guide publishes public-API drop-in replacements. Downgrading JAX would also require changing the validated Flax/Orbax stack and the later WSL CUDA wheel, while upgrading/replacing Brax is a materially larger framework change. The adapter is small, tested, training-only, and explicitly temporary.

The first retry passed initialization and emitted a finite step-0 evaluation, then exposed a second inherited compatibility mismatch in the repository's lightweight checkpoint saver. Inspection of the complete Orbax tree showed that Brax 0.14.2 supplies a three-item sequence `(normalizer, policy, value)`, while the old saver assumed a nested attribute. Failed attempts are retained under `results/gate4/ppo_smoke_retry1` and `ppo_smoke_retry2`; a tested adapter now accepts current, historical nested, and already inference-shaped layouts.

The third retry serialized the compact step-0 policy, then failed at the trainer's between-epoch reset because `use_pmap_on_reset=False` selects a Brax 0.14.2 branch that sends a batched `(num_envs, 2)` key directly into the already vectorized wrapper. Restoring Brax's documented/default `True` path keeps explicit device and environment axes and is the smallest configuration correction; it does not change the environment's single-key reset contract.

## D-008 — Preserve signed G1 reward totals

**Decision:** Sum the weighted G1 terms and multiply by the control timestep without applying the old environment's nonnegative clamp.

**Evidence:** The selected Playground G1 implementation preserves signed totals. The first 128-environment capacity probe showed that the inherited clamp converted the `-100` terminal cost into zero total reward at a fall, erasing the intended survival signal; evaluation reward fell from `0.0657` to `0.0200` over 65,536 steps and KL reached `4.85`. The corrected 512-environment probe retains a roughly `-2` terminal return and has an exact aggregation regression test. This is a reference-alignment correction, not reward tuning.

## D-009 — Laptop-GPU execution profile

**Decision:** Use WSL2 CUDA with JAX preallocation disabled, 512 environments, and an ignored persistent compilation cache for Gate 4 runs.

**Evidence:** Native Windows JAX exposes CPU only. WSL2 JAX 0.11.0 executed a compiled operation on the RTX 4060. A 512-environment/full-network probe completed without OOM and achieved about 831 environment steps/s; 128 environments achieved about 747 steps/s. Identical seed configurations can reuse JAX's local cache without changing numerical inputs or tracked source.

## D-010 — Survival shaping after measured early-termination exploitation

**Decision:** Enable the conventional upstream-defined alive term at scale `3.0` and use two PPO updates per batch at learning rate `5e-5` for the next matched pilot.

**Evidence:** With signed totals but no alive term, the 983,040-step policy increased reported return by collapsing mean episode length from `22.9` to `3.8` steps. Alive scale `1.0` plus gentler PPO controlled KL near `0.03–0.05`, but its 491,520-step final survival (`14.6`) still trailed the untrained policy (`19.25`). From the measured returns and alive accumulations, scale `2` only makes the observed 20.75-step and 8.125-step policies approximately tied; scale `3` creates a clear incentive margin. This tuning is explicitly provisional until the matched pilot shows survival improvement.

## D-011 — Contact classification and auxiliary-cost rebalance

**Decision:** Terminate non-foot ground contact and Playground's explicit foot/foot or foot/opposite-shin contacts; apply a nonterminal collision cost to same-side hand/thigh contact. Keep every contract-required auxiliary cost nonzero, but reduce vertical velocity, roll/pitch rate, action rate, acceleration, standstill, and pose weights where the measured falling transient overwhelmed survival and tracking.

**Evidence:** A fixed held-out rollout terminated at step 11 solely because `left_hand_collision` touched `left_thigh_collision` while both feet remained on the floor. The authoritative Playground G1 implementation penalizes that pair but does not terminate it. After correcting classification, the current-code PPO smoke's untrained mean episode length rose to `42.5`, but its accumulated weighted vertical velocity (`-72.9`), roll/pitch rate (`-64.7`), and standstill (`-56.2`) terms outweighed alive (`+127.5`), tracking, and phase terms before `dt`, producing `-3.97`. Playground currently sets several of these auxiliary weights to zero; the selected compromise keeps them testably signed and nonzero as the contract requires, while preventing the falling transient from dominating the learning objective.

## D-012 — Enforce serialized PPO runtime controls

**Decision:** Forward configured observation normalization, gradient clipping,
and action repeat explicitly to Brax through a separately tested mapping.

**Evidence:** Three 2,129,920-step runs serialized
`normalize_observations=true` and `max_grad_norm=1.0`, but source inspection
after repeated KL spikes (up to `9.2376`) showed that `train.py` omitted both
arguments. Brax 0.14.2 therefore used its materially different defaults:
normalization disabled and no gradient clipping. The invalidated runs are
retained. A corrected 1,024-step CUDA smoke checkpointed, and a corrected
491,520-step probe improved return `1.0062 -> 2.0245` and survival
`51.25 -> 64.75` while post-warmup KL stayed near `0.0034`.

## D-013 — Calibrate the G1 pelvis-height termination

**Decision:** Lower the minimum pelvis height from `0.45 m` to `0.25 m`, while
retaining inverted-torso, non-foot ground, cross-leg, high-pelvis, and invalid
state termination.

**Evidence:** In the fixed nominal-reset diagnostic, the zero-action controller
hit the `0.45 m` cutoff after 67 control steps while mean torso tilt was only
`15.76 deg`. At a diagnostic `0.25 m` cutoff it continued to 72 steps and
reached `0.218 m`; the other contact/orientation criteria remain available to
identify a physical collapse. Native MuJoCo rollouts of both Menagerie
keyframes showed that neither is passively stable and that `knees_bent` lasts
slightly longer before non-foot contact, so the selected Playground keyframe
is retained. The lower cutoff also avoids precluding the contract's possible
future squat range.

## D-014 -- Separate support and cross-contact foot geoms

**Decision:** Introspect and serialize the pinned model's distinct
`left/right_foot_box_collision` IDs. Use the three foot capsules only for
floor support, use the boxes for cross-foot/cross-shin termination, and
penalize non-foot ground contact without immediately terminating it.

**Evidence:** Menagerie `scene_mjx.xml` defines floor pairs on three capsule
geoms per foot but its explicit cross-leg pairs on a separate box geom. The
previous implementation and test constructed a cross-leg contact with a
capsule ID that the real model never uses for that pair. Playground's G1
termination is cross-leg/orientation based and does not terminate its
same-side hand/thigh collision; height and inverted-torso tests continue to
detect a completed fall in this full-collision Menagerie scene.

## D-015 -- Restore tracking dominance after measured survival exploitation

**Decision:** Reduce alive shaping from `3.0` to `0.5`, double the two tracking
weights to `2.0/1.5`, restore Playground's standstill/pose weights, narrow the
training commands to the fixed evaluation envelope, and probe at Playground's
`1e-4` learning rate before any matched rerun.

**Evidence:** The failed 10,485,760-step seed-0 checkpoint had 100% falls and
yaw RMSE `1.971` on the fixed evaluation. Its logged alive contribution was
about 5.2 times the two trackers combined after timestep integration. This
made increasing survival with command-insensitive high-rate motion more
valuable than tracking. The corrective profile keeps every contract-required
regularizer signed and nonzero and is frozen in `research/EXPERIMENT_PLAN.md`.

## D-016 -- Bound compiled epoch size with evaluation cadence

**Decision:** Use 21 evaluations for the 20,971,520-step matched family so
each Brax epoch contains 16 transition batches, and preserve the interrupted
five-evaluation attempt as a failed runtime experiment.

**Evidence:** The five-evaluation full seed compiled 80 batches into one epoch
and produced no step-0 callback after 25 minutes; the otherwise identical
probe and prior full profile compiled 16-batch epochs successfully. Evaluation
cadence does not change the total rollouts or PPO updates, so this is the
smallest reversible runtime correction. It does increase checkpoint frequency
and is recorded before the replacement seed starts.

## D-017 -- Restart interrupted seed 2 instead of partial-parameter resume

**Decision:** Preserve the interrupted seed-2 artifact at 10,485,760 steps and
restart seed 2 from step zero with the unchanged frozen Gate 4 configuration.
Do not count a continuation from the saved policy parameters as a matched final
seed.

**Evidence:** An interactive execution handle was closed by a status-message
interruption after the durable 10,485,760-step callback. Inspection of the
pinned Brax 0.14.2 PPO implementation shows that
`restore_checkpoint_path` restores the observation normalizer plus policy and
value parameters, but not the optimizer state, PRNG/environment stream, or
environment-step counter. A continuation would therefore restart optimization
and relabel its steps from zero, making it non-equivalent to uninterrupted seeds
0 and 1. The partial metrics and all checkpoints are retained as an interrupted
runtime result; the replacement is launched independently of the interactive
tool handle so later status questions cannot terminate it.

## D-018 -- Gate 4 fails the frozen three-seed decision rule

**Decision:** Record the authorized Gate 4 attempt as a failure and stop before
HOMIE Phases 5-7. Do not promote a visually preferred checkpoint or weaken the
predeclared rule after seeing reset seeds 2000-2002.

**Evidence:** Across 216 fixed held-out episodes, trained linear-vector RMSE
was `0.9254` versus `0.9878` untrained and `0.9942` standing, and mean duration
was `1.3742 s` versus `1.1364/1.1333 s`. However trained yaw RMSE was `1.0575`
versus `0.4539/0.3238`, every controller had fall rate `1.0`, no rollout
survived the 500-step horizon, and zero of three trained policy seeds passed
the matched rule. All values were finite. The exact aggregate is retained at
`results/gate4/final_aggregate/summary.json`.

## D-019 -- Preserve the discovered observation-timing defect

**Decision:** Do not change actor-observation timing underneath the completed
checkpoint family. Add strict expected-failure tests for the correct invariants,
document the defect as a Gate 4 limitation, and require a fix before any new
baseline family.

**Evidence:** `Joystick.step()` currently calls `_get_obs` before shifting
`last_act`, advancing phase, and resampling the command. Thus the returned
observation's "previous action" is one control step older than the just-applied
action, and at a resampling boundary its command differs from the returned
`info["command"]`. Moving observation construction would change every future
policy input and make the already-trained checkpoints incomparable. The frozen
family already fails on yaw and survival, so no additional training is justified
under the current semantics.

## D-020 -- Use a repository-local video encoder fallback

**Decision:** When MediaPy cannot find system `ffmpeg`, point it to the
platform-specific binary supplied by pinned `imageio-ffmpeg==0.6.0`; retain
OpenCV's MP4 writer as a last fallback. Do not install a system package merely
to finish representative evidence.

**Evidence:** The fixed trained rollout completed, then encoding failed because
WSL had no `ffmpeg` on `PATH`. The failed attempt is retained. The bundled
encoder was functionally checked on Windows and WSL, and a regression test
forces and verifies the OpenCV fallback.

## D-021 -- Start a new corrective family with functional transition timing

**Decision:** Treat every run after commit `27de436` as a new, non-comparable
corrective experiment family. Compute the reward from the command, phase,
previous action, and accumulated foot-air time that governed the transition;
then advance history/resample the command and construct the next observation.
Initialize reset actuator targets from the sampled joint pose and never mutate
the input state's `info` or `metrics` dictionaries.

**Evidence:** The preserved Gate 4 environment returned an observation with an
extra-step-stale previous action and, at resampling, a command different from
its returned `info`. It also reset foot-air time before reward evaluation, so a
genuine touchdown after `0.4 s` received the negative `-0.2` raw term instead
of a positive reward; every Gate 4 training log accumulated a negative
feet-air-time term. Finally, reset forwarded the pose with zero actuator
targets (summed absolute reset force about `380.1`) while metadata claimed the
pose was held. Integration tests now encode all four invariants. Playground's
current `step` also constructs its observation before history updates; that
ordering is intentionally corrected rather than copied because it violates the
stated observation semantics.

## D-022 -- Pinned Playground/native semantic comparison

**Decision:** Keep the pinned Menagerie/Brax seam for one corrected short
diagnostic, but make its differences from Playground explicit. Escalate to a
thin pinned Playground adapter if corrected short training still lacks a
plausible balance/gait signal. Do not copy an apparent stale upstream joint
range or known reward implementation defect merely to make arrays equal.

The comparison used Playground commit
`8a4b4642d8eba8a80ac99ed125cb62c16e1457ad`, including its G1 XML overlay,
environment, randomizer, and `locomotion_params.py`, against Menagerie commit
`71f066ad0be9cd271f7ed58c030243ef157af9f4` and the live native environment.

| Area | Pinned Playground | Native corrective status / decision |
|---|---|---|
| Model and scene | Feet-only flat MJX overlay; expects Menagerie `1b86ece...`; `36/35/29`, 31 bodies, 72 geoms, 5 pairs, 29 sensors | Pinned newer Menagerie `scene_mjx.xml`; same state/action/body counts, 63 geoms, 49 explicit full-collision pairs, 14 sensors. Preserve for the first diagnostic and instrument extra contacts. |
| Joint and actuator order | Same 29 names and one-to-one order | Exact equality is introspected and tested. |
| Joint/control ranges | Same except Playground right-hip-roll `[-0.5236, 2.9671]` | Menagerie has mirrored/right range `[-2.9671, 0.5236]`; retain the pinned model value rather than copy the apparent left-side range. Targets are tested against all live ranges. |
| Default pose | Same leg/waist pose; arm rolls `+/-0.2`, elbows `0.6` | Pinned `knees_bent` arm rolls `+/-0.22`, elbows `1.0`; retain exact model keyframe and serialize it. |
| Action | 29 position offsets, scale `0.5`; no environment-side normalized/target clip | Same scale; explicitly clip normalized actions and physical targets. Evaluation reports both action and target saturation. |
| PD and integration | Euler, 2 ms; joint damping mostly `2`, ankle pitch `1`, ankle roll/wrists `0.2`; gains mostly `75`, ankle pitch `20`, ankle roll/wrists `2` | 2 ms override but inherited ImplicitFast; actuator velocity gain `2`, zero joint damping; ankle roll/wrist gains `20`. This is the largest unresolved dynamics difference and the next controlled ablation if the corrected run fails. Both disable Euler damping and use solver/iterations/line-search `2/3/5`. |
| Control rate | 10 substeps, 20 ms / 50 Hz | Exact match. |
| Actor/critic observations | 103/216 values: local velocity, gyro, gravity, command, joint offsets/velocities, previous action, phase; privileged physical state/contact/site velocity/air time | Same dimensions, order, frames, and noise scales. Previous action, command, and phase now match returned next-state metadata; true foot-site velocity replaces body-origin velocity. |
| Action history/timing | Reward uses old action; current upstream returns obs before shifting history | Reward uses old action and returned obs uses just-applied action. Functional state update has regression tests. |
| Commands | Uniform independent `[vx,vy,yaw]`, 10% zero; ranges `+/-1.0`, `+/-0.5`, `+/-1.0`; effectively resamples after 501 steps | Same representation and zero probability; conservative `+/-0.5`, `+/-0.3`, `+/-0.5`; exact 500-step interval. Transition reward uses old command and next obs uses resampled command. |
| Reward kernels/signs | Exponential planar/yaw tracking; signed costs; no total clipping | Same tracking kernels/sign convention and signed aggregation. Native keeps prior measured nonzero regularizers and `alive=0.5`; Playground defaults several to zero, adds contact-force/hip/knee terms, and has tracking `1/0.75` versus native `2/1.5`. Touchdown ordering is corrected. |
| Foot slip | Current code uses pelvis speed times contact despite defining foot-site sensors | Native uses squared true foot-site planar velocity times support contact; retain the semantically correct quantity. |
| Termination | Inverted torso, cross-foot/cross-shin sensors, NaNs | Adds low/high pelvis and full-state finite check. Full-collision non-foot ground remains separately observable so evaluation can reject crawling/falls. |
| Contact detection | Two foot-floor found sensors in feet-only scene; selected self/cross sensors | Explicit named Menagerie pairs: three support capsules per foot, separate cross-contact boxes/shins, hand-thigh and non-foot-ground categories. All categories are tested. |
| Orientation and velocity frames | Torso up-vector; pelvis-local planar velocity/gyro yaw | Exact sensor convention. Tests prove positive yaw sign and the world-to-local `+90 deg` transform. |
| Reset | XY/yaw, joint multiplier `0.5..1.5`, base velocity `+/-0.5`; control initialized to sampled qpos | Same distribution (with joint-range clip), now correct initial control. A 2,000-reset probe found only 49.8% double support, so nominal reset is used only for the first diagnostic; broader reset is restored progressively. |
| Push/domain randomization | Push enabled; optional friction, frictionloss, armature, mass, torso-mass, and qpos0 randomization | Push and domain randomization remain off during diagnosis. Construction now explicitly rejects G1 domain randomization because the generic path lacks G1 torso/contact mapping and its qpos0 jitter cannot reach the fixed reset pose; it can no longer fail later or silently pretend to randomize. |
| PPO | 200M steps; 8192 envs; unroll 20; 32 minibatches; 4 updates; discount `.97`; LR `3e-4`; entropy `.005`; reward scale `1`; `512/256/128` actor and critic | Failed family used materially different low-entropy/low-LR/scaled-reward settings. New corrective profile preserves Playground optimizer/network semantics while scaling the batch to 512 laptop environments with `32*16=512`; budget starts at 262,144 steps. |

## D-023 -- Strengthen held-out gait and exploit diagnostics before tuning

**Decision:** A held-out episode cannot be successful if it terminates on its
final requested step. Evaluate action clipping/target saturation, non-foot
ground and self-contact rates, double/single/flight support fractions, per-foot
contact transitions, and transition cadence in addition to tracking, horizon,
tilt, height, effort, and slip. Use the environment-applied normalized action
for action-rate cost.

**Evidence:** The earlier evaluator's success predicate omitted `not fall`, and
its raw-policy action rate disagreed with the reward whenever a policy output
saturated. Contact asymmetry alone cannot distinguish alternating walking from
both-feet-planted or hopping motion, and the combined collision bit cannot
identify knee-crawling. The old foot velocity was measured at the ankle-body
origin; an independent MuJoCo site-Jacobian check found `0.038/0.022 m/s`
left/right differences on a moderate test velocity. The rigid-body point
formula now matches the site definition, and synthetic regression traces
exercise the reporting cases.

## D-024 -- Preserve the failed PPO profile and add a corrective one

**Decision:** Keep `ppo_g1` as the historical failed-family configuration and
register `ppo_g1_corrective` for new work. The corrective profile uses the
pinned Playground learning rate, entropy, discount, reward scale, update count,
and actor/critic widths, but starts at 512 environments and 262,144 steps.

**Evidence:** Gate 4 used reward scale `0.1`, discount `.98`, zero entropy,
different critic layers, and often a learning rate below Playground's tuned
profile. Those differences are too large to attribute the result solely to
environment bugs. Keeping both names makes old run manifests reproducible and
prevents post-result configuration drift.

## Gate status

- Gate 0: **passed 2026-09-08**. Clean install, dependency check, local logger tests, secret-pattern scan, and both loader modes passed; revisions/licenses are recorded in `research/SOURCE_LEDGER.md`.
- Gate 1: **passed 2026-09-08**. Six default-environment regression tests pass in both the working and clean validation environments. The manual reset/step completes, and `results/phase1_default_baseline_zero_action.mp4` was rendered and visually inspected. The three pre-existing defects above were corrected narrowly; no G1 assumptions entered the old environment.
- Gate 2: **passed 2026-09-08**. The viewer and training share one resolver at Menagerie commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`; Hydra selects the isolated G1 config/environment; all 29 joint-actuator mappings and foot contacts are introspected and tested; the complete clean-environment suite reports 16 passed. Deterministic metadata was regenerated after separating floor-support and cross-contact foot geoms and is in ignored `results/gate2_g1_metadata.json` (SHA-256 `55ddfd2bd586ba7a73590d8701e971b5e0c6eb1a27703212665472dad94a532c`). No locomotion behavior or quality is claimed at this gate.
- Gate 3: **passed 2026-09-08 (CPU smoke)**. The clean suite reports 24 passed, including 1,000 bounded control steps with finite state/observations. Reset is seeded; action and motor targets are clipped; termination covers torso orientation, pelvis height, undesired contact, and NaNs. The exact standing pose and contact mapping were visually/programmatically verified. Diagnostic videos are under ignored `results/gate3/`. Negative result: the untrained `0.02`-amplitude sinusoidal controller first terminates at step 69 and is visibly fallen by 1.6 seconds; this is pipeline evidence only.
- Gate 4: **failed 2026-09-09**. Corrective matched seeds 0, 1, and 2 each
  completed exactly 20,971,520 steps; the interrupted first seed-2 process is
  retained and its exact from-scratch replacement completed normally. All 216
  frozen held-out episodes were finite, and trained linear RMSE/duration beat
  both controls, but trained yaw RMSE was worse, every controller fell, no
  full-horizon rollout survived, and zero policy seeds passed individually.
  The late audit also found the previous-action/command observation-timing
  defect in D-019. A clean exported source tree independently passed its full
  suite and produced a 1,024-step PPO checkpoint plus finite fixed-command
  evaluation, satisfying pipeline reproducibility but not behavioral quality.
  This is not a verified locomotion baseline.
- Gates 5–7: out of current authorization.
