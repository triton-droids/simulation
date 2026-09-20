# Decisions and Gate Log

## D-037 -- Stop C12 unchanged despite full survival

C12 survives all eight development commands and retains sustained forward
gait, but yaw improves only 12.91% versus its initial policy. This misses the
frozen 20% extension prerequisite. Preserve the positive translation and
survival result alongside failed turning/stand yaw. Diagnose same-physics
turning against the oracle before changing reward or command curriculum;
no unchanged 4,014,080-step extension is permitted.

## D-036 -- Verify forward gait, then normalize newly introduced commands

C11 completes the declared forward-gait gate, including .26/.14 s completed
air medians in the fixed-phase diagnostic. Broader commands are now eligible
for a bounded curriculum. Existing forward-only normalizers have std 1e-6
on lateral/yaw inputs; blindly introducing .1 commands produces normalized
magnitude 100,000. The explicit prior preparation changes only these unused
zero-mean channels and their Welford variance, preserving the policy/value
function on the old vy=yaw=0 subspace. Six helper/checkpoint tests pass;
actual-checkpoint inference parity and exact reload are required next.
C12 also restores the original linear-to-angular tracking reward ratio while
introducing the full command grid. Its conditional extension is frozen in
advance; it is not a claimed isolated reward ablation or final held-out test.

## D-035 -- Preserve C10's numerical pass; require convincing support timing

C10 clears the frozen numerical forward checks on two reset seeds and the
fixed-phase survival check. Its visual gait remains irregular, with median
air intervals .06 s versus oracle .20 s. Retain that progress without calling
Gate 4 passed or broadening commands. C11 is a bounded local contact-phase
reward ablation, justified by saved-trace counterfactual scoring and tested
invariants. This optional term is not upstream reproduction. It uses old
phase/command with next-state contact, is disabled by default (including old
configs), and adds a stable reset/step metric only when enabled. A first
floating-point cancellation test exposed tiny double-support residuals; an
explicit single-support mask now makes stance/flight credit exactly zero.
The .12-second completed-air-interval requirement is prospective for C11,
not a retroactive reclassification of C10's numerical gate.

## D-034 -- Reject C08 and isolate forward gait acquisition from balance

C08's final checkpoint fails the frozen gait gate (71 steps, .598791 linear
RMSE, 8.45% single support, hopping/collapse video). Training reward gains and
fall-related contact transitions do not count as gait. Preserve the complete
run and controls; do not scale it unchanged. C09 changes initialization to
the locally trained C06 balance checkpoint, stays forward-only, and has a
separate 1,003,520-step budget and unchanged behavioral gate. It tests staged
learning, not omnidirectional warm-start or oracle imitation. All restored
normalizer/policy/value parameters and reset optimizer/PRNG semantics must be
recorded. Its checkpoint-0 control is `initial`, not `untrained`; the evaluator
now supports that explicit label while preserving the default frozen grid
and untrained-control behavior. See the C09 predeclaration before execution.

## D-033 -- Preserve transition bookkeeping across Playground full reset

The C07 successor audit on 2026-09-15 reproduced a concrete wrapper bug with
zero PPO steps. Pinned Playground's `BraxAutoResetWrapper(full_reset=True)`
resets `truncation`, `episode_done`, and `episode_metrics` on done transitions,
preserving only `steps`. A synthetic two-environment test failed separately
for physical termination (`episode_done=0`) and timeout (`truncation=0`).
Brax 0.14.2 PPO consumes these exact fields: its GAE uses `truncation`, and
its metrics aggregator uses the other two. Thus timeouts were treated as
physical terminals and training episode reports could disappear. The outer
EvalWrapper's reward/length metrics remain independent and valid.

A small local bridge now uses upstream's supported
`AutoResetWrapper_preserve_info` hook to carry the four EpisodeWrapper fields
through reset, while simulator/command/action/phase/contact history still
resets coherently. Both forced-terminal cases pass after the change, including
the following episode's accumulators. This changes future training semantics;
it does not alter bare-environment checkpoint evaluation or establish that the
bug caused C06's static stance. Preserve all previous experiments.

Saved C06 checkpoint 5007360 additionally passed 64-observation inference
parity: maximum training/evaluator action difference exactly 0; disabling
normalization changed actions by up to 1.9926244. Artifact:
`results/gate4_corrective/C07_audit/action_parity.json`.

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
normalization disabled and no gradient clipping. Those runs are invalid for the
declared normalized profile but retained; their identity evaluator is faithful
to the actual unintended policy function. A corrected 1,024-step CUDA smoke
checkpointed, and a corrected
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

**Evidence at the time:** The fixed evaluator reported 100% falls and yaw RMSE
`1.971`; D-030 later invalidated that PPO inference. Its valid logged alive contribution was
about 5.2 times the two trackers combined after timestep integration. This
made increasing survival with command-insensitive high-rate motion more
valuable than tracking. The corrective profile keeps every contract-required
regularizer signed and nonzero and is frozen in `research/EXPERIMENT_PLAN.md`.

## D-016 -- Bound compiled epoch size with evaluation cadence

**Decision:** Use 21 evaluations for the 20,971,520-step matched family so
each Brax epoch contains 16 transition batches. Preserve the five-evaluation
attempt as a documented runtime failure; a freeze audit found no distinct run
directory despite earlier prose saying one had been retained.

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

## D-018 -- Historical Gate 4 failure decision (superseded by D-030)

**Decision at the time:** Record the authorized Gate 4 attempt as a failure and stop before
HOMIE Phases 5-7. Do not promote a visually preferred checkpoint or weaken the
predeclared rule after seeing reset seeds 2000-2002.

**Historical evidence:** Across 216 fixed held-out episodes, trained linear-vector RMSE
was `0.9254` versus `0.9878` untrained and `0.9942` standing, and mean duration
was `1.3742 s` versus `1.1364/1.1333 s`. However trained yaw RMSE was `1.0575`
versus `0.4539/0.3238`, every controller had fall rate `1.0`, no rollout
survived the 500-step horizon, and zero of three trained policy seeds passed
the matched rule. All values were finite. The exact aggregate is retained at
`results/gate4/final_aggregate/summary.json`.

**Superseding correction:** D-030 establishes that the trained PPO network in
this evaluator omitted observation normalization. The raw files stay intact,
but the trained-policy metrics above no longer support a faithful failure
verdict. Gate 4 remains not passed because no valid three-seed held-out result
exists—not because D-018's numbers remain accepted.

## D-019 -- Preserve, then correct, the discovered observation-timing defect

**Decision:** Do not change actor-observation timing underneath the completed
checkpoint family. Add strict expected-failure tests for the correct invariants,
document the defect as a Gate 4 limitation, and require a fix before any new
baseline family.

**Evidence at discovery:** `Joystick.step()` called `_get_obs` before shifting
`last_act`, advancing phase, and resampling the command. Thus the returned
observation's "previous action" is one control step older than the just-applied
action, and at a resampling boundary its command differs from the returned
`info["command"]`. Moving observation construction would change every future
policy input and make the already-trained checkpoints incomparable. The frozen
family already fails on yaw and survival, so no additional training is justified
under the current semantics.

**Completed follow-up:** Commit `4426499` fixed these invariants for a new,
explicitly non-comparable corrective family and converted the strict timing
checks into passing regressions. Historical checkpoints remain tied to the old
semantics.

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

## D-025 -- Normalize cross-platform Git provenance and persist run manifests

**Decision:** Query Git with `core.autocrlf=true` for run provenance, timestamp
every local JSONL event in UTC, and write a per-run manifest containing the
exact command, Git state, Python/package versions, platform, JAX backend and
devices, GPU/driver/memory, and relevant runtime environment variables.

**Evidence:** Two otherwise identical clean-checkout deterministic scans were
recorded as dirty because WSL Git interpreted the Windows CRLF worktree as 35
whole-file modifications, while Windows Git correctly reported clean and
`git diff --ignore-space-at-eol` was empty. `git -c core.autocrlf=true status`
is clean on the same checkout and still reports semantic edits. Prior run
configurations could reconstruct versions from later evaluation, but did not
carry a complete timestamped per-run hardware/source manifest; new training
runs must do so before any learning evidence is accepted.

## D-026 -- Require a bounded corrected-learning diagnostic before scaling

**Decision:** Treat C00/C01 as pipeline checks only, then run one noise-free,
nominal-reset C02 diagnostic with the Playground-aligned PPO optimizer/network
profile. The requested 262,144 steps resolve to 286,720 actual environment
steps under Brax epoch rounding (`4` post-initial evaluation epochs, `7`
training iterations per epoch, and `10,240` environment steps per iteration).
Do not increase the budget unless C02 is finite, has controlled KL, improves
episode length/tracking, and shows plausible support transitions in a
fixed-command rollout. Record exact commands and elapsed time in all future
evaluation summaries as well as training manifests.

**Evidence:** C01 completed its exact 1,024-step checkpoint path on CUDA from a
clean commit, but its final KL was `3.3128`. D-030 invalidates the old
held-command tracking values. C01 remains checkpoint/provenance plumbing only,
which was sufficient reason not to treat it as learning evidence.

## D-027 -- Escalate to a thin pinned Playground runtime adapter

**Decision:** Do not scale the corrected native configuration after C02. Add a
separately registered `unitree_g1_playground` environment that loads only the
exact G1/mjx/wrapper modules from Playground commit `8a4b464...`, points them
at the existing Menagerie resolver checkout, and uses Playground's feet-only
MJX model, Euler/PD dynamics, reward scales, observation layout, contact
sensors, and Brax training wrapper. No robot XML, mesh, or texture is copied
into tracked source. Keep the native `unitree_g1` implementation and every
failed checkpoint intact.

Apply only narrow correctness adaptations around the authoritative task:
functional `info`/`metrics` dictionaries; normalized-action clipping;
qpos/qvel finite termination; synchronized returned command, prior-action,
phase, and critic air-time slices; coherent full-state auto-reset metadata; and
an optional nominal-reset diagnostic mode. The
reward for a transition remains the upstream reward computed before next-state
history changes. Resolve and record both upstream revisions independently.

**Evidence:** C02 was finite and its optimizer stabilized, but after 286,720
steps its valid training-side mean length improved only `72.0 -> 75.41` and
termination remained present. D-030 later invalidated its fixed-command
tracking/video evidence. The small signal plus the large documented dynamics
differences in D-022 still supported a reversible authoritative adapter rather
than another expensive native run. The adapter's pre-training tests load the official
`36/35/29`, 72-geom, 5-pair, 29-sensor model; a jitted CUDA reset and step are
finite and return observations consistent with next-state metadata. The
Playground overlay consumes the already-pinned Menagerie asset directory; mesh
files are byte-identical across Playground's expected Menagerie revision and
the repository pin, as recorded in D-022.

## D-028 -- Require coherent auto-reset and a bounded authoritative diagnostic

**Decision:** Bind Playground's supported Brax wrapper with `full_reset=True`.
Its default fast path restores cached simulator data and observations after a
terminal transition but deliberately preserves the prior episode's mutable
`info`; that makes the reset command, phase, action/contact history, and reward
inputs disagree. A forced-terminal batched wrapper test now proves that data,
observation slices, and returned history reset together.

After the 1,024-step C03 adapter smoke passes checkpoint restoration, run C04
for 262,144 requested (`286,720` actual) steps with the full corrective PPO
network/optimizer, nominal reset, and noise/push/randomization disabled. Retain
the official command ranges and reward/dynamics semantics. Apply the explicit
finite/KL/length/tracking/gait scale-up rule recorded in `research/RESULTS.md`;
C04 remains a diagnostic, not a Gate 4 PASS attempt.

**Evidence:** C03 completed from clean commit `0379401`, wrote checkpoints 0
and 1,024 plus exact dual-source/runtime provenance, and structurally restored
both policies through 24 finite held-command rollouts. Its final training KL
was `3.2936`. D-030 later invalidated the held-rollout behavior because
normalized preprocessing was missing; the run remains sufficient checkpoint
plumbing evidence but not gait evidence.

## D-029 -- Use the shipped policy as an oracle, then stage gait acquisition

**Decision:** Do not infer a dynamics/reward defect from a 0.287-million-step
run when the pinned upstream profile trains for 200 million steps. Before using
more PPO compute, execute the exact `experimental/sim2sim/onnx/g1_policy.onnx`
blob from the same Playground commit against the adapter's compiled model and
fixed command suite. Use it only as a behavioral oracle, not as a claimed local
training seed or final policy.

Because the oracle succeeds while C04 fails, change the next experiment rather
than scaling C04 unchanged: C06 first learns a bounded forward gait with
`vx=[0.2, 0.6]`, zero lateral/yaw commands, nominal reset, and a five-million-
step budget. It retains the authoritative reward/dynamics and full PPO network.
Only a quantitatively and visually verified alternating forward gait can be
warm-started into an omnidirectional command stage.

**Evidence:** C04 remained finite and KL stabilized near `0.038`, but its valid
training-side final episode length regressed to `61.75` steps. Its old
fixed-command policy behavior was later invalidated by D-030. C05b's exact
shipped policy completed all eight 500-step CPU-MuJoCo rollouts with no fall,
minimum pelvis height above `0.692 m`, and 74–83% single support. Forward and
yaw response were sign-correct, and video inspection confirms alternating
steps. A diff between the Playground-expected and repository-pinned Menagerie
G1 XML shows only formatting/statistic changes in the included dynamics files,
so a hidden joint/gain/contact revision mismatch is not a supported cause.

## D-030 -- Invalidate pre-f642 evaluations of normalized PPO checkpoints

**Decision:** Preserve all prior JSON, CSV, plots, and videos, but stop using
pre-`f642bc2` fixed behavior from checkpoints trained with normalization enabled
as scientific evidence. Gate 4 is now “not passed / held-out verdict
invalidated,” not a verified PASS and not a faithful three-seed failure.
Training-side Brax evaluation metrics and the independent C05b ONNX oracle
remain valid.

**Evidence:** `evaluate_g1.py` loaded the checkpoint tuple containing running
observation statistics but constructed `make_ppo_networks` with its identity
preprocessor. Brax training constructs the same network with
`brax.training.acme.running_statistics.normalize` whenever
`normalize_observations=true`. Commit
`f642bc2fbdbc575dc3d4865cdd2fc72a941029e2` restores that callback and adds
positive and negative regression tests. The historical normalized 216-episode
family, corrected/capacity/reference-scaled native evaluations, clean-export
smoke, C01-C04, and C06's first video evaluation used the wrong policy
function. Standing traces are not normalized, but trained-versus-control
claims from those summaries are invalid.

The earlier pre-D-012 pilots are a distinct case. Because `train.py` had not
yet forwarded the serialized flag, Brax actually trained them with its
`normalize_observations=False` default. Their identity-preprocessed evaluator
was faithful to that actual policy function, although the runs remain excluded
as unintended, misconfigured profiles. This distinction is protected by the
evaluator's enabled/disabled normalization tests.

## D-031 -- C06 learned balance, not forward gait

**Decision:** Mark C06 as a legitimate negative gait-acquisition result. Do not
warm-start it into an omnidirectional stage and do not rerun the same
configuration unchanged. Retain the final checkpoint because it is the first
locally trained policy validly shown to hold an upright stance for 10 seconds.

**Evidence:** C06 ran from clean commit `da70eda` for 5,007,360 actual steps
with the exact adapter/rewards, `vx=[0.2,0.6]`, y/yaw zero, nominal reset, and
the authoritative PPO optimizer/network. Training evaluation rose from length
69/reward `-1.6033` to length 500/reward `15.8929`, with final KL `.0946`.
After D-030, fixed `vx=0.5` evaluation on reset seed 2000 also survived 500/500
and stayed above `.756 m`, but vector RMSE was `.4999 m/s`, double support was
100%, single support 0%, and both feet made zero contact transitions. It
therefore fails the predeclared transition, support, tracking-success, and
visual-gait requirements even though it numerically beats controls that fall.
The six out-of-distribution lateral/yaw/backward/combined commands terminate in
3-116 steps. The corrected result is
`results/gate4_corrective/C06_playground_forward_curriculum_seed0_5000000/evaluation/checkpoint_5007360_normalized_seed2000_500/`.

## D-032 -- Freeze at C07 diagnostics before more PPO

**Decision:** The first successor action is C07, a zero-training-step
inference/invariant audit. It must (1) compare saved-checkpoint actions between
Brax training-time and evaluator network construction, (2) extend autoreset
coverage to the EpisodeWrapper bookkeeping keys, (3) run the corrected C06
eight-command grid on seeds 2000-2002, add a tested video-command selector, and
produce one corrected forward video while treating only stand/forward as
in-distribution, and (4) correctly reevaluate C02 and C04.
C06 used nominal reset, so those seeds are not randomized-pose robustness. The
old full native family may be evaluated only in a temporary worktree at
`27de436` (the closest recoverable historical-semantics snapshot, not a
per-run-proven exact commit) while backporting D-030.

**Reason:** This recovers information from already-spent compute and protects
against a second invalid report. The current `full_reset=True` path is not a
confirmed cause of C06—Brax's outer EvalWrapper preserves the done/step metrics
used for training evaluation—but its dedicated regression currently asserts
environment history rather than every EpisodeWrapper field. After C07,
quantify the C06 static reward optimum against C05b before freezing one bounded
seed-0 gait experiment. No three-seed or large-budget run is justified until
alternating support is both measured and visible.

## Gate status

- Gate 0: **passed 2026-09-08**. Clean install, dependency check, local logger tests, secret-pattern scan, and both loader modes passed; revisions/licenses are recorded in `research/SOURCE_LEDGER.md`.
- Gate 1: **passed 2026-09-08**. Six default-environment regression tests pass in both the working and clean validation environments. The manual reset/step completes, and `results/phase1_default_baseline_zero_action.mp4` was rendered and visually inspected. The three pre-existing defects above were corrected narrowly; no G1 assumptions entered the old environment.
- Gate 2: **passed 2026-09-08**. The viewer and training share one resolver at Menagerie commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`; Hydra selects the isolated G1 config/environment; all 29 joint-actuator mappings and foot contacts are introspected and tested; the complete clean-environment suite reports 16 passed. Deterministic metadata was regenerated after separating floor-support and cross-contact foot geoms and is in ignored `results/gate2_g1_metadata.json` (SHA-256 `55ddfd2bd586ba7a73590d8701e971b5e0c6eb1a27703212665472dad94a532c`). No locomotion behavior or quality is claimed at this gate.
- Gate 3: **passed 2026-09-08 (CPU smoke)**. The clean suite reports 24 passed, including 1,000 bounded control steps with finite state/observations. Reset is seeded; action and motor targets are clipped; termination covers torso orientation, pelvis height, undesired contact, and NaNs. The exact standing pose and contact mapping were visually/programmatically verified. Diagnostic videos are under ignored `results/gate3/`. Negative result: the untrained `0.02`-amplitude sinusoidal controller first terminates at step 69 and is visibly fallen by 1.6 seconds; this is pipeline evidence only.
- Gate 4: **not passed; historical held-out verdict invalidated 2026-09-11 UTC
  (2026-09-10 PDT)**.
  Three 20,971,520-step native runs completed, but D-030 invalidates their PPO
  held-out policy inference. C05b independently proves the model/interface can
  walk. C06 validly proves a local policy can balance for 10 seconds, but it
  remains stationary at `vx=0.5`, has zero foot transitions, and fails its gait
  gate. No conventional three-axis, three-seed locomotion baseline exists.
- Gates 5–7: out of current authorization.
