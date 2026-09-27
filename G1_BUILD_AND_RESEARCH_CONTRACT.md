# Triton Droids Unitree G1 Locomotion Retrofit and Research Contract

## Current authority — Milestone 1 only (2026-09-27)

The user superseded open-ended three-seed research with **Milestone 1: package,
verify and report the existing simulation controller**, then STOP and pause the
repeating task. Read `research/MILESTONES.md` and `research/milestone_status.json`
first (paths relative to repository root). Milestones 2/3 are deferred and require
new user authorization. All previous results/code/checkpoints remain preserved;
see `research/MILESTONE_2_RESUME.md`. Historical "active" queues and instructions
to continue until three seeds pass below are archival, not current authorization.
No new training or automatic research continuation. R03 was intentionally stopped.


Version: 1.2, staged delivery edition
Prepared: 2026-09-05  
Target repository: current Triton Droids RoboCup Simulator / PyCharm project  
Intended executor: Codex or another autonomous software/research agent

## 1. Executive decision

Use the current repository as the host for the G1 research work. Do not replace it with a second standalone locomotion project and do not convert the existing 12-actuator `default_humanoid_legs` environment in place.

The repository already contains two useful foundations:

1. A pinned Unitree G1 MuJoCo Menagerie loader in `scripts/load_unitree_g1.py`, including an MJX-scene option and hidden model cache.
2. An existing Brax/MJX PPO locomotion stack under `source/` with robot, environment, reward, randomization, training, playback, and configuration infrastructure.

The correct retrofit is to connect those foundations with a G1-specific environment and a small amount of shared infrastructure. Preserve the existing environment as a regression baseline. Extract common abstractions only after G1 and the existing task are both covered by tests.

## 2. Long-term mission (Milestones 2/3; deferred)

Build a reproducible Unitree G1 locomotion research path inside this repository and determine whether selected techniques from HOMIE improve commanded-velocity tracking and stability.

The first useful result is deliberately small: a simulated G1 that accepts planar velocity commands `[v_x, v_y, yaw_rate]`, remains upright, and tracks those commands measurably better than an untrained policy and a standing-only controller.

More advanced HOMIE features are gated behind a verified standard G1 baseline.

## 3. Current repository facts to verify before editing

The uploaded project snapshot indicates the following. Re-verify these facts against the live repository before making changes:

- `scripts/load_unitree_g1.py` resolves a pinned MuJoCo Menagerie checkout at commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`.
- The loader supports `unitree_g1/scene.xml` and `unitree_g1/scene_mjx.xml`.
- Downloaded Menagerie files live under `.cache/mujoco_menagerie/` and are intentionally not committed or exported.
- The existing training environment is `source/locomotion/default_humanoid_legs/joystick.py`.
- The existing environment and robot metadata are built around 12 actuators and robot-specific names such as `LL_HR`, `LR_KFE`, `foot_left`, `foot_right`, `l_foot`, `r_foot`, `torso`, and `imu`.
- Observation sizes, joint-noise arrays, pose weights, feet bookkeeping, and several reward helpers contain 12-joint assumptions.
- `source/robots/robot.py` currently assumes an in-repository XML plus `mj_model.json` metadata.
- `source/scripts/train.py` already uses Brax PPO but currently assumes Weights & Biases logging is available.
- `requirements.txt` is not yet a validated lockfile.
- The README explicitly states that G1 loads in MuJoCo but is not yet integrated with the locomotion training environment.

Do not treat any of the above as permanent architecture. They describe the starting point.

## 4. Authority and interpretation

Use this priority order when instructions conflict:

1. Explicit instructions from the user in the current task.
2. This contract.
3. Repository-local `AGENTS.md`, if present and not conflicting.
4. Current official upstream documentation and source code.
5. Existing repository documentation and comments.

Record material ambiguities in `research/DECISIONS.md`. Choose the smallest reversible option that preserves research validity and the existing project.

## 5. Fixed defaults

Unless the user explicitly changes them, use:

- Robot: Unitree G1, target 29-DoF configuration without dexterous hands.
- Physics: MuJoCo with an MJX-compatible G1 scene.
- Training: simulation only.
- Task: planar commanded-velocity tracking.
- Command vector: `[v_x, v_y, yaw_rate]`.
- Controller: bounded joint-position targets or offsets around a verified G1 standing pose, using the actuator semantics supplied by the selected upstream G1 baseline.
- Algorithm: PPO.
- Initial terrain: flat ground.
- Primary environment reference: MuJoCo Playground G1 joystick environment.
- Primary robot model: the repository's already-pinned MuJoCo Menagerie G1 source, unless a documented compatibility reason requires changing the pin.
- Existing `default_humanoid_legs` implementation: preserved intact enough to remain a regression baseline.
- Physical-robot deployment: forbidden unless separately and explicitly authorized.

If the selected upstream G1 model exposes a different controlled DoF count than expected, verify the exact model variant and stop only if the choice would materially alter the research question.

## 6. Non-goals

Do not implement any of the following during the initial assignment:

- HOMIE exoskeleton, gloves, pedal, cameras, or cockpit.
- Real-robot control or network communication with a Unitree robot.
- Sim-to-real deployment.
- Dexterous-hand control.
- Object manipulation or imitation learning.
- A complete reimplementation of HOMIE.
- A broad rewrite of the current simulator.
- Replacing the existing project with MuJoCo Playground wholesale.
- Automated pushes, pull requests, remote server access, or destructive Git operations.

## 7. Safety and repository rules

Before changing code:

1. Inspect repository tree, Git status, current branch, recent history, and every applicable `AGENTS.md`.
2. Treat existing and uncommitted work as user-owned. Never discard it.
3. Never read, print, copy, commit, or upload `.env`, private keys, tokens, passwords, SSH material, or credential-bearing configuration.
4. Do not use `git reset --hard`, forced pushes, branch deletion, or broad file deletion.
5. Do not connect to Triton Droids remote infrastructure without explicit authorization in the current task.
6. Do not deploy to hardware.
7. Keep checkpoints, videos, logs, datasets, Menagerie caches, and large generated outputs outside Git unless the user explicitly requests otherwise.
8. Preserve upstream licenses and attribution for every copied or adapted file.
9. Do not make W&B, a cloud account, or a credential mandatory for smoke tests or local evaluation.
10. Before any long or expensive training run, prove the environment with deterministic smoke and short PPO integration tests.

If a potential credential is discovered, report only the affected filename and recommended remediation. Never reproduce the value.

## 8. Required sources

Read locally supplied papers first when available under `research/references/papers/`. Otherwise use canonical sources.

### Papers

1. HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit  
   https://arxiv.org/abs/2502.13013

2. MuJoCo Playground  
   https://arxiv.org/abs/2502.08844

### Code and models

- OpenHOMIE: https://github.com/InternRobotics/OpenHomie
- OpenHOMIE RL: https://github.com/InternRobotics/OpenHomie/tree/main/HomieRL
- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground
- MuJoCo Playground G1 locomotion: `mujoco_playground/_src/locomotion/g1/`
- MuJoCo Menagerie G1: https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1
- Unitree RL Mjlab: https://github.com/unitreerobotics/unitree_rl_mjlab
- Unitree MuJoCo: https://github.com/unitreerobotics/unitree_mujoco

Create `research/SOURCE_LEDGER.md` before implementing paper-derived or upstream-derived behavior. For every imported idea, record:

- Source title and URL.
- Paper version or Git commit SHA.
- Exact section, equation, page, file, class, or function.
- Whether the local implementation is exact, adapted, or only inspired.
- License and attribution requirements.

Do not claim to reproduce a HOMIE component unless its definition and implementation were verified from the paper or official code.

## 9. Retrofit architecture

Do not add a disconnected top-level `g1_locomotion/` package. Fit G1 into the repository's existing `source/` organization.

Preferred shape:

```text
source/
├── config/
│   ├── config.py
│   ├── envs.py
│   ├── robots.py
│   ├── sim.py
│   └── ...
├── locomotion/
│   ├── default_humanoid_legs/        # preserve existing baseline
│   └── unitree_g1/
│       ├── __init__.py
│       ├── base.py                    # only if a G1-specific base is useful
│       ├── joystick.py                # standard G1 velocity baseline
│       ├── constants.py               # verified names and mappings
│       ├── observations.py            # split out only if it improves testing
│       ├── symmetry.py                # Phase 7
│       └── curriculum.py              # HOMIE extension phases
├── robots/
│   ├── default_humanoid_legs/
│   ├── unitree_g1/
│   │   ├── __init__.py
│   │   ├── model.py                   # shared model resolver and pin
│   │   ├── metadata.py                # model-name/introspection helpers
│   │   └── README.md
│   └── robot.py
├── rewards/
│   └── ...                            # keep genuinely shared rewards here
└── scripts/
    ├── train.py                       # reuse when practical
    ├── play.py
    ├── evaluate.py                    # add if missing
    └── ...

scripts/
├── load_unitree_g1.py                 # keep beginner-facing entry point
├── export_repo_zip.py
└── create_codex_research_bundle.py

research/
├── DECISIONS.md
├── SOURCE_LEDGER.md
├── EXPERIMENT_PLAN.md
├── RESULTS.md
├── references/
│   ├── README.md
│   └── papers/
└── results/
```

Rules:

- Keep G1-specific names and arrays in `unitree_g1` code until they are proven general.
- Do not make the old 12-actuator environment artificially generic by replacing every `12` with `nu` and assuming that is sufficient.
- Share reward functions only when semantics and required sensors are genuinely identical.
- Add explicit robot-name maps rather than relying on copied positional indices.
- Keep the existing beginner `scripts/load_unitree_g1.py` workflow working after refactoring.

## 10. Single source of truth for G1 assets

The current loader already has a good pinned-cache design. Reuse it for training instead of introducing a second G1 asset source.

Required approach:

1. Move or extract the reusable Menagerie resolution logic from `scripts/load_unitree_g1.py` into a library module such as `source/robots/unitree_g1/model.py`.
2. Keep `scripts/load_unitree_g1.py` as a thin beginner-facing wrapper around that library.
3. Keep the existing Menagerie commit pin unless a documented incompatibility requires changing it.
4. Use `scene_mjx.xml` for the MJX training path.
5. Keep `.cache/mujoco_menagerie/` ignored and excluded from upload bundles.
6. Support an explicit local model override for offline development.
7. Record the exact model commit and selected scene in every experiment.

Do not vendor the full Menagerie G1 mesh tree merely to satisfy the current `Robot` wrapper unless there is a clear reproducibility or offline-distribution need. Prefer a clean G1 adapter or a carefully generalized `Robot` interface that can resolve an external pinned model directory.

The current training resume behavior copies a single robot XML into run output. That may be insufficient for an include-based G1 model with dependent assets. For G1, record the model source, commit, scene, and any local modifications, then reconstruct through the resolver. Do not pretend a copied top-level XML is a self-contained model if it is not.

## 11. Known integration hazards to address explicitly

Before training G1, inspect and test every assumption below:

- Fixed length-12 joint and action arrays.
- Fixed 12-element observation and privileged-observation contributions.
- Hard-coded joint groups and names.
- Hard-coded foot body/site/sensor names.
- Hard-coded torso and IMU names.
- Existing `mj_model.json` generation and staleness risk.
- Default-pose keyframe naming differences (`home`, `stand`, `knees_bent`).
- Joint ordering and actuator ordering.
- Position-actuator ranges and action scaling.
- Base-height thresholds appropriate to the old model but not G1.
- Reward terms that depend on sensors absent from the G1 Menagerie scene.
- Foot-contact semantics and undesired self-contact detection.
- Observation dimensions declared in config versus dimensions actually constructed.
- Noise vectors tied to old joint categories.
- Domain randomization assumptions tied to old body names or actuator arrays.
- Current reward aggregation and clipping behavior. Verify intended semantics against the selected baseline rather than silently inheriting them.
- Current push application and state replacement semantics. Add a test proving a scheduled push actually changes the intended state.
- W&B behavior. Local smoke and test workflows must run with logging disabled or offline.

Every one of these should have either a regression test, a G1-specific implementation, or an explicit written decision.

## 12. Upstream integration strategy

### Primary route

Use MuJoCo Playground's maintained G1 joystick environment as the behavioral reference while retaining this repository's own config, registry, training, and evaluation workflow where practical.

Port or adapt only the pieces required for a verified G1 velocity baseline. Pin the exact MuJoCo Playground revision used. Preserve license attribution. Keep a source ledger mapping upstream functions to local equivalents.

The goal is not to fork Playground wholesale. The goal is to use its G1-specific model mappings, observations, reward definitions, contact logic, randomization conventions, and tested defaults to avoid reinventing fragile humanoid details.

### Secondary route

If direct adaptation into the current Brax/MJX stack becomes materially more complex or less reliable than using Playground as a runtime dependency, pause at a documented gate and compare two options:

- thin local wrapper around a pinned Playground G1 environment, or
- continued native port into the current environment API.

Choose the route that minimizes duplicated physics/task logic while preserving experiment control and reproducibility.

### Fallback route

If the current MuJoCo Playground G1 environment cannot be integrated after a bounded documented attempt, use Unitree RL Mjlab as an isolated reference baseline. Do not mix Mjlab and Brax abstractions inside one environment.

### Prohibited route

Do not begin by porting all of OpenHOMIE from Isaac Gym. OpenHOMIE is a methodological and reward reference for later phases, not the initial runtime dependency.

## 13. Technical research phases and gates (preserved for Milestone 2/3)

These legacy research phases are not the delivery milestones. Milestone 1 packages an existing development policy without claiming these research gates passed. Do not begin a later research phase until its gate is satisfied and the user has authorized that milestone.

### Phase 0: audit and reproducibility record

Produce:

- Repository inventory and architecture note.
- Current Git state and baseline commit.
- Existing test result before modifications.
- Python, CUDA, JAX, MuJoCo, MJX/Brax, driver, GPU, and OS versions.
- Source ledger with pinned upstream revisions.
- A written framework decision.
- A clean installation procedure.
- A tested local/offline logging mode.
- A dependency lock or other reproducible version record after compatibility is validated.

Gate 0:

- No secrets are included.
- A clean environment installs from documented commands.
- G1 viewer/no-viewer loader still works.
- All external revisions and licenses are recorded.

### Phase 1: preserve and characterize the existing baseline

Run the smallest safe checks required to characterize `default_humanoid_legs`. Do not tune it.

Verify:

- Existing MJCF loads.
- Reset and step return finite arrays with expected dimensions.
- Command sampling returns `[v_x, v_y, yaw_rate]` inside configured bounds.
- Existing velocity-tracking rewards move in the correct direction under controlled tests.
- A push test verifies that a scheduled perturbation changes the state when push is enabled.
- Observation config dimensions match constructed observations.
- Any historical rollout or current short rollout can be inspected.

Gate 1:

- Findings are recorded in `research/DECISIONS.md`.
- Pre-existing failures are clearly separated from new failures.
- Existing behavior is protected by automated regression tests before shared refactors.

### Phase 2: create the G1 integration seam

Implement only the infrastructure needed to make G1 a first-class selectable robot/environment without yet claiming locomotion.

Required work:

- Shared G1 asset resolver extracted from the current loader.
- `unitree_g1` robot/config registration.
- `unitree_g1` environment registration.
- Verified G1 joint and actuator name maps produced from the loaded model.
- Clean handling of external/include-based model assets.
- Training path can select G1 without breaking the existing default robot.
- W&B/cloud logging is optional.

Gate 2:

- Existing default environment tests still pass.
- G1 model resolves from the exact pinned source.
- G1 joint/actuator names and counts are printed or serialized deterministically.
- No locomotion quality claim is made yet.

### Phase 3: G1 MJX smoke environment

Implement or adapt the G1 environment far enough to reset and step safely.

Required behavior:

- Load the MJX-compatible G1 scene and verified standing keyframe.
- Reset deterministically from a seeded distribution.
- Advance at least 1,000 random or bounded-action simulation steps without NaNs or shape errors.
- Apply bounded actions using verified actuator semantics.
- Detect falls using orientation plus height/contact criteria appropriate to G1.
- Expose named observation/action dimensions.
- Produce a standing and bounded-action diagnostic video.

Gate 3:

- Smoke tests pass on CPU where feasible and GPU where required.
- Standing pose, joint order, actuator order, and contact bodies are visually and programmatically verified.
- Action clipping and joint limits are tested.

### Phase 4: standard G1 velocity baseline

Implement a conventional flat-ground velocity task before adding HOMIE components.

Minimum policy observation content:

- Projected gravity or equivalent torso orientation representation.
- Base angular velocity.
- Commanded `[v_x, v_y, yaw_rate]`.
- Joint positions relative to the verified default pose.
- Joint velocities.
- Previous action.

Add linear velocity to policy observation only if justified by the selected research baseline; privileged observations may contain additional state for the critic. Document the choice.

Minimum reward family:

- Planar linear-velocity tracking.
- Yaw-rate tracking.
- Vertical base-velocity penalty.
- Roll/pitch angular-velocity penalty.
- Upright-orientation penalty.
- Torque or effort penalty.
- Mechanical-power or energy penalty.
- Action-rate penalty.
- Joint-acceleration penalty.
- Joint-limit penalty.
- Foot-slip penalty.
- Undesired-contact penalty.
- Termination penalty.

For every reward, add a unit test for sign and monotonic direction. For exponential tracking rewards, make error scale explicit.

Start with conservative command ranges supported by the selected upstream baseline. Record the exact distribution.

Gate 4:

- Trained policy outperforms an untrained policy and standing-only controller on fixed held-out velocity commands.
- Evaluation metrics are finite.
- A representative video is reviewable.
- Training and evaluation commands work from a clean checkout.
- At least one short PPO integration test produces a checkpoint before long training is attempted.

### Phase 5: HOMIE upper-body-pose curriculum

Add randomly commanded upper-body joint poses while lower-body locomotion tracks velocity and balance.

Requirements:

- Define exact upper-body and lower-body joint sets by verified names.
- Never rely only on positional indices copied from another model.
- Start with small pose deviations.
- Increase range through a documented curriculum tied to training progress.
- Keep a no-curriculum control with the same compute budget and command distribution.

Gate 5:

- Curriculum and control runs use matched seeds and training budget.
- Held-out upper-body pose stability is measured quantitatively.

### Phase 6: height tracking and squatting

Only after Phase 5 succeeds, implement commanded torso-height tracking and HOMIE knee-related behavior.

Requirements:

- Extract the precise HOMIE height and knee reward definitions from the paper/code.
- State every kinematic adaptation explicitly.
- Evaluate stand-to-squat, squat hold, squat-to-stand, and walking at supported heights.
- Treat unsafe self-collision, knee-ground contact, or obvious reward exploitation as failures.

Gate 6:

- Height error, transition success, fall rate, and undesired-contact rate are reported across seeds.

### Phase 7: symmetry utilization

Implement symmetry only after a verified left-right joint and observation mapping exists.

Required tests:

- Mirroring twice returns the original observation/action within tolerance.
- Mirrored lateral velocity and yaw command signs are correct.
- Every left joint maps to the intended right joint.
- A constructed symmetric output has near-zero symmetry loss.

Compare symmetry augmentation/loss against a matched no-symmetry control.

## 14. Research design

Create `research/EXPERIMENT_PLAN.md` and freeze it before full training.

Run separate experiment families:

1. Standard G1 velocity baseline.
2. Pose robustness: baseline versus upper-body-pose curriculum.
3. Height control: pose-capable policy with and without HOMIE height/knee rewards.
4. Symmetry: matched policies with and without symmetry utilization.
5. Full system: best verified components combined versus the standard baseline.

Fair-comparison requirements:

- Identical training steps, environment count, command distribution, and evaluation suite within each ablation.
- At least three seeds for exploratory conclusions; five or more when compute permits.
- Fixed held-out evaluation seeds and command trajectories.
- No hyperparameter selection on held-out evaluation data.
- Preserve failed and negative runs.
- Report wall-clock time, environment steps, and hardware.
- Separate exploratory findings from confirmatory claims.

Required metrics:

- Linear-velocity RMSE and absolute error.
- Yaw-rate RMSE and absolute error.
- Fall rate and mean episode duration.
- Command-conditioned success rate.
- Torso orientation error.
- Mechanical work or clearly defined energy proxy.
- Torque and action smoothness.
- Foot-slip distance/rate.
- Undesired-contact rate.
- Joint-limit violations.
- Gait left-right asymmetry.
- Recovery success after standardized perturbations.
- Phase 5: held-out upper-body pose success.
- Phase 6: torso-height RMSE and squat-transition success.
- Sample efficiency and wall-clock efficiency.

Report seed-level values or uncertainty. Do not show only the best checkpoint or best video without a fixed selection rule.

## 15. Domain randomization and perturbations

Do not add aggressive domain randomization before the nominal baseline works.

When enabled, configure and log ranges for:

- Ground friction.
- Link mass and center-of-mass offsets.
- Actuator strength and PD gains.
- Joint damping and armature where appropriate.
- Sensor and observation noise.
- Control latency if modeled.
- External pushes.

Use separate nominal and randomized evaluations. Domain randomization is not evidence of real-world safety.

## 16. Reproducibility requirements

Every meaningful experiment must save:

- Full resolved configuration.
- Random seed.
- Git commit and dirty-state indication.
- Menagerie commit and scene path.
- Playground/OpenHOMIE commit SHAs when used.
- Environment/package versions.
- Hardware description.
- Start/end timestamps and wall-clock duration.
- Training curves and per-reward metrics.
- Evaluation JSON/CSV.
- Checkpoint-selection rule.
- Final checkpoint and at least one diagnostic video when practical.

Provide a fast smoke profile and a full research profile. A successful smoke test proves only pipeline health, not locomotion quality.

## 17. Required automated tests

At minimum:

- Existing default-humanoid regression tests.
- G1 asset resolution and exact pinned revision.
- G1 expected joints/actuators and name maps.
- Seeded reset reproducibility.
- Command bounds and resampling.
- Finite reset and step outputs.
- Observation/action shape consistency.
- Reward sign and monotonicity.
- Termination thresholds and NaN termination.
- Action clipping and joint limits.
- Foot/contact-body identification.
- Push actually changes intended state when enabled.
- Config serialization.
- Logging-disabled local run.
- Short PPO integration run that produces a checkpoint.
- Symmetry involution/sign tests when Phase 7 is implemented.

Never hide failures with broad exception handling, disabled assertions, or unconditional skips.

## 18. Research acceptance criteria (Milestone 2, not Milestone 1)

Milestone 1 acceptance is defined by research/MILESTONES.md. The broader research baseline remains incomplete until the following and the original full three-seed validation criteria are met:

1. Clean documented installation without bundled virtual environments.
2. Existing G1 loader still works and training uses the same pinned model source.
3. Verified target G1 joint/actuator ordering.
4. Passing default-environment regression tests.
5. Passing G1 smoke, environment, reward, and reproducibility tests.
6. Trainable PPO velocity-tracking baseline.
7. Fixed evaluation suite comparing trained, untrained, and standing-only controls.
8. At least three seeded baseline training runs unless compute is genuinely unavailable and documented.
9. Machine-readable metrics, training plots, and representative videos.
10. Concise `research/RESULTS.md` stating what worked, failed, and remains uncertain.
11. Exact reproduction commands.
12. No credentials, virtual environments, machine-specific absolute paths, or unauthorized remote actions in the deliverable.

HOMIE extension phases are separate milestones. Do not label the standard velocity baseline a HOMIE reproduction.

## 19. Reporting format

`research/RESULTS.md` must begin with:

- Outcome in five sentences or fewer.
- Which gates passed.
- Best-supported conclusion.
- Largest unresolved limitation.
- Exact command to reproduce the principal result.

Then include:

- Environment and algorithm.
- Experiment table.
- Metrics with seed-level results.
- Reward/curriculum ablations.
- Failure analysis and suspected reward exploitation.
- Visual evidence links or paths.
- Deviations from this contract.
- Recommended next experiment.

Use cautious language. Simulation performance is not real-robot safety.

## 20. Final executor checklist

Before declaring completion, verify:

- [ ] No secret or private infrastructure material is included.
- [ ] Existing `default_humanoid_legs` behavior was not silently broken.
- [ ] Beginner G1 loader still works.
- [ ] Training and viewer paths resolve the same pinned G1 source.
- [ ] All source revisions and licenses are recorded.
- [ ] Clean-install and smoke commands were executed.
- [ ] Tests were run and exact results reported.
- [ ] Evaluation uses held-out fixed scenarios.
- [ ] Runs are comparable within each ablation.
- [ ] Failed and negative results remain visible.
- [ ] Videos correspond to identified checkpoints/configs.
- [ ] No real robot or remote club infrastructure was accessed.
- [ ] Final report separates facts, inferences, and proposals.

## 21. Stop conditions

Stop and ask the user rather than guessing if:

- The only available path requires credentials or access not already authorized.
- The proposed action would push to or rewrite a shared remote repository.
- Robot deployment becomes necessary.
- A license appears incompatible with intended use or redistribution.
- The G1 DoF/model-variant choice cannot be resolved from authoritative sources and would materially change the work.
- Required compute would create a meaningful cost not already authorized.
- A missing user decision would materially change the research conclusion.

Otherwise proceed autonomously only within the currently authorized milestone. Complete Milestone 1, write its report, pause the repeating task and stop; do not resume Milestones 2/3 without new user authorization.
