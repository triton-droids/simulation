# TRITON DROIDS — ROBOCUP TEST FIELD 01
## Implementation and Research Contract for Codex / Claude Code
**Revision:** 1.0 | **Date:** 2026-10-24 | **Status:** Implementation specification, not evidence of completed software

### Agent assignment
Implement an inspectable, reproducible soccer-field simulation package for Triton Droids' RoboCup Humanoid Soccer League **Large Robot Division**. The deliverable must load a robot into a RoboCup-style field, allow an operator to **see it move**, support **N parallel environments**, and provide **configurable, measured, reproducible domain randomization**. Build and test working software; do not stop at architecture notes. Do not claim training or sim-to-real validation without executed evidence.

**Constraints/assumptions:** Team expects access to UCSD HPC compute, subject to actual GPU availability and scheduling constraints. Agents may solve dependency/integration issues, but physical fidelity must still be calibrated. One *field layout* is shared by N *simulation instances*. A simulation instance is not a distinct field geometry; its initial conditions and parameters can differ.

### Priority directives
1. Work in a new branch (`feat/robocup-test-field-01`) and preserve existing code and historical results.
2. Make the robot model an explicit input. Inventory the repository and document candidate humanoid URDF/MJCF/USD + meshes, kinematics, sensors, actuator metadata, dimensions and license. The supplied `pupper-simulations.zip` snapshot contains Bittle quadruped model assets, **not a verified Large humanoid model**. NEVER relabel Bittle as a humanoid or claim a placeholder is the team's robot.
3. If the real humanoid model is unavailable, implement field + randomizer + controls with a clearly marked proxy demo robot; make `--robot real` fail loudly with precise missing-asset instructions. Finish the unblocked work.
4. Select one pinned, tested simulator stack. **Default: Isaac Lab v3.0.0** (verify release tags and current API). Prefer a fast headless backend for RL; use a compatible browser-capable visualizer (Viser or Rerun where supported), or Isaac Sim/Kit for detailed meshes and cameras. Document backend-specific features, including any unsupported randomization.
5. Support one-field *geometry* in v1. **M-Field 14 m long × 9 m wide** is the default. It is the former AdultSize geometry and is an allowed option for Large Division under the current *draft* HSL rules; event-specific rules can differ. Make all geometry numbers configuration-driven; ship an optional future `large_22x14` geometry profile, but do NOT train on multiple field geometries in v1.
6. Source rules from the HSL-Rules repository, not blindly from 2021–2022 virtual soccer. The legacy `hlvs_webots` repository is a **reference**, not a drop-in Isaac Lab scene importer. Record URLs, git SHAs, license and measurements of imported/reference assets. If importing Webots assets is impractical, procedurally author geometry and explain why.
7. Do not introduce a completely new learning algorithm or make a working trained striker a prerequisite. Physics demo, scripted controller, and testable training environment come first.

### Ground-truth references
- Current HSL rule sources: https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules and, specifically, `rules/environment.tex`.
- HSL field source file: https://raw.githubusercontent.com/RoboCup-HumanoidSoccerLeague/HSL-Rules/main/rules/environment.tex
- Legacy RoboCup virtual humanoid world: https://github.com/RoboCup-Humanoid-TC/hlvs_webots (`worlds/robocup.wbt`; legacy Webots 2022b, not a current Large Division rule authority).
- Isaac Lab 3.0: https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0 and https://docs.nvidia.com/isaaclab/isaac-lab/installation
- Isaac Lab current randomization guidance: https://github.com/isaac-sim/IsaacLab/blob/main/skills/user/domain-randomization-events/reference.md

### Field specification
Use SI units, right-handed coordinate system, Z up, center at (0,0,0), goals at ±length/2. The `medium_14x9` baseline should use rule-sourced geometry as follows:

| Property | Default M-Field | Notes |
|---|---:|---|
| Field length | 14.0 m | X axis |
| Field width | 9.0 m | Y axis |
| Goal width | 2.5 m | Rule range 2.4–2.6 m |
| Goal height | 1.7 m | Rule range 1.5–1.9 m |
| Goal depth | 1.0 m | Rule range 0.7–1.2 m |
| Goal area | 1.0 m × 4.0 m | Length into field × full width |
| Penalty area | 3.0 m × 6.0 m | Length into field × full width |
| Penalty mark | 2.0 m from goal line | Per draft |
| Center circle diameter | 3.0 m | Per draft |
| Corner arcs | Radius 0.5 m | Per draft |
| Boundary line width | 0.05 m | Ensure lines lie within marked areas correctly |
| Turf | Synthetic, 20–30 mm nominal | Visual proxy unless validated contact model exists |
| Ball | Size 5 | Source/verify exact mass, dimensions, and ball-contact model |
| Lighting | Mostly white, realistic venue conditions | HSL recommends most of pitch >=300 lux, preferably >=400 lux |

Implement physical goal frame, ball, boundaries and correct field markings. Texture and lighting may be simplified in headless mode; physics must be validated separately. The model may use a flat collision plane with an empirically calibrated contact model for turf; do not present rendered blades of grass as physically simulated turf. Verify geometries with unit tests and report actual measured dimensions.

### Deliverable A — Field loader
`robocub.soccer/field/`: parameterized scene definition, ball/goal assets, scene generator, field-profile YAML. Support `field_profile=medium_14x9`. Provide clean USD/procedural scene sources compatible with pinned physics backend; do not depend on a Webots GUI for runtime. Include a provenance manifest if imported assets are copied.

### Deliverable B — Robot loader and viewer
`robocub.soccer/robots/` and `robocub.soccer/viewer/`:
- Accept robot URDF, MJCF or USD with meshes, inertias, collision shapes, joint limits and actuator metadata. Validate scaling (meters vs millimeters), joint axes, COM, contacts, collision groups and foot-ground penetration. Preserve the identity of imported asset.
- `--robot real --robot_asset PATH`: load team's exact physical model; fail with actionable error if absent. `--robot demo` permitted only when conspicuously labeled **DEMO/PROXY, NOT TRITON DROIDS HARDWARE**.
- 1-environment interactive mode with orbit, overhead and robot-follow views; pause/resume/reset; select camera; ball/robot reposition to legal areas; current field/seed/randomization values display. Prefer browser-served Viser/Rerun when the HPC login architecture permits, and provide local native visualization as alternative.
- Expose a deterministic scripted stand/walk or joint-command demonstration, or explicit manual actuation controls. No fake learned policy. Capture one screenshot and one short MP4 or documented recording workflow. Rendering must be separate from the fast headless train loop.
- No public unauthenticated internet-facing streaming service. Document SSH-port forwarding or approved cluster access method, subject to site rules.

### Deliverable C — Parallel vectorized environments
`robocub.soccer/envs/` with configurable `--num_envs N`:
- Functional acceptance at N=1, 16, 256 on supported GPU; benchmark 1024 and 4096 only if resource limits permit. Record peak VRAM, wall-clock steps/s, simulator steps/s, effective FPS, host/device IDs, physics backend, scene replication constraints and failure modes. Do not promise that 4096 *full-sized* pitches fit one GPU.
- Every instance has separate robot state, ball state, scenario, RNG and reset counter. Field geometry remains identical. No cross-instance collision or ball leakage. If tiled, separate physical instances and camera views safely.
- Allow deterministic per-env resets and fixed evaluation seed suites independent from training RNG.
- Include a minimal Gym-like observation/action/reward/termination interface or the native Isaac Lab task abstraction, documented with shapes and units. A scripted baseline must be able to run without RL training.

### Deliverable D — Randomization engine
Implement `robocub.soccer/randomization/` and `App/Configs/randomization_field01.yaml`.

**Design rule:** Maintain a registry of about 18–20 tunable parameter *families*, but enable only **5–8 active sampled families per episode** in the intermediate curriculum, including always-on initial ball and robot poses. This is a design choice to aid attribution and curriculum learning, not a physical law. Always log which knobs were actually varied. Support independent baseline runs with all randomization disabled and stress tests with more families enabled.

**Parameter registry and starting ranges** (engineering hypotheses, not measured tolerances; calibrate to team's hardware):

| Family | Initial phase(s) | Nominal sample/range | Scope / timing |
|---|---|---|---|
| Robot initial X,Y | 1+ | legal spawn area, no overlap | per environment, reset |
| Robot initial yaw | 1+ | plausible orientation/task-conditioned | per environment, reset |
| Ball initial X,Y | 1+ | legal ball area, no overlap | per environment, reset |
| Ball initial speed/direction | 2+ | stationary or small/task-conditioned | per environment, reset |
| Foot/turf static friction | 2+ | calibrated reference × [0.90, 1.10] | per env if backend supports; init/reset as safe |
| Foot/turf dynamic friction | 2+ | calibrated correlated value; dynamic <= static | material creation/valid reset |
| Ball/turf rolling loss | 2+ | calibrated reference × [0.85, 1.15] | if implemented/verified |
| Ball restitution | 2+ | calibrated reference × [0.90, 1.10] | material/init/reset if supported |
| Ball mass | 3+ | calibrated reference × [0.98, 1.02] | as supported; nominal size-5 ball |
| Motor strength / torque scale | 2+ | × [0.95, 1.05] | per robot, reset if valid |
| Actuator gain | 2+ | × [0.95, 1.05] | per robot, reset if valid |
| Joint damping | 3+ | × [0.90, 1.10] | per robot, valid timing |
| Control latency | 2+ | 0–1 control steps initially | per env, task/controller |
| Proprioceptive observation noise | 2+ | bounded from sensor specs | per env, observation |
| Observation delay/dropout | 3+ | 0–1 steps or rare missing frames | per env, observation |
| Lighting intensity/exposure | 3+ | realistic mostly-white venue range | vision/render backend only |
| Camera pose jitter / image noise | 3+ | calibrated camera tolerance | camera pipeline only |
| Robot COM/inertial offset | 4+ | small, validated perturbations | init/reset if supported |
| External pushes | 4+ | task appropriate, non-destructive | interval event if supported |
| Opponent scenario/behavior | 4+ | scripted policy bank or defined set | per episode / soccer policy |

**Critical constraints:**
- Do **not** randomize gravitational acceleration for v1; real matches take place on Earth. Note that some PhysX gravity operations are scene-wide rather than independent by environment.
- The official field dimensions, line positions, goal dimensions, ball type, and match rules are **fixed during training v1**. Do not randomize them just to increase diversity.
- Physics parameters must stay physically plausible. Correlate related variables (e.g. dynamic/static friction), don't independently sample contradictory materials. Sample *consistent physical domains* rather than contradictory random numbers.
- On Isaac Lab 3.0, select correct lifecycle modes (`prestartup`, `startup`, `reset`, `interval`) for each property. Some PhysX material operations have initialization/CPU costs; Isaac Sim Replicator material/color functions can be global or require Kit. If an advertised per-env randomization cannot be supported by chosen backend, move it to process/run scope and document it; do not silently apply one change to all instances while logging per-env variation.
- `--seed 123` plus env ID, episode index, process/rank and configuration hash must reproduce an environment's randomized parameters (and, as much as backend permits, its trajectory). Define exactly what “deterministic” means for GPU execution.
- Store master seed, code commit, configuration hash, sampled-parameter summary/histogram, enabled families, backend, asset version and experiment ID. Logging can sample a subset of training episodes to avoid HPC I/O overload, but evaluations must permit full sample/seed replay.
- Split train, in-distribution validation and **held-out stress** suites with named configurations. Do not leak fixed evaluation seeds into training. Preserve nominal baseline cases to ensure performance on normal turf does not collapse.
- Use curriculum presets: `nominal` (only initial poses), `mild` (5–8 active), `robust` (roughly 8–12 active as validated), `stress` (held-out ranges for evaluation, never assumed realistic without evidence).

### Suggested distribution strategy
For `mild`, always vary robot pose (X,Y and yaw) and ball initial state (X,Y, with speed optionally fixed). Add **2–4 independently selected *families*** from dynamics/control and sensing groups on each reset, constrained by task and backend. Assign coherent environment-level values; sample additional small reset-only noise within each. E.g. one instance may have slightly slippery turf + 1 control-step lag; another may have higher ball rolling resistance + motor-gain deviation. Avoid globally turning on every disturbance on day one.

Use episode-level randomization in N simultaneous instances. It is not necessary to pre-generate an explicit list of N permanent ‘unique fields’; environments change after each reset. Allow replay via `(run_id, seed, env_id, episode_id)` manifest.

### User-visible CLI contract
The project must supply scripts with equivalent functionality and documented actual commands. Preferred wrappers:
```
./robocub/tools/field01 doctor
./robocub/tools/field01 view --robot real --robot_asset /path/to/humanoid.urdf --field medium_14x9 --seed 42
./robocub/tools/field01 view --robot demo --field medium_14x9 --seed 42
./robocub/tools/field01 smoke --num_envs 16 --randomization mild --seed 42
./robocub/tools/field01 train --num_envs 256 --randomization mild --seed 42
./robocub/tools/field01 evaluate --suite nominal --checkpoint PATH --seed 12345
./robocub/tools/field01 evaluate --suite heldout --checkpoint PATH --seed 54321
./robocub/tools/field01 report --run_dir PATH
```
Do not pretend commands work until tested; adapt invocation to Isaac Lab v3.0 APIs, Windows laptop vs Linux HPC requirements and scheduler conditions. `doctor` should validate models, NVIDIA driver/backend, dependencies, model geometry, and output directories, and emit a clear actionable report.

### Required source layout
```
robocub.soccer/
  field/{profiles,geometry,ball,goals,provenance}/
  robots/{importer,validator,manifest}/
  envs/{scene,task,spawning,observations,actions,rewards,terminations}/
  randomization/{registry,sampler,lifecycle,logging}/
  viewer/{app,controls,recording}/
  evaluation/{suites,metrics,replay}/
App/Configs/{field01.yaml,randomization_field01.yaml,robots.example.yaml}
scripts/{field01,benchmark_field01.sh,slurm/}
tests/{test_field_geometry.py,test_robot_asset.py,test_randomization.py,test_env_isolation.py,test_ball_physics.py}
Documentation/Guides/{QUICKSTART.md,FIELD_SOURCES.md,RANDOMIZATION_DESIGN.md,HPC_RUNBOOK.md,IMPLEMENTATION_STATUS.md}
```
Equivalent Python package layout is acceptable if clearly documented and not needlessly duplicative.

### Milestones and strict acceptance criteria
**M0 — Discovery and pinned baseline**
- Inventory repository; branch clean; list candidate robot files and licenses; record upstream commit SHAs, Isaac Lab version, Python/CUDA/backend and available GPU details. State exactly which robot can be displayed. Produce source manifest and `doctor` output.

**M1 — Field correctness**
- One `medium_14x9` scene loads. Verify length, width, goal dimensions, lines, center circle, penalty/goal areas and legal spawn regions numerically. Drop and roll size-5 ball; collision with ground and goalposts stable. Do not claim validated physics from visuals alone.

**M2 — Robot display and control**
- Viewer opens one environment locally or over approved SSH forwarding. Real robot model loads when available, or a conspicuously labeled proxy mode runs. Orbit + follow + overhead camera, pause, reset, reposition ball, small scripted non-learning motion, and screenshot/recording function. No hidden manual setup.

**M3 — Parallelism and isolation**
- N=1,16,256 tested on available compatible GPU. Independent env states, resets and random seeds; reproducibility checks. Benchmark 1024/4096 conditionally. Do not spend an entire HPC allocation debugging a graphical viewer.

**M4 — Domain randomization and auditability**
- `nominal`, `mild`, `robust`, `stress` presets; deterministic seed/manifest sampling, parameter constraints, range validation, tests for impossible combinations, and distribution sanity plots/histograms. Unit tests demonstrate independent sampled values across env IDs and repeatability for the same seed. Verify backend changes truly take effect in physics or mark them unavailable.

**M5 — Research-ready handoff**
- Scripted baseline trajectories; basic reward/observation interface; train/evaluate CLI smoke tests; held-out evaluation split; saved sample videos and metrics; clear `README`/runbook and cluster job template. One end-to-end automated smoke test must succeed on real hardware/GPU resources before declaring completion.

### Definition of done / no-fake-success policy
Deliver a clean diff, exact executed commands, exit statuses, hashes, screenshots/video evidence, tests (pass/fail), what remains blocked and why. Passing unit tests on CPU is not proof that GPU simulation works. A proxy robot is not the team humanoid. Visual similarity does not prove physical validity. Report real throughput rather than guessing it. If job scheduler or GPUs unavailable, fully test local/unit portions and clearly mark GPU tests as unexecuted.

### Requested handoff summary
Return in a final agent report: **(1)** files changed, **(2)** current robot asset and fidelity, **(3)** field dimensions/reference revision, **(4)** view command + proof, **(5)** num_envs measured + VRAM/throughput, **(6)** randomization family table with timing/physical constraints, **(7)** test evidence, **(8)** unresolved blockers, **(9)** next recommended milestone.

### One user-supplied input that genuinely matters
Provide the **actual Large humanoid model** (URDF/MJCF/USD package with mesh and joint/actuator data, or an exact hardware model with an authoritative downloadable simulation package) when it is available. Until then, deliver the complete field + viewer + randomization framework with a transparently labeled demo robot.
