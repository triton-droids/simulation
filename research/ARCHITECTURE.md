# Repository Architecture and Phase 0 Audit

Current-state review: 2026-09-15 at code/research base
`f642bc2fbdbc575dc3d4865cdd2fc72a941029e2`. The dated Phase 0 observations
below are retained as historical baseline evidence; this section describes the
architecture a successor will actually inherit.

## Current architecture

The original `default_humanoid_legs` task remains separate and protected by
regression tests. G1 now has two selectable implementations sharing one robot
resolver and the same local-first PPO/checkpoint/provenance surfaces:

1. `source/locomotion/unitree_g1/joystick.py` is the corrected native Brax
   `PipelineEnv`. It uses the pinned full-collision Menagerie MJX scene and
   model-introspected joints, actuators, sites, contacts, ranges, and keyframe.
2. `source/locomotion/unitree_g1/playground_joystick.py` is the current primary
   baseline adapter. `playground_source.py` loads only the exact G1/MJX/wrapper
   modules from Playground commit `8a4b464...`, overlays its feet-only scene,
   and redirects model assets to the existing Menagerie checkout at
   `71f066a...`. It keeps local config/registry interfaces and repairs functional
   next-observation/history semantics without vendoring upstream source/assets.

Both expose 29 actions, actor observation 103, and privileged observation 216.
`source/scripts/train.py` runs Brax PPO with local JSONL, exact manifests, source
records, and restorable compact/full checkpoints. `evaluate_g1.py` supports
both backends and is the validated scientific playback path. Since `f642bc2`,
it reconstructs normalized PPO networks with the same running-statistics
preprocessor used in training. Older evaluator outputs for checkpoints trained
with normalization enabled (the D-012 runtime-control family onward) lack that
preprocessor and are invalid trained-policy behavior evidence. Earlier pilots
actually trained with Brax's normalization-disabled default; their identity
evaluations are faithful only to those unintended profiles.

`source/scripts/evaluate_playground_onnx.py` runs the exact shipped upstream
103-to-29 ONNX policy against the adapter-compiled CPU MuJoCo model. C05b proves
this physical/control architecture can walk. The current local C06 checkpoint
balances but remains in static double support; gait acquisition, not basic
model construction, is the open problem.

At the 2026-09-15 freeze, WSL compileall passed and the full repository suite
reported `93 passed, 25 warnings in 616.44s` with no skips, xfails, or failures.

Audit date: 2026-09-08  
Baseline Git commit: `6640663e5b9a50f25264e392a4c18703ca2c00e7` (`Robocup`, one local commit ahead of `origin/Robocup`)  
Initial worktree state: the contract, handoff README, and two local papers were staged additions; no source files were modified. A later untracked `outputs/` directory was produced by the documented pre-change Hydra smoke test and is intentionally retained as negative-result evidence.

## Architecture found

The repository has two useful but disconnected paths:

- `scripts/load_unitree_g1.py` is a beginner-facing MuJoCo loader. It resolves the Unitree G1 model from a sparse, hidden MuJoCo Menagerie checkout pinned to `71f066ad0be9cd271f7ed58c030243ef157af9f4`, and can select `scene.xml` or `scene_mjx.xml`.
- `source/` is an advanced Hydra + Brax/MJX locomotion stack. It contains one registered `PipelineEnv`, `source/locomotion/default_humanoid_legs`, one in-tree JSON-backed robot wrapper, a reward registry, domain randomization, PPO training/playback, and rollout rendering.

The current environment is robot-specific rather than a reusable 12-DoF template. Its joint names, foot sensors, pose weights, observation construction, noise arrays, and contact logic are all tied to `default_humanoid_legs`. The G1 retrofit therefore uses a separate `source/locomotion/unitree_g1` environment and a G1 model adapter; it does not replace literal `12` values with `nu` in the old task.

## Suitability decision

The project is suitable for a small, reversible retrofit. Brax PPO already accepts dictionary observations for an asymmetric actor/critic, and `brax.io.mjcf.load` successfully loads the pinned Menagerie `unitree_g1/scene_mjx.xml`. The existing `PipelineEnv` API will be retained for the Phase 4 baseline because this minimizes changes to training, evaluation, and checkpoints. This choice is deliberately local to the exploratory milestone: Brax warns that its pipelines are no longer actively maintained, so a later project-wide direction should compare a thin MuJoCo Playground runtime wrapper before expanding beyond Gate 4.

## Verified model facts

The pinned MJX scene loads with `nq=36`, `nv=35`, `nu=29`, `nbody=31`, `ngeom=63`, `nmesh=35`, `nkey=2`, timestep `0.004`, and 49 explicit contact pairs. `mjx.put_model` succeeds with the JAX implementation. The 29 position actuators map one-to-one, by name, to 29 scalar joints: 12 legs, 3 waist joints, 7 left-arm joints, and 7 right-arm joints. Training will use the `knees_bent` keyframe and position targets of the form `q_default + 0.5 * clip(action, -1, 1)`, followed by actuator-range clipping.

The foot sites are `left_foot` and `right_foot`, attached respectively to `left_ankle_roll_link` and `right_ankle_roll_link`. Ground contact uses three capsule collision geoms per foot (`left_foot1_collision` through `left_foot3_collision`, with right-side equivalents). Separate `left_foot_box_collision` and `right_foot_box_collision` geoms participate in the scene's explicit cross-foot and foot/opposite-shin pairs; they are not floor-support geoms. The torso body is `torso_link`; the model supplies pelvis and torso IMU sites and 14 velocity, accelerometer, gyro, up-vector, and orientation sensors. It does not supply MuJoCo Playground's custom foot-contact/foot-force sensors, so G1 contact state is derived from the actual explicit contacts.

## Pre-change baseline results

Commands were run with `.venv/Scripts/python.exe` from the repository root.

| Check | Result before source changes |
|---|---|
| `python -m compileall -q source scripts` | Pass |
| `python scripts/load_unitree_g1.py --no-viewer` | Pass: `nq=36`, `nv=35`, `nu=29`, `ngeom=72`, `nmesh=35`, `nkey=1` |
| `python scripts/load_unitree_g1.py --mjx-scene --no-viewer` | Pass: `nq=36`, `nv=35`, `nu=29`, `ngeom=63`, `nmesh=35`, `nkey=2` |
| `python -m pytest` | Fail before collection: `pytest` was not installed or declared |
| `python source/locomotion/test_joystick.py` | Fail: direct execution could not import `source` |
| `python -m source.locomotion.test_joystick` | Reset and step completed, then diagnostic printing failed because Brax `State` has no `q` attribute; positions live in `state.pipeline_state.q`/`qpos` |
| deterministic default-env reset and one zero-action step | Finite and reproducible; actor observation is 52 as configured |
| default privileged observation check | Pre-existing failure: constructed size 112, configured size 88 |
| default scheduled-push inspection | Pre-existing failure: `tree_replace` return value is discarded, so the computed velocity update is not applied |

The manual Hydra run created `outputs/2026-09-07/22-38-47/`; it is preserved to keep the negative result visible.

## Validated host environment

| Component | Version / value |
|---|---|
| OS | Windows 11 `10.0.26200` |
| Python | 3.12.7, 64-bit (Anaconda build) |
| MuJoCo | 3.10.0 |
| JAX / jaxlib | 0.11.0 / 0.11.0 |
| Brax | 0.14.2 |
| Hydra / OmegaConf | 1.3.4 / 2.3.1 |
| Flax / Orbax | 0.12.8 / 0.12.1 |
| NVIDIA driver | 572.61 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB |
| Native Windows JAX device | CPU only |
| WSL | WSL2 Ubuntu, Linux `6.6.87.2-microsoft-standard-WSL2-x86_64`; Python 3.12.3; isolated `.cache/g1_wsl_venv`; JAX reports `cuda:0` |
| Media / oracle extras | MediaPy 1.2.7, imageio-ffmpeg 0.6.0; optional ONNX Runtime 1.22.1 in WSL |

CUDA-capable PPO training uses the isolated WSL2 environment after both CPU and CUDA checkpoint smoke tests passed. No club host, remote branch, cloud runner, or physical robot is in scope.

## Installation profiles

CPU validation (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\load_unitree_g1.py --mjx-scene --no-viewer
```

GPU training (WSL2 Ubuntu, validated isolated environment):

```bash
python3.12 -m venv .cache/g1_wsl_venv
. .cache/g1_wsl_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade "jax[cuda12]==0.11.0"
python -c "import jax; print(jax.devices())"
```

Basic testing and training default to local logging and do not require W&B credentials.

## Gate 0 verification

A new ignored environment at `.cache/gate0_clean_venv` was created from the documented Python 3.12 commands and `requirements.txt`. Installation completed successfully in 94 seconds. In that environment, `pip check` reported no broken requirements, compileall passed, both credential-free logger tests passed, and both G1 loader modes worked with `--no-fetch-model`. The exact Windows CPU transitive dependency snapshot is preserved in `requirements-lock-windows.txt`.

A filename-only scan found no credential-bearing filenames in deliverable paths, and a high-confidence content-pattern scan found no private-key, common access-token, or assigned-secret patterns. Cache, virtual-environment, generated-output, binary paper, image, archive, and video paths were excluded from the content scan. No credential values were read or printed.

## Current Gate 4 architecture note

The bounded native seam and the exact Playground adapter are mechanically
reproducible, but no local conventional walking policy has passed Gate 4. The
native observation/action/command timing, touchdown air time, reset control,
finite-state, and foot-site velocity bugs described in the earlier audit were
fixed in `4426499` and protected by passing tests. Historical native
checkpoints remain tied to their original timing and cannot be silently
reevaluated under current semantics.

A separate late evaluator audit found an orthogonal defect: fixed evaluations
before `f642bc2` omitted the observation-normalization callback. This invalidates
evaluations of the normalization-enabled D-012-and-later checkpoints, including
the old three-seed held-out metrics/videos. It does not invalidate identity
inference for earlier pilots that actually trained without normalization, though
those remain excluded from the intended profile. C06 is the only saved
normalization-enabled checkpoint with a corrected evaluation: it
survives 10 seconds for stand and forward commands but remains stationary in
100% double support. C05b's exact shipped ONNX policy walks for all eight
10-second command rollouts. The architecture is capable of gait; the unresolved
research problem is local PPO gait acquisition and task/exploration alignment.

See root `ASTRA_HANDOFF.md` for the exact artifact map, validity boundary,
commands, and frozen C07 next step. HOMIE-specific work remains unstarted.
