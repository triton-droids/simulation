# Test Field 01 quickstart

This package is separate from the completed G1 locomotion milestone. No earlier
training results, policies or environments are replaced. Start here, not the old
research queue. `Documentation/Guides/field01/IMPLEMENTATION_STATUS.md` contains executed evidence.

## Simple full-pitch camera (tested native diagnostic viewer)

From the repository root on this Windows laptop:

```powershell
.venv/Scripts/python.exe robocub/tools/field01 view --backend mujoco --robot g1 --randomization disabled
```

The default overhead camera shows the entire field. O returns overhead, C selects
orbit (mouse drag/scroll), F follows the robot fixture. P pauses/resumes, R resets,
B puts the ball in the center, arrow keys reposition the ball in0.25m increments,
I/J/K/L reposition the fixture. Invalid/out-of-bounds or overlapping placements are
rejected. S captures a screenshot. Close the window to stop. A terminal state pauses
instead of hiding the failure with an automatic reset. R then P continues.

G1 is the user-selected default, using the pinned Menagerie 29-joint model. This
joint-holding demonstration does not load the earlier learned walking controller:
it can fall and the viewer pauses honestly. Exact team hardware revision remains
unverified. Explicit `--robot demo` selects the orange fixed-base two-joint fixture;
it is labeled DEMO/PROXY and must never be confused with humanoid validation.

A reproducible3-second recording, with explicit backend and labels:

```powershell
.venv/Scripts/python.exe robocub/tools/field01 view --backend mujoco --robot g1 --randomization disabled --record --steps 150 --run_dir Artifacts/Results/field01/my_video
```

Outputs: field01.png, field01.mp4, trajectory.json and manifest.json. Output paths
must be fresh. Native rendering is separate from the headless simulation loop.

## Default platform: Isaac Lab3.0

Use a SEPARATE Python3.12 environment. Do not install into the locomotion environment.
The default backend is isaaclab; a missing runtime errors rather than silently
switching engines. Pinned implementation: isaaclab3.0.0 + Newton1.6.1, MJWarp100iterations, implicitfast integration,
dt=.005, control decimation4. Full installed dependency lock is captured with evidence.

```powershell
.cache/field01_venv/Scripts/python.exe robocub/tools/field01 doctor
.cache/field01_venv/Scripts/python.exe robocub/tools/field01 smoke --num_envs 1 --randomization nominal --seed 42 --run_dir Artifacts/Results/field01/isaac_check
.cache/field01_venv/Scripts/python.exe robocub/tools/field01 view --robot g1 --randomization nominal --seed 42
```

Isaac's browser viewer uses pinned Viser1.0.26, bound to127.0.0.1:8080 with public
sharing disabled. Open that address after launch. The Soccer field panel starts
paused and offers Overhead/Orbit/Follow, reset, and legal robot/ball placement.
Uncheck Pause field to simulate. `--view_seconds 30` bounds a viewer smoke test.
`--record` writes a replayable `scene.viser` recording, not an MP4. The native MuJoCo
viewer supplies MP4 output. Neither is evidence of a learned soccer controller.

The default upstream CUDA13 installation requires a newer driver than this laptop's
572.61. Official PyTorch2.11.0 CUDA12.8 wheels successfully ran the actual G1 GPU smoke.
Consult the status report for measured scale and limitations. Do not upgrade a shared host driver casually.

## Real robot input

```powershell
.cache/field01_venv/Scripts/python.exe robocub/tools/field01 view --robot real --robot_asset C:/models/humanoid.usd --robot_metadata C:/models/robot.json
```

Metadata keys are described in App/Configs/robots.example.yaml (CLI input is JSON).
Identity, SI units, source license, initial height, every controlled joint's nominal
position, and actuator parameters must be explicit. USD is imported as an Isaac
articulation. URDF/MJCF use Isaac's optional importer dependencies; missing importer
packages are a blocker, not permission to replace the robot. The MuJoCo diagnostic
composition accepts a flattened robot-only MJCF. No auto conversion changes its
scale or silently invents actuator metadata. A model loading does not prove hardware
fidelity; contact/COM/joint checks and hardware-specific acceptance remain necessary.

## Headless, sampling, evaluation

```powershell
.venv/Scripts/python.exe robocub/tools/field01 smoke --backend mujoco --num_envs 16 --randomization mild --steps 100 --run_dir Artifacts/Results/field01/my_smoke
.venv/Scripts/python.exe robocub/tools/field01 sample --backend mujoco --num_envs 1000 --randomization mild --run_dir Artifacts/Results/field01/my_samples
.venv/Scripts/python.exe robocub/tools/field01 train --backend mujoco --smoke_only --num_envs 16 --steps 20 --run_dir Artifacts/Results/field01/my_train_smoke
.venv/Scripts/python.exe robocub/tools/field01 evaluate --backend mujoco --suite nominal --seed 12345 --checkpoint Artifacts/Results/field01/my_train_smoke/checkpoint.json --run_dir Artifacts/Results/field01/my_nominal
.venv/Scripts/python.exe robocub/tools/field01 evaluate --backend mujoco --suite heldout --seed 54321 --checkpoint Artifacts/Results/field01/my_train_smoke/checkpoint.json --run_dir Artifacts/Results/field01/my_heldout
.venv/Scripts/python.exe robocub/tools/field01 report --run_dir Artifacts/Results/field01/my_smoke
.venv/Scripts/python.exe -m pytest tests/field01 -q
```

`train --smoke_only` validates a scripted baseline and emits a clearly typed scripted
checkpoint. It performs NO learning. Real policy checkpoints are rejected until an
explicit policy adapter is provided; they are never silently replaced. `train`
without --smoke_only fails with an actionable explanation. An external trainer can
call SoccerEnv.reset/step: action[N,nu] is joint-position offset radians (slide joints
use meters), clipped±.5; observation[N,2nu+13] contains joint position/velocity,
ball xyz/velocity, root xyz/quaternion(wxyz in MuJoCo). Reward is ball +X displacement
minus effort cost, termination is invalid/out-of-region ball, truncation1500steps.
A G1 root below0.35m also terminates. Each done instance requires explicit reset. This is a research scaffold, not a
striker task acceptance criterion. Isaac uses its native tensor/articulation API;
its observation quaternion boundary is xyzw, not the MuJoCo wxyz order.


## Writable compiler caches on this Windows machine

Before Isaac commands, set these repository-local cache directories (already created):

```powershell
$env:TEMP=(Resolve-Path .cache/field01_tmp).Path
$env:TMP=$env:TEMP
$env:WARP_CACHE_PATH=(Join-Path (Resolve-Path .cache).Path 'field01_warp')
```

The isolated environment needs its pinned CUDA12.8 PyTorch wheels. Do not replace
these with pip's CUDA13 default on driver572.61. The G1 import uses the official
`mujoco-usd-converter` Newton path. The higher-level isolated Isaac Sim importer
was tested and failed on missing PhysX schema; its failure logs are preserved.
The upstream `G1_29DOF_CFG` asset unexpectedly contained43 joints and was rejected.
Our import asserts29 joints and records the original asset and mesh hashes.

MuJoCo uses `g1_mjx.xml`, Isaac uses `g1.xml`, both from the same pinned Menagerie
revision. Their contact representations and gains differ; cross-engine physical
parity is NOT claimed. No neural policy has been transferred between engines.


## Accepted GPU validation and performance

```powershell
.cache/field01_venv/Scripts/python.exe robocub/tools/field01 smoke --robot g1 --num_envs 3 --validate_physics --randomization disabled --steps 10 --run_dir Artifacts/Results/field01/fresh_physics
.cache/field01_venv/Scripts/python.exe robocub/tools/benchmark_field01.py --robot g1 --counts 1 16 256 --output Artifacts/Results/field01/fresh_benchmark
```

The benchmark refuses to run if the GPU drop/roll/goalpost preflight fails. Earlier
XPBD loop timings are invalidated because the ball state did not advance correctly.
Explicit Euler also produced nonfinite G1 states; implicitfast matches the pinned
G1 source model and passes the unchanged physics gates. See the report for accepted
runs, finite-state checks and exact limits. No training is launched by benchmarks.
