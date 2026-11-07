# Field #1 implementation report

Status: working local simulation infrastructure and a learned walking visual demo;
not a completed soccer controller or Isaac walking-policy transfer. Existing
locomotion research and its checkpoints were preserved. No new training occurred.

## Delivered and executed

- Procedural 14 x 9 m field, configurable 22 x 14 profile, markings, goals and ball.
  Geometry dimensions and rule-source uncertainties are in FIELD_SOURCES.md.
- Actual pinned Unitree G1 29-DOF model, source metadata and validation. The user
  selected G1; the team's exact hardware revision remains unconfirmed. Proxy mode
  is explicit and is not substituted for the G1 results below.
- Isaac Lab 3.0 / Newton MJWarp GPU environments, configurable count and spacing,
  independent reset checks, nominal controller, smoke checkpoint/evaluation path.
- Browser viewer with overhead/orbit/follow camera, reset, pause and object placement.
  Automated local-browser checks passed; screenshots are in evidence/.
- Seeded, per-environment/episode domain randomization with supported-family counts,
  replay records, held-out presets, actual ball mass/inertia write/readback tests.
- CPU diagnostic backend, 23 passing unit/physics/configuration tests, GPU physics
  preflight, benchmark script, local commands and unsubmitted HPC templates.
- One-command learned walking demo: `watch_soccer.cmd`. See WALKING_DEMO.md.

## Platform and physics validation

Windows RTX 4060 Laptop 8 GB, driver 572.61; isolated Python 3.12.7 environment,
Isaac Lab 3.0.0, Newton 1.6.1, Warp 1.17.0, Torch 2.11.0 + CUDA 12.8.
The accepted solver is MJWarp, 100 iterations, implicitfast, 0.005 s physics step
and four substeps per control step. GPU ball drop, roll and goal-post checks passed
with the actual G1 present. Nominal and held-out 16-environment, 80-step evaluations
completed with finite states and zero observed falls. These are short infrastructure
tests, not locomotion or soccer performance guarantees.

Initial XPBD measurements were invalidated after a ball-state persistence failure.
Explicit Euler produced G1 instability. Both failures remain saved; neither is
included as accepted evidence. Higher-level asset conversion failures were resolved
for G1 using the official native MuJoCo USD converter. Arbitrary third-party URDF/
USD model compatibility is not verified.

## Measured GPU throughput

Each row uses the actual G1, nominal joint hold, 100 control steps after warmup.
Setup/compilation is excluded from throughput; each run represents only 2 simulated
seconds. Aggregate environment steps/s is not per-robot real-time rate.

| Environments | Aggregate environment steps/s |
|---|---:|
| 1 | 58.97 |
| 16 | 507.85 |
| 256 | 3274.34 |
| 1024 | 3735.91 |

N=1024 completed with exit 0; stepping took 27.41 s and setup 16.12 s. N=16/256
isolation checks confirmed resetting environment zero preserves other ball states.
GPU memory telemetry samples the whole device and may miss peaks; it is not a
precise per-process VRAM allocation measurement. See saved execution JSON files.

## Randomization limits

Isaac supports ten registered families: three initial-pose families, ball velocity,
ball mass, angular rolling drag, control latency, proprioceptive noise, observation
delay and external ball pushes. CPU additionally supports actuator gains, motor
strength and joint damping. Unsupported friction/restitution mappings, lighting,
camera jitter, COM and opponents are excluded explicitly. Angular drag is a
documented approximation, not experimentally calibrated rolling friction.
The 1,000-row distribution report checks sampling, not 1,000 physical trajectories.

## Walking handoff

The T02 checkpoint drives G1 in its existing MuJoCo flat-ground simulation. The
same field geometry is drawn around it, with follow and whole-field cameras.
Goals and the ball in this demo are visual only. Its initial 600-step recording
showed 2.22 m net travel, minimum pelvis height 0.745 m and no printed fall reset;
six time-distributed video frames were inspected. Floor overlay flicker was then
removed. The final recording shows 2.365 m net travel over 600 steps with minimum
height 0.745 m; six distributed frames show upright stepping and turning. The native
launcher opened and exited successfully in a separate 12-second viewer smoke test.
A concurrent startup exposed MuJoCo model object serialization failure; the demo
now transfers explicit model bytes through a uniquely named local cache file
(the sandbox's system temporary directory was unwritable).
Full trajectories
and videos remain in Artifacts/Results/field01/t02_walk*/. This is not three-seed validation.

## Reproduce

```powershell
# Learned walking, native window (existing policy environment)
.\watch_soccer.cmd
# Infrastructure viewer, Isaac Lab GPU (separate isolated environment)
.\scripts\field01.cmd view --robot g1 --num_envs 1
# Fast tests
.venv\Scripts\python.exe -m pytest tests/field01 -q
```

QUICKSTART.md contains further smoke/evaluate/randomization commands;
HPC_RUNBOOK.md covers benchmark and cluster templates. Source/config hashes and
compact accepted evidence are in evidence_summary.json; larger raw outputs live
under Artifacts/Results/field01/. The environment lock file records installed versions.
Model caches, policy checkpoints and Python environments remain external local
dependencies and must be provisioned on a new machine.

## Remaining work

T02-to-Isaac observation/action/dynamics parity and learned control are unverified.
The visual walking demo does not implement kicking or goal collisions. Vision,
ball tracking, navigation, soccer strategy, calibrated randomization and hardware
deployment remain outside the delivered checks. Unsupported randomization families
and generic asset imports need their own implementations and validation before use.
No remote job, hardware action or upload was performed.
