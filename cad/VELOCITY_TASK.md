# Triton experimental velocity baseline

Simulation-only PPO task for ten joints, adapted from MjLab velocity control.
Actor inputs are gyro (3), projected gravity (3), position relative to standing (10), joint velocity (10), previous action (10), and public command [forward, lateral, yaw] (3). Plant axes are X right, Y forward, Z up. Actor has no ground-truth base linear velocity.

Control is 50 Hz. Position targets use action scale 0.2, verified joint clipping, 1 rad/s target slew, and deployed spring-term torque guards of 80/40/11 Nm by motor model. Motor gains are 100/100/100/80/22 and kd 2 per leg. The guard is not an absolute cap on total PD torque. Ankle linkage uses an approximate constant ratio of 0.957.

Training starts with 1500 iterations of zero commands, then alternates standing and slow forward commands with slew and frequent start/stop transitions. Maximum forward speed progresses from 0.1 to 0.2 m/s. Lateral and yaw commands stay zero. Plain velocity rewards are the baseline; AMP is absent and the optional style reward defaults to zero.

Only selected fitted actuator uncertainty is randomized. Confounded hip fits, contact, body mass, IMU noise and calibration are not broadly randomized. Hip parameters remain provisional. Observation delay is one control step; action delay spans 0–2 physics steps. Hardware communication faults and unhealthy ankle behavior remain unresolved hardware validation gates.

## Commands

From the simulation repository, with the MjLab environment installed:

```bash
PYTHONPATH=mjlab/src mjlab/.venv/bin/python cad/smoke_test_velocity.py --fit /path/to/sysid_params.json --num-envs 4 --device cuda:0 --train
PYTHONPATH=mjlab/src mjlab/.venv/bin/python cad/train_velocity.py --fit /path/to/sysid_params.json --num-envs 4096 --iterations 1000000 --hours 8
MUJOCO_GL=egl PYTHONPATH=mjlab/src mjlab/.venv/bin/python cad/record_velocity_videos.py logs/triton_velocity/<run>
```

Runs get fresh directories, checkpoints, ONNX exports, configuration and source snapshots. ONNX input is obs [1,39], output actions [1,10]. Changing the input contract requires a matching embedded runner; the old fixed-reference tracker is incompatible. No physical tests are launched by these scripts.

## First experiment

The 20261008T051847Z_baseline run completed its eight-hour deadline at iteration 23730, approximately 2.33 billion simulated environment steps. Saved deterministic 15-second standing and forward-command videos had no termination, but the forward-command trial achieved approximately 0.0002 m/s despite a 0.2 m/s request. This baseline has not demonstrated walking and is not deployment-ready. Videos, raw rollout arrays, final checkpoint, ONNX and run configuration are included in that run directory; intermediate checkpoints and TensorBoard event files remain on the training machine.

Next work is to diagnose command tracking and reward behavior, evaluate starts/stops and held-out model conditions, and compare against the old tracker. Reward and video alone do not qualify hardware deployment.
