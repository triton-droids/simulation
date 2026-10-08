# Triton locomotion: progress and remaining work

**Date:** October 8, 2026
**Branch:** `codex/system-id-velocity`
**Baseline implementation/results commit:** `174fa9d`
**Current assessment:** The training pipeline runs, but the first policy has not demonstrated commanded walking. It is not ready for hardware deployment.

## 1. What we are trying to build

One policy for a ten-joint robot that balances when the controller requests zero speed and walks slowly forward when commanded. It runs at 50 Hz and outputs joint-position targets. The interface keeps three command inputs: forward speed, lateral speed, and yaw rate. Only standing and straight forward walking are in the first milestone.

The actor uses 39 deployable signals: body gyro (3), projected gravity (3), joint position relative to the standing pose (10), joint velocity (10), previous action (10), and command (3). Ground-truth base linear velocity is available to training rewards and the critic, but not the actor.

The main objective is velocity tracking and balance. Reference motion is an optional style aid. This baseline uses neither reference tracking nor AMP.

## 2. What was completed

| Area | Completed | Limit of that evidence |
|---|---|---|
| Repository preservation | Work done in separate simulation and embedded worktrees; existing tracker, original checkouts and old runs preserved | Robot-host uncommitted files still need careful synchronization |
| Deployed-source inspection | Received and inspected the actual 50 Hz ONNX tracking runner/config and system-ID logger | Inspection is not a clean connected hardware test |
| Policy provenance | Old tracker ONNX SHA-256 matched the supplied deployment evidence | Does not demonstrate correct physical motion |
| Interface | New 39-input, 10-output ONNX export passed the smoke test | Matching embedded observations and action processing still need end-to-end validation |
| Simulation task | Flat ground, velocity commands, standing warmup, start/stop command transitions, deployed position clipping, 1 rad/s target slew and spring-term torque guard | Contact, hips, linkage and standing pose remain provisional |
| System identification | Added portable fitting/report/replay tools; reproduced a synthetic knee recovery check and held-out motor-9 replays | Limited joint evidence; not a validated whole-robot model |
| GPU training | Eight-hour PPO run with 4,096 parallel environments on an A40; checkpoints and ONNX saved | Large sample count and reward do not establish walking |
| Evaluation artifacts | Two 15-second, 50 FPS videos, raw rollout arrays and manifest saved and pushed | Two single-environment trials are not a robustness evaluation |

The actual hardware command guard bounds the spring term `kp * (target - position)` using motor-model caps of 80/40/11 Nm. It does **not** establish an absolute cap on total PD torque. That distinction must remain explicit in simulation and evaluation.

## 3. What the first training run actually achieved

Run: `logs/triton_velocity/20261008T051847Z_baseline/`

- Finished normally at its eight-hour wall-time limit.
- Final reported iteration: 23,730.
- Approximately 2.33 billion simulated environment steps.
- Late-run throughput: approximately 78,000 steps per second.
- Final checkpoint: `final.pt`; exported policy: `20261008T051847Z_baseline.onnx`.
- Training used a standing-only warmup, followed by alternating standing and slow forward commands up to 0.2 m/s.

The final training window reported full-length episodes and no fall terminations. This was initially encouraging, but the deterministic videos showed why that is insufficient:

| Recorded trial | Forward request | Mean body-frame forward velocity after first 2 s | Terminations |
|---|---:|---:|---:|
| Standing | 0 m/s | -0.000413 m/s | 0 |
| Forward command | 0.2 m/s | 0.000200 m/s | 0 |

**The forward-command rollout remained almost stationary.** Its mean forward-speed error was approximately 0.1998 m/s. These measurements support standing in these two tested conditions, not successful locomotion.

The videos use fitted nominal actuator values with the evaluation delay setting, rather than a broad sweep of randomized models. They are simulation recordings, not physical tests.

Artifacts:

- [Standing video](../logs/triton_velocity/20261008T051847Z_baseline/videos/standing.mp4)
- [Forward-command video](../logs/triton_velocity/20261008T051847Z_baseline/videos/forward_0p2.mp4)
- [Recording manifest](../logs/triton_velocity/20261008T051847Z_baseline/videos/manifest.json)
- Raw `.npz` files alongside both videos contain velocity, position, actions and termination flags.

## 4. Why walking failed: what is known and what is not

The observed failure is insufficient movement under a nonzero forward command. Its root cause has **not yet been established**. More training time alone is not a justified fix.

The next diagnosis must distinguish these possibilities:

1. **Command delivery or evaluation mismatch.** Trace the requested command through resampling, command slew, the actor's final three inputs and the reward's command. Verify public `[forward, lateral, yaw]` versus plant `[right, forward, yaw]`. Verify ONNX inference agrees with the checkpoint under identical observations and normalization.
2. **A standing solution is rewarded too generously.** Compare reward terms for staying still versus moving at 0.05, 0.1 and 0.2 m/s. Check tracking sensitivity and the balance between tracking, pose, smoothness, effort and gait terms. The current low requested speeds may make stationary behavior a competitive solution; this is a hypothesis to measure.
3. **Action processing prevents useful stepping.** Measure raw actions, targets before/after clipping, slew, spring-term guarding and actual joint motion. Check whether action scale, the provisional stance or model dynamics make stepping difficult. Keep verified hardware slew and limits; do not raise them just to improve a video.
4. **The curriculum produces an unhelpful local solution.** Measure actual time spent at zero/nonzero commands, start/stop durations and speed distribution. Compare checkpoints around the transition to walking.
5. **The plant model makes this gait unnecessarily difficult or inaccurate.** Inspect contact geometry, foot support, ankle linkage and joint dynamics. Validate changes with measurements rather than arbitrary parameter searches.

AMP could make movement resemble a reference, but it cannot repair a missing command, an incorrect actuator model or communication faults. It should not replace this diagnosis.

## 5. Hardware and system-identification status

The supplied robot-host preflight resolved the runner/config and displayed motor IDs, signs, gains, limits, 50 Hz control, 1 rad/s slew and a 15 ms feedback timeout. It failed the live-device checks because `can0` and the configured IMU serial device were absent. The robot was disconnected. This was a **static preflight, not a successful live feedback check**.

The October 2 logs include bus-wide missed replies and a knee-limit stop. Communication timing and reply freshness need investigation before blaming friction or armature. An inspected startup path also needs review: partial motor enable can occur before the interface marks itself connected, while cleanup depends on that connected state. The appropriate fix and its tests are still pending.

Existing archived trials provide useful initial fitting evidence. The held-out motor-9 replay improved position RMS error:

| Trial | Previous model | Fitted model |
|---|---:|---:|
| Small step | 0.368 degrees | 0.265 degrees |
| 0.5 Hz sine | 0.495 degrees | 0.240 degrees |

This is narrow knee-response evidence in the recorded gantry setup. It does not validate hips, foot contact, maximum torque/speed or free-standing balance. Hip fits were confounded by gantry motion; an unhealthy ankle trial was excluded. The ankle model still uses a constant linkage ratio, and contact/body parameters are not independently identified.

An actuator network is **not implemented**. First establish whether a physical delay/PD/friction/inertia model predicts held-out data. Consider a learned actuator model only if repeatable residual behavior remains and sufficient clean data exist.

## 6. Remaining work, in order

### A. Diagnose this policy before another overnight run

- Trace command values through actor observations and rewards during standing, forward and start/stop trials.
- Compare checkpoint and ONNX actions with the same observations.
- Quantify target saturation, slew/guard intervention, action variation, torque, joint speed, slip and signed forward travel.
- Inspect earlier checkpoints and compare moving versus stationary reward breakdowns.
- Run short controlled reward/curriculum experiments; require measurable forward response before committing another eight hours.

**Exit condition:** Commands affect policy behavior, and short evaluations show sustained forward travel with bounded speed error while preserving standing. Agree numerical acceptance thresholds before selecting a policy.

### B. Finish hardware feedback and actuator validation

- Resolve startup/cleanup handling and audit the existing independent E-stop/fault path.
- On the robot-connected machine, complete read-only preflight, then a user-triggered ten-second stationary pose-hold recording at 50 Hz.
- Record monotonic command/reply timestamps, freshness, pre/post-limit targets, position, velocity, reported torque, temperature, IMU, available voltage and stop reasons.
- If reply losses recur, stop motion testing and diagnose transport/timing.
- Prepare small one-joint tests with the other nine joints held; the user triggers each physical trial.
- Keep raw trials and manifests under a new `logs/system_id/<timestamp>/` directory.
- Fit delay and PD response first, then torque/speed, friction, reflected inertia, nonlinear ankle linkage and contact. Fit and validate on different trials, using identical post-slew commands.

**Exit condition:** Clean feedback and independently validated actuator predictions, with unresolved or unhealthy joints addressed.

### C. Retrain and evaluate the corrected baseline

- Use new run directories and the measured hardware model.
- Randomize supported uncertainty, keeping provisional assumptions visible.
- Evaluate multiple seeds/environments for standing, repeated starts/stops and straight walking under nominal, measured-delay and plausible extreme conditions.
- Report falls, signed travel, speed error, torque, joint speed, target saturation, foot slip and action variation.
- Compare the old tracker under the same plant model; this comparison is still pending.

**Exit condition:** Reliable commanded locomotion by behavioral metrics, rather than reward or appearance alone.

### D. Add gait style only after baseline success

- First try a separate low-weight cyclic, speed-tolerant reference-style run.
- Consider AMP as a separate experiment if it offers a clear benefit and has suitable reference data.
- Keep style only if it preserves balance, speed tracking and start/stop behavior.

### E. Complete deployment integration

- Verify embedded ONNX observations, units, joint signs, gains, limits, action processing and inference timing against simulation.
- Complete controller integration: trained speed bounds, dead zone, command slew, held deadman and stale-command timeout.
- Keep low-level pose hold for homing before policy handover, with independent E-stop/fault handling.
- Advance to short supported real trials only after clean feedback and acceptable sim/real agreement. The user triggers physical motion.

## 7. Current recommendation

Keep this run as a reproducible failed walking baseline. Do not deploy it or spend another night training the unchanged setup. The next useful milestone is a verified command/reward/action trace and a short run that actually moves forward on command. Hardware communication and actuator identification can proceed alongside that work when the robot is connected.
