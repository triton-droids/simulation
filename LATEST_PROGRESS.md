# Latest locomotion progress

Updated 2026-11-09 (Antioch run timestamps are UTC).

## Current state

The humanoid lower-body model has **not yet demonstrated reliable commanded walking** in Isaac Lab/PhysX. No MJCF, USD, mesh, linkage, or mass properties were changed during these locomotion experiments. The remaining problem is controller and policy behavior, not a newly discovered linkage closure failure.

The latest saved episode is [the teacher stability v3 preview](robots/humanoid_legs_5/locomotion_latest_episode_teacher_stability_v3.mp4). It stays upright for 10 seconds with no reset, but moves only 0.089 m forward; final-half forward speed is 0.0026 m/s. It is a standing result, not a walking result.

## Recent Isaac Lab evaluations

All rows below use the balance-stage21 checkpoint with a hip/knee gait overlay, a fixed body-Y command of 0.12 m/s, 0.22 action scale, reset noise 0.35, 32 environments, and a 10-second first-episode evaluation. Resets are counted before any reset state can inflate the trajectory.

| Setting | Mean forward speed | First-episode survival | Forward displacement | Heading drift | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Hip 0.10 rad, knee 0.08 rad, 0.35 Hz, balance blend 0.515 | 0.113 m/s | 5/32 | 0.590 m mean | 13.8° mean absolute | Moves, then most trials fall |
| Hip 0.12 rad, knee 0.10 rad, 0.35 Hz, balance blend 0.45 | 0.127 m/s | 4/32 | 0.566 m mean | 14.5° mean absolute | Slightly faster, worse survival |
| Baseline plus positive yaw-hip correction | 0.057 m/s | 4/32 | 0.625 m mean | 13.4° mean absolute | More falls; rejected |
| Baseline plus negative yaw-hip correction | 0.004 m/s | 28/32 | 0.020 m mean | 3.3° mean absolute | Stable but effectively stands; rejected |

The baseline gait moves for about 6.7 seconds on average before falling. The hip-offset correction did not solve the tradeoff: one direction increased falls, while the other suppressed the gait. The balance policy often saturates its hip outputs, so small external corrections are clipped by the action wrapper.

The final blend 0.25 evaluation was explicitly stopped before it completed; it has no result and should not be treated as evidence.

## Changes in this update

- `src/test_policy_gait_overlay.py` supports single-configuration sweeps and records per-environment first-episode displacement, survival, and heading. This prevents post-reset motion from making a failing gait appear successful.
- The failed yaw-offset experiment was removed from the reusable evaluator.
- No training job is currently running. The Antioch simulator service remains available.

## Next work when locomotion work resumes

1. Put heading and fall resistance into the trainable policy objective rather than relying on offsets that are clipped by the action wrapper.
2. Continue from the gait imitation checkpoint while preserving its stepping behavior; evaluate first-episode survival, commanded-speed tracking, lateral drift, and heading together.
3. Save a single-robot video and checkpoint only after a multi-environment run confirms sustained walking without resets.

Current branch: `antioch/humanoid_legs_5`. Best available walking candidate remains the gait-imitation checkpoint, but it is fragile; the most stable PPO checkpoint stands still. Neither is ready for locomotion training at scale or hardware use.
