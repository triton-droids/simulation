# Rewards Reference

Positive rewards encourage behavior; negative rewards penalize it.

| Term | Scale | Purpose |
|---|---:|---|
| `tracking_lin_vel` | 2.0 | Match planar speed |
| `tracking_ang_vel` | 1.5 | Match yaw rate |
| `lin_vel_z` | -0.01 | Reduce bouncing |
| `ang_vel_xy` | -0.15 | Reduce roll/pitch motion |
| `orientation` | -2.0 | Stay upright |
| `torques` | -0.00001 | Reduce effort |
| `energy` | -0.00001 | Reduce power |
| `action_rate` | -0.005 | Smooth actions |
| `dof_acc` | -0.000000025 | Smooth acceleration |
| `dof_pos_limits` | -1.0 | Respect joint limits |
| `feet_slip` | -0.25 | Prevent foot slip |
| `collision` | -0.1 | Avoid body contact |
| `termination` | -100.0 | Penalize failure |
| `alive` | 0.5 | Reward survival |
| `feet_air_time` | 2.0 | Reward swing time |
| `feet_phase` | 1.0 | Match gait phase |
| `stand_still` | -1.0 | Hold pose when stopped |
| `pose` | -0.1 | Limit pose deviation |

Tracking uses `exp(-squared_error / 0.25)`. Air time rewards `0.2`–`0.5 s`;
phase uses a `0.15 m` swing target. Full HOMIE phases are not implemented or
validated, so this must not be described as a completed HOMIE policy.
