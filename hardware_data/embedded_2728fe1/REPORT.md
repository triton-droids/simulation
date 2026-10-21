# Embedded tracking-log analysis

Source: `triton-droids/embedded` commit `2728fe19eb17908ac5bd4d88c8e9a131111a9afd`.

## What the existing data supports

These logs lock down the deployed 50 Hz command contract, joint order, signs, limits, gain sets, safety exits, target slew, and a closed-loop response envelope. They are enough to continue simulation training and domain randomization now.

The lag below is command-to-measured-velocity response. It combines PD response, mechanical dynamics, contact, filtering, and communications. It is not a pure transport-delay estimate and should be used to validate randomized rollouts, not copied directly into a delay queue.

## Gain-profile summary

| profile | runs | completed live | duration | tracking RMSE | response lag | median correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deployed | 4 | 1/3 | 19.6 s | 0.079 rad (0.075-0.083) | 80 ms (60-100) | 0.78 |
| high_gain_experimental | 6 | 3/6 | 19.2 s | 0.088 rad (0.061-0.102) | 80 ms (40-100) | 0.77 |

## Run inventory

| file | mode/profile | status | duration | fresh | guard clip | RMSE | max target slew |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| tracking_hold_20261002_023320.npz | hold/deployed | completed | 5.00 s | 99.6% | 0.0% | 0.458 rad | 1.00 rad/s |
| tracking_live_20261002_012813.npz | live/high_gain_experimental | completed | 5.98 s | 91.0% | 0.0% | 0.068 rad | 1.00 rad/s |
| tracking_live_20261002_020907.npz | live/high_gain_experimental | completed | 5.98 s | 99.7% | 0.0% | 0.061 rad | 1.00 rad/s |
| tracking_live_20261002_021538.npz | live/high_gain_experimental | joint_limit | 0.02 s | 10.0% | 0.0% | 0.047 rad | 0.00 rad/s |
| tracking_live_20261002_021556.npz | live/high_gain_experimental | missed_feedback | 0.76 s | 87.1% | 3.9% | 0.093 rad | 2.00 rad/s |
| tracking_live_20261002_021644.npz | live/high_gain_experimental | missed_feedback | 0.50 s | 80.4% | 3.2% | 0.102 rad | 2.00 rad/s |
| tracking_live_20261002_021735.npz | live/high_gain_experimental | completed | 5.98 s | 99.7% | 3.0% | 0.088 rad | 2.00 rad/s |
| tracking_live_20261002_023417.npz | live/deployed | completed | 5.98 s | 99.7% | 0.0% | 0.075 rad | 1.00 rad/s |
| tracking_live_20261002_033349.npz | live/deployed | missed_feedback | 3.54 s | 97.5% | 0.0% | 0.079 rad | 1.00 rad/s |
| tracking_live_20261002_033830.npz | live/deployed | missed_feedback | 5.12 s | 98.1% | 0.0% | 0.083 rad | 1.00 rad/s |

## Decision

Proceed with the velocity-policy training pipeline. Keep the two gain profiles separate, reproduce the measured closed-loop response envelope in simulation, and retain conservative dynamics randomization because these logs do not uniquely identify motor friction, link inertia, ground contact, or calibrated joint torque.

Before the first untethered policy deployment, collect dedicated per-joint tests: small step and chirp commands at the intended deployed gains, unloaded and loaded; log commanded position, measured position/velocity, timestamps, voltage, current, and calibrated output torque when available. Also measure total robot mass, segment masses/centres of mass, and foot-ground friction. Those tests can narrow the current randomization ranges; they do not block policy development.

Raw per-joint lag and correlation values are in `analysis.json`.
