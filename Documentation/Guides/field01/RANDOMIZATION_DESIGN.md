# Randomization design and capability boundary

Sampling is keyed by SHA256(master seed, rank, suite, environment ID, episode,
configuration hash). Order of resets does not change another environment's samples.
The hash includes geometry, backend and preset. JSONL logs preserve every reset;
replay rejects a different configuration. GPU trajectories are not promised bitwise
identical. Evaluation seeds 12345/54321 cannot be used for training; stress requires
heldout. Geometry and Earth gravity remain fixed.

The executable capability table is `robocub.soccer/randomization/registry.py`.

| Family | Timing | MuJoCo diagnostic | Isaac Newton |
|---|---|---|---|
| Robot XY/yaw, ball XY (3 families) | reset, per env | applied | applied |
| Ball initial velocity | reset, per env | applied | applied |
| Control latency | control, per env, 0–1 steps | applied | applied |
| Ball rolling loss | interval | angular viscous drag proxy | same explicit drag |
| Ball mass | reset | mass and inertia together | mass/inertia writes + readback |
| Motor strength | reset | actuator/joint effort limits | unavailable |
| PD gain | reset | gain and bias together | unavailable |
| Joint damping | reset | baseline damping multiplier | unavailable |
| Proprioception noise | observation | bounded deterministic noise | bounded deterministic noise |
| Observation delay | observation | one-step history | one-step history |
| External push | interval | bounded force on BALL | bounded force on BALL |
| Static/dynamic turf friction | startup/reset | separate coefficients unavailable | unavailable |
| Restitution | startup/reset | uncalibrated mapping unavailable | unavailable |
| Lighting/camera jitter | rendering | unavailable | unavailable |
| COM offset | prestartup | unavailable | unavailable |
| Opponent bank | reset | unavailable | unavailable |

Unavailable parameters are NEVER sampled or logged as applied. There is no hidden
scene-global physical update represented as independent per-env variation. A
multiplier on a zero nominal damping remains zero; logs expose the effective value.
Ranges are engineering assumptions, not measured hardware tolerances. Static and
dynamic nominal friction must be nonnegative and ordered. These describe intended
materials; separate Coulomb coefficients are not claimed in MuJoCo.

Disabled samples no families. Nominal varies three initial poses. Mild adds 2�4
supported families (Isaac5�7 total); robust adds5�8 (Isaac8�10 total); stress adds7�9
(Isaac10 total). Unsupported actuator/material/vision families remain excluded.
Isaac supports seven extras: ball velocity, angular rolling drag, mass, control
latency, proprioception noise, observation delay and ball pushes. The GPU reset
adapter writes both mass and inertia, checks readback, and clears force/history
buffers for only the selected environments. Wrenches are applied every physics
step; observations are produced every control step. Changing observations does not
change the open-loop scripted baseline's actions; a future policy must consume them.

`Artifacts/Results/field01/samples_mild` contains 1,000 samples, family counts and a distribution
plot. Unit tests cover repeatability, independent IDs, constraints and suite leakage.
Effective MuJoCo values are logged with reset manifests. Physics calibration needs
measured ball roll/bounce, foot contact and actuator data before sim-to-real use.
