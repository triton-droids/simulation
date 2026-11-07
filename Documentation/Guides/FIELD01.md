# Field01

The MuJoCo live field and Isaac Lab Field01 are different environments.

| Capability | MuJoCo field | Isaac Field01 |
|---|---|---|
| Live camera | Yes | No public viewer |
| Loads T02 | Yes | Transfer not implemented |
| Shows field/goals/ball | Yes | Config/assets exist |
| Verified ball interaction | No | No |
| Verified kicking/strategy | No | No |

Field01 targets Isaac Lab `3.0.0`, G1, 256 environments, a `0.005 s` physics
step and control decimation 4. The field is 14 × 9 m, the goal is
2.5 × 1.7 × 1.0 m, and the ball has radius `0.1098169107 m` and mass `0.43 kg`.
The selected `mild` preset samples 2–4 randomization dimensions.

Configuration is not evidence that T02 works there or that soccer training has
succeeded.
