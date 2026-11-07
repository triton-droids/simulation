# Configuration Reference

Native G1 uses a `0.002 s` physics step, solver `2`, 3 solver iterations, 5
line-search iterations, 103 policy observations, 216 privileged observations,
action scale `0.5`, and 10 physics frames per action.

Reset randomization uses XY ±`0.5 m`, yaw ±`3.14 rad`, joint scale `0.5`–`1.5`,
and base velocity ±`0.5`. Observation noise is enabled: joint position `0.03`,
joint velocity `1.5`, gravity `0.05`, linear velocity `0.1`, and gyro `0.2`.
Pushes and domain randomization are configured but disabled in the native G1
profile.

## Sources of truth

- G1: `robocub/config/g1.py`
- Playground: `robocub/config/g1_playground.py`
- PPO: `robocub/config/agents.py`
- Rewards: `robocub/locomotion/unitree_g1/rewards.py`
- Field: `App/Configs/field01.yaml`
- Randomization: `App/Configs/randomization_field01.yaml`
- Dimensions: `App/Configs/field_profiles/medium_14x9.yaml`

Code overrides prose if they ever disagree.
