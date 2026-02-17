# Task 3 Tips

## Environment Design Checklist
- Confirm all important body and joint IDs in `_init_env`.
- Keep randomization small first, then widen after learning starts.
- Ensure `reset` initializes every field used later in `step`.
- Keep observation keys stable across training and eval.

## Observation Suggestions
- Include robot joint positions and velocities.
- Include tool-center-point (TCP) position.
- Include cube and bin positions.
- Include relative vectors such as `tcp_to_cube` and `cube_to_bin`.

Good observation design usually matters more than adding many reward terms.

## Success And Grasp Conditions
- `check_grasp`: start with TCP-to-cube distance threshold.
- `check_success`: cube inside bin XY bounds plus height threshold.
- Add hysteresis or margin if success flickers on/off across steps.

## Reward Shaping Template
Use a weighted sum of simple terms:
- Reach reward: encourage TCP to approach cube.
- Grasp/lift reward: reward cube height increase only after grasp.
- Place reward: reward cube approaching bin center.
- Success bonus: sparse terminal reward when cube is in bin.
- Optional action penalty: discourage large unstable commands.

If training is unstable, reduce reward complexity and retune weights.

## PPO Training Tips
- Run a short smoke test with fewer timesteps first.
- Watch `eval/episode_reward` and success metrics together.
- If reward rises but success stays low, reward may be exploitable.
- If nothing improves, check observation scaling and action ranges.

## Common Failure Modes
- Undefined or inconsistent state fields in `state.info`.
- Reward terms that conflict (for example, reaching vs. placing).
- Success condition too strict to ever trigger.
- Domain randomization too large too early.

## Useful References
- Brax PPO implementation: https://github.com/google/brax/tree/main/brax/training/agents/ppo
- Brax environment base classes: https://github.com/google/brax/blob/main/brax/envs/base.py
- JAX sharp bits: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
- MuJoCo MJX docs: https://mujoco.readthedocs.io/en/stable/mjx.html
