# Task 3: Reinforcement Learning
This task moves from imitation to reinforcement learning. You will implement an MJX environment and train a PPO policy to pick and place a cube using reward feedback only.

## Outcome
By the end, you should be able to:
- Build a Brax `PipelineEnv` around a MuJoCo MJCF model.
- Define observations, success conditions, and reward shaping terms.
- Train PPO and evaluate rollouts.
- Diagnose RL-specific instability or reward design issues.

## Prerequisite
- Complete Task 1 first and ensure body names used in your XML match the environment code (`panda_hand`, `cube`, `bin` by default).

## Suggested Milestones
1. `PickAndPlace.reset` and `PickAndPlace.step` run without runtime errors.
2. Observation dictionary contains stable `state` and `privileged_state`.
3. Success and grasp checks are implemented and unit-tested with prints/visual checks.
4. Reward function returns meaningful dense signal before sparse success reward.
5. PPO runs on a short smoke test config.
6. Full training runs and saves checkpoints.

## Definition Of Done
- Environment compiles and trains with PPO end-to-end.
- Rollout video runs with the trained policy.
- Success metric increases relative to random initialization.
- You can explain how each reward term affects behavior.

## Practical Advice
- Keep the first reward simple and monotonic.
- Add one reward term at a time and observe behavior changes.
- Start with a short smoke-test run before long training.
- Log success rate and reward components separately when debugging.

See `tips.md` for implementation-level guidance.

## Recommended Resources
- MJX docs: https://mujoco.readthedocs.io/en/stable/mjx.html
- MuJoCo Python API: https://mujoco.readthedocs.io/en/stable/python.html
- Brax repository and examples: https://github.com/google/brax
- JAX docs: https://jax.readthedocs.io/en/latest/
- PPO background (Spinning Up): https://spinningup.openai.com/en/latest/algorithms/ppo.html

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/triton-droids/simulation/blob/onboarding/task3/reinforcement_learning.ipynb?copy=true)
