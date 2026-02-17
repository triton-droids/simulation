# Task 2: Imitation Learning
This task introduces behavior cloning (BC) for a MuJoCo pick-and-place setup. You will use a motion planner as the expert, collect trajectories, train a policy, and evaluate it in simulation.

## Outcome
By the end, you should be able to:
- Build a Gymnasium-compatible MuJoCo environment wrapper.
- Collect expert demonstrations from planner rollouts.
- Train a stochastic policy that predicts a Gaussian action distribution.
- Evaluate policy success and diagnose BC failure modes.

## Prerequisite
- Complete Task 1 first so `assets/descriptions/DropCubeInBinEnv.xml` includes the Panda arm, cube, bin, and a `"home"` keyframe.

## Why learners get stuck
Most difficulty comes from three places:
- Observation/action shape mismatches.
- Planner target poses in the wrong frame or orientation.
- Training loops that run but never improve due to data quality issues.

Use `tips.md` as your implementation checklist, not just theory notes.

## Suggested Milestones
1. Environment scaffold works.
2. Planner solves at least one episode reliably.
3. Dataset collection works on a small run (`20-50` successful episodes).
4. `TrajectoryDataset` returns valid `(obs, action)` tensors.
5. Actor forward pass returns `(mean, log_std)` with correct shapes.
6. Training loop runs and saves a best checkpoint.
7. Evaluation runs end-to-end in simulator.

## Definition Of Done
- Notebook executes without syntax errors.
- You can collect a non-empty dataset file.
- The trained policy loads and runs in the environment.
- You can explain one BC limitation observed during rollout.

## Practical Advice
- Start with small experiments first.
- Validate dimensions early with print/assert checks.
- Save intermediate outputs often (planner success rate, dataset size, eval success rate).
- Use deterministic policy mean for evaluation before trying stochastic sampling.

## Recommended Resources
- MuJoCo Python tutorial: https://mujoco.readthedocs.io/en/stable/python.html
- MuJoCo XML reference: https://mujoco.readthedocs.io/en/stable/XMLreference.html
- Gymnasium custom env guide: https://gymnasium.farama.org/main/introduction/create_custom_env/
- Gymnasium wrappers (`TimeLimit`, `RecordVideo`): https://gymnasium.farama.org/api/wrappers/misc_wrappers/
- PyTorch datasets/dataloaders: https://docs.pytorch.org/docs/stable/data.html
- PyTorch Normal distribution: https://docs.pytorch.org/docs/stable/distributions.html#normal

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/triton-droids/simulation/blob/onboarding/task2/imitation_learning.ipynb?copy=true)
