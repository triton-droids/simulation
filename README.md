# Triton Droids Humanoid Robot Simulator

Official repository for **Triton Droids** simulations.

## Installation and Configuration

Please refer to [setup.md] for installation and configuration steps.

## Project Overview

This simulator codebase provides a platform for:

- **Train**: Use the MJX simulation environment to learn a policy that maximizes the designed rewards.

- **Play**: Play the learned policies in the simulator, and return an evaluation metric. (IN PROGRESS)

- **Sim2Sim Transfer**: Transfer learned policies from one simulation environment to another to improve robustness and generalization of the agent. (IN PROGRESS)

- **Sim2Real Transfer**: Deploy the policy to a physical robot to achieve motion control. (IN PROGRESS)

## Code Structure

Here's an overview of the directory structure:

```Markdown
/mjx
│
├── /configs                          # Default config files for env, sim, and rl params
├── /envs                             # Contains all RL environments
│   ├── /locomotion                   # Locomotion tasks
├── /scripts                          # Train, play, evaluate scripts
├── /utils                            # Utility functions and task manager
```

## User Guide

`run.py` serves as the main entry point. The execution environment can be configured using the `instructs.yml` file, allowing users to override default settings. If no instructions are provided the default configuration will run.

### 1. Training

Run the following command to start training:

```shell
python run.py train --env=xxx --framework=brax_ppo 
```

Parameter Description

- `env`: Required parameter; values can be

(DefaultHumanoidJoystickFlatTerrain,
DefaultHumanoidJoystickRoughTerrain)

- `framework`: Required parameter; values can only be (brax_ppo) atm.
- `name`: Name of the run / experiment.
- `checkpoint`: Path to model checkpoint to resume training.

### 2. Play

Coming soon.

### 3. Sim2Sim

Coming soon.

## Future Plans

- **Ros Integration**
- **Isaac Lab**

## TODO

- train a policy and evaluate environment setup
- implement play.py

## Acknowledgments

This repository incorporates code from [mujoco_playground](https://github.com/google-deepmind/mujoco_playground).
