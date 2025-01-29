# Triton Droids Humanoid Robot Simulator
Official repository for **Triton Droids** simulations.

## Installation and Configuration
Please refer to [setup.md](doc\setup.md) for installation and configuration steps.

## Project Overview
This simulator codebase provides a platform for:
- **Train**: Use the MJX simulation environment to learn a policy that maximizes the designed rewards. 

- **Play**: Test and evaluate the trained policies in the simulator.

- **Sim2Sim Transfer**: Transfer learned policies from one simulation environment to another to improve robustness and generalization of the agent.
- **Sim2Real Transfer**: Deploy the policy to a physical robot to achieve motion control. 


## Code Structure
Here's an overview of the directory structure:

```
/mjx
│
├── /configs                          # Default config files for env, sim, and rl params
├── /envs                             # Contains all RL environments
│   ├── /locomotion                   # Locomotion tasks
├── /scripts                          # Train, play, evaluate scripts
├── /utils                            # Utility functions and task manager
```

## User Guide
`run.py` serves as the main entry point. The execution environment can be configured using the `instructs.yml` file, allowing users to override default settings. If not instructions are provided the default configuration will run. Refer to the provided [template files](doc\templates) to see which variables can be modified.


### 1. Training
Run the following command to start training:


```
python run.py --env=xxx --framework=brax_ppo 
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


## Acknowledgments
This repository incorporates code from [mujoco_playground](https://github.com/google-deepmind/mujoco_playground).


