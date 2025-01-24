# Triton Droids Humanoid Robot Simulator
Welcome to the **Triton Droids** (mjx) codebase! We are the first club at UCSD and second in the nation to embark on the mission of building a humanoid robot. This codebase serves as the foundation for training reinforcement learning models, evaluating learned policies, and enabling sim-to-sim transfer to improve the robustness and generalization of our robot. 

## Installation and Configuration
Please refer to [setup.md](doc\setup.md) for installation and configuration steps.

## Project Overview
This simulator codebase provides a platform for:
- **Train**: Use the MuJoCo simulation environment to let the robot interact with the MJX environment and learn a policy that maximizes the designed rewards. 

- **Play**: Test and evaluate the trained policies in the simulator.

- **Sim2Sim Transfer**: Transfer learned policies from one simulation environment to another to improve robustness and generalization of the agent.
- **Sim2Real Transfer**: Deploy the policy to a physical robot to achieve motion control. 


## Code Structure
Here's an overview of the directory structure:

```
/mjx
│
├── /locomotion                   # Contains locomotion task environments
│   ├── /default_humanoid         # MuJoCo's default humanoid model
│       ├── xmls                  # MJCF xml files and assets 
|       ├── base.py               # base env class
|       ├── joystick.py           # Joystick task implementation
|
```

## User Guide
...



## Future Plans
- **Ros Integration**
- **Isaac Lab**

## Acknowledgments
This repository incorporates code from [mujoco_playground](https://github.com/google-deepmind/mujoco_playground).


