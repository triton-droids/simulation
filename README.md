# Triton Droids Humanoid Robot Simulator

Official repository for **Triton Droids** simulations.

This codebase provides a framework to train, evaluate, and test reinforcement learning (RL) policies using MJX and MuJoCo.

## Installation and Configuration

Please refer to [setup.md] for installation and configuration steps.

For the lower-body tracking ONNX bench and its dedicated Python environment, see
[standalone policy benchmark](scripts/policy_bench/README.md). The sibling
`embedded` repository contains the ROS 2 50 Hz integration and native topic-rate
test instructions in `docs/tracking_policy_ros2.md`.

## 🚀 Features

This simulator codebase provides a platform for:

- **Train**: Learn policies using MJX and Brax PPO.

- **Play**: Test trained policies inside MuJoCo environments.

- **Sim2Sim Transfer**: Transfer policies between simulators to improve robustness. (WIP)

- **Sim2Real Transfer**: Deploy policies to physical robots for real-world control. (Future goal)

## 📂 Project Structure

```Markdown
/simulation
├── /scripts                              # Shell scripts to automate training execution
├── /src
│   ├── /configs                          # Dataclass config files for environmnets, sim, and rl params
│   ├── /locomotion                       # (mjx) Locomotion task environments
│   ├── /rewards                          # Reward functions
│   ├── /robots                           # Robot object to represent MuJoCo model
│   ├── /scripts                          # Python entry points for training and policy playback
│   ├── /sim                              # MuJoCo utilities (physics rendering and sim-state management)
│   ├── /tools                            # Experiment tools
│   ├── /utils                            # General-purpose uitilites
```

## Usage

### 1. Training

Run the following command to start training:

```shell
./scripts/single_train_mjx.sh
```

Parameter Description

- `VIDEO`: Flag to enable/disable video recording during training.

- `VIDEO_LENGTH`: Number of simulation steps (frames) to record per video. Controls how long each video lasts.
- `VIDEO_INTERVAL`:Interval (in training steps or timesteps) between video recordings.

### 2. Play

Coming soon.

### 3. Sim2Sim

Coming soon.

## Future Plans

- **Ros Integration**
- **Isaac Lab**

## Acknowledgments

This repository incorporates code from [toddlerbot](https://github.com/hshi74/toddlerbot).
