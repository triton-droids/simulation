# Humanoid Simulation Validation

A MuJoCo-based validation environment for testing humanoid policies trained in IsaacSim. This repository provides CPU-based simulation environments for disturbance rejection and locomotion tasks, along with visualization and debugging tools.

## Installation

### Prerequisites

- Python 3.12+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd simulation
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

3. Install the package:
```bash
# Install in editable mode (for development)
pip install -e .

# Or install normally
pip install .
```

#### Optional: Development Tools

To install additional development dependencies (pytest, black, ruff):
```bash
pip install -e ".[dev]"
```

**Note**: Before publishing or sharing, update the author information and repository URLs in `pyproject.toml`.

## Quick Start: Policy Evaluation Workflow

### 1. Add Your Policy

Place your trained policy file (`.pth` format) in the `policies/` directory:

```bash
policies/
└── your_policy_name.pth
```

### 2. Run Visualization

Use `visualize.py` to evaluate your policy and generate output videos:

#### Disturbance Rejection Environment (default):
```bash
# With standing pose (no trained policy)
python visualize.py -d

# With your trained policy
python visualize.py -d --policy policies/disturbance_rejection.pth
```

#### Locomotion Environment:
```bash
# With standing pose
python visualize.py -l

# With your trained policy
python visualize.py -l --policy policies/your_policy_name.pth
```

#### Additional Options:
```bash
# Specify video duration and FPS
python visualize.py --duration 10.0 --fps 60 --policy policies/your_policy_name.pth

# Show help
python visualize.py --help
```

### 3. View Output

Generated videos are saved in the `output/` directory:
- `.gif` format for quick preview
- `.mp4` format for high-quality playback

Output naming convention:
```
output/
├── test_disturbance_standing.gif      # Disturbance env, standing pose
├── test_disturbance_standing.mp4
├── test_disturbance_trained.gif       # Disturbance env, trained policy
├── test_disturbance_trained.mp4
├── test_locomotion_standing.gif       # Locomotion env, standing pose
└── test_locomotion_standing.mp4
```

## Repository Structure

### Core Environment Files

#### `disturbance_env.py`
- **Purpose**: Validation environment for disturbance rejection tasks
- **Features**:
  - Applies random force and torque disturbances to the robot
  - Default: 0% disturbance probability (configurable up to 100%)
  - Max disturbance force: 25N
  - Max disturbance torque: 10Nm
- **Observation Space**: Gravity vector, angular velocity, joint positions/velocities, joint torques, previous actions (frame-stacked)
- **Use Case**: Testing policy robustness to external forces

#### `locomotion_env.py`
- **Purpose**: Validation environment for locomotion tasks
- **Features**:
  - Includes velocity command observations (x, y)
  - Reduced disturbances (5N force, 2Nm torque, 2% probability)
  - Tracks linear velocity in body frame
- **Observation Space**: Gravity vector, angular velocity, velocity commands, joint states, torques, previous actions (frame-stacked)
- **Use Case**: Testing walking/running policies

Both environments:
- Run at 50 Hz control frequency (configurable)
- Use 10 physics substeps per control step (0.002s physics timestep)
- Support frame stacking (default: 3 frames)
- Include MuJoCo rendering capabilities

### Visualization and Testing

#### `visualize.py`
Main script for policy evaluation and video generation.

**Command-line Arguments**:
- `-l, --locomotion`: Use locomotion environment
- `-d, --disturbance`: Use disturbance environment (default)
- `--policy PATH`: Path to trained policy `.pth` file
- `--duration FLOAT`: Video duration in seconds (default: 5.0)
- `--fps INT`: Frames per second (default: 50)

**Policy Support**:
- Automatically loads PyTorch checkpoint files
- Supports A2C network architecture (256-128-128 hidden layers)
- Applies observation normalization if scaler is present in checkpoint
- Falls back to standing pose if no policy provided

**Output**:
- Generates both `.gif` and `.mp4` formats
- Saves to `output/` directory with descriptive filenames

#### `sim.py`
Interactive MuJoCo viewer for debugging and model inspection.

**Usage**:
```bash
# Load full scene (robot + floor + lighting)
python sim.py

# Load robot only (no environment)
python sim.py -h
```

**Features**:
- Hot reload: Close viewer window to reload model
- Displays model statistics (bodies, joints, DOF, actuators, meshes)
- Shows base height, COM position, and model type
- Useful for debugging MJCF files and testing poses

### Robot Description

#### `robot_description/`
Contains MuJoCo MJCF files and assets:

- `scene.xml`: Complete simulation scene with humanoid, floor, and lighting
  - Includes keyframes for standing poses
  - Configures camera positions and rendering settings

- `humanoid.xml`: Standalone humanoid robot definition
  - Joint configuration and limits
  - Actuator definitions
  - Mass and inertia properties

- `stls/`: STL mesh files for robot visualization
- `robot_meshes/`: Processed mesh files
- `utils/`: Utility scripts (e.g., MJCF to URDF conversion)

### Configuration

#### `pyproject.toml`
Modern Python package configuration file that defines:
- Project metadata (name, version, description, authors)
- Dependencies and version requirements
- Optional development dependencies (pytest, black, ruff)
- Build system configuration
- Tool configurations for linters and formatters

This file enables easy installation with `pip install -e .` and ensures consistent dependency management.

### Documentation

#### `docs/STARTING_POSE.md`
Guide for finding stable standing poses for the humanoid robot. Includes workflow for:
- Using dual terminal setup with `sim.py`
- Iteratively refining keyframes
- Tips for efficient pose tuning using binary search methodology

## Advanced Usage

### Custom Environment Parameters

You can modify environment parameters programmatically:

```python
from disturbance_env import HumanoidDisturbanceEnv

env = HumanoidDisturbanceEnv(
    xml_path="robot_description/scene.xml",
    frame_stack=3,                    # Number of observation frames to stack
    disturbance_force_max=25.0,       # Maximum disturbance force (N)
    disturbance_torque_max=10.0,      # Maximum disturbance torque (Nm)
    disturbance_prob=0.1,             # Probability of disturbance (0-1)
    action_scale=1.0,                 # Action scaling factor (radians)
    dt=0.02                           # Control timestep (seconds)
)

obs = env.reset()
for _ in range(1000):
    action = your_policy(obs)
    obs = env.step(action)
    frame = env.render()
```

### Policy Requirements

Trained policies should be saved as PyTorch checkpoint files (`.pth`) with the following structure:

```python
checkpoint = {
    'model': model.state_dict(),  # Required
    'scaler': {                    # Optional (for observation normalization)
        'mean': mean_values,
        'std': std_values
    }
}
```

Expected model architecture (A2C):
- Input: Observation dimension (varies by environment and frame stack)
- Hidden layers: [256, 128, 128] with ELU activation
- Output: Action dimension (number of actuators)

### Working with Different Robots

To use a different humanoid model:

1. Create/modify MJCF files in `robot_description/`
2. Ensure keyframe named `standing_pose` exists in the scene
3. Ensure torso body is named `torso`
4. Update observation/action dimensions if needed

## Troubleshooting

### Policy Loading Issues

If you encounter errors loading a policy:
- Ensure the `.pth` file contains a `'model'` key
- Verify observation dimensions match your environment configuration
- Check that action dimensions match the number of actuators

### Visualization Issues

If videos aren't generating:
- Ensure `output/` directory exists (created automatically)
- Check that opencv-python and imageio are installed
- Verify sufficient disk space for video files

### Simulation Instability

If the robot falls or behaves erratically:
- Review the standing pose in `scene.xml`
- Reduce action scale or disturbance parameters
- Check joint limits and control gains
- See `docs/STARTING_POSE.md` for pose tuning guidance

## Contributing

When adding new features:
1. Follow existing code structure and naming conventions
2. Update this README with new functionality
3. Test with both environments (disturbance and locomotion)
4. Document any new command-line arguments or parameters

## License

[Add your license information here]
