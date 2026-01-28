"""
Validation environment video generator with trained policy
Usage:
    python visualize_2.py                    # Uses disturbance_env.py with standing pose policy (default)
    python visualize_2.py -l                 # Uses locomotion_env.py with standing pose policy
    python visualize_2.py -d                 # Uses disturbance_env.py with standing pose policy
    python visualize_2.py --policy PATH      # Uses trained policy from PATH
    python visualize_2.py --help             # Show help
"""

import argparse
import numpy as np
import torch
import os

from debug import format_obs_detailed, policies, height_policies
from policy_exporter import save_as_gif, save_as_mp4

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate validation videos for humanoid environments')
parser.add_argument('--duration', type=float, default=5.0,
                    help='Video duration in seconds (default: 5.0)')
parser.add_argument('--fps', type=int, default=50,
                    help='Frames per second (default: 50)')
parser.add_argument('--policy', type=str, default=None,
                    help='Path to policy .pth file (default: None, uses standing pose policy)')
args = parser.parse_args()

include_height = args.policy in height_policies if args.policy is not None else False



# Import appropriate environment (default to disturbance)
if args.policy:
    env_name = policies[args.policy]
    if env_name == "locomotion_env":
        from envs.locomotion_env import HumanoidLocomotionEnv as Env
        env_name = "Locomotion"
        print("Using locomotion_env.py")
    else:
        from envs.disturbance_env import HumanoidDisturbanceEnv as Env
        env_name = "Disturbance"
        print("Using disturbance_env.py")
else:
    from envs.disturbance_env import HumanoidDisturbanceEnv as Env
    env_name = "Disturbance"
    print("Using disturbance_env.py")

# Create environment to get dimensions (60Hz control frequency)
env = Env(xml_path="robot_description/scene.xml", include_height=include_height)
obs = env.reset()

#Debugging
torso_quat = env.data.xquat[env._torso_body_id]
up_world = np.array([0, 0, 1.0])
up_b = env._rotate_vector(up_world, torso_quat, inverse=True)
up_cmd = env._rotate_xy(up_b, env._cmd_yaw_cos, env._cmd_yaw_sin)

print(f"Up in body frame: {up_b}")
print(f"Up in command frame: {up_cmd}")

obs_dim = obs.shape[0]
action_dim = env._nu

print(f"\nObservation dimension: {obs_dim}")
print(f"Action dimension: {action_dim}")


# Load trained policy if provided
use_trained_policy = args.policy is not None

# Setup policy based on mode
if use_trained_policy: 
    policy = torch.jit.load(args.policy, map_location = "cpu")
    policy.eval()
    print("Using trained policy")

else:
    # Use standing pose policy
    standing_joint_positions = env._standing_qpos[env._q_joint_start:]
    standing_actions = standing_joint_positions / env._action_scale
    print(f"Using standing policy")
    policy = None

# Calculate number of steps
num_steps = int(args.duration * args.fps)
print(f"Recording {args.duration}s at {args.fps} FPS ({num_steps} steps)")

# Run simulation with policy
frames = []
if use_trained_policy:
    # Use trained neural network policy
    with torch.no_grad():  # No gradient computation needed for inference
        for step in range(num_steps):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension

            # Get action from policy
            action_tensor = policy(obs_tensor)


            if step == 0:
                print(f"First action from trained policy: {action_tensor}")

            action = action_tensor.squeeze(0).numpy()  # Remove batch dimension, convert to numpy

            # Clip action to valid range
            action = np.clip(action, -1.0, 1.0)

            # Step environment
            obs = env.step(action)
            #print(obs)
            frames.append(env.render())

            if (step + 1) % args.fps == 0:
                print(format_obs_detailed(obs, env, include_height))
                print(f"  {(step + 1) / args.fps:.1f}s")
else:
    # Use standing pose policy
    for step in range(num_steps):
        obs = env.step(standing_actions)
        frames.append(env.render())

        if (step + 1) % args.fps == 0:
            print(_format_obs_detailed(obs, env))
            print(f"  {(step + 1) / args.fps:.1f}s")

print(f"✓ Collected {len(frames)} frames")

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Generate filename suffix based on policy type
policy_suffix = "trained" if use_trained_policy else "standing"

# Save as GIF
gif_filename = f"output/test_{env_name.lower()}_{policy_suffix}.gif"
save_as_gif(frames, gif_filename, args.fps)

# Convert to MP4
mp4_filename = f"output/test_{env_name.lower()}_{policy_suffix}.mp4"
save_as_mp4(frames, mp4_filename, args.fps)

print(f"\n{'='*60}")
print(f"Environment: {env_name}")
print(f"Policy: {args.policy if use_trained_policy else 'Standing pose'}")
print(f"Duration: {args.duration}s @ {args.fps} FPS")
print(f"GIF: {gif_filename}")
print(f"MP4: {mp4_filename}")
print(f"{'='*60}")
