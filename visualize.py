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
from PIL import Image
import cv2
import torch
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate validation videos for humanoid environments')
group = parser.add_mutually_exclusive_group()
group.add_argument('-l', '--locomotion', action='store_true',
                   help='Use locomotion_env.py')
group.add_argument('-d', '--disturbance', action='store_true',
                   help='Use disturbance_env.py (default)')
parser.add_argument('--duration', type=float, default=5.0,
                    help='Video duration in seconds (default: 5.0)')
parser.add_argument('--fps', type=int, default=50,
                    help='Frames per second (default: 50)')
parser.add_argument('--policy', type=str, default=None,
                    help='Path to policy .pth file (default: None, uses standing pose policy)')
args = parser.parse_args()

# Import appropriate environment (default to disturbance)
if args.locomotion:
    from locomotion_env import HumanoidLocomotionEnv as Env
    env_name = "Locomotion"
    print("Using locomotion_env.py")
else:
    from disturbance_env import HumanoidDisturbanceEnv as Env
    env_name = "Disturbance"
    print("Using disturbance_env.py")

# Load trained policy if provided
use_trained_policy = args.policy is not None

if use_trained_policy:
    print(f"\nLoading policy from {args.policy}...")
    if not os.path.exists(args.policy):
        raise FileNotFoundError(f"Policy file not found: {args.policy}")

    # Load with weights_only=False for compatibility
    policy_data = torch.load(args.policy, map_location='cpu', weights_only=False)
    print(f"✓ Policy loaded successfully")
    print(f"Policy keys: {list(policy_data.keys())}")

    # Extract the actual model state dict
    if 'model' in policy_data:
        state_dict = policy_data['model']
        print("✓ Extracted 'model' from checkpoint")
        print(f"State dict type: {type(state_dict)}")

        # Inspect the architecture from state dict
        if isinstance(state_dict, dict):
            print(f"\nModel layers:")
            for key in state_dict.keys():
                print(f"  {key}: {state_dict[key].shape}")
    else:
        raise KeyError("Could not find 'model' key in checkpoint")
else:
    print("\nNo policy provided - will use standing pose policy")

# Create environment to get dimensions
env = Env(xml_path="robot_description/scene.xml")
obs = env.reset()
obs_dim = obs.shape[0]
action_dim = env._nu

print(f"\nObservation dimension: {obs_dim}")
print(f"Action dimension: {action_dim}")

# Setup policy based on mode
if use_trained_policy:
    # Create model architecture matching the checkpoint
    import torch.nn as nn

    class A2CNetwork(nn.Module):
        """A2C network matching the trained model structure"""
        def __init__(self, obs_dim, action_dim):
            super().__init__()

            # Actor MLP (policy network)
            self.actor_mlp = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU()
            )

            # Mean action output
            self.mu = nn.Linear(128, action_dim)

            # Learned standard deviation
            self.sigma = nn.Parameter(torch.zeros(action_dim))

            # Value head (critic)
            self.value = nn.Linear(128, 1)

        def forward(self, x):
            features = self.actor_mlp(x)
            mu = self.mu(features)
            return mu

    class ValueMeanStd(nn.Module):
        """Running mean/std normalization for values"""
        def __init__(self):
            super().__init__()
            self.register_buffer('running_mean', torch.zeros(1))
            self.register_buffer('running_var', torch.ones(1))
            self.register_buffer('count', torch.zeros([], dtype=torch.long))

    class ActorCriticModel(nn.Module):
        """Full model wrapper"""
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.a2c_network = A2CNetwork(obs_dim, action_dim)
            self.value_mean_std = ValueMeanStd()

        def forward(self, x):
            return self.a2c_network(x)

    # Create model instance
    policy = ActorCriticModel(obs_dim, action_dim)

    # Load state dict
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"✓ Policy loaded and ready for inference")

    # Check if there's a scaler for observations
    scaler = policy_data.get('scaler', None)
    if scaler is not None:
        print(f"✓ Found observation scaler")
        # Scaler might have mean and std
        if hasattr(scaler, 'mean') and hasattr(scaler, 'std'):
            print(f"  Scaler mean shape: {scaler.mean.shape}")
            print(f"  Scaler std shape: {scaler.std.shape}")
else:
    # Use standing pose policy
    standing_joint_positions = env._standing_qpos[env._q_joint_start:]
    standing_actions = standing_joint_positions / env._action_scale
    print(f"\nStanding joint positions: {standing_joint_positions}")
    print(f"Standing actions (normalized): {standing_actions}")
    print(f"✓ Standing pose policy ready")
    policy = None
    scaler = None

# Reset environment
obs = env.reset()
print(f"\n✓ Environment initialized. Obs shape: {obs.shape}")

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

            # Apply scaler if available
            if scaler is not None and hasattr(scaler, 'mean') and hasattr(scaler, 'std'):
                obs_tensor = (obs_tensor - scaler.mean) / (scaler.std + 1e-8)

            # Get action from policy
            action_tensor = policy(obs_tensor)
            action = action_tensor.squeeze(0).numpy()  # Remove batch dimension, convert to numpy

            # Clip action to valid range
            action = np.clip(action, -1.0, 1.0)

            # Step environment
            obs = env.step(action)
            frames.append(env.render())

            if (step + 1) % args.fps == 0:
                print(f"  {(step + 1) / args.fps:.1f}s")
else:
    # Use standing pose policy
    for step in range(num_steps):
        obs = env.step(standing_actions)
        frames.append(env.render())

        if (step + 1) % args.fps == 0:
            print(f"  {(step + 1) / args.fps:.1f}s")

print(f"✓ Collected {len(frames)} frames")

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Generate filename suffix based on policy type
policy_suffix = "trained" if use_trained_policy else "standing"

# Save as GIF
gif_filename = f"output/test_{env_name.lower()}_{policy_suffix}.gif"
print(f"Saving {gif_filename}...")
pil_frames = [Image.fromarray(frame) for frame in frames]
pil_frames[0].save(
    gif_filename,
    save_all=True,
    append_images=pil_frames[1:],
    duration=int(1000 / args.fps),  # milliseconds per frame
    loop=0
)
print(f"✓ Saved {gif_filename}")

# Convert to MP4
mp4_filename = f"output/test_{env_name.lower()}_{policy_suffix}.mp4"
print(f"Converting to {mp4_filename}...")
height, width = frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(mp4_filename, fourcc, args.fps, (width, height))

for frame in frames:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)

out.release()
print(f"✓ Saved {mp4_filename}")

print(f"\n{'='*60}")
print(f"Environment: {env_name}")
print(f"Policy: {args.policy if use_trained_policy else 'Standing pose'}")
print(f"Duration: {args.duration}s @ {args.fps} FPS")
print(f"GIF: {gif_filename}")
print(f"MP4: {mp4_filename}")
print(f"{'='*60}")