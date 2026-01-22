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

from policy_exporter import save_as_gif, save_as_mp4

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
parser.add_argument('--debug', action='store_true',
                    help='Print detailed debug information')
args = parser.parse_args()

# Import appropriate environment (default to disturbance)
if args.locomotion:
    from envs.locomotion_env import HumanoidLocomotionEnv as Env
    env_name = "Locomotion"
    print("Using locomotion_env.py")
else:
    from envs.disturbance_env import HumanoidDisturbanceEnv as Env
    env_name = "Disturbance"
    print("Using disturbance_env.py")

device = "cpu"

# Create environment to get dimensions (60Hz control frequency)
env = Env(xml_path="robot_description/scene.xml")
obs = env.reset()

# Debugging - Check orientation frames
print("\n=== ORIENTATION CHECK ===")
torso_quat = env.data.xquat[env._torso_body_id]
gravity_world = np.array([0, 0, -1.0])
up_b = env._rotate_vector(gravity_world, torso_quat, inverse=True)
up_cmd = env._rotate_xy(up_b, env._cmd_yaw_cos, env._cmd_yaw_sin)

print(f"Up in body frame: {up_b}")
print(f"Up in command frame: {up_cmd}")
print(f"Expected when standing: body ≈ [0, 0, -1], command ≈ [0, 0, -1]")

obs_dim = obs.shape[0]
action_dim = env._nu

print(f"\nObservation dimension: {obs_dim}")
print(f"Action dimension: {action_dim}")

# === NEW: DETAILED OBSERVATION CHECK ===
if args.debug:
    print("\n=== RESET OBSERVATION BREAKDOWN ===")
    print(f"Obs shape: {obs.shape}")
    print(f"Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"\nFirst frame (indices 0-{env._single_frame_size-1}):")
    idx = 0
    print(f"  Height [0]: {obs[idx]:.3f}")
    idx += 1
    print(f"  Lin vel [1-3]: {obs[idx:idx+3]}")
    idx += 3
    print(f"  Ang vel scaled [4-6]: {obs[idx:idx+3]}")
    idx += 3
    print(f"  Up vec [7-9]: {obs[idx:idx+3]}")
    idx += 3
    print(f"  Commands [10-12]: {obs[idx:idx+3]}")
    idx += 3
    print(f"  Joint pos scaled [13-22]: {obs[idx:idx+action_dim]}")
    idx += action_dim
    print(f"  Joint vel scaled [23-32]: {obs[idx:idx+action_dim]}")
    idx += action_dim
    print(f"  Prev actions [33-42]: {obs[idx:idx+action_dim]}")
    
    # Check for issues
    if np.any(np.isnan(obs)):
        print("\n⚠️  WARNING: NaN values detected in observation!")
    if np.any(np.abs(obs) > 50):
        print("\n⚠️  WARNING: Extremely large values in observation!")
    clip_count = np.sum((obs == 10.0) | (obs == -10.0))
    if clip_count > 0:
        print(f"\n⚠️  WARNING: {clip_count} values hitting clip limits (±10)")

# Load trained policy if provided
use_trained_policy = args.policy is not None

# Setup policy based on mode
if use_trained_policy: 
    policy = torch.jit.load(args.policy, map_location="cpu")
    policy.eval()
    print(f"\nLoaded policy from: {args.policy}")
    
    # === NEW: TEST POLICY OUTPUT ===
    if args.debug:
        print("\n=== POLICY OUTPUT TEST ===")
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_tensor = policy(obs_tensor)
            action = action_tensor.squeeze(0).numpy()
            
            print(f"Action shape: {action.shape}")
            print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
            print(f"Actions: {action}")
            
            # Check for suspicious patterns
            if np.all(action == action[0]):
                print("⚠️  WARNING: All actions are identical!")
            if np.all(np.abs(action) > 0.99):
                print("⚠️  WARNING: All actions are saturated at ±1!")
            if np.any(np.isnan(action)):
                print("⚠️  WARNING: NaN in policy output!")
else:
    # Use standing pose policy
    standing_joint_positions = env._standing_qpos[env._q_joint_start:]
    standing_actions = standing_joint_positions / env._action_scale  # ISSUE: Should this include per-joint scaling?
    print(f"\nUsing standing policy")
    print(f"Standing joint positions: {standing_joint_positions}")
    print(f"Standing actions: {standing_actions}")
    policy = None

# Calculate number of steps
num_steps = int(args.duration * args.fps)
print(f"\nRecording {args.duration}s at {args.fps} FPS ({num_steps} steps)")

# Run simulation with policy
frames = []
if use_trained_policy:
    # Use trained neural network policy
    with torch.no_grad():
        for step in range(num_steps):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            # Get action from policy
            action_tensor = policy(obs_tensor)
            action = action_tensor.squeeze(0).numpy()

            # Clip action to valid range
            action = np.clip(action, -1.0, 1.0)

            # === NEW: Debug first few steps ===
            if args.debug and step < 3:
                print(f"\n--- Step {step} ---")
                print(f"Action: {action}")
                print(f"Height before step: {env.data.xpos[env._torso_body_id, 2]:.3f}")

            # Step environment
            obs = env.step(action)
            
            if args.debug and step < 3:
                print(f"Height after step: {env.data.xpos[env._torso_body_id, 2]:.3f}")
                print(f"New obs[0] (height): {obs[0]:.3f}")
            
            frames.append(env.render())

            if (step + 1) % args.fps == 0:
                if not args.debug:  # Don't print this if already printing debug info
                    print(f"  {(step + 1) / args.fps:.1f}s - Height: {env.data.xpos[env._torso_body_id, 2]:.3f}")
else:
    # Use standing pose policy
    for step in range(num_steps):
        obs = env.step(standing_actions)
        frames.append(env.render())

        if (step + 1) % args.fps == 0:
            print(f"  {(step + 1) / args.fps:.1f}s - Height: {env.data.xpos[env._torso_body_id, 2]:.3f}")

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