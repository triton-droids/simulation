"""
Validation environment video generator (robot attempts to maintain starting pose)
Usage:
    python visualize.py           # Uses locomotion_env.py (default)
    python visualize.py -l        # Uses locomotion_env.py
    python visualize.py -d        # Uses disturbance_env.py
    python visualize.py --help    # Show help
"""

import argparse
import numpy as np
from PIL import Image
import cv2

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate validation videos for humanoid environments')
group = parser.add_mutually_exclusive_group()
group.add_argument('-l', '--locomotion', action='store_true', 
                   help='Use locomotion_env.py (default)')
group.add_argument('-d', '--disturbance', action='store_true', 
                   help='Use disturbance_env.py')
parser.add_argument('--duration', type=float, default=5.0,
                    help='Video duration in seconds (default: 5.0)')
parser.add_argument('--fps', type=int, default=50,
                    help='Frames per second (default: 50)')
args = parser.parse_args()

# Import appropriate environment
if args.disturbance:
    from disturbance_env import HumanoidDisturbanceEnv as Env
    env_name = "Disturbance"
    print("Using disturbance_env.py")
else:
    from locomotion_env import HumanoidLocomotionEnv as Env
    env_name = "Locomotion"
    print("Using locomotion_env.py")

# Create environment
env = Env(xml_path="robot_description/scene.xml")

# Reset
obs = env.reset()
print(f"✓ Environment initialized. Obs shape: {obs.shape}")

# Get the standing pose actions (normalized)
standing_joint_positions = env._standing_qpos[env._q_joint_start:]
standing_actions = standing_joint_positions / env._action_scale

print(f"Standing joint positions: {standing_joint_positions}")
print(f"Standing actions (normalized): {standing_actions}")

# Calculate number of steps
num_steps = int(args.duration * args.fps)
print(f"Recording {args.duration}s at {args.fps} FPS ({num_steps} steps)")

# Run simulation
frames = []
for step in range(num_steps):
    obs = env.step(standing_actions)
    frames.append(env.render())
    
    if (step + 1) % args.fps == 0:
        print(f"  {(step + 1) / args.fps:.1f}s")

print(f"✓ Collected {len(frames)} frames")

# Save as GIF
gif_filename = f"output/test_{env_name.lower()}.gif"
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
mp4_filename = f"output/test_{env_name.lower()}.mp4"
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
print(f"Duration: {args.duration}s @ {args.fps} FPS")
print(f"GIF: {gif_filename}")
print(f"MP4: {mp4_filename}")
print(f"{'='*60}")