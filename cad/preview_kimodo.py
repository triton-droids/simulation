"""Create a simple, root-following two-view preview of a Kimodo SOMA clip."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from kimodo.skeleton.registry import build_skeleton

p = argparse.ArgumentParser()
p.add_argument('motion', type=Path)
args = p.parse_args()
z = np.load(args.motion)
joints = z['posed_joints']
skeleton = build_skeleton(joints.shape[1])
names = skeleton.bone_order_names
parents = skeleton.joint_parents.tolist()
frames = []
for t, pose in enumerate(joints):
    image = Image.new('RGB', (900, 600), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((20, 16), f'Kimodo human reference | {t / 30:.2f}s | not robot playback', fill='black')
    for center, axis, label in [(230, 0, 'Front'), (670, 2, 'Side')]:
        draw.text((center - 20, 50), label, fill='black')
        draw.line((center - 200, 550, center + 200, 550), fill='#bbbbbb', width=2)
        points = np.column_stack((center + (pose[:, axis] - pose[0, axis]) * 260, 550 - pose[:, 1] * 260))
        for i, parent in enumerate(parents):
            if parent < 0 or any(word in names[i] for word in ('Hand', 'Eye', 'Jaw')):
                continue
            color = '#2171b5' if names[i].startswith('Left') else '#d95f0e' if names[i].startswith('Right') else '#555555'
            draw.line(tuple(points[parent]) + tuple(points[i]), fill=color, width=4)
    frames.append(image)
output = args.motion.with_suffix('.gif')
frames[0].save(output, save_all=True, append_images=frames[1:], duration=33, loop=0)
root = z['root_positions']
report = {'frames': len(joints), 'fps': 30, 'duration_seconds': len(joints) / 30,
          'finite': bool(np.isfinite(joints).all()),
          'root_displacement_m': (root[-1] - root[0]).tolist(),
          'horizontal_mean_speed_m_s': float(np.linalg.norm(np.diff(root[:, [0, 2]], axis=0), axis=1).sum() / ((len(root) - 1) / 30)),
          'foot_contact_fractions': z['foot_contacts'].mean(0).tolist(),
          'status': 'Generated human candidate; encoder warnings and motion quality require review before training.'}
args.motion.with_suffix('.audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(output)
print(json.dumps(report, indent=2))
