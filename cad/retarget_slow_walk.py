"""Retarget a Kimodo candidate using measured XML leg segments, then audit FK."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import mujoco
from holosoma_leg_config import make_config, HOLOSOMA
sys.path.insert(0, str(HOLOSOMA.parent))
from holosoma_retargeting.examples.robot_retarget import create_task_constants, build_retargeter_kwargs_from_config, create_ground_points
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter

BASE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--frames', type=int, default=180)
ap.add_argument('--motion', type=Path, default=BASE/'motions/slow_walk_seed42.npz')
ap.add_argument('--output-dir', type=Path, default=BASE/'motions')
ap.add_argument('--prepare-only', action='store_true')
args = ap.parse_args()
OUT=args.output_dir
OUT.mkdir(parents=True,exist_ok=True)
source = np.load(args.motion)
# Use the exact SOMA joint ordering written into the Kimodo skeleton definition.
names = json.loads((BASE / 'motions/soma_skeleton.json').read_text())['joint_names']
idx = {n: i for i, n in enumerate(names)}
p = source['posed_joints'][:args.frames].copy()[..., [0, 2, 1]]
# Point-target convention: +X is the XML's named left, +Y is its forward
# knee-flexion convention, +Z is up. This reflects the source sagittal axis
# relative to the usual Y-up -> Z-up rotation; no source joint angles are used.
neutral = json.loads((BASE / 'holosoma_lower_body_mapping.json').read_text())['neutral_landmark_positions_m']
neutral = {n: np.array(v) for n, v in neutral.items()}
robot_lengths = {}
human_lengths = {}
for side in ('Left', 'Right'):
    a, b, c = (side + x for x in ('Leg', 'Shin', 'Foot'))
    robot_lengths[side] = [np.linalg.norm(neutral[b]-neutral[a]), np.linalg.norm(neutral[c]-neutral[b])]
    human_lengths[side] = [np.linalg.norm(p[:,idx[b]]-p[:,idx[a]],axis=-1).mean(), np.linalg.norm(p[:,idx[c]]-p[:,idx[b]],axis=-1).mean()]
scale = np.mean([sum(robot_lengths[s])/sum(human_lengths[s]) for s in robot_lengths])
target = p * scale
target[..., :2] -= target[0,idx['Hips'],:2].copy()
# Preserve source leg directions while enforcing robot thigh/shin lengths and hip width.
# Rotate pelvis-local robot hip offsets with the source pelvis orientation.
R = source['global_rot_mats'][:args.frames,idx['Hips']]
C = np.array([[1,0,0],[0,0,1],[0,1,0]])
R = C @ R @ C.T
for side in ('Left', 'Right'):
    a, b, c = (side+x for x in ('Leg','Shin','Foot'))
    target[:,idx[a]] = target[:,idx['Hips']] + np.einsum('tij,j->ti',R,neutral[a]-neutral['Hips'])
    for parent, child, length in ((a,b,robot_lengths[side][0]),(b,c,robot_lengths[side][1])):
        direction = p[:,idx[child]]-p[:,idx[parent]]
        target[:,idx[child]] = target[:,idx[parent]] + direction / np.linalg.norm(direction,axis=-1,keepdims=True)*length
# Ground level from source toe minima; ankle targets keep their source contact clearance.
toe_min = target[:,[idx['LeftToeBase'],idx['RightToeBase']],2].min()
target[...,2] -= toe_min
cfg = make_config(BASE / 'motions','slow_walk',BASE / 'motions',names)
constants = create_task_constants(cfg.robot_config,cfg.motion_data_config,cfg.task_config,'robot_only')
retargeter = InteractionMeshRetargeter(**build_retargeter_kwargs_from_config(cfg.retargeter,constants,None,'robot_only'))
q_init = retargeter.robot_model.qpos0.copy()
mujoco.mj_forward(retargeter.robot_model,retargeter.robot_data)
q_init[:3] += target[0,idx['Hips']] - retargeter.robot_data.body('retarget_pelvis').xpos
objects = np.tile([0,0,0,1,0,0,0],(len(target),1))
# Upstream unconditionally writes augmented object coordinates into the last
# seven qpos entries, even for ground-only models. Preserve those robot joints
# here; ground mode never transforms an augmented object and debug is disabled.
assert retargeter.object_name == 'ground' and retargeter.nq == 17 and not retargeter.debug
ground_augmented = np.tile(q_init[-7:],(len(target),1))
ground = create_ground_points(cfg.task_config.ground_range,cfg.task_config.ground_range,cfg.task_config.ground_size)
contacts = source['foot_contacts'][:args.frames]
sticking = [{'LeftToeBase':bool(row[:3].any()),'RightToeBase':bool(row[3:].any())} for row in contacts]
output = OUT / f'legs_retarget_{len(target)}frames.npz'
np.savez(OUT/'robot_scaled_targets.npz',positions=target,joint_names=names,fps=30,foot_contacts=contacts)
(OUT/'retarget_scaling.json').write_text(json.dumps({'scale':float(scale),'robot_segment_lengths_m':{s:np.array(v).tolist() for s,v in robot_lengths.items()},'human_segment_lengths_m':{s:np.array(v).tolist() for s,v in human_lengths.items()},'note':'XML-based segment and hip-width fit; no CAD joint recalibration.'},indent=2))
if args.prepare_only:
    print('Prepared targets',OUT)
    sys.exit(0)
retargeter.retarget_motion(human_joint_motions=target, object_poses=objects,
    object_poses_augmented=ground_augmented, object_points_local_demo=ground,object_points_local=ground,
    foot_sticking_sequences=sticking,q_a_init=q_init,q_nominal_list=None,original=True,dest_res_path=str(output))
print('Saved',output)
