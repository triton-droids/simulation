"""Create zero-mass anatomical landmarks for Holosoma without altering joints."""
from pathlib import Path
import json
import mujoco
import numpy as np

BASE = Path(__file__).resolve().parent
spec = mujoco.MjSpec.from_file(str(BASE / 'chrobot_16kg_actuated.xml'))
mapping = {'Hips': 'retarget_pelvis'}
landmarks = [('hip', 'retarget_pelvis', [-.155587, -.029495, .577401])]
for side in ('left', 'right'):
    for human, body, joint in [('Leg', f'{side}_leg2', f'{side}_hip2_joint'),
                              ('Shin', f'{side}_leg4', f'{side}_knee_joint'),
                              ('Foot', f'{side}_foot', f'{side}_ankle_joint')]:
        marker = f'retarget_{side}_{human.lower()}'
        mapping[f'{side.title()}{human}'] = marker
        landmarks.append((body, marker, spec.joint(joint).pos.copy()))
for body, marker, pos in landmarks:
    spec.body(body).add_body(name=marker, pos=pos)
original = mujoco.MjSpec.from_file(str(BASE / 'chrobot_16kg_actuated.xml')).compile()
model = spec.compile()
assert model.nq == original.nq and model.nv == original.nv
assert abs(model.body_mass.sum() - original.body_mass.sum()) < 1e-9
for item in spec.meshes:
    if item.file:
        item.file = str((BASE / spec.meshdir / item.file).resolve())
spec.meshdir = ''
(BASE / 'chrobot_holosoma_landmarks.xml').write_text(spec.to_xml())
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
positions = {human: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, marker)].tolist() for human, marker in mapping.items()}
assert positions['LeftLeg'][0] > positions['RightLeg'][0]
assert positions['LeftLeg'][2] > positions['LeftShin'][2] > positions['LeftFoot'][2]
assert positions['RightLeg'][2] > positions['RightShin'][2] > positions['RightFoot'][2]
(BASE / 'holosoma_lower_body_mapping.json').write_text(json.dumps({
    'status': 'landmark geometry validated; solver adapter not yet run',
    'source_format': 'kimodo', 'robot_dof': 10, 'joints_mapping': mapping,
    'foot_sticking_links': ['retarget_left_foot', 'retarget_right_foot'],
    'neutral_landmark_positions_m': positions,
    'notes': ['No upper-body targets.', 'Marker bodies have no mass or joints.',
              'Hip positions are derived from the existing XML, not new Fusion calibration.',
              'Do not use this extra-body model to generate training body arrays; use the original model.']}, indent=2)+'\n')
print('Prepared seven landmarks; preserved 10 joints and 16 kg:', positions)
