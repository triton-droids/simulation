"""Bounded analytic leg-target kinematics; never a locomotion validation."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def load_model():
    """Use the same pinned, local asset resolution as the training adapter."""
    import mujoco
    from source.locomotion.unitree_g1.playground_source import (
        resolve_playground_source, load_playground_g1_modules,
    )
    from source.robots.unitree_g1 import UnitreeG1Model
    source = resolve_playground_source(fetch=False)
    robot = UnitreeG1Model(fetch=False)
    mjx_env, _, _ = load_playground_g1_modules(source)
    mjx_env.MENAGERIE_PATH = mjx_env.epath.Path(robot.resolution.scene_path.parents[1])
    from mujoco_playground._src.locomotion.g1.base import get_assets
    xml = source.root/'mujoco_playground/_src/locomotion/g1/xmls/scene_mjx_feetonly_flat_terrain.xml'
    return mujoco.MjModel.from_xml_string(xml.read_text(), assets=get_assets()), xml


def pitch_offsets(phase, moving=True):
    """Fixed symmetric flexion primitive in radians; no fitted parameters."""
    lift = np.maximum(np.cos(np.asarray(phase)), 0.) if moving else np.zeros_like(np.asarray(phase))
    return lift[..., None] * np.array([-.15, .30, -.15])


def main():
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model, xml = load_model()
    data = mujoco.MjData(model)
    q0 = model.keyframe('knees_bent').qpos.copy()
    sites = [model.site(side+'_foot').id for side in ('left','right')]
    joints = [[model.joint(side+'_'+name+'_joint').id for name in ('hip_pitch','knee','ankle_pitch')] for side in ('left','right')]
    addresses = [[int(model.jnt_qposadr[j]) for j in leg] for leg in joints]
    data.qpos[:] = q0
    mujoco.mj_forward(model,data)
    baseline = data.site_xpos[sites].copy()
    rows=[]
    for phase in np.linspace(0,2*np.pi,32,endpoint=False):
        offsets=pitch_offsets([phase,phase+np.pi])
        data.qpos[:]=q0
        for adr,delta in zip(addresses,offsets): data.qpos[adr]+=delta
        mujoco.mj_forward(model,data)
        values=np.array([data.qpos[a] for a in addresses])
        lower=np.array([-1.57,0.,-.4]);upper=np.array([1.57,1.57,.4])
        contacts=[{'geom1':model.geom(int(c.geom1)).name,'geom2':model.geom(int(c.geom2)).name,'distance':float(c.dist)} for c in data.contact if c.dist < -1e-4 and int(c.geom1)!=model.geom('floor').id and int(c.geom2)!=model.geom('floor').id]
        rows.append({'phase':float(phase),'joint_targets':values.tolist(),'foot_delta':(data.site_xpos[sites]-baseline).tolist(),'restricted_limits_ok':bool(np.all(values>=lower)&np.all(values<=upper)),'penetrating_nonfloor_contacts':contacts})
    peak=np.max(np.array([r['foot_delta'] for r in rows])[:,:,2],axis=0)
    report={'kind':'fixed_base_kinematic_probe','full_validation':False,'training_steps':0,'physics_rollout':False,'xml':str(xml),'joint_names':[[model.joint(j).name for j in leg] for leg in joints],'qpos_addresses':addresses,'peak_foot_lift_m':peak.tolist(),'prerequisite_pass':bool(np.all(peak>=.01) and all(r['restricted_limits_ok'] and not r['penetrating_nonfloor_contacts'] for r in rows)),'rows':rows}
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))

if __name__=='__main__': main()
