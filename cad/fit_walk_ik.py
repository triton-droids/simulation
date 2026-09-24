"""Direct bounded position IK diagnostic after Holosoma's poor lower-body fit."""
from pathlib import Path
import json
import argparse
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

BASE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--motion-dir',type=Path,default=BASE/'motions')
ap.add_argument('--contact-aware',action='store_true')
args=ap.parse_args()
OUT=args.motion_dir
z=np.load(OUT/'robot_scaled_targets.npz')
targets=z['positions']; names=z['joint_names'].tolist()
mapping=json.loads((BASE/'holosoma_lower_body_mapping.json').read_text())['joints_mapping']
model=mujoco.MjModel.from_xml_path(str(BASE/'chrobot_holosoma_landmarks.xml'))
data=mujoco.MjData(model)
ids=[model.body(n).id for n in mapping.values()]
indices=[names.index(n) for n in mapping]
q=model.qpos0.copy(); data.qpos[:]=q; mujoco.mj_forward(model,data)
q[:3]+=targets[0,names.index('Hips')]-data.xpos[ids[0]]
x=np.r_[q[:3],np.zeros(3),q[7:]]
low=np.r_[[-10]*3,[-np.pi]*3,model.jnt_range[1:,0]]
high=np.r_[[10]*3,[np.pi]*3,model.jnt_range[1:,1]]
if args.contact_aware:
    low[[10,15]]=-.55
    high[[10,15]]=.55
    targets=targets.copy()
    targets[:,:,2]-=.04337423529321807
    contacts=z['foot_contacts'].reshape(len(targets),2,3).any(axis=2)
    for side,k in enumerate((3,6)):
        anchor=None
        for t in range(len(targets)):
            if contacts[t,side]:
                if t==0 or not contacts[t-1,side]:
                    anchor=targets[t,indices[k],:2].copy()
                targets[t,indices[k],:2]=anchor
    foot_geoms=[model.geom(n).id for n in ('left_foot_collision_box','right_foot_collision_box')]
    bottom=[np.array([[a,b,-1] for a in (-1,1) for b in (-1,1)])*model.geom_size[g] for g in foot_geoms]
x=np.clip(x,low+1e-7,high-1e-7)
poses=[]; errors=[]
def forward(v):
    quat=Rotation.from_rotvec(v[3:6]).as_quat()
    data.qpos[:]=np.r_[v[:3],quat[[3,0,1,2]],v[6:]]
    mujoco.mj_forward(model,data)
    return data.xpos[ids].copy()
for t,target in enumerate(targets[:,indices]):
    previous=x.copy()
    def residual(v):
        actual=forward(v)
        normals=np.array([data.body(n).xmat.reshape(3,3)[:,2] for n in ('left_foot','right_foot')])
        terms=[(actual-target).ravel()*10,(normals-[0,0,1]).ravel()*(3 if args.contact_aware else .3),(v[3:]-previous[3:])*.05]
        if args.contact_aware:
            for side,g in enumerate(foot_geoms):
                world=bottom[side]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
                terms.append((world[:,2] if contacts[t,side] else np.minimum(world[:,2],0))*30)
        return np.concatenate(terms)
    result=least_squares(residual,x,bounds=(low,high),max_nfev=100,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    x=result.x; actual=forward(x)
    poses.append(data.qpos.copy()); errors.append(np.linalg.norm(actual-target,axis=1))
errors=np.array(errors); poses=np.array(poses)
output=OUT/'legs_walk_ik_candidate.npz'
np.savez(output,qpos=poses,fps=30,errors=errors,human_joints=targets)
report={'contact_aware':args.contact_aware,'status':'Diagnostic kinematic fit; contact quality not yet validated for RL',
    'method':'bounded scipy least_squares over free-base pose and ten joints',
    'mean_position_error_m':dict(zip(mapping,errors.mean(0).tolist())),
    'max_position_error_m':float(errors.max()),'joint_min_rad':poses[:,7:].min(0).tolist(),
    'joint_max_rad':poses[:,7:].max(0).tolist()}
output.with_suffix('.audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
