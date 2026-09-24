"""Convert full-speed qpos playback to 50 Hz original-model tracking states."""
from pathlib import Path
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp

BASE=Path(__file__).resolve().parent
OUT=BASE/'motions/calm91_contacts'
z=np.load(OUT/'legs_walk_playback.npz'); q=z['qpos']; fps=float(z['fps'])
t=np.arange(len(q))/fps; new=np.arange(int(np.floor(t[-1]*50))+1)/50
sample=np.column_stack([np.interp(new,t,q[:,i]) for i in range(q.shape[1])])
r=Slerp(t,Rotation.from_quat(q[:,[4,5,6,3]]))(new).as_quat()
sample[:,3:7]=r[:,[3,0,1,2]]
m=mujoco.MjModel.from_xml_path(str(BASE/'chrobot_16kg_actuated.xml'));d=mujoco.MjData(m)
order=json.loads((BASE/'tracking_body_order.json').read_text())
assert [m.body(i).name for i in range(1,m.nbody)]==order['body_names']
assert [m.joint(i).name for i in range(1,m.njnt)]==order['joint_names']
velocity=[]
for i in range(len(sample)):
    a=max(0,i-1);b=min(len(sample)-1,i+1);v=np.zeros(m.nv)
    mujoco.mj_differentiatePos(m,v,(b-a)/50,sample[a],sample[b]);velocity.append(v)
velocity=np.array(velocity)
pos=[];quat=[];lin=[];ang=[]
for q,v in zip(sample,velocity):
    d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d)
    pos.append(d.xpos[1:].copy());quat.append(d.xquat[1:].copy())
    body_vel=[]
    for b in range(1,m.nbody):
        bv=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,b,bv,0);body_vel.append(bv)
    body_vel=np.array(body_vel);ang.append(body_vel[:,:3]);lin.append(body_vel[:,3:])
output=OUT/'walking_reference_50hz.npz'
arrays=dict(fps=np.array(50),joint_pos=sample[:,7:],joint_vel=velocity[:,6:],
    body_pos_w=np.array(pos),body_quat_w=np.array(quat),body_lin_vel_w=np.array(lin),body_ang_vel_w=np.array(ang))
assert all(np.isfinite(a).all() for a in arrays.values())
np.savez(output,**arrays)
print(output,'frames',len(sample),'max_joint_speed',np.max(np.abs(velocity[:,6:])))
