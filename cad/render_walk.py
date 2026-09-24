"""Render reference playback and audit foot collision mesh heights."""
from pathlib import Path
import json
import argparse
import mujoco
import numpy as np
from PIL import Image

BASE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--motion-dir',type=Path,default=BASE/'motions')
ap.add_argument('--rollout',type=Path)
args=ap.parse_args()
OUT=args.motion_dir
source=OUT/'legs_walk_ik_candidate.npz'
if args.rollout:
    source=args.rollout
z=np.load(source); q=z['qpos'].copy(); fps=float(z['fps'])
if args.rollout:
    resets=np.flatnonzero(z['reset'])
    if len(resets):
        q=q[:max(1,int(resets[0]))]
spec=mujoco.MjSpec.from_file(str(BASE/'chrobot_16kg_actuated.xml'))
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,size=[5,5,.1],rgba=[.85,.85,.85,1])
spec.worldbody.add_light(pos=[1,1,3],dir=[-1,-1,-2])
m=spec.compile(); d=mujoco.MjData(m)
feet=[m.body(n).id for n in ('left_foot','right_foot')]
geoms=[[g for g in range(m.ngeom) if m.geom_bodyid[g]==b and m.geom_group[g]==3] for b in feet]
assert all(geoms)
def heights():
    h=[]
    for gs in geoms:
        vertices=[]
        for g in gs:
            mesh=m.geom_dataid[g]
            if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
                local=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*m.geom_size[g]
            else:
                assert mesh>=0
                a=m.mesh_vertadr[mesh]; n=m.mesh_vertnum[mesh]
                local=m.mesh_vert[a:a+n]
            vertices.append(local @ d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
        h.append(np.vstack(vertices)[:,2].min())
    return h
sole=[]
for v in q:
    d.qpos[:]=v; mujoco.mj_forward(m,d); sole.append(heights())
sole=np.array(sole)
# One constant world-height correction; do not independently move frames/feet.
lift=0.0 if args.rollout else -sole.min(); q[:,2]+=lift; sole+=lift
if not args.rollout:
    np.savez(OUT/'legs_walk_playback.npz',qpos=q,fps=fps)
audit={'fps':fps,'duration_seconds':len(q)/fps,'constant_ground_lift_m':float(lift),'sole_height_min_m':sole.min(0).tolist(),
 'sole_height_max_m':sole.max(0).tolist(),'ankle_limit_frame_fraction':(np.abs(q[:,[11,16]])>.599).mean(0).tolist(),
 'status':'Kinematic playback only; contact and actuator checks required before training.'}
if not args.rollout:
    (OUT/'playback_audit.json').write_text(json.dumps(audit,indent=2))
renderer=mujoco.Renderer(m,height=480,width=600)
cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
cam.distance=1.4; cam.azimuth=135; cam.elevation=-12
frames=[]
for v in q:
    d.qpos[:]=v; mujoco.mj_forward(m,d)
    cam.lookat[:]=d.body('hip').xpos+[ -.155587,-.029495,.3]
    renderer.update_scene(d,camera=cam)
    frames.append(Image.fromarray(renderer.render().copy()))
renderer.close()
stem=args.rollout.with_suffix('') if args.rollout else OUT/'legs_walk_playback'
frames[0].save(stem.with_suffix('.gif'),save_all=True,append_images=frames[1:],duration=round(1000/fps),loop=0)
frames[min(60,len(frames)-1)].save(stem.with_suffix('.png'))
print(json.dumps(audit,indent=2))
