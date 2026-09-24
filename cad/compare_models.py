"""Visual comparison only: original articulated model and assembled Fusion STL."""
from pathlib import Path
import mujoco
import numpy as np
import trimesh
import viser
from mjlab.entity import EntityCfg
from mjviser import Viewer
base=Path(__file__).resolve().parent
mesh=trimesh.load(base.parent/'Lower_Body_Reassembled_No_3d.stl',process=False)
# Keep every original triangle and its winding; weld exact duplicate vertices.
# Sorting triangle indices destroys winding and produces incorrect shading.
v,inverse=np.unique(np.asarray(mesh.vertices),axis=0,return_inverse=True)
v=v*.001
f=inverse[mesh.faces]
v[:,0]+= .7-(mesh.bounds[:,0].mean()*.001)
v[:,2]-=mesh.bounds[0,2]*.001+.027
model=EntityCfg(spec_fn=lambda:mujoco.MjSpec.from_file(str(base/'chrobot_16kg_candidate.xml'))).build().compile()
data=mujoco.MjData(model)
mujoco.mj_forward(model,data)
server=viser.ViserServer(host='127.0.0.1',port=8080)
server.scene.add_mesh_simple('/fusion',vertices=v,faces=f,color=(70,155,190),side='double')
server.scene.add_label('/old_label','Existing MJCF — 16 kg candidate',position=(0,0,.85))
server.scene.add_label('/cad_label','Fusion STL — static geometry reference',position=(.7,0,1.3))
@server.on_client_connect
def camera(client):
 client.camera.look_at=(.35,0,.55)
 client.camera.position=(1.7,-2.4,1.25)
viewer=Viewer(model,data,server=server)
viewer._paused=True
print('Comparison ready:',len(f),'display triangles; CAD bounds',mesh.bounds,flush=True)
viewer.run()
