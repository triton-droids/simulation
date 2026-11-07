"""Executed GPU ball checks against the actual field collision scene."""
import torch

def check_ball_physics(scene,sim,field,device):
    ball=scene['ball']
    if scene.num_envs<3:raise ValueError('Physics checks need at least3 environments')
    ids=torch.tensor([0,1,2],device=device,dtype=torch.int32)
    pose=ball.data.default_root_pose.torch[:3].clone();vel=torch.zeros((3,6),device=device)
    pose[:,:3]=torch.tensor([[0,0,1.],[0,0,field.ball_radius+.01],[field.length/2-.6,field.goal_width/2+field.post_width/2,.2]],device=device)
    pose[:,:3]+=scene.env_origins[:3];vel[1,0]=1.;vel[2,0]=2.
    ball.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids);ball.write_root_velocity_to_sim_index(root_velocity=vel,env_ids=ids)
    print('PHYSICS_PROBE', {'floating':ball.root_view.is_floating_base,'requested':pose.tolist(),'readback':ball.data.root_pose_w.torch[:3].tolist()},flush=True)
    low=10.;post_max=-100.
    for _ in range(100):
        for _ in range(4):scene.write_data_to_sim();sim.step();scene.update(.005)
        pos=ball.data.root_pos_w.torch[:3]-scene.env_origins[:3]
        low=min(low,float(pos[0,2]));post_max=max(post_max,float(pos[2,0]))
    final=pos.detach().cpu().tolist()
    result={'drop_min_height':low,'drop_final_height':final[0][2],'roll_distance':final[1][0],'post_max_x':post_max,'final_positions':final}
    if low<field.ball_radius-.03 or abs(final[0][2]-field.ball_radius)>.03:raise RuntimeError(f'GPU ball drop failed: {result}')
    if not .1<final[1][0]<2.2:raise RuntimeError(f'GPU roll failed: {result}')
    if post_max>field.length/2+.04:raise RuntimeError(f'GPU post barrier failed: {result}')
    result['passed']=True
    return result
