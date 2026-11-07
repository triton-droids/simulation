"""Verified-capability adapter for Isaac Newton ball dynamics and task observations."""
import numpy as np
import torch

class IsaacRandomizer:
    def __init__(self,robot,ball,scene,sampler,device):
        self.robot=robot;self.ball=ball;self.scene=scene;self.sampler=sampler;self.device=device
        self.mass=ball.data.body_mass.torch.clone()
        self.inertia=ball.data.body_inertia.torch.clone()
        self.history=self.raw().clone()
    def raw(self):
        r,b=self.robot,self.ball
        return torch.cat((r.data.joint_pos.torch,r.data.joint_vel.torch,
            b.data.root_pos_w.torch-self.scene.env_origins,b.data.root_vel_w.torch[:,:3],
            r.data.root_pos_w.torch-self.scene.env_origins,r.data.root_pose_w.torch[:,3:]),dim=1)
    def reset(self,ids,records):
        idx=torch.tensor(ids,device=self.device,dtype=torch.int32)
        scales=torch.tensor([records[i]['values'].get('ball_mass',1.) for i in ids],device=self.device,dtype=torch.float32)
        mass=self.mass[ids]*scales[:,None]
        inertia=self.inertia[ids]*scales[:,None,None]
        self.ball.set_masses_index(masses=mass,env_ids=idx)
        self.ball.set_inertias_index(inertias=inertia,env_ids=idx)
        effective=self.ball.data.body_mass.torch[ids]
        if not torch.allclose(effective,mass):raise RuntimeError('Newton mass readback mismatch')
        self.ball.permanent_wrench_composer.reset(env_ids=idx)
        self.history[ids]=self.raw()[ids]
        for j,i in enumerate(ids):
            records[i]['effective']={'ball_mass_kg':float(effective[j,0]),'inertia_scaled_with_mass':True,
                'rolling_model':'angular viscous drag, uncalibrated','push_target':'ball',
                'observation_quaternion':'xyzw'}
    def forces(self,records,steps):
        force=torch.zeros((len(records),1,3),device=self.device)
        torque=torch.zeros_like(force)
        for i,record in enumerate(records):
            v=record['values']
            if 'external_push' in v and steps[i]%100==50:force[i,0,0]=v['external_push']
            if 'ball_rolling_loss' in v:torque[i,0]=-.02*v['ball_rolling_loss']*self.ball.data.root_vel_w.torch[i,3:]
        self.ball.permanent_wrench_composer.set_forces_and_torques_index(forces=force,torques=torque,is_global=True)
    def observe(self,records,steps):
        raw=self.raw();obs=raw.clone();n=self.robot.data.joint_pos.torch.shape[1]
        for i,record in enumerate(records):
            v=record['values']
            if v.get('observation_delay',0):obs[i]=self.history[i]
            sigma=v.get('proprio_noise',0.)
            if sigma:
                noise=self.sampler.observation_rng(record,steps[i]).uniform(-sigma,sigma,size=2*n)
                obs[i,:2*n]+=torch.as_tensor(noise,device=self.device,dtype=obs.dtype)
        self.history=raw.clone()
        return obs
