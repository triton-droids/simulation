"""Isolated CPU worlds for diagnostics, not GPU vectorization claims."""
from collections import deque
import math
import numpy as np
import mujoco
from robocub.soccer.field.geometry import Field
from robocub.soccer.field.scene import scene_xml
from robocub.soccer.robots.importer import require_robot
from robocub.soccer.randomization.sampler import Sampler

class SoccerEnv:
    backend='mujoco'
    dt=.02
    def __init__(self,num_envs=1,field=None,seed=42,preset='mild',suite='train',rank=0,robot='demo',robot_asset=None,robot_metadata=None,config=None):
        if not 1<=num_envs<=4096:raise ValueError('num_envs must be 1..4096')
        self.num_envs=num_envs;self.field=field or Field();self.info=require_robot(robot,robot_asset,robot_metadata)
        self.sampler=Sampler(self.field,seed,preset,self.backend,rank,suite,config)
        xml=scene_xml(self.field,self.info,self.info['path'] if robot=='g1' else robot_asset if robot=='real' else None)
        self.models=[mujoco.MjModel.from_xml_string(xml) for _ in range(num_envs)]
        self.data=[mujoco.MjData(m) for m in self.models];m=self.models[0]
        self.ball_joint=m.joint('ball_free').id;self.bq=int(m.jnt_qposadr[self.ball_joint]);self.bv=int(m.jnt_dofadr[self.ball_joint]);self.ball_id=m.body('soccer_ball').id
        self.act_joints=m.actuator_trnid[:,0].astype(int);self.n_actions=m.nu
        if self.n_actions==0:raise ValueError('Robot has no actuators; explicit actuator mapping required')
        if any(m.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE) for j in self.act_joints):raise ValueError('Only scalar joint actuators supported')
        self.jq=m.jnt_qposadr[self.act_joints];self.jv=m.jnt_dofadr[self.act_joints]
        self.nominal=np.zeros(self.n_actions);self.kp=20.;self.kd=1.;self.effort=12.
        self.root_id=m.body('proxy_root').id if robot=='demo' else int(m.body_rootid[m.jnt_bodyid[self.act_joints[0]]])
        self.free=[j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and j!=self.ball_joint]
        if robot in ('real','g1'):
            meta=self.info['operator_metadata'];names=[m.joint(int(j)).name for j in self.act_joints]
            if not set(names)<=set(meta['nominal_joints']):raise ValueError('Metadata must supply every controlled joint nominal pose')
            self.nominal=np.array([meta['nominal_joints'][n] for n in names]);self.kp=meta['kp'];self.kd=meta['kd'];self.effort=meta['effort_limit']
        self.is_g1=robot=='g1'
        self.position_actuators=m.actuator_biastype==mujoco.mjtBias.mjBIAS_AFFINE
        self.baseline_gain=[x.actuator_gainprm.copy() for x in self.models]
        self.baseline_bias=[x.actuator_biasprm.copy() for x in self.models]
        self.baseline_force=[x.actuator_forcerange.copy() for x in self.models]
        self.baseline_joint_force=[x.jnt_actfrcrange.copy() for x in self.models]
        self.baseline_damping=[x.dof_damping.copy() for x in self.models]
        self.baseline_mass=[(float(x.body_mass[self.ball_id]),x.body_inertia[self.ball_id].copy()) for x in self.models]
        self.episodes=np.full(num_envs,-1,int);self.steps=np.zeros(num_envs,int);self.done=np.zeros(num_envs,bool)
        self.records=[None]*num_envs;self.actions=[deque(maxlen=2) for _ in range(num_envs)];self.obs_history=[deque(maxlen=2) for _ in range(num_envs)]
        self.reset()

    def reset(self,env_ids=None,episode=None):
        ids=list(range(self.num_envs)) if env_ids is None else list(env_ids)
        for i in ids:
            if not 0<=i<self.num_envs:raise IndexError(i)
            self.episodes[i]=int(episode) if episode is not None else self.episodes[i]+1
            rec=self.sampler.sample(i,self.episodes[i]);self.records[i]=rec;v=rec['values'];m,d=self.models[i],self.data[i]
            mujoco.mj_resetData(m,d);self.steps[i]=0;self.done[i]=False
            mass,inertia=self.baseline_mass[i];scale=v.get('ball_mass',1.)
            m.body_mass[self.ball_id]=mass*scale;m.body_inertia[self.ball_id]=inertia*scale
            m.dof_damping[:]=self.baseline_damping[i]*v.get('joint_damping',1.)
            m.actuator_gainprm[:]=self.baseline_gain[i]*v.get('actuator_gain',1.)
            m.actuator_biasprm[:]=self.baseline_bias[i]*v.get('actuator_gain',1.)
            m.actuator_forcerange[:]=self.baseline_force[i]*v.get('motor_strength',1.)
            m.jnt_actfrcrange[:]=self.baseline_joint_force[i]*v.get('motor_strength',1.)
            # Recompute mass-dependent constants after reset-time parameter updates.
            mujoco.mj_setConst(m,d)
            yaw=v['robot_spawn_yaw'];xy=v['robot_spawn_xy']
            if self.free:
                j=self.free[0];q=int(m.jnt_qposadr[j]);d.qpos[q:q+3]=[*xy,self.info['operator_metadata']['initial_height']];d.qpos[q+3:q+7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
            else:
                m.body_pos[self.root_id,:2]=xy;m.body_quat[self.root_id]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
            d.qpos[self.jq]=self.nominal
            d.qpos[self.bq:self.bq+3]=[*v['ball_spawn_xy'],self.field.ball_radius+.01];d.qpos[self.bq+3:self.bq+7]=[1,0,0,0]
            d.qvel[self.bv:self.bv+2]=v.get('ball_velocity',[0,0]);mujoco.mj_forward(m,d)
            if self.free and any(d.contact[k].dist < -.02 for k in range(d.ncon)):raise ValueError('Initial robot contact penetration exceeds2cm; fix supplied initial pose/scaling')
            rec['effective']={'ball_mass_kg':float(m.body_mass[self.ball_id]),'joint_damping':m.dof_damping[self.jv].tolist(),'sliding_friction':float(m.geom_friction[m.geom('turf').id,0]),'active':rec['enabled'],'rolling_model':'angular viscous drag proxy, not coefficient of rolling resistance'}
            self.actions[i].clear();self.actions[i].append(np.zeros(self.n_actions));self.obs_history[i].clear();self.obs_history[i].append(self._raw(i))
        return self.observe()

    def _raw(self,i):
        d=self.data[i]
        return np.concatenate([d.qpos[self.jq],d.qvel[self.jv],d.qpos[self.bq:self.bq+3],d.qvel[self.bv:self.bv+3],d.xpos[self.root_id],d.xquat[self.root_id]])

    def observe(self):
        result=[]
        for i in range(self.num_envs):
            v=self.records[i]['values'];raw=self._raw(i)
            if v.get('observation_delay',0) and self.obs_history[i]:raw=self.obs_history[i][-1].copy()
            else:raw=raw.copy()
            sigma=v.get('proprio_noise',0.)
            if sigma:
                rng=self.sampler.observation_rng({k:x for k,x in self.records[i].items() if k!='effective'},self.steps[i]);raw[:2*self.n_actions]+=rng.uniform(-sigma,sigma,2*self.n_actions)
            result.append(raw)
        return np.array(result,dtype=np.float32)

    def scripted_action(self):
        # Genuine bounded joint command, NOT a learned walking controller.
        if self.is_g1:
            a=np.zeros((self.num_envs,self.n_actions))
            for k,j in enumerate(self.act_joints):
                if 'elbow' in self.models[0].joint(int(j)).name:a[:,k]=.05*np.sin(self.steps*self.dt*2)
            return a
        return np.array([.2*np.sin(self.steps[i]*self.dt*2+np.arange(self.n_actions)*math.pi) for i in range(self.num_envs)])

    def step(self,action):
        a=np.asarray(action,float)
        if a.shape!=(self.num_envs,self.n_actions) or not np.isfinite(a).all():raise ValueError(f'actions require finite {(self.num_envs,self.n_actions)}')
        rewards=np.zeros(self.num_envs);terminated=np.zeros(self.num_envs,bool);truncated=np.zeros(self.num_envs,bool)
        for i,(m,d) in enumerate(zip(self.models,self.data)):
            if self.done[i]:raise RuntimeError(f'Environment{i} terminated; reset explicitly')
            v=self.records[i]['values'];old_ball=d.qpos[self.bq];self.obs_history[i].append(self._raw(i))
            delayed=self.actions[i][-1] if v.get('control_latency',0) else np.clip(a[i],-.5,.5)
            self.actions[i].append(np.clip(a[i],-.5,.5).copy())
            for _ in range(4):
                torque=self.kp*v.get('actuator_gain',1.)*(self.nominal+delayed-d.qpos[self.jq])-self.kd*d.qvel[self.jv]
                d.ctrl[:]=np.clip(torque*v.get('motor_strength',1.),-self.effort,self.effort)
                d.ctrl[self.position_actuators]=(self.nominal+delayed)[self.position_actuators]
                d.xfrc_applied[:]=0
                if 'ball_rolling_loss' in v:
                    d.xfrc_applied[self.ball_id,3:]=-.02*v['ball_rolling_loss']*d.qvel[self.bv+3:self.bv+6]
                if 'external_push' in v and self.steps[i]%100==50:d.xfrc_applied[self.ball_id,0]=v['external_push']
                mujoco.mj_step(m,d)
            self.steps[i]+=1;ball=d.qpos[self.bq:self.bq+3]
            terminated[i]=(bool(self.free) and d.xpos[self.root_id,2]<.35) or not np.isfinite(d.qpos).all() or abs(ball[0])>self.field.length/2+self.field.goal_depth+.5 or abs(ball[1])>self.field.width/2+.5
            truncated[i]=self.steps[i]>=1500
            rewards[i]=float(ball[0]-old_ball)-.0001*float(np.sum(d.ctrl**2))
            self.done[i]=terminated[i] or truncated[i]
        return self.observe(),rewards,terminated,truncated,{'records':self.records,'reward':'ball +X progress minus effort; no soccer success claim'}

    def reposition(self,ball=None,robot=None,env_id=0):
        i=env_id;m,d=self.models[i],self.data[i]
        if ball is not None:
            if not self.field.legal(*ball,self.field.ball_radius):raise ValueError('Ball outside legal spawn region')
            if np.linalg.norm(np.array(ball)-d.xpos[self.root_id,:2])<.7:raise ValueError('Ball overlaps robot')
            d.qpos[self.bq:self.bq+3]=[*ball,self.field.ball_radius+.01];d.qvel[self.bv:self.bv+6]=0
        if robot is not None:
            if not self.field.legal(*robot,.5):raise ValueError('Robot outside legal spawn region')
            if np.linalg.norm(np.array(robot)-d.qpos[self.bq:self.bq+2])<.7:raise ValueError('Robot overlaps ball')
            if self.free:
                q=int(m.jnt_qposadr[self.free[0]]);d.qpos[q:q+2]=robot;d.qvel[:]=0
            else:m.body_pos[self.root_id,:2]=robot
        mujoco.mj_forward(m,d)
        self.obs_history[i].clear();self.obs_history[i].append(self._raw(i))

    def close(self):pass


