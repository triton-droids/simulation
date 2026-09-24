"""Evaluate a learned policy under simulation physics; save one real rollout."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tracking_task import make_leg_tracking_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

ap=argparse.ArgumentParser()
ap.add_argument('checkpoint',type=Path)
ap.add_argument('--steps',type=int,default=500)
ap.add_argument('--num-envs',type=int,default=64)
args=ap.parse_args(); log=args.checkpoint.parent
agent=json.loads((log/'agent.json').read_text())
cfg=make_leg_tracking_cfg(log/'reference.npz');cfg.scene.num_envs=args.num_envs
cfg.seed=123
cfg.observations['actor'].enable_corruption=False
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    wrapped=RslRlVecEnvWrapper(env,clip_actions=agent['clip_actions'])
    runner=MotionTrackingOnPolicyRunner(wrapped,agent,None,'cuda:0')
    runner.load(str(args.checkpoint))
    policy=runner.get_inference_policy(device='cuda:0')
    obs,_=wrapped.reset()
    robot=env.scene['robot']; durations=[]; rewards=[]; metrics={}; trajectory=[]; done_frames=[]
    failures=0; timeouts=0; effort=[]; base_start=None; base_end=None
    with torch.inference_mode():
        base_start=(robot.data.root_link_pos_w-env.scene.env_origins).clone()
        for step in range(args.steps):
            before=env.episode_length_buf.clone()
            obs,reward,done,extra=wrapped.step(policy(obs))
            failures+=int(env.reset_terminated.sum().item())
            timeouts+=int(env.reset_time_outs.sum().item())
            durations.extend(((before[done.bool()]+1)*env.step_dt).cpu().tolist())
            rewards.append(reward.mean().item())
            effort.append(robot.data.actuator_force.abs().cpu().numpy())
            cmd=env.command_manager.get_term('motion')
            for k,v in cmd.metrics.items():
                if k.startswith('error_'):
                    metrics.setdefault(k,[]).append(v.mean().item())
            q=torch.cat((robot.data.root_link_pos_w[0]-env.scene.env_origins[0],robot.data.root_link_quat_w[0],robot.data.joint_pos[0]))
            trajectory.append(q.cpu().numpy());done_frames.append(bool(done[0]))
            if step==args.steps-2:
                base_end=(robot.data.root_link_pos_w-env.scene.env_origins).clone()
        if base_end is None:
            base_end=(robot.data.root_link_pos_w-env.scene.env_origins).clone()
    effort=np.array(effort)
    result={'checkpoint':str(args.checkpoint.resolve()),'num_envs':args.num_envs,'steps':args.steps,
        'evaluation_seconds_per_env':args.steps*env.step_dt,'completed_episodes':len(durations),
        'failure_resets':failures,'time_limit_resets':timeouts,
        'mean_completed_episode_seconds':float(np.mean(durations)) if durations else None,
        'longest_completed_episode_seconds':max(durations) if durations else None,
        'surviving_episode_seconds_at_end':(env.episode_length_buf*env.step_dt).cpu().tolist(),
        'mean_reward_per_step':float(np.mean(rewards)),
        'actuator_peak_abs_nm':effort.max(axis=(0,1)).tolist(),
        'actuator_mean_abs_nm':effort.mean(axis=(0,1)).tolist(),
        'actuator_fraction_above_datasheet_peak':[float((effort[...,i]>limit).mean()) for i,limit in enumerate([60,60,120,120,17,60,60,120,120,17])],
        'body_forward_displacement_m_by_env':(base_end[:,1]-base_start[:,1]).cpu().tolist(),
        'mean_tracking_errors':{k:float(np.mean(v)) for k,v in metrics.items()},
        'note':'Deterministic policy, no actor observation noise; simulated friction/encoder bias retained. Random reference starts and automatic resets; duration alone does not prove walking.'}
    (log/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez(log/'policy_rollout.npz',qpos=trajectory,fps=1/env.step_dt,reset=done_frames)
    print(json.dumps(result,indent=2))
finally:
    env.close()
