"""Short local reference-tracking learning test; simulation only."""
import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import shutil
import numpy as np
import torch
from tracking_task import make_leg_tracking_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', type=Path, default=Path(__file__).with_name('standing_reference.npz'))
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--resume', type=Path, help='Continue policy and optimizer from a checkpoint')
    args = parser.parse_args()
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this execution environment')
    cfg = make_leg_tracking_cfg(args.motion)
    cfg.scene.num_envs = args.num_envs
    ref = np.load(args.motion)
    frames = ref['joint_pos'].shape[0]
    required = {'joint_pos': (10,), 'joint_vel': (10,), 'body_pos_w': (13, 3),
                'body_quat_w': (13, 4), 'body_lin_vel_w': (13, 3), 'body_ang_vel_w': (13, 3)}
    for key, trailing in required.items():
        if ref[key].shape != (frames, *trailing) or not np.isfinite(ref[key]).all():
            raise ValueError(f'Invalid reference array: {key}')
    log = Path(__file__).resolve().parent.parent / 'logs' / 'legs_tracking' / datetime.now().strftime('%Y%m%d_%H%M%S')
    log.mkdir(parents=True)
    shutil.copy2(args.motion, log / 'reference.npz')
    for name in ('chrobot_16kg_candidate.xml', 'chrobot_16kg_actuated.xml', 'tracking_body_order.json', 'tracking_task.py'):
        shutil.copy2(Path(__file__).with_name(name), log / name)
    (log / 'run.json').write_text(json.dumps({'motion': str(args.motion.resolve()), 'num_envs': args.num_envs, 'iterations': args.iterations, 'resume': str(args.resume.resolve()) if args.resume else None}, indent=2))
    agent = unitree_g1_tracking_ppo_runner_cfg()
    agent.experiment_name = 'legs_tracking'
    agent.logger = 'tensorboard'
    agent.actor.hidden_dims = (128, 64)
    agent.critic.hidden_dims = (128, 64)
    agent.actor.distribution_cfg['init_std'] = 0.2
    agent.save_interval = args.save_interval
    (log / 'agent.json').write_text(json.dumps(asdict(agent), indent=2))
    print(f'Training directory: {log}', flush=True)
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MotionTrackingOnPolicyRunner(wrapped, asdict(agent), str(log), 'cuda:0')
        if args.resume is not None:
            runner.load(str(args.resume.resolve()), map_location='cuda:0')
            print(f'Resumed from {args.resume}; iteration {runner.current_learning_iteration}', flush=True)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        print(f'Training finished: {log}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
