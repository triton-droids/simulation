"""Record deterministic ONNX rollouts without touching hardware or training files."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch
from evaluate_velocity import condition_cfg, OnnxPolicy
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer.viewer_config import ViewerConfig


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('run', type=Path)
  parser.add_argument('--seconds', type=float, default=15)
  args = parser.parse_args()
  torch.set_num_threads(4)
  run = args.run.resolve()
  onnx = run / (run.name + '.onnx')
  output = run / 'videos'
  output.mkdir(exist_ok=False)
  policy = OnnxPolicy(onnx)
  manifest = {'onnx': str(onnx), 'sha256': hashlib.sha256(onnx.read_bytes()).hexdigest(),
              'condition': 'fitted nominal values, measured-delay setting; no parameter randomization', 'trials': {}}
  for name, forward in [('standing', 0.0), ('forward_0p2', 0.2)]:
    cfg = condition_cfg('latency', run / 'sysid_params.json')
    cfg.scene.num_envs = 1
    cfg.seed = 123
    cfg.viewer.width, cfg.viewer.height = 960, 720
    cfg.viewer.distance, cfg.viewer.azimuth, cfg.viewer.elevation = 2.6, 135, -20
    cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_BODY
    cfg.viewer.entity_name, cfg.viewer.body_name = 'robot', 'hip'
    cfg.viewer.max_extra_envs = 0
    env = ManagerBasedRlEnv(cfg, device='cuda:0', render_mode='rgb_array')
    frames, velocities, positions, actions, terminations = [], [], [], [], []
    path = output / (name + '.mp4')
    try:
      with torch.no_grad(), imageio.get_writer(path, fps=50, codec='libx264', quality=8,
                                               macro_block_size=16) as writer:
        obs, _ = env.reset()
        command = env.command_manager.get_term('twist')
        robot = env.scene['robot']
        for step in range(round(args.seconds / env.step_dt)):
          command.set_external_command(forward)
          action = policy(obs['actor'])
          obs, reward, terminated, truncated, info = env.step(action)
          writer.append_data(env.render())
          velocities.append(robot.data.root_link_lin_vel_b[0].cpu().numpy().copy())
          positions.append(robot.data.root_link_pos_w[0].cpu().numpy().copy())
          actions.append(action[0].cpu().numpy().copy())
          terminations.append(bool(terminated[0]))
      np.savez_compressed(output / (name + '.npz'), velocity_body=np.asarray(velocities),
                          position_world=np.asarray(positions), action=np.asarray(actions),
                          terminated=np.asarray(terminations), command_forward=forward, dt=env.step_dt)
      manifest['trials'][name] = {'video':str(path), 'seconds':args.seconds, 'forward_command_m_s':forward,
        'termination_count':sum(terminations), 'mean_forward_speed_m_s':float(np.asarray(velocities)[100:,1].mean())}
      print(json.dumps(manifest['trials'][name]), flush=True)
    finally:
      env.close()
  (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
  main()
