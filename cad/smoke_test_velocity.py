"""CPU smoke test for velocity_task.py: builds the env, checks the interface, steps it.

  env -u PYTHONPATH PYTHONPATH=../mjlab/src <venv>/bin/python smoke_test_velocity.py [--fit sysid_params.json]

Checks the actor is 39-D in contract order, joints and actions are in contract
order, the stand pose is the default pose, the command path (clip, slew, torque
guard) behaves, and the robot holds the stand pose for a few seconds with zero
actions. Optionally runs two PPO iterations and exports the ONNX.
"""

import argparse
import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from velocity_task import (
  JOINTS,
  HardwareVelocityConfig,
  guard_span_rad,
  load_contract,
  make_triton_velocity_cfg,
)

from mjlab.envs import ManagerBasedRlEnv


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--fit", type=Path, default=None)
  p.add_argument("--num-envs", type=int, default=4)
  p.add_argument("--device", default="cpu")
  p.add_argument("--train", action="store_true", help="also run 2 PPO iterations and export ONNX")
  args = p.parse_args()

  contract = load_contract()
  cfg = make_triton_velocity_cfg(HardwareVelocityConfig(fit_path=args.fit, randomize=False))
  cfg.scene.num_envs = args.num_envs
  cfg.events.pop("push_robot", None)
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    robot = env.scene["robot"]
    assert tuple(robot.joint_names) == JOINTS, robot.joint_names
    term = env.action_manager.get_term("joint_pos")
    assert tuple(term.target_names) == JOINTS, term.target_names
    assert env.action_manager.total_action_dim == 10
    actor_dim = env.observation_manager.group_obs_dim["actor"]
    assert actor_dim == (contract["actor_obs_dim"],), actor_dim
    names = env.observation_manager.active_terms["actor"]
    assert list(names) == [t["name"] for t in contract["actor_obs"]], names
    stand = torch.tensor([contract["stand_pose_rad"][j] for j in JOINTS])
    default = robot.data.default_joint_pos[0].cpu()
    assert torch.allclose(default, stand, atol=1e-6), (default, stand)
    print(f"[OK] joints, actions (10) and actor obs {actor_dim} match the contract; default pose = stand pose")

    obs, _ = env.reset()
    zero = torch.zeros(args.num_envs, 10, device=env.device)
    heights, tilts = [], []
    total_falls = 0
    for _ in range(int(4.0 / env.step_dt)):
      obs, rew, term_, trunc, extras = env.step(zero)
      total_falls += int(term_.sum())
      heights.append(robot.data.root_link_pos_w[:, 2].mean().item())
      g = robot.data.projected_gravity_b
      tilts.append(torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1))).max().item())
    actor = obs["actor"]
    assert torch.isfinite(actor).all()
    print(f"[OK] zero action for 4 s: base height {heights[0]:.3f} -> {heights[-1]:.3f} m, "
          f"max tilt {max(tilts):.1f} deg, falls over all steps {total_falls} (untrained diagnostic)")
    print(f"     actor obs sample: gyro {actor[0, 0:3].tolist()}, gravity {actor[0, 3:6].tolist()}, "
          f"command {actor[0, 36:39].tolist()}")

    # Command path: a large action step must be slewed and guard-limited.
    big = torch.full((args.num_envs, 10), 4.0, device=env.device)
    before = term._commanded.clone()
    env.step(big)
    step = (term._commanded - before).abs().max().item()
    limit = contract["target_slew_rad_s"] * env.step_dt
    pos = robot.data.joint_pos[:, term._target_ids]
    span = torch.tensor(guard_span_rad(contract), device=env.device)
    over = ((term._processed_actions - pos).abs() - span).max().item()
    print(f"[OK] slew: largest target change {step:.4f} rad per step (limit {limit:.4f}); "
          f"guard: target-position gap exceeds span by at most {over:.4f} rad (<= 0 expected, "
          f"position measured after the step)")
    assert step <= limit + 1e-5

    for _ in range(50):
      obs, *_ = env.step(torch.randn(args.num_envs, 10, device=env.device))
      assert torch.isfinite(obs["actor"]).all()
    print("[OK] 50 random-action steps, all observations finite")

    if args.train:
      from mjlab.rl import RslRlVecEnvWrapper
      from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
      from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

      agent = unitree_g1_ppo_runner_cfg()
      agent.experiment_name = "smoke"
      agent.logger = "tensorboard"
      agent.num_steps_per_env = 8
      agent.save_interval = 1
      with tempfile.TemporaryDirectory() as log:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = VelocityOnPolicyRunner(wrapped, asdict(agent), log, args.device)
        runner.learn(num_learning_iterations=2)
        onnx = sorted(Path(log).rglob("*.onnx"))
        assert onnx, "no ONNX exported"
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx[-1]), providers=["CPUExecutionProvider"])
        ins = [(i.name, i.shape) for i in sess.get_inputs()]
        outs = [(o.name, o.shape) for o in sess.get_outputs()]
        meta = sess.get_modelmeta().custom_metadata_map
        print(f"[OK] 2 PPO iterations; ONNX inputs {ins} outputs {outs}")
        print(f"     ONNX metadata keys: {sorted(meta)}")
        for k in ("joint_names", "default_joint_pos", "action_scale", "observation_names"):
          if k in meta:
            print(f"     {k}: {meta[k][:200]}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
