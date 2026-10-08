"""Train the Triton legs velocity policy (stand at zero command, walk slowly forward).

  python cad/train_velocity.py --fit /path/to/sysid_params.json [--num-envs 4096]

Every run gets a new directory with the contract, the fit, the task code and the
agent config copied in, so the exported ONNX can always be traced back to them.
The runner exports `<run>/<checkpoint>.onnx` at every save (obs[1,39] -> actions[1,10]).
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import signal
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from velocity_task import JOINTS, HardwareVelocityConfig, load_contract, make_triton_velocity_cfg

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--fit", type=Path, default=None, help="sysid_params.json from the system-ID fit")
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--iterations", type=int, default=10000)
  p.add_argument("--hours", type=float, default=8.0, help="Training wall time; 0 disables deadline")
  p.add_argument("--save-interval", type=int, default=200)
  p.add_argument("--run-dir", type=Path, help="new output directory; must not exist")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--resume", type=Path, help="continue from a checkpoint")
  p.add_argument("--no-randomize", action="store_true", help="debug only: disable hardware randomisation")
  p.add_argument("--style-clip", type=Path, default=None,
                 help="reference clip npz (joint_pos [N,10]) for the optional soft gait-style reward")
  p.add_argument("--style-weight", type=float, default=0.0, help="0 = plain PPO baseline")
  args = p.parse_args()
  torch.set_num_threads(4)
  if args.hours < 0:
    p.error('--hours must be nonnegative')
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; pass --device cpu only for debugging")
  if args.fit is not None and not args.fit.is_file():
    raise FileNotFoundError(args.fit)
  if args.fit is None:
    print("[WARN] no --fit given: using codex-branch actuator estimates instead of the measured fit")

  contract = load_contract()
  if args.style_weight > 0 and (args.style_clip is None or not args.style_clip.is_file()):
    raise FileNotFoundError(f"--style-weight needs an existing --style-clip, got {args.style_clip}")
  hardware = HardwareVelocityConfig(fit_path=args.fit, randomize=not args.no_randomize,
                                    style_clip=args.style_clip, style_weight=args.style_weight)
  cfg = make_triton_velocity_cfg(hardware)
  cfg.scene.num_envs = args.num_envs

  log = args.run_dir or HERE.parent / "logs" / "triton_velocity" / datetime.now().strftime("%Y%m%d_%H%M%S")
  log.mkdir(parents=True, exist_ok=False)
  shutil.copy2(HERE.parent / 'mjlab' / 'uv.lock', log / 'uv.lock')
  packages = subprocess.run([__import__('sys').executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
  (log / 'environment.txt').write_text(packages.stdout if packages.returncode == 0 else packages.stderr)
  for name in (
    "velocity_contract.json",
    "velocity_task.py",
    "train_velocity.py",
    "hardware_tracking_task.py",
    "tracking_task.py",
    "chrobot_hardware_candidate.xml",
    "chrobot_hardware_actuated.xml",
    "VELOCITY_TASK.md",
  ):
    if (HERE / name).is_file():
      shutil.copy2(HERE / name, log / name)
  if args.fit is not None:
    shutil.copy2(args.fit, log / "sysid_params.json")
  if args.style_clip is not None and args.style_weight > 0:
    shutil.copy2(args.style_clip, log / "style_clip.npz")
  git = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True)
  dirty = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain"], capture_output=True, text=True)

  agent = unitree_g1_ppo_runner_cfg()
  agent.upload_model = False
  agent.experiment_name = "triton_velocity"
  agent.logger = "tensorboard"
  agent.actor.hidden_dims = (256, 128, 64)
  agent.critic.hidden_dims = (256, 128, 64)
  agent.actor.distribution_cfg["init_std"] = 0.5
  agent.save_interval = args.save_interval
  agent.max_iterations = args.iterations
  (log / "agent.json").write_text(json.dumps(asdict(agent), indent=2))
  (log / "run.json").write_text(json.dumps({
    "started": datetime.now().isoformat(timespec="seconds"),
    "num_envs": args.num_envs,
    "iterations": args.iterations,
    "hours": args.hours,
    "device": args.device,
    "resume": str(args.resume.resolve()) if args.resume else None,
    "fit": str(args.fit.resolve()) if args.fit else None,
    "fit_sha256": sha256(args.fit) if args.fit else None,
    "contract_sha256": sha256(HERE / "velocity_contract.json"),
    "hardware": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(hardware).items()},
    "git_head": git.stdout.strip(),
    "git_dirty": [ln for ln in dirty.stdout.splitlines() if ln.strip()],
  }, indent=2))
  print(f"Training directory: {log}", flush=True)

  env = ManagerBasedRlEnv(cfg, device=args.device)
  requested_stop = threading.Event()
  timer = None
  runner = None
  reason = 'iteration_limit'
  try:
    assert tuple(env.scene["robot"].joint_names) == JOINTS
    assert tuple(env.action_manager.get_term("joint_pos").target_names) == JOINTS
    assert env.observation_manager.group_obs_dim["actor"] == (contract["actor_obs_dim"],)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = VelocityOnPolicyRunner(wrapped, asdict(agent), str(log), args.device)
    if args.resume is not None:
      runner.load(str(args.resume.resolve()), map_location=args.device)
      print(f"Resumed from {args.resume}; iteration {runner.current_learning_iteration}", flush=True)
    original_step = env.step
    def timed_step(actions):
      if requested_stop.is_set():
        raise KeyboardInterrupt
      return original_step(actions)
    env.step = timed_step
    def stop(signum, frame):
      nonlocal reason
      reason = f'signal_{signum}'
      requested_stop.set()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.time()
    (log / 'status.json').write_text(json.dumps({'state':'training','started_epoch':started,
      'deadline_epoch':started+args.hours*3600 if args.hours else None,'pid':__import__('os').getpid()},indent=2))
    if args.hours:
      def deadline():
        nonlocal reason
        reason = 'time_limit'
        requested_stop.set()
      timer = threading.Timer(args.hours*3600, deadline)
      timer.daemon = True
      timer.start()
    try:
      runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
    except KeyboardInterrupt:
      print(f'Stopping at safe Python step boundary: {reason}',flush=True)
    runner.save(str(log / 'final.pt'))
    (log / 'status.json').write_text(json.dumps({'state':'completed','stop_reason':reason,
      'finished_epoch':time.time(),'iteration':runner.current_learning_iteration,
      'final_checkpoint':'final.pt'},indent=2))
    print(f"Training finished: {log}")
  except BaseException as error:
    (log / 'status.json').write_text(json.dumps({'state':'failed','error':repr(error)},indent=2))
    raise
  finally:
    if timer is not None:
      timer.cancel()
    env.close()


if __name__ == "__main__":
  main()
