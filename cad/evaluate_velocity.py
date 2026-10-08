"""Evaluate an exported velocity-policy ONNX across command profiles and conditions.

  python cad/evaluate_velocity.py <run>/<run>.onnx --fit sysid_params.json [--device cuda:0]

Runs the ONNX itself (the file that goes on the robot), not the training
checkpoint. Command profiles are held fixed (no resampling):
  stand        zero command                         -> drift and sway while standing
  fwd_0.2/0.4  constant forward command              -> speed tracking, falls
  start_stop   0 -> 0.3 m/s -> 0 in 4 s segments     -> transitions
  yaw          0.3 rad/s turn on the spot            -> yaw tracking
Conditions:
  nominal      fitted actuator values, no randomisation, no delay
  latency      nominal plus 1 control step observation delay and 1 physics step action delay
  randomized   the training randomisation with the latency above
  extreme      worst-case corners of the training randomisation: highest friction
               and armature, weakest kp, strongest kd, heavy links, low foot
               friction, maximum pushes
Prints a table and writes JSON with per-condition, per-profile metrics.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from velocity_task import JOINTS, HardwareVelocityConfig, load_contract, make_triton_velocity_cfg

from mjlab.envs import ManagerBasedRlEnv

PROFILES = {
  "stand": [(10.0, (0.0, 0.0, 0.0))],
  "fwd_0.1": [(10.0, (0.0, 0.1, 0.0))],
  "fwd_0.2": [(10.0, (0.0, 0.2, 0.0))],
  "start_stop": [(4.0, (0.0, 0.0, 0.0)), (4.0, (0.0, 0.1, 0.0)),
                 (4.0, (0.0, 0.0, 0.0)), (4.0, (0.0, 0.2, 0.0)), (4.0, (0.0, 0.0, 0.0))],
}

CONDITIONS = ("nominal", "latency", "randomized", "extreme")


def condition_cfg(condition: str, fit: Path | None):
  delays = {"nominal": ((0, 0), (0, 0))}.get(condition, ((1, 1), (1, 1)))
  hardware = HardwareVelocityConfig(
    fit_path=fit,
    observation_delay_steps=delays[0],
    action_delay_physics_steps=delays[1],
    randomize=condition in ("randomized", "extreme"),
  )
  cfg = make_triton_velocity_cfg(hardware)
  cfg.curriculum = {}
  cfg.episode_length_s = 1e6
  twist = cfg.commands["twist"]
  twist.resampling_time_range = (1e6, 1e6)
  twist.rel_standing_envs = 0.0
  twist.debug_vis = False
  cfg.observations["actor"].enable_corruption = condition != "nominal"
  cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
  cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
  if condition in ("nominal", "latency"):
    cfg.events.pop("push_robot", None)


  if condition == "extreme":

    for name, term in cfg.events.items():
      if name == "pd_gains":
        # weakest kp and strongest kd the training randomisation allows
        term.params["kp_range"] = (term.params["kp_range"][0],) * 2
        term.params["kd_lo"] = list(term.params["kd_hi"])
      elif name.split("_")[0] in ("frictionloss", "armature", "damping"):
        hi = term.params["ranges"][1]
        term.params["ranges"] = (hi, hi)

  return cfg


class OnnxPolicy:
  def __init__(self, path: Path):
    import onnxruntime as ort

    self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    self.inp = self.sess.get_inputs()[0].name
    self.out = self.sess.get_outputs()[0].name
    self.batch = self.sess.get_inputs()[0].shape[0]

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    x = obs.detach().cpu().numpy().astype(np.float32)
    if self.batch == 1:
      y = np.concatenate([self.sess.run([self.out], {self.inp: row[None]})[0] for row in x])
    else:
      y = self.sess.run([self.out], {self.inp: x})[0]
    return torch.as_tensor(y, device=obs.device)


def run_profile(env, policy, profile, contract) -> dict:
  robot = env.scene["robot"]
  twist = env.command_manager.get_term("twist")
  with torch.no_grad():
    obs, _ = env.reset()
  start_xy = (robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).clone()
  caps = torch.tensor([contract["torque_cap_motor_nm"][m] for m in contract["motor_model"]], device=env.device)
  falls = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  seg_stats, peak_tau, peak_qd, over_cap = [], torch.zeros(10, device=env.device), torch.zeros(10, device=env.device), 0
  with torch.no_grad():
    for seconds, cmd in profile:
      steps = int(round(seconds / env.step_dt))
      vf, vr, wz = [], [], []
      for k in range(steps):
        twist.set_external_command(cmd[1])
        obs = env.get_observations()
        obs, _, terminated, _, _ = env.step(policy(obs["actor"]))
        falls |= terminated.bool() & ~falls
        if k >= steps // 3:   # skip the transient at the start of each segment
          v = robot.data.root_link_lin_vel_b
          vr.append(v[:, 0].clone()); vf.append(v[:, 1].clone())
          wz.append(robot.data.root_link_ang_vel_b[:, 2].clone())
        tau = robot.data.actuator_force.abs()
        peak_tau = torch.maximum(peak_tau, tau.max(dim=0).values)
        peak_qd = torch.maximum(peak_qd, robot.data.joint_vel.abs().max(dim=0).values)
        over_cap += int((tau > caps).any(dim=1).sum())
      ok = ~falls
      def mean(xs):
        return float(torch.stack(xs)[:, ok].mean()) if ok.any() and xs else float("nan")
      seg_stats.append({"cmd": cmd, "v_forward": mean(vf), "v_right": mean(vr), "yaw_rate": mean(wz)})
  end_xy = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  drift = torch.linalg.vector_norm(end_xy - start_xy, dim=1)
  return {
    "falls": int(falls.sum()),
    "envs": env.num_envs,
    "segments": seg_stats,
    "displacement_m_median": float(drift.median()),
    "peak_torque_nm": peak_tau.cpu().round(decimals=2).tolist(),
    "peak_torque_over_cap_steps": over_cap,
    "peak_joint_speed_rad_s": peak_qd.cpu().round(decimals=2).tolist(),
  }


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("onnx", type=Path)
  p.add_argument("--fit", type=Path, default=None)
  p.add_argument("--conditions", default=",".join(CONDITIONS))
  p.add_argument("--profiles", default=",".join(PROFILES))
  p.add_argument("--num-envs", type=int, default=32)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--output", type=Path, default=None)
  args = p.parse_args()
  fit = args.fit or (args.onnx.parent / "sysid_params.json" if (args.onnx.parent / "sysid_params.json").is_file() else None)
  contract = load_contract()
  policy = OnnxPolicy(args.onnx)
  results = {"onnx": str(args.onnx.resolve()), "fit": str(fit) if fit else None, "conditions": {}}
  for condition in args.conditions.split(","):
    cfg = condition_cfg(condition, fit)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 123
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
      assert tuple(env.scene["robot"].joint_names) == JOINTS
      results["conditions"][condition] = {
        name: run_profile(env, policy, PROFILES[name], contract) for name in args.profiles.split(",")
      }
    finally:
      env.close()
  print(f"\n{'condition':<11} {'profile':<11} {'falls':>7}  per-segment (cmd fwd -> actual fwd m/s, yaw rad/s)   "
        f"peak |tau| hip1/hip2/thigh/knee/ankle (Nm)")
  for cond, profs in results["conditions"].items():
    for name, r in profs.items():
      segs = "  ".join(f"{s['cmd'][1]:.1f}->{s['v_forward']:+.2f}" + (f" w{s['yaw_rate']:+.2f}" if s['cmd'][2] else "")
                       for s in r["segments"])
      t = r["peak_torque_nm"]
      tau = "/".join(f"{max(t[i], t[i + 5]):.1f}" for i in range(5))
      print(f"{cond:<11} {name:<11} {r['falls']:>3}/{r['envs']:<3}  {segs:<52} {tau}")
  if args.output:
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
  main()
