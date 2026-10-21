"""Convert a teleoperation joint log into a tracking reference (reference.npz).

The teleoperation rig maps human leg motion 1:1 onto the robot's joint space, so
it already produces *retargeted* motion: no kinematics solver is needed, only
forward kinematics on the robot model to recover body poses.

Input (``--input``, .npz or .csv):
  - ``joint_pos``   (N, 10) joint positions in the hardware joint order
  - ``fps``         scalar, optional (default 50)
  - ``root_pos``    (N, 3) optional base position; if absent the base is held at
                    the standing height, which yields a walk-in-place reference
  - ``root_quat``   (N, 4) optional wxyz base orientation, default identity
  - ``time``        (N,) optional timestamps; fps is inferred from them

Output matches ``logs/legs_tracking/*/reference.npz``:
  fps, joint_pos (N,10), joint_vel, body_pos_w (N,13,3), body_quat_w (N,13,4),
  body_lin_vel_w (N,13,3), body_ang_vel_w (N,13,3)

Body order is the one used by ``cad/tracking_task.py``: floating_base, torso,
hip, then leg1..foot for the left and right legs.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

BODY_NAMES = (
  "floating_base",
  "torso",
  "hip",
  "left_leg1",
  "left_leg2",
  "left_leg3",
  "left_leg4",
  "left_foot",
  "right_leg1",
  "right_leg2",
  "right_leg3",
  "right_leg4",
  "right_foot",
)
STANCE_HEIGHT = 0.699


def load_input(path: Path) -> dict:
  if path.suffix == ".csv":
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    return {"joint_pos": data[:, :10]}
  arrays = np.load(path)
  return {key: arrays[key] for key in arrays.files}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--model",
    type=Path,
    default=Path(__file__).with_name("triton_lower_body_actuated.xml"),
  )
  parser.add_argument("--fps", type=float, default=None)
  parser.add_argument(
    "--joint-order",
    type=Path,
    default=Path(__file__).with_name("tracking_body_order.json"),
  )
  parser.add_argument(
    "--start-offset",
    type=float,
    default=0.0,
    help="Drop the first N seconds (teleop settling)",
  )
  args = parser.parse_args()

  data = load_input(args.input)
  joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
  order = json.loads(args.joint_order.read_text())["joint_names"]
  if joint_pos.shape[1] != len(order):
    raise ValueError(f"Expected {len(order)} joint columns, got {joint_pos.shape[1]}")

  fps = args.fps or float(data.get("fps", 50.0))
  if "time" in data:
    times = np.asarray(data["time"], dtype=np.float64)
    if len(times) > 1:
      fps = 1.0 / float(np.median(np.diff(times)))
  start = int(round(args.start_offset * fps))
  joint_pos = joint_pos[start:]

  model = mujoco.MjSpec.from_file(str(args.model)).compile()
  joint_ids = [model.joint(name).qposadr[0] for name in order]
  body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in BODY_NAMES]
  data_mj = mujoco.MjData(model)

  frames = joint_pos.shape[0]
  root_pos = np.asarray(data.get("root_pos", np.zeros((frames, 3))), dtype=np.float64)
  root_pos = root_pos[start:]
  if not len(root_pos):
    root_pos = np.zeros((frames, 3))
  root_quat = np.asarray(data.get("root_quat", np.zeros((frames, 4))), dtype=np.float64)
  root_quat = root_quat[start:]
  if not len(root_quat):
    root_quat = np.zeros((frames, 4))
  default_root = np.allclose(root_pos, 0.0)
  if default_root:
    root_pos = np.tile(np.array([0.0, 0.0, STANCE_HEIGHT]), (frames, 1))
  default_quat = np.allclose(root_quat, 0.0)
  if default_quat:
    root_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 1))

  body_pos = np.zeros((frames, len(BODY_NAMES), 3))
  body_quat = np.zeros((frames, len(BODY_NAMES), 4))
  for i in range(frames):
    data_mj.qpos[:] = 0.0
    data_mj.qpos[0:3] = root_pos[i]
    data_mj.qpos[3:7] = root_quat[i] / np.linalg.norm(root_quat[i])
    for adr, value in zip(joint_ids, joint_pos[i], strict=True):
      data_mj.qpos[adr] = value
    mujoco.mj_forward(model, data_mj)
    for b, body_id in enumerate(body_ids):
      body_pos[i, b] = data_mj.xpos[body_id]
      body_quat[i, b] = data_mj.xquat[body_id]

  dt = 1.0 / fps
  body_lin_vel = np.gradient(body_pos, dt, axis=0)
  joint_vel = np.gradient(joint_pos, dt, axis=0)
  # Angular velocity from finite differences of the quaternion.
  body_ang_vel = np.zeros_like(body_pos)
  for b in range(len(BODY_NAMES)):
    q = body_quat[:, b]
    dq = np.gradient(q, dt, axis=0)
    # dq = 0.5 * omega_quat * q  ->  omega = 2 * dq * conj(q), take vector part.
    w = np.zeros((frames, 3))
    for i in range(frames):
      qd = np.array([0.0, *dq[i]])
      qc = np.array([q[i, 0], -q[i, 1], -q[i, 2], -q[i, 3]])
      prod = np.array(
        [
          qd[0] * qc[0] - qd[1] * qc[1] - qd[2] * qc[2] - qd[3] * qc[3],
          qd[0] * qc[1] + qd[1] * qc[0] + qd[2] * qc[3] - qd[3] * qc[2],
          qd[0] * qc[2] - qd[1] * qc[3] + qd[2] * qc[0] + qd[3] * qc[1],
          qd[0] * qc[3] + qd[1] * qc[2] - qd[2] * qc[1] + qd[3] * qc[0],
        ]
      )
      w[i] = 2.0 * prod[1:]
    body_ang_vel[:, b] = w

  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    args.output,
    fps=np.array(fps),
    joint_pos=joint_pos.astype(np.float32),
    joint_vel=joint_vel.astype(np.float32),
    body_pos_w=body_pos.astype(np.float32),
    body_quat_w=body_quat.astype(np.float32),
    body_lin_vel_w=body_lin_vel.astype(np.float32),
    body_ang_vel_w=body_ang_vel.astype(np.float32),
  )
  print(
    f"Wrote {args.output}: {frames} frames at {fps:.1f} Hz"
    + (" (walk-in-place root)" if default_root else "")
  )


if __name__ == "__main__":
  main()
