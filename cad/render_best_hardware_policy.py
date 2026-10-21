"""Render the saved best ONNX in the matching hardware simulator."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw, ImageFont
from evaluate_hardware_tracking import make_condition_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path)
args = ap.parse_args()
log = args.run_dir.resolve()
cad = Path(__file__).resolve().parent
for name in (
  "hardware_tracking_task.py",
  "tracking_task.py",
  "triton_lower_body_actuated.xml",
  "triton_lower_body.xml",
):
    if (cad / name).read_bytes() != (log / name).read_bytes():
        raise RuntimeError(f"Current model differs from run snapshot: {name}")
selection = json.loads((log / "best.json").read_text())
cfg = make_condition_cfg(log / "reference.npz", "nominal")
cfg.scene.num_envs = 1
cfg.seed = 123
cfg.commands["motion"].sampling_mode = "start"
agent = json.loads((log / "agent.json").read_text())
env = ManagerBasedRlEnv(cfg, device="cuda:0")
session = ort.InferenceSession(str(log / "best.onnx"), providers=["CPUExecutionProvider"])
renderer = None
writer = None
try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent["clip_actions"])
    obs, _ = wrapped.reset()
    model = env.sim.mj_model
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 480
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    cameras = []
    for azimuth in (135, 225):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 2.15
        cam.azimuth = azimuth
        cam.elevation = -15
        cameras.append(cam)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    output = log / "best_policy_rollout.mp4"
    writer = imageio.get_writer(str(output), fps=50, codec="libx264", quality=8, macro_block_size=2,
                               ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    events = []
    states = []
    reference_frames = []
    previous_frame = -1
    notice = ""
    notice_until = -1
    for step in range(500):
        cmd = env.command_manager.get_term("motion")
        frame = int(cmd.time_steps[0])
        if frame < previous_frame:
            events.append({"time_s": step * .02, "event": "reference_restart"})
            notice = "REFERENCE RESTART"
            notice_until = step + 50
        previous_frame = frame
        qpos = env.sim.data.qpos[0].detach().cpu().numpy().copy()
        states.append(qpos)
        reference_frames.append(frame)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        canvas = Image.new("RGB", (1280, 560), (18, 23, 31))
        for i, cam in enumerate(cameras):
            cam.lookat[:] = [qpos[0], qpos[1], 0.45]
            renderer.update_scene(data, camera=cam, scene_option=option)
            canvas.paste(Image.fromarray(renderer.render().copy()), (i * 640, 52))
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 9), f"BEST SAVED POLICY | {Path(selection['checkpoint']).name} | nominal hardware simulation", font=font, fill="white")
        draw.text((18, 534), f"t = {step*.02:5.2f} s  |  reference frame {frame:03d}/298  |  real-time playback, 50 fps", font=small, fill=(205, 215, 230))
        if step < notice_until:
            draw.rectangle((930, 530, 1279, 559), fill=(125, 35, 35))
            draw.text((944, 534), notice, font=small, fill="white")
        writer.append_data(np.asarray(canvas))
        if step in (0, 125, 250, 375):
            canvas.save(log / f"best_policy_preview_{step:03d}.png")
        actor = obs["actor"].detach().cpu().numpy().astype(np.float32)
        action = session.run(["actions"], {"obs": actor, "time_step": np.array([[frame]], np.float32)})[0]
        with torch.inference_mode():
            obs, _, done, _ = wrapped.step(torch.as_tensor(action, device="cuda:0"))
        if bool(done[0]):
            failure = bool(env.reset_terminated[0])
            events.append({"time_s": (step+1)*.02, "event": "failure_reset" if failure else "episode_timeout"})
            notice = "FAILURE RESET" if failure else "EPISODE TIMEOUT"
            notice_until = step + 51
        if step % 100 == 0:
            print(f"Rendered {step}/500 frames", flush=True)
    np.savez(log / "best_policy_rollout.npz", qpos=np.stack(states), reference_frame=reference_frames, fps=50)
    (log / "best_policy_video.json").write_text(json.dumps({
        "policy": str(log / "best.onnx"), "selected_checkpoint": selection["checkpoint"],
        "condition": "nominal", "seed": 123, "sampling_mode": "start", "seconds": 10,
        "fps": 50, "events": events,
        "note": "Actual physics rollout, not reference playback. All resets are retained and labeled. This single start-frame rollout differs from the 64-environment randomized-start evaluation."
    }, indent=2))
    print(f"Saved {output}", flush=True)
finally:
    if writer is not None:
        writer.close()
    if renderer is not None:
        renderer.close()
    env.close()
