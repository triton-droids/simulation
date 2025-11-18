import os
import argparse

#add argparse argumets
parser = argparse.ArgumentParser(description="Play a trained policy in Mujoco or MJX")
parser.add_argument("--run_dir", type=str, default=None, help="Run directory to load the policy from.")
parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint to load.")

args = parser.parse_args()

import functools
from brax.training.agents.ppo import networks as ppo_networks
from etils import epath
from omegaconf import OmegaConf
from src.locomotion import get_env_class
from src.robots.robot import Robot
from brax.io import model
import jax
from moviepy import VideoFileClip, clips_array

import mediapy as media


#Load play with hydra
def main():
    try:
        run_dir = epath.Path(args.run_dir).resolve()
        policy_path = run_dir / "logs" / "checkpoints" / f"{args.checkpoint}" / "policy"

        config_dir = str(run_dir / ".hydra" / "config.yaml")
        config = OmegaConf.load(config_dir)
    except FileNotFoundError as e:
        print(f"Error: Could not find required file: {e}")
        print(f"Please check that the run_dir '{args.run_dir}' and checkpoint files exist.")
        return

    env_cfg = config.sim
    agent_cfg = config.agent

    EnvClass = get_env_class('default_humanoid_legs')
    robot = Robot(config.robot.name)
    env = EnvClass(
        config.robot.name,
        robot,
        config.env.terrain, 
        env_cfg)

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=agent_cfg.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=agent_cfg.value_hidden_layer_sizes,
    )
   
    ppo_network = make_networks_factory(
        env.obs_size, env.action_size
    )

    make_policy = ppo_networks.make_inference_fn(ppo_network)

    params = model.load_params(policy_path)
    inference_fn = make_policy(params, deterministic=True)

    # initialize the state
    jit_reset = jax.jit(env.reset)

    jit_step = jax.jit(env.step)

    jit_inference_fn = jax.jit(inference_fn)


    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
     #state.info["command"] = jp.array([1.0, 0.0, 0.0])

   
    rollout = [state.pipeline_state]
    for i in range(1000):
        ctrl, _ = jit_inference_fn(state.obs, rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)

        if state.done:
            break

    video_paths = []

    # Render and save videos for each camera
    for camera in ["back", "side"]:
        video_path = f"{camera}.mp4"
        media.write_video(
            video_path,
            env.render(
                rollout[::2],
                height=480,
                width=640,
                camera=camera,
            ),
            fps=1.0 / env.dt / 2,
        )
        video_paths.append(video_path)
    
     # Load the video clips using moviepy
    clips = [VideoFileClip(path) for path in video_paths]
    # Arrange the clips in a 2x2 grid
    final_video = clips_array([[clips[0], clips[1]]])
    # Save the final concatenated video
    final_video.write_videofile("play.mp4", logger=None)

    

if __name__ == "__main__":
    main()