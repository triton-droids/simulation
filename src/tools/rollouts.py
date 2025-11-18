import os
import jax
import mediapy as media
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from moviepy import VideoFileClip, clips_array
from src.utils.IO_utils import SuppressOutput

def get_rollout(policy_path, env, make_networks_factory, num_steps):
    """
    Get a rollout from the policy
    """
    ppo_network = make_networks_factory(
        env.obs_size, env.action_size
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    params = model.load_params(policy_path)
    inference_fn = make_policy(params, deterministic=True)

    # initialize the state
    jit_reset = jax.jit(env.reset)
    # jit_reset = env.reset
    jit_step = jax.jit(env.step)
    # jit_step = env.step
    jit_inference_fn = jax.jit(inference_fn)
    # jit_inference_fn = inference_fn

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)

    rollout = [state.pipeline_state]
    for i in range(num_steps):
        ctrl, _ = jit_inference_fn(state.obs, rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        if state.done:
            break
       
    return rollout

def save_rollout(save_path, policy_path, env, make_networks_factory, num_steps) -> None:
    """
    Save a rollout to a file
    """
    
    rollout = get_rollout(policy_path, env, make_networks_factory, num_steps)
    render_video(env, rollout, save_path, render_every=2)

def render_video(
    env,
    rollout,
    save_path: str,
    render_every: int = 2,
    height: int = 360,
    width: int = 640,
):
    """Renders and saves a video of the environment from multiple camera angles.

    """
   
    
    # Define paths for each camera's video
    video_paths = []

    # Render and save videos for each camera
    for camera in ["back", "side"]:
        video_path = save_path + f"-{camera}.mp4"
        media.write_video(
            video_path,
            env.render(
                rollout[::render_every],
                height=height,
                width=width,
                camera=camera,
            ),
            fps=1.0 / env.dt / render_every,
        )
        video_paths.append(video_path)

    with SuppressOutput():
        # Load the video clips using moviepy
        clips = [VideoFileClip(path) for path in video_paths]
        # Arrange the clips in a 2x2 grid
        final_video = clips_array([[clips[0], clips[1]]])
        # Save the final concatenated video
        final_video.write_videofile(save_path + "-eval.mp4", logger=None)