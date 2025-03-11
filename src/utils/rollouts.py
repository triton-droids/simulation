from typing import Callable

import jax
import jax.numpy as jp
import numpy as np
from brax import envs
from brax.mjx.base import State as mjxState
import mujoco
import mediapy as media

InferenceFn = Callable[[jp.ndarray, jp.ndarray], tuple[jp.ndarray, jp.ndarray]]


def mjx_rollout(
    eval_env: envs.Env,
    inference_fn: InferenceFn,
    episode_length: int = 1000,
    seed: int = 0,
) -> list[mjxState]:
    """Rollout a trajectory using MJX

    Args:
        env: Brax environment
        inference_fn: Inference function #Not sure if we can obtain inference func during training. Neeed another way
        episode_length: Length of episode (timesteps)
        seed: Random seed

    Returns:
        The rollout trajectory for a provided length or until the episode ends
    """

    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)
    rollout = [state]

    # Run evaluation rollout
    for i in range(episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)
        if state.done:
            break
    return rollout


def render_mjx_rollout(
    env: envs.Env,
    inference_fn: InferenceFn,
    episode_length: int = 1000,
    render_every: int = 2,
    seed: int = 0,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Rollout a trajectory using MJX and render it.

    Args:
        env: Brax environment
        inference_fn: Inference function
        episode_length: Number of steps to rollout
        render_every: Render every nth step
        seed: Random seed
        width: width of rendered frame in pixels
        height: height of rendered frame in pixels

    Returns:
        A list of renderings of the policy rollout with dimensions (T, H, W, C)
    """
    rollout = mjx_rollout(env, inference_fn, episode_length, seed)
    traj = rollout[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    frames = env.render(
        traj, camera="track", height=height, width=width, scene_option=scene_option
    )

    return np.array(frames)


def save_mjx_rollout(
    env: envs.Env,
    inference_fn: InferenceFn,
    name: str,
    episode_length: int = 1000,
    render_every: int = 2,
    seed: int = 0,
    width: int = 640,
    height: int = 480,
):

    rollout = mjx_rollout(env, inference_fn, episode_length, seed)
    traj = rollout[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    frames = env.render(
        traj, camera="track", height=height, width=width, scene_option=scene_option
    )

    fps = 1.0 / env.dt / render_every
    media.write_video(f"{name}.mp4", frames, fps=fps)
