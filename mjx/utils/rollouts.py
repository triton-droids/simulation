from typing import Callable

import jax
import jax.numpy as jp
import numpy as np
from brax import envs
from brax.mjx.base import State as mjxState
import mujoco

InferenceFn = Callable[[jp.ndarray, jp.ndarray], tuple[jp.ndarray, jp.ndarray]]


def mjx_rollout(
        eval_env: envs.Env,
        inference_fn: InferenceFn,
        episode_length: int = 1000,
        seed: int = 0,
) -> list[mjxState]:
    """ Rollout a trajectory using MJX

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
    """Rollout a trajectory using MuJoCo and render it.

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
    rollout = mjx_rollout(env, inference_fn, episode_length, render_every, seed)
    traj = rollout[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    frames = env.render(
        traj,
        camera="track",
        height=height, 
        width=width, 
        scene_option=scene_option
    )

    return np.array(frames)

def render_mujoco_rollout(
    env: mujoco.MjModel,
    inference_fn: InferenceFn,
    n_steps: int = 1000,
    render_every: int = 2,
    seed: int = 0,
    width: int = 320,
    height: int = 240,
) -> np.ndarray:
    """Rollout a trajectory using MuJoCo.

    Args:
        env: Brax environment
        inference_fn: Inference function
        n_steps: Number of steps to rollout
        render_every: Render every nth step
        seed: Random seed
        width: width of rendered frame in pixels
        height: height of rendered frame in pixels

    Returns:
        A list of images of the policy rollout (T, H, W, C)
    """
    print(f"Rolling out {n_steps} steps with MuJoCo")
    model = env.sys.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    ctrl = jp.zeros(model.nu)

    images: list[np.ndarray] = []
    rng = jax.random.PRNGKey(seed)
    for step in tqdm(range(n_steps)):
        act_rng, seed = jax.random.split(rng)
        obs = env._get_obs(mjx.put_data(model, data), ctrl)
        # TODO: implement methods in envs that avoid having to use mjx in a hacky way...
        # print(obs)
        ctrl, _ = inference_fn(obs, act_rng)
        data.ctrl = ctrl
        for _ in range(env._n_frames):
            mujoco.mj_step(model, data)

        if step % render_every == 0:
            renderer.update_scene(data, camera="side")
            images.append(renderer.render())

    return np.array(images)