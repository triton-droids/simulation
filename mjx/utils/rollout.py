from typing import Callable

import jax
from brax import envs
from brax.mjx.base import State as mjxState

def mjx_rollout(
        eval_env: envs.Env,
        inference_fn: Callable,
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

