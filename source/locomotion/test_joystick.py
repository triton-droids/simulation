"""Manual MJX environment smoke test for the default humanoid legs task."""

import jax
import jax.numpy as jnp
from jax import disable_jit

from source.robots.robot import Robot
from source.locomotion import get_env_class

# For illustration, we'll use a built-in env as a stand-in:
from brax.envs import create
from source.locomotion.default_humanoid_legs.joystick import Joystick
from source.config.sim import MJXConfig
from source.config.agents import PPOConfig

import hydra

@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    """Run one no-JIT reset/step smoke test for the configured joystick env.

    Args:
        cfg: Hydra config tree containing robot, environment, and simulator
            settings.

    Side effects:
        Prints initial/post-step state values to stdout.

    Failure cases:
        Missing MuJoCo/JAX/Brax dependencies or invalid robot scene paths will
        stop the smoke test before the step is printed.
    """

    rng = jax.random.PRNGKey(0)
    robot = Robot(cfg.robot.name)

    EnvClass = get_env_class(cfg.env.name)
    env_cfg = cfg.sim
    train_cfg = cfg.agent

    
    env = EnvClass(
        cfg.robot.name,
        robot,
        cfg.env.terrain, 
        env_cfg)

    with disable_jit():
        print("Initializing environment (no jit)...")
        state = env.reset(rng)
        print("\nInitial Position:")
    

        # Random action
        action = jnp.zeros(env.nu)
        state = env.step(state, action)

        print("\nPost-Step Position:")
        print(state.q)

        print("\nVelocity:")
        print(state.qd)

        print("\nReward:", state.reward)
        print("Done:", state.done)


if __name__ == "__main__":
    main()
