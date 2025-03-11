import jax
from absl import logging
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mjx.utils.rollouts import render_mjx_rollout


def play(env, cfg, seed, model_path):
    env_cfg = cfg.env
    ppo_cfg = cfg.brax_ppo_agent

    rng = jax.random.PRNGKey(seed)
    env.reset(rng)

    logging.info(
        "Loaded environment %s with env.observation_size: %s and env.action_size: %s",
        cfg.env.name,
        env.observation_size,
        env.action_size,
    )
    params = model.load_params(model_path)

    if ppo_cfg.normalize_observations:
        normalize = running_statistics.normalize

    policy_network = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        **ppo_cfg.network_factory,
    )
    params = (params[0], params[1].policy)
    # Params are a tuple of (processor_params, PolicyNetwork)
    inference_fn = ppo_networks.make_inference_fn(policy_network)(params)
    print(f"Loaded params from {model_path}")
    print(inference_fn)

    # if args.use_mujoco:
    #     images_thwc = render_mujoco_rollout(env, inference_fn, n_steps, render_every, width=width, height=height)
    # else:
    #     images_thwc = render_mjx_rollout(env, inference_fn, n_steps, render_every, width=width, height=height)
    # print(f"Rolled out {len(images_thwc)} steps")
