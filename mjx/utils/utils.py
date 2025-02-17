from typing import any, List, Tuple
from brax.training.agents.ppo import networks as ppo_networks

def make_inference_fn(env, ppo_params, params):
    policy_network = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        **ppo_params.network_factory,
    )

    inference_fn = ppo_networks.make_inference_fn(policy_network)((params[0], params[1]['policy']), deterministic=True)

    return inference_fn
