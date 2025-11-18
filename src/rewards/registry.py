from . import locomotion_rewards

REWARD_REGISTRY = {}

def register_reward(name):
    def decorator(fn):
        REWARD_REGISTRY[name] = fn
        return fn
    return decorator

def get_reward_function(name):
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Reward '{name}' not found.")
    return REWARD_REGISTRY[name]