"""Small registry that maps reward names to reward functions."""

from . import locomotion_rewards

REWARD_REGISTRY = {}

def register_reward(name):
    """Register a reward function under a config-facing name.

    Args:
        name: Reward term name used in `source/config/sim.py`.

    Returns:
        Decorator that stores the function in `REWARD_REGISTRY` and returns the
        function unchanged.
    """

    def decorator(fn):
        """Store a reward function without changing its call signature.

        Args:
            fn: Reward function that accepts environment, pipeline state, info,
                and action arguments.

        Returns:
            The same function so normal decoration works.
        """

        REWARD_REGISTRY[name] = fn
        return fn
    return decorator

def get_reward_function(name):
    """Look up a reward function by name.

    Args:
        name: Registered reward name.

    Returns:
        Callable reward function.

    Failure cases:
        Raises ValueError when the config references an unknown reward term.
    """

    if name not in REWARD_REGISTRY:
        raise ValueError(f"Reward '{name}' not found.")
    return REWARD_REGISTRY[name]
