
from src.locomotion.default_humanoid_legs.joystick import Joystick
import functools

_envs = {
    "default_humanoid_legs": Joystick

}

def get_env_class(env_name: str, terrain: str = None):
    """Returns the environment class associated with the given environment name.

    Args:
        env_name (str): The name of the environment to retrieve.

    Returns:
        Type[MJXEnv]: The class of the specified environment.

    Raises:
        ValueError: If the environment name is not found in the registry.
    """
    if env_name not in _envs:
        raise ValueError(f"Unknown env: {env_name}")
    
    return _envs[env_name]
