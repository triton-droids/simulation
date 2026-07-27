"""Register Hydra config groups for the simulator training entry points."""

from hydra.core.config_store import ConfigStore
from source.config.config import Config
from source.config.envs import HumanoidLegsEnv
from source.config.agents import PPOConfig
from source.config.robots import DefaultHumanoidLegsRobot
from source.config.sim import MJXConfig

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

# env group
cs.store(group="env", name="default_humanoid_legs", node=HumanoidLegsEnv)

# agent group
cs.store(group="agent", name="ppo", node=PPOConfig)

# robot group
cs.store(group="robot", name="humanoid_legs", node=DefaultHumanoidLegsRobot)

# sim group
cs.store(group="sim", name="mjx", node=MJXConfig)

def get_config(config_name: str):
    """Fetch a registered Hydra config by name.

    Args:
        config_name: Name previously registered with Hydra's ConfigStore.

    Returns:
        The registered config node.

    Failure cases:
        Hydra raises if the name is not registered.
    """

    return cs.get(config_name)
