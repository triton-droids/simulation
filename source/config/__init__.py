"""Register Hydra config groups for the simulator training entry points."""

from hydra.core.config_store import ConfigStore
from source.config.config import Config
from source.config.envs import HumanoidLegsEnv, UnitreeG1Env
from source.config.agents import G1PPOConfig, G1PPOSmokeConfig, PPOConfig
from source.config.robots import DefaultHumanoidLegsRobot, UnitreeG1Robot
from source.config.sim import MJXConfig
from source.config.g1 import G1MJXConfig

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

# env group
cs.store(group="env", name="default_humanoid_legs", node=HumanoidLegsEnv)
cs.store(group="env", name="unitree_g1", node=UnitreeG1Env)

# agent group
cs.store(group="agent", name="ppo", node=PPOConfig)
cs.store(group="agent", name="ppo_g1_smoke", node=G1PPOSmokeConfig)
cs.store(group="agent", name="ppo_g1", node=G1PPOConfig)

# robot group
cs.store(group="robot", name="humanoid_legs", node=DefaultHumanoidLegsRobot)
cs.store(group="robot", name="unitree_g1", node=UnitreeG1Robot)

# sim group
cs.store(group="sim", name="mjx", node=MJXConfig)
cs.store(group="sim", name="unitree_g1", node=G1MJXConfig)

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
