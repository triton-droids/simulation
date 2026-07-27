"""Top-level Hydra config dataclass tying robot, environment, agent, and sim defaults together."""

from dataclasses import dataclass, field
from source.config.envs import HumanoidLegsEnv
from source.config.agents import PPOConfig
from source.config.robots import DefaultHumanoidLegsRobot
from source.config.sim import MJXConfig

@dataclass
class Config:
    """Top-level Hydra configuration for simulator training.

    This object groups the default environment, agent, robot, simulator, and
    seed settings that the training entry point receives as one config tree.
    """

    task: str = "locomotion" 
    env: object = field(default_factory=HumanoidLegsEnv)
    agent: object = field(default_factory=PPOConfig)
    robot: object = field(default_factory=DefaultHumanoidLegsRobot)
    sim: object = field(default_factory=MJXConfig)
    seed: int = 42

