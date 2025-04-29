from dataclasses import dataclass, field
from src.config.envs import HumanoidLegsEnv
from src.config.agents import PPOConfig
from src.config.robots import DefaultHumanoidLegsRobot
from src.config.sim import MJXConfig

@dataclass
class Config:
    task: str = "locomotion" 
    env: object = field(default_factory=HumanoidLegsEnv)
    agent: object = field(default_factory=PPOConfig)
    robot: object = field(default_factory=DefaultHumanoidLegsRobot)
    sim: object = field(default_factory=MJXConfig)
    seed: int = 42

