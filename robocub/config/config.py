"""Top-level Hydra config dataclass tying robot, environment, agent, and sim defaults together."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

@dataclass
class Config:
    """Top-level Hydra configuration for simulator training.

    This object groups the default environment, agent, robot, simulator, and
    seed settings that the training entry point receives as one config tree.
    """

    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"env": "default_humanoid_legs"},
            {"agent": "ppo"},
            {"robot": "humanoid_legs"},
            {"sim": "mjx"},
        ]
    )
    task: str = "locomotion"
    env: Any = MISSING
    agent: Any = MISSING
    robot: Any = MISSING
    sim: Any = MISSING
    seed: int = 42

