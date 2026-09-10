"""Robot selection defaults used by the existing training configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DefaultHumanoidLegsRobot:
    """Default robot selector for the existing training environment.

    This remains separate from the registered ``UnitreeG1Robot`` selector so
    G1 assumptions cannot leak into the 12-actuator regression baseline.
    """

    name: str = "default_humanoid_legs"


@dataclass
class UnitreeG1Robot:
    """Resolver options for the pinned external Unitree G1 model."""

    name: str = "unitree_g1"
    model_path: Optional[str] = None
    menagerie_root: Optional[str] = None
    cache_root: Optional[str] = None
    fetch_model: bool = True
