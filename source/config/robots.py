"""Robot selection defaults used by the existing training configuration."""

from dataclasses import dataclass

@dataclass
class DefaultHumanoidLegsRobot:
    """Default robot selector for the existing training environment.

    Unitree G1 is intentionally loaded by `scripts/load_unitree_g1.py` for now;
    this config still points at the original robot used by the MJX task.
    """

    name: str = "default_humanoid_legs"
    
