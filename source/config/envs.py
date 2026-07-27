"""Environment selection defaults for the locomotion training workflow."""

from dataclasses import dataclass

@dataclass
class HumanoidLegsEnv:
    """Default locomotion task selector.

    The training code uses this to choose the environment implementation and
    scene terrain for the existing default humanoid legs robot.
    """

    name: str = "default_humanoid_legs"
    terrain: str = "flat"
