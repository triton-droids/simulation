from dataclasses import dataclass

@dataclass
class JointState:
    """Data class for storing joint state information"""

    time: float
    pos: float
    vel: float = 0.0
    tor: float = 0.0


    
