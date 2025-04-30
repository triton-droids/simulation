import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from src.robots import Robot

class BaseRef(ABC):
    """Abstract class for generating motion references"""
    def __init__(
        self, 
        name: str,
        motion_type: str,
        robot: Robot,
        dt: float,
        ):
        """ Initializes the motion controller for a robot with specified parameters
        
        Args:
            name: Name of the motion reference
            motion_type: Type of motion reference
            robot: Robot object
            dt: Time step
            com_kp: Proportional gain for center of mass
        """
        self.name = name
        self.motion_type = motion_type
        self.robot = robot
        self.dt = dt


    def _setup_waist(self):
        pass

    def _setup_legs(self):
        pass

    def _setup_mjx(self):
        pass


    def get_motion_ref(self):
        pass