import os
from src.motion_ref.base_ref import BaseRef
from src.motion_generation.zmp_walk import ZMPWalk
from src.robots.robot import Robot

class WalkZMPRef(BaseRef):
    """ Class for generating a ZMP-based walking refernce for a robot. """

    def __init__(
        self,
        robot: Robot,
        dt: float,
        cycle_time: float,
        waist_roll_max: float
    ):
        """ Initalizes the walk ZMP controller.

         Args:
            robot (Robot): The robot instance to be controlled.
            dt (float): The time step for the controller.
            cycle_time (float): The duration of one walking cycle.
            waist_roll_max (float): The maximum allowable roll angle for the robot's waist.
        """
        super().__init__("walk_zmp", "periodic", robot, dt)

        self.cycle_time = cycle_time
        self.waist_roll_max = waist_roll_max

        self._setup_zmp()
        
    def _setup_zmp(self):
        """ Initalizes the Zero Moment Point (ZMP) walking parameters and lookup tables for the robot.
        
        This method sets up the necessary inidices for hip joints.
        """
        self.zmp_walk = ZMPWalk(
            self.robot,
            self.cycle_time,
            self.waist_roll_max
        )
    
