import mujoco
import numpy as np
import numpy.typing as npt
from typing import Dict
from time import time

from src.robots.robot import Robot
from src.sim.mujoco_utils import MuJoCoRenderer, MuJoCoViewer
from src.sim.sim_types import JointState


class MujocoSim:
    """ 
    A class for the MuJoCo simulation environment.

    It provides an interface for interacting with a model in the MuJoCo simulation environment, 
    allowing functions to control and observe the model's behavior. It supports retrieving motor and joint states, 
    setting joint angles, advancing the simulation, visualizing the robot's movements, and computing transformation matrices. 
    Includes features like applying random forces to the robot and resetting the simulation.
    """
    def __init__(
        self,
        robot: Robot,
        n_frames: int = 20,
        dt: float = 0.001,
        xml_path: str = "",
        vis_type: str = "render",
    ):
        """ Initalizes the simulation environment for a robot using the MuJoCo physics engine. """

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.dt = dt
        self.n_frames = n_frames
        self.model.opt.timestep = self.dt
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.model.opt.iterations = 1
        self.model.opt.ls_iterations = 4

        self.robot = robot

        if vis_type == "render":
            self.visualizer = MuJoCoRenderer(self.model)
        elif vis_type == "view":
            self.visualizer = MuJoCoViewer(robot, self.model, self.data)
        
    
    def get_body_transform(self, body_name: str):
        """Computes the transformation matrix for a specified body.

        Args:
            body_name (str): The name of the body for which to compute the transformation matrix.

        Returns:
            np.ndarray: A 4x4 transformation matrix representing the position and orientation of the body.
        """
        transformation = np.eye(4)
        body_pos = self.data.body(body_name).xpos.copy()
        body_mat = self.data.body(body_name).xmat.reshape(3, 3).copy()
        transformation[:3, :3] = body_mat
        transformation[:3, 3] = body_pos
        return transformation

    def get_site_transform(self, site_name: str):
        """Retrieves the transformation matrix for a specified site.

        This method constructs a 4x4 transformation matrix for the given site name,
        using the site's position and orientation matrix. The transformation matrix
        is composed of a 3x3 rotation matrix and a 3x1 translation vector.

        Args:
            site_name (str): The name of the site for which to retrieve the transformation matrix.

        Returns:
            numpy.ndarray: A 4x4 transformation matrix representing the site's position and orientation.
        """
        transformation = np.eye(4)
        site_pos = self.data.site(site_name).xpos.copy()
        site_mat = self.data.site(site_name).xmat.reshape(3, 3).copy()
        transformation[:3, :3] = site_mat
        transformation[:3, 3] = site_pos
        return transformation
        

    def get_motor_state(self) -> Dict[str, JointState]: 
        """Retrieve the current state of each motor in the robot.

        Returns:
            Dict[str, JointState]: A dictionary mapping each motor's name to its
            current state, including position, velocity, and torque.
        """
        motor_state_dict = {}
        for name in self.robot.actuators.keys():
            motor_state_dict[name] = JointState(
                time=time(),
                pos=self.data.joint(name).qpos.item(),
                vel=self.data.joint(name).qvel.item(),
                tor=self.data.actuator(name).force.item(),
            )

        return motor_state_dict
    
    def get_motor_angles(
        self, 
        type: str = "dict"
    ) -> Dict[str, float] | npt.NDArray[np.float32]:
        """Retrieves the current angles of the robot's motors.

        Args:
            type (str): The format in which to return the motor angles.
                Options are "dict" for a dictionary format or "array" for a NumPy array.
                Defaults to "dict".

        Returns:
            Dict[str, float] or npt.NDArray[np.float32]: The motor angles in the specified format.
            If "dict", returns a dictionary with motor names as keys and angles as values.
            If "array", returns a NumPy array of motor angles.
        """
        motor_angles = {}
        for name in self.robot.actuators.keys():
            motor_angles[name] = self.data.joint(name).qpos.item()

        if type == "array":
            motor_pos_arr = np.array(list(motor_angles.values()), dtype=np.float32)
            return motor_pos_arr
        else:
            return motor_angles
        
    def get_joint_state(self) -> Dict[str, JointState]:
        """Retrieves the current state of each joint in the robot.

        Returns:
            Dict[str, JointState]: A dictionary mapping each joint's name to its current state,
            which includes the timestamp, position, and velocity.
        """
        joint_state_dict = {}
        for name in self.robot.joints.keys():
            joint_state_dict[name] = JointState(
                time=time(),
                pos=self.data.joint(name).qpos.item(),
                vel=self.data.joint(name).qvel.item(),
            )

        return joint_state_dict

    def get_joint_angles(
        self, 
        type: str = "dict"
    ) -> Dict[str, float] | npt.NDArray[np.float32]:
        """Retrieves the current joint angles of the robot.

        Args:
            type (str): The format in which to return the joint angles.
                Options are "dict" for a dictionary format or "array" for a NumPy array.
                Defaults to "dict".

        Returns:
            Dict[str, float] or npt.NDArray[np.float32]: The joint angles of the robot.
                Returns a dictionary with joint names as keys and angles as values if
                `type` is "dict". Returns a NumPy array of joint angles if `type` is "array".
        """
        joint_angles = {}
        for name in self.robot.joints.keys():
            joint_angles[name] = self.data.joint(name).qpos.item()

        if type == "array":
            joint_pos_arr = np.array(list(joint_angles.values()), dtype=np.float32)
            return joint_pos_arr
        else:
            return joint_angles
        
    def get_sensor_data(
        self, 
        type: str = "dict"
    ) -> Dict[str, float] | npt.NDArray[np.float32]:
        """ Retrieve the current sensor values of the robot.
        
        Args:
            type (str): The format in which to return the sensor values.
                Options are "dict" for a dictionary format or "array" for a NumPy array.
                Defaults to "dict".

        Returns:
            Dict[str, float] or npt.NDArray[np.float32]: The sensor values of the robot.
                Returns a dictionary with sensor names as keys and values as values if
                `type` is "dict". Returns a NumPy array of sensor values if `type` is "array".
        """
        sensor_values = {}
        for name in self.robot.sensors.keys():
            sensor_values[name] = self.data.sensor(name).data

        if type == "array":
            sensor_arr = np.array(sensor_values.values())
            return sensor_arr
        else:
            return sensor_values
    
    def set_joint_angles(
        self, 
        joint_angles: Dict[str, float] | npt.NDArray[np.float32]
    ):
        """Sets the joint angles of the robot.

        This method updates the joint positions of the robot based on the provided joint angles. 

        Args:
            joint_angles (Dict[str, float] | npt.NDArray[np.float32]): A dictionary mapping joint names to their respective angles, 
            or a NumPy array of joint angles in the order specified by the robot's joint ordering.
        """
        if not isinstance(joint_angles, dict):
            joint_angles = dict(zip(self.robot.joints.keys(), joint_angles))

        for name in joint_angles:
            self.data.joint(name).qpos = joint_angles[name]


    def set_qpos(self, qpos: npt.NDArray[np.float32]):
        """Set the position of the system's generalized coordinates.

        Args:
            qpos (npt.NDArray[np.float32]): An array representing the desired positions of the system's generalized coordinates.
        """
        self.data.qpos = qpos
        
    def reset(self):
        """Reset the simulation to its initial state."""
        mujoco.mj_resetData(self.model, self.data)
        
    def step(self):
        """Advances the simulation by a specified number of frames and updates the visualizer.

        This method iterates over the number of frames defined by `n_frames`, 
        and advances the simulation state using Mujoco's `mj_step` function. 
        If a visualizer is provided, it updates the visualization with the current simulation data.
        """
        for _ in range(self.n_frames):
            mujoco.mj_step(self.model, self.data)

        self.visualizer.visualize(self.data)

    def forward(self):
        """Advances the simulation forward by a specified number of frames and visualizes the result if a visualizer is available.

        Iterates through the simulation for the number of frames specified by `self.n_frames`, updating the model state at each step. 
        If a visualizer is provided, it visualizes the current state of the simulation data.
        """
        for _ in range(self.n_frames):
            mujoco.mj_step(self.model, self.data)

        self.visualizer.visualize(self.data)

    def get_obs(self):
        """ Get the observation from the simulation."""
        pass

    def close(self):
        """ Closes the visualizer if it is currently open."""
        self.visualizer.close()
    
    def push(
        self, 
        low: float = 100, 
        high: float = 200
    ):
        """ Apply a random push to the robot.
        
        Args:
            low (float): The lower bound of the random force magnitude.
            high (float): The upper bound of the random force magnitude.
        """
        theta = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(low, high)  

        force = np.array([
            np.cos(theta) * magnitude,  
            np.sin(theta) * magnitude,  
            0.0,                        
            0.0, 0.0, 0.0               
        ])
        
        self.data.xfrc_applied[self.robot.bodies["torso"]["body_id"]] = force




