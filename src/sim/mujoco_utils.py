import os
import time
import warnings
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import mujoco.rollout
import mujoco.viewer
import numpy as np
from moviepy import VideoFileClip, clips_array

from src.robots.robot import Robot

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")


def mj_render(model, data, lib="plt"):
    """Renders a MuJoCo simulation scene using the specified library.

    Args:
        model: The MuJoCo model to be rendered.
        data: The simulation data associated with the model.
        lib (str): The library to use for rendering. Options are "plt" for matplotlib and any other value for OpenCV. Defaults to "plt".

    Raises:
        ValueError: If the specified library is not supported.
    """
    renderer = mujoco.Renderer(model)
    renderer.update_scene(data)
    pixels = renderer.render()

    if lib == "plt":
        plt.imshow(pixels)
        plt.show()
    else:
        pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.imshow("Simulation", pixels_bgr)
        cv2.waitKey(1)  # This ensures the window updates without blocking


class MujocoViewer:
    """A class for visualizing MuJoCo simulation data.
    
    Provides an interface for launching a passive MuJoCo viewer and visualizing 
    simulation-specific elements such as the robot's center of mass and support polygon. 
    """

    def __init__(self, robot: Robot, model: Any, data: Any):
        """Initializes the class with a robot, model, and data, and sets up the viewer and foot geometry.

        Args:
            robot (Robot): The robot instance containing configuration and state information.
            model (Any): The model object representing the simulation environment.
            data (Any): The data object containing the simulation state.

        Attributes:
            robot (Robot): Stores the robot instance.
            model (Any): Stores the model object.
            viewer: Launches a passive viewer for the simulation using the provided model and data.
            foot_names (list of str): List of foot collision geometry names based on the robot's foot name.
            local_bbox_corners (np.ndarray): Local coordinates of the bounding box corners for the foot geometry.
        """
        self.robot = robot
        self.model = model

        self.viewer = mujoco.viewer.launch_passive(model, data)

    def visualize(
        self,
        data: Any,
        vis_flags: List[str] = ["com"],
    ):
        """Visualizes specified components of the data using the viewer.

        Args:
            data (Any): The data to be visualized.
            vis_flags (List[str], optional): A list of visualization flags indicating which components to visualize. Defaults to ["com", "support_poly"].
        """
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            if "com" in vis_flags:
                self.visualize_com(data)
   
        self.viewer.sync()

    def visualize_com(self, data: Any):
        """Visualize the center of mass (COM) of a given body in the simulation environment.

        This function adds a visual representation of the center of mass for a specified body
        to the simulation viewer. The COM is depicted as a small red sphere.

        Args:
            data (Any): The simulation data object containing information about the bodies,
                including their center of mass positions.
        """
        i = self.viewer.user_scn.ngeom
        com_pos = np.array(data.body(0).subtree_com, dtype=np.float32)
        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.01, 0.01, 0.01]),  # Adjust size of the sphere
            pos=com_pos,
            mat=np.eye(3).flatten(),
            rgba=[1, 0, 0, 1],
        )
        self.viewer.user_scn.ngeom = i + 1


    def close(self):
        """Closes the viewer associated with the current instance."""
        self.viewer.close()


class MuJoCoRenderer:
    """A class for rendering MuJoCo simulation data and recording video output.

    Provides functionality to record and visualize the robot's motion over time. 
    """

    def __init__(self, model: Any, height: int = 360, width: int = 640):
        """Initializes the object with a given model and sets up a renderer.

        Args:
            model (Any): The model to be used for rendering.
            height (int, optional): The height of the rendering window. Defaults to 360.
            width (int, optional): The width of the rendering window. Defaults to 640.
        """
        self.model = model
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.qpos_data = []
        self.qvel_data = []


    def visualize(self, data: Any, vis_data: Dict[str, Any] = {}):
        """Visualizes the given data by updating pose, position, and velocity information.

        This method processes the input data to update the animation pose and appends
        the position and velocity data to their respective lists for further visualization.

        Args:
            data (Any): The input data containing pose, position, and velocity information.
            vis_data (Dict[str, Any], optional): Additional visualization data. Defaults to an empty dictionary.
        """
        self.qpos_data.append(data.qpos.copy())
        self.qvel_data.append(data.qvel.copy())

    def save_recording(
        self,
        name: str = "mujoco.mp4",
        exp_folder_path: str = "",
        render_every: int = 2,
        dt: float = 0.001,
    ):
        """Saves a recording of the simulation from multiple camera angles.

        Args:
            exp_folder_path (str): The path to the folder where the recording and data will be saved.
            dt (float): The time step duration for rendering frames.
            render_every (int): The interval at which frames are rendered.
            name (str, optional): The name of the final video file. Defaults to "mujoco.mp4".
        """

        # Define paths for each camera's video
        video_paths = []
        # Render and save videos for each camera
        for camera in ["back", "side"]:
            video_path = os.path.join(exp_folder_path, f"{camera}.mp4")
            video_frames = []
            for qpos, qvel in zip(
                self.qpos_data[::render_every], self.qvel_data[::render_every]
            ):
                d = mujoco.MjData(self.model)
                d.qpos, d.qvel = qpos, qvel
                mujoco.mj_forward(self.model, d)
                self.renderer.update_scene(d, camera=camera)
                video_frames.append(self.renderer.render())

            media.write_video(video_path, video_frames, fps=1.0 / dt / render_every)
            video_paths.append(video_path)

        # Delay to ensure the video files are fully written
        time.sleep(1)
        
        # Load the video clips using moviepy
        clips = [VideoFileClip(path) for path in video_paths]
        # Arrange the clips in a 2x2 grid
        final_video = clips_array([[clips[0], clips[1]]])
        # Save the final concatenated video
        final_video.write_videofile(os.path.join(exp_folder_path, name))


    def close(self):
        """Closes the renderer associated with the current instance.

        This method ensures that the renderer is properly closed and any resources
        associated with it are released.
        """
        self.renderer.close()
