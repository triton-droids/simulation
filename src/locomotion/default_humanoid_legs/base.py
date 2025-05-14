from src.utils.file_utils import find_robot_file_path
from brax.envs.base import PipelineEnv
from brax.io import mjcf
import jax
from typing import Any
import jax.numpy as jp
from src.tools.mjx import get_sensor_data

class DefaultHumanoidEnv(PipelineEnv):
    def __init__(self, 
            name: str,
            robot: any,
            scene: str,
            cfg,
            **kwargs: Any,
    ):
        self.name = name
        self.robot = robot
        self.cfg = cfg

        scene_xml_path = find_robot_file_path(robot.name, scene, '.xml')

        sys = mjcf.load(
            scene_xml_path,
        )
        sys = sys.tree_replace(
            {
                "opt.timestep": cfg.sim.timestep,
                "opt.solver": cfg.sim.solver,
                "opt.iterations": cfg.sim.iterations,
                "opt.ls_iterations": cfg.sim.ls_iterations,
            }
        )
        kwargs["n_frames"] = cfg.action.n_frames
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

    def get_gravity(self, pipeline_state) -> jax.Array:
        """Get gravity vector from the model."""
        return get_sensor_data(
            self.sys.mj_model, pipeline_state, "upvector"
        )

    def get_global_linvel(self, pipeline_state) -> jax.Array:
        """Return the linear velocity of the robot in the world frame."""
        return get_sensor_data(
            self.sys.mj_model, pipeline_state, "global_linvel"
        )

    def get_global_angvel(self, pipeline_state) -> jax.Array:
        """Return the angular velocity of the robot in the world frame."""
        return get_sensor_data(
            self.sys.mj_model, pipeline_state, "global_angvel"
        )

    def get_local_linvel(self, pipeline_state) -> jax.Array:
        """Return the linear velocity of the robot in the local frame."""
        return get_sensor_data(
            self.sys.mj_model, pipeline_state, "local_linvel"
        )

    def get_accelerometer(self, pipeline_state) -> jax.Array:
        """Return the accelerometer readings in the local frame."""
        return get_sensor_data(
            self.sys.mj_model, pipeline_state, "accelerometer"
        )

    def get_gyro(self, pipeline_state) -> jax.Array:
        """Return the gyroscope readings in the local frame."""
        return get_sensor_data(self.sys.mj_model, pipeline_state, "gyro")
    
    def get_feet_pos(self, pipeline_state) -> jax.Array:
        """Return the position of the feet in the world frame."""
        return jp.vstack([
            get_sensor_data(self.sys.mj_model, pipeline_state, sensor_name)
            for sensor_name in self.feet_pos_sensor_names
        ])
    
