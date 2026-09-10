# Copyright 2026 Triton Droids
# Adapted from MuJoCo Playground commit
# 8a4b4642d8eba8a80ac99ed125cb62c16e1457ad (Apache-2.0).
"""Brax/MJX base adapter for the repository-pinned Unitree G1 scene."""

from __future__ import annotations

from typing import Any

from brax.envs.base import PipelineEnv
from brax.io import mjcf
import jax

from source.tools.mjx import get_sensor_data


class UnitreeG1Env(PipelineEnv):
    """Load the external pinned G1 scene without an in-repository asset copy."""

    def __init__(self, name: str, robot: Any, scene: str, cfg: Any, **kwargs: Any):
        if name != "unitree_g1":
            raise ValueError(f"UnitreeG1Env received robot name {name!r}")
        if scene != "flat":
            raise ValueError("The Gate 4 G1 prototype supports flat terrain only.")
        if robot.resolution.scene != "scene_mjx.xml":
            raise ValueError("G1 training requires the MJX-compatible scene_mjx.xml.")

        self.name = name
        self.robot = robot
        self.cfg = cfg
        self.model_source = robot.resolution
        self.metadata = robot.metadata

        sys = mjcf.load(str(robot.scene_path))
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

        self.nu = self.sys.nu
        self.nq = self.sys.nq
        self.nv = self.sys.nv
        self.obs_size = cfg.obs.num_single_obs
        self.privileged_obs_size = cfg.obs.num_single_privileged_obs
        self.add_domain_rand = cfg.domain_rand.add_domain_rand

    def get_sensor(self, pipeline_state: Any, name: str) -> jax.Array:
        return get_sensor_data(self.sys.mj_model, pipeline_state, name)

    def get_local_linvel(self, pipeline_state: Any, frame: str = "pelvis") -> jax.Array:
        return self.get_sensor(pipeline_state, f"local_linvel_{frame}")

    def get_global_linvel(self, pipeline_state: Any, frame: str = "pelvis") -> jax.Array:
        return self.get_sensor(pipeline_state, f"global_linvel_{frame}")

    def get_global_angvel(self, pipeline_state: Any, frame: str = "pelvis") -> jax.Array:
        return self.get_sensor(pipeline_state, f"global_angvel_{frame}")

    def get_gyro(self, pipeline_state: Any, frame: str = "pelvis") -> jax.Array:
        return self.get_sensor(pipeline_state, f"gyro_{frame}")

    def get_gravity(self, pipeline_state: Any, frame: str = "pelvis") -> jax.Array:
        return self.get_sensor(pipeline_state, f"upvector_{frame}")

    def get_accelerometer(self, pipeline_state: Any, frame: str = "pelvis") -> jax.Array:
        return self.get_sensor(pipeline_state, f"accelerometer_{frame}")
