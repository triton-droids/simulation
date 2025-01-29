# Copyright 2025 Triton Droids

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Locomotion environments."""


import functools
from typing import Any, Dict

from mujoco_playground import MjxEnv
from .default_humanoid import joystick as default_humanoid_joystick

#Register all locomotion enviornmnets here
_envs = {
    "DefaultHumanoidJoystickFlatTerrain": functools.partial(
        default_humanoid_joystick.Joystick, terrain="flat_terrain"
    ),
    "DefaultHumanoidJoystickRoughTerrain": functools.partial(
        default_humanoid_joystick.Joystick, terrain="rough_terrain"
    )
}

#Register 'default' config paths here
_cfgs = {
    "DefaultHumanoidJoystickFlatTerrain": 'mjx\configs\DefaultHumanoid\locomotion_default.yml',
    "DefaultHumanoidJoystickRoughTerrain": 'mjx\configs\DefaultHumanoid\locomotion_default.yml'
}


ALL = list(_envs.keys()) 

def get_default_config(env_name: str) -> str:
  """Get the path to the default configuration for an environment."""
  if env_name not in _cfgs:
    raise ValueError(
        f"Env '{env_name}' not found in default configs. Available configs:"
        f" {list(_cfgs.keys())}"
    )
  return _cfgs[env_name]

def load(
    env_name: str,
    config: str
) -> MjxEnv:
  """Get an environment instance with the given configuration.

  Args:
      env_name: The name of the environment.
      config: The configuration to use. 

  Returns:
      An instance of the environment.
  """

  if env_name not in _envs:
    raise ValueError(f"Environment {env_name} not found in locomotion.")
  
  return _envs[env_name](config=config)

