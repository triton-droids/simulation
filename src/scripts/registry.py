# Copyright 2025 Triton Droids
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Registry for all environments."""
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
from ml_collections import ConfigDict
from mujoco import mjx
import functools

from ..envs import locomotion
from mujoco_playground._src import mjx_env
from ml_collections import config_dict

DomainRandomizer = Optional[
    Callable[[mjx.Model, jax.Array], Tuple[mjx.Model, mjx.Model]]
]

ALL_ENVS = locomotion.ALL


def get_default_config(env_name: str) -> str:
    if env_name in locomotion.ALL:
        return locomotion.get_default_config(env_name)

    raise ValueError(f"Env '{env_name}' not found in default configs.")


def load(
    env_name: str,
    config: ConfigDict,
) -> mjx_env.MjxEnv:
    if env_name in locomotion.ALL:
        return locomotion.load(env_name, config)

    raise ValueError(f"Env '{env_name}' not found. Available envs: {ALL_ENVS}")


def get_domain_randomizer(env_name: str) -> Optional[DomainRandomizer]:
    if env_name in locomotion.ALL:
        return locomotion.get_domain_randomizer(env_name)
