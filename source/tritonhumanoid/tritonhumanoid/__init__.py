# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments when Isaac Lab is available.
try:
    from .tasks import *
except ModuleNotFoundError as exc:
    if exc.name != "isaaclab_tasks":
        raise

# Register UI extensions when Omniverse is available.
try:
    from .ui_extension_example import *
except ModuleNotFoundError as exc:
    if exc.name != "omni":
        raise
