# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments when IsaacLab is available. MuJoCo eval helpers are
# intentionally importable in lightweight Python environments without IsaacLab.
try:
    from .tasks import *  # noqa: F401,F403
except ModuleNotFoundError as exc:
    if exc.name not in {"isaaclab", "isaaclab_tasks", "isaacsim"}:
        raise

# Register UI extensions when IsaacLab/Omniverse dependencies are available.
try:
    from .ui_extension_example import *  # noqa: F401,F403
except ModuleNotFoundError as exc:
    if exc.name not in {"isaaclab", "isaaclab_tasks", "isaacsim", "omni"}:
        raise
