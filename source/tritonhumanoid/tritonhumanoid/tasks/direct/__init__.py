# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

try:
    import gymnasium as gym  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "gymnasium":
        raise
