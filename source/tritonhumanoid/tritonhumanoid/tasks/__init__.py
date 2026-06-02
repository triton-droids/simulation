# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

try:
    from isaaclab_tasks.utils import import_packages
except ModuleNotFoundError as exc:
    if exc.name != "isaaclab_tasks":
        raise
else:
    # The blacklist is used to prevent importing configs from sub-packages
    _BLACKLIST_PKGS = ["utils", ".mdp"]
    # Import all configs in this package
    import_packages(__name__, _BLACKLIST_PKGS)
