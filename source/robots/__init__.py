"""Robot factory preserving the in-tree baseline and pinned external G1."""

from __future__ import annotations

from typing import Any

from source.robots.robot import Robot
from source.robots.unitree_g1 import UnitreeG1Model


def make_robot(robot_cfg: Any, *, config_path: str | None = None, xml_path: str | None = None):
    """Construct the selected robot adapter from a Hydra/dataclass config."""

    name = robot_cfg.name
    if name == "unitree_g1":
        if config_path is not None or xml_path is not None:
            raise ValueError(
                "G1 runs reconstruct the pinned include-based model through its resolver; "
                "copied XML/config resume paths are not supported."
            )
        return UnitreeG1Model(
            model_path=getattr(robot_cfg, "model_path", None),
            menagerie_root=getattr(robot_cfg, "menagerie_root", None),
            cache_root=getattr(robot_cfg, "cache_root", None),
            fetch=getattr(robot_cfg, "fetch_model", True),
        )
    if name == "default_humanoid_legs":
        return Robot(name, config_path=config_path, xml_path=xml_path)
    raise ValueError(f"Unknown robot: {name}")
