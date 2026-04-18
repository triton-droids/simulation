#!/usr/bin/env python3
"""Test script to verify all environment configs load correctly."""

import sys
sys.path.insert(0, '.')

from envs.locomotion_env import HumanoidLocomotionEnv
import mujoco

configs = [
    ("slippery/slippery_low", "Slippery Low"),
    ("slippery/slippery_medium", "Slippery Medium"),
    ("slippery/slippery_high", "Slippery High"),
    ("bouncy/bouncy_low", "Bouncy Low"),
    ("bouncy/bouncy_medium", "Bouncy Medium"),
    ("bouncy/bouncy_high", "Bouncy High"),
]

print("=" * 60)
print("Testing all environment configs")
print("=" * 60)

for config_path, config_name in configs:
    print(f"\n{config_name}:")
    print("-" * 40)

    try:
        env = HumanoidLocomotionEnv(
            xml_path="robot_description/scene.xml",
            system_config_path=f"robot_description/system_configs/{config_path}.json"
        )

        # Check floor friction
        floor_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        friction = env.model.geom_friction[floor_geom_id]
        solref = env.model.geom_solref[floor_geom_id]
        solimp = env.model.geom_solimp[floor_geom_id]

        print(f"  Friction: [{friction[0]:.2f}, {friction[1]:.4f}, {friction[2]:.4f}]")
        print(f"  Solref: [{solref[0]:.2f}, {solref[1]:.2f}]")
        print(f"  Solimp: [{solimp[0]:.2f}, {solimp[1]:.3f}, {solimp[2]:.3f}]")
        print(f"  ✓ Loaded successfully")

    except Exception as e:
        print(f"  ✗ Failed to load: {e}")

print("\n" + "=" * 60)
print("All configs tested!")
print("=" * 60)
