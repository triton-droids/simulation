"""Gate 2 tests for G1 provenance, mappings, and registration."""

from __future__ import annotations

import json
from pathlib import Path

from hydra import compose, initialize
import numpy as np
from omegaconf import OmegaConf
import pytest

import source.config  # noqa: F401 - imports register structured Hydra configs.
from source.config.g1 import G1MJXConfig
from source.locomotion import get_env_class
from source.locomotion.unitree_g1.joystick import Joystick
from source.robots import make_robot
from source.robots.unitree_g1 import MENAGERIE_COMMIT, UnitreeG1Model
from source.robots.unitree_g1.model import resolve_unitree_g1_model
from source.robots.unitree_g1.metadata import (
    ACTUATOR_JOINT_NAMES,
    CROSS_CONTACT_FOOT_GEOMS,
    DEFAULT_KEYFRAME,
    FOOT_BODIES,
    FOOT_COLLISION_GEOMS,
    FOOT_SITES,
    HAND_COLLISION_GEOMS,
    SHIN_COLLISION_GEOMS,
    THIGH_COLLISION_GEOMS,
)


@pytest.fixture(scope="module")
def g1_robot():
    return UnitreeG1Model(fetch=False)


def test_resolver_uses_exact_pin_and_mjx_scene(g1_robot):
    source = g1_robot.resolution
    assert source.pinned is True
    assert source.revision == MENAGERIE_COMMIT
    assert source.scene == "scene_mjx.xml"
    assert source.scene_path.name == "scene_mjx.xml"
    assert source.scene_path.parent.name == "unitree_g1"


def test_explicit_local_model_override_is_supported_and_marked_unpinned(g1_robot):
    override = resolve_unitree_g1_model(
        model_path=g1_robot.scene_path,
        mjx_scene=True,
        fetch=False,
    )
    assert override.scene_path == g1_robot.scene_path
    assert override.source == "local-override"
    assert override.pinned is False


def test_joint_actuator_and_contact_metadata_are_exact(g1_robot):
    metadata = g1_robot.metadata
    assert (metadata.nq, metadata.nv, metadata.nu) == (36, 35, 29)
    assert metadata.joint_names == ACTUATOR_JOINT_NAMES
    assert metadata.actuator_names == ACTUATOR_JOINT_NAMES
    assert metadata.actuator_joint_names == ACTUATOR_JOINT_NAMES
    assert metadata.qpos_addresses == tuple(range(7, 36))
    assert metadata.dof_addresses == tuple(range(6, 35))
    assert metadata.default_keyframe == DEFAULT_KEYFRAME
    assert len(metadata.default_qpos) == 36
    assert metadata.foot_sites == FOOT_SITES
    assert metadata.foot_bodies == FOOT_BODIES
    assert metadata.foot_geom_names == FOOT_COLLISION_GEOMS
    assert metadata.cross_contact_foot_geom_names == CROSS_CONTACT_FOOT_GEOMS
    assert set(metadata.cross_contact_foot_geom_ids).isdisjoint(
        set(metadata.foot_geom_ids[0]) | set(metadata.foot_geom_ids[1])
    )
    assert metadata.shin_geom_names == SHIN_COLLISION_GEOMS
    assert metadata.hand_geom_names == HAND_COLLISION_GEOMS
    assert metadata.thigh_geom_names == THIGH_COLLISION_GEOMS
    assert metadata.foot_link_ids == (6, 12)
    assert metadata.torso_body_id == 16
    assert metadata.pelvis_body_id == 1
    np.testing.assert_allclose(metadata.joint_ranges, metadata.actuator_ctrl_ranges, atol=1e-12)


def test_metadata_serialization_is_deterministic(g1_robot):
    first = json.dumps(g1_robot.metadata.to_dict(), sort_keys=True, separators=(",", ":"))
    second = json.dumps(g1_robot.metadata.to_dict(), sort_keys=True, separators=(",", ":"))
    assert first == second


def test_g1_assets_are_not_duplicated_in_source_tree():
    package_dir = Path("source/robots/unitree_g1")
    copied_assets = [
        path
        for path in package_dir.rglob("*")
        if path.suffix.lower() in {".xml", ".stl", ".obj", ".dae", ".png"}
    ]
    assert copied_assets == []


def test_hydra_and_environment_registration_select_g1():
    with initialize(version_base=None, config_path=None):
        cfg = compose(
            config_name="config",
            overrides=["env=unitree_g1", "robot=unitree_g1", "sim=unitree_g1"],
        )
    assert cfg.env.name == "unitree_g1"
    assert cfg.robot.name == "unitree_g1"
    assert cfg.sim.obs.num_single_obs == 103
    assert cfg.sim.obs.num_single_privileged_obs == 216
    assert get_env_class(cfg.env.name) is Joystick

    serialized = json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True)
    restored = OmegaConf.create(json.loads(serialized))
    assert restored.env.name == "unitree_g1"
    assert restored.robot.name == "unitree_g1"
    assert restored.sim.obs.num_single_obs == 103


def test_training_constructor_loads_external_scene(g1_robot):
    cfg = OmegaConf.structured(G1MJXConfig())
    env = get_env_class("unitree_g1")("unitree_g1", g1_robot, "flat", cfg)
    assert (env.nq, env.nv, env.nu) == (36, 35, 29)
    assert env.obs_size == 103
    assert env.privileged_obs_size == 216
    assert env.model_source.revision == MENAGERIE_COMMIT
    assert env.sys.mj_model.keyframe(DEFAULT_KEYFRAME).id >= 0


def test_robot_factory_selects_g1_config():
    config = OmegaConf.create(
        {
            "name": "unitree_g1",
            "model_path": None,
            "menagerie_root": None,
            "cache_root": None,
            "fetch_model": False,
        }
    )
    robot = make_robot(config)
    assert isinstance(robot, UnitreeG1Model)
    assert robot.resolution.revision == MENAGERIE_COMMIT
