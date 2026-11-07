"""Verified name maps for the pinned 29-actuator Unitree G1 model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np


ACTUATOR_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

FOOT_SITES = ("left_foot", "right_foot")
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_COLLISION_GEOMS = (
    (
        "left_foot1_collision",
        "left_foot2_collision",
        "left_foot3_collision",
    ),
    (
        "right_foot1_collision",
        "right_foot2_collision",
        "right_foot3_collision",
    ),
)
CROSS_CONTACT_FOOT_GEOMS = (
    "left_foot_box_collision",
    "right_foot_box_collision",
)
SHIN_COLLISION_GEOMS = (
    ("left_shin_collision", "left_linkage_brace_collision"),
    ("right_shin_collision", "right_linkage_brace_collision"),
)
HAND_COLLISION_GEOMS = ("left_hand_collision", "right_hand_collision")
THIGH_COLLISION_GEOMS = ("left_thigh_collision", "right_thigh_collision")
TORSO_BODY = "torso_link"
PELVIS_BODY = "pelvis"
TORSO_IMU_SITE = "imu_in_torso"
PELVIS_IMU_SITE = "imu_in_pelvis"
DEFAULT_KEYFRAME = "knees_bent"
EXPECTED_SENSORS = (
    "local_linvel_torso",
    "local_linvel_pelvis",
    "accelerometer_torso",
    "accelerometer_pelvis",
    "gyro_torso",
    "gyro_pelvis",
    "upvector_torso",
    "upvector_pelvis",
    "orientation_torso",
    "orientation_pelvis",
    "global_linvel_torso",
    "global_linvel_pelvis",
    "global_angvel_torso",
    "global_angvel_pelvis",
)


@dataclass(frozen=True)
class G1Metadata:
    """Deterministically serializable model mapping used by tests and runs."""

    nq: int
    nv: int
    nu: int
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    actuator_joint_names: tuple[str, ...]
    qpos_addresses: tuple[int, ...]
    dof_addresses: tuple[int, ...]
    joint_ranges: tuple[tuple[float, float], ...]
    actuator_ctrl_ranges: tuple[tuple[float, float], ...]
    default_keyframe: str
    default_qpos: tuple[float, ...]
    foot_sites: tuple[str, str]
    foot_bodies: tuple[str, str]
    foot_body_ids: tuple[int, int]
    foot_link_ids: tuple[int, int]
    foot_geom_names: tuple[tuple[str, ...], tuple[str, ...]]
    foot_geom_ids: tuple[tuple[int, ...], tuple[int, ...]]
    cross_contact_foot_geom_names: tuple[str, str]
    cross_contact_foot_geom_ids: tuple[int, int]
    shin_geom_names: tuple[tuple[str, ...], tuple[str, ...]]
    shin_geom_ids: tuple[tuple[int, ...], tuple[int, ...]]
    hand_geom_names: tuple[str, str]
    hand_geom_ids: tuple[int, int]
    thigh_geom_names: tuple[str, str]
    thigh_geom_ids: tuple[int, int]
    floor_geom_id: int
    torso_body_id: int
    pelvis_body_id: int
    torso_imu_site_id: int
    pelvis_imu_site_id: int
    sensor_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, index: int) -> str:
    value = mujoco.mj_id2name(model, obj, index)
    if value is None:
        raise ValueError(f"Unnamed {obj!s} at index {index}")
    return value


def introspect_g1_model(model: mujoco.MjModel) -> G1Metadata:
    """Validate and return all fragile G1 mappings from the loaded model."""

    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError(
            f"Unexpected G1 dimensions {(model.nq, model.nv, model.nu)}; expected (36, 35, 29)."
        )

    joint_names = tuple(
        _name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(1, model.njnt)
    )
    actuator_names = tuple(
        _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    )
    actuator_joint_ids = tuple(int(value) for value in model.actuator_trnid[:, 0])
    actuator_joint_names = tuple(model.joint(index).name for index in actuator_joint_ids)
    if joint_names != ACTUATOR_JOINT_NAMES:
        raise ValueError(f"Unexpected G1 joint order: {joint_names}")
    if actuator_names != ACTUATOR_JOINT_NAMES:
        raise ValueError(f"Unexpected G1 actuator order: {actuator_names}")
    if actuator_joint_names != ACTUATOR_JOINT_NAMES:
        raise ValueError(f"Unexpected actuator-to-joint map: {actuator_joint_names}")

    qpos_addresses = tuple(int(model.jnt_qposadr[index]) for index in actuator_joint_ids)
    dof_addresses = tuple(int(model.jnt_dofadr[index]) for index in actuator_joint_ids)
    if qpos_addresses != tuple(range(7, 36)) or dof_addresses != tuple(range(6, 35)):
        raise ValueError(
            f"Unexpected scalar addresses: qpos={qpos_addresses}, dof={dof_addresses}"
        )

    joint_ranges_array = np.asarray(model.jnt_range[1:], dtype=float)
    ctrl_ranges_array = np.asarray(model.actuator_ctrlrange, dtype=float)
    if not np.allclose(joint_ranges_array, ctrl_ranges_array, atol=1e-12):
        raise ValueError("G1 actuator control ranges do not match their joint ranges.")

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, DEFAULT_KEYFRAME)
    if key_id < 0:
        raise ValueError(f"Required G1 keyframe {DEFAULT_KEYFRAME!r} is missing.")
    default_qpos_array = np.asarray(model.key_qpos[key_id], dtype=float)
    default_joints = default_qpos_array[7:]
    if np.any(default_joints < joint_ranges_array[:, 0]) or np.any(
        default_joints > joint_ranges_array[:, 1]
    ):
        raise ValueError("G1 default keyframe violates a joint range.")

    foot_site_ids = tuple(model.site(name).id for name in FOOT_SITES)
    foot_body_ids = tuple(int(model.site_bodyid[index]) for index in foot_site_ids)
    actual_foot_bodies = tuple(model.body(index).name for index in foot_body_ids)
    if actual_foot_bodies != FOOT_BODIES:
        raise ValueError(f"Unexpected G1 foot-site bodies: {actual_foot_bodies}")

    foot_geom_ids = tuple(
        tuple(model.geom(name).id for name in side) for side in FOOT_COLLISION_GEOMS
    )
    for side_ids, body_id in zip(foot_geom_ids, foot_body_ids, strict=True):
        if any(int(model.geom_bodyid[geom_id]) != body_id for geom_id in side_ids):
            raise ValueError("A G1 foot collision geom is attached to the wrong body.")

    cross_contact_foot_geom_ids = tuple(
        model.geom(name).id for name in CROSS_CONTACT_FOOT_GEOMS
    )
    for geom_id, body_id in zip(
        cross_contact_foot_geom_ids, foot_body_ids, strict=True
    ):
        if int(model.geom_bodyid[geom_id]) != body_id:
            raise ValueError("A G1 cross-contact foot geom is attached to the wrong body.")

    shin_geom_ids = tuple(
        tuple(model.geom(name).id for name in side) for side in SHIN_COLLISION_GEOMS
    )
    hand_geom_ids = tuple(model.geom(name).id for name in HAND_COLLISION_GEOMS)
    thigh_geom_ids = tuple(model.geom(name).id for name in THIGH_COLLISION_GEOMS)
    expected_named_bodies = (
        (shin_geom_ids, ("left_knee_link", "right_knee_link")),
        ((hand_geom_ids,), ("left_wrist_yaw_link", "right_wrist_yaw_link")),
        ((thigh_geom_ids,), ("left_hip_yaw_link", "right_hip_yaw_link")),
    )
    for grouped_ids, body_names in expected_named_bodies:
        if len(grouped_ids) == 1:
            grouped_ids = tuple((geom_id,) for geom_id in grouped_ids[0])
        actual = tuple(
            model.body(int(model.geom_bodyid[side_ids[0]])).name
            for side_ids in grouped_ids
        )
        if actual != body_names:
            raise ValueError(
                f"Unexpected collision-geom bodies {actual}; expected {body_names}."
            )

    sensor_names = tuple(
        _name(model, mujoco.mjtObj.mjOBJ_SENSOR, index)
        for index in range(model.nsensor)
    )
    if sensor_names != EXPECTED_SENSORS:
        raise ValueError(f"Unexpected G1 sensor order: {sensor_names}")

    return G1Metadata(
        nq=model.nq,
        nv=model.nv,
        nu=model.nu,
        joint_names=joint_names,
        actuator_names=actuator_names,
        actuator_joint_names=actuator_joint_names,
        qpos_addresses=qpos_addresses,
        dof_addresses=dof_addresses,
        joint_ranges=tuple(tuple(float(x) for x in row) for row in joint_ranges_array),
        actuator_ctrl_ranges=tuple(tuple(float(x) for x in row) for row in ctrl_ranges_array),
        default_keyframe=DEFAULT_KEYFRAME,
        default_qpos=tuple(float(x) for x in default_qpos_array),
        foot_sites=FOOT_SITES,
        foot_bodies=FOOT_BODIES,
        foot_body_ids=foot_body_ids,
        foot_link_ids=tuple(body_id - 1 for body_id in foot_body_ids),
        foot_geom_names=FOOT_COLLISION_GEOMS,
        foot_geom_ids=foot_geom_ids,
        cross_contact_foot_geom_names=CROSS_CONTACT_FOOT_GEOMS,
        cross_contact_foot_geom_ids=cross_contact_foot_geom_ids,
        shin_geom_names=SHIN_COLLISION_GEOMS,
        shin_geom_ids=shin_geom_ids,
        hand_geom_names=HAND_COLLISION_GEOMS,
        hand_geom_ids=hand_geom_ids,
        thigh_geom_names=THIGH_COLLISION_GEOMS,
        thigh_geom_ids=thigh_geom_ids,
        floor_geom_id=model.geom("floor").id,
        torso_body_id=model.body(TORSO_BODY).id,
        pelvis_body_id=model.body(PELVIS_BODY).id,
        torso_imu_site_id=model.site(TORSO_IMU_SITE).id,
        pelvis_imu_site_id=model.site(PELVIS_IMU_SITE).id,
        sensor_names=sensor_names,
    )
