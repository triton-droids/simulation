"""Initial simulation-only ankle balance experiment, not a hardware controller."""
import mujoco
import numpy as np


def balance_step(model, data, lateral_target=0.0):
    mujoco.mj_forward(model, data)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'floating_base')
    rotation = data.xmat[root].reshape(3, 3)
    tilt = np.arctan2(rotation[2, 1], rotation[2, 2])
    # World angular velocity projected onto the root's local ankle-pitch axis.
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, root, velocity, 1)
    target = np.clip(tilt + 0.1 * velocity[0], -0.4, 0.4)
    data.ctrl[:] = 0
    for name in ('left_hip2_act', 'right_hip2_act'):
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = lateral_target
    for name in ('left_ankle_act', 'right_ankle_act'):
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = target
    mujoco.mj_step(model, data)


def weight_shift_step(model, data):
    # Tested for 20 seconds at 0.2 radians, 0.1 Hz; both feet stayed loaded.
    # Larger shifts unloaded a foot and destabilized this simple controller.
    target = 0.2 * np.sin(2 * np.pi * 0.1 * max(0.0, data.time - 2.0))
    balance_step(model, data, target)


def lateral_balance_step(model, data):
    mujoco.mj_forward(model, data)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'floating_base')
    rotation = data.xmat[root].reshape(3, 3)
    sideways_tilt = np.arctan2(-rotation[2, 0], rotation[2, 2])
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, root, velocity, 1)
    requested_shift = 0.24 * np.sin(2 * np.pi * 0.1 * max(0.0, data.time - 2.0))
    correction = -2.0 * (sideways_tilt + 0.1 * velocity[1])
    balance_step(model, data, np.clip(requested_shift + correction, -0.4, 0.4))
