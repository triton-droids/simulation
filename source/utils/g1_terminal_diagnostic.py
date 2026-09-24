"""Read-only decomposition of the pinned G1 termination predicate."""
import numpy as np

SENSORS = {
    "foot_foot": "_right_foot_left_foot_found_sensor",
    "left_foot_right_shin": "_left_foot_right_shin_found_sensor",
    "right_foot_left_shin": "_right_foot_left_shin_found_sensor",
}

def terminal_signals(env, data):
    gravity = np.asarray(env.get_gravity(data, "torso"))
    signals = {"inverted": np.asarray(gravity[-1] < 0),
               "invalid": np.asarray(not (np.isfinite(np.asarray(data.qpos)).all()
                                           and np.isfinite(np.asarray(data.qvel)).all()))}
    for name, attribute in SENSORS.items():
        address = env.mj_model.sensor_adr[getattr(env, attribute)]
        signals[name] = np.asarray(np.asarray(data.sensordata)[address] > 0)
    return {"torso_gravity": gravity, **{"terminal/" + k: v for k, v in signals.items()}}
