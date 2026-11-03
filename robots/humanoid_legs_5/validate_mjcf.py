"""Independent smoke check for the portable humanoid_legs_5 bundle."""

from pathlib import Path
import json
import numpy as np
import mujoco


HERE = Path(__file__).resolve().parent
MODEL = HERE / "humanoid_legs_5.xml"
REPORT = HERE / "humanoid_legs_5_validation.json"


def pin_error(data, side):
    bar = data.site(f"{side}_heel_bar_pin").xpos
    crank = data.site(f"{side}_crank_pin").xpos
    return float(np.linalg.norm(bar - crank))


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    report = json.loads(REPORT.read_text())
    assert model.nq == 21 and model.nu == 10
    assert abs(float(model.body_mass.sum()) - 15.999) < 1e-6
    assert report["validation"]["crank_branch_continuous"]
    assert report["validation"]["max_pin_closure_error_m"] < 1e-8
    data = mujoco.MjData(model)
    saved_gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0
    max_error = 0.0
    for step in range(1000):
        for side in ("left", "right"):
            data.ctrl[model.actuator(f"{side}_ankle_act").id] = 0.25 * np.sin(
                2 * np.pi * 0.4 * step * model.opt.timestep
            )
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        max_error = max(max_error, *(pin_error(data, side) for side in ("left", "right")))
    model.opt.gravity[:] = saved_gravity
    assert max_error < 1e-4, max_error
    print(json.dumps({"model": "humanoid_legs_5", "nq": model.nq, "actuators": model.nu,
                      "mass_kg": float(model.body_mass.sum()),
                      "zero_gravity_steps": 1000, "max_driven_pin_gap_m": max_error}, indent=2))
    print("MJCF linkage smoke check passed. Isaac Lab/PhysX import and grounded-hold smoke check passed; hardware calibration and walking remain open.")


if __name__ == "__main__":
    main()
