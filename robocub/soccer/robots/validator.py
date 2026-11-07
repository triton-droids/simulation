"""Numerical asset validation; simulation metadata never verifies physical identity."""
from pathlib import Path
import numpy as np
from .importer import inspect_asset

def validate_mjcf(path):
    import mujoco
    result=inspect_asset(path);model=mujoco.MjModel.from_xml_path(str(Path(path).resolve()));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    dynamic=model.body_mass[1:]>0
    result.update(total_mass_kg=float(model.body_mass.sum()),joints=model.njnt,actuators=model.nu,sensors=model.nsensor,
        moving_body_coms=model.body_ipos[1:].tolist(),inertia_positive=bool(np.all(model.body_inertia[1:][dynamic]>0)),
        joint_axes_unit=bool(np.allclose(np.linalg.norm(model.jnt_axis[model.jnt_type>=2],axis=1),1)),
        finite_state=bool(np.isfinite(data.qpos).all()),collision_geom_count=int(np.sum((model.geom_contype!=0)|(model.geom_conaffinity!=0))),
        explicit_contact_pairs=model.npair,initial_max_penetration_m=float(max([0.]+[-data.contact[i].dist for i in range(data.ncon)])),
        root_height_m=float(data.qpos[2]) if model.nq>=7 else None,
        scaling_note='SI assumption requires operator confirmation; dimensions/COM inspected, not calibrated',
        control_ranges=model.actuator_ctrlrange.tolist(),effort_ranges=model.actuator_forcerange.tolist())
    if not result['inertia_positive'] or not result['joint_axes_unit'] or not result['finite_state']:raise ValueError('Invalid inertias/joint axes/state')
    return result
