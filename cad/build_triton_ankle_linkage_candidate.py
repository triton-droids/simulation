"""Add articulated heel bars to a separate Triton lower-body inspection model.

This model adds passive joints and provisional crank inertia. It intentionally
does not replace the 10-joint policy environment or claim calibrated dynamics.
"""
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh
from scipy.optimize import least_squares

from build_ankle_linkage_inspection import BASE, OUT, closure, fmt, rot, wrap, angle

MESHES = BASE.parent / "mjlab/src/mjlab/asset_zoo/robots/triton_robot/meshes"


def measure_old_mesh_axes():
    """Register bore axes, not unrelated Fusion and legacy mesh origins."""
    foot = trimesh.load(MESHES / "visual/left_foot.stl")
    flanges = [c for c in foot.split(only_watertight=False)
               if len(c.faces) > 100 and c.bounds[1, 1] < -50]
    centers, residuals = [], []
    for flange in flanges:
        p = flange.vertices[:, 1:]
        p = p[abs(np.linalg.norm(p-[-74.4, 31.5], axis=1)-10.4875) < .8]
        fit = least_squares(lambda center: np.linalg.norm(p-center, axis=1)-10.4875, [-74.4, 31.5])
        centers.append(fit.x)
        residuals.append(float(max(abs(fit.fun))))
    if len(centers) != 2 or np.linalg.norm(centers[0]-centers[1]) > .001:
        raise ValueError("Legacy heel-flange registration is inconsistent")
    shin = trimesh.load(MESHES / "visual/left_leg4.stl")
    motor = max(shin.split(only_watertight=False), key=lambda c: len(c.faces))
    p = motor.vertices[:, 1:]
    center = motor.bounds[:, 1:].mean(axis=0)
    p = p[abs(np.linalg.norm(p-center, axis=1)-39.25) < .1]
    fit = least_squares(lambda a: np.linalg.norm(p-a[:2], axis=1)-a[2], [*center, 39.25])
    return np.mean(centers, axis=0)*.001, fit.x[:2]*.001, {
        "heel_bore_fit_max_residual_mm": max(residuals),
        "motor_shell_fit_max_residual_mm": float(max(abs(fit.fun))),
        "motor_axis_note": "Estimated from legacy cylindrical motor mesh, not a measured hardware shaft axis",
    }


def set_pose(model, data, rigs, q):
    for side, r in rigs.items():
        b = r["a"] + rot(q) @ (r["b"]-r["a"])
        d = closure(b, r["c"], r["length"], r["radius"], r["d"])
        for name, value in ((side+"_ankle_joint", q),
                            (side+"_heel_bar_joint", wrap(angle(d-b)-r["rod_angle"]-q)),
                            (side+"_ankle_crank_joint", wrap(angle(d-r["c"])-r["crank_angle"]))):
            data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)


def main():
    source = json.loads((OUT / "validation.json").read_text())
    heel_yz, motor_yz, registration = measure_old_mesh_axes()
    root = ET.parse(BASE / "triton_lower_body.xml").getroot()
    root.set("model", "triton_lower_body_linkage_candidate_not_policy_compatible")
    root.insert(0, ET.Comment("Inspection candidate: extra passive DOFs, unreconciled masses and 1 g crank proxies. Do not load existing policy blindly."))
    root.find("compiler").set("meshdir", ".")
    assets = root.find("asset")
    for mesh in assets.findall("mesh"):
        mesh.set("file", os.path.relpath(MESHES / mesh.get("file"), OUT))
    actuator = ET.parse(BASE / "triton_lower_body_actuated.xml").getroot().find("actuator")
    root.append(actuator)
    equality = ET.SubElement(root, "equality")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1200", offheight="1000")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="1 -1 2", dir="-1 1 -2", diffuse=".8 .8 .8")
    ET.SubElement(world, "light", pos="-1 1 2", dir="1 -1 -2")
    rigs, report = {}, {"scope": "Full robot geometric inspection only", "registration": registration,
                        "limitations": ["Extra passive DOFs require observation/action mapping changes before policy use",
                                        "Existing direct ankle actuators retained; motor-side control not calibrated",
                                        "Bar masses added to existing lumped links; mass/inertia reconciliation pending",
                                        "1 g crank proxies; no identified rotor inertia"], "sides": {}}
    source_model = mujoco.MjModel.from_xml_path(str(OUT / "ankles.xml"))
    for label, side in (("L", "left"), ("R", "right")):
        measure = source["sides"][label]
        shin = root.find(f".//body[@name='{side}_leg4']")
        foot = root.find(f".//body[@name='{side}_foot']")
        a = np.fromstring(foot.find("joint").get("pos"), sep=" ")
        # Legacy foot meshes are mirrored across x=-0.1556355 m. Align the
        # bar between their heel bearings, preserving the old model's naming.
        x_bar_left = .02137446069113918 - source["sides"]["L"]["pivot_B"][0]
        x_bar = x_bar_left if side == "left" else -.311271-x_bar_left
        b, c = np.r_[x_bar, heel_yz], np.r_[x_bar, motor_yz]
        length, radius = measure["bar_hole_spacing_m"], measure["crank_radius_m"]
        hint = b + [0, 0, length]
        d = closure(b, c, length, radius, hint)
        old_bar_vector = np.array(measure["pivot_D_closed"])-measure["pivot_B"]
        # Inspection and legacy models face opposite y directions. Rotate the
        # recovered mesh by a proper rotation, then align its two hole axes.
        turn = np.diag([-1., -1., 1.])
        initial_vector = turn @ old_bar_vector
        correction = wrap(angle(d-b)-angle(initial_vector))
        mesh = trimesh.load(OUT / "meshes" / (label+"_heel_linkage.stl"))
        mesh.vertices = np.asarray(mesh.vertices) @ turn.T @ rot(correction).T
        mesh_name = side+"_heel_linkage_recovered"
        mesh.export(OUT / "meshes" / (mesh_name+".stl"))
        ET.SubElement(assets, "mesh", name=mesh_name, file="meshes/"+mesh_name+".stl")
        bar = ET.SubElement(foot, "body", name=side+"_heel_linkage", pos=fmt(b))
        ET.SubElement(bar, "joint", name=side+"_heel_bar_joint", limited="false", axis="1 0 0", damping=".01", armature=".0001")
        mass = float(source_model.body(label+"_heel_linkage").mass[0])
        ET.SubElement(bar, "geom", mesh=mesh_name, type="mesh", mass=str(mass),
                      contype="0", conaffinity="0", group="2", rgba="1 .45 .05 1")
        ET.SubElement(bar, "site", name=side+"_heel_bar_pin", pos=fmt(d-b), size=".002", rgba="1 .8 0 1")
        crank = ET.SubElement(shin, "body", name=side+"_ankle_crank", pos=fmt(c))
        ET.SubElement(crank, "joint", name=side+"_ankle_crank_joint", limited="false", axis="1 0 0", damping=".01", armature=".0001")
        ET.SubElement(crank, "geom", type="capsule", fromto=fmt(np.r_[np.zeros(3),d-c]),
                      size=".003", mass=".001", contype="0", conaffinity="0", group="2", rgba=".2 .65 1 1")
        ET.SubElement(crank, "site", name=side+"_crank_pin", pos=fmt(d-c), size=".002", rgba="1 .8 0 1")
        ET.SubElement(equality, "connect", site1=side+"_heel_bar_pin", site2=side+"_crank_pin", solref=".002 1")
        rigs[side] = dict(a=a,b=b,c=c,d=d,length=length,radius=radius,
                          rod_angle=angle(d-b),crank_angle=angle(d-c))
        report["sides"][side] = {"A": a.tolist(), "B": b.tolist(), "C": c.tolist(), "D": d.tolist(),
                                "bar_mass_kg": mass, "length_m": length, "crank_radius_m":radius}
    ET.indent(root)
    path = OUT / "triton_lower_body_linkage_candidate.xml"
    ET.ElementTree(root).write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    baseline = mujoco.MjModel.from_xml_path(str(BASE / "triton_lower_body_actuated.xml"))
    data = mujoco.MjData(model)
    errors = []
    for q in np.linspace(-.6,.6,241):
        set_pose(model,data,rigs,float(q))
        for side in rigs:
            errors.append(float(np.linalg.norm(data.site(side+"_heel_bar_pin").xpos-data.site(side+"_crank_pin").xpos)))
    assert max(errors)<1e-8
    assert model.nu == baseline.nu == 10
    report["validation"] = {"angles_tested_per_side":241,"max_pin_closure_error_m":max(errors),
                            "baseline_nq":baseline.nq,"candidate_nq":model.nq,
                            "actuators":model.nu,"baseline_mass_kg":float(baseline.body_mass.sum()),
                            "candidate_mass_kg":float(model.body_mass.sum())}
    saved_gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0
    driven = mujoco.MjData(model)
    driven_errors = []
    for step in range(1000):
        for side in rigs:
            driven.ctrl[model.actuator(side+"_ankle_act").id] = .25*np.sin(2*np.pi*.4*step*model.opt.timestep)
        mujoco.mj_step(model, driven)
        assert np.isfinite(driven.qpos).all() and np.isfinite(driven.qvel).all()
        for side in rigs:
            driven_errors.append(float(np.linalg.norm(driven.site(side+"_heel_bar_pin").xpos-driven.site(side+"_crank_pin").xpos)))
    assert max(driven_errors) < .0001
    model.opt.gravity[:] = saved_gravity
    report["validation"].update(zero_gravity_driven_steps=1000, zero_gravity_driven_finite=True,
                                zero_gravity_driven_max_closure_m=max(driven_errors))
    (OUT/"full_robot_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[0,0,.34];camera.distance=1.55;camera.elevation=-13;camera.azimuth=145
    options=mujoco.MjvOption();options.geomgroup[3]=0
    with mujoco.Renderer(model,height=1000,width=1200) as renderer:
        set_pose(model,data,rigs,0)
        renderer.update_scene(data,camera,scene_option=options)
        imageio.imwrite(OUT/"triton_lower_body_linkage.png",renderer.render())
        camera.lookat[:]=[0,-.035,.16];camera.distance=.88;camera.azimuth=180
        renderer.update_scene(data,camera,scene_option=options)
        imageio.imwrite(OUT/"triton_ankle_detail.png",renderer.render())
        with imageio.get_writer(OUT/"triton_ankle_sweep.mp4",fps=30) as video:
            for q in .45*np.sin(np.linspace(0,2*np.pi,150)):
                set_pose(model,data,rigs,float(q))
                renderer.update_scene(data,camera,scene_option=options)
                video.append_data(renderer.render())


if __name__ == "__main__":
    main()
