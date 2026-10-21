"""Build a CAD-derived, closed-loop ankle inspection rig, not a training model.

Run from simulation: uv run --project mjlab --no-sync python cad/build_ankle_linkage_inspection.py
The pin axes define a planar four-bar. A small CAD assembly mismatch is resolved
by closing the mechanism analytically, rather than stretching the heel bar.
"""
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import trimesh

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "ankle_audit/fusion_export_20261020_233848"
OUT = BASE / "ankle_linkage_inspection"


def fmt(values):
    return " ".join(f"{v:.12g}" for v in np.asarray(values).ravel())


def rot(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def angle(vector):
    return math.atan2(vector[2], vector[1])


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def closure(b, c, length, radius, hint):
    """Intersect circles in the hinge-normal plane; preserve the CAD branch."""
    delta = b[1:] - c[1:]
    distance = np.linalg.norm(delta)
    along = (radius**2 - length**2 + distance**2) / (2 * distance)
    height2 = radius**2 - along**2
    if height2 < -1e-12:
        raise ValueError("Four-bar cannot close at the requested angle")
    direction = delta / distance
    perpendicular = np.array([-direction[1], direction[0]])
    candidates = [c[1:] + along * direction + sign * np.sqrt(max(height2, 0)) * perpendicular
                  for sign in (-1, 1)]
    picked = min(candidates, key=lambda p: np.linalg.norm(p - hint[1:]))
    return np.r_[b[0], picked]


def build():
    OUT.mkdir(exist_ok=True)
    (OUT / "meshes").mkdir(exist_ok=True)
    manifest = json.loads((SOURCE / "ankle_linkage.json").read_text())
    occurrences = {o["component_id"]: o for o in manifest["occurrences"]}
    root = ET.Element("mujoco", model="cad_ankle_linkage_inspection_only")
    ET.SubElement(root, "compiler", angle="radian", meshdir="meshes")
    ET.SubElement(root, "option", timestep="0.001", gravity="0 0 0", integrator="implicitfast")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1200", offheight="800")
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "geom", contype="0", conaffinity="0")
    ET.SubElement(defaults, "joint", damping="0.01", armature="0.0001")
    assets = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 1 -2", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "light", pos="1 1 1", dir="-1 -1 -1")
    ET.SubElement(world, "geom", type="plane", size="1 1 .02", pos="0 0 -.08", rgba=".13 .16 .21 1")
    equality = ET.SubElement(root, "equality")
    rigs = {}
    report = {"scope": "CAD-derived closed-loop kinematic inspection; not calibrated dynamics or a training replacement",
              "source": str(SOURCE.relative_to(BASE)), "sides": {}}

    for side, offset in (("L", .18), ("R", -.18)):
        components = {c["name"][2:]: c for c in manifest["components"] if c["name"].startswith(side + " ")}

        def transform(name):
            return np.array(occurrences[components[name]["id"]]["transform2"]).reshape(4, 4)

        # Both sides share a proper right-handed view frame. Do not infer an
        # existing URDF frame from component names or silently mirror geometry.
        view = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])

        def cylinders_world(name, radius):
            t = transform(name)
            return [(view @ (t[:3, :3] @ a["origin_cm"] + t[:3, 3])) * .01
                    for a in components[name]["bodies"][0]["cylinder_axes_native"]
                    if abs(a["radius_cm"] - radius) < 1e-6]

        a = cylinders_world("Ankle Rod 1", .6)[0]
        b_raw, d_raw = sorted(cylinders_world("Heel Linkage", .6), key=lambda p: p[2])
        c_raw = cylinders_world("R02 Connector", 2.2)[0]
        pin_raw = cylinders_world("Top Heel Linkage Rod", .6)[0]
        mesh_bar = trimesh.load(SOURCE / components["Heel Linkage"]["bodies"][0]["mesh"])

        def vertices_world(mesh, name):
            t = transform(name)
            return ((np.asarray(mesh.vertices) * .1 @ t[:3, :3].T + t[:3, 3]) @ view.T) * .01

        bar_world = vertices_world(mesh_bar, "Heel Linkage")
        # Points on an axis can differ axially; pin closure is measured normal to it.
        plane_x = float(bar_world[:, 0].mean())
        b_raw[0] = d_raw[0] = c_raw[0] = pin_raw[0] = plane_x
        b, c, d_cad, pin = (p - a for p in (b_raw, c_raw, d_raw, pin_raw))
        length = float(np.linalg.norm((d_cad - b)[1:]))
        radius = float(np.linalg.norm((pin - c)[1:]))
        d = closure(b, c, length, radius, d_cad)
        correction = wrap(angle(d - b) - angle(d_cad - b))
        crank_angle = angle(d - c)
        rod_angle = angle(d - b)

        base = ET.SubElement(world, "body", name=side + "_shin_mount", pos=fmt([offset, 0, 0]))
        foot = ET.SubElement(base, "body", name=side + "_foot")
        ET.SubElement(foot, "joint", name=side + "_ankle", type="hinge", axis="1 0 0",
                      limited="true", range="-.6 .6")

        # Add the exported foot parts in their measured placement. Exclude the
        # long bar, which gets its own moving body below.
        for name, component in components.items():
            if not component["bodies"] or name == "Heel Linkage":
                continue
            if "Foot Sub-Assembly" in component["owner_occurrence"]:
                parent, origin, color = foot, a, ".65 .69 .75 1"
            elif name in ("R02 Connector", "Ankle Bearing 1", "Ankle Bearing 2"):
                parent, origin, color = base, a, ".35 .43 .54 1"
            else:
                continue
            for index, body in enumerate(component["bodies"]):
                mesh = trimesh.load(SOURCE / body["mesh"])
                mesh.vertices = vertices_world(mesh, name) - origin
                mesh_name = side + "_" + component["id"] + "_" + str(index)
                mesh.export(OUT / "meshes" / (mesh_name + ".stl"))
                ET.SubElement(assets, "mesh", name=mesh_name, file=mesh_name + ".stl")
                ET.SubElement(parent, "geom", type="mesh", mesh=mesh_name,
                              mass=str(body["mass_kg"]), rgba=color)

        # The crank's detailed rotor mesh/inertia was not exported. This explicit
        # 1 g proxy is for inspecting closure only, never a motor dynamics model.
        crank = ET.SubElement(base, "body", name=side + "_crank", pos=fmt(c))
        ET.SubElement(crank, "joint", name=side + "_crank_joint", axis="1 0 0")
        ET.SubElement(crank, "geom", type="capsule", fromto=fmt(np.r_[np.zeros(3), d-c]),
                      size=".003", mass=".001", rgba=".2 .65 1 1")
        ET.SubElement(crank, "site", name=side + "_crank_pin", pos=fmt(d-c), size=".003", rgba="1 1 0 1")

        bar = ET.SubElement(foot, "body", name=side + "_heel_linkage", pos=fmt(b))
        ET.SubElement(bar, "joint", name=side + "_bar_joint", axis="1 0 0")
        mesh_bar.vertices = (bar_world - b_raw) @ rot(correction).T
        mesh_name = side + "_heel_linkage"
        mesh_bar.export(OUT / "meshes" / (mesh_name + ".stl"))
        ET.SubElement(assets, "mesh", name=mesh_name, file=mesh_name + ".stl")
        ET.SubElement(bar, "geom", type="mesh", mesh=mesh_name,
                      mass=str(components["Heel Linkage"]["bodies"][0]["mass_kg"]), rgba="1 .48 .08 1")
        ET.SubElement(bar, "site", name=side + "_bar_pin", pos=fmt(d-b), size=".003", rgba="1 1 0 1")
        ET.SubElement(equality, "connect", name=side + "_closure", site1=side+"_bar_pin",
                      site2=side+"_crank_pin", solref=".002 1")
        rigs[side] = dict(b=b, c=c, d=d, length=length, radius=radius,
                          rod_angle=rod_angle, crank_angle=crank_angle)
        report["sides"][side] = {
            "bar_hole_spacing_m": length, "crank_radius_m": radius,
            "exported_upper_pin_misalignment_m": float(np.linalg.norm((pin-d_cad)[1:])),
            "bar_pose_correction_deg": math.degrees(correction),
            "upper_pin_pose_adjustment_m": float(np.linalg.norm((d-pin)[1:])),
            "pivot_A": [0, 0, 0], "pivot_B": b.tolist(), "pivot_C": c.tolist(),
            "pivot_D_closed": d.tolist(), "crank_inertia": "1 g capsule proxy; not identified",
        }
    ET.indent(root)
    path = OUT / "ankles.xml"
    ET.ElementTree(root).write(path, encoding="unicode")
    return path, rigs, report


def pose(model, data, rigs, q):
    for side, rig in rigs.items():
        b = rot(q) @ rig["b"]
        d = closure(b, rig["c"], rig["length"], rig["radius"], rig["d"])
        values = {side + "_ankle": q,
                  side + "_bar_joint": wrap(angle(d-b) - rig["rod_angle"] - q),
                  side + "_crank_joint": wrap(angle(d-rig["c"]) - rig["crank_angle"])}
        for name, value in values.items():
            data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)


def main():
    path, rigs, report = build()
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    errors, crank_history = [], {s: [] for s in rigs}
    for q in np.linspace(-.6, .6, 241):
        pose(model, data, rigs, q)
        for side in rigs:
            errors.append(float(np.linalg.norm(data.site(side+"_bar_pin").xpos-data.site(side+"_crank_pin").xpos)))
            crank_history[side].append(float(data.qpos[model.joint(side+"_crank_joint").qposadr[0]]))
    assert max(errors) < 1e-8, max(errors)
    for side, values in crank_history.items():
        derivative = np.gradient(np.unwrap(values), .005)
        assert np.all(np.isfinite(derivative)) and np.max(np.abs(np.diff(values))) < .1
        report["sides"][side]["crank_angle_per_foot_angle_range"] = [float(derivative.min()), float(derivative.max())]
    report["validation"] = {"angles_tested_per_side": 241, "foot_angle_range_rad": [-.6,.6],
                            "max_pin_closure_error_m": max(errors), "branch_continuous": True,
                            "range_note": "Existing simulator range, not verified hardware travel"}
    pose(model, data, rigs, 0.)
    for _ in range(100):
        mujoco.mj_step(model, data)
    report["validation"]["zero_gravity_100_step_finite"] = bool(np.all(np.isfinite(data.qpos)))
    report["validation"]["zero_gravity_100_step_max_closure_m"] = max(float(np.linalg.norm(data.site(s+"_bar_pin").xpos-data.site(s+"_crank_pin").xpos)) for s in rigs)
    (OUT / "validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)

    import imageio.v2 as imageio
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, .025, .1]
    camera.distance = .85
    camera.elevation = -18
    camera.azimuth = 125
    with mujoco.Renderer(model, height=800, width=1200) as renderer:
        pose(model, data, rigs, 0)
        renderer.update_scene(data, camera)
        imageio.imwrite(OUT / "ankles.png", renderer.render())
        with imageio.get_writer(OUT / "ankle_sweep.mp4", fps=30) as video:
            for q in .45*np.sin(np.linspace(0, 2*np.pi, 150)):
                pose(model, data, rigs, float(q))
                renderer.update_scene(data, camera)
                video.append_data(renderer.render())


if __name__ == "__main__":
    main()
