#!/usr/bin/env python3

# NOTE THIS DOESN"T WRITE THE VISUAL MESH TAGS PROPERLY BUT COMPUTATIONS ARE GOOD

"""
MJCF -> URDF converter for the humanoid legs.

Assumptions (matching your setup and the working finger converter):
- Every body except the root ("hip") has exactly one hinge joint.
- joint.pos in MJCF is an ABSOLUTE position in the same global CAD frame
  as the meshes (after dividing by 1000).
- All body pos="0 0 0" so the geometry and joints are anchored by their
  absolute coordinates, not by body.pos.
- Meshes are defined in <asset><mesh name="..."> with a "file" and
  (optionally) a "scale" that we copy to URDF.

Strategy:
1. Traverse the MJCF worldbody, record:
   - body -> parent
   - body -> MJCF body element
   - body -> joint info (pos, axis, range, type) if present
2. Treat each joint.pos as an absolute position in the CAD/world frame.
3. URDF joint origin:
      origin(child) = joint_abs(child) - joint_abs(nearest_revolute_parent)
   For children of the root “hip”, parent_abs is 0.
4. Visual/collision compensation:
      visual_origin(body) = - joint_abs(body)
   For bodies without a joint (e.g., root hip), inherit parent’s comp
   or default to (0,0,0).
5. Copy mesh filenames and scale from MJCF <asset><mesh>.
"""

import xml.etree.ElementTree as ET
import numpy as np
import argparse
import os
from typing import Dict, List, Optional


class HumanoidMJCFToURDF:
    def __init__(self, mjcf_file: str):
        self.mjcf_file = mjcf_file
        self.tree = ET.parse(mjcf_file)
        self.root = self.tree.getroot()

        # MJCF data
        self.asset_meshes = {}       # mesh_name -> {"file": ..., "scale": [sx,sy,sz]}
        self.body_elems = {}         # body_name -> <body> element
        self.parent_map = {}         # body_name -> parent_body_name or None
        self.body_joint = {}         # body_name -> dict(joint info) or None
        self.joint_abs_pos = {}      # body_name -> np.array absolute joint position

        # Parse MJCF
        self._parse_assets()
        self._parse_bodies()

        # Choose root body explicitly (in your model it's "hip")
        self.root_body = "hip"
        if self.root_body not in self.body_elems:
            raise RuntimeError(f'Root body "{self.root_body}" not found in MJCF')

    # ------------------------------------------------------------------
    # Parsing MJCF
    # ------------------------------------------------------------------
    def _parse_assets(self):
        asset = self.root.find("asset")
        if asset is None:
            return
        for mesh in asset.findall("mesh"):
            name = mesh.get("name")
            file_attr = mesh.get("file")
            if not name or not file_attr:
                continue
            basename = os.path.basename(file_attr)
            scale_str = mesh.get("scale", "1 1 1")
            parts = [float(x) for x in scale_str.split()]
            if len(parts) == 1:
                parts = [parts[0], parts[0], parts[0]]
            elif len(parts) != 3:
                parts = [parts[0], parts[0], parts[0]]
            self.asset_meshes[name] = {
                "file": basename,
                "scale": parts,
            }

    def _parse_bodies(self):
        worldbody = self.root.find("worldbody")
        if worldbody is None:
            raise RuntimeError("No <worldbody> found in MJCF")

        # Recursive traversal
        def traverse(body_elem, parent_name=None):
            body_name = body_elem.get("name")
            if not body_name:
                return

            self.body_elems[body_name] = body_elem
            self.parent_map[body_name] = parent_name

            # Assume at most one joint per body (true for your humanoid)
            joint_elem = body_elem.find("joint")
            if joint_elem is not None:
                pos_str = joint_elem.get("pos", "0 0 0")
                pos = np.array([float(x) for x in pos_str.split()])
                axis_str = joint_elem.get("axis", "0 0 1")
                axis = [float(x) for x in axis_str.split()]
                jtype = joint_elem.get("type", "hinge")
                jrange = joint_elem.get("range", "-1.57 1.57")
                name = joint_elem.get("name", f"{body_name}_joint")

                self.body_joint[body_name] = {
                    "name": name,
                    "pos_abs": pos,     # treat as absolute
                    "axis": axis,
                    "type": jtype,
                    "range": jrange,
                }
                self.joint_abs_pos[body_name] = pos
            else:
                self.body_joint[body_name] = None

            for child in body_elem.findall("body"):
                traverse(child, body_name)

        for root_body in worldbody.findall("body"):
            traverse(root_body, parent_name=None)

    # ------------------------------------------------------------------
    # Helpers: joint positions & visual compensation
    # ------------------------------------------------------------------
    def _nearest_parent_revolute_abs(self, body_name: str) -> np.ndarray:
        """Return absolute position of nearest parent that has a hinge/revolute joint."""
        parent = self.parent_map.get(body_name)
        while parent is not None:
            jinfo = self.body_joint.get(parent)
            if jinfo is not None and jinfo["type"] in ("hinge", "revolute"):
                return jinfo["pos_abs"]
            parent = self.parent_map.get(parent)
        # No parent joint => treat as world origin
        return np.zeros(3)

    def _joint_origin_relative(self, body_name: str) -> np.ndarray:
        """URDF origin for the joint that connects this body to its parent."""
        jinfo = self.body_joint.get(body_name)
        if jinfo is None:
            # Fixed link with no own joint (shouldn't happen in your humanoid except maybe root)
            return np.zeros(3)
        child_abs = jinfo["pos_abs"]
        parent_abs = self._nearest_parent_revolute_abs(body_name)
        return child_abs - parent_abs

    def _visual_compensation(self, body_name: str) -> np.ndarray:
        """
        Visual/collision compensation offset:
        - If body has a joint: - joint_abs
        - Else: inherit from parent (so root hip typically gets 0 0 0)
        """
        jinfo = self.body_joint.get(body_name)
        if jinfo is not None:
            return -jinfo["pos_abs"]
        parent = self.parent_map.get(body_name)
        if parent is None:
            return np.zeros(3)
        return self._visual_compensation(parent)

    # ------------------------------------------------------------------
    # Mesh lookup helpers
    # ------------------------------------------------------------------
    def _get_mesh_info(self, mesh_name: str) -> Optional[Dict]:
        return self.asset_meshes.get(mesh_name)

    # ------------------------------------------------------------------
    # Generate URDF
    # ------------------------------------------------------------------
    def generate_urdf(self) -> ET.ElementTree:
        robot = ET.Element("robot", {"name": "ch_robot"})

        # Process bodies in a traversal order starting from root_body
        ordered_bodies = self._traversal_order()

        # First: create all links
        for body_name in ordered_bodies:
            self._create_link(robot, body_name)

        # Second: create joints (skip root)
        for body_name in ordered_bodies:
            if body_name == self.root_body:
                continue
            self._create_joint(robot, body_name)

        _indent_xml(robot)
        return ET.ElementTree(robot)

    def _traversal_order(self) -> List[str]:
        """Depth-first traversal from root_body."""
        order = []

        def dfs(bname):
            order.append(bname)
            body_elem = self.body_elems[bname]
            for child in body_elem.findall("body"):
                cname = child.get("name")
                if cname:
                    dfs(cname)

        dfs(self.root_body)
        return order

    def _create_link(self, robot_elem: ET.Element, body_name: str):
        link = ET.SubElement(robot_elem, "link", {"name": body_name})

        # Simple placeholder inertial
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", {"value": "0.1"})
        ET.SubElement(inertial, "inertia", {
            "ixx": "0.001", "ixy": "0", "ixz": "0",
            "iyy": "0.001", "iyz": "0", "izz": "0.001",
        })

        body_elem = self.body_elems[body_name]
        comp = self._visual_compensation(body_name)
        comp_str = f"{comp[0]:.6g} {comp[1]:.6g} {comp[2]:.6g}"

        # Convert geoms: group=0 or contype=0/conaffinity=0 => visual, else collision
        for geom in body_elem.findall("geom"):
            gtype = geom.get("type", "mesh")
            if gtype != "mesh":
                # For now, ignore non-mesh geoms (like the hip box). You can add
                # box/capsule conversion here later if needed.
                continue

            mesh_name = geom.get("mesh")
            if not mesh_name:
                continue

            group = geom.get("group", "1")
            contype = geom.get("contype", "1")
            conaff = geom.get("conaffinity", "1")

            mesh_info = self._get_mesh_info(mesh_name)
            if mesh_info is None:
                continue

            filename = f"robot_meshes/{mesh_info['file']}"
            sx, sy, sz = mesh_info["scale"]
            scale_str = f"{sx} {sy} {sz}"

            is_visual = (group == "0") or (contype == "0" and conaff == "0")

            tag_name = "visual" if is_visual else "collision"
            sub = ET.SubElement(link, tag_name)

            # Compensation so the mesh attaches at the joint
            ET.SubElement(sub, "origin", {"xyz": comp_str, "rpy": "0 0 0"})

            geom_xml = ET.SubElement(sub, "geometry")
            ET.SubElement(geom_xml, "mesh", {
                "filename": filename,
                "scale": scale_str,
            })

    def _create_joint(self, robot_elem: ET.Element, body_name: str):
        parent_name = self.parent_map.get(body_name)
        if parent_name is None:
            return

        jinfo = self.body_joint.get(body_name)
        if jinfo is None:
            # Fixed joint (no own hinge)
            jname = f"{body_name}_fixed"
            jtype = "fixed"
            axis = [0.0, 0.0, 1.0]
            lower, upper = "0", "0"
        else:
            jname = jinfo["name"]
            jtype = "revolute" if jinfo["type"] in ("hinge", "revolute") else "fixed"
            axis = jinfo["axis"]
            rvals = jinfo["range"].split()
            lower, upper = rvals[0], rvals[1] if len(rvals) == 2 else ("-1.57", "1.57")

        joint_elem = ET.SubElement(robot_elem, "joint", {
            "name": jname,
            "type": jtype,
        })
        ET.SubElement(joint_elem, "parent", {"link": parent_name})
        ET.SubElement(joint_elem, "child", {"link": body_name})

        # Relative origin
        rel = self._joint_origin_relative(body_name)
        rel_str = f"{rel[0]:.6g} {rel[1]:.6g} {rel[2]:.6g}"
        ET.SubElement(joint_elem, "origin", {"xyz": rel_str, "rpy": "0 0 0"})

        # Axis and limits for revolute joints
        ET.SubElement(joint_elem, "axis", {
            "xyz": f"{axis[0]} {axis[1]} {axis[2]}",
        })
        if jtype == "revolute":
            ET.SubElement(joint_elem, "limit", {
                "lower": lower,
                "upper": upper,
                "effort": "100",
                "velocity": "5.0",
            })
        ET.SubElement(joint_elem, "dynamics", {"damping": "1.0"})


def _indent_xml(elem, level=0):
    """Pretty-print XML in-place."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def main():
    parser = argparse.ArgumentParser(
        description="Convert humanoid MJCF (ch_robot) to URDF with proper joint origins and mesh scales."
    )
    parser.add_argument("mjcf_file", help="Input MJCF XML file, e.g. ch_robot.xml")
    parser.add_argument(
        "-o",
        "--output",
        default="ch_robot_converted.urdf",
        help="Output URDF file (default: ch_robot_converted.urdf)",
    )
    args = parser.parse_args()

    conv = HumanoidMJCFToURDF(args.mjcf_file)
    urdf_tree = conv.generate_urdf()
    urdf_tree.write(args.output, encoding="unicode", xml_declaration=True)

    print(f"Written URDF to: {args.output}")
    print("\nJoint absolute vs relative (debug):")
    for body, jinfo in conv.body_joint.items():
        if jinfo is None:
            continue
        abs_pos = jinfo["pos_abs"]
        rel = conv._joint_origin_relative(body)
        print(f"  {body}: abs={abs_pos}, rel={rel}")

    print("\nVisual compensations (debug):")
    for body in conv.body_elems.keys():
        comp = conv._visual_compensation(body)
        print(f"  {body}: comp={comp}")


if __name__ == "__main__":
    main()
