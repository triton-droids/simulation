"""Run in Fusion's Scripts and Add-Ins with the assembled robot open.

Read-only CAD export. Preserves native frames and assembly placements separately;
does not infer a linkage ratio or modify joints, visibility, or the timeline.
"""

import json
import re
import traceback
from datetime import datetime
from pathlib import Path

import adsk.core
import adsk.fusion


def xyz(value):
    return [value.x, value.y, value.z]


def read_property(obj, name, convert=lambda value: value):
    try:
        value = getattr(obj, name)
        return None if value is None else convert(value)
    except Exception as exc:
        return {"unavailable": str(exc)}


def occurrence_path(occurrence):
    return occurrence.fullPathName if occurrence else None


def collect_occurrences(root):
    """Never merge components using entity tokens from linked documents."""
    records, components = [], []

    def visit(occurrence):
        cid = f"component_{len(components):04d}"
        component = occurrence.component
        components.append((cid, component, occurrence.fullPathName))
        records.append({
            "path": occurrence.fullPathName,
            "component_id": cid,
            "component_name": component.name,
            "assembly_context": occurrence_path(occurrence.assemblyContext),
            "transform2": list(occurrence.transform2.asArray()),
            "native_transform2": list((occurrence.nativeObject or occurrence).transform2.asArray()),
        })
        for child in occurrence.childOccurrences:
            visit(child)

    visit(root)
    return records, components


def cylinder_axes(body):
    """Export native bore/shaft geometry even if CAD has no assembly joints."""
    result = []
    for index, face in enumerate(body.faces):
        cylinder = adsk.core.Cylinder.cast(face.geometry)
        if cylinder:
            result.append({"face_index": index, "origin_cm": xyz(cylinder.origin),
                           "axis": xyz(cylinder.axis), "radius_cm": cylinder.radius})
    return result


def joint_record(joint):
    record = {
        "name": joint.name,
        "type": joint.objectType,
        "parent_component": joint.parentComponent.name,
        "assembly_context": occurrence_path(joint.assemblyContext),
        "occurrence_one": occurrence_path(joint.occurrenceOne),
        "occurrence_two": occurrence_path(joint.occurrenceTwo),
        "suppressed": joint.isSuppressed,
        "frames": {},
    }
    # Raw native-frame matrices; their owning component is explicitly recorded.
    for field in ("geometryOneTransform", "geometryTwoTransform", "transform"):
        if hasattr(joint, field):
            record["frames"][field] = read_property(joint, field, lambda m: list(m.asArray()))
    motion = joint.jointMotion
    record["motion_type"] = motion.objectType
    for field in ("rotationAxisVector", "rotationValue"):
        if hasattr(motion, field):
            record[field] = read_property(motion, field, xyz if field.endswith("Vector") else float)
    if hasattr(motion, "rotationLimits"):
        limits = motion.rotationLimits
        record["rotation_limits"] = {
            field: read_property(limits, field)
            for field in ("isMinimumValueEnabled", "minimumValue", "isMaximumValueEnabled", "maximumValue")
        }
    for field in ("isFlipped", "isLocked", "errorOrWarningMessage"):
        if hasattr(joint, field):
            record[field] = read_property(joint, field)
    return record


def run(context):
    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            raise RuntimeError("Open the assembled robot in Fusion's Design workspace.")
        roots = [o for o in design.rootComponent.allOccurrences
                 if o.component.name == "Lower Body Reassembled"]
        if len(roots) != 1:
            selection = ui.selectEntity("Select the assembled lower-body occurrence in the browser", "Occurrences")
            if not selection:
                return
            roots = [adsk.fusion.Occurrence.cast(selection.entity)]
            if not roots[0]:
                raise RuntimeError("Select an assembly occurrence, not an individual body.")
        dialog = ui.createFolderDialog()
        dialog.title = "Choose where to save the ankle linkage export"
        if dialog.showDialog() != adsk.core.DialogResults.DialogOK:
            return
        output = Path(dialog.folder) / datetime.now().strftime("ankle_linkage_%Y%m%d_%H%M%S")
        output.mkdir(exist_ok=False)
        manifest = {
            "exporter_version": 2,
            "document": app.activeDocument.name,
            "api_version": app.version,
            "units": {"transforms_and_bounds": "cm", "mesh": "mm", "angles": "rad", "mass": "kg"},
            "frame_note": "Native body meshes and native joint frames. Resolve using component placements; do not treat every matrix as world coordinates.",
            "occurrences": [], "components": [], "root_joint_records": [], "warnings": [],
        }
        manifest["occurrences"], components = collect_occurrences(roots[0])
        # Include joints owned above the selected subtree, as well as linked parts.
        for collection in (design.rootComponent.allJoints, design.rootComponent.allAsBuiltJoints):
            for joint in collection:
                try:
                    manifest["root_joint_records"].append(joint_record(joint))
                except Exception as exc:
                    manifest["warnings"].append(f"Root joint {joint.name}: {exc}")
        selected = ("heel", "ankle", "r02 connector", "foot", "lower leg", "configuration 1")
        for cid, component, owner_path in components:
            record = {"id": cid, "name": component.name, "owner_occurrence": owner_path,
                      "bodies": [], "joints": []}
            for collection in (component.joints, component.asBuiltJoints):
                for joint in collection:
                    try:
                        record["joints"].append(joint_record(joint))
                    except Exception as exc:
                        manifest["warnings"].append(f"{cid}/{joint.name}: {exc}")
            if any(word in component.name.lower() for word in selected):
                for index, body in enumerate(component.bRepBodies):
                    if not body.isSolid:
                        continue
                    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", component.name)
                    filename = f"{cid}_{safe_name}_body_{index}.stl"
                    props = body.getPhysicalProperties(adsk.fusion.CalculationAccuracy.HighCalculationAccuracy)
                    bounds = body.boundingBox
                    entry = {
                        "name": body.name, "mesh": filename,
                        "native_bounds_cm": [xyz(bounds.minPoint), xyz(bounds.maxPoint)],
                        "mass_kg": props.mass, "native_com_cm": xyz(props.centerOfMass),
                        "principal_moments": list(props.getPrincipalMomentsOfInertia()),
                        "origin_moments": list(props.getXYZMomentsOfInertia()),
                        "cylinder_axes_native": cylinder_axes(body),
                    }
                    options = design.exportManager.createSTLExportOptions(body, str(output / filename))
                    options.isBinaryFormat = True
                    options.sendToPrintUtility = False
                    options.unitType = adsk.fusion.DistanceUnits.MillimeterDistanceUnits
                    options.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementHigh
                    if not design.exportManager.execute(options):
                        raise RuntimeError(f"Mesh export failed: {filename}")
                    record["bodies"].append(entry)
            manifest["components"].append(record)
        heels = [c for c in manifest["components"] if "heel linkage" in c["name"].lower()
                 and "top heel" not in c["name"].lower()]
        missing = [c["owner_occurrence"] for c in heels if not c["bodies"]]
        complete = len(heels) >= 2 and not missing
        if not complete:
            manifest["warnings"].append(f"INCOMPLETE: expected two heel bars with meshes; found {len(heels)}; missing bodies: {missing}")
        joint_count = len(manifest["root_joint_records"]) + sum(len(c["joints"]) for c in manifest["components"])
        if not joint_count:
            manifest["warnings"].append("No joints found. Exported cylinder axes require geometric reconstruction; joint limits remain unknown.")
        manifest["validation"] = {"heel_meshes_complete": complete, "heel_occurrences": len(heels),
                                  "joint_record_count_including_duplicates": joint_count,
                                  "kinematics_validated": False}
        (output / "ankle_linkage.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        status = "Heel meshes exported" if complete else "INCOMPLETE heel geometry"
        ui.messageBox(f"{status}\nSaved to:\n{output}\n{joint_count} joint records; {len(manifest['warnings'])} warnings.\nCAD was not modified.")
    except Exception:
        ui.messageBox("Ankle export failed:\n" + traceback.format_exc())
