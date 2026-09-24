"""Run inside Autodesk Fusion; exports only the named lower-body occurrence."""

import json
import traceback

import adsk.core
import adsk.fusion


def vector(value):
    return [value.x, value.y, value.z]


def properties(entity):
    props = entity.getPhysicalProperties(
        adsk.fusion.CalculationAccuracy.HighCalculationAccuracy
    )
    axes_ok, axis1, axis2, axis3 = props.getPrincipalAxes()
    moments_ok, i1, i2, i3 = props.getPrincipalMomentsOfInertia()
    xyz_ok, xx, yy, zz, xy, yz, xz = props.getXYZMomentsOfInertia()
    if not (axes_ok and moments_ok and xyz_ok):
        raise RuntimeError("Fusion could not calculate complete inertia properties")
    return {
        "mass_kg": props.mass,
        "center_of_mass_cm": vector(props.centerOfMass),
        "principal_moments_kg_cm2": [i1, i2, i3],
        "principal_axes": [vector(axis1), vector(axis2), vector(axis3)],
        "world_origin_moments_kg_cm2": {
            "xx": xx, "yy": yy, "zz": zz, "xy": xy, "yz": yz, "xz": xz
        },
    }


def export_occurrence(occurrence):
    record = {
        "path": occurrence.fullPathName,
        "component_name": occurrence.component.name,
        "transform2_array": occurrence.transform2.asArray(),
        "aggregate_properties": properties(occurrence),
        "direct_bodies": [],
        "children": [],
    }
    # Body and child records support subsequent grouping into rigid robot links.
    # Aggregate parent properties must not be added to their descendants.
    for body in occurrence.bRepBodies:
        if body.isSolid:
            record["direct_bodies"].append({
                "name": body.name,
                "material_name": body.material.name if body.material else None,
                "properties": properties(body),
            })
    for child in occurrence.childOccurrences:
        record["children"].append(export_occurrence(child))
    return record


def run(context):
    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            raise RuntimeError("Open the full robot assembly in Fusion first")
        candidates = [
            occurrence for occurrence in design.rootComponent.allOccurrences
            if occurrence.component.name == "Lower Body Reassembled"
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "Expected exactly one Lower Body Reassembled occurrence; "
                f"found {len(candidates)}. Check the component name in the browser."
            )
        dialog = ui.createFileDialog()
        dialog.title = "Save lower-body physical properties"
        dialog.filter = "JSON files (*.json)"
        dialog.initialFilename = "lower_body_physical_properties.json"
        if dialog.showSave() != adsk.core.DialogResults.DialogOK:
            return
        lower = export_occurrence(candidates[0])
        mass = lower["aggregate_properties"]["mass_kg"]
        result = {
            "document_name": app.activeDocument.name,
            "measured_hardware_mass_kg": 16.0,
            "fusion_lower_body_mass_kg": mass,
            "difference_from_scale_kg": mass - 16.0,
            "notes": [
                "Only the Lower Body Reassembled subtree was exported.",
                "Parent aggregate masses include descendants: do not sum all records.",
                "Raw Fusion properties retain cm and kg*cm^2 units.",
                "Visibility is not a physical exclusion rule; inspect for battery or duplicates.",
                "Coordinate frames and rigid-link grouping require validation before MJCF conversion.",
            ],
            "lower_body": lower,
        }
        with open(dialog.filename, "w", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
        ui.messageBox(
            f"Exported lower body: {mass:.6f} kg\n"
            f"Difference from measured 16 kg: {mass - 16.0:+.6f} kg\n"
            f"Saved to {dialog.filename}"
        )
    except Exception:
        ui.messageBox("Lower-body export failed:\n" + traceback.format_exc())
