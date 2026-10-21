# ExportAnkleLinkage v2

1. Open the assembled robot in Fusion's Design workspace.
2. Replace the previous ExportAnkleLinkage script folder with this folder.
3. In Utilities → Scripts and Add-Ins, add/select ExportAnkleLinkage and run it.
4. If the expected assembly name is absent or ambiguous, the script asks you
   to select the lower-body assembly occurrence in the browser. No renaming or
   code editing is required.
5. Choose an output folder. Send the entire new timestamped folder as a ZIP.

This version gives each occurrence its own record. It does not deduplicate
linked parts by entity-token strings: that merged distinct components in the
first export and omitted both heel bars. It also exports root-context joints
and native cylinder axes for the selected ankle/heel/foot bodies.

The final dialog reports whether both heel meshes were exported. Zero joint
records remains a warning, not proof that the mechanism is a direct hinge;
cylinder axes can support geometric reconstruction but do not establish limits.
The script does not modify the CAD model, move joints, or change visibility.

Validation: two offline identity regression tests pass. Replaying all 948
occurrences from the first ZIP preserves both heel-bar identities with no
record merging. Fusion API execution still needs to be verified on the Mac.

API references:
- https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/fusion_Component.htm
- https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/core_Cylinder.htm
