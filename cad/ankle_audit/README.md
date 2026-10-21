# Missing heel linkage audit

The user confirmed the missing part is the long flat two-hole heel linkage
shown in `Left_Foot_Sub-Assembly.png`. These PNGs are original Fusion archive
previews, not generated illustrations.

## Findings

- The Fusion dependency inventory names `L Heel Linkage` and `R Heel Linkage`.
- The validated Fusion export is `fusion_export_20261020_233848`. It contains
  both heel-linkage bars, surrounding ankle geometry, occurrence transforms,
  and cylinder axes. Preliminary mesh-recovery artifacts were archived outside
  the repository after this direct export replaced them.
- Each heel bar has approximately 166.248 mm hole-center spacing. The inferred
  ankle crank radius is 45 mm. These dimensions support the inspection model;
  they do not independently calibrate the real motor axis or transmission.
- Current `left_foot.stl` only extends to about 53.26 mm above its mesh origin.
  The detailed and simplified shin meshes share essentially the same bounds;
  replacing `_dumb.stl` with the detailed mesh is not a demonstrated repair.
- Four humanoid URDF variants on each of `isaac_humanoid_locomotion` and
  `isaaclab_humanoid_v2` use one revolute ankle joint per foot, with no separate
  heel-linkage joint. None supplies a ready-made articulated replacement.
- `cad/triton_lower_body_actuated.xml` drives the foot hinge directly. Its own
  comment explicitly calls the ankle transmission an approximation pending
  measured linkage geometry.

The existing training model has **not** been edited. A cosmetic bar rigidly
attached to the foot would give the wrong motion; adding a separate dynamic
bar without correcting lumped masses could also double-count mass.

## Fusion exporter

On the Mac, open the assembled robot in Autodesk Fusion. Use Utilities →
Scripts and Add-Ins to add and run the folder `cad/ExportAnkleLinkage`.
The assembly must contain the occurrence named `Lower Body Reassembled`.
Choose an output directory; the script writes a timestamped folder containing
`ankle_linkage.json` and component-local STL meshes.

The script exports native joint frames, owning components, occurrence
placements, motion limits, and selected ankle/heel/foot geometry. CAD distances
are recorded in cm; STL vertices are explicitly exported in mm. It records
unavailable properties and missing joint data instead of inventing pivots.
It does not move the robot or change joints or visibility. This exporter has
passed Python syntax validation, but has not been executed inside Fusion here.

The final export is already present. The assembly's CAD joints are mostly rigid,
so motor axis, zero offset, travel, backlash, and motor-to-foot calibration are
still required from hardware measurements.

## Validation after export

1. Resolve assembly frames and compare both legs with the neutral CAD pose.
2. Build a separate articulated MJCF candidate with the linkage closure and
   correct motor attachment. Preserve the original training model for comparison.
3. Sweep the permitted ankle range; verify pin closure, travel, branch
   continuity, and motor-angle to boot-angle mapping on both sides.
4. Reconcile link masses/COM/inertias with the existing lumped foot/shin masses.
5. Validate motor-to-joint torque and velocity conversion before retraining.

## API references

- [Joint frames](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/fusion_Joint.htm)
- [As-built joint frames](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/fusion_AsBuiltJoint.htm)
- [STL export](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/fusion_ExportManager_createSTLExportOptions.htm)
- [Explicit mesh units](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/fusion_STLExportOptions.htm)
