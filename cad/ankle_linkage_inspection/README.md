# Triton heel-linkage inspection candidates

The missing long heel bars are restored as independently hinged links with
closed-loop pin constraints. Orange is the actual exported bar geometry; blue
is a simplified crank proxy. These models are for inspecting the mechanism,
not training or hardware deployment.

## Which robot and which file?

- `../triton_lower_body_legacy_actuated.xml`: older normalized 16 kg model.
- `../triton_lower_body_actuated.xml`: current hardware-aware Triton baseline,
  used by the hardware tracking and velocity tasks. Unchanged by this work.
- `triton_lower_body_linkage_candidate.xml`: full-robot inspection copy with
  the articulated heel bars. Includes the same 10 actuator names, plus four
  passive joints; it is not a drop-in checkpoint/policy replacement.
- `ankles.xml`: isolated two-ankle CAD-frame inspection rig.

`triton_ankle_sweep.mp4` shows an analytically positioned ankle-angle sweep,
not a learned policy or a walking evaluation. `ankle_sweep.mp4` shows the
isolated mechanism. `triton_lower_body_linkage.png` shows the full robot.

## Evidence and validation

Source: `../ankle_audit/fusion_export_20261020_233848/ankle_linkage.json`.
Both native heel meshes and cylinder axes are present. The CAD bar pin spacing
is 166.2476 mm and the inferred crank radius is 45 mm. The exported left upper
pin and bar hole differ by 2.649 mm, while the right aligns. The isolated rig
closes the geometry without stretching the bar, adjusting its neutral pose.

The legacy robot and the new CAD use different coordinate frames. The full
candidate registers the lower pin to the existing heel-bearing mesh bores and
estimates the motor axis from the legacy cylindrical motor shell. The existing
left/right foot meshes are mirrored; this registration keeps their labels.
Motor-shell fit residual is approximately 0.0445 mm; that is not independent
hardware calibration. Full pivot coordinates are in `full_robot_validation.json`.

Both models compile. Each side passes 241 poses across [-0.6, +0.6] radians
(the existing simulator's range, not verified hardware travel), with pin gaps
below 1e-8 m. The full candidate also passes 1,000 zero-gravity simulation
steps with sinusoidal ankle position commands; maximum pin gap is about
0.013 mm. This checks constraint behavior, not grounded locomotion.

## Remaining integration work

The original lumped link inertias are retained, and the bar masses plus 1 g
crank proxies are added. Consequently the candidate is 16.06097 kg versus the
16 kg baseline. Reassign the bar/crank mass and inertia from the existing
lumped links using validated rigid-body grouping before dynamics training.
The proxy crank inertia is not an identified rotor inertia.

The model still uses effective foot-joint position actuators. A calibrated
motor-side model must account for the linkage angle/torque mapping, signs,
zero offsets, limits, friction and reflected inertia. Candidate nq=21 versus
baseline nq=17; select the ten controlled joints by name when rebuilding
observations and actions, and verify policy and deployment contracts.

Contact geometry on the new bars/cranks is disabled. Mechanical interference
and travel limits still need inspection against the actual assembly.

## Rebuild

From the simulation repository:

```sh
UV_CACHE_DIR=/tmp/droids-uv-cache uv run --project mjlab --no-sync python cad/build_ankle_linkage_inspection.py
UV_CACHE_DIR=/tmp/droids-uv-cache uv run --project mjlab --no-sync python cad/build_triton_ankle_linkage_candidate.py
```

The scripts regenerate meshes, XMLs, validation JSON, stills and videos. They
write only the inspection directory and never overwrite the baseline models.
