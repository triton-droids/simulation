# Lower-body CAD extraction

The archive `/home/thewheelneedsme/Projects/droids/Full Assembly with PLA Arms.f3z` contains a document
named **Lower Body Reassembled**, with **Left Leg Assembly**,
**Pelvis Sub-Assembly**, and **Right Leg Assembly** dependencies.
`lower_body_archive_inventory.json` lists its 164 distinct document
dependencies. These are not counts of assembled instances or measured masses.

The binary Fusion geometry could not be evaluated in this Linux workspace.
Use the prepared script in Autodesk Fusion to calculate the physical properties:

1. Open/import `Full Assembly with PLA Arms.f3z` in Fusion.
2. Open **Scripts and Add-Ins**, add the `ExportLowerBody` script folder,
   and run `ExportLowerBody` as a Python script.
3. Save `lower_body_physical_properties.json` in this `cad` directory.

The script finds the exact named lower-body occurrence and exports its
aggregate mass plus nested occurrence/body properties. It never traverses
the upper-body subtree. If Fusion changes the component name on import,
the script stops rather than choosing an arbitrary assembly.

This script has been syntax checked, but cannot be executed without Fusion.
It exports raw properties using the documented Fusion API; geometry, frame
interpretation, material assignments, and link grouping still need validation.
Parent aggregate masses must not be summed with their children.

After export, compare the assembly total to the 16 kg hardware measurement,
check for any battery included in that subtree, and map bodies into rigid
links separated by actuated joints. Convert centimeters to meters and
kg*cm² to kg*m². MuJoCo needs inertia about each link's center of mass in
the link frame, so world-origin moments cannot be pasted directly into XML.
The original model remains unchanged. See the candidate below.

## Export results and 16 kg candidate

The received export reports a lower-body total of **16.270112574 kg**:

| Assembly | CAD mass (kg) |
| --- | ---: |
| Pelvis | 3.523294888 |
| Left leg | 6.385427639 |
| Right leg | 6.361390046 |

No occurrence name in this exported subtree contains `battery`, `upper body`,
or `torso`; this name check alone does not prove physical battery exclusion.
The CAD total is 0.270112574 kg (1.69%) above the hardware measurement.
The cause of that small discrepancy is not established.

`chrobot_16kg_candidate.xml` is a separate simulation candidate. It matches
the named subassemblies to the XML chain and normalizes their masses to
15.999 kg, preserving the existing 0.001 kg torso/IMU placeholder for a
16 kg compiled total. `mass_mapping_candidate.json` records every mapping
and mass change. Use the candidate XML path explicitly when loading a model;
existing training and playback commands still use their original models.

**This is not a completed CAD inertial calibration.** The original centers
of mass are retained and each original full inertia tensor is scaled by
new mass / old mass. The exported CAD centers of mass and tensors have not
been applied. CAD coordinates differ substantially from the old XML frame,
and assembly naming does not verify which parts move together across each
joint (particularly motor housings/rotors and hip connectors). These must
be resolved before treating this candidate as hardware dynamics.

Validation: MuJoCo compiled the candidate at 15.999999999993 kg; all massive
bodies have positive principal inertias; forward dynamics and 100 unforced
physics steps remained finite. This does not validate standing, walking,
policy compatibility, or hardware deployment. No trained policy or rollout
has been changed.

API reference: https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/fusion_PhysicalProperties.htm
