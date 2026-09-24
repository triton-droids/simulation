# Lower-body mass audit

Target hardware: approximately 16 kg, lower body only, external battery.
These totals were verified by compiling the checked-in XMLs with MuJoCo.

Update: the Fusion export now reports **16.270112574 kg** for Lower Body
Reassembled. A separate `cad/chrobot_16kg_candidate.xml` compiles to **16 kg**
using CAD subassembly mass proportions. Original XMLs remain unchanged.
See `cad/README.md` and `cad/mass_mapping_candidate.json` for the approximate
link mapping and retained COM/scaled original inertia limitations.

| Model | Compiled mass |
| --- | ---: |
| `mjlab/src/mjlab/asset_zoo/robots/triton_robot/xmls/chrobot.xml` | 24.696541 kg |
| `src/robots/default_humanoid_legs/default_humanoid_legs.xml` | 23.893370785 kg |
| Older flat and rough scenes | 23.893370785 kg each |

The older `mj_model.json` reports 24 kg and rounded link masses; it is not
the authoritative compiled mass of the current XML.

## mjlab link masses

| Body | Mass (kg) |
| --- | ---: |
| torso | 0.001000 |
| hip | 5.159055 |
| left_leg1 | 2.189611 |
| left_leg2 | 1.709674 |
| left_leg3 | 3.156742 |
| left_leg4 | 1.980828 |
| left_foot | 0.765723 |
| right_leg1 | 2.189548 |
| right_leg2 | 1.722840 |
| right_leg3 | 3.074969 |
| right_leg4 | 1.980828 |
| right_foot | 0.765723 |

The legs alone total 19.536486 kg. The full model exceeds 16 kg by
8.696541 kg. Removing that difference from the hip would require negative
hip mass. There is no separately named battery body in this model.

## Correcting the model

Measure the pelvis and link masses, including mounted motors, electronics,
and cables. Identify where any historical battery allowance was assigned.
Update center of mass and inertia as well as mass when removing a localized
component. Matching total mass alone does not identify the correct dynamics.

If only total mass is available, uniform scaling by 16 / 24.696541 can
produce an approximate 16 kg model. Scale each full inertia tensor by the
same factor when assuming unchanged geometry and normalized mass
distribution. This is an approximation, not a measured battery-removal
model, and has not been applied to the source XML.

The older model infers mass and inertia from geom dimensions and default
density; it is a separate embodiment, not a copy of the CAD-based mjlab
robot. Its training code also supports restoring a saved run XML, so editing
the source does not necessarily change resumed training. Confirm the actual
model used by each training/playback command and preserve historical run
artifacts. The rollout videos alone do not establish their model provenance.

## UMR and walking

[UMR](https://github.com/hanyang9/UMR) learns surface correspondence and
retargets motion references through kinematic optimization. It does not
convert a G1 locomotion policy to a new embodiment by swapping XMLs.
Its published examples use full humanoids; a legs-only surface needs its
own correspondence validation.

For command-driven walking, configure and train a locomotion policy for
the actual lower-body model. For imitation, first retarget suitable walking
references, then train a tracking policy on the corrected dynamics. Existing
G1 task settings may serve as a starting point, but joint ordering, limits,
actuator capability, observations, gains, and action scales must match the
legs. Evaluate a policy trained for the corrected model before hardware use.
If the tether supports weight or constrains motion, those forces also need
to be represented; an external power cable alone is not an onboard payload.
