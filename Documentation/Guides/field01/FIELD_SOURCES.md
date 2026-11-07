# Field sources, revisions, and physical assumptions

Pinned reference SHAs are in source_refs.json. Retrieved2026-10-24.

- HSL-Rules: https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules
  commit06c44d4b0ea2fd3cd4fdcc2a7cb907f6ace400dc, rules/environment.tex.
- IsaacLabv3.0.0: https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0
  commite3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4; BSD-3-Clause.
  Code targets v3 launch_simulation/instantiate/*_index APIs, not v2 assumptions.
- Legacy reference: https://github.com/RoboCup-Humanoid-TC/hlvs_webots
  commit87829b1cc23a7474608cb8494e30d96c7bec879b. No Webots assets were copied.
  Repository license lookup did not establish licenses for HSL/legacy; they are
  measurement references only. No unknown-license meshes or worlds are redistributed.
- Ball size/mass source: https://www.theifab.com/laws/latest/the-ball/
  size5 circumference68–70cm, mass410–450g. We select69cm/.43kg, radius .10981691m.
  Pressure/deformation/internal air are not simulated; contact response is uncalibrated.
- Existing candidate UnitreeG1: Menagerie71f066ad0be9cd271f7ed58c030243ef157af9f4;
  unitree_g1/LICENSE carries Unitree's3-clause redistribution license. Original assets
  remain in the existing cache. License and mesh hashes are recorded by doctor.
  No supplied pupper-simulations.zip was found among these request attachments.

## Measurements and discrepancies

medium_14x9 uses14×9m, goal clear2.5×1.7m/depth1m, post width.08m, goal area1×4m,
penalty area3×6m, mark2m from goal line, center circle diameter3m, corner radius.5m,
line width.05m, center/penalty mark.1m. Measurements use outside line edges. Lines
are non-colliding thin visual boxes; circular lines use96segments (corners16 each),
so the curved rendering is an approximation. Collision ground is flat and uniform.
It is not physically simulated20–30mm grass. Goalposts and crossbars collide;
rear/base supports are simplified, no compliant net is modeled. Out-of-bounds is a
termination condition, not an invisible sideline wall. Gravity is fixed at9.81m/s².

The pinned draft allows Large Division on M or L fields, but its goal table lists
Large Division width2.6–3.1m and height1.8–2m whereas the M-field table supplies
2.4–2.6m width. The user contract explicitly chooses2.5×1.7m. We preserve that
configured test geometry and DO NOT certify it as event-compliant Large Division.
Confirm event rules before competition use. large_22x14 is an optional explicit
profile; geometries never change inside a training run or randomization episode.

Field sources are our procedural boxes/lines/USD definitions. Webots world imports
were unnecessary and would bring unrelated simulator dependencies and legacy rules.
White rendering lights are artistic defaults, not calibrated lux/exposure sensors.
All physics randomization ranges remain engineering hypotheses until measured.
