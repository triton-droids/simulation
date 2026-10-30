# humanoid_legs_5 for Antioch

This is the small deployment branch for Triton Droids' lower-body linkage
inspection asset. Start with [the VS Code notebook](notebooks/ANTIOCH_HUMANOID_LEGS_5.ipynb).
The portable [MJCF and videos](robots/humanoid_legs_5/) are included with all
24 meshes. `validate_mjcf.py` checks the MuJoCo linkage locally.

The linkage is geometrically closed and mass-accounted, but it has not passed
Isaac Sim/PhysX import or hardware calibration. See the robot README before
training. Development history and Fusion extraction tools remain on the
`codex/hardware-tracking-retrain` branch (source commit `1d01b69`).
