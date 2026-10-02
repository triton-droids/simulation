# Hardware tracking changes

Relative to `logs/legs_tracking/20260914_170827`:

- Actuator peak torque: all joints ±120 Nm → hip1/knee ±120, hip2/thigh ±60, ankle ±16 Nm. These are motor datasheet peaks with an estimated 0.95 ankle linkage factor; the deployed controller log also reports lower software command caps (80/40/11 Nm), which are **not yet imposed by this model**.
- Torque–speed: unlimited below the fixed torque caps → piecewise RS04/RS03/RS02 curves supplied in the training task, with 0.9 speed scaling for the estimated 42 V bus versus 48 V curves. Updated once per 20 ms control step, not every 5 ms physics substep.
- PD gains: kp 100/100/100/80/20 per leg unchanged; actuator kv 1 → 2 Nm·s/rad. Passive damping 1.0 → 0.1 (estimate). Per-episode kp scale 0.8–1.2 and kd scale 0.7–1.5 (estimates).
- Joint friction: 0 → RS04 0.3–1.0 Nm (knee measured approximately 0.53 and 0.81), RS03 0.2–0.8 Nm and RS02 0.1–0.4 Nm (estimates).
- Command path: instantaneous joint targets → per-joint safety clips plus 1 rad/s target slew matching the latest hardware log. This is a **target** limit, not motor shaft speed. The log has no verified acceleration-limit value, so none is modeled.
- Latency: zero → actor IMU/joint observation delay 0–1 control step and actuator delay 0–1 physics step. The hardware report estimates approximately 18 ms feedback age and 1–3 ms command transport.
- IMU: projected-gravity noise zero → uniform ±0.05; per-episode mounting tilt up to 2° (small-angle approximation) and gyro bias ±0.05 rad/s. A 44 Hz low-pass response is not yet modeled.
- Randomization: base COM absent → ±2 cm, physical-link mass and inertia scale about 0.9–1.1, armature 0.01 nominal ×0.5–2, and small velocity pushes every 3–6 s. Foot friction and encoder bias remain randomized. Massless floating base is excluded from inertia randomization.
- Reward: adds penalties for torque above 80% of the datasheet peak and above the stalled torque rating (28.5/15/6 Nm by motor).
- Interface: 50 Hz, 10 ordered joints, 56 actor observations, action scale 0.2, and the 299-frame reference are unchanged. ONNX includes the same named inputs and outputs.

Known model gaps: the ankle is still a single effective hinge instead of a measured linkage; the CAD masses and inertias are estimates; the 299-frame reference reaches 5.13 rad/s while the deployed position-target slew is 1 rad/s; and torque-limiting software and IMU filtering need confirmation before physical deployment. This simulation result alone does not authorize a hardware walk test.
