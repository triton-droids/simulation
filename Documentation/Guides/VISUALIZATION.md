# Visualization

## See the robot walk

```powershell
.venv\Scripts\python.exe Scripts\visualize.py
```

The console first restores checkpoint `1003520` and JAX may compile for about a
minute. MuJoCo then opens in a separate window named `MuJoCo : g1 scene`. If
the console prints commands but you see no camera, use Alt-Tab and select that
window.

The default demo cycles through standing and movement commands so the saved
policy can be inspected. It uses:

- Robot: Unitree G1
- Physics/viewer: MuJoCo
- Policy: T02 `rate_cost`
- Weights: `App/Models/T02/logs/checkpoints/1003520/policy`
- Scene: visible RoboCup-style field, goals, and ball

## What the viewer proves

It proves that the saved policy can be restored and driven in the MuJoCo
viewer. It does not prove that the policy perceives the ball, kicks, plays
soccer, runs in Isaac Lab, or transfers to the physical G1.

## Flat developer scene

The public script deliberately opens the field. Developers can request the
flat scene through the CLI:

```powershell
.venv\Scripts\python.exe robocub.py watch --scene flat
```

## Record a field run

```powershell
.venv\Scripts\python.exe robocub.py watch --record runs\field_capture
```

Recording behavior and supported output files are controlled by the internal
field viewer. Use the live window first before treating a recording as an
evaluation result.
