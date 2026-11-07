# Watch G1 walk on Field #1

From this repository, run:

```powershell
.venv\Scripts\python.exe Scripts\visualize.py
```

It loads the existing T02 checkpoint and opens a native MuJoCo window after
policy compilation. Startup can take a minute or longer. The robot automatically
tours forward walking, turns, sideways and backward commands. This is the actual
learned G1 controller, not the field infrastructure's joint-motion smoke test.

Arrow keys command forward/backward/turning. End requests standing, Enter resumes
the tour, Backspace resets, Home follows G1, and F6 shows the entire field.
Close the viewer to stop its simulation child process. Falls are printed in the
console and cause an explicit automatic reset after two seconds.

## Scope

This demo uses **MuJoCo**, preserving T02's tested flat-ground dynamics and policy
observations. Field markings, goals and the displayed ball are **visual overlays**;
goal collisions and kicking are not implemented here. The separate Isaac Lab 3.0
field has physical field/ball geometry but does not yet run T02. No Isaac policy
transfer, soccer-playing ability, new training or hardware validation is claimed.

Dependencies are the existing `.venv`, pinned local model/playground caches, and
`Artifacts/Results/post_f1_t02/rate_cost/train` checkpoint/config. These large local artifacts
are not bundled in Git. A missing checkpoint produces an explicit startup error.
The original viewer implementations are preserved under `robocub/tools/`; the public
field entry point is `App/Scripts/visualize.py`.

To record a bounded 600-step / 12-simulation-second clip, choose a NEW directory:

```powershell
.\watch_soccer.cmd --record Artifacts/Results/field01/my_walk
```

The recording contains every fifth simulation frame at 10 fps, plus all 600 root
positions, a whole-field screenshot and numeric summary. Rendering can take longer
than real time. The interactive window drops old display frames to stay responsive.
