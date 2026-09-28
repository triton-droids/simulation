# T02 simulation controller handoff

Development prototype only. No hardware, perception, kicking or get-up controller.
The fixed T02 checkpoint is 1003520 (seed11, warmstarted training); it is NOT a
three-independent-seed result. Historical nominal tests survived24/24 episodes;
randomized backward seed6001 failed at109 steps. Preserve that limitation.

## Local setup and demo

Use Linux/WSL with the existing verified `.cache/g1_wsl_venv` environment. The
checkpoint is ignored by Git. Create an offline handoff from the provisioned repo:

```sh
.cache/g1_wsl_venv/bin/python source/scripts/package_g1_controller.py --output-dir results/milestone1/package01
```

The package contains source.zip, run configs/checkpoint, historical numeric evidence,
runtime_versions.txt and SHA256 manifest. It does not bundle the Python environment
or external model/source caches. Extract source.zip into a clean directory, provision
the recorded Python dependencies, and explicitly point MUJOCO_MENAGERIE_PATH and
MUJOCO_PLAYGROUND_PATH to the pinned cache checkouts recorded in run/*source.json.
Use the package run directory with --run-dir. Source pins must match; no downloads
are performed by the demo. The dependency snapshot describes the tested environment,
not a guarantee that arbitrary Windows installations are supported.

From the provisioned repository root:

```sh
.cache/g1_wsl_venv/bin/python source/scripts/demo_g1_controller.py --output-dir results/my_t02_demo --video
```

Output must be a new directory. This headless demo saves MP4 and JSON for forward,
turning and stopping in one continuous episode. `--profile fixed` runs eight
independent nominal commands; `--profile smoke` checks restoration with five steps.
Failures terminate the episode and are retained in output. Read summary.json;
a produced video or successful process exit alone is not a behavioral pass.

## Command interface

See examples/g1_velocity_commands.py. SimulationController.reset(seed) explicitly
starts a nominal episode. set_command(vx,vy,yaw_rate) takes body-frame forward/left
velocities in m/s and positive left yaw rate in rad/s. Each step advances0.02s;
command updates apply at the next step. Allowed input envelope is vx[-.25,.45],
vy[-.25,.25], yaw[-.4,.4]; arbitrary simultaneous combinations are not validated.
The eight fixed commands are in research/queues/final_f1_commands.json.
stop() requests[0,0,0]; it does not freeze physics or promise instantaneous arrest.
A terminated episode requires an explicit reset; there is no fall recovery.
Observation normalization is restored with the policy. This API owns simulated
state and must not be treated as a hardware transport.

Verification results and measured limitations are recorded in
research/MILESTONE_1_REPORT.md. Milestone 1 verification passed on 2026-09-28. Existing Milestone2 research remains preserved and deferred.
