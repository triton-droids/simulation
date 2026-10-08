# System identification from the v2 handoff

These are offline tools. They do not connect to hardware or modify the active
robot checkout. The archived raw data remains under
`/opt/him/triton_droids/handoff-review/triton_velocity_handoff_20261007_v2/sysid/session`.

`sysid_fit.py` and `sysid_report.py` originate from the hash-checked handoff.
The fitting tool now accepts `--sim-repo` and `--mjcf-ref`, and plot input paths
come from the raw session rather than assuming the output directory is inside it.

`validate_fit.py` replays the saved post-slew/post-guard commands for held-out
trials, pins the MJCF commit recorded in the fit, and retains prediction and
measurement arrays as NPZ. It refuses to overwrite an existing output directory.

```bash
cd /opt/him/triton_droids/simulation-system-id
/opt/him/table-tennis/envs/tt-mujoco/bin/python cad/system_id/validate_fit.py \
  --session ../handoff-review/triton_velocity_handoff_20261007_v2/sysid/session \
  --params ../handoff-review/triton_velocity_handoff_20261007_v2/sysid/sysid_params.json \
  --motor 9 --output reports/holdout_m9_new
```

This existing CPU interpreter contains MuJoCo, SciPy and NumPy; it was not
upgraded. Its location does not imply it is a validated PPO environment.

October 8 motor-9 replay in `reports/holdout_m9_20261008`:

| Held-out command | Defaults RMS | Fitted RMS |
|---|---:|---:|
| 3-degree step | 0.368 degrees | 0.265 degrees |
| 0.5 Hz sine | 0.495 degrees | 0.240 degrees |

This supports the fitted knee model for the measured supported trials. It does
not validate peak torque/speed, foot contact, or the unhealthy motor-5 ankle.
The fitter estimates receive-time measurements using half the measured round
trip; transport asymmetry is an assumption, not directly identified delay.
The source archive's hip fits are explicitly gantry-confounded. A fitted
actuator net is not warranted solely by these low-amplitude trials: first
evaluate repeatable residuals across held-out speed/load/pose conditions.

The deployed tracking runner/logger sources are missing from the pushed repo.
See `embedded-system-id/docs/SYSTEM_ID_AUDIT_20261008.md` for the exact source
hashes, communication failures, static preflight and source-sharing requirements.
Do not import the old 56-observation/time-indexed policy into a 39-input actor.
