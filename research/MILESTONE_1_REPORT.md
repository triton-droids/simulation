# Milestone 1 report — completed simulation prototype handoff

Date: 2026-09-28 UTC. Scope: the existing T02 rate_cost policy only; zero training.
Milestones 2 and 3 remain deferred. This is not Gate4 or three-seed success.

## Outcome

The packaged controller restored successfully, passed eight fixed nominal command
checks and a continuous 30-second forward/turn/stop sequence, and ran from an
isolated extracted source directory with explicitly provisioned external caches.
The verification queue ran01:55:24–02:15:49 UTC (about20min25s, within the2h cap).
Source revision:5018a8bb82f20fe6e3bb5f210b239610a84c1a3c.

## Deliverables and use

- Interface: `source/locomotion/unitree_g1/simulation_controller.py`.
- Setup/API guide: `docs/G1_CONTROLLER.md`.
- Integration example: `examples/g1_velocity_commands.py`.
- Local package: `results/milestone1/package01/`; SHA256 list in manifest.json.
  Includes fixed checkpoint1003520, configs, normalizer-bearing policy,
  provenance, historical numeric evidence, source.zip and runtime_versions.txt.
- Demo videos and raw per-step JSON: `results/milestone1/fixed01/` and
  `results/milestone1/transitions01/`. Summary JSON in each directory.
- Clean-directory proof: `results/milestone1/clean01/summary.json`.
- Visual contact sheets and derived metrics: `results/milestone1/review/`.

From the provisioned repository root in WSL/Linux:

```sh
.cache/g1_wsl_venv/bin/python source/scripts/demo_g1_controller.py --output-dir results/my_t02_demo --video
```

Choose a new output directory. The demo renders to MP4; it is not an interactive
hardware controller. Commands are body-frame vx/vy in m/s and yaw rate in rad/s,
updated at50Hz. See the guide for bounds and explicit reset/stop semantics.
The package and checkpoint are local ignored artifacts, NOT included in Git.
Archive the package separately if the project might be deleted. No upload occurred.

## Measured behavior

All fixed episodes used nominal reset6000 and lasted500steps/10s without termination.
Minimum pelvis height across them was0.745m (required>.6m). The frozen gates were
stand linear/yaw RMSE<=.15; moving linear<=.35/yaw<=.5. All passed.

| Command | Linear RMSE m/s | Yaw RMSE rad/s | Measured mean requested-axis speed |
|---|---:|---:|---:|
| Stand | .11860 | .13308 | vx -.0405, vy -.0228 m/s |
| Forward .45 | .11589 | .12215 | .4315 m/s |
| Backward -.25 | .12121 | .12010 | -.2332 m/s |
| Left .25 | .17593 | .14468 | .1137 m/s |
| Right -.25 | .12309 | .10645 | -.1750 m/s |
| Turn left .4 | .11180 | .14391 | .3270 rad/s |
| Turn right -.4 | .09674 | .19635 | -.3297 rad/s |
| Combined [.35,-.15,-.3] | .13720 | .13626 | [.3202,-.0563,-.3423] |

The continuous episode changed commands after500 and1000steps without resetting
physics. All1500steps survived. The final standing segment had whole-segment
linear/yaw RMSE .11745/.11341; final250steps .11872/.11476, passing the frozen
settled-standing thresholds. This demonstrates the tested transition only, not
arbitrary command switching or instant stopping.

Visual review inspected12 time-distributed frames from each of all9 videos:
forward/backward/lateral stepping, both turn directions, combined motion, standing,
and continuous movement/turn/stopping. Sampled frames show upright posture and
changing support/heading consistent with numeric motion. Standing retains small
foot/posture adjustments and drift. No collapse is visible in these samples;
full per-step termination/height records also passed. This sampled visual review
is suitable for this prototype handoff; it is not continuous-frame contact/slip
certification or final research gait validation. All full videos remain available.

## Portability and verification

All packaged checksums passed. Extracted source.zip restored the packaged policy
and produced a rendered smoke video in a clean working directory. The test shared
an explicitly named existing Python environment and pinned caches; it did NOT test
a fresh dependency installation. Runtime includes JAX/JAXlib .11.0, Brax .14.2,
MuJoCo/MJX3.10.0, Orbax .12.1; full exact snapshot is packaged.
Menagerie pin:71f066ad0be9cd271f7ed58c030243ef157af9f4.
Playground pin:8a4b4642d8eba8a80ac99ed125cb62c16e1457ad.
The first smoke caught a file-versus-directory checkpoint check; it was repaired
before the successful verification. Eight interface tests passed before launch.
Final regression output is saved in `results/milestone1/review/tests.log`.

## Limitations and preserved research

Lateral motion substantially undershoots requested speed, especially leftward.
Standing means bounded drift/adjustments, not motionless planted feet. The input
bounds are an envelope, not proof of every simultaneous combination. Only the
listed nominal reset/commands and transition were checked for this handoff.

Historical full randomized evaluation remains23/24 survivors: backward command
[-.25,0,0] at disturbed reset6001 failed after109steps. Its episodes.csv/summary
are retained in the package's historical_evidence/full_randomized directory and
original results/post_f1_t02/rate_cost/full_randomized. Nominal24/24 and extra
forward12/12 historical survival do not erase this failure. No robustness,
three-independent-seed, soccer, get-up or real-hardware claim is made.

T02 is development seed11 warmstarted from B07airtime_gate, with1,003,520 continuation
steps and action-rate penalty-.1. It is not an independently trained final recipe.
All Milestone2 work, including intentionally stopped R03 and checkpoints, remains
preserved. Resume only with new user authorization and research/MILESTONE_2_RESUME.md.
Milestone1 is complete; no further experiments are authorized by this handoff.

Final targeted regression result:44 passed, one upstream JAXopt deprecation warning.

The recurring automation was paused and its saved status verified as PAUSED. The completed milestone status is a persistent stop marker.
