# G1 Build and Research Results

**Outcome:** Gates 0-3 passed, but the first authorized Gate 4 standard
commanded-velocity family **failed** its frozen acceptance rule. That negative
result remains unchanged below. A separate corrective campaign is now active:
the observation/history, command-resampling, touchdown-air-time, reset-control,
finite-state, foot-site-velocity, and evaluation invariants have been fixed and
tested. The new campaign has passed deterministic and PPO plumbing diagnostics
but has not yet produced locomotion evidence or a Gate 4 PASS. Phases 5-7 remain
unstarted.

## Active corrective campaign (not yet a Gate 4 result)

Every run in this table is intentionally non-comparable with the preserved
failed family because the transition semantics changed. Generated artifacts are
retained under `results/gate4_corrective/` and remain ignored by Git.

| Experiment | Commit / configuration | Seed and budget | Runtime | Result and interpretation |
|---|---|---:|---:|---|
| C00c deterministic scan | `3f437e3f711d0478f036d3e5656fcc8df81f1ba6`; corrected native G1; sinusoidal action amplitude `0.02`; CUDA; clean checkout | seed 1707; 1,000 control steps | 41.19 s | All state and observations were finite. The untrained diagnostic first terminated at step 74 and ended at pelvis height `0.1323 m`; this passes deterministic pipeline execution only and is negative behavior evidence. |
| C01 PPO integration | same clean commit; `ppo_g1_smoke`; 16 environments; randomized reset and observation noise; 32x32 networks | seed 0; 1,024 requested/actual steps | 280.03 s | Emitted both callbacks and `run_end`, wrote restorable checkpoints 0 and 1,024, and improved training-side return `-4.8356 -> -2.8073` with length `42.25 -> 43.75`. KL was `3.3128`, so the change is not learning evidence; C01 validates only training/checkpoint/provenance plumbing. |
| C01 fixed-command restore | checkpoint 1,024 versus its checkpoint-0 network and standing control; all eight command axes; nominal reset seed 2000; 64 steps (`1.28 s`) | 24 episodes | about 3.5 min external wall time | Both checkpoints restored and every value was finite. Trained/untrained/standing linear-vector RMSE was `0.5452/0.5476/0.6149`; yaw RMSE was `1.7945/1.9813/0.1697`; trained fall rate was `0.125` and success was `0`. This is the expected negative result for a tiny random policy and proves the evaluator path, not locomotion. |

C02 is predeclared as a nominal-reset, noise-free, 262,144-step corrective PPO
diagnostic (Brax rounds this to 286,720 actual environment steps). It may scale
only if metrics stay finite, PPO KL is not grossly unstable, episode length and
tracking improve beyond controls, and a fixed-command rollout shows plausible
balance/alternating support rather than leaning, hopping, or contact abuse.

## Principal reproduction commands

The matched family used the following WSL2 command, with `SEED` set to `0`,
`1`, and `2` and a distinct absolute `RUN_DIR` for each run:

```bash
python source/scripts/train.py --logger local --seed "$SEED" \
  env=unitree_g1 robot=unitree_g1 sim=unitree_g1 agent=ppo_g1 \
  robot.fetch_model=false \
  agent.num_envs=2048 agent.batch_size=64 agent.num_minibatches=32 \
  agent.num_updates_per_batch=5 agent.learning_rate=0.0001 \
  agent.num_timesteps=20971520 agent.num_evals=21 agent.num_eval_envs=32 \
  hydra.run.dir="$RUN_DIR" hydra.job.chdir=true
```

The frozen held-out evaluation command was run once per selected checkpoint,
using reset seeds `2000,2001,2002`:

```bash
python source/scripts/evaluate_g1.py \
  --run-dir "$RUN_DIR" --checkpoint "$CHECKPOINT" \
  --untrained-checkpoint 0 --steps 500 --seeds 2000,2001,2002 \
  --output-dir "$RUN_DIR/evaluation/checkpoint_$CHECKPOINT"
```

The three runs were combined without rerunning or selecting on held-out data:

```powershell
.venv\Scripts\python.exe source\scripts\report_g1_baseline.py `
  --run-dir results\gate4\baseline_corrective_seed0_20971520 `
  --run-dir results\gate4\baseline_corrective_seed1_20971520 `
  --run-dir results\gate4\baseline_corrective_seed2_20971520 `
  --output-dir results\gate4\final_aggregate
```

## Gate status

| Gate | Status | Evidence |
|---|---|---|
| 0 - audit/reproducibility | Passed | Clean CPU environment, dependency check, two loader modes, local logger, pinned sources/licenses, and credential-pattern scan |
| 1 - old baseline | Passed | Default 12-actuator environment protected by regression tests; narrow pre-existing observation, push, and manual-smoke defects fixed |
| 2 - G1 seam | Passed | Separate registered G1 robot/environment; exact resolver pin; deterministic 29-joint/actuator and contact metadata |
| 3 - MJX smoke | Passed | Seeded reset, finite 1,000-step scan, bounded position targets, termination checks, and two diagnostic videos |
| 4 - velocity baseline | **Failed** | Three matched 20,971,520-step seeds and 216 held-out episodes completed, but yaw, survival, per-seed, and observation-semantics requirements failed |
| 5-7 - HOMIE extensions | Not authorized / not started | Work stopped at Gate 4 as required |

## Environment and algorithm

- Robot: Unitree G1, `nq=36`, `nv=35`, `nu=29`.
- Model: MuJoCo Menagerie `unitree_g1/scene_mjx.xml` at commit
  `71f066ad0be9cd271f7ed58c030243ef157af9f4`.
- Behavioral reference: MuJoCo Playground G1 joystick at commit
  `8a4b4642d8eba8a80ac99ed125cb62c16e1457ad`.
- Runtime: repository-native Brax `PipelineEnv` over MJX; Brax PPO with an
  asymmetric 103-value actor observation and 216-value critic observation.
- Task: flat ground, commands `[v_x, v_y, yaw_rate]`; training ranges x
  `[-0.5, 0.5]`, y `[-0.3, 0.3]`, and yaw `[-0.5, 0.5]`.
- Action: 29 normalized values clipped to `[-1, 1]`, mapped to absolute
  position targets around the verified `knees_bent` pose, then clipped to the
  actual actuator ranges.
- Training: 20,971,520 environment steps per seed; 2,048 train environments;
  32 evaluation environments; episode length 1,000; unroll 32; batch 64; 32
  minibatches; 5 PPO updates; learning rate `1e-4`; discount `0.98`; GAE
  `0.95`; reward scale `0.1`; observation normalization enabled; gradient
  norm clipped at `1.0`; no pushes or domain randomization.
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB) through WSL2 CUDA.
  Native Windows JAX remained CPU-only.
- Logging: local JSONL and local checkpoints; no W&B account, cloud service,
  Triton infrastructure, remote branch, or physical robot was used.

## Final matched training family

Training-side checkpoint selection was frozen as the checkpoint with the
highest logged mean evaluation reward. These values are useful for diagnosing
the large training/held-out gap; they are not held-out success metrics.

| Policy seed | Selected step | Training eval return | Training eval length | Linear tracking accumulation | Yaw tracking accumulation | KL | Logged train + eval wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17,825,792 | 32.8730 | 629.97 | 868.67 | 592.39 | 0.03109 | 74.0 min at selection; 86.4 min through final callback |
| 1 | 19,922,944 | 31.7124 | 594.78 | 831.90 | 593.22 | 0.03252 | 80.8 min at selection; 85.0 min through final callback |
| 2 | 20,971,520 | 35.2003 | 696.50 | 1,059.98 | 707.47 | 0.03738 | 79.1 min through final callback |

All three exact runs contain 21 metric records, a `run_end` record, resolved
configuration, source provenance, and restorable full and compact checkpoints.
The matched family consumed 62,914,560 environment steps. Seed 0 and seed 1
regressed after their selected checkpoints; evaluation correctly used the
predeclared training-side selector instead of the final checkpoint.

## Frozen held-out evaluation

Each policy seed was tested deterministically on the same eight commands and
three randomized reset seeds. The untrained comparison uses checkpoint zero of
the matched network; the standing controller uses zero action. There are 24
episodes per controller per policy seed and 216 episodes total.

### Aggregate across policy seeds

| Metric | Trained | Untrained | Standing | Gate interpretation |
|---|---:|---:|---:|---|
| Linear velocity vector RMSE (m/s) | **0.9254** | 0.9878 | 0.9942 | Pass: trained is lower than both |
| Yaw-rate RMSE (rad/s) | **1.0575** | 0.4539 | 0.3238 | **Fail:** trained is substantially worse |
| Mean episode duration (s) | **1.3742** | 1.1364 | 1.1333 | Pass: trained is longer than both |
| Fall rate | **1.0000** | 1.0000 | 1.0000 | **Fail:** no full-horizon rollout |
| Episode success rate | 0.0000 | 0.0000 | 0.0000 | Fail |
| Finite rate | 1.0000 | 1.0000 | 1.0000 | Pass |

For the trained family, sample standard deviations across policy seeds were
`0.0456` for linear-vector RMSE, `0.1120` for yaw RMSE, and `0.1517 s` for
duration. The corresponding 95% t intervals were `[0.8121, 1.0387]`,
`[0.7794, 1.3357]`, and `[0.9973, 1.7511]`. With only three seeds these wide
intervals are descriptive, not confirmatory.

### Trained result by policy seed

| Policy seed | Selected checkpoint | Linear RMSE | Yaw RMSE | Duration (s) | Fall rate | Matched rule |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 17,825,792 | 0.9596 | 1.1567 | 1.3792 | 1.0 | Fail |
| 1 | 19,922,944 | 0.9429 | 1.0797 | 1.5233 | 1.0 | Fail |
| 2 | 20,971,520 | 0.8736 | 0.9361 | 1.2200 | 1.0 | Fail |

All three seeds improved linear RMSE and duration over their matched controls,
but all three failed yaw and survival. Therefore the frozen requirement that
at least two seeds pass is not close to satisfied.

### Additional diagnostics

The trained controller used much more effort and exhibited rougher motion than
the controls: mean actuator effort was `190.21` versus `81.94` untrained and
`75.37` standing; the mechanical-energy proxy was `398.90` versus `44.81` and
`35.81`; mean action-rate cost was `7.150` versus `0.037` and `0.000`. Trained
torso tilt was lower (`15.23 degrees` versus about `20.7/20.3`), but that did
not yield survival. Trained tracking-success fraction was only `0.0383`, below
both controls (`0.0469/0.0576`), and gait contact asymmetry was `0.3295` versus
`0.1274/0.1082`.

The strong late training-side returns and episode lengths did not generalize to
the frozen reset/command grid. Per-command inspection shows excessive yaw
motion even for commands that request no turn, followed by low-pelvis or
cross-contact termination. This is consistent with unstable, command-insensitive
motion rather than a valid velocity-tracking gait.

## Representative visual evidence

- Gate 1: `results/phase1_default_baseline_zero_action.mp4`.
- Gate 3 standing: `results/gate3/standing_zero_action.mp4`.
- Gate 3 bounded action: `results/gate3/bounded_action.mp4`.
- Gate 4 comparison plots:
  `results/gate4/final_aggregate/heldout_comparison.png` and
  `results/gate4/final_aggregate/training_curves.png`.
- Gate 4 fixed representative rollout: seed-0 checkpoint `17,825,792`, combined
  command `[0.4, 0.2, 0.35]`, reset seed `2000`, for trained, matched untrained,
  and standing controllers under
  `results/gate4/baseline_corrective_seed0_20971520/evaluation/representative_checkpoint_17825792_combined_seed2000/`.

All three MP4s decode at 640x480 and 25 fps. The trained clip contains 38
frames (`1.52 s` encoded, 74 control steps/`1.48 s` evaluated); the two controls
contain 27 frames (`1.08 s` encoded, 53 control steps/`1.06 s` evaluated).
Visual inspection confirms that the trained robot leans, fails to establish a
stable gait, and collapses sideways by the final frame; both controls also
collapse by their final frames. All three representative episodes terminated
on low pelvis height. On this fixed combined command, trained linear/yaw RMSE
was `0.680/1.110`, untrained was `0.811/0.568`, and standing was
`0.815/0.419`, matching the aggregate conclusion: some transient linear
improvement, much worse yaw, and no useful survival.

The first representative render completed the trained rollout but failed at
encoding because WSL lacked a system `ffmpeg`; that failure is retained under a
`_failed_no_ffmpeg` suffix. The evaluator now prefers the already-installed,
platform-specific `imageio-ffmpeg` binary and retains an OpenCV MP4 fallback,
with a regression test. Generated metrics, checkpoints, plots, and media remain
ignored by Git and are not included in the review zip.

## Retained failures and negative results

| Experiment | Status and finding |
|---|---|
| Initial PPO smoke | Failed at removed JAX pmap helper; led to a narrow, tested Brax compatibility adapter |
| Checkpoint retries 1-3 | Exposed compact-tree and reset-axis assumptions; every failed artifact was retained |
| 1,024-step CPU/CUDA PPO integration | Passed and produced restorable compact checkpoints before long training |
| 65,536-step capacity probes | Revealed reward clipping erased termination cost, then showed signed-reward throughput and unstable KL |
| 983,040-step signed-reward pilot | Increased return by collapsing survival to 3.8 steps; explicit early-termination exploitation |
| Alive-shaping retunes | Demonstrated that survival shaping could dominate command tracking |
| Fixed-command pilot | Exposed an incorrect terminal hand/thigh contact classification |
| Three 2,129,920-step nominal baseline runs | Invalidated because serialized normalization/gradient clipping were not actually forwarded to Brax |
| 10,485,760-step reference-scaled seed 0 | Failed held-out yaw (`1.971`) and had 100% falls; alive contribution dominated tracking |
| Five-evaluation full launch | Preserved compile-efficiency failure: 80 transition batches were compiled into one epoch |
| First final seed-2 process | Externally interrupted at 10,485,760 steps; preserved and replaced from scratch because policy-only restore is not continuation-equivalent |
| Final matched family | Completed normally but failed frozen yaw, survival, and per-seed criteria |
| First representative video encoding | Failed for missing WSL system `ffmpeg`; preserved and rerun with a local bundled encoder |
| Exported-source PPO smoke | Passed training/checkpoint/evaluation plumbing; 1,024-step KL was `6.11` and all diagnostic episodes fell, so it is not learning evidence |

## Final reproducibility verification

- Working Windows environment: `pip check` passed; the full suite reported 58
  passed and the two declared observation-timing tests xfailed.
- Independently created Gate 0 environment: `pip check` passed and the same
  suite reported 58 passed plus the same two expected failures.
- Clean exported source tree: extracted outside the repository, pointed at the
  exact pinned Menagerie checkout through `MUJOCO_MENAGERIE_PATH`, and reported
  58 passed plus two expected failures. It contained no `.git`, cache, virtual
  environment, generated result, log, checkpoint, or media directory.
- Clean-source training: the local 1,024-step PPO profile ran on `cuda:0`,
  emitted both progress callbacks and `run_end`, improved its integration-only
  evaluation reward from `-5.2194` to `-0.9420`, and wrote checkpoint 1,024.
- Clean-source evaluation: loaded that checkpoint, evaluated all eight commands
  for reset seed 2000, wrote 24 controller episodes plus JSON/CSV, and reported
  finite rate `1.0`. Its summary explicitly records that Git provenance is
  unavailable in the export rather than failing or inventing a revision.
- Both ordinary and MJX G1 loader modes passed offline with the exact Menagerie
  pin. Compileall passed for `source`, `scripts`, and `tests`; the metadata
  inspector again reported 29 one-to-one joint/actuator mappings.
- The retained clean-source integration artifact is
  `results/gate4/clean_snapshot_ppo_smoke_20260909/`. It is separate from the
  final matched research family.

## Remaining limitations

1. `Joystick.step()` constructs its returned observation before advancing
   `last_act`, phase, and command resampling. This makes previous action one
   additional step stale and makes the observation command disagree with
   `info["command"]` at a resampling boundary. Two strict expected-failure
   tests preserve the required invariants without changing completed-run
   semantics. This must be fixed before any new baseline training.
2. Every held-out rollout fell in roughly 1-1.5 seconds. Linear RMSE improvement
   during such short, unstable transients is not evidence of useful locomotion.
3. No standardized push-recovery or domain-randomization evaluation was run;
   both were intentionally disabled for the conventional first baseline.
4. The three-seed budget is about 20.97M steps per seed, far below Playground's
   large published G1 profile. Compute may matter, but a larger budget should
   not be spent until observation timing and task alignment are corrected.
5. The native Brax environment is a reversible integration seam, not a durable
   platform recommendation; Brax warns that its pipelines are not actively
   maintained. A thin pinned Playground runtime remains a credible alternative.
6. Results are simulation-only on one laptop GPU. There was no system
   identification, sim-to-real validation, physical G1 connection, or safety
   case.

## Deviation from the contract layout

The supplied papers were located directly under `research/references/` rather
than the preferred `research/references/papers/` directory. They were read from
their supplied paths and left in place so user-staged files were not moved or
duplicated.

## Recommendation

Do **not** proceed to HOMIE upper-body curriculum, height/knee rewards, or
symmetry yet. First fix and test the observation-update ordering, then run a
small command-conditioned yaw/reset diagnostic against the authoritative
Playground behavior. If the native port still diverges, compare it with a thin
pinned Playground wrapper before spending another full three-seed budget. Only
a new conventional family that passes finite tracking, yaw, survival, and
reviewable-rollout criteria would justify requesting authorization for Phases
5-7.
