# G1 Build and Research Results

## Successor C07 update (2026-09-16; zero new PPO steps)

Gate 4 remains unpassed. Saved C06 inference exactly matches Brax training
construction on 64 nontrivial inputs; an autoreset regression exposed and fixed
lost timeout/episode bookkeeping (D-033). C06's corrected eight-command,
three-reset-seed evaluation confirms static stance for stand/forward, with
zero foot transitions across all six focal episodes. The same-MJX oracle
comparison isolates reward/exploration as a plausible remaining bottleneck:
the walking oracle earns only 14.4% more reward per step than C06 stance.

| C07 diagnostic | Result | Artifact under `results/gate4_corrective/` |
|---|---|---|
| Saved C06 inference parity | max action difference `0`; identity preprocessing differs by `1.9926244` | `C07_audit/action_parity.json` |
| C06, nominal seeds 2000/2001/2002, forward | 500/500 steps each; RMSE `.499923/.499994/.499635`; 0% single support; 0 transitions each foot | `C07_C06_normalized_three_seed/episodes.csv` |
| C06 corrected forward video | inspected six frames across 10 s: stationary upright stance | `C07_C06_normalized_three_seed/trained_forward_seed2000.mp4` and `forward_montage.png` |
| C04 normalized, nominal seed 2000 | all eight trained commands fall at 60-62 steps; mean linear RMSE `.754139`, yaw `.327019`; no hidden gait | `C07_C04_normalized/` |
| C02 normalized, nominal seed 2000 | all eight trained commands fall at 72-83 steps; forward RMSE `.680555`; no sustained gait | `C07_C02_normalized/` |
| C06 in same-MJX comparison | 500 steps; mean vx `.000573`; RMSE `.499670`; min pelvis `.756060`; 0% single support | `C07_same_mjx_reward_comparison/C06_trace.npz` |
| Exact oracle in same MJX | 500 steps; mean vx `.547185`; RMSE `.173076`; min pelvis `.707041`; 74.8% single support; 30/32 foot transitions | `C07_same_mjx_reward_comparison/oracle_trace.npz` |

The same-MJX comparison fixes phase frequency to 1.5 Hz, reset seed 2000,
nominal pose and command `[.5,0,0]`. Both use the actual adapter's clipped
actions and synchronized observations. Oracle maximum absolute action is
`.985630`, so clipping does not explain the difference. It remains a shipped
policy diagnostic, not a local training result. Weighted reward totals before
dt are `1.450191` (stance) and `1.658568` (gait); stance still earns `.368905`
linear tracking, `.739729` yaw tracking and `.466800` phase reward. Gait earns
`.892206/.613743/.693651` respectively, but incurs more contact-force and
movement costs. Reweighting linear tracking alone from 1 to 3 on these fixed
traces gives `2.188001/3.442980` (57.4% oracle margin); this is counterfactual
reward arithmetic, not evidence that PPO can acquire the gait.

Focused evaluator/adapter regressions: 22 passed, one deliberately deselected
real-MJX compile test in 115.66 s. The subsequent complete suite passed
**97 tests, 25 warnings in 671.76 s**, including real MJX and all expanded reset
cases, with no skipped/xfail/failed tests. Compileall and `git diff --check`
also pass. C02 recovery is complete; historical-native recovery is underway.
No new training has been launched at this update. The earlier dated freeze is
retained verbatim as provenance.

**Outcome at the 2026-09-15 handoff freeze:** Gates 0-3 passed. Gate 4 has
**not passed**. The original three-seed Gate 4 training family completed, but
commit `f642bc2` proved that earlier fixed-command evaluators loaded checkpoint
running statistics but reconstructed policy networks without Brax's
`running_statistics.normalize` preprocessor. This invalidates evaluation of
checkpoints actually trained with normalization enabled, including the old
matched family; those held-out metrics/videos establish neither a PASS nor a
faithful definitive failure. Early pre-D-012 pilots actually trained with
normalization disabled, so their identity evaluations remain faithful only to
those unintended profiles. Files and raw values remain below.

The corrective campaign fixed and tested the observation/history,
command-resampling, touchdown-air-time, reset-control, finite-state,
foot-site-velocity, reward, PPO-propagation, and evaluator invariants. C05b
proves the exact physical/control interface supports a real alternating gait.
C06 then produced a 10-second nominal static stance for stand/forward at reset
seed 2000, not forward walking: under a correctly normalized `vx=0.5`
evaluation it used 100% double support, made zero
foot transitions, and had vector RMSE `0.4999 m/s`. No conventional locomotion
baseline is verified. Phases 5-7 remain unstarted.

> **Validity boundary:** pre-`f642bc2` fixed evaluation omitted observation
> normalization. It is invalid when the checkpoint was trained with
> normalization enabled (D-012 runtime-control family onward), but faithful to
> the actual identity-preprocessed function for earlier normalization-disabled
> pilots. Training-side PPO metrics, standing-controller results, source
> introspection, and C05b are unaffected. The only corrected saved evaluation
> of a normalization-enabled checkpoint at this freeze is C06's
> `checkpoint_5007360_normalized_seed2000_500` directory.

Historical native run directories predate per-run Git manifests. Their
surviving evaluator summaries record base HEAD
`6640663e5b9a50f25264e392a4c18703ca2c00e7` with `dirty=true`, but the evolving
dirty source diff was not archived separately. Accordingly, an exact source
commit is not recorded for those runs; `27de436` is only the closest later
committed snapshot of their final semantics. The clean-export smoke correctly
records Git as unavailable. Corrective C00c-C06 rows below carry exact clean
commits.

## Active corrective campaign (not yet a Gate 4 result)

Every run in this table is intentionally non-comparable with the preserved
failed family because the transition semantics changed. Generated artifacts are
retained under `results/gate4_corrective/` and remain ignored by Git.

| Experiment | Commit / configuration | Seed and budget | Runtime | Result and interpretation |
|---|---|---:|---:|---|
| C00c deterministic scan | `3f437e3f711d0478f036d3e5656fcc8df81f1ba6`; corrected native G1; sinusoidal action amplitude `0.02`; CUDA; clean checkout | seed 1707; 1,000 control steps | 41.19 s | All state and observations were finite. The untrained diagnostic first terminated at step 74 and ended at pelvis height `0.1323 m`; this passes deterministic pipeline execution only and is negative behavior evidence. |
| C01 PPO integration | same clean commit; `ppo_g1_smoke`; 16 environments; randomized reset and observation noise; 32x32 networks | seed 0; 1,024 requested/actual steps | 280.03 s | Emitted both callbacks and `run_end`, wrote restorable checkpoints 0 and 1,024, and improved training-side return `-4.8356 -> -2.8073` with length `42.25 -> 43.75`. KL was `3.3128`, so the change is not learning evidence; C01 validates only training/checkpoint/provenance plumbing. |
| C01 fixed-command restore | checkpoint 1,024 versus its checkpoint-0 network and standing control; all eight command axes; nominal reset seed 2000; 64 steps (`1.28 s`) | 24 episodes | about 3.5 min external wall time | Checkpoint deserialization and finite simulation succeeded, but trained-policy inference omitted observation normalization. The raw `0.5452/0.5476/0.6149` linear and `1.7945/1.9813/0.1697` yaw values are preserved but invalid for policy comparison. |
| C02 native corrected-learning diagnostic | `8a7fce5c0635a602d9b2f1af2b48bf49f0b7c475`; `ppo_g1_corrective`; nominal reset; no observation noise/push/domain randomization; 512 environments; clean checkout | seed 0; 262,144 requested, 286,720 actual | 805.62 s | Training-side return rose `0.3319 -> 0.9259`, final KL was `0.03973`, and length rose only `72.0 -> 75.41` steps; its training-side evaluation still terminated. This weak signal did not justify scaling the native profile. |
| C02 fixed-command/video diagnosis | checkpoint 286,720 versus checkpoint 0 and standing; eight commands; nominal reset seed 2000; 200-step (`4 s`) request; videos enabled | 24 episodes | 322.44 s | **Invalid trained inference:** normalization was omitted. The raw duration/RMSE/contact numbers and videos remain for provenance but cannot diagnose the trained checkpoint. C02's training-side length still did not justify scaling; a normalized reevaluation remains cheap and valid under the corrected native semantics. |
| C03 pinned-Playground PPO integration | `03794010c6d994e5439c203febc16bae8497bfdf`; exact pinned Playground G1 adapter; coherent full reset; nominal reset; no observation noise/push/domain randomization; `ppo_g1_smoke`; clean checkout | seed 0; 1,024 requested/actual steps | 505.64 s recorded training wall time | Both callbacks and `run_end` were emitted, source/effective configs and the complete runtime manifest were written, and checkpoints 0 and 1,024 restore. Return changed `-0.6917 -> -2.4834`, length `64.0 -> 63.75`, and KL was `3.2936`; as predeclared, this is successful infrastructure evidence and negative/non-comparable learning evidence. The long first compile is retained as a practical cost of the JAX authoritative graph plus coherent full resets. |
| C03 held-command restore | checkpoint 1,024 versus checkpoint 0 and standing; eight commands; nominal reset seed 2000; 32 steps (`0.64 s`) | 24 episodes | 131.71 s recorded evaluation wall time | Checkpoint loading and finite stepping succeeded, but normalized trained-policy inference was not reconstructed. Its tracking/contact values are invalid learning evidence. |
| C04 authoritative full-command diagnostic | `16bb816181247628788fb59c797a94f4f15226f3`; exact pinned Playground G1 JAX task/rewards; coherent full reset; nominal reset; no noise/push/domain randomization; full official command ranges; `ppo_g1_corrective`; clean checkout | seed 0; 262,144 requested, 286,720 actual | 655.77 s | Optimizer KL stabilized from `2.219` at 71,680 to `0.0390/0.0380/0.0384`, and return rose late from `-2.5898` to `-1.7536`. However, final evaluation length regressed from `68.91` to `61.75` steps and the termination term remained `-100` at every callback. C04 fails its predeclared length and behavior gates and will not be scaled unchanged. |
| C04 fixed-command/video diagnosis | checkpoint 286,720 versus checkpoint 0 and standing; eight commands; nominal reset seed 2000; 200-step (`4 s`) request; videos enabled | 24 episodes | 233.10 s | **Invalid trained inference:** normalization was omitted. Preserve the raw files, but do not use their tracking numbers or trained video. C04 remains a no-scale result because its valid training-side final length regressed to `61.75` and termination stayed `-100`. |
| C05a shipped-policy oracle implementation failure | `16bb816181247628788fb59c797a94f4f15226f3`; exact Playground-shipped `g1_policy.onnx`; first local CPU evaluator attempt | seed N/A (deterministic/no RNG); eight commands; intended 500 steps | 57.6 s external wall time | The diagnostic incorrectly treated the torso frame-z-axis sensor as projected gravity, inverted the termination test, and stopped every rollout after one step. No policy conclusion is drawn. The failed directory is preserved; C05b is a distinct corrected run. |
| C05b exact shipped ONNX oracle | same clean commit; exact Playground ONNX blob SHA-256 `db2eb258494c1297c43d2b9ffa94cdbde97654c2a44cbab0b40fd4b990752a5b`; ONNX Runtime `1.22.1`; MuJoCo CPU; corrected `upvector_torso` termination | seed N/A (deterministic/no RNG); eight fixed commands; 500 steps (`10 s`) each | 66.37 s recorded wall time | All eight rollouts completed 10 s, finite and upright, with minimum pelvis height `0.692–0.715 m`, `74.2–83.0%` single support, and roughly 30–48 transitions per foot. For command `vx=0.5`, mean `vx=0.580`; yaw commands `+/-0.5` produced mean `+0.487/-0.382 rad/s`. Lateral and combined tracking are imperfect, but the video visibly shows a stable alternating gait. This falsifies a broken-model/dynamics hypothesis and identifies insufficient/difficult from-scratch learning as the current problem. Its preserved JSON incorrectly retains the internal C05a experiment label; use the C05b directory and verified policy hash to identify it. |
| C06 forward-only training | `da70eda3ce353b9527b95dc8069e603b82c7a2f5`; exact Playground adapter/rewards; `vx=[0.2,0.6]`, `vy=yaw=0`, 10% zero; nominal/no noise/push/randomization; 512 envs; authoritative corrective PPO | seed 0; 5,000,000 requested / 5,007,360 actual | 1,272.85 s | Training eval return rose `-1.6033 -> 15.8929`, length `69 -> 500`, final KL was `.0946`, and termination contribution changed `-100 -> 0`. No training video was generated or claimed. This is valid balance acquisition, not by itself locomotion. |
| C06 first fixed evaluation | final checkpoint; eight commands; nominal seed 2000; videos | 24 episodes; 500-step request | 205.14 s | **Invalid trained inference:** observation normalization was omitted. Preserve `checkpoint_5007360_nominal_seed2000_500`, but do not cite its trained metrics or MP4s. |
| C06 corrected fixed evaluation | `f642bc2`; same checkpoint/config; observation normalization restored; eight commands; nominal seed 2000; no video | 24 episodes; 500-step request | 232.94 s | Stand and forward survived 500/500. Stand vector RMSE was `.0147`; forward was `.4999`, with 100% double support, 0% single support, zero transitions, min pelvis `.7561 m`, and success false. Backward survived 116 steps; lateral/yaw/combined only 3–27. C06 learned static stance, fails its predeclared gait gate, and cannot seed an omnidirectional stage. |

C02 did not pass its training-side scale-up rule, so the native profile was not
scaled unchanged. C03 passed its infrastructure-only gate. C04's valid
training-side final length regressed and its termination term remained `-100`,
so it likewise was not scaled unchanged. Their old fixed-command behavior must
not be cited because normalized inference was missing.

C05b proves that the exact model/observation/action loop supports a robust
policy and that the shipped policy uses the same 103-to-29 interface; the
upstream tuned PPO budget is 200 million steps, versus C04's 0.287 million.

C06 was predeclared as a staged gait-acquisition diagnostic rather than a full
command repeat: 5,000,000 requested (`5,007,360` actual) steps, seed 0,
500-step episodes, nominal/noiseless/no-push reset, and commands restricted to
forward `vx in [0.2, 0.6]` with `vy=yaw=0` (plus upstream's 10% zero command).
It retained the authoritative dynamics, reward scales, action semantics, and
PPO networks/optimizer. A warm-start omnidirectional stage was allowed only if
C06 was finite, final KL was below `0.2`, fixed `vx=0.5` survival reached at
least 400/500 steps, forward RMSE beat both checkpoint 0 and standing by at
least 20%, pelvis height remained above `0.6 m`, both feet transitioned, single
support was between 35% and 95%, and video showed an alternating gait.

C06 is complete and the gate is false. Corrected inference verifies balance
and height but shows static double support and zero transitions. No
omnidirectional warm-start is allowed. The next experiment is C07, a
zero-training-step audit: establish evaluator/training action parity, cover
autoreset bookkeeping, run the corrected full eight-command grid across seeds
2000-2002 with one valid video, and correctly reevaluate existing C02/C04
checkpoints. Only stand/forward are in-distribution for C06; because its reset
is nominal, the three seeds do not test randomized initial poses. A new PPO
configuration must not be frozen until those cheaper diagnostics identify one
specific gait-acquisition hypothesis.

C02 quantitative evidence and videos are under
`results/gate4_corrective/C02_nominal_noisefree_seed0_262144/evaluation/checkpoint_286720_nominal_seed2000_200/`.
The three `*_combined_seed2000.mp4` files decode successfully; review montages
at frames 0/10/20/35 are retained in its `frame_montages/` subdirectory.
C03 training and restore evidence is under
`results/gate4_corrective/C03_playground_ppo_integration_seed0_1024/`.
C04 evidence is under
`results/gate4_corrective/C04_playground_nominal_seed0_262144/`; C05a and C05b
are under their respective `results/gate4_corrective/C05*onnx_oracle/`
directories. The representative oracle video is `onnx_combined.mp4` in C05b.
C06's valid corrected JSON/CSV is under
`C06_playground_forward_curriculum_seed0_5000000/evaluation/checkpoint_5007360_normalized_seed2000_500/`.
Its sibling `...nominal_seed2000_500/` and all normalization-enabled PPO
evaluator outputs named above predate the fix. Their trained/checkpoint-0 policy
portions are forensic artifacts only; zero-action standing traces are unaffected.

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

The following command was historically run once per selected checkpoint using
reset seeds `2000,2001,2002`. The old outputs are invalid because the evaluator
then omitted normalized preprocessing; the current command is correct after
`f642bc2`, but faithful historical evaluation also requires the old transition
semantics in a temporary worktree based on `27de436`, the closest recoverable
snapshot rather than a per-run-proven exact commit:

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
| 4 - velocity baseline | **Not passed / held-out verdict invalidated** | Three matched 20,971,520-step native runs completed, but their fixed evaluation omitted observation normalization. C06 validly learned stance but no gait. No three-seed conventional baseline is verified. |
| 5-7 - HOMIE extensions | Not authorized / not started | Work stopped at Gate 4 as required |

## Environment and algorithm

- Robot: Unitree G1, `nq=36`, `nv=35`, `nu=29`.
- Model: MuJoCo Menagerie `unitree_g1/scene_mjx.xml` at commit
  `71f066ad0be9cd271f7ed58c030243ef157af9f4`.
- Behavioral reference: MuJoCo Playground G1 joystick at commit
  `8a4b4642d8eba8a80ac99ed125cb62c16e1457ad`.
- Runtime: both the preserved repository-native Brax `PipelineEnv` and the
  current thin exact-Playground JAX/MJX adapter; Brax PPO uses an asymmetric
  103-value actor observation and 216-value critic observation.
- Current diagnostic task: flat ground and commands `[v_x, v_y, yaw_rate]`.
  C06 used forward x `[0.2, 0.6]`, y/yaw zero; the older native family used x
  `[-0.5, 0.5]`, y `[-0.3, 0.3]`, yaw `[-0.5, 0.5]`.
- Action: 29 normalized values clipped to `[-1, 1]`, mapped to absolute
  position targets around the verified `knees_bent` pose, then clipped to the
  actual actuator ranges.
- Historical native matched training: 20,971,520 environment steps per seed;
  2,048 environments; details below. Current C06: one 5,007,360-step seed,
  512 environments, authoritative optimizer/network/reward settings, nominal
reset, and no noise/push/domain randomization.
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB) through WSL2 CUDA.
  Native Windows JAX remained CPU-only.
- Logging: local JSONL and local checkpoints; no W&B account, cloud service,
  Triton infrastructure, remote branch, or physical robot was used.

## Pre-D-012 fixed evaluations (faithful to unintended profiles)

Before runtime-control propagation, Brax actually trained with observation
normalization disabled. The old evaluator's identity preprocessing therefore
matches these policies, although the runs remain invalid for the *declared*
normalized experiment:

- `eval_smoke/`: alive-3 checkpoint 393,216, seed 1000, eight commands and
  three controllers for two steps; all finite, no video, runtime not recorded.
- `pilot_alive3_eval_checkpoint393216/`: same checkpoint/grid for 500 requested
  steps; trained/untrained/standing durations `.22/.22/.64 s`, linear RMSE
  `.790/.793/1.132`, yaw RMSE `1.376/1.380/.610`; all fell, no video, runtime
  not recorded.
- `pilot_balanced_eval_checkpoint491520/`: seed 1000, same 500-step grid;
  trained/untrained/standing durations all `.64 s`, linear RMSE
  `1.118/1.119/1.132`, yaw RMSE `1.248/1.195/.610`; all fell, no video, runtime
  not recorded.
- `baseline_seed0_2129920/evaluation/prelim_checkpoint1916928_seed1000/`:
  seed 1000, same grid; durations all `.64 s`, linear RMSE
  `1.122/1.120/1.130`, yaw RMSE `1.047/1.192/.608`; all fell, no video, runtime
  not recorded.

These are negative evidence for the actual normalization-disabled profiles and
are not comparable with D-012-and-later normalized runs.

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

## Historical frozen held-out evaluation (trained inference invalid)

Each policy seed was historically tested on the same eight commands and three
randomized reset seeds. There are 216 raw episode rows. The evaluator restored
normalizer parameters but failed to install `running_statistics.normalize` in
the network, so the trained and checkpoint-0 PPO actions do not represent the
saved policy under training inference. This section preserves the numbers that
were reported at the time; none of its PASS/FAIL cells is a valid current Gate
4 decision.

### Aggregate across policy seeds

| Metric | Trained | Untrained | Standing | Historical interpretation (invalid) |
|---|---:|---:|---:|---|
| Linear velocity vector RMSE (m/s) | **0.9254** | 0.9878 | 0.9942 | Was called pass; unusable for current policy comparison |
| Yaw-rate RMSE (rad/s) | **1.0575** | 0.4539 | 0.3238 | Was called fail; unusable for current policy comparison |
| Mean episode duration (s) | **1.3742** | 1.1364 | 1.1333 | Unusable for current policy comparison |
| Fall rate | **1.0000** | 1.0000 | 1.0000 | Unusable for current policy comparison |
| Episode success rate | 0.0000 | 0.0000 | 0.0000 | Unusable for current policy comparison |
| Finite rate | 1.0000 | 1.0000 | 1.0000 | Finite plumbing only |

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

These rows were originally interpreted as linear/duration improvement plus yaw
and survival failure. That interpretation is withdrawn. A faithful evaluation
must use normalized inference and the exact historical native environment
semantics; current transition timing is intentionally different.

### Additional diagnostics

Under the invalid inference function, the nominally trained controller used
much more effort and exhibited rougher motion than
the controls: mean actuator effort was `190.21` versus `81.94` untrained and
`75.37` standing; the mechanical-energy proxy was `398.90` versus `44.81` and
`35.81`; mean action-rate cost was `7.150` versus `0.037` and `0.000`. Trained
torso tilt was lower (`15.23 degrees` versus about `20.7/20.3`), but that did
not yield survival. Trained tracking-success fraction was only `0.0383`, below
both controls (`0.0469/0.0576`), and gait contact asymmetry was `0.3295` versus
`0.1274/0.1082`.

Those diagnostic traces describe the wrong policy function and cannot establish
generalization or gait. The strong late training-side returns and lengths remain
an unresolved reason to perform a faithful normalized reevaluation rather than
discard the checkpoints.

## Representative visual evidence

- Gate 1: `results/phase1_default_baseline_zero_action.mp4`.
- Gate 3 standing: `results/gate3/standing_zero_action.mp4`.
- Gate 3 bounded action: `results/gate3/bounded_action.mp4`.
- Historical Gate 4 comparison plots (invalid trained inference):
  `results/gate4/final_aggregate/heldout_comparison.png` and
  `results/gate4/final_aggregate/training_curves.png`.
- Historical Gate 4 representative rollout (invalid trained inference): seed-0 checkpoint `17,825,792`, combined
  command `[0.4, 0.2, 0.35]`, reset seed `2000`, for trained, matched untrained,
  and standing controllers under
  `results/gate4/baseline_corrective_seed0_20971520/evaluation/representative_checkpoint_17825792_combined_seed2000/`.

All three MP4s decode at 640x480 and 25 fps. The trained clip contains 38
frames (`1.52 s` encoded, 74 control steps/`1.48 s` evaluated); the two controls
contain 27 frames (`1.08 s` encoded, 53 control steps/`1.06 s` evaluated).
Visual inspection confirms that this **incorrectly reconstructed** trained
policy function leans and collapses; it does not establish the saved
checkpoint's behavior. Its `0.680/1.110` linear/yaw values and the comparison
with controls are preserved only for forensic history. Do not reuse these
MP4s as representative policy evidence.

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
| 10,485,760-step reference-scaled seed 0 | Valid training metrics showed alive contribution dominated tracking; its old held-out yaw/fall result used invalid unnormalized inference |
| Five-evaluation full launch | About 25 minutes exposed a compile-efficiency failure: 80 transition batches were compiled into one epoch. Contrary to earlier prose, no distinct artifact directory can be found; only the narrative survives. |
| First final seed-2 process | Externally interrupted at 10,485,760 steps; preserved and replaced from scratch because policy-only restore is not continuation-equivalent |
| Final matched family | Completed normally; its frozen fixed-evaluation verdict was invalidated by the normalization audit, so it remains unverified rather than passed |
| First representative video encoding | Failed for missing WSL system `ffmpeg`; preserved and rerun with a local bundled encoder |
| Exported-source PPO smoke | Passed training/checkpoint plumbing; KL was `6.11`. Its trained-policy fixed evaluation is unnormalized and invalid learning evidence. |
| C05a oracle | Inverted the torso up-vector termination check and stopped after one step; preserved separately from valid C05b |
| C06 forward curriculum | Learned 10-second static stance; correctly normalized forward trace has 100% double support and zero transitions, so it fails gait acquisition |

## Reproducibility verification

- 2026-09-15 handoff freeze: WSL `compileall` passed and the complete current
  suite reported **93 passed, 25 warnings in 616.44 s**, with zero skips,
  xfails, or failures. The warnings are existing JAX/Brax deprecation/runtime
  warnings.

- Historical working Windows and independent Gate 0 environments passed
  dependency checks and the then-current 58-test suite plus two timing xfails.
  Those timing xfails were subsequently fixed and converted to passing
  regression tests.
- Clean exported source tree: extracted outside the repository, pointed at the
  exact pinned Menagerie checkout through `MUJOCO_MENAGERIE_PATH`, and reported
  the then-current suite. It contained no `.git`, cache, virtual
  environment, generated result, log, checkpoint, or media directory.
- Clean-source training: the local 1,024-step PPO profile ran on `cuda:0`,
  emitted both progress callbacks and `run_end`, improved its integration-only
  evaluation reward from `-5.2194` to `-0.9420`, and wrote checkpoint 1,024.
- Clean-source evaluation loaded that checkpoint and wrote finite JSON/CSV, but
  its trained-policy behavior is invalid after discovering the missing
  observation-normalization preprocessing. Its Git-provenance handling remains
  valid plumbing evidence.
- Both ordinary and MJX G1 loader modes passed offline with the exact Menagerie
  pin. Compileall passed for `source`, `scripts`, and `tests`; the metadata
  inspector again reported 29 one-to-one joint/actuator mappings.
- The retained clean-source integration artifact is
  `results/gate4/clean_snapshot_ppo_smoke_20260909/`. It is separate from the
  final matched research family.

## Remaining limitations

1. The historical three-seed native family has never received a faithful
   fixed held-out evaluation. Reproducing it requires its exact old transition
   semantics plus the evaluator-normalization fix; today's native environment
   intentionally changed history/touchdown/reset timing.
2. C06 is one seed and one narrow forward curriculum. It balances but does not
   lift either foot or track forward velocity; no corrected C06 video exists.
3. C06's all-command aggregate includes axes outside its training distribution
   and is diagnostic only. Lateral/yaw/combined policies remain untrained.
4. No standardized push-recovery or domain-randomization evaluation was run;
   both were intentionally disabled for the conventional first baseline.
5. The largest adapter run is 5.0M steps/512 environments, far below the exact
   upstream 200M/8,192 profile. Compute may matter, but C06's static optimum
   must be diagnosed before scaling.
6. The current `full_reset=True` regression proves environment data/observation
   history coherence. It does not explicitly test every Brax EpisodeWrapper
   bookkeeping key; this is a coverage gap, not a confirmed cause of C06.
7. `source/scripts/play.py` and the training-time `--video` helper have not been
   audited for normalized-checkpoint preprocessing. Use the fixed
   `evaluate_g1.py --video` path for scientific evidence.
8. Results are simulation-only on one laptop GPU. There was no system
   identification, sim-to-real validation, physical G1 connection, or safety
   case.

## Deviation from the contract layout

The supplied papers were located directly under `research/references/` rather
than the preferred `research/references/papers/` directory. They were read from
their supplied paths and left in place so user-staged files were not moved or
duplicated.

## Recommendation

Do **not** proceed to HOMIE upper-body curriculum, height/knee rewards, or
symmetry yet. First complete C07's zero-training-step inference/autoreset audit,
correctly reevaluate the saved C06 policy across the declared seeds/command
grid with a valid video, and recover information from C02/C04. Quantify why the
C06 static stance
earns high reward versus the C05b gait, then change one gait-acquisition factor
in a bounded seed-0 run. Only a conventional three-seed family that passes
finite forward/lateral/yaw tracking, sustained survival, genuine alternating
support, exploit checks, and reproducible checkpoint evaluation can justify a
request to start Phases 5-7.
