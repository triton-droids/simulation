# GPT-6 Astra Successor Handoff: Unitree G1 Locomotion

## Active C17 fixed-phase diagnostic (2026-09-21)

C17 nominal gait gate and visual review PASS, exact 17-leaf restore. Both 500 steps, linear .125/.135, single support ~.79, completed air >=.28 s. However zero-yaw RMSE .414/.425 and video turning require correction. Active queue research/queues/c17_fixed_phase.json; status results/gate4_corrective/C17_fixed_phase_queue/status.json. It performs no training, only required 1.5-Hz seed-2000 matched oracle comparison. Review summary and traces before choosing intervention; independent final multi-seed/full-command Gate 4 is still outstanding.

## Active C17 gait queue (2026-09-21)

C16 PASSES balance but is static (all four 500, pelvis >=.753890; forward RMSE ~.5005, single support zero). Video reviewed. Active next queue research/queues/c17_gait.json; inspect results/gate4_corrective/C17_queue/status.json. C17 restores C16 final, trains 2,007,040 with established gait rewards, audits actor/normalizer restore and evaluates forward gait. Numeric pass still needs video and fixed-phase diagnostic before command broadening; no unchanged extension on failure.

## C16 evaluation recovery (2026-09-20)

Active replacement: research/queues/c16_evaluation_recovery.json; inspect results/gate4_corrective/C16_evaluation_recovery_queue/status.json. C16 completed 3,368,960 training steps but evaluation failed on comma-separated command labels. Recovery evaluates the saved final checkpoint only. Do not retrain; the original queue error is preserved.

Automation change 2026-09-20: user explicitly requested token-efficient local
orchestration and same-chat wakes. Read `research/QUEUE_WORKFLOW.md` and inspect
`results/gate4_corrective/C16_queue/status.json` before doing further work.
The runner/plan are being committed for launch. Do not duplicate an existing
queue or manually poll an active job. C15 evaluation is complete: trained
seeds 3000/3001 fall at 113/107 steps; video reviewed and fails upright gait.
C16 remains a balance-only prerequisite, not final success. A scheduled wake
should review terminal evidence and then autonomously freeze the next batch.

Immediate 2026-09-20: C15 fresh seed 7 completed 2,007,040 steps from clean
`cb4c604`, final training survival 108.8125. Final forward reset 3000 falls
at 113 steps, RMSE 1.513; both its gait and extension conditions fail.
`C15_forward_gate/` is completing controls/video; inspect before relaunch.
C16 is now frozen in EXPERIMENT_PLAN: fresh seed 7, original upstream rewards
1/.75/1/contact0, forward-only nominal task, 3,368,960 steps for balance
acquisition under the corrected wrapper. Inspect C16 artifacts/processes
before launching. C14 remains the successful robust-gait reference; final
independent acquisition and multi-seed Gate 4 remain outstanding.

Latest 2026-09-20: C14 PASSES its full development gate: all 8 nominal and
16 randomized episodes survive 500, mean errors nominal .159320/.105013 and
randomized .207868/.140616; worst randomized errors .341152/.338696, minimum
pelvis .678900. Nominal/randomized combined videos show real upright gait.
Final multi-seed Gate 4 is still outstanding. C15 is now frozen in the
leading EXPERIMENT_PLAN section: fresh seed 7, no restore, forward-only,
rewards 3/9/3/2, nominal reset, 2,007,040 steps, to validate independent
acquisition with a shorter recipe. Inspect live processes/artifacts before
launching. Do not retrain C14 unchanged or claim final success prematurely.

Immediate 2026-09-20: C14 completed normally at final 2007040, directory
`results/gate4_corrective/C14_randomreset_seed0_2007040`, clean training
`2a2b07b`, 942.87 s, final training survival 388.125 and KL .124018. Exact
checkpoint-zero actor/normalizer restore verified. Final nominal assessment
is running/written in `C14_nominal_seed4000/`, log in `C07_audit/`; inspect
process/output before relaunch. Randomized 5000/5001 assessment still needed.
An optional batching prototype failed real-physics parity despite passing
unit tests; production evaluator stays serial. Preserve failed audit in
`C14_batch_physics_audit/`. Do not select intermediate best 1146880 before
the frozen final-checkpoint assessment. No C15 is yet declared or launched.

Latest 2026-09-20: C13 (`C13_yaw9_seed0_1003520`, clean `0831d4f`) passes its
full nominal eight-command gate at final 1003520: all 500 steps, mean linear/
yaw .153616/.130862, forward support 85.2%, air .32/.32 s, combined video
shows real alternating gait and turning. However randomized development
seed 5000 fails four of eight commands (107/111/119/95 steps), mean errors
.630911/.455851. Do not claim robust Gate 4. Same-MJX oracle comparison from
that exact randomized forward start is next, in
`results/gate4_corrective/C13_randomized_oracle_seed5000/`. Inspect actual
logs/processes before relaunching. That diagnostic is complete: identical
initial states, oracle survives 500 while C13 falls at 112. C14 is now frozen
in EXPERIMENT_PLAN: only enable randomized reset, 2,007,040 steps from C13,
then nominal retention plus randomized seeds 5000/5001. Inspect C14 artifacts
and live processes before any launch; do not duplicate a running experiment.
Older entries below are history; preserve the failed and successful evidence.

Latest assessment 2026-09-19: C12 completed 2,007,040 steps and survives all
eight nominal development commands for 500 steps. Mean linear/yaw errors
.140362/.270396; yaw improves only 12.91% versus initial, failing the frozen
20% extension prerequisite. **Do not extend C12 unchanged.** C11's real
forward gait is retained (C12 forward single support 84%, air .34/.32 s),
but stand/turn yaw remain poor. Same-MJX turning diagnostic finds C12 mean
yaw .206 for .5 commanded, versus oracle .551; current reward slightly
favors C12. C13 changes only angular tracking weight 2.25 -> 9 after saved
trace scoring reverses that preference. Its 1,003,520-step pilot is frozen
in EXPERIMENT_PLAN. Three final seeds and Gate 4 remain pending.
See the leading RESULTS and decision D-037; earlier updates below are history.

Current update 2026-09-19: C11 passes its complete forward gait gate (both
reset seeds, visual alternation, fixed phase and sustained air intervals).
C12 is predeclared in EXPERIMENT_PLAN: first audit an explicit variance prior
for unseen lateral/yaw inputs, then a bounded broader-command curriculum.
Inspect active processes/generated C12 directories before launching. The
earlier status below is history; full three-seed Gate 4 remains outstanding.

Current status, 2026-09-16: C07 audit and checkpoint recovery are complete;
autoreset metadata was fixed and normalized saved inference verified. C08
from scratch failed (71 steps). C09, warm-started from local C06 balance,
acquired movement but failed the support gate. C10 increased feet_phase to 3
and now passes numerical forward checks on seeds 3000/3001 and fixed 1.5 Hz:
500 steps each, RMSE .184/.134/.134, single support 38.2%/35.6%/35.2%.
Its short irregular lifts (median air .06 s) and yaw drift are unresolved;
do not call Gate 4 passed or broaden commands yet.

C11 is predeclared in `research/EXPERIMENT_PLAN.md` and D-035: an optional
local phase-aligned contact reward, weight 2, disabled by default, from C10
for 1,003,520 forward-only steps. Tests must pass first. Inspect generated
C11 artifacts/processes before launching anything; never duplicate an active
run. Final checkpoint first, two development reset seeds, fixed-phase
survival and per-foot median completed air intervals >=.12 s. This is neither
oracle imitation nor HOMIE. The three final policy seeds and genuinely new
held-out command/reset evidence remain outstanding. No remote push occurred.
The freeze below is historical; leading research sections hold current evidence.

Freeze date: 2026-09-15 (America/Los_Angeles)

Branch: `Robocup`

Audited code/research base: `f642bc2fbdbc575dc3d4865cdd2fc72a941029e2`
Handoff commit: the current `Robocup` tip whose subject is
`Freeze Astra G1 successor handoff`; run `git rev-parse HEAD` for its exact
hash. A Git commit cannot contain its own hash without changing that hash.

## 1. Purpose

Triton Droids requested a short, modular investigation of Unitree G1 humanoid
locomotion: start with a conventional velocity-commanded reinforcement-learning
baseline, use MuJoCo Playground as the primary behavioral reference, and use
HOMIE only as a reward/methodology reference. The work must remain easy to stop
or redirect rather than becoming a robotics-platform rewrite.

The current goal is still to obtain a robust standard G1 policy that walks,
stays upright, and tracks forward velocity, lateral velocity, and yaw rate on
held-out commands across at least three training seeds. No such local policy
has been verified yet. HOMIE-specific Phases 5-7 are out of scope until the
ordinary baseline genuinely passes; do not begin them automatically.

## 2. Current repository state

### Git

- Branch: `Robocup`, tracking `origin/Robocup`.
- Audited pre-handoff HEAD: `f642bc2fbdbc575dc3d4865cdd2fc72a941029e2`
  (`Restore PPO observation normalization in G1 evaluation`).
- At the start of this freeze the tracked worktree was clean and the branch was
  11 local commits ahead. The final handoff commit adds this file, reconciles
  the research documentation, and tracks the ONNX-oracle CLI. Nothing was
  pushed.
- Final handoff snapshot after the single planned commit: `Robocup` is 12 local
  commits ahead of `origin/Robocup`, with no tracked or untracked changes and
  only the generated paths listed below ignored by `.gitignore`. Verify with
  `git status --short --branch` and `git status --ignored --short`.
- Freeze verification: WSL `compileall` passed and the complete suite reported
  `93 passed, 25 warnings in 616.44s`; no test was skipped, xfailed, or failed.
- Twenty-seven important documentation/source/checkpoint/video paths listed
  below were checked with `Test-Path`; all existed. Both cache revisions and
  the ONNX SHA-256 were re-read and exactly matched the declared pins.
- Do not discard ignored experiment directories: they contain the only local
  checkpoints, metrics, and videos. They are not in Git or the normal export.

### Runtime actually validated

| Surface | Exact value |
|---|---|
| Windows host | Windows 11 `10.0.26200`; Python 3.12.7 (Anaconda build) |
| WSL training host | WSL2 Ubuntu, Linux `6.6.87.2-microsoft-standard-WSL2-x86_64`; Python 3.12.3 |
| JAX / jaxlib | `0.11.0 / 0.11.0` |
| Brax | `0.14.2` |
| MuJoCo | `3.10.0` |
| Hydra / OmegaConf | `1.3.4 / 2.3.1` |
| Flax / Orbax | `0.12.8 / 0.12.1` |
| MediaPy / imageio-ffmpeg | `1.2.7 / 0.6.0` |
| Optional oracle runtime | ONNX Runtime `1.22.1` in WSL only |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB, driver `572.61` |
| JAX devices | native Windows: CPU; WSL venv: `[CudaDevice(id=0)]` |
| WSL environment | ignored `.cache/g1_wsl_venv` (about 6.6 GB) |

Useful long training runs were executed only through WSL CUDA. Basic loading,
tests, logging, and CPU diagnostics require no W&B account or cloud service.

### Authoritative pins

- MuJoCo Menagerie Unitree G1:
  `71f066ad0be9cd271f7ed58c030243ef157af9f4`; training scene
  `unitree_g1/scene_mjx.xml`; ignored cache `.cache/mujoco_menagerie`.
- MuJoCo Playground G1 joystick:
  `8a4b4642d8eba8a80ac99ed125cb62c16e1457ad`; ignored sparse checkout
  `.cache/mujoco_playground`.
- Playground-shipped ONNX policy SHA-256:
  `db2eb258494c1297c43d2b9ffa94cdbde97654c2a44cbab0b40fd4b990752a5b`.
- The local Playground adapter deliberately points at the repository's one
  Menagerie resolver. There is no second tracked G1 asset tree.

## 3. What has been built

- A separate native G1 robot/environment under
  `source/robots/unitree_g1` and `source/locomotion/unitree_g1`. It introspects
  the real 36-qpos/35-qvel/29-actuator model instead of generalizing the old
  12-actuator arrays.
- A separately registered thin adapter, `unitree_g1_playground`, which loads
  the exact pinned Playground G1 task/feet-only MJX scene while reusing the
  pinned Menagerie checkout. The original native implementation remains for
  comparison.
- Hydra configurations for native and Playground G1 plus historical and
  corrective PPO profiles. PPO supports asymmetric actor/critic dictionaries
  (`state=103`, `privileged_state=216`).
- Local-first JSONL logging, timestamped run manifests, exact source/config
  records, compact inference checkpoints, full checkpoint trees, and explicit
  parameter warm-start semantics. W&B is optional.
- `source/scripts/evaluate_g1.py`: deterministic fixed-command evaluation of a
  trained checkpoint, its matched checkpoint-0 policy, and a zero-action
  standing controller. It records tracking, survival, height/tilt, action and
  target saturation, effort/power, contacts, support phases, transition cadence,
  terminal causes, JSON, CSV, trajectories, and optional MP4s.
- `source/scripts/evaluate_playground_onnx.py`: reproducible local execution of
  the exact shipped ONNX policy as a behavioral oracle. This is not a training
  result.
- Deterministic G1 reset/step/1,000-step smoke paths and video fallbacks that do
  not require a system `ffmpeg`.
- Regression tests protecting `source/locomotion/default_humanoid_legs`; it is
  still a working, separate 12-actuator baseline.
- Research provenance in `research/SOURCE_LEDGER.md`, decision records in
  `research/DECISIONS.md`, experiment declarations/results, clean-export logic,
  and tests for the export exclusions.

## 4. Experiment history

Exact resolved configs and source/runtime manifests live inside each ignored
run directory. “Invalid fixed evaluation” below means a checkpoint trained with
normalization enabled was evaluated without Brax's
`running_statistics.normalize` callback. Checkpoint bytes and training-side PPO
metrics remain valid; the reported trained rollout behavior, tracking numbers,
and videos do not. Before D-012, the trainer failed to forward the flag and Brax
actually trained with normalization disabled, so those early identity
evaluations are faithful to their actual but unintended profiles.

### Historical native Gate-4 campaign

Revision caveat applying to every row in this historical table: surviving
evaluator records report HEAD
`6640663e5b9a50f25264e392a4c18703ca2c00e7` with `dirty=true`, while the older
training directories predate per-run Git manifests. The evolving dirty source
diff was not archived separately, so an exact source commit is **not recorded**
for these runs; `6640663...` is only the recorded base HEAD. The clean-export
smoke intentionally reports Git unavailable. This is a provenance defect, not
a reason to invent a revision.

| Experiment | Hypothesis and key configuration | Seed / steps / runtime | Quantitative and visual outcome | Conclusion, artifact, comparability |
|---|---|---|---|---|
| PPO smoke/retries | Native full ranges, randomized reset/noise, 16 envs, 64-step episodes, unroll 4, batch/minibatches 4/4, one update, 32x32 nets; exercise PPO/checkpoints | seed 42; 1,024 target; failed-attempt runtimes not recorded | Initial/retries found removed JAX pmap helpers, checkpoint-tree mismatch, and reset-axis mismatch. Completed CPU retry: reward `.00234 -> .00301`, length `5.75 -> 4`, KL `.341`, 60.85 s. GPU: `.00244 -> .00316`, length `5.75 -> 4`, KL `.614`, 170.78 s. No video. | Infrastructure only; `ppo_smoke`, `ppo_smoke_retry{1..4}`, `ppo_smoke_gpu`; not learning evidence |
| Capacity probe, clamped | Native full network, 128 envs, inherited nonnegative total reward | seed 42; 65,536; 286.53 s | Reward `.0657 -> .0200`, length `22.875 -> 19.5`, KL `4.848`; proved clamp erased `-100` termination | `capacity_probe_128`; direct reward-semantics ablation only |
| Capacity probe, signed | Same capacity question with signed total, 512 envs | seed 42; 65,536; 285.16 s | Reward `-2.090 -> -2.188`, length `19.125 -> 13.125`, KL `.239`, about 831 steps/s | `capacity_probe_signed_512`; no locomotion |
| Signed-reward pilot | No alive reward, LR `1e-4`, five updates, 512 envs | seed 0; 983,040; 811.96 s | Reward improved `-2.323 -> -2.002` while length collapsed from `22.875` to about 3; final KL `.106` | Explicit early-termination exploit; `pilot_signed_seed0_983040` |
| Standing probe | Native zero action, deterministic | seed 0; 1,000 control steps; 84.43 s | Finite; first termination 67; final pelvis `.13294 m`; no video claim | `standing_probe`; controller diagnostic only |
| Alive-1 / alive-3 pilots | 512 envs, LR `5e-5`, two updates; survival shaping 1 then 3 | seed 0; 491,520 each; 634.94 / 613.21 s | Alive-1 reward `-1.704 -> -1.811`, length `19.25 -> 14.625`, KL `.0313`; alive-3 `-.934 -> -.882`, length `19.25 -> 20`, KL `.0332`. | `pilot_alive_seed0_491520`, `pilot_alive3_seed0_491520`; survival shaping insufficient |
| Alive-3 fixed evaluator smoke/full | Checkpoint 393,216 vs checkpoint 0/standing; randomized reset seed 1000; no observation noise; eight commands; actual training/eval normalization disabled | seed 1000; 2 then 500 requested steps; runtime not recorded | `eval_smoke` ran 0.04 s/episode finite with no video. Full evaluation: trained/untrained/standing duration `.22/.22/.64 s`, linear RMSE `.790/.793/1.132`, yaw RMSE `1.376/1.380/.610`; all fell; no video. | `eval_smoke`, `pilot_alive3_eval_checkpoint393216`; faithful to the actual unintended unnormalized profile, not comparable to D-012-and-later runs |
| Contact/cost probes | Make hand-thigh penalized not terminal; reduce dominant auxiliary scales; alive 3 | seed 0; 1,024 / 65,536 / 491,520; 146.24 / 301.09 / 639.12 s | Smoke length 42.5; tuning length `49.875 -> 42.375`; balanced pilot reward `.917 -> 1.403`, length `49.875 -> 57.875`, KL `.0350`. | `ppo_smoke_contactfix`, `tuning_probe_balanced_65536`, `pilot_balanced_seed0_491520`; pre-runtime-control family |
| Balanced-pilot fixed evaluation | Checkpoint 491,520 vs checkpoint 0/standing; randomized reset seed 1000; no noise; eight commands; actual normalization disabled | seed 1000; 500 requested steps; runtime not recorded | All fell after `.64 s`; trained/untrained/standing linear RMSE `1.118/1.119/1.132`, yaw `1.248/1.195/.610`; no video. | `pilot_balanced_eval_checkpoint491520`; faithful negative result for the actual unnormalized profile |
| Initial nominal family | 512 envs, 2,129,920 steps, LR `5e-5`, two updates, 11 evals, full ranges, random reset/noise, alive 3 | seeds 0/1/2; 22.80 / 18.01 / 18.08 min | Best reward/length: `1.6496/58.25`, `2.3442/67.5`, `1.4797/54.75`; max KL `.1505/.5393/9.2376`. Serialized normalization, clipping, and action repeat were not forwarded. | Invalid intended training family; `baseline_seed{0,1,2}_2129920`; internally comparable as the same actual unnormalized profile |
| Initial-family preliminary fixed evaluation | Seed-0 checkpoint 1,916,928 vs checkpoint 0/standing; randomized reset seed 1000; eight commands; actual normalization disabled | seed 1000; 500 requested steps; runtime not recorded | All fell after `.64 s`; trained/untrained/standing linear RMSE `1.122/1.120/1.130`, yaw `1.047/1.192/.608`; no video. | `baseline_seed0_2129920/evaluation/prelim_checkpoint1916928_seed1000`; faithful to the misconfigured family, not evidence for the declared normalized profile |
| Runtime-control smoke/probe | Actually forward normalization, max-grad 1, action repeat 1 | seed 0; 1,024 / 491,520; 162.35 / 591.02 s | Smoke reward `-.369 -> .242`, length `41.5 -> 45`, KL `4.632`; probe reward `1.006 -> 2.0245`, length `51.25 -> 64.75`, KL `.00337` | `ppo_smoke_runtime_controls`, `probe_runtime_controls_seed0_491520`; supports propagation fix |
| Corrected 2.13M native | Balanced profile with controls active | seed 0; 2,129,920; 988.21 s | Best reward `2.8531` at 1,916,928, length `67.25`; final reward `1.2201`, length 48. Old fixed eval invalid; standing traces still helped calibrate height cutoff to `.25 m`. | `baseline_corrected_seed0_2129920`; post-runtime-control family |
| 2,048-env capacity/height | Min pelvis `.25`, batch 64, 32 minibatches, five updates, LR `2.5e-5` | seed 0; 327,680; 547.96 s | Best reward `1.3686` at 196,608, length `55.75`, KL `.00536`; >5 GiB GPU headroom, about 7.7k steps/s. Old fixed eval invalid. | `probe_2048env_seed0_327680`; capacity only |
| Reference-scaled native | 2,048 envs, 11 evals, five updates, LR `2.5e-5`, alive 3, full ranges | seed 0; 10,485,760; 45.94 min | Valid training reward `.7687 -> 5.9996`, length `51.19 -> 114.16`, KL `.00919`; alive contribution about 5.2x trackers. Old 100%-fall/yaw `1.971` fixed eval is invalid. | `final_seed0_10485760`; retain training diagnostics only |
| Contact/reward probe | Support/cross geom fix; alive `.5`; tracking `2/1.5`; narrowed `+/-0.5,+/-0.3,+/-0.5`; LR `1e-4` | seed 0; 2,097,152; 786.32 s logged / 916.5 s external | At 1,572,864: length `50.34 -> 53.38`, return `-1.470 -> -.516`, linear/yaw accumulation `28.63/27.43 -> 34.55/43.91`, KL `.0131` | `reward_probe_seed0_2097152`; passed training-side go/no-go |
| Five-evaluation compile failure | Intended 20.97M corrective run but 80 transition batches compiled into one epoch | seed 0; no callback; about 25 min | Stopped before step 0. Later changed to 21 evals/16 batches per epoch. | Narrative only: freeze audit found **no distinct artifact directory**, despite older prose claiming preservation |
| Interrupted seed-2 | Test recoverability of stopped full run | seed 2; 10,485,760; 44.00 min | Reward `.8972`, length `88.66`; checkpoint durable, but compact restore omits optimizer/PRNG/env stream/step | Excluded and preserved at `baseline_corrective_seed2_20971520_interrupted_at_10485760`; replacement trained from scratch |
| Frozen matched native family | 2,048 envs, 20,971,520 steps, 21 evals, LR `1e-4`, five updates | seeds 0/1/2; 86.42 / 85.02 / 79.11 min | Selected steps 17,825,792 / 19,922,944 / 20,971,520. Training return `32.873/31.712/35.200`, length `629.97/594.78/696.50`, KL `.0311/.0325/.0374`. Published 216-episode eval/videos used incorrect unnormalized inference. | Gate 4 **not passed**, but prior definitive failure invalid. `baseline_corrective_seed{0,1,2}_20971520`, `final_aggregate`; faithful reevaluation needs historical semantics + normalization |
| Clean-export smoke | Source-only archive with no Git metadata; train/checkpoint/evaluate | seed 314; 1,024; 147.35 s | Training reward `-5.2194 -> -.9420`, length `42 -> 51.5`, KL `6.114`; checkpoint valid, old fixed behavior invalid | `clean_snapshot_ppo_smoke_20260909`; reproducibility plumbing only |

### Corrective and authoritative campaign

| ID | Hypothesis and exact configuration | Seed / steps / runtime | Quantitative and visual outcome | Conclusion, artifact, comparability |
|---|---|---|---|---|
| C00/C00b/C00c | Revisions `4426499` dirty for C00/C00b and clean `3f437e3f711d0478f036d3e5656fcc8df81f1ba6` for C00c; correct native timing/state invariants, then deterministic sinusoidal scan (`0.02` amplitude) | seed 1707; 1,000 control steps each; 64.85 / 43.41 / 41.19 s | Finite; untrained controller terminated at step 74, pelvis `0.1323 m`. C00/C00b looked dirty only because WSL interpreted CRLF; C00c records a clean tree. | Pipeline pass only; `results/gate4_corrective/C00*`; new semantics, not comparable with old Gate 4 |
| C01 | Clean `3f437e3f711d0478f036d3e5656fcc8df81f1ba6`; native PPO plumbing, randomized reset/noise, 16 envs, 32x32 nets | seed 0; 1,024; 280.03 s | Checkpoints restore; training return `-4.8356 -> -2.8073`, length `42.25 -> 43.75`, KL `3.3128`. Its 24-episode fixed evaluation is unnormalized and invalid. | Infrastructure only; `C01_ppo_integration_seed0_1024` |
| C02 | Clean `8a7fce5c0635a602d9b2f1af2b48bf49f0b7c475`; corrected native learning diagnostic, nominal/noiseless, 512 envs, authoritative PPO widths/optimizer | seed 0; 262,144 requested / 286,720 actual; 805.62 s | Training return `.3319 -> .9259`, length `72.0 -> 75.41`, final KL `.03973`. The old 322.44 s rollout/video diagnosis is unnormalized and cannot establish behavior. | Did not justify scaling on training length; `C02_nominal_noisefree_seed0_262144`; normalized reevaluation is still possible under current native semantics |
| C03 | Clean `03794010c6d994e5439c203febc16bae8497bfdf`; exact pinned Playground adapter PPO/checkpoint smoke; nominal/no noise/push/randomization | seed 0; 1,024; 505.64 s | Both checkpoints written; return `-.6917 -> -2.4834`, length `64 -> 63.75`, KL `3.2936`. The old held evaluation is unnormalized and invalid. | Adapter infrastructure only; `C03_playground_ppo_integration_seed0_1024` |
| C04 | Clean `16bb816181247628788fb59c797a94f4f15226f3`; full authoritative G1 commands/rewards, nominal/noiseless, 512 envs, corrective PPO | seed 0; 262,144 requested / 286,720 actual; 655.77 s | KL settled near `.038`; return `-2.5898 -> -1.7536`; final training-eval length fell to `61.75` and termination stayed `-100`. Old fixed videos/numbers are unnormalized and invalid. | Training metrics reject scaling unchanged; `C04_playground_nominal_seed0_262144`; comparable with C03 only as adapter diagnostics |
| C05a | Clean `16bb816181247628788fb59c797a94f4f15226f3`; first exact shipped-ONNX oracle attempt | seed N/A (deterministic/no RNG); eight commands; intended 500 each; 5.85 s recorded / about 57.6 s external | Incorrectly inverted the up-vector termination convention and stopped every rollout at one step | Preserved evaluator failure; `C05_exact_playground_shipped_onnx_oracle`; no policy conclusion |
| C05b | Clean `16bb816181247628788fb59c797a94f4f15226f3`; corrected exact shipped ONNX policy, same compiled model and raw 103-to-29 loop, CPU MuJoCo | seed N/A (deterministic/no RNG); eight commands; 500 each; 66.37 s | All eight survived 10 s; min pelvis `.692-.715 m`; single support `74.2-83.0%`; 30-48 transitions/foot; visibly alternating gait | Valid behavioral oracle, not a local training seed. `C05b_exact_playground_shipped_onnx_oracle`; its preserved JSON has the stale internal label `C05_exact_playground_shipped_onnx_oracle`, so distinguish it by directory/hash; not numerically comparable to PPO controls |
| C06 training | Clean `da70eda3ce353b9527b95dc8069e603b82c7a2f5`; forward-only gait acquisition: `vx=[.2,.6]`, `vy=yaw=0`, upstream 10% zero; nominal/no noise/push/domain randomization; authoritative rewards; 512 envs, batch 32, 16 minibatches, four updates, LR `3e-4`, entropy `.005`, 512/256/128 networks, 500-step episodes | seed 0; 5,000,000 requested / 5,007,360 actual; 1,272.85 s | Training eval return `-1.603 -> 15.893`, length `69 -> 500`, final KL `.0946`, termination term `-100 -> 0`; no training video was generated or claimed | Valid training-side evidence of learned balance. `C06_playground_forward_curriculum_seed0_5000000`; narrow curriculum, so OOD axes are diagnostic only |
| C06 first fixed evaluation | Clean `da70eda3ce353b9527b95dc8069e603b82c7a2f5`; same final checkpoint, eight commands, reset seed 2000, nominal reset, videos | 24 controller episodes; 500 requested; recorded 205.14 s | Evaluator omitted observation normalization; reported trained falls and videos reflect the wrong policy function | **Invalid and preserved** at `evaluation/checkpoint_5007360_nominal_seed2000_500`; do not cite its trained behavior or MP4s |
| C06 corrected fixed evaluation | Commit `f642bc2`; same checkpoint/config; normalization restored; eight commands, reset seed 2000, no video | 24 episodes; 500 requested; 232.94 s | Trained stand and forward both survived 500/500. Stand RMSE `.0147`; forward RMSE `.4999`, 100% double support, zero transitions, success false. Backward lasted 116; lateral/yaw/combined lasted 3-27. | C06 produced a 10-second nominal static stance for stand/forward at reset seed 2000, not gait, so its predeclared gait gate fails and omnidirectional warm-start is forbidden. Valid artifact: `evaluation/checkpoint_5007360_normalized_seed2000_500` |

## 5. Most important current fact: C05b

The exact policy at
`.cache/mujoco_playground/mujoco_playground/experimental/sim2sim/onnx/g1_policy.onnx`
has the verified SHA-256 above, accepts one 103-value observation, and returns
29 position-offset actions. C05b drove the same adapter-compiled G1 model with
the same 20 ms control interval, 0.5 action scale, default pose, sensor frames,
phase, and action history expected by the local interface.

It completed every one of the eight fixed 10-second rollouts without terminal
failure. The forward `0.5 m/s` command produced mean `vx=0.580 m/s` and vector
RMSE `0.194 m/s`. Yaw `+0.5/-0.5 rad/s` produced mean `+0.487/-0.382 rad/s`.
The lateral and combined tracking is imperfect, but all rollouts remained
upright, pelvis height never fell below `0.692 m`, single support was
`74.2-83.0%`, and the video `onnx_combined.mp4` visibly shows genuine alternating
steps.

This proves that the physical model, actuator ordering, default pose, sensors,
action scale, and basic control interface can support stable G1 walking. It
falsifies “the G1 model is fundamentally broken,” “the Menagerie revision
cannot walk,” and “the 103-to-29 interface is incapable of gait” as primary
explanations. The remaining bottleneck is local gait acquisition/training (or
a still-unisolated difference in training semantics), not basic feasibility.

## 6. Bugs already found and fixed

The protecting tests are named so they can be run directly with `pytest -k`.

| Bug / correction | Regression protection |
|---|---|
| Default humanoid privileged observation was configured as 88 but built as 112 | `test_model_and_robot_specific_shapes_are_preserved` |
| Default scheduled push discarded immutable replacement state | `test_scheduled_push_changes_planar_velocity` |
| Manual smoke read nonexistent top-level state fields | `test_manual_smoke_reads_pipeline_state_fields` plus the live reset/step coverage in `test_seeded_reset_and_zero_action_step_are_finite_and_shaped` |
| Clean setup omitted the pytest runner dependency | `test_pytest_runner_is_declared_for_clean_install` |
| Local logger unnecessarily implied cloud credentials | `test_local_logger_writes_jsonl_without_importing_wandb`, `test_disabled_logger_needs_no_files_or_cloud_package` |
| Brax called JAX pmap helpers removed in JAX 0.11 | `test_brax_replication_helper_is_available_and_preserves_leading_device_axis` |
| Brax's non-pmap reset path collapsed the device/environment reset-key axes | `test_brax_pmap_reset_axis_workaround_remains_enabled`; all supported PPO profiles inherit `use_pmap_on_reset=True` |
| Compact checkpoint saver assumed the wrong Brax parameter tree | all four tests in `tests/test_checkpoints.py` |
| PPO serialized normalization, max-grad norm, and action repeat but did not propagate them | `test_repository_runtime_controls_override_brax_defaults`, `test_invalid_runtime_controls_are_rejected` |
| G1 reward total was clamped nonnegative, erasing termination penalties | `test_signed_aggregation_does_not_erase_termination_penalty` |
| Support capsules and cross-contact foot boxes were conflated; hand-thigh collision was misclassified terminal | `test_joint_actuator_and_contact_metadata_are_exact`, `test_contact_classes_match_playground_semantics_and_ground_safety` |
| Native returned observation used an extra-step-stale action and stale command at resampling | `test_step_observation_contains_just_applied_previous_action`, `test_resampled_command_matches_returned_observation` |
| Transition reward could use the resampled next command rather than the command that governed the action | `test_transition_reward_uses_command_that_governed_action` |
| Returned phase lagged next-state phase | `test_returned_observation_contains_advanced_phase` |
| Touchdown air time was cleared before its reward was computed | `test_touchdown_reward_uses_air_time_before_contact_reset` |
| Reset claimed pose-hold targets while sending zero control, creating about 380 summed absolute actuator force | `test_reset_initializes_pd_targets_to_the_actual_joint_pose` |
| Step mutated the input state's Python `info`/`metrics` dictionaries | `test_step_does_not_mutate_input_state_history`, `test_adapter_step_is_functional_clips_action_and_repairs_upstream_staleness` |
| Termination checked NaN but not all infinities/qvel invalidity | `test_termination_catches_height_contact_and_nonfinite_state` |
| Native foot velocity used the ankle body origin rather than the foot site | `test_foot_site_velocity_includes_rigid_body_angular_motion` |
| G1 domain randomization could crash late or pretend to jitter a reset pose it did not control | `test_g1_rejects_unimplemented_domain_randomization` (explicit early rejection) |
| Playground adapter inherited stale action/command/phase/air-time observation slices | `test_transition_observation_uses_returned_command_action_phase_and_air_time`, `test_adapter_step_is_functional_clips_action_and_repairs_upstream_staleness` |
| Fast autoreset restored data/obs while leaving environment history stale | `test_playground_training_auto_reset_restores_matching_history`; current adapter uses `full_reset=True` |
| Evaluator held-command resampling could change the command behind the policy/reward | `test_held_command_is_used_for_reward_and_restored_after_resampling` |
| Evaluator could label terminal-on-last-step as success | `test_terminal_on_final_step_is_not_labeled_successful` |
| Evaluator action-rate used raw rather than applied/clipped action and conflated target limits | `test_action_diagnostics_use_applied_action_and_separate_target_limits`, `test_action_rate_uses_clipped_actions_not_raw_policy_outputs` |
| Evaluator could not distinguish planted/hopping/fall transitions or collision classes | `test_contact_breakdown_separates_nonfoot_ground_and_self_collision`, `test_contact_breakdown_does_not_label_clean_foot_support_as_collision`, `test_gait_summary_reports_support_and_contact_transitions` |
| Evaluator reconstructed normalized PPO checkpoints with identity preprocessing | `test_evaluation_network_reuses_training_observation_normalization` and disabled-normalization companion; fixed in `f642bc2` |
| The first shipped-policy oracle inverted Playground's torso up-vector sign and treated upright as terminal | `test_oracle_termination_uses_positive_upvector_as_upright`; the failed C05a artifact remains preserved |
| WSL video failed when no system `ffmpeg` existed | `test_video_falls_back_to_opencv_when_ffmpeg_is_missing`; bundled imageio-ffmpeg is preferred first |
| Windows export traversed excluded WSL symlink/stat failures and could include generated `outputs/`/`runs/` | `test_export_skips_excluded_tree_before_file_stat`, `test_export_excludes_generated_output_directories` |

Known but not yet closed: the `full_reset=True` wrapper test proves environment
history coherence but does not explicitly assert preservation of Brax
`EpisodeWrapper` terminal bookkeeping. Add that cheap behavioral test before
another PPO run. Also avoid `source/scripts/play.py` and training-time
`--video` for scientific evidence until their normalized-checkpoint inference
path is audited; `evaluate_g1.py --video` is the validated path.

D-013's `0.45 -> 0.25 m` native pelvis cutoff was an explicitly measured
experimental calibration, not a correctness bug; the termination test follows
the configured threshold rather than pinning that research choice forever.

## 7. Current open question

Why does the authoritative shipped policy walk while local from-scratch PPO has
only learned balance/static stance so far?

C06 narrows the question. Its final network is finite, stable for 10 seconds,
and command-conditioned enough to become unstable on out-of-distribution axes,
but for `vx=0.5` it produces effectively zero forward velocity, keeps both feet
down for 100% of the rollout, and makes zero contact transitions. Plausible
causes, not yet distinguished, are:

1. the five-million-step/512-environment run is still far below the official
   200-million-step/8,192-environment optimization regime;
2. the narrow forward reward has an attractive static local optimum (zero
   velocity still earns nonzero exponential tracking reward plus perfect yaw
   and phase averages);
3. reduced reset diversity/noise/push removed useful exploration pressure;
4. batch/entropy/update statistics on an 8 GiB laptop are inadequate for gait
   discovery;
5. local synchronization/full-reset adaptations subtly change the training
   distribution even though they improve stated invariants; or
6. PPO/evaluation bookkeeping still has a wrapper-level discrepancy.

Do not answer this by guessing or immediately spending another 60M steps.

## 8. C06 declaration, outcome, and next experiment

C06 was predeclared as a forward-only gait-acquisition stage: seed 0;
5,000,000 requested / 5,007,360 actual steps; `vx=[0.2,0.6]`, `vy=yaw=0`, and
upstream 10% zero command; 500-step episodes; nominal reset; observation noise,
pushes, and domain randomization disabled; exact authoritative dynamics,
rewards, action mapping, and PPO profile; 512 environments; batch 32;
16 minibatches; four updates; unroll 20; LR `3e-4`; entropy `.005`; discount
`.97`; reward scale `1`; 512/256/128 actor and critic.

Its gate required finite training with final KL below `.2`, fixed `vx=.5`
survival of at least 400/500, forward RMSE at least 20% below checkpoint 0 and
standing, minimum pelvis above `.6 m`, both feet transitioning, single support
between 35% and 95%, and visually alternating gait. Only after all of those
could an omnidirectional warm-start begin.

C06 is now complete. Corrected inference passes KL, survival, height, and the
numerical RMSE comparison, but fails the decisive gait conditions: 0% single
support, 100% double support, zero transitions on either foot, no forward
tracking success, and no corrected video. Therefore C06 fails and no warm-start
omnidirectional stage is authorized.

The next experiment should be **C07, an evaluation/invariant audit with zero
new PPO steps**:

1. prove action equality between the fixed evaluator and Brax training-time
   inference on a saved normalized checkpoint;
2. add an autoreset test that simultaneously checks environment history and
   `EpisodeWrapper` terminal metrics;
3. rerun C06 corrected evaluation on the full eight-command grid for seeds
   2000-2002; add a tested evaluator video-command selector and generate one
   corrected **forward** video. Interpret stand/forward as the only
   in-distribution focal commands and the other axes as diagnostic;
4. rerun C02 and C04 selected checkpoints with normalization to determine what
   was hidden by the evaluator defect; and
5. for the old matched native family, use a temporary worktree at `27de436`,
   the closest recoverable snapshot of its historical transition semantics,
   plus only the evaluator-normalization correction. Per-run provenance does
   not prove that snapshot byte-identical to the dirty training tree, and the
   current native timing is not checkpoint-compatible.

C06 uses `reset.randomize=false`, so seeds 2000-2002 do not vary initial pose
or velocity. This repetition can catch inference nondeterminism and other RNG
state differences, but it is not randomized-reset robustness evidence.

Only after C07 should a new training configuration be frozen. A likely C08
candidate is a one-seed authoritative forward-gait diagnostic that removes the
10% zero command, avoids velocities near zero, and tests one evidence-backed
exploration/reward change at a time. Do not precommit a large budget or start
three seeds until a short run visibly alternates support and tracks velocity.

## 9. Cheap diagnostics before expensive runs

- Run the complete tests and the real jitted Playground reset/step test.
- Compare one saved checkpoint's first action through training and evaluator
  network construction; normalization must be bitwise/numerically identical.
- Inspect normalizer means/variances and action distributions for saturation or
  NaNs.
- Test autoreset terminal output and the following transition: command,
  last-action, phase, air time, steps, episode_done, truncation, and episode
  metrics must be mutually consistent.
- Reevaluate existing C02/C04/C06 checkpoints correctly before training new
  ones. Existing compute is sunk; evaluation is much cheaper.
- Plot reward-term contribution per step for a true zero-velocity stance versus
  an oracle gait at `vx=.5`; quantify the static local optimum.
- Compare the C05b oracle action/contact/phase trace with C06 for the same
  initial state and command. Check action amplitude, cadence, target saturation,
  and which joints create foot clearance.
- Run CPU MuJoCo one-step checks for action sign/scale, PD targets, sensor frames,
  yaw sign, and foot sensors after any environment change.
- Use a tiny/short PPO run with an early stop rule. Reject runs with high KL,
  NaNs, worsening length, no foot transitions, planted stance, hopping, or
  reward/termination exploits.
- Freeze a new held-out command-sequence hash and quantitative criteria before
  any final three-seed run; do not tune on seeds 2000-2002 again.

## 10. Do-not-repeat list

- Do not generalize `default_humanoid_legs` by replacing literal 12s with 29.
- Do not create another G1 asset tree or change the Menagerie pin without model
  evidence.
- Do not treat rising reward, longer episodes, or falling support transitions
  as walking.
- Do not use pre-`f642bc2` fixed evaluations of normalization-enabled
  checkpoints (D-012 runtime-control family onward) as behavior evidence.
  Earlier identity evaluations apply only to their actual normalization-disabled,
  misconfigured profiles.
- Do not reevaluate old native checkpoints under today's corrected transition
  semantics and call that faithful historical evaluation.
- Do not scale C02, C04, or C06 unchanged. C06 found the static-standing local
  optimum; C04 training length regressed.
- Do not restore a compact checkpoint and describe it as an exact training
  continuation. Optimizer state, PRNG/environment stream, and step counter are
  not restored.
- Do not use a very low evaluation count that compiles tens of transition
  batches into one enormous epoch on this laptop.
- Do not enable the generic native G1 domain-randomization path; it is explicitly
  rejected until a G1 mapping is implemented and tested.
- Do not assume the model/dynamics are incapable of walking; C05b falsifies it.
- Do not lower success criteria, select a prettier checkpoint after held-out
  inspection, hide negative runs, or start HOMIE features to bypass baseline
  failure.

## 11. Exact reproduction commands

Run from the repository root. The commands below were checked against the
current CLIs. Public upstream fetches are optional after the caches exist; no
Triton infrastructure is involved.

### Windows CPU setup and tests

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m pytest -q
```

The authoritative freeze result was obtained in WSL with:

```bash
.cache/g1_wsl_venv/bin/python -m compileall -q source scripts tests
.cache/g1_wsl_venv/bin/python -m pytest -q
```

Result: `93 passed, 25 warnings in 616.44s`.

### WSL2 CUDA setup

```bash
cd /mnt/c/Users/brand/PycharmProjects/Robocub
python3.12 -m venv .cache/g1_wsl_venv
. .cache/g1_wsl_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade 'jax[cuda12]==0.11.0'
python -c 'import jax; print(jax.devices())'
```

### Resolve sources and smoke G1

```powershell
.venv\Scripts\python.exe scripts\load_unitree_g1.py --mjx-scene --no-viewer
.venv\Scripts\python.exe source\scripts\smoke_g1.py --steps 1000 --seed 7 --no-fetch-model
```

For the adapter checkout, the first construction may fetch the exact public
Playground pin. Offline runs use `robot.fetch_model=false` and
`sim.playground.fetch_source=false` after both caches exist.

### Exact shipped-policy oracle

The regular sparse checkout does not require the optional policy blob. Add it
once and install the CPU-only oracle runtime:

```bash
cd /mnt/c/Users/brand/PycharmProjects/Robocub
. .cache/g1_wsl_venv/bin/activate
python -m pip install 'onnxruntime==1.22.1'
git -C .cache/mujoco_playground sparse-checkout add \
  /mujoco_playground/experimental/sim2sim/onnx/g1_policy.onnx \
  /mujoco_playground/experimental/sim2sim/play_g1_joystick.py
python source/scripts/evaluate_playground_onnx.py \
  --steps 500 --video \
  --output-dir results/gate4_corrective/C05c_oracle_reproduction
```

The output directory must be new. The script refuses a policy whose hash does
not match the pinned blob.

### C06 exact training command (historical reproduction; do not rerun blindly)

```bash
cd /mnt/c/Users/brand/PycharmProjects/Robocub
. .cache/g1_wsl_venv/bin/activate
export JAX_DEFAULT_MATMUL_PRECISION=highest
python source/scripts/train.py --logger local --seed 0 \
  env=unitree_g1_playground robot=unitree_g1 \
  sim=unitree_g1_playground agent=ppo_g1_corrective \
  robot.fetch_model=false sim.playground.fetch_source=false \
  sim.reset.randomize=false sim.noise.add_noise=false \
  sim.push.add_push=false sim.domain_rand.add_domain_rand=false \
  'sim.commands.lin_vel_x=[0.2,0.6]' \
  'sim.commands.lin_vel_y=[0.0,0.0]' \
  'sim.commands.ang_vel_yaw=[0.0,0.0]' \
  agent.num_timesteps=5000000 agent.num_evals=4 \
  agent.num_eval_envs=16 agent.episode_length=500 \
  hydra.run.dir=/mnt/c/Users/brand/PycharmProjects/Robocub/results/gate4_corrective/C06_REPRO_NEW \
  hydra.job.chdir=true
```

### Correct checkpoint evaluation and video

The first command below runs the full eight-command grid. At this freeze,
`evaluate_g1.py --video` renders only its hard-coded `combined` episode; for C06
that command is out of distribution and terminates almost immediately. The
second command therefore validates corrected rendering/inference only. As part
of C07, add and test a command selector before citing a representative forward
video.

```bash
python source/scripts/evaluate_g1.py \
  --run-dir results/gate4_corrective/C06_playground_forward_curriculum_seed0_5000000 \
  --checkpoint 5007360 --steps 500 --seeds 2000,2001,2002 \
  --nominal-reset \
  --output-dir results/gate4_corrective/C06_playground_forward_curriculum_seed0_5000000/evaluation/C07_normalized_three_seed

python source/scripts/evaluate_g1.py \
  --run-dir results/gate4_corrective/C06_playground_forward_curriculum_seed0_5000000 \
  --checkpoint 5007360 --steps 500 --seeds 2000 \
  --nominal-reset --video --render-every 2 \
  --output-dir results/gate4_corrective/C06_playground_forward_curriculum_seed0_5000000/evaluation/C07_normalized_video
```

### Checkpoint restoration without training

```bash
python - <<'PY'
from brax.io import model
p = model.load_params('results/gate4_corrective/C06_playground_forward_curriculum_seed0_5000000/logs/checkpoints/5007360/policy')
print(type(p), len(p))
PY
```

`train.py --resume --checkpoint PATH` is only a parameter warm-start, not an
exact resume. Use a new run directory and label it non-comparable if invoked.

### Clean export

```powershell
.venv\Scripts\python.exe scripts\export_repo_zip.py
```

## 12. Artifact map

These paths existed and were verified during the freeze. They are ignored and
will not survive an ordinary clone unless copied separately.

| Path | Contents / status |
|---|---|
| `results/gate4/` | about 451 MB / 2,595 files; historical native pilots, three full checkpoints, invalid old fixed evaluations, plots, interrupted and clean-export runs |
| `results/gate4/final_aggregate/` | old 216-episode summary/plots; trained inference invalid after normalization audit—retain only as historical evidence |
| `results/gate4_corrective/` | about 87.5 MB / 699 files; C00-C06 runs, checkpoints, evaluator outputs, oracle evidence |
| `results/gate4_corrective/C05b_exact_playground_shipped_onnx_oracle/` | valid `summary.json`, 1.96 MB `onnx_combined.mp4`, extracted review frames |
| `results/gate4_corrective/C06_playground_forward_curriculum_seed0_5000000/` | complete 5.007M run, checkpoints 0/1,669,120/3,338,240/5,007,360, configs, manifest, JSONL |
| `.../evaluation/checkpoint_5007360_nominal_seed2000_500/` | unnormalized evaluation with three MP4s; trained/checkpoint-0 policy evidence is invalid, while the zero-action standing trace is unaffected |
| `.../evaluation/checkpoint_5007360_normalized_seed2000_500/` | valid corrected one-reset-seed JSON/CSV; no video |
| `.cache/mujoco_menagerie/` | exact 40 MB public Menagerie checkout |
| `.cache/mujoco_playground/` | exact sparse public Playground checkout plus optional shipped ONNX |
| `.cache/g1_wsl_venv/` | validated 6.6 GB WSL CUDA environment; local only |
| `.cache/gate0_clean_venv/` | independent Gate-0 clean-install validation environment |
| `.cache/jax_compilation_cache/` | reusable ignored compilation cache |
| `.venv/` | ignored Windows CPU development/export environment |
| `outputs/` | preserved pre-change Hydra smoke output |
| `exports/simulation_robocup_export.zip` | generated source-only archive; regenerate after the handoff commit for the newest docs |

Important existing checkpoints were verified at:

- `results/gate4/baseline_corrective_seed0_20971520/logs/checkpoints/17825792/policy`
- `results/gate4/baseline_corrective_seed1_20971520/logs/checkpoints/19922944/policy`
- `results/gate4/baseline_corrective_seed2_20971520/logs/checkpoints/20971520/policy`
- `results/gate4_corrective/C04_playground_nominal_seed0_262144/logs/checkpoints/286720/policy`
- `results/gate4_corrective/C06_playground_forward_curriculum_seed0_5000000/logs/checkpoints/5007360/policy`

## 13. Success definition

Do not weaken the assignment to fit existing results. A PASS requires all of:

1. representative rollouts visibly show a stable alternating walking gait;
2. the large majority of ordinary held-out command episodes complete the full
   intended horizon, rather than surviving only 1-2 seconds;
3. forward and lateral velocity tracking materially beat both the matched
   untrained policy and standing controller;
4. yaw-rate tracking also materially beats both controls and has correct sign;
5. the result reproduces across at least three matched training seeds, not one
   lucky seed;
6. no reward clipping, termination, contact, stance, hopping, crawling, or
   evaluator exploit explains the score;
7. final evaluation uses frozen command sequences/reset streams not used to
   tune the final policy; and
8. rerunning from each saved checkpoint reproduces the numbers and videos.

Before the final expensive family, freeze a versioned command-sequence suite
and its hash. Unless a stronger rule is declared, interpret “large majority” as
at least 80% full-horizon survival overall and a majority within each forward,
lateral, and yaw group; interpret “materially” as at least 20% lower grouped
RMSE than **both** controls. At least two of three trained seeds must
individually satisfy the tracking, survival, and gait requirements, while all
three seeds must be reported. Require both feet to transition repeatedly and a
nontrivial single-support fraction; exact cadence/support bounds must be frozen
before looking at the final held-out traces.

Reward rising, one seed balancing, one command surviving, or linear RMSE over a
short falling transient is not a PASS. C06's 10-second static stance is useful
progress but is explicitly not locomotion.

## 14. Authority and safety boundaries

- Work locally. Public authoritative source fetches are allowed; Triton Droids
  remote infrastructure is not.
- Do not push, create PRs, modify remote branches, expose credentials, or require
  cloud logging.
- Do not connect to, deploy on, or test a physical G1.
- Preserve `source/locomotion/default_humanoid_legs` and its regression suite.
- Preserve every failed/interrupted/invalid experiment and label it honestly.
- Small coherent local commits are allowed and encouraged.
- No HOMIE Phases 5-7 until the conventional three-axis baseline passes and the
  user separately authorizes them.

## 15. Recommended successor strategy

Ranked by expected information value:

1. **Close inference confidence first.** Add action-parity and full autoreset
   bookkeeping tests, then run the corrected multi-seed C06 eight-command grid
   and one video, focusing conclusions on in-distribution stand/forward. This
   is cheap relative to PPO and prevents another invalid report.
2. **Recover information from existing checkpoints.** Correctly reevaluate C02
   and C04. Evaluate the old full native family only in a temporary worktree at
   its training semantics with normalization backported; do not contaminate the
   main branch or relabel old artifacts.
3. **Quantify the static local optimum.** Compare per-step reward terms and
   action/contact traces for C06 stance versus C05b gait at the same forward
   command. Decide from numbers whether command distribution, tracking kernel,
   gait/clearance rewards, or exploration is the first isolated change.
4. **Run one bounded seed-0 diagnostic.** Make one reversible change, use early
   gait gates, and stop if feet remain planted. Avoid three seeds until behavior
   is plausible.
5. **Scale progressively.** Short -> medium -> full only after visual and
   quantitative gait evidence. Reintroduce broader commands/reset diversity in
   stages, then freeze a new untouched held-out suite.
6. **Final confirmation.** Train at least three from-scratch matched seeds,
   evaluate all axes and mixed sequences, rerun checkpoint evaluation, and
   produce representative videos. Stop at Gate 4 and request authorization
   before any HOMIE-specific work.

Start by reading this file, `G1_BUILD_AND_RESEARCH_CONTRACT.md`,
`research/RESULTS.md`, `research/DECISIONS.md`, and the supplied papers at
`research/references/HOMIE_2502.13013.pdf` and
`research/references/MuJoCo_Playground_2502.08844.pdf`.
The most dangerous mistake is to trust the old fixed-evaluation plots or to
interpret C06 balance as walking.
