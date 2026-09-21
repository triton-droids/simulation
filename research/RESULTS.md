# G1 Build and Research Results

## C15 direct fresh acquisition fails; C16 balance stage frozen (2026-09-20)

C15 ran from clean `cb4c604`, fresh seed 7 with no restore, for 2,007,040
steps in 1023.80 s (trainer wall time). Checkpoint zero has normalizer count
zero and zero means; the fresh-initialization audit is saved with the run.
Initial survival was 70; final training-side survival 108.8125, return
14.156712, KL .060773. First-update KL .719571 settled to .041739 at the
second callback; no sustained later KL >=.2 occurred. Final checkpoint
2007040, assessed first, falls at 113 steps on reset 3000 with linear RMSE
1.513 and yaw RMSE .365. This already fails both the gait gate and extension
prerequisite. Reset 3001 also falls at 107 steps, linear RMSE 1.547577 and
yaw RMSE .633306. Complete controls/video are saved in `C15_forward_gate/`.
The reviewed 0-2 s montage shows loss of balance and falling backward;
minimum pelvis is negative in both trained episodes. No upright gait claim.
No unchanged C15 extension and no gait-acquisition success claim.

C06's preserved learning curve reached 500-step balance by 3,338,240 steps
using original upstream rewards. C16 therefore tests a distinct fresh-seed
balance stage with those weights and the corrected current wrapper; its
budget and stage-only balance criteria are frozen before launch. This does
not weaken the final gait/command/randomization/multi-seed requirements.

## C14 passes nominal retention and randomized recovery (2026-09-20)

The final checkpoint 2007040 passes the full frozen development gate. All
eight nominal seed-4000 episodes and all 16 randomized seed-5000/5001 episodes
survive 500 steps, with finite metrics. Nominal mean linear/yaw RMSE is
.159320/.105013; randomized means .207868/.140616. Randomized initial C13
has .394935/.295736, mean duration 8.04 s, fall fraction .25; standing has
1.124011/.669696, duration .93 s, fall fraction 1. C14's randomized linear/yaw
improvements are 47.37%/52.45% over initial. Worst randomized linear/yaw
RMSE .341152/.338696 remain below the frozen .35 per-episode cap. Minimum
pelvis across randomized episodes is .678900 m.

Nominal forward retains 86.6% single support and .34/.34 s median completed
air. Dense nominal and randomized combined-command frames show upright
alternating steps; randomized startup recovers and continues turning.
Artifacts: `C14_nominal_seed4000/` and `C14_randomized_seeds5000_5001/` under
`results/gate4_corrective/`, with full controls, videos, and montages.
This is a successful single acquisition lineage, not final multi-seed Gate 4.
C15 is predeclared to test a shorter fresh-acquisition recipe before final
independent training seeds and new held-out command/reset data are frozen.

## C14 training complete; batching optimization rejected (2026-09-20)

C14 completed 2,007,040 steps from clean `2a2b07b` in 942.87 s. Initial
randomized training-side mean survival was 285; final 388.125, return
106.127953, KL .124018. Intermediate survival varied (peak logged 472.25),
and the best-return checkpoint was 1146880. Assess final 2007040 first as
declared, not that intermediate checkpoint. All 17 checkpoint-zero actor/
normalizer leaves exactly match C13. Fixed nominal/randomized checks remain
pending; training-side variation alone does not establish forgetting because
the evaluator samples new initial states across callbacks.

An optional vectorized evaluator prototype passed 26 unit tests, including
ordering/padding invariants, but failed the prospective real-MJX parity
audit. At 50 steps on C13, forward/turn-left maximum yaw differences were
.118722/.156177 versus the .01 tolerance; turn-left reward difference was
.047769. Termination masks agreed; at most one contact sample differed.
This can arise from floating-point/contact sensitivity, but equivalence was
not established. The production batching CLI and path were removed; retain
serial numeric evaluation. Diagnostic helper/script, tests, and all audit
evidence remain: `results/gate4_corrective/C14_batch_physics_audit/` and
`source/scripts/audit_g1_batching.py`, clean audit commit `bb2f452`.

## Randomized recovery diagnostic supports C14 (2026-09-20)

`C13_randomized_oracle_seed5000/` records exactly equal initial qpos, qvel,
and phase_dt for C13 and the oracle. The split reset key and sampled gait
frequency match the evaluator protocol. C13 falls at 112 steps (111 in the
scan-based evaluator; floating-point graph differences are not hidden).
The oracle survives 500, linear RMSE .218454, yaw RMSE .381689, minimum
pelvis .654218, single support 74.8%. This demonstrates recoverability,
not oracle perfection on every metric. C14 tests only the previously absent
randomized-reset training distribution; its budget, retention requirements,
and conditional extension are frozen in EXPERIMENT_PLAN before training.

## C13 passes nominal commands; randomized reset gap remains (2026-09-20)

Final checkpoint 1003520 passes the complete frozen C13 development gate.
All eight nominal seed-4000 commands survive 500 steps, with mean linear/yaw
RMSE .153616/.130862. Initial C12 has .140362/.270396: yaw improves 51.60%,
with a small translation tradeoff. Standing has 1.091086/.174654 and falls
at 69 steps. Forward has 85.2% single support and .32/.32 s median completed
air intervals. Dense combined-motion video shows upright alternating steps
and clear turning. Artifacts: `C13_grid_seed4000/` under
`results/gate4_corrective/`, including both controls and all three videos.

| Nominal command | Linear RMSE | Yaw RMSE |
|---|---:|---:|
| stand | .084105 | .109015 |
| forward | .175391 | .158643 |
| backward | .207785 | .124199 |
| left | .184701 | .155313 |
| right | .212621 | .093865 |
| turn_left | .122610 | .136489 |
| turn_right | .128620 | .105786 |
| combined | .113097 | .163588 |

A subsequent explicitly developmental randomized-reset diagnostic on seed
5000 exposes missing robustness. Stand/forward/turn-left/turn-right terminate
at 107/111/119/95 steps; the other four survive 500. Mean linear/yaw RMSE
.630911/.455851, mean duration 6.08 s, fall fraction .5. Initial C12 also
falls on four commands, mean errors .784635/.758311 and duration 6.0025 s;
standing falls on all commands at 39 steps. Full evidence is preserved in
`C13_randomized_seed5000/`. This does not invalidate the nominal pass, but
prevents a robust baseline claim. Compare oracle recovery from the same
initial state before a bounded randomized-reset curriculum. No Gate 4 yet.

## C13 training completed; behavioral assessment pending (2026-09-19)

Clean source `0831d4f`, 1,003,520 steps, 678.76 s wall time; final training
survival 500, return 142.925140, KL .119197. No logged post-initial KL reaches
.2. Checkpoint-zero actor/normalizer exactly matches all 17 C12 saved leaves
(`C13_yaw9_seed0_1003520/restore_parity.json`). Final checkpoint 1003520 is
being assessed first against the unchanged eight-command gate plus the
stricter C13 mean-yaw criteria. Reward alone is not a pass.

C12's full combined-motion video and dense 0.2-second montage show sustained
alternating foot lifts and upright posture, consistent with its contact
metrics. Artifacts: `C12_combined_video/` under `results/gate4_corrective/`.
The evaluator now accepts mutually exclusive explicit nominal/randomized
reset switches, retaining saved task configuration by default. Its existing
22 tests pass (9.34 s, one deprecation warning). C13 evaluation remains
nominal as frozen; randomized robustness has not yet been demonstrated.

## C12 turning diagnosis and C13 rationale (2026-09-19)

Fixed-phase turn-left in the same MJX environment, command [0,0,.5], reset
2000, 500 steps: both C12 and oracle survive. C12 mean yaw .205543 and std
.183886 yield RMSE .347159; oracle mean .551168 and std .228875 yield .234525.
The main excess error is under-turning. Summed mean reward before dt still
favors C12 7.290668 vs 7.168455. Reweighting only angular tracking 2.25 -> 9
on these saved traces reverses the ranking to 11.894755 vs 12.957972 (oracle
+8.94%). Artifacts: `results/gate4_corrective/C12_turn_diagnostic/`, including
both full traces and `yaw_weight_counterfactual.json`. C13 is a bounded
single-factor pilot, frozen in EXPERIMENT_PLAN; no training success inferred.

## C12 full-command assessment: survival passes, yaw fails (2026-09-19)

C12 completed normally at 2,007,040 steps (944.48 s wall time), final training
survival 500, return 76.516464, KL .071242. The final checkpoint, assessed
first on all eight nominal commands at reset seed 4000, survives all eight
500-step episodes with finite metrics. Mean linear RMSE is .140362 versus
initial .241257 (41.82% improvement), but yaw RMSE is .270396 versus initial
.310485 (12.91% improvement). Standing controls have linear/yaw RMSE
1.090840/.173969 and fall at 69 steps on every command.

| Command | Linear RMSE | Yaw RMSE |
|---|---:|---:|
| stand | .110889 | .216696 |
| forward | .123631 | .229476 |
| backward | .173478 | .210292 |
| left | .179932 | .187056 |
| right | .188356 | .276369 |
| turn_left | .092981 | .377552 |
| turn_right | .144448 | .371975 |
| combined | .109180 | .293751 |

Full precision: `results/gate4_corrective/C12_grid_seed4000/episodes.csv`.
Forward retains 84% single support and .34/.32 s median completed air
intervals. Nevertheless stand yaw, pure turns, and mean yaw fail the frozen
success rule. The predeclared unchanged extension is **not eligible** because
yaw improvement is below 20%. Do not extend C12 unchanged or claim Gate 4.
Next: zero-training same-MJX turning comparison with the shipped oracle,
recording mean yaw, oscillation, and reward components before choosing a
new bounded experiment.

## C12 preparation and launch (2026-09-19)

The first prior preparation passed inference comparison but failed at save
because the script used an unavailable Orbax utility namespace. The writer
now uses the same `flax.training.orbax_utils` helper as training; the failed
log and empty output directory remain preserved. The separate v2 preparation
passes 64-input old-subspace action parity and full-checkpoint round-trip,
both with maximum difference zero; actor/critic weights are unchanged.
Artifact: `results/gate4_corrective/C12_command_prior_v2/audit.json`.

C12's bounded broader-command run starts from clean `fa3e112`, with exactly
the frozen commands and reward weights. Its saved checkpoint-zero actor and
normalizer exactly match the prepared checkpoint. Initial training-side
mean survival is 479.8125 steps and return 57.594376. Inspect
`results/gate4_corrective/C12_allcommands_seed0_2007040/` for live status;
no broader-command success is claimed yet.

## C11 verified forward gait (completed/recovered 2026-09-19)

Training from clean `7c531b1` completed 1,003,520 steps in 677.49 s; final
mean survival 500, return 58.332001, KL .070638. Checkpoint-0 actor/normalizer
matches C10 exactly. The first evaluator was interrupted during the pause
after saving trained seed 3000 and its video. A fresh numeric recovery run
preserved those files and exactly reproduced that episode.

| Reset seed | Steps | Linear RMSE | Single support | L/R median completed air (s) |
|---|---:|---:|---:|---:|
| 3000 | 500 | .104055 | 66.4% | .28/.18 |
| 3001 | 500 | .098 (rounded) | 68.4% | .28/.24 |

Dense video shows clear alternating leg lifts. Fixed 1.5 Hz/reset 2000 also
passes: 500 steps, linear RMSE .121543, minimum pelvis .730030 m, single
support 65.8%, and median completed air .26/.14 s. Thus C11's declared
forward-gait gate passes. Yaw remains imperfect (.314 rad/s RMSE on seed
3000); broader commands and three final training seeds remain unverified.

Artifacts: `C11_contact_phase2_seed0_1003520/`, partial
`C11_forward_gate_seeds3000_3001/` with video and dense montage, complete
`C11_forward_gate_recovery_20260919/`, and `C11_fixed_phase_gate/`, all under
`results/gate4_corrective/`. A zero-training normalization audit finds lateral
and yaw command std 1e-6; a .1 command would normalize to 100,000 without a
prior. The opt-in prior helper passes six checkpoint/invariance tests and
requires an actual-checkpoint parity audit before the C12 curriculum.

C11 prerequisites passed: 37 targeted adapter/evaluator/PPO-configuration
tests, three warnings, 162.16 s in WSL CUDA, including real MJX with contact
weight 0 and 2. Log: `results/gate4_corrective/C07_audit/C11_tests.log`.
Four local pure invariant tests also pass. The optional reward is tested
before training; these results do not establish learned gait quality.
The evaluator now records per-foot median completed air intervals, excludes
boundary-censored swings, and incrementally persists episode JSONL before
video rendering. Its complete 16-test module passes locally (6.68 s); an
initial Windows temp-folder permission error was resolved with a fresh
workspace test directory. Training source is frozen separately at `7c531b1`.

## C10 completed: numerical forward pass, gait-quality hold (2026-09-16)

Clean training base `253f9bc`, 1,003,520 steps in 640.71 s. Exact 17-leaf
actor/normalizer restore equality. Final training survival 500, return
46.126392, KL .075201. Final checkpoint assessed first as declared.

| Nominal reset seed | Steps | Linear RMSE | Min pelvis | Single support | L/R transitions | Yaw RMSE |
|---|---:|---:|---:|---:|---:|---:|
| 3000 | 500 | .183816 | .716093 | 38.2% | 58/49 | .471 |
| 3001 | 500 | .133795 | .738901 | 35.6% | 54/56 | .299 |

Initial C09 also survives both, but has RMSE .225252/.202612 and single support
25.4%/28%. Standing falls at 69 steps (RMSE .849567). C10 passes the declared
numerical criteria on both seeds. Fixed phase 1.5 Hz/reset 2000 also survives
500, RMSE .133596, minimum pelvis .747606, single support 35.2%.

Dense video and foot-position traces show brief irregular lifts rather than
the oracle's sustained cycles. Pooled median air interval is .06 s versus
oracle .20 s; foot-site p95 heights .0594/.0531 m versus .0998/.0927 m.
Consecutive-contact foot-site XY speed averages .1243 versus .1034 m/s; this
is a finite-difference site-motion proxy, not exact contact-point slip and
does not alone establish sliding. Do not conflate numerical pass with a
finished visual gait or satisfactory yaw tracking. Broader commands remain
on hold. C11's prospective sustained-support diagnostic is separately frozen.

Artifacts: `C10_phase3_seed0_1003520/`, `C10_forward_gate_seeds3000_3001/`
(CSV, three videos, full/dense montages), and `C10_fixed_phase_gate/`
(matched traces, `gait_quality.json/png`, counterfactual contact-phase scores),
all under `results/gate4_corrective/`.

## C10 zero-training shaping diagnostic (2026-09-16)

The same-MJX comparison at fixed 1.5 Hz/reset 2000 exposed C09 fragility:
457 steps before termination, whole-trace RMSE .783747 and 31.29% single
support, versus oracle 500 steps, .174590 and 75%. C09's termination allows
the pelvis to drop far below the gait gate height; full-trace minimum is
-.617559. This is not equivalent to its nominal seed-3000 gate rollout.
The first fixed 250 steps remain upright and show 25.6% single support.
Increasing existing feet_phase weight 1 -> 3 on those saved traces raises
oracle's reward advantage from 10.8% to 18.2%. The post-hoc diagnostic window
and full negative trace are both disclosed in `C10_gait_shaping_diagnostic/`.
This supports a bounded phase-weight pilot, not a claim of gait acquisition.

## C09 completed: movement acquired, gait gate still failed (2026-09-16)

From clean `c321fc0`, C06 parameter warm-start plus tracking weight 3 ran
1,003,520 additional steps in 451.94 s. All 17 saved actor/normalizer leaves
at checkpoint 0 exactly matched C06; initial mean survival was 500 steps.
Final training survival 450.5625, return 27.291819, KL .078490; all interval
KL values .077-.099. Optimizer and PRNG restarted as declared.

Final fixed-forward seed-3000 diagnostic: 500/500 steps, linear RMSE
.225252, minimum pelvis .741102, 25.4% single support, 74.6% double support,
no flight and 54/62 foot transitions. Initial C06: 500 steps, RMSE .500030,
zero transitions. Standing: 69 steps, RMSE .849567. All finite. C09 passes
survival/height/error/transition-count thresholds but fails the frozen >=35%
single-support threshold. Twelve-frame video inspection shows stiff legs and
low foot clearance rather than a convincing alternating gait. Yaw RMSE is
.427, another limitation even though C09's forward gate did not threshold it.
Do not broaden commands or call Gate 4 passed.

Artifacts under `results/gate4_corrective/`: `C09_C06_tracking3_seed0_1003520/`
(including `restore_parity.json`) and `C09_forward_gate_seed3000/` (CSV,
summary, three videos, trained montage). Next is a zero-training same-MJX
reward/contact comparison with the oracle, before choosing gait shaping.

## C08 completed: forward gait gate failed (2026-09-16)

Tracking weight 3, from scratch, clean training base `688eaa3`, exactly
2,007,040 steps in 1,048.35 s. Final training length 94.6875, return 2.580198,
KL .055301; first-update KL 1.144106, later values .038-.055. Normal run end.
The final checkpoint was evaluated first, irrespective of the earlier
training-return peak. Fixed forward .5, nominal seed 3000, 500 requested steps:

| Controller | Steps | Linear RMSE | Min pelvis | Single support | L/R transitions |
|---|---:|---:|---:|---:|---:|
| Trained | 71 | .598791 | .381873 | 8.45% | 9/1 |
| Untrained | 69 | .808679 | .102939 | 4.35% | 1/6 |
| Standing | 69 | .849567 | .063025 | 10.14% | 2/9 |

All finite. The trained policy modestly improves error against falling
controls but fails absolute tracking, survival, height and gait requirements.
Six-frame inspection shows hopping then collapse, not alternating walking.
Artifacts: `results/gate4_corrective/C08_tracking3_seed0_2007040/`,
`C08_forward_gate_seed3000/{episodes.csv,summary.json,trained_montage.png}`
and three corresponding forward videos. Evaluation used highest matmul
precision, matching training. Gate 4 remains unpassed. C09 is a distinct
forward-only balance-to-gait initialization diagnostic; see its declaration.

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

Historical recovery subsequently completed for all three training seeds on
reset seed 2000, using isolated `.cache/C07_historical_native` at `27de436`
with only observation normalization backported. The exact patch is saved in
`C07_audit/historical_normalization.patch`. This is the closest recoverable
snapshot, not proof of byte-identical historical dirty training source. These
are exploratory recovery results, not a replacement final held-out campaign.

| Native training seed | Mean duration (s) | Fall fraction | Mean linear RMSE | Mean yaw RMSE | Forward RMSE |
|---|---:|---:|---:|---:|---:|
| 0 | 9.0500 | .125 | .284160 | .298757 | .502259 |
| 1 | 8.1325 | .250 | .312494 | .302764 | .498085 |
| 2 | 6.9700 | .500 | .520929 | .385661 | .500871 |

The normalization omission substantially understated survival. All three
still fail forward tracking at command `.5`; aggregate RMSE gains against
falling controls do not establish walking. Outputs are
`C07_historical_seed{0,1,2}_normalized/` and
`C07_audit/historical_recovery_summary.json`. The audit and fixes are committed
as `e2cbf41`. C08's 1,024-step wrapper/checkpoint smoke starts from that clean
commit; the bounded gait pilot remains gated on its completion.

C08 smoke subsequently completed in 297.54 s from clean `e2cbf41`, writing
checkpoints 0 and 1024 and a normal `run_end`. Return `-1.340551 -> -1.379093`
is infrastructure-only evidence, not improvement. Reloading checkpoint 1024
passed 64-input action parity with maximum difference 0 (identity inference
differs by 1.964582). No source changes occurred during training; only research
documentation changed before the separate reload audit. Artifacts:
`C08_wrapper_smoke_seed0_1024/` and `C07_audit/C08_smoke.log`.
All predeclared prerequisites for the bounded C08 gait pilot are satisfied.

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
