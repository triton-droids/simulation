# Milestone 2 preserved resumption record

Deferred by user on2026-09-27 to finish Milestone1 first. Do not resume automatically.
No checkpoints, evidence, source adaptations, plans or logs were deleted or moved.
Historical research commit before scope change:08c6173.

## Work preserved

- T02 rate_cost: `results/post_f1_t02/rate_cost/`; final checkpoint1003520.
  Nominal24/24 numeric pass, extraforward12/12; random23/24 with backward6001
  failure109steps and aggregate linear failure. M1 prototype, not full validation.
- B08 seed11 and R02seed22/33: `results/post_f1_b08/`, `results/post_f1_r02/`.
  Both R02 seeds stand precisely but lack forward gait; randomized0/8 each.
- M01 mapping trial and A16 diagnostics: `results/post_f1_m01/`,
  `results/post_f1_a16/`; failed, branch closed. Optional code remains default-off.
- R03 matched original-objective seed22 control: `research/queues/post_f1_r03.json`
  and `results/post_f1_r03/`. Intentionally stopped through queue/STOP at user
  scope change; last recorded training step14336000. Checkpoints0,2867200,
  5734400,8601600,11468800,14336000 preserved. No completed final20M result;
  do not score this interruption as scientific rejection or auto-repair it.
- Earlier F1/C/B/T/O/L/A studies, failed outputs, recipe/code variants, environment
  pins, tests and decision history remain in their original locations and Git.
- Detailed evidence: PIPELINE_AUDIT_A01.md, POST_F1_CONTINUATION.md, DECISIONS.md,
  EXPERIMENT_PLAN.md, SOURCE_LEDGER.md and historical RESULTS.md sections.

## Before any future resumption

Get explicit user authorization for M2 and read MILESTONES.md first. Preserve M1
frozen interface/artifact. Reassess faithful baseline versus further local tuning;
choose a finite compute budget and decision point. Inspect whether R03 saved
artifacts include optimizer/PRNG state before proposing a true resume: parameter
warm-start is not equivalent to uninterrupted training. Do not remove STOP or
reuse old output directories automatically. Keep fresh validation seeds distinct
from all used development seeds/grids. All original full criteria still apply.

Preservation check at scope change: T02 candidate directory and R03 checkpoint14336000 exist; R03 status records STOP termination with child_pid=null. No research source code or evidence deleted.
