# G1 milestone plan — current scope

Effective 2026-09-27 by explicit user request. This plan supersedes older
"continue until three seeds pass" instructions and historical active-queue notes.
Milestone 1 is complete; no milestone is active. Milestones 2 and 3 require a new user go-ahead.

## Milestone 1 — usable simulation controller handoff (COMPLETE — 2026-09-28)

Deliver a working, documented simulation prototype for teammates to integrate.
Use the existing T02 rate_cost final checkpoint1003520, not a new training search:
`results/post_f1_t02/rate_cost/train/logs/checkpoints/1003520/policy`.
Keep its saved configuration/normalization and pinned model/source dependencies.
Do not replace it with a newer or higher-reward policy without a documented
artifact/compatibility blocker. No training, retuning, new seeds or research
extensions are authorized under this milestone.

### Completion checklist

- [x] Identify/checksum the checkpoint and required configs, source pins,
  dependency versions and provenance; preserve originals outside Git.
- [x] Provide a repository-relative, one-command simulation demo with clear
  setup/preflight errors and a documented `[vx, vy, yaw_rate]` interface:
  units, coordinate frame, tested bounds, update timing, reset and stop behavior.
  Stop means request zero velocity; do not promise instantaneous motion arrest.
- [x] Provide a minimal scripted integration example using that same interface
  so teammates can later connect ball-approach logic. No perception/kicking scope.
- [x] Run/document fixed nominal forward, backward, lateral, turning and stand
  examples. Reuse validated evaluator/runtime; verify command changes and
  move-to-stop behavior rather than assuming independent episodes prove transitions.
  Record actual supported scope; a broken basic advertised command is not a pass.
- [x] Include reviewable videos and machine-readable checks, identify checkpoint,
  commands and resets; visually inspect walking/turning/stopping evidence.
- [x] Include the known disturbed backward failure (seed6001) and its existing
  evidence, with limitations plainly visible. No robustness/three-seed claim.
- [x] Verify documented setup/demo from a clean temporary checkout or equivalent
  clean working directory with explicitly provisioned external caches/checkpoint.
  Do not claim the ignored checkpoint is bundled in Git. Provide a local packaging
  or restore procedure and manifest; no upload or remote distribution required.
- [x] Run targeted interface/restore/command tests and relevant regression checks.
- [x] Write `research/MILESTONE_1_REPORT.md` and update `research/RESULTS.md` with
  the result, deliverable paths/commands, verification evidence, measured behavior,
  failures, limitations, provenance, and preserved Milestone 2 resumption pointer.
- [x] Mark status complete only after all required checks actually pass; notify
  the user once with the report/demo links, then pause the existing repeating task.

### Boundaries and budget

This is delivery of one existing development policy, not research validation.
The previously recorded nominal24/24 and extra-forward12/12 successes support
candidate choice; full randomized23/24 and incomplete final visual review remain
failures/limitations. Do not erase them or claim Gate4/three-seed success.
Use short smoke checks first, then a bounded local verification queue if needed.
Freeze each verification job before launching. Initial allowance: at most two
GPU-hours of evaluation/demo runtime (zero training), excluding fixing genuine
execution errors. If the prototype cannot meet basic handoff requirements, report
an honest blocker and stop expansion; never silently start Milestone 2.

## Milestone 2 — reproducible validated locomotion recipe (DEFERRED)

Preserve all existing work in place; see `MILESTONE_2_RESUME.md`.
When separately authorized: audit differences from the pinned upstream baseline,
benchmark the intended training configuration and freeze a finite compute budget.
Establish full development numeric/control/visual success before a separately
frozen test of three predetermined fresh independent training seeds and fresh
held-out scenarios. All three must pass the original full criteria. No lucky-seed
selection, gate weakening, or relabeling development data as held out.
Intermediate phases: faithful baseline -> complete development pass -> frozen
three-seed confirmation -> evidence/report and stop for review. A baseline
reproduction and the team's additional acceptance criteria must be distinguished.
M1 completion does not grant permission to resume this work automatically.

## Milestone 3 — soccer integration (DEFERRED)

After explicit scoping: simulation ball perception/state estimation, approach
behavior, motion-skill transitions, kicking and a small end-to-end soccer demo.
These are separate acceptance tasks, not abilities implied by a walking policy.
Getting up after a fall is a separate skill. Hardware transfer/real-robot testing,
remote infrastructure and paid compute require separate explicit authorization;
none is automatically included by Milestone 3 or completion of Milestone 2.

## Persistence and stop protocol

`research/milestone_status.json` is the compact work state. Read it first on
scheduled wakes. Follow this document, not old queue prompts or historical notes.
Use statuses active, verifying, blocked, complete with evidence paths and next
concrete action. A fresh running verification queue needs no logs/polling/model
waiting. Do not edit tracked files while a local queue is running.
On complete: report, request pause through the automation tool, verify persistence.
If app persistence fails, keep a completed stop marker and explicitly report the
schedule problem; future wakes do no research or repeat the completion report.
On blockers: record evidence and notify required user action; do not invent a pass.
