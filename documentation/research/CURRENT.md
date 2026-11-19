# Current research handoff - 2026-11-18 resume

User explicitly requested resume from the latest checkpoint Nov18. Active manifest: `documentation/research/queues/local_rb07_resume_20261118.json`. Receipt/status: `Artifacts/Results/local_rb07_resume_20261118/launch_receipt.json` and `batch/status.json`.

Run local_rb07t_202 from local_rb07s_202 checkpoint258048000 (cumulative346931200), remaining54476800, then original unlaunched local_rb07s_303 (remaining312524800). Completed local_rb07s_101 remains untouched. Review all three together. Original controls/gates unchanged; fixed cumulative401408000 each. Parameter continuation, not exact optimizer/PRNG/environment resume.

New cap104100s (28h55m):202 queue28500/train17700;303 queue75600/train64800. Prior batch consumed102951.937809s; new cap is below remaining122228.062191s. No extension. Preserve both prior STOP/pause records. User pause is revoked for this fresh batch only. Do not replay old batches. Audit is complete at Artifacts/Results/local_rb07_resume_20261117/replication_audit.md; no repeated audit. Four-hour checks; healthy runs need no logs or polling. No source/dependency edits while active. Next hypothesis comparison still requires separate approval.

## Historical Nov17 handoff (superseded paths)

# Current research handoff - 2026-11-17

**User explicitly authorized resuming laptop GPU training and scheduled review on Nov17.** Check the new batch receipt/status before any action; never replay the old paused batch.

Active continuation plan: `documentation/research/queues/local_rb07_resume_20261117.json`.
Runtime receipt and status: `Artifacts/Results/local_rb07_resume_20261117/launch_receipt.json` and `batch/status.json`.
Runs: local_rb07s_101 -> local_rb07s_202 -> local_rb07s_303, serial on one laptop RTX4060. Do not launch competing GPU jobs.

Seed101 starts from local_rb07r_101 checkpoint5734400, cumulative100352000, with301056000 steps remaining. Seeds202/303 start from original88883200 parents with312524800 steps remaining each. Final cumulative target401408000 each; original checkpoint0 controls and all original evaluation gates remain unchanged. No final three-seed validation is claimed. Continuations reset optimizer/PRNG/environment state; they are not exact resumes.

Historical paused batch consumed1573.299775seconds; charge1620seconds conservatively and deduct that from the previous63h future cap. New batch cap225180seconds (62h33m). Including priorcluster5.060833h and reservedprobe.75h, total maximum68.810833h remains below approved72h. See the new resume_budget.json for accounting. No silent budget extension or automatic scientific redesign.

Training/environment/config source files are unchanged from24ceee9. UI/path/launch-tool changes and new frozen plans are committed before launch. Do not edit tracked source or dependencies while this batch runs. Preserve the historical STOP and pause record. A new stop request applies to the NEW batch output path.

The approved investigation remains Artifacts/Results/local_rb07_recovery/investigation_plan.md: one replication audit while training runs, review all three fixed-final outcomes together, then at most TWO separately user-approved hypothesis comparisons before mandatory review. No lucky checkpoints, mixed recipes or gate weakening. Further compute/scientific comparisons require explicit budgets.

GUI: Scripts/dashboard.py. Policy paths come from Policies/catalog.json; demo is Policies/g1-walking-demo and experiments are Policies/g1-walking-experimental/replica-1 through replica-3. Cleanup is complete; never reapply queued cleanup or old migration scripts. Old outputs and receipts are immutable evidence.
