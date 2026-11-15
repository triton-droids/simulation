# Current research handoff - 2026-11-14

**Training and the scheduled review are PAUSED at the user's request. Do not restart either without explicit instruction.**

No three-seed solution has been certified. The approved investigation first audits
paper/configuration differences, reviews the bounded longer-training comparison,
then permits at most two separately approved hypothesis comparisons before a
mandatory review. No automatic architecture change, budget extension or gate weakening.

Seed101 local continuation stopped with saved5734400 steps after original94617600
(cumulative100352000). Seeds202/303 local continuations did not start; original
parents remain88883200 each. Optimizer/PRNG/environment state are not saved:
future continuation is not an exact resume. Reconcile remaining steps and actual
compute cost before any new launch; never replay the old batch blindly.

## Authoritative records

- [Approved investigation](../../Artifacts/Results/local_rb07_recovery/investigation_plan.md)
- [User pause record](../../Artifacts/Results/local_rb07_recovery/USER_PAUSED.txt)
- [Batch status](../../Artifacts/Results/local_rb07_recovery/batch/status.json)
- [Campaign history and gates](archive/CLUSTER_CAMPAIGN.md)
- [Historical decisions](archive/README.md)

The three local paths above are generated records and may be absent in a fresh clone.
The older NEXT_CAMPAIGN_DECISION proposal was subsequently approved; its old title
does not mean approval is still pending. Use this handoff first.

User-facing cleanup is now applied: GUI remains Scripts/dashboard.py,
Policies contains the complete T02 bundle and named references to experimental weights, Artifacts/Videos indexes
recordings without moving originals. No desktop shortcut. Preserve original paths.

Current scope: [CONTRACT](CONTRACT.md). Source-layout migration details: [layout report](../../Artifacts/Results/layout_consolidation/report.md).
