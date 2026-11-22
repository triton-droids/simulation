# Current handoff - RT01 turn dead-zone comparison on Atlantis (2026-11-21)

User approved comparison 2 of 2 ("test that") on 2026-11-21, as proposed after TS01. Fix arm: separate yaw-rate tracking sigma .05 (upstream shared .25) plus `in_place_turn_probability` .15, where 15% of moving command draws become zero-linear turns with |yaw| in [.2,1.0] and zero draws stay zero (about 13.5% of all draws). Control: upstream reward and sampler. Both arms: lr 1e-4 (RF01 used 3e-4), narrow_feet 0, continuing each fixed final RB07 401408000 policy for 51609600 steps with exact restore parity, the KL guard, screens and full evaluations unchanged, plus an unconditional `turn_pure_yaw` diagnostic (TS01 grid) on the final checkpoint. Queues: `cluster_rt01_{101,202,303}_{fix,ctl}.json`, six Atlantis RTX2080Ti jobs. This is the last approved comparison; mandatory review follows.

Decision rule, frozen before launch. SUPPORT only if every fix arm (1) passes nominal turn_left and turn_right yaw (max yaw_rate_rmse <= .25) in full_c15_nominal, (2) has mean yaw_rate_rmse <= .25 at pure yaw +.3 and -.3 in turn_pure_yaw, and (3) shows no survival, foot-contact or other gate failure absent from its matched control, and the matched controls do not also clear (1) and (2) on all three seeds (otherwise lr/duration explains it). REJECT otherwise, with no extension or retry. A fix arm that is KL-stopped counts as a failure; a KL-stopped control is reported and leaves that seed unmatched. Even if supported, fresh-seed final validation needs separate approval.

## Previous handoff - TS01 CONFIRMED, comparison 2 needs approval (2026-11-21)

TS01 is CONFIRMED under its frozen rule (review: `Artifacts/Results/cluster_ts01_review.md`). All seven policies have a stand-still dead zone for small commands. Pure yaw .2-.3 stands in 7/7 policies, yaw .4 flips by seed and direction, and yaw .6 and above, or yaw .4 with forward speed, steps and tracks (yaw error about .1). The cause is the reward shape: standing earns 70% of the yaw reward at .3. In-place turns are also rare in the sampler (about 1%). Comparison 2 of 2 is unused and needs explicit approval of its hypothesis, budget, commands and decision rule.

## Previous handoff - TS01 turn-command diagnostic on Atlantis (2026-11-21)

User approved an evaluation-only turn sweep and cancellation of frozen job 7983 (RF01_101_fix, cancelled after 7h52m at step 0) on 2026-11-21. TS01 is not a hypothesis comparison: no training, no promotion, gates unchanged, comparison 2 of 2 is still unused. Hypothesis: nominal turn failures are an in-place-turn deficit (the policy stands still for zero-linear yaw commands), because the upstream sampler gives |v|<.1 with |yaw|>=.3 in only about 1% of training commands. Seven policies: RB07 finals 101/202/303 (RF01 checkpoint 0, with an exact actor/critic/normalizer parity check against the parent) and RF01 202/303 fix/ctl at 51609600. Three nominal grids on seeds 6000-6002: `ts01_{pure_yaw,moving_yaw,mixed_yaw}_commands.json` (labels are evaluator slots; the CSV records the actual commands). Plans: `cluster_ts01_*.json`. Budget: seven 1-GPU jobs, at most 4h each.

Reading rule, frozen before launch. CONFIRMED if, for each policy/direction failing pure yaw .4 (yaw_rate_rmse > .25), most failing pure-yaw episodes show single_support_fraction < .05 (standing), and adding forward speed (vx .1-.45, same yaw) brings mean yaw_rate_rmse to <= .25 with stepping. REFUTED if yaw tracking still fails with forward speed, or the failures show stepping (single_support >= .2): that would make it general turn-tracking error, not an in-place deficit. Anything else is MIXED and is reported as such. Command-curriculum training is proposed only if CONFIRMED.

## Previous handoff - RF01 rejected, decision required (2026-11-21)

RF01 narrow-foot penalty is REJECTED under its frozen rule. 303 fix had 3 foot-contact terminations against 1 in its control, and 202 fix and 202 control both had 0, so the penalty did not cause the improvement. Every arm fails nominal turning yaw (.36-.40 > .25): that is now the shared blocker. 101 control was stopped by the KL guard (kl 11.25), and 101 fix froze at compile with no result. Review: `Artifacts/Results/cluster_rf01_review.md`. Comparison 1 of 2 is used; the next comparison needs separate approval. Final validation is not unlocked.

## Previous handoff - RF01 narrow-foot comparison on Atlantis (2026-11-21)

User approved comparison 1 of 2 on 2026-11-21 and lifted the 72 GPU-hour cap for Atlantis work. Diagnosis: 3/4 RB07 randomized failures are foot-foot/foot-shin contact terminations while upright (pelvis >.74m), not falls; passive references also end on foot contact at reset6002. RF01 tests the existing narrow_feet cost (weight -4, width .16m) from each fixed final 401408000 policy for 51609600 more steps, with a matched penalty-off control per seed: queues `cluster_rf01_{101,202,303}_{fix,ctl}.json`, six Atlantis RTX2080Ti jobs. Gates unchanged; full evaluations run unconditionally so every arm reports foot-contact terminations.

Decision rule, frozen before launch: support only if all three fix arms have zero undesired-contact terminations across screen, full randomized and 12 forward-reset evaluations with no failure absent from the matched control. Reject if fix arms do not reduce foot-contact terminations versus controls or regress survival/gait relative to controls; no extension or retry. Turn/backward tracking misses are out of scope and would need comparison 2. Final fresh-seed validation is not unlocked by RF01.

## Previous handoff - batch complete, decision required

All three cumulative401408000-step development runs completed. None passes all original gates. Scheduled research paused pending separately approved next work. Report: `Artifacts/Results/local_rb07_resume_20261118/mandatory_review.md`. No automatic extensions, new seeds or final-validation launch. Audit complete. Preserve all prior runs.

## Historical handoff

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
