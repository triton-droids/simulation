# Proposed bounded training-duration comparison — awaiting user decision

No three-seed solution has passed. RD14–RD17 did not satisfy their frozen
mechanistic prerequisites, and their branches remain closed.

## Concrete proposal
Reopen training duration once, rather than continue optimizer diagnostics:
three FROM-SCRATCH development runs, seeds101/202/303, each401,408,000steps,
using the saved RN01 **base** recipe (2048env,batch128,unroll20,4updates,
lr3e-4,gamma.97,clip.2,original reward/full random resets,reproducible compiler
profile), with no late stage. No shared trained ancestor or parameter warmstart.
Compare all three final results against their saved RE03 20,070,400-step base
results. Total proposed training budget1,204,224,000steps. No checkpoint shopping.
Checkpoint/evaluation cadence should retain the existing2,867,200-step interval
(141evaluations) for operational monitoring;only final401,408,000 enters gates.
Keep nonfinite and post-first-two-consecutiveKL>=.2 stop criteria unchanged.
All original screens,conditional full numeric/control gates,and visual review
remain binding. Partial improvement does not permit final validation. A final
three-unused-seed/fresh-held-out campaign requires its own later frozen recipe.

## Why this question, and its limits
EXPERIMENT_PLAN.md records the paper's400M-step/32,768-environment configuration
and explains that smaller budgets originated with an8GiB laptop prototype.
Current cluster runs are20Msteps:about5% of that recorded step budget. This is
not evidence that400M will work. The proposal retains2048env and repository
adaptations,so it is explicitly NOT an exact paper replication. It tests duration
only, not a secretly changed objective, larger rollout batch, or selected seed.
The saved RN01_202 base took2334.8seconds for20M;linear extrapolation suggests
about13hours per400Mrun,roughly39GPUhours total,with substantial uncertainty.
Proposed hard caps:20htraining,22hqueue,24hSlurm perrun,72GPUhours maximum across
three jobs. OneGPU/3CPU/16GiB each,allready submitted together with paceduploads;
actual concurrency is scheduler-dependent. No extra jobs to fill GPUs.

## Approval and implementation status
This deliberately exceeds the current20M bounded experiments and8h launcher
limit and reopens the explicit no-unchanged-extension restriction. It must not
be launched as an ordinary repair or disguised continuation. User approval of
this concrete duration/budget change is required before implementation/submission.
The user explicitly approved the bounded longer-training comparison on Nov12
Pacific time. RB07 plans implement this approval;all other closed branches remain
closed. Launcher now accepts explicit1�24hour limits,default8unchanged. Tenlauncher
tests pass;threeplansvalidate;trainingargv match RN01base except approvedsteps,
evalcadence andoutputpaths. All evaluationdefinitions match original screens/full
gates except paths andfixedfinalcheckpoint. No earlycheckpointpromotion.

## Frozen rejection/progression proposal
Allthree must retain original nominal gates and not lose randomized survival
versusRE03(4/8,8/8,6/8). Seed101 must acquire originalforwardgait andseed303 must
survive both6001 commands;otherwise reject duration hypothesis,no extension,
extra seeds,or longer-duration sweep. Even if this prerequisite is met,only full
original development numeric/control/visual success permits finalvalidation.

If the user declines this budget,do not silently relax gates or launch another
knob sweep. Return with a redesigned research scope for review. Preserve all work.
