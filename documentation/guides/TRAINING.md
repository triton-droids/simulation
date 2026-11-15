# Training: examples versus research

| Workflow | Purpose | Results |
|---|---|---|
| Smoke test | Check reset, PPO, logging and checkpoint creation | `Artifacts/Runs/` |
| Historical public trainer | Preserved default profile | `Artifacts/Runs/` |
| Example parameter matrix | Exercise trainer/world combinations | `Artifacts/Sweeps/` |
| Frozen research queue | Test a predeclared scientific hypothesis | `Artifacts/Results/` |

A smoke test, high reward or a selected demo is not a three-seed pass. Do not
start another training job while the local research batch owns the GPU.

## Verify the pipeline when compute is free

```powershell
.venv\Scripts\python.exe Scripts\train.py --smoke
```

This runs 1,024 environment steps with a small network, recording checkpoints,
metrics, configuration and a summary. Plot/video generation depends on runtime;
inspect outputs rather than assuming a video exists. Without `--smoke`, this
entry uses the historical profile, not the current research recipe.
Native Windows JAX on this host uses CPU. GPU research uses verified WSL or an
approved cluster allocation; selecting a dropdown does not migrate a run.

## What settings control

| File | Controls | Explanation |
|---|---|---|
| `Configs/sweeps/training_hyperparameters.json` | Budget and PPO updates | [PPO](PPO_REFERENCE.md) |
| `Configs/sweeps/environment_hyperparameters.json` | Task/world bundles | [Rewards](REWARDS_REFERENCE.md), [configuration](CONFIGURATION_REFERENCE.md) |
| Frozen queue JSON | Approved commands, seeds, budgets, checks and outputs | Active research contract |

The example matrix pairs every trainer bundle with every environment bundle:

```powershell
.venv\Scripts\python.exe Scripts\sweep.py --smoke
```

Its leaderboard ranks reward; it does not establish gait quality or permit lucky
seed/checkpoint selection. It is not a substitute for the bounded investigation.

## Research workflow

Inspect progress/results in the dashboard. Start approved plan only offers
registered reviewed plans with fresh outputs. An empty catalog is valid.
The research contract is [CLUSTER_CAMPAIGN.md](../research/archive/CLUSTER_CAMPAIGN.md);
current status and continuation handoffs are saved with each run in `Artifacts/Results/`.
The approved investigation distinguishes replication, training duration,
implementation defects and architecture limits. No automatic unbounded sweep.

Evaluate every prescribed seed/control with the original numeric/control/visual
gates. Preserve failures. Parameters without optimizer/PRNG/environment state
support parameter continuation, not exact resume. Recover evaluation from saved
weights without repeating training. New hypothesis comparisons need approval.
