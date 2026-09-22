# Continued research after F1 failure

The user explicitly authorizes continued bounded research until a recipe passes
three fresh independent training seeds. This supersedes the F1 stop-after-failure
instruction; it does not permit selecting three lucky seeds or weakening gates.
F1 remains a failed, incomplete validation study. Seeds 11/22 and command grid
6000-6002 are now development evidence, not fresh final validation data.

D01 freezes a zero-training comparison: evaluate seed11 linear checkpoint
1003520 on the same randomized grid as the failed recovery checkpoint.
Budget: 72 episodes of 500 steps maximum including two controls; one gait
video per controller; three-hour queue cap. Exact commands and unchanged
numeric gates: queues/post_f1_d01.json. Outputs are fresh and evidence preserved.
If the parent materially outperforms recovery, investigate the recovery stage;
otherwise investigate the earlier learned behavior. Do not blindly add training.

At each completed batch, use numeric gates and video to choose the cheapest
discriminating next experiment, freeze hypothesis/budget/commands/gates, test
changed logic, commit and launch locally. Resolve the seed22 Orbax rename
PermissionError before further training; preserve interrupted stage artifacts.
Do not claim parameter warm-start restores optimizer/PRNG training state.
Once a development recipe passes all required tests, freeze a new validation
study with three predetermined fresh training seeds and fresh held-out tests.
All three must pass; failures return to development, never disappear.

Hourly model wakes read only compact status while healthy. No active waiting,
no full logs except relevant errors, no hardware or remote pushes. Notify only
meaningful findings, errors requiring attention, and verified final success.
