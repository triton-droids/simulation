# Frozen F1 independent replication

Authorized 2026-09-21: repeat the successful development recipe with fresh
training seeds 11, 22, 33. No seed selection, replacement, or automatic tuning.
C22 learning rate 1e-4 passed all 8 nominal and 16 randomized development
rollouts and both reviewed gait videos. The 3e-5 candidate failed randomized
survival. This is development evidence, not independent replication.

Each seed starts from scratch and follows balance, gait, yaw, commands,
linear tracking, and randomized recovery stages. Warm starts stay within
that seed. Final recovery uses learning rate 1e-4. Exact commands, budgets,
restore audits and gates are frozen in queues/final_f1.json and
queues/final_f1_protocol.json. Total: 10,393,600 steps per seed; 31,180,800
across all three. Sequential RTX 4060 CUDA execution; 24-hour queue cap is
a safety limit, not a completion estimate.

Final checkpoints only. Eight fresh command vectors and reset seeds
6000/6001/6002 are assessed in nominal and randomized regimes, with true
untrained initialization and standing controls: 432 episodes total.
Existing absolute survival, height, tracking and gait gates remain.
Each seed/regime additionally requires at least 20% lower mean linear
tracking RMSE and greater mean survival than both matched controls.
Report every seed and mean/sample standard deviation across seeds.
Final success requires all numeric gates and visual confirmation of upright
alternating gait in all six representative trained videos.

Python handles all stages, audits, evaluations, videos and report without
model calls. Four-hour scheduled checks read compact state and stop quietly
while healthy work runs. One final substantive review determines pass/fail;
a failure does not authorize another tuning campaign. Execution faults may
be repaired without repeating completed training; preserve all evidence.

Keep the machine and WSL running, and Codex open for scheduled follow-up.
