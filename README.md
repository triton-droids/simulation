# humanoid_legs_5 for Antioch

This is the small deployment branch for Triton Droids' lower-body linkage
inspection asset. Start with [Getting Started](notebooks/GETTING_STARTED.ipynb) to install the tools, sign in to Antioch, open the packaged USD in Isaac Sim.
The portable [MJCF and videos](robots/humanoid_legs_5/) are included with all
24 meshes. `validate_mjcf.py` checks the MuJoCo linkage locally.

The torso placeholder has been removed: the hip assembly is now the physical
floating root and the packaged USD has been regenerated. Isaac Lab/PhysX
fixed-root joint and linkage checks pass. A first flat-ground PPO task and
checkpoint evaluation are in `src/train_locomotion.py` and
`src/evaluate_locomotion.py`; they use the packaged USD directly and control
the 10 motors while leaving four linkage joints passive.

## Locomotion experiment

From the repository root in the Antioch project, run:

```bash
uv run antioch service exec -- python src/train_locomotion.py \
  --num-envs 256 --iterations 500 --steps-per-env 24 \
  --output outputs/locomotion_run
```

Evaluate a checkpoint on a fixed forward command (the current task assumes
body Y is forward based on mesh and joint-axis layout; verify this against the
CAD assembly before treating it as a robot convention):

```bash
uv run antioch service exec -- python src/evaluate_locomotion.py \
  --checkpoint outputs/locomotion_run/humanoid_legs_5_latest.pt \
  --num-envs 64 --duration 10 --vy 0.35 \
  --output outputs/locomotion_run/eval_forward.json
```

Record a streamed Isaac Sim preview from the checkpoint you trained:

```bash
uv run antioch service exec --stream -- python src/record_locomotion.py \
  --checkpoint outputs/locomotion_run/humanoid_legs_5_latest.pt \
  --seconds 8 --command-y 0.35 \
  --output robots/humanoid_legs_5/locomotion_preview.mp4
```

The branch includes the selected review candidate at
`robots/humanoid_legs_5/checkpoints/humanoid_legs_5_ppo_candidate_1200.pt`.
To replay its fixed-command evaluation, substitute that path for
`outputs/locomotion_run/humanoid_legs_5_latest.pt` in the evaluation command.
The included preview was recorded from that candidate.

The current checkpoint is **not a walking policy**. After a 500-iteration
fine-tune, a fixed 0.35 m/s body-Y request produced about 0.004 m/s mean
body-Y velocity over a five-second evaluation. Use it to reproduce the
experiment, not to claim locomotion success. See the [training
assessment](robots/humanoid_legs_5/locomotion_training_assessment.md), which
also lists the next model and contact fixes. The reviewed candidate checkpoint
and short preview are in the robot bundle; full training logs remain local.

See `robots/humanoid_legs_5/isaaclab_physx_check.json` for import, drive, and
linkage validation, plus remaining model calibration. Development history and
Fusion extraction tools remain on the `codex/hardware-tracking-retrain`
branch (source commit `1d01b69`).

Antioch is configured at this repository root. Run `uv sync --locked` here,
then follow the notebook. The robot meshes are in
`robots/humanoid_legs_5/meshes/`. Older CAD exports and scripts are preserved
locally in `local_archive/cad_exports/`. This archive is excluded from Git;
keep it out of Antioch uploads.
