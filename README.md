# Triton Droids RoboCup Simulator

## Run Unitree G1

1. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Open this file:

   ```text
   scripts/load_unitree_g1.py
   ```

3. Press Run.

The first run downloads the pinned public Unitree G1 model automatically into a
hidden local cache. MuJoCo should open and display the robot.

## Research status

The separate G1 environment, deterministic MJX smoke path, local PPO workflow,
and three-seed Gate 4 evaluation are implemented. The frozen Gate 4 family did
**not** produce a verified locomotion baseline: yaw tracking was worse than both
controls and every held-out rollout fell. See
[`research/RESULTS.md`](research/RESULTS.md) for exact metrics, artifacts,
limitations, and the recommendation to stop before HOMIE Phases 5-7.

## PyCharm

1. Open the repository.
2. Select or create a Python interpreter.
3. Run:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Open `scripts/load_unitree_g1.py`.
5. Right-click the file and select Run, or press the green Run button.

No committed PyCharm project settings are required.

## VS Code

1. Open the repository.
2. Select a Python interpreter.
3. Run:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Open `scripts/load_unitree_g1.py`.
5. Use the standard Run Python File button.

No custom launch configuration is required.

## Command Line

Open the viewer:

```bash
python scripts/load_unitree_g1.py
```

Validate without opening the viewer:

```bash
python scripts/load_unitree_g1.py --no-viewer
```

Validate the MJX-compatible training scene and deterministic G1 control path:

```bash
python scripts/load_unitree_g1.py --mjx-scene --no-viewer
python source/scripts/smoke_g1.py --steps 1000 --seed 7 --no-fetch-model
```

Show options:

```bash
python scripts/load_unitree_g1.py --help
```

Create a clean upload zip:

```bash
python scripts/export_repo_zip.py
```

If an IDE run window closes before you can read it, run:

```bash
python scripts/export_repo_zip.py --pause
```

The zip is written to:

```text
exports/simulation_robocup_export.zip
```

## Project Layout

```text
scripts/load_unitree_g1.py  Press Run here to load Unitree G1
source/scripts/smoke_g1.py  Deterministic G1 reset/step validation
source/scripts/train.py     Local-first PPO training entry point
source/scripts/evaluate_g1.py  Fixed-command G1 policy evaluation
scripts/export_repo_zip.py  Create a clean upload zip
source/                     Simulator and training internals
research/                   Source ledger, decisions, experiment plan, results
requirements.txt            Python dependencies
README.md                   Setup and basic instructions
```

Local generated folders such as `.cache/`, `.venv/`, `venv/`, `exports/`,
`outputs/`, `results/`, and `__pycache__/` are ignored by Git.

## Model Source

The loader uses the public Unitree G1 model from MuJoCo Menagerie:

```text
https://github.com/google-deepmind/mujoco_menagerie
```

Pinned commit:

```text
71f066ad0be9cd271f7ed58c030243ef157af9f4
```

Default scene:

```text
unitree_g1/scene.xml
```

The beginner viewer defaults to that full scene. G1 smoke, PPO training, and
policy evaluation use the MJX-compatible `unitree_g1/scene_mjx.xml` from the
same pinned checkout.

The downloaded model is stored locally at:

```text
.cache/mujoco_menagerie/
```

That cache is not committed and is excluded from export zips.

## Troubleshooting

- PyCharm says no interpreter:
  select or create a Python interpreter, then install `requirements.txt`.
- MuJoCo is not installed:
  run `python -m pip install -r requirements.txt` in the selected interpreter.
- The model download fails:
  install Git, or run `python scripts/load_unitree_g1.py --model path/to/scene.xml`.
- The viewer does not open:
  run `python scripts/load_unitree_g1.py --no-viewer`. If that works, the model
  loaded and the remaining issue is local graphics/OpenGL support.

## Advanced Development

The `source/` folder contains the simulator internals: config, locomotion,
rewards, robot definitions, MuJoCo utilities, MJX helpers, and training/playback
code. Unitree G1 is integrated as the separate
`source/locomotion/unitree_g1` task; the original 12-actuator
`default_humanoid_legs` environment remains a regression baseline.

Run the short, credential-free PPO integration profile from the repository
root:

```bash
python source/scripts/train.py --logger local --seed 0 env=unitree_g1 robot=unitree_g1 sim=unitree_g1 agent=ppo_g1_smoke hydra.run.dir=results/g1_ppo_smoke hydra.job.chdir=true
```

That command writes JSONL metrics and restorable checkpoints locally. W&B is
optional and is never required for basic tests or training. To evaluate its
final checkpoint on a fixed command/reset trace:

```bash
python source/scripts/evaluate_g1.py --run-dir results/g1_ppo_smoke --checkpoint 1024 --steps 100 --seeds 2000 --output-dir results/g1_ppo_smoke/evaluation/checkpoint_1024
```

Native Windows JAX uses CPU on the validated host. Useful long PPO runs use the
same commands from WSL2 with a CUDA-enabled JAX environment; the exact bounded
Gate 4 profile and results are recorded in `research/RESULTS.md` and
`research/EXPERIMENT_PLAN.md`.

Docker, ROS, Isaac Lab, GPU compute, and reinforcement learning are not required
to load Unitree G1.

## Attribution

This repository contains Triton Droids simulator code and code derived from
[toddlerbot](https://github.com/hshi74/toddlerbot). The Unitree G1 loading
workflow uses model assets from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), where
the `unitree_g1` model is provided under a BSD-3-Clause license.
