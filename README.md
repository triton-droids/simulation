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
scripts/export_repo_zip.py  Create a clean upload zip
source/                     Simulator and training internals
requirements.txt            Python dependencies
README.md                   Setup and basic instructions
```

Local generated folders such as `.cache/`, `.venv/`, `venv/`, `exports/`, and
`__pycache__/` are ignored by Git.

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
code. Unitree G1 currently loads in MuJoCo but is not yet integrated with the
existing locomotion training environment.

Docker, ROS, Isaac Lab, GPU compute, and reinforcement learning are not required
to load Unitree G1.

## Attribution

This repository contains Triton Droids simulator code and code derived from
[toddlerbot](https://github.com/hshi74/toddlerbot). The Unitree G1 loading
workflow uses model assets from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), where
the `unitree_g1` model is provided under a BSD-3-Clause license.
