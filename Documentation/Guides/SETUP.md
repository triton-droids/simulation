# Setup

## Fast path on Windows

From the repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe robocub.py doctor
```

The doctor command checks Python, the saved T02 artifacts, and every public
launcher. A complete installation prints `PASS` for every row.

## What gets downloaded

The Unitree G1 MuJoCo model comes from the public MuJoCo Menagerie pin recorded
in the repository. The local model cache is `.cache/mujoco_menagerie/` and is
not committed. Some Playground adapter workflows can also populate a local
robocub/cache directory.

## GPU reality

- Opening the MuJoCo viewer does not require a training GPU.
- The verified native Windows JAX path uses CPU on this host.
- Long useful PPO runs are intended for a CUDA-enabled JAX environment, such
  as WSL2 or a suitable remote machine.
- Access to any named external cluster is not configured by this repository.

## Warp

`warp-lang==1.17.0` is pinned in `requirements.txt` and imports as `warp`.
That does not install `mujoco_warp`. The currently pinned MuJoCo version is
`3.10.0`, while the checked `mujoco_warp` package requires a newer MuJoCo, so
those packages should not be mixed without a deliberate compatibility update.

## PyCharm

1. Set the project interpreter to `.venv\Scripts\python.exe`.
2. Select one of the checked-in numbered run configurations.
3. Use `1 Watch Robot` for the live camera or `2 Train Smoke` to verify PPO.

The green button runs the selected configuration, not necessarily the open
file. To run an arbitrary open Python file, right-click the editor and choose
**Run**, or create a temporary current-file configuration in PyCharm.
