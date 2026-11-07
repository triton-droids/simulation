# Hyperparameters to Test

This folder is the only place a normal user needs to edit to define a sweep.

- `training_hyperparameters.json` contains PPO/trainer settings.
- `environment_hyperparameters.json` contains MJX G1 world/task settings.

Each file is a JSON list. Each object is one bundle of values. The sweep tries
every training object against every environment object. Two rows in each file
therefore produce `2 × 2 = 4` training runs.

```powershell
.venv\Scripts\python.exe Scripts\sweep.py --smoke
```

Trainer keys normally begin with `agent.` and environment keys with `sim.`. An
empty object `{}` means “use the selected profile's defaults for this side.”

These files control the MJX locomotion task. `App/Configs/field01.yaml` configures
the separate Isaac Lab field infrastructure and is not silently swept because
the G1 training policy is not integrated with that runtime yet.
