# Start here

## 1. Open the control panel

Run **Scripts/dashboard.py** from the project. No desktop shortcut or installed application is required.
In PyCharm choose **6 RoboCup Dashboard** in the run dropdown, then Run.
The selected configuration determines what starts, not the open editor file.

Terminal alternative from the project root:

```powershell
.venv\Scripts\python.exe Scripts\dashboard.py
```

If the interpreter is missing, follow [Setup](../../guides/SETUP.md).

## 2. Choose an action

- **Inspect training:** Current work & history, Refresh saved progress, Inspect
  selected and Open results. Saved state is not always live scheduler state.
- **Check compute:** choose laptop or Atlantis and click Check compute. This
  does not transfer an active job. Unknown/stale is not ready.
- **Watch the robot:** Watch robot on field opens the saved T02 demo, not the
  latest selected training checkpoint. Avoid extra simulation load during training.
- **Start training:** Start approved plan only accepts registered reviewed plans.
  An empty list means nothing new is registered; it is not a broken installation.
- **Stop:** Stop selected requests a queue/job stop, preserving saved checkpoints.
  It is not exact pause/resume. Serial research batches have a supervisor; consult
  the active campaign before stopping them.

See [Control panel](../../guides/DASHBOARD.md) for detailed button behavior.

## 3. Find your files

| Need | Location |
|---|---|
| Public launchers | `Scripts/` |
| Example trainer/world settings | `Configs/sweeps/` |
| Named policies and provenance | `Policies/` (references original weights) |
| Browse recordings | `Artifacts/Videos/index.html` |
| Ordinary training outputs | `Artifacts/Runs/` |
| Frozen research outputs | `Artifacts/Results/` |
| Parameter explanations | [Manual index](../../guides/README.md) |

The root `Scripts/` holds legacy remote utilities, not the GUI. Do not use its
sweep script for the active research campaign. No editing `robocub/` is needed
to open the panel. The field demo is locomotion, not a complete soccer player.
