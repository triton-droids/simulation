# Triton Droids RoboCup

## Open the GUI

Right-click **Scripts/dashboard.py** in PyCharm and choose Run, or select
**6 RoboCup Dashboard** in the run dropdown. No desktop shortcut or installed app.

```powershell
.venv\Scripts\python.exe Scripts\dashboard.py
```

The panel provides compute checks, saved progress, policies, recordings and a saved
robot viewer. Opening it does not resume training. **Training and scheduled reviews
are paused; resume only when explicitly requested.**

| Goal | Run or read |
|---|---|
| Watch the saved T02 walker | `Scripts/visualize.py` |
| See the selected weights | `Scripts/status.py` |
| Check installation | `Scripts/robocub_cli.py doctor` |
| Browse policies | [Policies](Policies/README.md) |
| Browse recordings | [Video library](Artifacts/Videos/README.md) |
| Understand controls | [Dashboard guide](documentation/guides/DASHBOARD.md) |
| Current research status | [CURRENT](documentation/research/CURRENT.md) |

## Folder map

```text
Scripts/                  Things you run directly
  dashboard.py            GUI entry point
  examples/               Small controller demonstrations
  legacy/                 Historical remote shell workflows
Configs/                  User-facing configuration
Policies/                 One folder per named policy; T02 includes actual weights
Artifacts/                Training outputs, videos and research evidence
documentation/            One documentation index, guides and research archive
robocub/                  Implementation; normally not edited by operators
tests/                    Regression checks
```

Read [the documentation index](documentation/README.md) for setup, training and
parameter explanations. Example training/sweep scripts are not the approved research
campaign. Selecting laptop or Atlantis affects a future launch; it never migrates
active work. An empty approved-plan list is valid.

The T02 scene demonstrates simulated locomotion, not kicking, soccer strategy,
hardware or Isaac transfer. A demo, reward leaderboard or one successful seed is
not three-seed certification. Every original numeric/control/visual gate remains.

`.cache/` holds downloaded assets and the WSL GPU runtime; moving it breaks existing
environment paths. `.run/` is PyCharm's required shared-run-configuration directory.
Both are excluded from the normal IDE project tree. Turn off **Show Excluded Files**
if your IDE displays them. Pytest temporary files now live under `.cache/pytest/`;
old temporary files are preserved under `.cache/test-history/`.

## Why these files stay at the root

| File | Purpose |
|---|---|
| `README.md` | Main entry point for the project |
| `.gitignore` | Keeps caches, credentials and generated data out of Git |
| `.env` | Private local settings used by existing tools; do not commit |
| `pytest.ini` | Automatic test discovery and cache configuration |
| `LICENSE`, `NOTICE` | License terms and attribution |

Python dependency lists are organized under
[Configs/environments](Configs/environments/README.md). No root `START_HERE.md`
is needed: its instructions were merged here and its old text is archived.

### Quick checkpoint workflow

In **Policies & saved checkpoints**, choose a named policy, select its timestamped checkpoint, click **Load checkpoint**, then **Watch loaded checkpoint**. The viewer uses that exact saved policy. **Prepare training selection** saves a draft only; training remains paused. Research IDs and original checkpoint files remain unchanged.
