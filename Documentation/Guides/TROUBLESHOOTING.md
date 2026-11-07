# Troubleshooting

Start with `.venv\Scripts\python.exe robocub.py doctor`.

- **Text but no camera:** wait for compilation, then Alt-Tab to
  `MuJoCo : g1 scene`; PyCharm's Run panel is only the console.
- **Red filenames:** usually untracked Git files, not Python errors. Check
  `git status`; do not commit checkpoints just to remove the color.
- **Wrong green-button target:** select the desired run configuration, or
  right-click the open editor and choose Run.
- **Missing weights:** restore the local, Git-ignored file
  `App/Models/T02/logs/checkpoints/1003520/policy`.
- **Training only prints numbers:** expected; MJX training has no camera.
- **`import warp` fails:** use `.venv\Scripts\python.exe` and reinstall
  `requirements.txt`.
