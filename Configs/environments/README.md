# Python environment records

| File | Purpose |
|---|---|
| `requirements.txt` | Direct pinned dependencies for ordinary setup |
| `requirements-lock-windows.txt` | Historical fully resolved Windows CPU environment snapshot |

Install from the repository root using the configured interpreter:

```powershell
.venv\Scripts\python.exe -m pip install -r Configs/environments/requirements.txt
```

These files were relocated unchanged. This cleanup did not install packages or
modify the existing Windows/WSL environments. The historical lock is evidence,
not a claim that it exactly describes today's GPU runtime. Training remains paused.
