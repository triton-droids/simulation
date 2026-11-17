"""Read-only policy discovery. Browsing never loads JAX or changes training."""
from datetime import datetime, timezone
import json
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Policy path escapes the project')
    return path


def policies(root):
    entries = read(root / 'Policies/catalog.json')
    return entries if isinstance(entries, list) else []


def checkpoints(root, entry):
    run = inside(root, entry['run_dir'])
    rows = []
    for folder in (run / 'logs/checkpoints').glob('*'):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        weights = inside(root, str(folder / 'policy'))
        metadata = read(folder / '_CHECKPOINT_METADATA')
        stamp = metadata.get('commit_timestamp_nsecs')
        source = 'Checkpoint commit'
        if not isinstance(stamp, (int, float)) or stamp <= 0:
            stamp = weights.stat().st_mtime if weights.is_file() else folder.stat().st_mtime
            source = 'File modified (may be transfer time)'
        else:
            stamp /= 1e9
        step = int(folder.name)
        rows.append(dict(step=step, total=step + (entry.get('ancestor_steps') or 0),
                         saved=datetime.fromtimestamp(stamp, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
                         timestamp_source=source, run=run, weights=weights,
                         ready=weights.is_file() and (run/'resolved_config.json').is_file()))
    return sorted(rows, key=lambda row: row['step'], reverse=True)


def viewer_command(root, executable, row):
    run = inside(root, str(row['run']))
    weights = run/'logs/checkpoints'/str(int(row['step']))/'policy'
    if not weights.is_file() or not (run/'resolved_config.json').is_file():
        raise ValueError('Checkpoint weights or configuration are missing')
    return [executable, str(root/'robocub/tools/watch_g1_soccer.py'),
            '--run-dir', str(run), '--checkpoint', str(row['step'])]
