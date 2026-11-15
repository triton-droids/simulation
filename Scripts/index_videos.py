"""Refresh the recording library without moving or copying experiment videos."""
from pathlib import Path
import html
import json
import os
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


def build_index(root=ROOT):
    output = root / 'Artifacts/Videos'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for base in ('Artifacts', 'Policies'):
        for folder, dirs, files in os.walk(root/base, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in {'.git', '.cache', '.venv', '__pycache__', 'node_modules', 'source', 'src', 'snapshots'} and not Path(folder,d).is_symlink())
            for name in sorted(files):
                path = Path(folder,name)
                if path.suffix.lower() in {'.mp4','.webm','.avi','.mov','.mkv'}:
                    records.append({'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size})
    records.sort(key=lambda r:r['path'])
    (output/'index.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    rows = ''.join('<li><a href="'+quote(os.path.relpath(root/r['path'],output).replace('\\','/'),safe='/')+'">'+html.escape(r['path'])+'</a> ('+str(r['bytes'])+' bytes)</li>' for r in records)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>RoboCup recordings</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto}li{margin:10px 0}</style><h1>RoboCup recordings</h1><p>Original recordings remain with their runs. Names identify their source; inclusion is not validation or policy promotion.</p><p>'+str(len(records))+' recordings. Refresh with Scripts/index_videos.py.</p><ul>'+rows+'</ul>',encoding='utf-8')
    return records


if __name__ == '__main__':
    print(f'Indexed {len(build_index())} recordings in Artifacts/Videos/index.html. Originals unchanged.')
