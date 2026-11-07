"""Reproducible local evidence manifest with dirty source hashes."""
from pathlib import Path
import hashlib,json,subprocess
from robocub.soccer.field.geometry import ROOT

def source_manifest():
    files={}
    for directory in ('robocub/soccer','App/Configs/field_profiles'):
        for p in sorted((ROOT/directory).rglob('*')):
            if p.is_file() and p.suffix in ('.py','.yaml','.json'):files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ('App/Configs/field01.yaml','App/Configs/randomization_field01.yaml'):
        p=ROOT/name;files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return {'files':files,'tree_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
