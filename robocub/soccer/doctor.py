from pathlib import Path
import json,platform,subprocess,sys,importlib.util,importlib.metadata
from robocub.soccer.field.geometry import ROOT
from robocub.soccer.robots.importer import inspect_asset

def doctor():
    candidates=[]
    for base in [ROOT/'source',ROOT/'assets',ROOT/'.cache/mujoco_menagerie/unitree_g1']:
        if base.exists():
            for p in base.rglob('*'):
                if p.suffix in ('.urdf','.mjcf','.usd','.usda') or (p.name.startswith('g1') and p.suffix=='.xml'):
                    try:candidates.append(inspect_asset(p))
                    except Exception as e:candidates.append({'path':str(p),'error':str(e)})
    try:gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,uuid','--format=csv,noheader'],text=True).strip()
    except Exception as e:gpu=str(e)
    deps={}
    for m in ['isaaclab','newton','warp-lang','torch','mujoco','numpy','PyYAML']:
        try:deps[m]=importlib.metadata.version(m)
        except importlib.metadata.PackageNotFoundError:deps[m]=None
    blockers=[]
    if deps['isaaclab']!='3.0.0':blockers.append('Default backend requires isaaclab==3.0.0 in an isolated Python3.12 environment.')
    cuda=None
    try:
        import torch
        cuda={'available':torch.cuda.is_available(),'runtime':torch.version.cuda}
    except ImportError:pass
    if not cuda or not cuda['available']:blockers.append('CUDA PyTorch unavailable in this interpreter; isolated cu128 environment is required.')
    blockers.append('G1 selected for simulation; exact physical hardware revision and calibrated contact parameters remain unverified.')
    return {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'gpu':gpu,'dependencies':deps,'cuda':cuda,'default_backend':'isaaclab','candidates':candidates,'blockers':blockers,'demo':'Procedural fixed-base joint fixture, not humanoid hardware','output_writable':ROOT.exists()}
