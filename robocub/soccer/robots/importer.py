"""Explicit robot identity and dependency inventory; no implicit hardware substitution."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path

LABEL='DEMO/PROXY, NOT TRITON DROIDS HARDWARE (fixed-base joint fixture)'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def inspect_asset(path):
    p=Path(path).resolve()
    if not p.is_file():raise FileNotFoundError(f'Robot asset absent: {p}. Supply URDF/MJCF/USD plus meshes, license and actuator metadata.')
    result={'path':str(p),'sha256':sha(p),'format':p.suffix.lower(),'hardware_identity_verified':False}
    if p.suffix.lower() in ('.xml','.mjcf','.urdf'):
        root=ET.parse(p).getroot();result['name']=root.get('model',root.get('name','unnamed'))
        result['format']='urdf' if root.tag=='robot' else 'mjcf'
        result['joints']=[dict(e.attrib) for e in root.iter('joint') if e.get('name')]
        result['sensors']=[dict(e.attrib,kind=e.tag) for parent in root.iter('sensor') for e in parent]
        result['actuators']=[dict(e.attrib,kind=e.tag) for parent in root.iter('actuator') for e in parent]
        files=[];compiler=root.find('compiler');meshdir=compiler.get('meshdir','') if compiler is not None else ''
        for element in root.iter():
            value=element.get('file') or element.get('filename')
            if value:
                if value.startswith('package://'):files.append({'reference':value,'resolved':False});continue
                candidate=p.parent/(meshdir if element.tag=='mesh' else '')/value
                files.append({'reference':value,'path':str(candidate.resolve()),'exists':candidate.is_file(),'sha256':sha(candidate) if candidate.is_file() else None})
        result['dependencies']=files
    elif p.suffix.lower() not in ('.usd','.usda','.usdc'):
        raise ValueError('Expected URDF, MJCF/XML or USD')
    else:result['validation']='USD articulation validation requires Isaac importer runtime'
    licenses=list(p.parent.glob('*LICENSE*'))
    result['licenses']=[{'path':str(x),'sha256':sha(x)} for x in licenses]
    return result

def require_robot(mode,asset=None,metadata=None):
    if mode=='g1':
        from .g1 import g1_info
        return g1_info()
    if mode=='demo':return {'name':LABEL,'format':'procedural','hardware_identity_verified':False}
    if not asset:raise ValueError('--robot real requires --robot_asset PATH and --robot_metadata JSON with identity, units=m, license, initial_height, nominal_joints, kp, kd and effort_limit. Existing G1 assets are candidates, not confirmed team hardware.')
    info=inspect_asset(asset)
    if not metadata:raise ValueError('Real robot needs --robot_metadata; dimensions/actuator settings cannot be guessed.')
    meta=json.loads(Path(metadata).read_text())
    required={'identity','units','license','initial_height','nominal_joints','kp','kd','effort_limit'}
    if not required<=meta.keys() or meta.get('units')!='m':raise ValueError('Missing metadata fields or non-meter asset; see App/Configs/robots.example.yaml')
    for key in ('initial_height','kp','kd','effort_limit'):
        if not isinstance(meta[key],(int,float)) or not 0<meta[key]<100000:raise ValueError(f'Invalid {key}')
    info['operator_metadata']=meta
    return info

