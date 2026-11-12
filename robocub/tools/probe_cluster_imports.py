import sys,json,hashlib,base64,importlib.metadata as md,traceback
from pathlib import Path
out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True)
errors=[]
def trace(frame,event,arg):
 if event=='exception' and '/trimesh/path' in frame.f_code.co_filename:
  typ,value,tb=arg
  errors.append({'file':frame.f_code.co_filename,'line':frame.f_lineno,'type':typ.__name__,'message':str(value)})
 return trace
def integrity():
 d=md.distribution('trimesh');rows={}
 for f in d.files or []:
  if str(f).startswith('trimesh/') and str(f).endswith('.py'):
   p=d.locate_file(f);raw=p.read_bytes();h=hashlib.sha256(raw).digest()
   if f.hash and f.hash.mode=='sha256':assert base64.urlsafe_b64encode(h).rstrip(b'=').decode()==f.hash.value,str(f)
   rows[str(f)]=h.hex()
 return rows
report={'full_validation':False,'training_steps':0}
try:
 report['before']=integrity();sys.settrace(trace)
 from robocub.locomotion import get_env_class
 import trimesh
 from mujoco import mjx
 import brax
 sys.settrace(None)
 report['after']=integrity();assert report['before']==report['after']
 report['versions']={n:md.version(n) for n in ['trimesh','numpy','scipy','mujoco','brax','jax','jaxlib']}
 report['ok']=True
except BaseException:
 sys.settrace(None);report['ok']=False;report['error']=traceback.format_exc()
finally:
 report['caught_path_exceptions']=errors;out.write_text(json.dumps(report,indent=2))
if not report['ok']:raise RuntimeError(report['error'])
