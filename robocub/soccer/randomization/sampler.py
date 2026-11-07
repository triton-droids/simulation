"""Order-independent per-environment sampling and fully replayable manifests."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import yaml
from robocub.soccer.field.geometry import ROOT,Field
from .registry import REGISTRY,ALWAYS,capabilities

SUITES={'train':42,'validation':12345,'nominal':12345,'heldout':54321}

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class Sampler:
    def __init__(self,field=None,seed=42,preset='mild',backend='isaaclab',rank=0,suite='train',config=None):
        self.field=field or Field();self.seed=int(seed);self.preset=preset;self.backend=backend;self.rank=int(rank);self.suite=suite
        self.config=config or yaml.safe_load((ROOT/'App/Configs/randomization_field01.yaml').read_text())
        if preset not in self.config['presets']:raise ValueError('Unknown preset')
        if suite not in SUITES:raise ValueError('Unknown suite')
        if preset=='stress' and suite!='heldout':raise ValueError('Stress is held-out only')
        if suite=='train' and seed in (12345,54321):raise ValueError('Reserved evaluation seed cannot be used for training')
        if not self.config['fixed_geometry'] or self.config['fixed_gravity'] != [0,0,-9.81]:raise ValueError('Geometry and Earth gravity must stay fixed')
        for name,(lo,hi) in self.config['ranges'].items():
            if name not in REGISTRY or not all(math.isfinite(x) for x in (lo,hi)) or lo>hi:raise ValueError('Invalid randomization range')
            if name not in ('com_offset',) and lo<0:raise ValueError('Negative physical range')
            if name in ('control_latency','observation_delay') and (lo!=int(lo) or hi!=int(hi) or hi>1):raise ValueError('Delay must be integer 0..1')
        nom=self.config['nominals']
        if not 0<=nom['dynamic_friction']<=nom['sliding_friction'] or nom['ball_mass']<=0:raise ValueError('Invalid nominal material or mass')
        if not all(math.isfinite(float(x)) for x in nom.values()):raise ValueError('Nonfinite nominal parameter')
        self.hash=digest({'config':self.config,'field':self.field.__dict__,'backend':backend,'preset':preset})

    def sample(self,env_id,episode):
        if env_id<0 or episode<0:raise ValueError('IDs must be nonnegative')
        key={'seed':self.seed,'rank':self.rank,'suite':self.suite,'env_id':int(env_id),'episode':int(episode),'config_hash':self.hash}
        rng=np.random.default_rng(int(digest(key)[:16],16))
        available=[k for k,v in capabilities(self.backend).items() if v['supported'] and k not in ALWAYS and (self.preset in ('robust','stress') or k not in ('external_push','ball_mass','joint_damping','observation_delay'))]
        low,high=self.config['presets'][self.preset]
        if low>len(available):raise ValueError(f'{self.backend}: {self.preset} needs {low} extra families but only {len(available)} are supported')
        count=int(rng.integers(low,min(high,len(available))+1));extra=sorted(rng.choice(available,count,replace=False).tolist())
        enabled=[] if self.preset=='disabled' else list(ALWAYS)+extra
        f=self.field
        robot=[-2.,0.,0.];ball=[0.,0.]
        if enabled:
            robot=[float(rng.uniform(-f.length/2+.7,f.length/2-.7)),float(rng.uniform(-f.width/2+.7,f.width/2-.7)),float(rng.uniform(-math.pi,math.pi))]
            for _ in range(1000):
                ball=[float(rng.uniform(-f.length/2+.3,f.length/2-.3)),float(rng.uniform(-f.width/2+.3,f.width/2-.3))]
                if np.linalg.norm(np.array(ball)-robot[:2])>1.:break
            else:raise ValueError('Cannot find nonoverlapping spawn')
        values={'robot_spawn_xy':robot[:2],'robot_spawn_yaw':robot[2],'ball_spawn_xy':ball}
        for name in extra:
            lo,hi=self.config['ranges'][name]
            if self.preset=='stress' and name not in ('control_latency','observation_delay'):
                mid=(lo+hi)/2;half=(hi-lo)/2*self.config['stress_range_scale'];lo=max(0,mid-half);hi=mid+half
            values[name]=int(rng.integers(lo,hi+1)) if name in ('control_latency','observation_delay') else float(rng.uniform(lo,hi))
        if 'ball_velocity' in values:
            a=rng.uniform(-math.pi,math.pi);values['ball_velocity']=[values['ball_velocity']*math.cos(a),values['ball_velocity']*math.sin(a)]
        # Materials stay nominal until a backend supports BOTH coefficients coherently.
        nominal=self.config['nominals'];material={'static':nominal['sliding_friction'],'dynamic':min(nominal['dynamic_friction'],nominal['sliding_friction'])}
        return {**key,'preset':self.preset,'backend':self.backend,'enabled':enabled,'values':values,'nominal_material':material,'unsupported':[k for k,v in capabilities(self.backend).items() if not v['supported']], 'trajectory_determinism':'CPU same stack tolerance only; GPU not bitwise guaranteed'}

    def observation_rng(self,record,step):
        return np.random.default_rng(int(digest({'record':record,'step':int(step),'stream':'observation'})[:16],16))

    def replay(self,record):
        restored=self.sample(record['env_id'],record['episode'])
        if restored!={k:v for k,v in record.items() if k!='effective'}:raise ValueError('Replay mismatch: seed/rank/suite/config/backend differ')
        return restored

