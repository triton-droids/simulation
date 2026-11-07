"""Capability registry: sampled does not mean applied. Never hide unsupported knobs."""
from dataclasses import dataclass,asdict

@dataclass(frozen=True)
class Family:
    timing: str
    scope: str
    constraint: str
    mujoco: bool = False
    isaaclab: bool = False

REGISTRY = {
 'robot_spawn_xy': Family('reset','env','legal, no ball overlap',True,True),
 'robot_spawn_yaw': Family('reset','env','[-pi,pi]',True,True),
 'ball_spawn_xy': Family('reset','env','legal, no robot overlap',True,True),
 'ball_velocity': Family('reset','env','bounded speed and angle',True,True),
 'foot_turf_static_friction': Family('startup/reset','env','requires separate verified static coefficient'),
 'foot_turf_dynamic_friction': Family('startup/reset','env','dynamic <= static; coupled material'),
 'ball_rolling_loss': Family('interval','env','explicit angular viscous drag proxy, not calibrated turf',True,True),
 'ball_restitution': Family('startup/reset','env','[0,1]; needs calibrated backend mapping'),
 'ball_mass': Family('reset','env','positive; inertia scales with mass',True,True),
 'motor_strength': Family('reset','env','positive effort scale',True),
 'actuator_gain': Family('reset','env','positive PD gain',True),
 'joint_damping': Family('reset','env','nonnegative',True),
 'control_latency': Family('reset/control','env','integer delayed actions 0..1',True,True),
 'proprio_noise': Family('observation','env','bounded rad/rad-s noise, engineering assumption',True,True),
 'observation_delay': Family('reset/observation','env','integer delay, reset clears history',True,True),
 'lighting': Family('startup','run','render only; unsupported until calibrated'),
 'camera_jitter': Family('startup','run','camera pipeline required'),
 'com_offset': Family('prestartup','env','positive inertia, validate dynamics first'),
 'external_push': Family('interval','env','bounded force, never hardware',True,True),
 'opponent': Family('reset','env','requires actual scripted opponent bank'),
}
ALWAYS=('robot_spawn_xy','robot_spawn_yaw','ball_spawn_xy')

def capabilities(backend):
    if backend not in ('mujoco','isaaclab'):raise ValueError('Unknown backend')
    return {k:dict(asdict(v),supported=getattr(v,backend)) for k,v in REGISTRY.items()}

