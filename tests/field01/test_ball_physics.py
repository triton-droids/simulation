import numpy as np
import mujoco
from robocub.soccer.envs.task import SoccerEnv


def test_drop_and_roll_stability():
    e=SoccerEnv(preset='disabled');d=e.data[0];m=e.models[0]
    d.qpos[e.bq+2]=1.;mujoco.mj_forward(m,d)
    minimum=10
    for _ in range(400):
        mujoco.mj_step(m,d);minimum=min(minimum,d.qpos[e.bq+2])
    assert minimum>e.field.ball_radius-.02
    assert abs(d.qpos[e.bq+2]-e.field.ball_radius)<.01
    d.qvel[e.bv]=1
    start=d.qpos[e.bq]
    for _ in range(200):mujoco.mj_step(m,d)
    assert .1<d.qpos[e.bq]-start<1.2
    assert np.isfinite(d.qpos).all()


def test_goalpost_collision():
    e=SoccerEnv(preset='disabled');d=e.data[0];m=e.models[0]
    d.qpos[e.bq:e.bq+3]=[6.4,1.29,.2];d.qvel[e.bv]=2.;mujoco.mj_forward(m,d)
    hit=False;post=m.geom('post_1_1').id;ball=m.geom('ball').id
    for _ in range(150):
        mujoco.mj_step(m,d)
        hit |= any({d.contact[j].geom1,d.contact[j].geom2}=={post,ball} for j in range(d.ncon))
    assert hit
    assert d.qpos[e.bq]<7.04
    assert np.isfinite(d.qpos).all()
