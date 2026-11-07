import numpy as np
import pytest
from robocub.soccer.envs.task import SoccerEnv


def test_isolated_worlds_and_resets():
    e=SoccerEnv(2,preset='mild');a=e.data[1].qpos.copy();r=e.records[1].copy()
    e.reset([0]);assert np.array_equal(a,e.data[1].qpos);assert r==e.records[1]
    e.data[0].qpos[e.bq]=1000
    assert e.data[1].qpos[e.bq]!=1000
    assert e.models[0] is not e.models[1]


def test_cpu_same_stack_replay():
    a,b=SoccerEnv(preset='mild'),SoccerEnv(preset='mild')
    for _ in range(15):
        oa=a.step(a.scripted_action())[0];ob=b.step(b.scripted_action())[0]
        np.testing.assert_allclose(oa,ob,atol=1e-7)
    a.reset([0],episode=0);b.reset([0],episode=0)
    np.testing.assert_allclose(a.observe(),b.observe())


def test_action_and_reposition_guards():
    e=SoccerEnv(preset='disabled')
    with pytest.raises(ValueError):e.step([[float('nan'),0]])
    with pytest.raises(ValueError):e.reposition(ball=[100,0])
    with pytest.raises(ValueError):e.reposition(ball=[-2,0])
    e.reposition(ball=[1,1]);np.testing.assert_allclose(e.data[0].qpos[e.bq:e.bq+2],[1,1])


def test_effective_mass_and_damping():
    e=SoccerEnv(preset='robust')
    for episode in range(20):
        e.reset([0],episode=episode);r=e.records[0]
        assert e.models[0].body_mass[e.ball_id]==pytest.approx(.43*r['values'].get('ball_mass',1))
        assert r['effective']['joint_damping']==e.models[0].dof_damping[e.jv].tolist()
