import copy
import numpy as np
import pytest
from robocub.soccer.randomization.sampler import Sampler
from robocub.soccer.randomization.registry import REGISTRY


def test_replay_and_reset_order_independence():
    s=Sampler(backend='mujoco');a=s.sample(7,2);s.sample(5,100)
    assert a==s.sample(7,2)==s.replay(a)
    assert a['values']!=s.sample(8,2)['values']
    assert a['values']!=Sampler(backend='mujoco',rank=1).sample(7,2)['values']
    assert len(REGISTRY)==20

@pytest.mark.parametrize('preset,count',[('disabled',(0,0)),('nominal',(3,3)),('mild',(5,7)),('robust',(8,11)),('stress',(10,12))])
def test_counts_and_legal_spawns(preset,count):
    s=Sampler(backend='mujoco',preset=preset,suite='heldout' if preset=='stress' else 'train')
    for i in range(100):
        r=s.sample(i,0);v=r['values'];assert count[0]<=len(r['enabled'])<=count[1]
        assert s.field.legal(*v['robot_spawn_xy']);assert s.field.legal(*v['ball_spawn_xy'],.11)
        assert np.linalg.norm(np.array(v['robot_spawn_xy'])-v['ball_spawn_xy'])>1
        assert r['nominal_material']['dynamic']<=r['nominal_material']['static']
        assert not set(r['enabled']) & set(r['unsupported'])


def test_no_evaluation_leakage_or_unsupported_sampling():
    with pytest.raises(ValueError):Sampler(seed=54321)
    with pytest.raises(ValueError):Sampler(preset='stress',suite='train')
    assert 8<=len(Sampler(backend='isaaclab',preset='robust').sample(0,0)['enabled'])<=10
    assert 5<=len(Sampler(backend='isaaclab').sample(0,0)['enabled'])<=7
    assert len(Sampler(backend='isaaclab',preset='stress',suite='heldout',seed=54321).sample(0,0)['enabled'])==10


def test_invalid_ranges_rejected():
    cfg=copy.deepcopy(Sampler().config);cfg['ranges']['ball_mass']=[-1,1]
    with pytest.raises(ValueError):Sampler(config=cfg)
    cfg=copy.deepcopy(Sampler().config);cfg['fixed_gravity']=[0,0,-1]
    with pytest.raises(ValueError):Sampler(config=cfg)
