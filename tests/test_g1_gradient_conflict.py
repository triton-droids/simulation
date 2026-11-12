import functools
import numpy as np
import pytest
from robocub.scripts.audit_g1_gradient_conflict import compatibility


def test_direction_and_degeneracy():
    a=np.array([1.,2.,3.])
    assert compatibility(a,a)['cosine']==pytest.approx(1.)
    assert compatibility(a,-a)['cosine']==pytest.approx(-1.)
    with pytest.raises(ValueError): compatibility(a,np.zeros(3))
    with pytest.raises(ValueError): compatibility(a,np.array([np.nan,0,0]))


def test_real_ppo_policy_gradient_without_update():
    jax=pytest.importorskip('jax')
    import jax.numpy as jp
    from brax.training import types
    from brax.training.acme import running_statistics as rs
    from brax.training.agents.ppo import networks
    from robocub.scripts.audit_g1_gradient_conflict import policy_gradient
    net=networks.make_ppo_networks({'state':(4,), 'privileged_state':(4,)},2,
        policy_hidden_layer_sizes=(8,),value_hidden_layer_sizes=(8,),
        policy_obs_key='state',value_obs_key='privileged_state',preprocess_observations_fn=rs.normalize)
    norm=rs.init_state({'state':jp.zeros(4),'privileged_state':jp.zeros(4)})
    actor=net.policy_network.init(jax.random.PRNGKey(1));critic=net.value_network.init(jax.random.PRNGKey(2))
    before=[np.asarray(x).copy() for x in jax.tree.leaves(actor)]
    x=jax.random.normal(jax.random.PRNGKey(3),(4,20,4));obs={'state':x,'privileged_state':x}
    policy=networks.make_inference_fn(net)((norm,actor,critic),deterministic=False)
    action,extras=policy(obs,jax.random.PRNGKey(4))
    data=types.Transition(observation=obs,action=action,reward=x[...,0],discount=jp.ones((4,20)),
        next_observation=obs,extras={'policy_extras':extras,'state_extras':{'truncation':jp.zeros((4,20))}})
    f=jax.jit(functools.partial(policy_gradient,net,norm,actor,critic,discount=.97,gae=.95,clip=.1))
    loss,g=f(data=data,key=jax.random.PRNGKey(5))
    assert np.isfinite(float(loss)) and np.isfinite(g).all()
    assert compatibility(g,g)['cosine']==pytest.approx(1.,abs=1e-6)
    assert all(np.array_equal(a,np.asarray(b)) for a,b in zip(before,jax.tree.leaves(actor)))

    from brax.training.agents.ppo.losses import PPONetworkParams
    from robocub.scripts.audit_g1_gradient_budget import gradient_budget
    cfg={'entropy_cost':.005,'discounting':.97,'reward_scaling':1.,'gae_lambda':.95,'clipping_epsilon':.1,'max_grad_norm':1.}
    budget=jax.jit(functools.partial(gradient_budget,net,PPONetworkParams(actor,critic),norm,config=cfg))(batch=data,key=jax.random.PRNGKey(5))
    b={k:float(v) for k,v in budget.items()}
    assert all(np.isfinite(v) for v in b.values())
    assert b['total_norm']==pytest.approx(np.hypot(b['actor_norm'],b['critic_norm']),rel=1e-5)
    assert b['shared_clip_multiplier']==pytest.approx(min(1.,1./b['total_norm']))
    assert b['policy_loss']==pytest.approx(float(loss),abs=1e-6)
