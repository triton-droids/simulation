import jax
import jax.numpy as jp
import numpy as np
from source.utils.g1_phase_diagnostic import replace_policy_phase


def test_phase_intervention_is_local_and_nonmutating():
    obs={"state":jp.arange(103,dtype=jp.float32),"privileged_state":jp.arange(216,dtype=jp.float32)}
    assert replace_policy_phase(obs,None) is obs
    for angle in [0.,np.pi/2]:
        changed=jax.jit(lambda o:replace_policy_phase(o,angle))(obs)
        expected=[np.cos(angle),np.cos(angle+np.pi),np.sin(angle),np.sin(angle+np.pi)]
        for key in obs:
            np.testing.assert_allclose(changed[key][99:103],expected,atol=2e-7)
            np.testing.assert_array_equal(changed[key][:99],obs[key][:99])
            np.testing.assert_array_equal(changed[key][103:],obs[key][103:])
            np.testing.assert_array_equal(obs[key],np.arange(len(obs[key])))


def test_double_stance_encodes_both_legs_at_pi_without_other_changes():
    obs={"state":jp.arange(103,dtype=jp.float32),"privileged_state":jp.arange(216,dtype=jp.float32)}
    out=jax.jit(lambda o:replace_policy_phase(o,None,True))(obs)
    for k in obs:
        np.testing.assert_allclose(out[k][99:103],[-1,-1,0,0],atol=2e-7)
        np.testing.assert_array_equal(out[k][:99],obs[k][:99])
        np.testing.assert_array_equal(out[k][103:],obs[k][103:])
        np.testing.assert_array_equal(obs[k],np.arange(len(obs[k])))
