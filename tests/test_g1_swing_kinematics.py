import numpy as np
from source.scripts.probe_g1_swing_kinematics import pitch_offsets

def test_fixed_primitive_is_symmetric_and_disabled_when_stopped():
    phases=np.linspace(0,2*np.pi,32,endpoint=False)
    a=pitch_offsets(np.stack((phases,phases+np.pi),axis=-1))
    b=pitch_offsets(np.stack((phases+np.pi,phases+2*np.pi),axis=-1))
    np.testing.assert_allclose(a[:,::-1],b,atol=1e-14)
    np.testing.assert_allclose(a.sum(axis=-1),0,atol=1e-14)
    np.testing.assert_array_equal(pitch_offsets(phases,False),np.zeros((32,3)))
    assert np.max(np.abs(a))<=.3
