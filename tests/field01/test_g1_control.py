import numpy as np
import pytest
from robocub.soccer.envs.task import SoccerEnv
from robocub.soccer.field.geometry import ROOT

@pytest.mark.skipif(not (ROOT/'.cache/mujoco_menagerie/unitree_g1/g1_mjx.xml').exists(),reason='Pinned G1 assets absent')
def test_g1_identity_position_control_and_contacts():
    e=SoccerEnv(robot='g1',preset='disabled')
    m=e.models[0]
    assert e.n_actions==29 and all(e.position_actuators)
    assert e.effort==139
    assert m.npair>=12
    action=e.scripted_action();e.step(action)
    np.testing.assert_allclose(e.data[0].ctrl,e.nominal+action[0])
    assert np.any(m.jnt_actfrcrange[:,1]>0)
    e.close()
