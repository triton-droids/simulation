import pytest
from robocub.soccer.robots.importer import require_robot,inspect_asset
from robocub.soccer.field.geometry import ROOT


def test_real_never_silently_substituted():
    with pytest.raises(ValueError,match='requires'):require_robot('real')
    assert 'NOT TRITON' in require_robot('demo')['name']


def test_available_g1_identity_is_not_team_confirmation():
    p=ROOT/'.cache/mujoco_menagerie/unitree_g1/g1_mjx.xml'
    if not p.exists():pytest.skip('Optional external G1 cache absent')
    info=inspect_asset(p)
    assert 'g1' in info['name'];assert info['licenses'];assert not info['hardware_identity_verified']
    assert all(x['exists'] for x in info['dependencies'])
    assert len(info['actuators'])==29
