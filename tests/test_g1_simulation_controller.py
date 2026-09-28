import pytest
from source.locomotion.unitree_g1.simulation_controller import validate_command,SimulationController

@pytest.mark.parametrize('command',[(0,0,0),(.45,0,0),(-.25,.25,.4)])
def test_valid_commands(command):
    assert validate_command(command)==command

@pytest.mark.parametrize('command',[(1,0,0),(0,0,float('nan')),(0,0),(0,0,0,0)])
def test_reject_invalid_commands(command):
    with pytest.raises(ValueError):validate_command(command)

def test_stop_changes_command_without_resetting_state():
    c=SimulationController.__new__(SimulationController);c.state=object();state=c.state
    c.set_command(.45,0,0);c.stop()
    assert c.command==(0,0,0) and c.state is state
