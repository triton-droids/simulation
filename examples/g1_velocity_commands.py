"""Run from repository root: python -m examples.g1_velocity_commands"""
from source.locomotion.unitree_g1.simulation_controller import SimulationController

def main():
    robot=SimulationController();robot.reset(seed=6000)
    for command in [(0.45,0,0),(0,0,0.4),(0,0,0)]:
        robot.set_command(*command)
        for _ in range(500):
            state=robot.step()
            if bool(state.done):
                print('Simulation terminated; no automatic reset.')
                return
if __name__=='__main__':main()
