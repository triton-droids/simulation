'''
Loads the humanoid into a MuJoCo visualizer for debugging purposes
'''

import mujoco
import mujoco.viewer

mjcf_path = "robot_description/test.xml"

def load_model(mjcf_path = mjcf_path):
    """ Loads mjcf model into Mujoco visualizer """

    # Loads model from MJCF into Mujoco 
    model = mujoco.MjModel.from_xml_path(mjcf_path)

    data = mujoco.MjData(model)
    
    # Initialize with default XML values
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    
    print(f"Model stats: {model.nbody} bodies, {model.njnt} joints, "
                f"{model.nv} DOF, {model.nu} actuators, {model.nmesh} meshes")
    
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    load_model()