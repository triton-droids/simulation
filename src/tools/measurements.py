import mujoco
import numpy as np
from mujoco import viewer

# Load the MuJoCo model
model_path = "/home/anthony-roumi/Desktop/toddlerbot/toddlerbot/descriptions/toddlerbot/toddlerbot_scene.xml"  # Replace with your model path
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Create a simulation loop
def simulation_loop():
    # Initialize the viewer
    with viewer.launch_passive(model, data) as viewer_window:
        # Reset the simulation
        mujoco.mj_resetData(model, data)
        
        # To set the model to a keyframe position:
        # First check if the keyframe exists
        # keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        
        # if keyframe_id >= 0:
        #     # If keyframe exists, copy its data to qpos
        #     mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        #     print("Set pose to 'home' keyframe")
        # else:
        #     # If no keyframe found, just use the default position
        #     print("Keyframe 'home' not found, using default position")
        #     # The default position is already set by mj_resetData above
        
        # Forward kinematics to update all positions based on the joint values
        mujoco.mj_forward(model, data)
        
        # Get joint positions
        def get_joint_position(joint_name):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                print(f"Joint {joint_name} not found!")
                return None
            
            # Get the body ID that this joint belongs to
            body_id = model.jnt_bodyid[joint_id]
            
            # Get the body position in world coordinates
            body_pos = np.zeros(3)
            body_pos[:] = data.xpos[body_id]
            return body_pos
        
        # Get body positions
        def get_body_position(body_name):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                print(f"Body {body_name} not found!")
                return None
            
            body_pos = np.zeros(3)
            body_pos[:] = data.xpos[body_id]
            return body_pos
        
        
        # # Example of getting positions
        # left_hip_pos = get_joint_position("2xc430") 
        # right_hip_pos = get_joint_position("2xc430_2")
        
        # print(f"Left hip position: {left_hip_pos}")
        # print(f"Right hip position: {right_hip_pos}")
        
        # # Example of getting a body position
        left_foot_pos = get_body_position("ank_roll_link")
        right_foot_pos = get_body_position("ank_roll_link_2")
        
        print(f"Left hip position: {left_foot_pos}")
        print(f"Right hip position: {right_foot_pos}")
        
        # # Calculate the foot to com y distance (similar to your existing code)
        if left_foot_pos is not None and right_foot_pos is not None:
            foot_to_com_y = abs(left_foot_pos[1] - right_foot_pos[1]) / 2
            print(f"Foot to COM Y distance: {foot_to_com_y}")

            foot_com_x = abs(left_foot_pos[0] - right_foot_pos[0]) / 2
            print(f"Foot COM X distance: {foot_com_x}")

        
        
        # Run the simulation with visualization
        while viewer_window.is_running():
            # Step physics
            mujoco.mj_step(model, data)
            
            # Update viewer with new state
            viewer_window.sync()

# Run the simulation
if __name__ == "__main__":
    simulation_loop()