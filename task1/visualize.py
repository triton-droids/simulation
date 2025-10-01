import mujoco as mj

#!/usr/bin/env python3
import time
import mujoco
import mujoco.viewer

def main():
    """Standalone script to spawn and view the hand model with default XML configuration."""
    
    # Load model & data
    # model = mujoco.MjModel.from_xml_path("meshes/lego_hand/lego_hand.xml")
    model = mujoco.MjModel.from_xml_path("../assets/descriptions/DropCubeInBinEnv.xml")
    data = mujoco.MjData(model)
    
    # Initialize with default XML values
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    print(f"Model loaded successfully!")
    print(f"Bodies: {model.nbody}")
    print(f"Joints: {model.njnt}")
    print(f"Actuators: {model.nu}")
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer launched. Press ESC to exit.")
        
        while viewer.is_running():
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Small delay to control simulation speed
            time.sleep(0.01)

if __name__ == "__main__":
    main()