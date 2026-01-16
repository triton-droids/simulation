'''
Loads the humanoid into a MuJoCo visualizer for debugging purposes
Press 'Ctrl+R' in terminal to reload the model
'''

import mujoco
import mujoco.viewer
import time
import sys
import select
import termios
import tty

mjcf_path = "robot_description/test.xml"

def load_model_with_hotreload(mjcf_path=mjcf_path):
    """ Loads mjcf model into Mujoco visualizer with hot reload """
    
    print("=" * 60)
    print("MuJoCo Model Viewer with Hot Reload")
    print("=" * 60)
    print("Controls:")
    print("  - Close viewer window to reload model")
    print("  - Ctrl+C to exit")
    print("=" * 60)
    
    iteration = 0
    
    while True:
        iteration += 1
        
        try:
            # Load model
            print(f"\n[{iteration}] Loading model from {mjcf_path}...")
            model = mujoco.MjModel.from_xml_path(mjcf_path)
            data = mujoco.MjData(model)
            
            # Initialize with default XML values
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            
            print(f"✓ Model loaded successfully!")
            print(f"  Stats: {model.nbody} bodies, {model.njnt} joints, "
                  f"{model.nv} DOF, {model.nu} actuators, {model.nmesh} meshes")
            
            # Run diagnostics
            if model.nq >= 7:  # Has freejoint
                com = data.subtree_com[1] if len(data.subtree_com) > 1 else data.subtree_com[0]
                print(f"  Base height: {data.qpos[2]:.4f}m")
                print(f"  COM position: x={com[0]:.3f}, y={com[1]:.3f}, z={com[2]:.3f}")
            
            print("\n🚀 Launching viewer... (close window to reload)\n")
            
            # Launch viewer
            mujoco.viewer.launch(model, data)
            
            # If we get here, viewer was closed
            print("\n📝 Viewer closed. Reloading in 0.5 seconds...")
            time.sleep(0.5)
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            print("Fix the error and close/reopen the viewer to try again...")
            time.sleep(2)

if __name__ == "__main__":
    load_model_with_hotreload()