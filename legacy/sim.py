'''
Loads the humanoid into a MuJoCo visualizer for debugging purposes
Press 'Ctrl+R' in terminal to reload the model

Usage:
    python sim.py       # Load scene with robot (default)
    python sim.py -h    # Load robot only (no scene)
'''

import mujoco
import mujoco.viewer
import time
import sys

# Define model paths
SCENE_PATH = "robot_description/scene.xml"      # Full scene with floor/lights
ROBOT_PATH = "robot_description/humanoid.xml"     # Robot only, no scene

def load_model_with_hotreload(mjcf_path, model_type="scene"):
    """ Loads mjcf model into Mujoco visualizer with hot reload """
    
    print("=" * 60)
    print(f"MuJoCo Model Viewer - Loading '{model_type}' model")
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
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "torso_top")
                if site_id != -1:
                    base_height = data.site_xpos[site_id, 2]
                else:
                    base_height = data.qpos[2]
                print(f"  Base height: {base_height:.4f}m")
                print(f"  COM position: x={com[0]:.3f}, y={com[1]:.3f}, z={com[2]:.3f}")
                print(f"  Model type: Mobile (with freejoint)")
            else:
                print(f"  Model type: Fixed base (no freejoint)")
            
            print("\n🚀 Launching viewer... (close window to reload)\n")
            
            # Launch viewer
            mujoco.viewer.launch(model, data)
            
            # If we get here, viewer was closed
            print("\n📝 Viewer closed. Reloading in 0.5 seconds...")
            time.sleep(0.5)
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except FileNotFoundError as e:
            print(f"\n❌ Error: Model file not found!")
            print(f"   Looking for: {mjcf_path}")
            print(f"   Make sure the file exists and the path is correct.")
            break
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            print("Fix the error and close/reopen the viewer to try again...")
            time.sleep(2)

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '-h':
        # Load humanoid/robot only
        mjcf_path = ROBOT_PATH
        model_type = 'humanoid'
    else:
        # Default: load scene
        mjcf_path = SCENE_PATH
        model_type = 'scene'
    
    # Load and run
    load_model_with_hotreload(mjcf_path, model_type)
