# Task 1: Scene Construction

# Setup Code / Packages
# Follow the instructions to download MuJoCo and setup your environment [here](../docs/setup.md).

# Import Packages
import mujoco as mj
import mediapy
import os
from utils import render


# Add the desired objects into the xml file. Fix the camera viewing angle inside of the xml to get a good view of the scene.
xml_file = os.path.abspath("../assets/descriptions/DropCubeInBinEnv.xml")
render(xml_file, camera_name="fixed")