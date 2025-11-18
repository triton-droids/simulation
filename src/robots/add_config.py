import os
import json
import argparse
import xml.etree.ElementTree as ET
import mujoco
import tempfile

def print_mj_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)

    with tempfile.NamedTemporaryFile(mode='r+', delete=True) as tmpfile:
    # tmpfile.name is the path to pass to the C function
        mujoco.mj_printModel(model, tmpfile.name)
        tmpfile.seek(0)
        mj_model_str = tmpfile.read()
    return mj_model_str

def parse_mj_model(xml_path: str):
    relevant_sections = set(['BODY', 'JOINT', 'DOF', 'GEOM', 'SITE', 'CAMERA', 'ACTUATOR', 'SENSOR'])
    result = {section: {} for section in relevant_sections}

    mj_model_str = print_mj_model(xml_path)
    lines = mj_model_str.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue

        current_section = line.split()[0]
        if current_section in relevant_sections:
            result[current_section][line[:-1]] = {}
            i += 1

            # Iterate through the section's lines
            while i < len(lines) and lines[i] and lines[i].split()[0] not in relevant_sections:
                section_name = lines[i].split()[0]
                result[current_section][line[:-1]][section_name] = ' '.join(lines[i].split()[1:])
                i += 1
        else:
            i += 1

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a config to a robot")
    parser.add_argument("--robot", type=str, required=True, help="The name of the robot. Must match the name in robots")
    args = parser.parse_args()

    robot_dir = os.path.join("src", "robots", args.robot)
    robot_xml = os.path.join(robot_dir, args.robot + ".xml")

    mj_model = parse_mj_model(robot_xml)

    os.makedirs(robot_dir, exist_ok=True)
    file_path = os.path.join(robot_dir, "mj_model.json")
    with open(file_path, "w") as f:     
        json.dump(mj_model, f, indent=4)

    print(f"Mujoco model saved to {file_path}")
