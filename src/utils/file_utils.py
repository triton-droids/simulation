import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_robot_file_path(robot_name: str, scene: str, suffix: str = ".urdf") -> str:
    """
    Dynamically finds the file path for a given robot name.

    This function searches for a .urdf file in the directory corresponding to the given robot name.
    It raises a FileNotFoundError if no URDF file is found.

    Args:
        robot_name: The name of the robot (e.g., 'robotis_op3').

    Returns:
        The file path to the robot's URDF file.

    Raises:
        FileNotFoundError: If no URDF file is found in the robot's directory.

    Example:
        robot_urdf_path = find_urdf_path("robotis_op3")
        print(robot_urdf_path)
    """
   
    robot_dir = os.path.join(project_dir, "robots", robot_name)
    if os.path.exists(robot_dir):
        file_path = os.path.join(robot_dir, robot_name + '_' + scene + '_scene' + suffix)
        if os.path.exists(file_path):
            return file_path

    raise FileNotFoundError(f"No {suffix} file found for robot '{robot_name}'.")