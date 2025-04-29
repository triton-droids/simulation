import json
import os
from typing import Any, Dict, List
from src.utils.math_utils import degrees_to_radians, radians_to_degrees

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class Robot:
    """ This class defines data structure of a robot"""

    def __init__(self, robot_name: str, config_path: str = None, xml_path: str = None):
        """Initalizes a robot with specified configurations and paths
        
        Args:
            robot_name (str): The name of the robot. Must match the name of the directory in the robots folder
        """
        self.name = robot_name

        self.root_path = os.path.join(project_root, "src", "robots", robot_name)
        self.model_config_path = os.path.join(self.root_path, "mj_model.json") if config_path is None else config_path
        self.xml_path = os.path.join(self.root_path, self.name + ".xml") if xml_path is None else xml_path
        
        with open(self.xml_path, "r") as f:
            self.xml = f.read()

        self.load_robot_config()
        self.initalize()

    def load_robot_config(self):
        """Load the robot's configuration and collision configuration from JSON files.

        Raises:
            FileNotFoundError: If the main configuration file or the collision configuration file does not exist at the specified paths.
        """
        if os.path.exists(self.model_config_path):
            with open(self.model_config_path, "r") as f:
                self.model_config = json.load(f)

        else:
            raise FileNotFoundError(f"No config file found for robot '{self.name}'.")

            
    def initalize(self):
        """Initialize the robot's configuration based on the loaded configuration data.
        
        Loads motor ordering, foot names, joint limits, and stores the names of geoms, bodies, and sensors.
        """

        self.joints = {}
        self.joint_limits = {}
        for joint in self.model_config["JOINT"].values():
            self.joints[joint["name"]] = {
                "qposadr": joint["jnt_qposadr"],
                "jnt_bodyid": joint["jnt_bodyid"],
            }
            self.joint_limits[joint["name"]] = {
                "min": joint["jnt_range"][0],
                "max": joint["jnt_range"][1],
            }

        self.geoms = {}
        for geom in self.model_config["GEOM"].values():
            self.geoms[geom["name"]] = {
                "geom_bodyid": geom["geom_bodyid"],
            }
            
        self.bodies = {}
        for id, body in self.model_config["BODY"].items():
            id = int(id.split(" ")[1])
            if body["name"] == "world":
                continue
            self.bodies[body["name"]] = {
                "body_id": id,
                "body_parentid": body["body_parentid"],
                "body_jntadr": body["body_jntadr"],
                "body_geom_adr": body["body_geomadr"],
            }
        
        self.sensors = {}
        for sensor in self.model_config["SENSOR"].values():
            self.sensors[sensor["name"]] = {
                "sensor_type": sensor["sensor_type"],
                "sensor_objtype": sensor["sensor_objtype"],
                "sensor_dim": sensor["sensor_dim"],
                "sensor_adr": sensor["sensor_adr"],
            }

        self.cameras = []
        for camera in self.model_config["CAMERA"].values():
            self.cameras.append(camera["name"])
        
        self.sites = {}
        for site in self.model_config["SITE"].values():
            self.sites[site["name"]] = {
                "site_bodyid": site["site_bodyid"],
                "site_type": site["site_type"],
            }

        self.actuators = {}
        for actuator in self.model_config["ACTUATOR"].values():
            self.actuators[actuator["name"]] = {
                "actuator_acc0": actuator["actuator_acc0"],
                "actuator_dyntype": actuator["actuator_dyntype"],
                "actuator_gainprm": actuator["actuator_gainprm"],
                "actuator_biastype": actuator["actuator_biastype"],
                "actuator_biasprm": actuator["actuator_biasprm"],
                "actuator_ctrlrange": actuator["actuator_ctrlrange"],
                "actuator_forcerange": actuator["actuator_forcerange"],
                
            }
        
    def get_joint_attrs(self, attr: str):
        """Get the attributes of a a model component.
        
        Args:
            key_name (str): The name of the key to search for.
            key_value (str): The value of the key to search for.
            attr (str): The attribute to get.
        """

        attrs = []
        for item in self.joints.values():
            attrs.append(item[attr].split(" "))
        return attrs
            
        # # Load foot name if specified
        # if "foot_name" in self.config["general"]:
        #     self.foot_name = self.config["general"]["foot_name"]

       

            
