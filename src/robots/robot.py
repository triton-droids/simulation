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
                "min": joint["jnt_range"].split(" ")[0],
                "max": joint["jnt_range"].split(" ")[1],
            }

        self.geoms = {}
        for geom in self.model_config["GEOM"].values():
            self.geoms[geom["name"]] = {
                "geom_bodyid": geom["geom_bodyid"],
            }
            
        self.bodies = {}
        body_data = {int(k.split()[1]): v for k, v in self.model_config["BODY"].items() 
                     if int(k.split()[1]) != 0}
        for id, body in body_data.items():
            self.bodies[body["name"]] = {
                "weld_id": body["body_weldid"],
                "parent_id": body["body_parentid"],
                "jnt_adr": body["body_jntadr"],
                "geom_adr": body["body_geomadr"],
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
        self.nu = len(self.actuators)
        self.body_to_joint_mapping = self.body_to_joint_mapping()

        self.default_joint_angles = self.model_config["default_joint_angles"]
        self.default_motor_ctrls = self.model_config["default_motor_ctrls"]
        
        # Create mapping between joint names and their initial angles based on qposadr
        self.init_joint_angles = {}
        for joint_name, joint_info in list(self.joints.items())[1:]:
            qpos_index = int(joint_info["qposadr"])
            if qpos_index < len(self.default_joint_angles):
                self.init_joint_angles[joint_name] = self.default_joint_angles[qpos_index]

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

    def body_to_joint_mapping(self):
        body_data = {int(k.split()[1]): v for k, v in self.model_config["BODY"].items() 
                     if int(k.split()[1]) != 0}
        joint_data = {int(k.split()[1]): v for k, v in self.model_config["JOINT"].items()}

        # Build body id to name and name to id mappings
        body_id_to_name = {bid: binfo['name'] for bid, binfo in body_data.items()}
        body_name_to_id = {v: k for k, v in body_id_to_name.items()}

        # Build parent-child relationship
        body_children = {}
        body_parents = {}
        for bid, binfo in body_data.items():
            pid = int(binfo['body_parentid'])
            body_children.setdefault(pid, []).append(bid)
            body_parents[bid] = pid

        # Map joints to the body they belong to
        joints_by_body = {}
        for jid, jinfo in joint_data.items():
            b_id = int(jinfo['jnt_bodyid'])
            joints_by_body.setdefault(b_id, []).append(jinfo['name'])

        # Recursive function to get all joints under a body (including sub-bodies)
        def collect_all_joints(body_id):
            joints = set(joints_by_body.get(body_id, []))
            for child_id in body_children.get(body_id, []):
                joints.update(collect_all_joints(child_id))
            return joints

        # Final mapping: body name → all joints inside it (including descendants)
        body_to_all_joints = {
            body_id_to_name[bid]: collect_all_joints(bid) for bid in body_id_to_name
        }

        return body_to_all_joints
            
