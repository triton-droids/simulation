"""Configuration adapter for the existing URKL Holosoma checkout."""
from pathlib import Path
import json
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
HOLOSOMA = BASE.parent.parent / 'URKL-Simulations/motion_retargeting/holosoma/src/holosoma_retargeting/holosoma_retargeting'
sys.path.insert(0, str(HOLOSOMA.parent))


def make_config(data_path, task_name, save_dir, demo_joints=None):
    from holosoma_retargeting.config_types.robot import RobotConfig
    from holosoma_retargeting.config_types.data_type import MotionDataConfig
    from holosoma_retargeting.config_types.retargeting import RetargetingConfig
    from holosoma_retargeting.config_types.retargeter import RetargeterConfig
    mapping = json.loads((BASE / 'holosoma_lower_body_mapping.json').read_text())
    defaults = {'triton_legs': {'robot_dof': 10, 'robot_height': 0.7, 'object_name': 'ground'}}
    robot = RobotConfig(robot_type='triton_legs', robot_defaults=defaults,
        robot_name='triton_legs', robot_urdf_file=str(BASE / 'chrobot_holosoma_landmarks.xml'),
        foot_sticking_links=mapping['foot_sticking_links'],
        manual_lb={str(i): -1.0 for i in range(3, 7)},
        manual_ub={str(i): 1.0 for i in range(3, 7)},
        manual_cost={}, nominal_tracking_indices=np.arange(7, 17))
    return RetargetingConfig(task_type='robot_only', robot='triton_legs',
        data_format='kimodo', data_path=Path(data_path), task_name=task_name,
        save_dir=Path(save_dir), robot_config=robot,
        motion_data_config=MotionDataConfig(data_format='kimodo', robot_type='triton_legs',
            robot_defaults=defaults, joints_mapping=mapping['joints_mapping'], demo_joints=demo_joints),
        retargeter=RetargeterConfig(visualize=False))
