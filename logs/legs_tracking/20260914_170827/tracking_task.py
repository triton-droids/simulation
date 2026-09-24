"""Lower-body BeyondMimic-style tracking baseline; explicit motion file required."""
from pathlib import Path
import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs.mdp.observations import projected_gravity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

BASE = Path(__file__).resolve().parent


def robot_spec():
    spec = mujoco.MjSpec.from_file(str(BASE / 'chrobot_16kg_actuated.xml'))
    spec.add_sensor(name='imu_ang_vel', type=mujoco.mjtSensor.mjSENS_GYRO,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname='imu')
    # Privileged critic only: a real IMU does not directly measure linear velocity.
    spec.add_sensor(name='imu_lin_vel', type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname='imu')
    return spec


def make_leg_tracking_cfg(motion_file):
    path = Path(motion_file).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {'robot': EntityCfg(
        spec_fn=robot_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0, 0, 0.71173)),
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=('.*_joint',)),),
            soft_joint_pos_limit_factor=0.95))}
    cfg.scene.sensors = (ContactSensorCfg(
        name='self_collision',
        primary=ContactMatch(mode='subtree', pattern='floating_base', entity='robot'),
        secondary=ContactMatch(mode='subtree', pattern='floating_base', entity='robot'),
        fields=('found', 'force'), reduce='none', num_slots=1, history_length=4),)
    motion = cfg.commands['motion']
    motion.motion_file = str(path)
    motion.anchor_body_name = 'hip'
    motion.body_names = ('floating_base', 'hip', 'left_leg1', 'left_leg2', 'left_leg3',
                         'left_leg4', 'left_foot', 'right_leg1', 'right_leg2',
                         'right_leg3', 'right_leg4', 'right_foot')
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.joint_position_range = (0.0, 0.0)
    cfg.actions['joint_pos'].scale = 0.2
    actor = cfg.observations['actor'].terms
    for name in ('motion_anchor_pos_b', 'motion_anchor_ori_b', 'base_lin_vel'):
        actor.pop(name)
    actor['gravity'] = ObservationTermCfg(func=projected_gravity)
    cfg.events.pop('push_robot')
    cfg.events.pop('base_com')
    cfg.events['foot_friction'].params['asset_cfg'].geom_names = (
        'left_foot_collision_box', 'right_foot_collision_box')
    cfg.terminations['ee_body_pos'].params['body_names'] = ('left_foot', 'right_foot')
    cfg.viewer.body_name = 'hip'
    return cfg
