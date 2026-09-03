"""EngineAI T800 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


T800_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "engineai_t800" / "xmls" / "serial_t800.xml"
)
assert T800_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(T800_XML))


# Actuator config

ARMATURE_HIP_KNEE = 0.2427264
ARMATURE_HIP_ROLL = 0.14110848
ARMATURE_HIP_YAW = 0.0448737
ARMATURE_LIMB = 0.0354625
ARMATURE_SMALL = 0.00671625

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_HIP_KNEE = ARMATURE_HIP_KNEE * NATURAL_FREQ**2
STIFFNESS_HIP_ROLL = ARMATURE_HIP_ROLL * NATURAL_FREQ**2
STIFFNESS_HIP_YAW = ARMATURE_HIP_YAW * NATURAL_FREQ**2
STIFFNESS_LIMB = ARMATURE_LIMB * NATURAL_FREQ**2
STIFFNESS_SMALL = ARMATURE_SMALL * NATURAL_FREQ**2

DAMPING_HIP_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_HIP_KNEE * NATURAL_FREQ
DAMPING_HIP_ROLL = 2.0 * DAMPING_RATIO * ARMATURE_HIP_ROLL * NATURAL_FREQ
DAMPING_HIP_YAW = 2.0 * DAMPING_RATIO * ARMATURE_HIP_YAW * NATURAL_FREQ
DAMPING_LIMB = 2.0 * DAMPING_RATIO * ARMATURE_LIMB * NATURAL_FREQ
DAMPING_SMALL = 2.0 * DAMPING_RATIO * ARMATURE_SMALL * NATURAL_FREQ

T800_ACTUATOR_HIP_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*HIP_PITCH.*", ".*KNEE_PITCH.*"),
  stiffness=STIFFNESS_HIP_KNEE,
  damping=DAMPING_HIP_KNEE,
  effort_limit=415.0,
  armature=ARMATURE_HIP_KNEE,
)

T800_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*HIP_ROLL.*",),
  stiffness=STIFFNESS_HIP_ROLL,
  damping=DAMPING_HIP_ROLL,
  effort_limit=370.0,
  armature=ARMATURE_HIP_ROLL,
)

T800_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*HIP_YAW.*", "J12_TORSO_YAW"),
  stiffness=STIFFNESS_HIP_YAW,
  damping=DAMPING_HIP_YAW,
  effort_limit=222.0,
  armature=ARMATURE_HIP_YAW,
)

T800_ACTUATOR_LIMB = BuiltinPositionActuatorCfg(
  target_names_expr=(".*ANKLE.*", ".*SHOULDER.*", ".*ELBOW_PITCH.*"),
  stiffness=STIFFNESS_LIMB,
  damping=DAMPING_LIMB,
  effort_limit=160.0,
  armature=ARMATURE_LIMB,
)

T800_ACTUATOR_SMALL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*ELBOW_YAW.*", ".*HEAD.*"),
  stiffness=STIFFNESS_SMALL,
  damping=DAMPING_SMALL,
  effort_limit=52.0,
  armature=ARMATURE_SMALL,
)


STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.03),
  joint_pos={
    ".*HIP_PITCH.*": -0.12,
    ".*KNEE_PITCH.*": 0.24,
    ".*ANKLE_PITCH.*": -0.12,
    ".*": 0.0,
  },
  joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  contype=1,
  conaffinity=1,
  condim=3,
)

T800_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    T800_ACTUATOR_HIP_KNEE,
    T800_ACTUATOR_HIP_ROLL,
    T800_ACTUATOR_HIP_YAW,
    T800_ACTUATOR_LIMB,
    T800_ACTUATOR_SMALL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_t800_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=STANDING_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=T800_ARTICULATION,
  )


T800_ACTION_SCALE: dict[str, float] = {}
for a in T800_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    T800_ACTION_SCALE[n] = 0.25 * e / s
