from ml_collections import ConfigDict
from wandbmon import monitor
from .scripts import brax_train_policy

class Trainer:
    def __init__(self, aConfig: ConfigDict, anEnv: str, aFramework: str):
        self.environment = anEnv
        brax_train_policy(
                     env_name="DefaultHumanoidJoystickFlatTerrain",
                cfg=aConfig)

