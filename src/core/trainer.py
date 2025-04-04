from ml_collections import ConfigDict

class Trainer:
    def __init__(self, aConfig: ConfigDict, anEnv: str, aFramework: str):
        self.environment = anEnv

