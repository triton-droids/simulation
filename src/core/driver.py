from .trainer import Trainer
from ..utils import Configurations, Logger
from ..utils.misc.enums import TDroidType

        
class Driver:
    def __init__(self):
        self.configs = Configurations()

        ## Configure High Level Environment
        theTrainConfig = self.configs.instructions.train
        theEnvironment, theFramework = theTrainConfig.environment, theTrainConfig.framework
        self.theTrainer = Trainer(self.configs.model, theEnvironment, theFramework)

    def train(self):
        pass

    def evaluate(self):
        pass
