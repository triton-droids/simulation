from .trainer import Trainer
from ..utils import Parser
from ..utils import Configurations

class Driver:
    def __init__(self):
        self.configs = Configurations()
        ## Configure High Level Environment
        theInstructions = self.configs["instructions"]
        theTrainer = Trainer(self.configs["model"])
