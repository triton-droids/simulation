from .trainer import Trainer
from ..utils import Parser, Configurations, Logger
from ..utils.misc.enums import TDroidType

class Driver:
    def __init__(self):
        self.configs = Configurations()
        general_logger = Logger(TDroidType.Log.GENERAL)
        train_logger = Logger(TDroidType.Log.TRAIN)
        sim_logger = Logger(TDroidType.Log.SIMULATE)


        general_logger.info("Hello from general")
        train_logger.info("Hello from train")
        sim_logger.warn("Hello from sim")
        ## Configure High Level Environment
        theInstructions = self.configs["instructions"]
        theTrainer = Trainer(self.configs["model"])
