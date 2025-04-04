from .logger import Logger
from .helper import Helper
from ml_collections import ConfigDict
from glob import glob
import yaml
import os

# Access via config_files[n]: returns a list (directory_path, [files]) of the nth directory
theFiles = [
    ("tools/configs", glob("tools/configs/*.yml")),
    ("tools/configs/models", glob("tools/configs/models/**/*.yml", recursive=True))
]

class Configurations:
    def __init__(self):
        with open(theFiles[0][1][0], 'r') as theData:
            theInstructions = Helper.toConfigDict(yaml.safe_load(theData))

        with open(theFiles[1][1][0], 'r') as theData:
            theModel = Helper.toConfigDict(yaml.safe_load(theData))
        
        self.configs = ConfigDict({
            'instructions': theInstructions,
            'model': theModel,
        })
        # Global configurations for loggers
        logger = Logger.configure(theInstructions)
         
    def __getattr__(self, aKey):
        return getattr(self.configs, aKey) 

