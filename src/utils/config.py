from .logger import Logger
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
            theInstructions = yaml.safe_load(theData)

        with open(theFiles[1][1][0], 'r') as theData:
            theModel = yaml.safe_load(theData)

        self.configs = {
                'instructions': theInstructions,
                'model': theModel,
        }
        # Global configurations for loggers
        logger = Logger.configure(theInstructions)
         
        
    def __getitem__(self, key):
        return self.configs[key]


