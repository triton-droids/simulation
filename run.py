from src.core import Driver, Trainer, Tester
from src.utils import Logger
from src.utils.misc.enums import TDroidType
import logging
import numpy as np

if __name__ == "__main__":
    theDriver = Driver()
    theDriver.train()
    theDriver.evaluate()


