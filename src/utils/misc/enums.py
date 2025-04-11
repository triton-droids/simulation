from enum import Enum

class Log(Enum):
    NULL = 0
    GENERAL = 1
    TRAIN = 2
    SIMULATE = 3

class Errors(Enum):
    INVALID_ENUM = 50
    INVALID_CONFIG = 100

class TDroidType:
    Log = Log
    Errors = Errors

