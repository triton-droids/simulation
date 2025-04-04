from .misc.enums import TDroidType
from enum import Enum
from ml_collections import ConfigDict
from typing import Type, TypeVar

T = TypeVar('T', bound=Enum)

class Helper:
    _enum_map = {
        "general": TDroidType.Log.GENERAL,
        "train": TDroidType.Log.TRAIN,
        "simulate": TDroidType.Log.SIMULATE
    }
    
    _string_map = {
        TDroidType.Errors.INVALID_ENUM: "Invalid TDroidType, please check for conversion compatibility between string and TDroidType",
        TDroidType.Errors.INVALID_CONFIG: "Invalid instructions, please check the .yml files"
    }

    @staticmethod
    def toEnum(aStr: str, aType: Type[T]) -> T:
        theStr = aStr.upper()
        theResult = getattr(aType, theStr, TDroidType.Errors.INVALID_ENUM)
        if theResult == TDroidType.Errors.INVALID_ENUM:
            theError = Helper.toString(theResult)
            raise ValueError(f"{theError}, error:{aStr}")
        return theResult

    @staticmethod
    def toString(anEnum: Type[T]) -> str:
        return Helper._string_map[anEnum]

    @staticmethod
    def toConfigDict(aDict: dict) -> ConfigDict:
        if isinstance(aDict, dict):
            return ConfigDict({k: Helper.toConfigDict(v) for k, v in aDict.items()})
        elif isinstance(aDict, list):
            return [Helper.toConfigDict(i) for i in aDict]
        elif isinstance(aDict, (str, int, float, bool, type(None))):
            return aDict
        else:
            raise ValueError(f"Unsupported config type: {type(aDict)} — value: {repr(aDict)}")

 
