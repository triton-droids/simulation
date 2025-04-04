from .helper import Helper
from .misc.enums import TDroidType
from ml_collections import ConfigDict
import logging
import os

# Global singleton, configured once during configuration initialization via .configure.
class Logger:
    _instance = None
    _enabled_levels = {TDroidType.Log.GENERAL: False, TDroidType.Log.TRAIN: False, TDroidType.Log.SIMULATE: False}
    _file_handlers = {}

    def __new__(cls, aType: TDroidType.Log):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize_logger()
        if not cls._enabled_levels[aType]:
            return cls._no_logger()
        return cls.get_logger(aType)

    @classmethod
    def configure(cls, aConfig: ConfigDict):
        os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True) # Root Logs Directory
        for section, content in aConfig.items():
            if 'settings' in content and 'enable_logging' in content.settings:
                theName = Helper.toEnum(section, TDroidType.Log)
                cls._enabled_levels[theName] = content.settings.enable_logging
                cls.create_directory(theName) # Subsequent Log Directories

    @classmethod
    def create_directory(cls, aType: TDroidType.Log):
        if not cls._enabled_levels[aType]:
            return

        log_type_dir = os.path.join(os.getcwd(), "logs", aType.name.lower())
        os.makedirs(log_type_dir, exist_ok=True)
        log_file = os.path.join(log_type_dir, f"{aType.name.lower()}.log")
        file_handler = logging.FileHandler(log_file, mode='a') # Don't recreate new log files everytime, but append them.
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - '
                                      '%(filename)s - %(funcName)s - %(lineno)d')
        file_handler.setFormatter(formatter)
        cls._file_handlers[aType] = file_handler
   
    @classmethod
    def get_logger(self, aType):
        logger = logging.getLogger(f"{aType.name} LOGGER")
        logger.setLevel(logging.DEBUG)

        if aType in self._file_handlers:
            file_handler = self._file_handlers[aType]
            logger.addHandler(file_handler)

        return logger

    @classmethod
    def _no_logger(self):
        class NoLogger:
            def debug(self, msg, *args, **kwargs): pass
            def info(self, msg, *args, **kwargs): pass
            def warning(self, msg, *args, **kwargs): pass
            def error(self, msg, *args, **kwargs): pass
            def critical(self, msg, *args, **kwargs): pass
            def exception(self, msg, *args, **kwargs): pass
            def log(self, level, msg, *args, **kwargs): pass
        return NoLogger()

    def _initialize_logger(self):
        self.logger = logging.getLogger("ApplicationLogger")
        self.logger.setLevel(logging.DEBUG)

        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(logging.DEBUG)
        self.stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.stream_handler)


