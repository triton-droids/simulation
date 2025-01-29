"""
parser code for parsing configuration files
"""
from pathlib import Path

from ml_collections import ConfigDict
import yaml


def parse_cfg(cfg_path: str = None, default_cfg_path: str = None) -> ConfigDict:
    """
    Parses a config file and returns a ConfigDict object. Priority is provided config, 
    then the default config if it exists.
    """
    
    def load_config(path: str) -> ConfigDict:
        """Load a YAML file into a ConfigDict"""
        with open(path, 'r') as file:
            return ConfigDict(yaml.safe_load(file))

    base = ConfigDict()

    if default_cfg_path is not None:
        base = load_config(default_cfg_path)
        
        if 'base_config' in base:
            base_config = base.pop('base_config')
            if isinstance(base_config, str):
                base_config = [base_config]
            if not isinstance(base_config, list):
                raise ValueError("base_config must be a string or list of strings")
            
            old_cfg = None
            for path in base_config:
                new_path = Path(default_cfg_path).parent / Path(path)
                new_cfg = parse_cfg(default_cfg_path=str(new_path))
                if old_cfg is not None:
                    new_cfg = old_cfg.update(new_cfg)
                old_cfg = new_cfg

            base.update(old_cfg)

    # If cfg_path is provided, load and merge it
    if cfg_path is not None:
        cfg = load_config(cfg_path)
        base.update(cfg)

    return base