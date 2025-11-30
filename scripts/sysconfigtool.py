import json
import os

# Get the directory where this script is located.
_script_dir = os.path.dirname(os.path.abspath(__file__))
# The JSON file is in the same directory.
_config_file = os.path.join(_script_dir, 'sysconfig.json')

def _load_config():
    """Loads the configuration from sysconfig.json."""
    if not os.path.exists(_config_file):
        return {}
    with open(_config_file, 'r') as f:
        return json.load(f)

def _save_config(data):
    """Saves the configuration to sysconfig.json."""
    with open(_config_file, 'w') as f:
        json.dump(data, f, indent=4)

def read(system, key):
    """
    Reads a value from the sysconfig.json file.

    Args:
        system (str): The system name (e.g., "RORQUAL").
        key (str): The configuration key (e.g., "HF_HOME").

    Returns:
        The value of the configuration key, or None if not found.
    """
    config = _load_config()
    return config.get(system, {}).get(key)

def write(system, key, value):
    """
    Writes a value to the sysconfig.json file.

    Args:
        system (str): The system name (e.g., "RORQUAL").
        key (str): The configuration key (e.g., "HF_HOME").
        value (any): The value to write.
    """
    config = _load_config()
    if system not in config:
        config[system] = {}
    config[system][key] = value
    _save_config(config)
