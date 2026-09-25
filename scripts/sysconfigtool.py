# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read and write per-cluster settings from sysconfig.json.

Values may contain ``${VAR}`` tokens, which are expanded against the process
environment on read. This lets a section express repo-relative paths such as
``${PROJECT_DIR}/containers/llamafactory.sif``. Unset tokens are left verbatim
so a caller can detect and report them.
"""

import json
import os
import re


# Get the directory where this script is located.
_script_dir = os.path.dirname(os.path.abspath(__file__))
# The JSON file is in the same directory.
_config_file = os.path.join(_script_dir, "sysconfig.json")

_TOKEN_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _load_config():
    """Load the configuration from sysconfig.json."""
    if not os.path.exists(_config_file):
        return {}

    with open(_config_file, encoding="utf-8") as f:
        return json.load(f)


def _save_config(data):
    """Save the configuration to sysconfig.json."""
    with open(_config_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def expand_value(value):
    """Expand ``${VAR}`` tokens in a value against the environment.

    Non-string values are returned unchanged. Tokens whose variable is unset are
    left verbatim rather than replaced with an empty string, so that a missing
    ``PROJECT_DIR`` surfaces as a visible ``${PROJECT_DIR}`` instead of a path
    that silently starts at the filesystem root.

    Args:
        value: The raw configuration value.

    Returns:
        The value with known tokens expanded.
    """
    if not isinstance(value, str):
        return value

    def _sub(match):
        return os.environ.get(match.group(1), match.group(0))

    return _TOKEN_RE.sub(_sub, value)


def read(system, key):
    """Read a value from the sysconfig.json file.

    Args:
        system (str): The system name (e.g. "RORQUAL", "PORTABLE").
        key (str): The configuration key (e.g. "HF_HOME").

    Returns:
        The expanded value of the configuration key, or None if not found.
    """
    config = _load_config()
    return expand_value(config.get(system, {}).get(key))


def read_all(system):
    """Read all configuration key-value pairs for a system.

    Args:
        system (str): The system name (e.g. "RORQUAL", "PORTABLE").

    Returns:
        dict: Mapping of configuration keys to expanded values for the system,
            or an empty dict if the system is not found.
    """
    config = _load_config()
    return {key: expand_value(value) for key, value in config.get(system, {}).items()}


def write(system, key, value):
    """Write a raw value to the sysconfig.json file.

    The value is stored verbatim; ``${VAR}`` tokens are preserved so they expand
    on read.

    Args:
        system (str): The system name (e.g. "RORQUAL").
        key (str): The configuration key (e.g. "HF_HOME").
        value: The value to write.
    """
    config = _load_config()
    if system not in config:
        config[system] = {}

    config[system][key] = value
    _save_config(config)
