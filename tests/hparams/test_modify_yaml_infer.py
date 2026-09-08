# Copyright 2026 the LlamaFactory team.
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

import importlib.util
from pathlib import Path


def _load_modify_yaml():
    path = Path(__file__).resolve().parents[2] / "scripts" / "utils" / "modify_yaml.py"
    spec = importlib.util.spec_from_file_location("modify_yaml", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_infer_data_type_parses_json_object():
    infer = _load_modify_yaml().infer_data_type
    assert infer('{"num_cycles": 0.4}') == {"num_cycles": 0.4}
    assert infer("null") is None
    assert infer("true") is True
    assert infer("8") == 8
    assert infer("0.02") == 0.02
    assert infer("/tmp/control_tokens.yaml") == "/tmp/control_tokens.yaml"
