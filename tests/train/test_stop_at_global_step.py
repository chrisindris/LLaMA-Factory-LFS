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

from types import SimpleNamespace

from llamafactory.train.callbacks import StopAtGlobalStepCallback


def _control() -> SimpleNamespace:
    return SimpleNamespace(should_training_stop=False, should_save=False)


def test_stop_at_global_step_requests_stop_and_save():
    callback = StopAtGlobalStepCallback(3)
    control = _control()

    callback.on_step_end(SimpleNamespace(), SimpleNamespace(global_step=2), control)
    assert control.should_training_stop is False
    assert control.should_save is False

    callback.on_step_end(SimpleNamespace(), SimpleNamespace(global_step=3), control)
    assert control.should_training_stop is True
    assert control.should_save is True
