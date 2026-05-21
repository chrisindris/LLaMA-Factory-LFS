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

r"""Efficient fine-tuning of large language models.

Level:
  api, webui > chat, eval, train > data, model > hparams > extras

Disable version checking: DISABLE_VERSION_CHECK=1
Enable VRAM recording: RECORD_VRAM=1
Force using torchrun: FORCE_TORCHRUN=1
Set logging verbosity: LLAMAFACTORY_VERBOSITY=WARN
Use modelscope: USE_MODELSCOPE_HUB=1
Use openmind: USE_OPENMIND_HUB=1
"""

try:
    import transformers

    if not hasattr(transformers, "HybridCache"):
        # Compatibility shim for older PEFT expecting HybridCache.
        try:
            from transformers import DynamicCache as _HybridCache
        except Exception:
            from transformers import Cache as _HybridCache
        transformers.HybridCache = _HybridCache
except Exception:
    pass

try:
    from transformers.models.auto import modeling_auto as _modeling_auto

    if not hasattr(_modeling_auto, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"):
        if hasattr(_modeling_auto, "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES"):
            _modeling_auto.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = (
                _modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
            )
        else:
            _modeling_auto.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = {}
except Exception:
    pass

from .extras.env import VERSION


__version__ = VERSION
