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
import json
import os
import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union, Optional

from ..extras import logging
from ..extras.constants import IMAGE_PLACEHOLDER
from .data_utils import Role
from .data_packing.h5_image_store import can_resolve_h5_image

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .mm_plugin import AudioInput, ImageInput, VideoInput
    from .parser import DatasetAttr

    MediaType = Union[ImageInput, VideoInput, AudioInput]


logger = logging.get_logger(__name__)


@dataclass
class DatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"

    def _sample_image_indices(self, num_images: int) -> list[int]:
        r"""Select image indices according to the configured sampling mode."""
        if num_images <= 0:
            return []

        if self.data_args.image_sample_count > 0:
            sample_count = min(self.data_args.image_sample_count, num_images)
            if sample_count >= num_images:
                return list(range(num_images))
            if sample_count == 1:
                return [0]

            return [round(i * (num_images - 1) / (sample_count - 1)) for i in range(sample_count)]

        stride = self.data_args.image_sample_stride
        if stride <= 1:
            return list(range(num_images))

        return list(range(0, num_images, stride))

    def _subsample_image_placeholders(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        r"""Keep image placeholders aligned with the sampled image indices."""
        total_placeholders = sum(message.get("content", "").count(IMAGE_PLACEHOLDER) for message in messages)
        selected_indices = set(self._sample_image_indices(total_placeholders))
        if len(selected_indices) == total_placeholders:
            return messages

        image_placeholder = IMAGE_PLACEHOLDER
        image_pattern = re.compile(re.escape(image_placeholder))
        placeholder_index = 0
        new_messages = []
        for message in messages:
            content = message.get("content", "")
            if image_placeholder not in content:
                new_messages.append(message)
                continue

            new_content_parts = []
            last_end = 0
            for match in image_pattern.finditer(content):
                new_content_parts.append(content[last_end:match.start()])
                if placeholder_index in selected_indices:
                    new_content_parts.append(image_placeholder)
                placeholder_index += 1
                last_end = match.end()
            new_content_parts.append(content[last_end:])

            new_message = dict(message)
            new_message["content"] = "".join(new_content_parts)
            new_messages.append(new_message)

        return new_messages

    def _find_medias(self, medias: Union["MediaType", list["MediaType"], None]) -> list["MediaType"] | None:
        r"""Optionally concatenate media path to media dir when loading from local disk."""
        
        logger.info_rank0(f"DEBUG _find_medias: load_from={self.dataset_attr.load_from}, media_dir={self.data_args.media_dir}")
        
        if medias is None:
            return None
        elif not isinstance(medias, list):
            medias = [medias]
        elif len(medias) == 0:
            return None
        else:
            medias = medias[:]

        if self.dataset_attr.load_from in ["script", "file"]:
            if isinstance(medias[0], str):
                for i in range(len(medias)):
                    original = medias[i]
                    media_path = os.path.join(self.data_args.media_dir, original)
                    if os.path.isfile(media_path):  # filesystem path under media_dir
                        medias[i] = media_path
                    elif os.path.isfile(original):  # already absolute / cwd-relative file
                        medias[i] = original
                    elif can_resolve_h5_image(original) or can_resolve_h5_image(media_path):
                        # Keep a resolvable path string for lazy H5 decode in mm_plugin.
                        # Prefer the annotation-relative key (original) for index normalize.
                        medias[i] = original if can_resolve_h5_image(original) else media_path
                    else:
                        logger.warning_rank0_once(
                            f"Media {media_path} does not exist in `media_dir` and is not in H5 indexes."
                        )
                        raise ValueError(f"Failed to resolve image path: original={original!r}, media_path={media_path!r}")
            elif isinstance(medias[0], list):  # for processed video frames
                # medias is a list of lists, e.g., [[frame1.jpg, frame2.jpg], [frame3.jpg, frame4.jpg]]
                for i in range(len(medias)):
                    for j in range(len(medias[i])):
                        original = medias[i][j]
                        if not isinstance(original, str):
                            continue
                        media_path = os.path.join(self.data_args.media_dir, original)
                        if os.path.isfile(media_path):
                            medias[i][j] = media_path
                        elif os.path.isfile(original):
                            medias[i][j] = original
                        elif can_resolve_h5_image(original) or can_resolve_h5_image(media_path):
                            medias[i][j] = original if can_resolve_h5_image(original) else media_path
                        else:
                            logger.warning_rank0_once(
                                f"Media {medias[i][j]} does not exist in `media_dir` and is not in H5 indexes."
                            )

        return medias

    def _find_images(self, images: Union["ImageInput", list["ImageInput"], None]) -> Optional[list["ImageInput"]]:
        r"""Optionally subsample image lists before resolving local paths."""
        if images is None:
            return None
        if not isinstance(images, list):
            images = [images]
        elif len(images) == 0:
            return None
        else:
            images = images[:]

        if not isinstance(images[0], list):
            images = [images[index] for index in self._sample_image_indices(len(images))]

        return self._find_medias(images)

    @abstractmethod
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r"""Convert a single example in the dataset to the standard format."""
        ...


@dataclass
class AlpacaDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        prompt = []
        if self.dataset_attr.history and isinstance(example[self.dataset_attr.history], list):
            for old_prompt, old_response in example[self.dataset_attr.history]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        query = []
        if self.dataset_attr.prompt and example[self.dataset_attr.prompt]:
            query.append(example[self.dataset_attr.prompt])

        if self.dataset_attr.query and example[self.dataset_attr.query]:
            query.append(example[self.dataset_attr.query])

        prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "prompt\nquery"

        if self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], str)
            and isinstance(example[self.dataset_attr.rejected], str)
        ):  # pairwise example
            response = [
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.chosen]},
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.rejected]},
            ]
        elif self.dataset_attr.response and isinstance(example[self.dataset_attr.response], str):  # normal example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
        else:  # unsupervised
            response = []

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_images(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
            "_question_id": (
                str(example[self.dataset_attr.question_id])
                if self.dataset_attr.question_id and example.get(self.dataset_attr.question_id) is not None
                else None
            ),
        }
        output["_prompt"] = self._subsample_image_placeholders(output["_prompt"])
        return output


@dataclass
class SharegptDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag, self.dataset_attr.observation_tag)
        even_tags = (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)
        messages = example[self.dataset_attr.messages]
        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping[message[self.dataset_attr.role_tag]],
                    "content": message[self.dataset_attr.content_tag],
                }
            )

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning_rank0(f"Invalid message count in {messages}.")
            broken_data = True

        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_images(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
            "_question_id": (
                str(example[self.dataset_attr.question_id])
                if self.dataset_attr.question_id and example.get(self.dataset_attr.question_id) is not None
                else None
            ),
        }
        output["_prompt"] = self._subsample_image_placeholders(output["_prompt"])
        return output


@dataclass
class OpenAIDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }

        messages = example[self.dataset_attr.messages]
        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example.get(self.dataset_attr.system, "") if self.dataset_attr.system else ""

        aligned_messages = []
        tool_responses = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            role = message[self.dataset_attr.role_tag]
            content = message[self.dataset_attr.content_tag]

            if role in [self.dataset_attr.assistant_tag, self.dataset_attr.function_tag]:
                if tool_calls := message.get("tool_calls"):
                    tool_calls_list = [tool["function"] for tool in tool_calls]
                    content = json.dumps(tool_calls_list, ensure_ascii=False)
                    role = self.dataset_attr.function_tag

            if role == self.dataset_attr.observation_tag:
                tool_responses.append(content)
                continue
            elif len(tool_responses) > 0:
                _content = "\n</tool_response>\n<tool_response>\n".join(tool_responses)
                aligned_messages.append(
                    {
                        "role": Role.OBSERVATION.value,
                        "content": _content,
                    }
                )
                tool_responses = []

            aligned_messages.append(
                {
                    "role": tag_mapping[role],
                    "content": content,
                }
            )

        odd_tags = (Role.USER.value, Role.OBSERVATION.value)
        even_tags = (Role.ASSISTANT.value, Role.FUNCTION.value)
        accept_tags = (odd_tags, even_tags)
        for turn_idx, message in enumerate(aligned_messages):
            if message["role"] not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken_data = True
                break

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning_rank0(f"Invalid message count in {messages}.")
            broken_data = True

        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        tools = example.get(self.dataset_attr.tools, "") if self.dataset_attr.tools else ""
        if isinstance(tools, dict) or isinstance(tools, list):
            tools = json.dumps(tools, ensure_ascii=False)

        short_system_prompt = "detailed thinking off"
        if not system:
            if not tools:
                system = short_system_prompt
            else:
                pass
        else:
            if not tools:
                if "detailed thinking on" in system or "detailed thinking off" in system:
                    pass
                else:
                    system += "\n" + short_system_prompt
            else:
                system += "\n"

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": tools,
            "_images": self._find_images(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        output["_prompt"] = self._subsample_image_placeholders(output["_prompt"])
        return output


DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,
    "openai": OpenAIDatasetConverter,
}


def register_dataset_converter(name: str, dataset_converter: type["DatasetConverter"]) -> None:
    r"""Register a new dataset converter."""
    if name in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} already exists.")

    DATASET_CONVERTERS[name] = dataset_converter


def get_dataset_converter(name: str, dataset_attr: "DatasetAttr", data_args: "DataArguments") -> "DatasetConverter":
    r"""Get a dataset converter."""
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")

    return DATASET_CONVERTERS[name](dataset_attr, data_args)


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Align the dataset to a specific format.

    Aligned dataset:
    _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
    _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
    _system: "..."
    _tools: "..."
    _images: []
    _videos: []
    _audios: []
    """
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )

    dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
    ds_name = dataset_attr.dataset_name

    def _convert_with_question_id(example: dict[str, Any], idx: int) -> dict[str, Any]:
        r"""Convert one example and ensure a stable QUESTION_ID.

        Prefer an annotation column mapped via dataset_info columns.question_id
        (see scripts/assign_question_ids.py). If missing, fall back to
        ``{dataset_name}_{idx}`` using the row index after load (file order for
        map-style datasets).
        """
        output = dataset_converter(example)
        qid = output.get("_question_id")
        if qid is None or qid == "":
            output["_question_id"] = f"{ds_name}_{idx}"
        else:
            output["_question_id"] = str(qid)
        return output

    return dataset.map(
        _convert_with_question_id,
        with_indices=True,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )

