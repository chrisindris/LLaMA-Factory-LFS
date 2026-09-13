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
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Union

import fsspec
from datasets import Dataset, DatasetDict, IterableDataset, concatenate_datasets, interleave_datasets

from ..extras import logging


if TYPE_CHECKING:
    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


SLOTS = list[Union[str, set[str], dict[str, str]]]


@unique
class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


class DatasetModule(TypedDict):
    train_dataset: Optional[Union["Dataset", "IterableDataset"]]
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]


def merge_dataset(
    all_datasets: list[Union["Dataset", "IterableDataset"]], data_args: "DataArguments", seed: int
) -> Union["Dataset", "IterableDataset"]:
    r"""Merge multiple datasets to a unified dataset."""
    if len(all_datasets) == 1:
        return all_datasets[0]

    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning_rank0_once("The samples between different datasets will not be mixed in streaming mode.")

        return concatenate_datasets(all_datasets)

    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning_rank0_once("We recommend using `mix_strategy=concat` in non-streaming mode.")

        strategy_map: str = {
            "interleave_under": "first_exhausted",
            "interleave_over": "all_exhausted",
            "interleave_once": "all_exhausted_without_replacement",
        }[data_args.mix_strategy]

        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=seed,
            stopping_strategy=strategy_map,  # type: ignore
        )

    else:
        raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}.")


def _content_key(prompt: Any, response: Any) -> str:
    r"""Stable content fingerprint for aligned `_prompt` / `_response` fields."""
    if prompt is None and response is None:
        return ""
    return json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False, sort_keys=True, default=str)


def _iter_eval_datasets(
    eval_dataset: Union["Dataset", "IterableDataset", dict[str, "Dataset"]],
) -> list[Union["Dataset", "IterableDataset"]]:
    if isinstance(eval_dataset, dict):
        return [data for data in eval_dataset.values() if data is not None]
    return [eval_dataset]


def _eval_names_subset_of_train(data_args: "DataArguments") -> bool:
    train_names = set(data_args.dataset or [])
    eval_names = set(data_args.eval_dataset or [])
    return bool(train_names and eval_names and eval_names <= train_names)


def _collect_eval_overlap_keys(
    eval_dataset: Union["Dataset", "IterableDataset", dict[str, "Dataset"]],
) -> tuple[set[str], set[str]]:
    r"""Collect eval `_question_id`s and prompt/response fingerprints from map-style eval sets."""
    eval_ids: set[str] = set()
    eval_contents: set[str] = set()
    skipped_streaming = False
    for dataset in _iter_eval_datasets(eval_dataset):
        if not isinstance(dataset, Dataset):
            skipped_streaming = True
            continue
        columns = getattr(dataset, "column_names", None) or []
        if "_question_id" in columns:
            for qid in dataset["_question_id"]:
                if qid is None or qid == "":
                    continue
                eval_ids.add(str(qid))
        if "_prompt" in columns and "_response" in columns:
            for prompt, response in zip(dataset["_prompt"], dataset["_response"], strict=True):
                key = _content_key(prompt, response)
                if key:
                    eval_contents.add(key)

    if skipped_streaming:
        logger.warning_rank0(
            "`exclude_eval_from_train` skipped streaming eval dataset(s); map-style eval is required to collect holdout keys."
        )
    return eval_ids, eval_contents


def _exclude_eval_rows_from_train(
    dataset: Union["Dataset", "IterableDataset"],
    eval_ids: set[str],
    eval_contents: set[str],
) -> Union["Dataset", "IterableDataset"]:
    if not eval_ids and not eval_contents:
        return dataset

    def _keep(example: dict[str, Any]) -> bool:
        qid = example.get("_question_id")
        if qid not in (None, "") and str(qid) in eval_ids:
            return False
        key = _content_key(example.get("_prompt"), example.get("_response"))
        return not (key and key in eval_contents)

    return dataset.filter(_keep)


def _split_test_size(size: float) -> int | float:
    r"""Match `val_size` units: integer count if `size > 1`, otherwise a fraction."""
    return int(size) if size > 1 else size


def _downsample_train_to_val_size_equivalent(
    dataset: Union["Dataset", "IterableDataset"],
    size: float,
    seed: int,
    streaming: bool,
) -> Union["Dataset", "IterableDataset"]:
    r"""Keep the train slice `--val_size` would have kept; discard the rest (do not use it as eval)."""
    if streaming:
        discarded = int(size)
        logger.info_rank0(
            f"`val_size_equivalent={size}`: skipping {discarded} streamed examples as unused val-equivalent; "
            "eval_dataset is the real eval."
        )
        return dataset.skip(discarded)

    test_size = _split_test_size(size)
    split_result = dataset.train_test_split(test_size=test_size, seed=seed)
    n_full = len(dataset)
    n_train = len(split_result["train"])
    n_discarded = len(split_result["test"])
    logger.info_rank0(
        f"`val_size_equivalent={size}`: keeping {n_train}/{n_full} train examples "
        f"(discarded {n_discarded} as unused val-equivalent; eval_dataset is the real eval)."
    )
    return split_result["train"]


def split_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]],
    data_args: "DataArguments",
    seed: int,
) -> tuple[dict, dict]:
    r"""Split the dataset and returns two dicts containing train set and validation set.

    Support both map dataset and iterable dataset.

    Returns:
        train_dict: Dictionary containing training data with key "train"
        eval_dict: Dictionary containing evaluation data with keys "validation" or "validation_{name}"
    """
    if eval_dataset is not None and data_args.val_size > 1e-6:
        raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

    # the train and eval better to in dict dtype and separately return for cpode clearly and good handle outside
    train_dict, eval_dict = {}, {}

    if dataset is not None:
        if data_args.streaming:
            dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

        if data_args.val_size > 1e-6:
            if data_args.streaming:
                eval_dict["validation"] = dataset.take(int(data_args.val_size))
                train_dict["train"] = dataset.skip(int(data_args.val_size))
            else:
                val_size = _split_test_size(data_args.val_size)
                split_result = dataset.train_test_split(test_size=val_size, seed=seed)
                train_dict["train"] = split_result["train"]
                eval_dict["validation"] = split_result["test"]
        elif data_args.val_size_equivalent > 1e-6:
            train_dict["train"] = _downsample_train_to_val_size_equivalent(
                dataset,
                data_args.val_size_equivalent,
                seed,
                streaming=data_args.streaming,
            )
        else:
            train_dict["train"] = dataset

    if eval_dataset is not None:
        if isinstance(eval_dataset, dict):
            for name, data in eval_dataset.items():
                eval_dict[f"validation_{name}"] = data
        else:
            if data_args.streaming:
                eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

            eval_dict["validation"] = eval_dataset

    if (
        data_args.exclude_eval_from_train
        and data_args.eval_dataset
        and "train" in train_dict
        and eval_dict
        and not _eval_names_subset_of_train(data_args)
    ):
        eval_ids, eval_contents = _collect_eval_overlap_keys(
            {key: value for key, value in eval_dict.items() if value is not None}
        )
        if eval_ids or eval_contents:
            train_dataset = train_dict["train"]
            before = len(train_dataset) if isinstance(train_dataset, Dataset) else None
            train_ids: set[str] = set()
            if isinstance(train_dataset, Dataset) and "_question_id" in (train_dataset.column_names or []):
                train_ids = {str(qid) for qid in train_dataset["_question_id"] if qid not in (None, "")}
            train_dict["train"] = _exclude_eval_rows_from_train(train_dataset, eval_ids, eval_contents)
            dropped = None if before is None else before - len(train_dict["train"])
            missing = sorted(eval_ids - train_ids) if train_ids else []
            if dropped is None:
                logger.info_rank0(
                    f"Excluding eval overlap from train ({len(eval_ids)} eval question_ids, "
                    f"{len(eval_contents)} eval content keys)."
                )
            else:
                logger.info_rank0(
                    f"Excluded {dropped}/{before} train examples overlapping eval_dataset "
                    f"({len(eval_ids)} eval question_ids, {len(eval_contents)} eval content keys)."
                )
            if missing:
                logger.info_rank0(f"eval question_ids not found in train: {missing}")
        else:
            logger.warning_rank0(
                "`exclude_eval_from_train` is enabled but eval_dataset has no `_question_id` or prompt/response keys."
            )
    elif data_args.exclude_eval_from_train and _eval_names_subset_of_train(data_args):
        logger.info_rank0(
            "Skipping `exclude_eval_from_train`: eval dataset names are a subset of train dataset names."
        )

    return train_dict, eval_dict


def get_dataset_module(dataset: Union["Dataset", "DatasetDict"]) -> "DatasetModule":
    r"""Convert dataset or dataset dict to dataset module."""
    dataset_module: DatasetModule = {}
    if isinstance(dataset, DatasetDict):  # dataset dict
        if "train" in dataset:
            dataset_module["train_dataset"] = dataset["train"]

        if "validation" in dataset:
            dataset_module["eval_dataset"] = dataset["validation"]
        else:
            eval_dataset = {}
            for key in dataset.keys():
                if key.startswith("validation_"):
                    eval_dataset[key[len("validation_") :]] = dataset[key]

            if len(eval_dataset):
                dataset_module["eval_dataset"] = eval_dataset

    else:  # single dataset
        dataset_module["train_dataset"] = dataset

    return dataset_module


def setup_fs(path: str, anon: bool = False) -> "fsspec.AbstractFileSystem":
    r"""Set up a filesystem object based on the path protocol."""
    storage_options = {"anon": anon} if anon else {}
    if path.startswith("s3://"):
        fs = fsspec.filesystem("s3", **storage_options)
    elif path.startswith(("gs://", "gcs://")):
        fs = fsspec.filesystem("gcs", **storage_options)
    else:
        raise ValueError(f"Unsupported protocol in path: {path}. Use 's3://' or 'gs://'.")

    if not fs.exists(path):
        raise ValueError(f"Path does not exist: {path}.")

    return fs


def _read_json_with_fs(fs: "fsspec.AbstractFileSystem", path: str) -> list[Any]:
    r"""Helper function to read JSON/JSONL files using fsspec."""
    with fs.open(path, "r") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)


def read_cloud_json(cloud_path: str) -> list[Any]:
    r"""Read a JSON/JSONL file from cloud storage (S3 or GCS).

    Args:
        cloud_path: str
            Cloud path in the format:
            - 's3://bucket-name/file.json' for AWS S3
            - 'gs://bucket-name/file.jsonl' or 'gcs://bucket-name/file.jsonl' for Google Cloud Storage
    """
    try:
        fs = setup_fs(cloud_path, anon=True)  # try with anonymous access first
    except Exception:
        fs = setup_fs(cloud_path)  # try again with credentials

    # filter out non-JSON files
    files = [x["Key"] for x in fs.listdir(cloud_path)] if fs.isdir(cloud_path) else [cloud_path]
    files = list(filter(lambda file: file.endswith(".json") or file.endswith(".jsonl"), files))
    if not files:
        raise ValueError(f"No JSON/JSONL files found in the specified path: {cloud_path}.")

    return sum([_read_json_with_fs(fs, file) for file in files], [])
