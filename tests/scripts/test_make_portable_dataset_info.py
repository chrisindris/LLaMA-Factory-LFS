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
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import make_portable_dataset_info as mpdi  # noqa: E402


def test_absolute_file_name_becomes_dataset_scoped_relative_path():
    registry = {"Scene30k": {"file_name": "/abs/hub/snap/data/train-00000-of-00001.parquet"}}
    new_registry, links = mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert new_registry["Scene30k"]["file_name"] == "Scene30k/train-00000-of-00001.parquet"
    assert links == [("Scene30k/train-00000-of-00001.parquet", "/abs/hub/snap/data/train-00000-of-00001.parquet")]


def test_absolute_directory_entry_keeps_trailing_slash_and_links_slash_free():
    # Seven entries in the repo's registry name a directory this way. basename()
    # returns "" for them, which would put the symlink at "SQA3D/" -- unbuildable
    # once "SQA3D" exists as its own parent.
    registry = {"SQA3D": {"file_name": "/abs/hub/snap/data/"}}
    new_registry, links = mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert new_registry["SQA3D"]["file_name"] == "SQA3D/data/"
    assert links == [("SQA3D/data", "/abs/hub/snap/data")]


def test_main_links_a_directory_entry(tmp_path):
    target = tmp_path / "real" / "data"
    target.mkdir(parents=True)
    (target / "shard.parquet").write_text("x", encoding="utf-8")
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": f"{target}/"}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    assert mpdi.main(["--source", str(source), "--dest", str(dest)]) == 0

    link = tmp_path / "annotations" / "D" / "data"
    assert link.is_symlink()
    assert (link / "shard.parquet").read_text(encoding="utf-8") == "x"


def test_main_reports_a_blocked_link_instead_of_raising(tmp_path):
    target = tmp_path / "real" / "x.parquet"
    target.parent.mkdir(parents=True)
    target.write_text("data", encoding="utf-8")
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": str(target)}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"
    # A real directory sits exactly where the symlink must go.
    (tmp_path / "annotations" / "D" / "x.parquet").mkdir(parents=True)

    rc = mpdi.main(["--source", str(source), "--dest", str(dest), "--require", "D"])

    assert rc == 1
    # Atomic publish: a registry naming a link that was never created is worse
    # than no registry, because preflight's existence check would accept it.
    assert not dest.exists()


def test_override_redirects_the_link_target():
    # The registry records another user's unreadable path; site.env redirects it.
    registry = {"Scene30k": {"file_name": "/scratch/someone-else/hub/snap/data/train.parquet"}}
    new_registry, links = mpdi.rewrite_registry(
        registry, "/repo/data", "/repo/data/annotations", {"Scene30k": "/my/own/train.parquet"}
    )

    assert links == [("Scene30k/train.parquet", "/my/own/train.parquet")]
    assert new_registry["Scene30k"]["file_name"] == "Scene30k/train.parquet"


def test_override_applies_to_a_directory_entry():
    registry = {"SQA3D": {"file_name": "/scratch/someone-else/snap/data/"}}
    _, links = mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations", {"SQA3D": "/my/own/shards/"})

    assert links == [("SQA3D/shards", "/my/own/shards")]


def test_main_override_wins_over_the_registry(tmp_path):
    target = tmp_path / "mine" / "train.parquet"
    target.parent.mkdir(parents=True)
    target.write_text("ok", encoding="utf-8")
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"Scene30k": {"file_name": "/unreadable/train.parquet"}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    rc = mpdi.main(
        ["--source", str(source), "--dest", str(dest), "--require", "Scene30k", "--override", f"Scene30k={target}"]
    )

    assert rc == 0
    assert (tmp_path / "annotations" / "Scene30k" / "train.parquet").read_text(encoding="utf-8") == "ok"


def test_main_malformed_override_is_rejected(tmp_path):
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": "x.jsonl"}}), encoding="utf-8")

    with pytest.raises(SystemExit):
        mpdi.main(["--source", str(source), "--dest", str(tmp_path / "d.json"), "--override", "no-equals-sign"])


def test_main_require_rejects_an_unknown_dataset(tmp_path):
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": "x.jsonl"}}), encoding="utf-8")

    with pytest.raises(SystemExit):
        mpdi.main(["--source", str(source), "--dest", str(tmp_path / "d.json"), "--require", "Typo"])


def test_unrequired_missing_source_does_not_fail(tmp_path):
    # The seven SQA3D ablation entries must not fail a run that does not use them.
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"SQA3Devery24": {"file_name": "/definitely/missing/data/"}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    rc = mpdi.main(["--source", str(source), "--dest", str(dest)])

    assert rc == 0
    assert dest.exists()


def test_dest_registry_is_not_written_when_a_required_source_is_missing(tmp_path):
    # A registry with dangling links would satisfy preflight's existence check and
    # the job would then die after the allocation starts.
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"Scene30k": {"file_name": "/definitely/missing/x.parquet"}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    rc = mpdi.main(["--source", str(source), "--dest", str(dest), "--require", "Scene30k"])

    assert rc == 1
    assert not dest.exists()


def test_a_previously_good_registry_survives_a_failed_run(tmp_path):
    dest = tmp_path / "annotations" / "dataset_info.json"
    dest.parent.mkdir(parents=True)
    dest.write_text('{"good": {}}', encoding="utf-8")
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"Scene30k": {"file_name": "/definitely/missing/x.parquet"}}), encoding="utf-8")

    assert mpdi.main(["--source", str(source), "--dest", str(dest), "--require", "Scene30k"]) == 1
    assert json.loads(dest.read_text(encoding="utf-8")) == {"good": {}}


def test_summary_counts_only_links_actually_created(tmp_path, capsys):
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": "sub/x.jsonl"}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    mpdi.main(["--source", str(source), "--dest", str(dest), "--no-symlinks"])

    assert "0 links in place" in capsys.readouterr().out


def test_relative_file_name_is_reanchored_to_new_dir():
    registry = {"3DThinker10k": {"file_name": "3DThinker-10K/out/3dthinker10k_cot.jsonl"}}
    new_registry, links = mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert new_registry["3DThinker10k"]["file_name"] == "../3DThinker-10K/out/3dthinker10k_cot.jsonl"
    assert links == []


def test_hub_url_entries_are_left_alone():
    registry = {"alpaca_en_demo": {"hf_hub_url": "llamafactory/alpaca_en"}}
    new_registry, links = mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert new_registry == registry
    assert links == []


def test_source_registry_is_not_mutated():
    registry = {"Scene30k": {"file_name": "/abs/x.parquet"}}
    mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert registry["Scene30k"]["file_name"] == "/abs/x.parquet"


def test_other_keys_are_preserved():
    registry = {
        "Scene30k": {
            "file_name": "/abs/x.parquet",
            "formatting": "alpaca",
            "columns": {"prompt": "q", "response": "cot"},
        }
    }
    new_registry, _ = mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert new_registry["Scene30k"]["formatting"] == "alpaca"
    assert new_registry["Scene30k"]["columns"] == {"prompt": "q", "response": "cot"}


def test_main_writes_dest_registry(tmp_path):
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": "sub/x.jsonl"}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    rc = mpdi.main(["--source", str(source), "--dest", str(dest), "--no-symlinks"])

    assert rc == 0
    written = json.loads(dest.read_text(encoding="utf-8"))
    assert written["D"]["file_name"] == "../sub/x.jsonl"


def test_main_reports_missing_absolute_target(tmp_path, capsys):
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": "/definitely/missing/x.parquet"}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    rc = mpdi.main(["--source", str(source), "--dest", str(dest), "--no-symlinks", "--require", "D"])

    assert rc == 1
    assert "missing" in capsys.readouterr().out.lower()


def test_main_creates_symlink(tmp_path):
    target = tmp_path / "real" / "x.parquet"
    target.parent.mkdir(parents=True)
    target.write_text("data", encoding="utf-8")
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": str(target)}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    rc = mpdi.main(["--source", str(source), "--dest", str(dest)])

    assert rc == 0
    assert (tmp_path / "annotations" / "D" / "x.parquet").resolve() == target.resolve()


def test_main_is_idempotent(tmp_path):
    target = tmp_path / "real" / "x.parquet"
    target.parent.mkdir(parents=True)
    target.write_text("data", encoding="utf-8")
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": str(target)}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    assert mpdi.main(["--source", str(source), "--dest", str(dest)]) == 0
    assert mpdi.main(["--source", str(source), "--dest", str(dest)]) == 0
