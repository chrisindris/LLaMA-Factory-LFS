# Copyright 2025 the LlamaFactory team.

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


h5py = pytest.importorskip("h5py")
_MODULE_PATH = Path(__file__).resolve().parents[2] / "src/llamafactory/data/data_packing/h5py_data.py"
_SPEC = spec_from_file_location("scannet_h5py_data_test", _MODULE_PATH)
h5py_data = module_from_spec(_SPEC)
_SPEC.loader.exec_module(h5py_data)


def _write_scene(root: Path, name: str, images: list[bytes]) -> Path:
    scene = root / name
    scene.mkdir()
    with h5py.File(scene / "images.hdf5", "w") as handle:
        data = handle.create_dataset("binary_data", (len(images),), dtype=h5py.vlen_dtype(np.dtype("uint8")))
        for index, image in enumerate(images):
            data[index] = np.frombuffer(image, dtype="uint8")

    (scene / "image_mapping.json").write_text(json.dumps({str(index): index for index in range(len(images))}))
    return scene


@pytest.fixture(autouse=True)
def clear_reader_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SCANNET_H5_DIR", str(tmp_path))
    reset = getattr(h5py_data, "_reset_scannet_cache", lambda: None)
    reset()
    yield
    reset()


def test_reuses_scene_file_and_mapping_without_changing_image_bytes(tmp_path):
    scene = _write_scene(tmp_path, "scene0000_00", [b"first", b"second"])
    real_h5_open = h5py.File
    real_json_open = open
    opened_h5 = []
    opened_json = []

    def open_h5(path, mode):
        opened_h5.append(str(path))
        return real_h5_open(path, mode)

    def open_json(path, mode):
        opened_json.append(str(path))
        return real_json_open(path, mode)

    with (
        patch.object(h5py_data.h5py, "File", side_effect=open_h5),
        patch.object(h5py_data, "open", open_json, create=True),
    ):
        assert h5py_data.retrieve_image(output_dir=tmp_path, scene_name=scene.name, image_name="0.jpg") == b"first"
        assert h5py_data.retrieve_image(output_dir=tmp_path, scene_name=scene.name, image_name="1.jpg") == b"second"

    assert opened_h5 == [str(scene / "images.hdf5")]
    assert opened_json == [str(scene / "image_mapping.json")]


def test_reopens_evicted_scene_without_reordering_images(tmp_path):
    scenes = [_write_scene(tmp_path, f"scene{index:04d}_00", [f"image{index}".encode()]) for index in range(9)]
    real_h5_open = h5py.File
    opened = []
    handles = []

    def open_h5(path, mode):
        opened.append(str(path))
        handle = real_h5_open(path, mode)
        handles.append(handle)
        return handle

    with patch.object(h5py_data.h5py, "File", side_effect=open_h5):
        for index, scene in enumerate(scenes):
            assert h5py_data.retrieve_image(output_dir=tmp_path, scene_name=scene.name, image_name="0.jpg") == (
                f"image{index}".encode()
            )

        assert not handles[0].id.valid
        assert handles[8].id.valid
        assert (
            h5py_data.retrieve_image(output_dir=tmp_path, scene_name=scenes[0].name, image_name="0.jpg") == b"image0"
        )
        assert (
            h5py_data.retrieve_image(output_dir=tmp_path, scene_name=scenes[1].name, image_name="0.jpg") == b"image1"
        )

    assert opened.count(str(scenes[0] / "images.hdf5")) == 2
    assert opened.count(str(scenes[1] / "images.hdf5")) == 2
    assert opened.count(str(scenes[8] / "images.hdf5")) == 1


def test_pid_change_discards_inherited_handle(tmp_path):
    scene = _write_scene(tmp_path, "scene0000_00", [b"payload"])
    real_open = h5py.File
    opened = []
    handles = []

    def open_h5(path, mode):
        opened.append(str(path))
        handle = real_open(path, mode)
        handles.append(handle)
        return handle

    with patch.object(h5py_data.h5py, "File", side_effect=open_h5):
        assert h5py_data.retrieve_image(output_dir=tmp_path, scene_name=scene.name, image_name="0.jpg") == b"payload"
        assert handles[0].id.valid
        with patch.object(h5py_data.os, "getpid", return_value=-1):
            assert (
                h5py_data.retrieve_image(output_dir=tmp_path, scene_name=scene.name, image_name="0.jpg") == b"payload"
            )
        assert not handles[0].id.valid

    assert opened == [str(scene / "images.hdf5")] * 2


def test_relative_paths_do_not_reuse_another_working_directory(tmp_path, monkeypatch):
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        _write_scene(root, "scene0000_00", [name.encode()])
        monkeypatch.chdir(root)
        assert h5py_data.retrieve_image(output_dir=".", scene_name="scene0000_00", image_name="0.jpg") == name.encode()
