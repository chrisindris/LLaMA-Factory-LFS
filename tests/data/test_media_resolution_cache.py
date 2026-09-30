# Copyright 2025 the LlamaFactory team.

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from llamafactory.data.converter import _resolve_media_path, get_dataset_converter
from llamafactory.data.data_packing.h5_image_store import _probe_scannet_scene
from llamafactory.data.parser import DatasetAttr
from llamafactory.hparams import DataArguments


class TestMediaResolutionCache(TestCase):
    def setUp(self):
        _resolve_media_path.cache_clear()
        _probe_scannet_scene.cache_clear()
        self.converter = get_dataset_converter("alpaca", DatasetAttr("file", "test"), DataArguments(media_dir="media"))

    def tearDown(self):
        _resolve_media_path.cache_clear()
        _probe_scannet_scene.cache_clear()

    def test_repeated_images_probe_once_and_keep_order(self):
        paths = ["a.jpg", "b.jpg", "a.jpg"]
        with (
            patch("os.path.isfile", return_value=False) as files,
            patch("llamafactory.data.converter.can_resolve_h5_image", return_value=True) as probe,
        ):
            assert self.converter._find_medias(paths) == paths
            assert self.converter._find_medias(paths) == paths
            assert probe.call_count == 2
            assert files.call_count == 4

    def test_real_file_precedes_h5(self):
        with TemporaryDirectory() as directory:
            image = Path(directory) / "a.jpg"
            image.touch()
            self.converter.data_args.media_dir = directory
            with patch("llamafactory.data.converter.can_resolve_h5_image") as probe:
                assert self.converter._find_medias(["a.jpg"]) == [str(image)]
                probe.assert_not_called()

    def test_joined_h5_fallback_and_video_frames(self):
        with (
            patch("os.path.isfile", return_value=False),
            patch(
                "llamafactory.data.converter.can_resolve_h5_image", side_effect=lambda p: p.startswith("media/")
            ) as probe,
        ):
            assert self.converter._find_medias([["a.jpg"]]) == [["media/a.jpg"]]
            assert probe.call_count == 2

    def test_failed_lookup_is_retried(self):
        with (
            patch("os.path.isfile", return_value=False),
            patch("llamafactory.data.converter.can_resolve_h5_image", return_value=False) as probe,
        ):
            with self.assertRaises(ValueError):
                self.converter._find_medias(["a.jpg"])
            probe.return_value = True
            assert self.converter._find_medias(["a.jpg"]) == ["a.jpg"]

    def test_root_change_invalidates_media_cache(self):
        with (
            patch("os.path.isfile", return_value=False),
            patch("llamafactory.data.converter.can_resolve_h5_image", return_value=True) as probe,
        ):
            for root in ("one", "two"):
                with patch.dict(os.environ, {"SCANNET_H5_DIR": root}):
                    self.converter._find_medias(["a.jpg"])
            assert probe.call_count == 2

    def test_scene_cache_reuses_checks_and_tracks_root(self):
        with TemporaryDirectory() as directory:
            for root in ("one", "two"):
                scene = Path(directory) / root / "scene0000_00"
                scene.mkdir(parents=True)
                (scene / "images.hdf5").touch()
                (scene / "image_mapping.json").touch()
                with patch.dict(os.environ, {"SCANNET_H5_DIR": str(scene.parent)}):
                    args = (
                        str(Path(directory) / "absent"),
                        "scene0000_00",
                        str(scene.parent),
                        os.getcwd(),
                        os.getpid(),
                    )
                    assert _probe_scannet_scene(*args)
                    with patch.object(Path, "is_file", side_effect=AssertionError("repeated filesystem check")):
                        assert _probe_scannet_scene(*args)
            assert _probe_scannet_scene.cache_info().misses == 2

    def test_worker_and_cwd_changes_invalidate_media_cache(self):
        with (
            patch("os.path.isfile", return_value=False),
            patch("llamafactory.data.converter.can_resolve_h5_image", return_value=True) as probe,
        ):
            for pid, cwd in ((1, "/one"), (2, "/one"), (2, "/two")):
                with patch("os.getpid", return_value=pid), patch("os.getcwd", return_value=cwd):
                    self.converter._find_medias(["a.jpg"])
            assert probe.call_count == 3

    def test_missing_scene_can_be_staged_after_failure(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {"SCANNET_H5_DIR": "/missing/root"}):
            args = (directory, "scene0000_00", "/missing/root", os.getcwd(), os.getpid())
            with self.assertRaises(FileNotFoundError):
                _probe_scannet_scene(*args)
            scene = Path(directory) / "scene0000_00"
            scene.mkdir()
            (scene / "images.hdf5").touch()
            (scene / "image_mapping.json").touch()
            assert _probe_scannet_scene(*args)
