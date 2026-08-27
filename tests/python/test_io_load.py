# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for lichtfeld.io loading functionality."""

import pytest


class TestIOBasics:
    """Basic I/O module tests."""

    def test_io_module_exists(self, lf):
        """Test that io submodule exists."""
        assert hasattr(lf, "io")

    def test_supported_formats(self, lf):
        """Test get_supported_formats returns non-empty list."""
        formats = lf.io.get_supported_formats()
        assert isinstance(formats, list)
        assert len(formats) > 0

    def test_supported_extensions(self, lf):
        """Test get_supported_extensions returns expected extensions."""
        extensions = lf.io.get_supported_extensions()
        assert isinstance(extensions, list)
        assert ".ply" in extensions


class TestDatasetDetection:
    """Tests for dataset path detection."""

    def test_is_dataset_path_colmap(self, lf, bicycle_dataset):
        """Test COLMAP dataset detection."""
        assert lf.io.is_dataset_path(str(bicycle_dataset))

    def test_is_dataset_path_false_for_file(self, lf, tmp_output):
        """Test that regular file is not detected as dataset."""
        fake_file = tmp_output / "test.ply"
        fake_file.touch()
        assert not lf.io.is_dataset_path(str(fake_file))

    def test_is_dataset_path_nonexistent(self, lf):
        """Test that nonexistent path is not detected as dataset."""
        assert not lf.io.is_dataset_path("/nonexistent/path/to/data")


class TestLoadResult:
    """Tests for LoadResult properties."""

    @pytest.mark.slow
    def test_load_dataset_returns_load_result(self, lf, bicycle_dataset):
        """Test loading dataset returns LoadResult."""
        result = lf.io.load(
            str(bicycle_dataset), resize_factor=8, images_folder="images_8"
        )

        assert hasattr(result, "loader_used")
        assert hasattr(result, "load_time_ms")
        assert hasattr(result, "warnings")
        assert hasattr(result, "is_dataset")
        assert hasattr(result, "scene_center")

    @pytest.mark.slow
    def test_load_dataset_is_dataset_true(self, lf, bicycle_dataset):
        """Test that loading a dataset sets is_dataset=True."""
        result = lf.io.load(
            str(bicycle_dataset), resize_factor=8, images_folder="images_8"
        )
        assert result.is_dataset is True

    @pytest.mark.slow
    def test_load_dataset_has_loader_used(self, lf, bicycle_dataset):
        """Test that loader_used is set after loading."""
        result = lf.io.load(
            str(bicycle_dataset), resize_factor=8, images_folder="images_8"
        )
        assert result.loader_used
        assert len(result.loader_used) > 0

    @pytest.mark.slow
    def test_load_dataset_has_load_time(self, lf, bicycle_dataset):
        """Test that load_time_ms is recorded."""
        result = lf.io.load(
            str(bicycle_dataset), resize_factor=8, images_folder="images_8"
        )
        assert result.load_time_ms >= 0


class TestLoadPLY:
    """Tests for loading PLY files."""

    @pytest.mark.slow
    def test_load_ply_returns_splat_data(self, lf, benchmark_ply):
        """Test loading PLY returns SplatData."""
        result = lf.io.load(str(benchmark_ply))

        assert result.is_dataset is False
        assert result.splat_data is not None

    @pytest.mark.slow
    def test_load_ply_splat_data_has_points(self, lf, benchmark_ply):
        """Test loaded SplatData has points."""
        result = lf.io.load(str(benchmark_ply))
        splat = result.splat_data

        assert splat.num_points > 0

    @pytest.mark.slow
    def test_load_ply_splat_data_tensors(self, lf, benchmark_ply, numpy):
        """Test SplatData has expected tensor attributes."""
        result = lf.io.load(str(benchmark_ply))
        splat = result.splat_data

        # Test raw tensor access (means_raw is a property, not a method)
        means = splat.means_raw
        assert means.shape[0] == splat.num_points
        assert means.shape[1] == 3

        # Test computed getters
        means_computed = splat.get_means()
        numpy.testing.assert_allclose(
            means.numpy(), means_computed.numpy(), rtol=1e-5
        )


class TestLoadErrors:
    """Tests for error handling in loading."""

    def test_load_nonexistent_file(self, lf):
        """Test loading nonexistent file raises error."""
        with pytest.raises(RuntimeError, match="Failed to load"):
            lf.io.load("/nonexistent/path/to/file.ply")

    def test_load_invalid_file(self, lf, tmp_output):
        """Test loading invalid file raises error."""
        bad_file = tmp_output / "bad.ply"
        bad_file.write_text("This is not a valid PLY file")

        with pytest.raises(RuntimeError):
            lf.io.load(str(bad_file))


class TestLoadProgress:
    """Tests for progress callback during loading."""

    @pytest.mark.slow
    def test_load_with_progress_callback(self, lf, bicycle_dataset):
        """Test that progress callback is called during loading."""
        progress_calls = []

        def on_progress(progress, message):
            progress_calls.append((progress, message))

        result = lf.io.load(
            str(bicycle_dataset),
            resize_factor=8,
            images_folder="images_8",
            progress=on_progress,
        )

        # Progress callback should have been called
        assert len(progress_calls) > 0

        # Progress values should be in [0, 100] (percentage)
        for progress, _ in progress_calls:
            assert 0.0 <= progress <= 100.0


_ONE_PIXEL_PNG = bytes(
    [
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
        0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00,
        0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
        0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ]
)


class TestMissingDatasetImages:
    """Python-visible load warnings for missing dataset images."""

    def test_transforms_load_warns_and_keeps_missing_camera(self, lf, tmp_path):
        dataset = tmp_path / "missing_images"
        dataset.mkdir()
        (dataset / "present.png").write_bytes(_ONE_PIXEL_PNG)
        (dataset / "transforms.json").write_text(
            """{
  "w": 1,
  "h": 1,
  "fl_x": 1.0,
  "fl_y": 1.0,
  "cx": 0.5,
  "cy": 0.5,
  "frames": [
    {
      "file_path": "present.png",
      "transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    },
    {
      "file_path": "missing.png",
      "transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }
  ]
}
"""
        )

        result = lf.io.load(str(dataset))
        assert result.is_dataset is True
        assert any("missing.png" in warning and "missing" in warning for warning in result.warnings)

        cameras = result.cameras
        assert cameras is not None
        assert len(cameras) == 1
        assert cameras[0].image_name == "present.png"
        assert cameras[0].has_image is True

        records = cameras.cameras()
        assert len(records) == 2
        by_name = {camera.image_name: camera for camera in records}
        assert by_name["present.png"].has_image is True
        assert by_name["missing.png"].has_image is False

    def test_transforms_all_missing_fails(self, lf, tmp_path):
        dataset = tmp_path / "all_missing_images"
        dataset.mkdir()
        (dataset / "transforms.json").write_text(
            """{
  "w": 1,
  "h": 1,
  "fl_x": 1.0,
  "fl_y": 1.0,
  "cx": 0.5,
  "cy": 0.5,
  "frames": [
    {
      "file_path": "missing.png",
      "transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }
  ]
}
"""
        )

        with pytest.raises(RuntimeError, match="missing"):
            lf.io.load(str(dataset))
