# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for lichtfeld.scene module."""

import pytest


class TestSceneModule:
    """Basic scene module tests."""

    def test_scene_module_exists(self, lf):
        """Test that scene submodule exists."""
        assert hasattr(lf, "scene")

    def test_get_scene_returns_none_without_context(self, lf):
        """Test get_scene returns None when no training/GUI context."""
        scene = lf.get_scene()
        assert scene is None


class TestSplatDataAfterLoad:
    """Tests for SplatData access after loading a PLY."""

    @pytest.fixture
    def loaded_splat(self, lf, benchmark_ply):
        """Load a PLY and return SplatData."""
        result = lf.io.load(str(benchmark_ply))
        return result.splat_data

    @pytest.mark.slow
    def test_splat_data_num_points(self, loaded_splat):
        """Test SplatData has correct point count."""
        assert loaded_splat.num_points > 0

    @pytest.mark.slow
    def test_splat_data_sh_degree(self, loaded_splat):
        """Test SplatData has SH degree set."""
        assert loaded_splat.sh_degree >= 0

    @pytest.mark.slow
    def test_splat_data_means_shape(self, loaded_splat, numpy):
        """Test means tensor has correct shape."""
        means = loaded_splat.get_means()
        assert means.ndim == 2
        assert means.shape[0] == loaded_splat.num_points
        assert means.shape[1] == 3

    @pytest.mark.slow
    def test_splat_data_opacities_shape(self, loaded_splat, numpy):
        """Test opacities tensor has correct shape."""
        opacities = loaded_splat.get_opacities()
        assert opacities.ndim == 2
        assert opacities.shape[0] == loaded_splat.num_points
        assert opacities.shape[1] == 1

    @pytest.mark.slow
    def test_splat_data_scalings_shape(self, loaded_splat, numpy):
        """Test scalings tensor has correct shape."""
        scalings = loaded_splat.get_scalings()
        assert scalings.ndim == 2
        assert scalings.shape[0] == loaded_splat.num_points
        assert scalings.shape[1] == 3

    @pytest.mark.slow
    def test_splat_data_rotations_shape(self, loaded_splat, numpy):
        """Test rotations tensor has correct shape."""
        rotations = loaded_splat.get_rotations()
        assert rotations.ndim == 2
        assert rotations.shape[0] == loaded_splat.num_points
        assert rotations.shape[1] == 4  # quaternion

    @pytest.mark.slow
    def test_splat_data_features_dc(self, loaded_splat, numpy):
        """Test features_dc tensor has correct shape."""
        features_dc = loaded_splat.get_features_dc()
        assert features_dc.ndim == 3
        assert features_dc.shape[0] == loaded_splat.num_points
        # DC is (N, 1, 3) typically
        assert features_dc.shape[2] == 3

    @pytest.mark.slow
    def test_reserve_capacity_grows_plain_splat(self, loaded_splat):
        """Plain (non-interop) SplatData can reserve extra capacity."""
        n = loaded_splat.num_points
        loaded_splat.reserve_capacity(n + 1024)
        assert loaded_splat.num_points == n
        assert loaded_splat.means_raw.shape[0] == n

        # Renderer-backed (vulkan_external_buffer / splat.exportable) models
        # must raise from Python. That assertion needs the GUI app and is
        # skipped here.


class TestTensorOperations:
    """Tests for tensor operations within scene context."""

    def test_mat4_creation(self, lf, numpy):
        """Test mat4 creates 4x4 matrix tensor."""
        identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        mat = lf.mat4(identity)

        assert mat.shape == (4, 4)
        numpy.testing.assert_allclose(mat.numpy(), numpy.eye(4))

    def test_mat4_translation(self, lf, numpy):
        """Test mat4 can represent translation."""
        translation = [
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, 10.0],
            [0.0, 0.0, 1.0, 15.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        mat = lf.mat4(translation)

        assert mat.shape == (4, 4)
        assert abs(mat.numpy()[0, 3] - 5.0) < 1e-6
        assert abs(mat.numpy()[1, 3] - 10.0) < 1e-6
        assert abs(mat.numpy()[2, 3] - 15.0) < 1e-6

    def test_mat4_wrong_rows(self, lf):
        """Test mat4 raises error for wrong number of rows."""
        bad_matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        with pytest.raises(RuntimeError, match="4 rows"):
            lf.mat4(bad_matrix)

    def test_mat4_wrong_cols(self, lf):
        """Test mat4 raises error for wrong number of columns."""
        bad_matrix = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
        with pytest.raises(RuntimeError, match="4 columns"):
            lf.mat4(bad_matrix)


class TestPointCloudAccess:
    """Tests for PointCloud access from loaded data."""

    @pytest.fixture
    def point_cloud(self, lf, benchmark_ply):
        """Load a PLY and return as PointCloud if available."""
        result = lf.io.load(str(benchmark_ply))
        # Access point cloud through splat_data
        splat = result.splat_data
        if splat is None:
            pytest.skip("No splat data in loaded result")
        return splat

    @pytest.mark.slow
    def test_point_cloud_size(self, point_cloud):
        """Test point cloud reports correct size."""
        assert point_cloud.num_points > 0

    @pytest.mark.slow
    def test_point_cloud_means_are_valid(self, point_cloud, numpy):
        """Test point cloud means contain finite values."""
        means = point_cloud.get_means()
        arr = means.cpu().numpy()
        assert numpy.all(numpy.isfinite(arr))

    @pytest.mark.slow
    def test_point_cloud_opacities_in_range(self, point_cloud, numpy):
        """Test opacities are in valid range after activation."""
        opacities = point_cloud.get_opacities()
        arr = opacities.cpu().numpy()
        # After sigmoid, should be in (0, 1)
        assert numpy.all(arr >= 0.0)
        assert numpy.all(arr <= 1.0)


class TestListScene:
    """Tests for list_scene utility function."""

    def test_list_scene_runs_without_context(self, lf, capsys):
        """Test list_scene doesn't crash without scene context."""
        lf.list_scene()
        captured = capsys.readouterr()
        assert "No scene available" in captured.out


class TestHelpFunction:
    """Tests for help utility function."""

    def test_help_function_exists(self, lf):
        """Test help function exists."""
        assert hasattr(lf, "help")
        assert callable(lf.help)

    def test_help_prints_info(self, lf, capsys):
        """Test help prints usage information."""
        lf.help()
        captured = capsys.readouterr()
        assert "LichtFeld Python API" in captured.out
        assert "get_scene" in captured.out
        assert "context" in captured.out
