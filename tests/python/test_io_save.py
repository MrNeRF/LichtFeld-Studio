# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for lichtfeld.io saving functionality."""

import pytest


def _read_ply_header(path):
    with open(path, "rb") as f:
        header_lines = []
        for line in f:
            text = line.decode("utf-8", errors="ignore").rstrip("\r\n")
            header_lines.append(text)
            if text == "end_header":
                break
    return "\n".join(header_lines)


class TestSavePLY:
    """Tests for PLY save functionality."""

    @pytest.mark.slow
    def test_save_ply_creates_file(self, lf, benchmark_ply, tmp_output):
        """Test save_ply creates output file."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "output.ply"

        lf.io.save_ply(result.splat_data, str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.slow
    def test_save_ply_binary_format(self, lf, benchmark_ply, tmp_output):
        """Test save_ply creates binary format PLY."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "output.ply"

        lf.io.save_ply(result.splat_data, str(output_path), binary=True)

        # Check that it's binary format
        header = _read_ply_header(output_path)

        assert "ply" in header
        assert "binary_little_endian" in header

    @pytest.mark.slow
    def test_save_ply_roundtrip(self, lf, benchmark_ply, tmp_output, numpy):
        """Test PLY save/load roundtrip preserves data."""
        original = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "roundtrip.ply"

        lf.io.save_ply(original.splat_data, str(output_path))
        reloaded = lf.io.load(str(output_path))

        # Point count should match
        assert reloaded.splat_data.num_points == original.splat_data.num_points

        # Means should be close (means_raw is a property, not a method)
        orig_means = original.splat_data.means_raw.numpy()
        reload_means = reloaded.splat_data.means_raw.numpy()
        numpy.testing.assert_allclose(orig_means, reload_means, rtol=1e-4)

    @pytest.mark.slow
    def test_save_ply_with_extra_scalar_attribute(self, lf, benchmark_ply, tmp_output, numpy):
        """Test save_ply writes an extra scalar per-vertex property."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "with_scalar_attr.ply"

        confidence = numpy.linspace(
            0.0, 1.0, result.splat_data.num_points, dtype=numpy.float32
        )

        lf.io.save_ply(
            result.splat_data,
            str(output_path),
            extra_attributes={"confidence": confidence},
        )

        header = _read_ply_header(output_path)
        assert "property float confidence" in header

    @pytest.mark.slow
    def test_save_ply_with_extra_vector_attribute(self, lf, benchmark_ply, tmp_output, numpy):
        """Test save_ply expands multi-column attributes into indexed PLY properties."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "with_vector_attr.ply"

        velocity = numpy.zeros((result.splat_data.num_points, 3), dtype=numpy.float32)
        velocity[:, 0] = 1.0
        velocity[:, 1] = 2.0
        velocity[:, 2] = 3.0

        lf.io.save_ply(
            result.splat_data,
            str(output_path),
            extra_attributes={"velocity": velocity},
        )

        header = _read_ply_header(output_path)
        assert "property float velocity_0" in header
        assert "property float velocity_1" in header
        assert "property float velocity_2" in header

    @pytest.mark.slow
    def test_save_ply_rejects_reserved_extra_attribute_name(self, lf, benchmark_ply, tmp_output, numpy):
        """Test save_ply rejects extra attributes that collide with reserved PLY property names."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "reserved_attr.ply"

        with pytest.raises(RuntimeError, match="reserved"):
            lf.io.save_ply(
                result.splat_data,
                str(output_path),
                extra_attributes={
                    "opacity": numpy.ones(result.splat_data.num_points, dtype=numpy.float32)
                },
            )

    @pytest.mark.slow
    def test_save_ply_rejects_duplicate_expanded_extra_attribute_name(self, lf, benchmark_ply, tmp_output, numpy):
        """Test save_ply rejects duplicate property names after vector expansion."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "duplicate_attr.ply"

        velocity = numpy.zeros((result.splat_data.num_points, 3), dtype=numpy.float32)
        velocity[:, 0] = 1.0
        velocity[:, 1] = 2.0
        velocity[:, 2] = 3.0

        with pytest.raises(RuntimeError, match="Duplicate PLY property name 'velocity_0'"):
            lf.io.save_ply(
                result.splat_data,
                str(output_path),
                extra_attributes={
                    "velocity": velocity,
                    "velocity_0": numpy.ones(result.splat_data.num_points, dtype=numpy.float32),
                },
            )


    @pytest.mark.slow
    def test_save_ply_rejects_reserved_name_after_vector_expansion(self, lf, benchmark_ply, tmp_output, numpy):
        """Test save_ply rejects names that only become reserved after [N,C] expansion."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "reserved_expanded_attr.ply"

        with pytest.raises(RuntimeError, match="scale_0"):
            lf.io.save_ply(
                result.splat_data,
                str(output_path),
                extra_attributes={
                    "scale": numpy.ones((result.splat_data.num_points, 3), dtype=numpy.float32),
                },
            )

    @pytest.mark.slow
    def test_save_ply_rejects_extra_attribute_row_count_mismatch(self, lf, benchmark_ply, tmp_output, numpy):
        """Test save_ply rejects extra attributes whose rows do not match the export size."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "mismatched_rows_attr.ply"

        with pytest.raises(RuntimeError, match="row count"):
            lf.io.save_ply(
                result.splat_data,
                str(output_path),
                extra_attributes={
                    "confidence": numpy.ones(result.splat_data.num_points - 1, dtype=numpy.float32),
                },
            )

    @pytest.mark.slow
    def test_save_point_cloud_ply_with_extra_attribute(self, lf, tmp_output, numpy):
        """Test save_point_cloud_ply forwards extra per-vertex properties."""
        scene = lf.get_scene()
        if scene is None or not scene.is_valid():
            pytest.skip("Scene not available")

        points = lf.Tensor.from_numpy(
            numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=numpy.float32)
        )
        colors = lf.Tensor.from_numpy(
            numpy.array([[255, 0, 0], [0, 255, 0]], dtype=numpy.uint8)
        )
        node_name = f"pytest_point_cloud_export_{tmp_output.name}"
        node_id = scene.add_point_cloud(node_name, points, colors)

        try:
            node = scene.get_node_by_id(node_id)
            assert node is not None
            point_cloud = node.point_cloud()
            assert point_cloud is not None

            output_path = tmp_output / "point_cloud_with_attr.ply"
            lf.io.save_point_cloud_ply(
                point_cloud,
                str(output_path),
                extra_attributes={"confidence": numpy.array([0.25, 0.75], dtype=numpy.float32)},
            )
        finally:
            scene.remove_node(node_name)

        header = _read_ply_header(output_path)
        assert "property float confidence" in header


class TestSaveSPZ:
    """Tests for SPZ (Niantic compressed) save functionality."""

    @pytest.mark.slow
    def test_save_spz_creates_file(self, lf, benchmark_ply, tmp_output):
        """Test save_spz creates output file."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "output.spz"

        lf.io.save_spz(result.splat_data, str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.slow
    def test_save_spz_compressed(self, lf, benchmark_ply, tmp_output):
        """Test SPZ is compressed (smaller than PLY)."""
        result = lf.io.load(str(benchmark_ply))

        ply_path = tmp_output / "output.ply"
        spz_path = tmp_output / "output.spz"

        lf.io.save_ply(result.splat_data, str(ply_path), binary=True)
        lf.io.save_spz(result.splat_data, str(spz_path))

        # SPZ should be smaller than binary PLY
        assert spz_path.stat().st_size < ply_path.stat().st_size


class TestSaveSOG:
    """Tests for SOG (SuperSplat) save functionality."""

    @pytest.mark.slow
    def test_save_sog_creates_file(self, lf, benchmark_ply, tmp_output):
        """Test save_sog creates output file."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "output.sog"

        lf.io.save_sog(result.splat_data, str(output_path), kmeans_iterations=5)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.slow
    def test_save_sog_with_gpu(self, lf, benchmark_ply, tmp_output, gpu_available):
        """Test save_sog with use_gpu=True."""
        if not gpu_available:
            pytest.skip("GPU not available")

        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "output_gpu.sog"

        lf.io.save_sog(
            result.splat_data, str(output_path), kmeans_iterations=5, use_gpu=True
        )

        assert output_path.exists()


class TestExportHTML:
    """Tests for HTML viewer export."""

    @pytest.mark.slow
    def test_export_html_creates_file(self, lf, benchmark_ply, tmp_output):
        """Test export_html creates output file."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "viewer.html"

        lf.io.export_html(result.splat_data, str(output_path), kmeans_iterations=5)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.slow
    def test_export_html_is_valid(self, lf, benchmark_ply, tmp_output):
        """Test exported HTML is valid HTML."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "viewer.html"

        lf.io.export_html(result.splat_data, str(output_path), kmeans_iterations=5)

        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content or "<html" in content


class TestSaveProgress:
    """Tests for progress callbacks during saving."""

    @pytest.mark.slow
    def test_save_ply_with_progress(self, lf, benchmark_ply, tmp_output):
        """Test progress callback during PLY save."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "progress.ply"

        progress_calls = []

        def on_progress(progress, stage):
            progress_calls.append((progress, stage))
            return True  # Continue

        lf.io.save_ply(result.splat_data, str(output_path), progress=on_progress)

        # File should be created (progress callbacks may not be implemented for all operations)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.slow
    def test_save_ply_completes_with_callback(self, lf, benchmark_ply, tmp_output):
        """Test that save completes successfully with progress callback provided."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "with_callback.ply"

        def on_progress(progress, stage):
            return True  # Continue

        lf.io.save_ply(result.splat_data, str(output_path), progress=on_progress)
        assert output_path.exists()
