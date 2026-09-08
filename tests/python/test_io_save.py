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

    def test_save_ply_default_writes_provenance(self, lf, benchmark_ply, tmp_output):
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "default_provenance.ply"

        lf.io.save_ply(result.splat_data, str(output_path))

        assert b"lichtfeld_provenance" in output_path.read_bytes()

    def test_save_ply_strip_writes_minimal_stamp(self, lf, benchmark_ply, tmp_output):
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "no_provenance.ply"

        lf.io.save_ply(result.splat_data, str(output_path), include_provenance=False)

        data = output_path.read_bytes()
        assert b"lichtfeld_provenance" in data
        assert b"build_commit" in data
        assert b"export_id" not in data

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

    def test_save_spz_default_writes_v4(self, lf, benchmark_ply, tmp_output):
        """Default save_spz writes SPZ v4 (NGSP/zstd container)."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "default_v4.spz"

        lf.io.save_spz(result.splat_data, str(output_path))

        with open(output_path, "rb") as f:
            assert f.read(4) == b"NGSP"

    def test_save_spz_version3_writes_gzip(self, lf, benchmark_ply, tmp_output):
        """save_spz(version=3) writes legacy gzip container."""
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "escape_v3.spz"

        lf.io.save_spz(result.splat_data, str(output_path), version=3)

        with open(output_path, "rb") as f:
            assert f.read(2) == b"\x1f\x8b"

    def test_save_spz_default_writes_provenance(self, lf, benchmark_ply, tmp_output):
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "default_provenance.spz"

        lf.io.save_spz(result.splat_data, str(output_path))

        assert b"lichtfeld_provenance" in output_path.read_bytes()

    def test_save_spz_strip_writes_minimal_stamp(self, lf, benchmark_ply, tmp_output):
        result = lf.io.load(str(benchmark_ply))
        output_path = tmp_output / "no_provenance.spz"

        lf.io.save_spz(result.splat_data, str(output_path), include_provenance=False)

        data = output_path.read_bytes()
        assert b"lichtfeld_provenance" in data
        assert b"build_commit" in data
        assert b"export_id" not in data


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


class TestSaveSSOG:
    """Nightly CUDA roundtrip using a small synthetic splat, without a GUI."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_ssog_roundtrip(self, lf, numpy, gpu_available, tmp_output):
        if not gpu_available:
            pytest.skip("GPU not available")
        n = 256
        fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                  "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        points = numpy.zeros(n, dtype=[(name, "<f4") for name in fields])
        rng = numpy.random.default_rng(42)
        for name in ("x", "y", "z"):
            points[name] = rng.uniform(-1, 1, n)
        for name in ("f_dc_0", "f_dc_1", "f_dc_2"):
            points[name] = rng.uniform(-0.5, 0.5, n)
        for name in ("scale_0", "scale_1", "scale_2"):
            points[name] = -3.0
        points["rot_0"] = 1.0
        points["opacity"] = 2.0
        source = tmp_output / "synthetic.ply"
        header = "ply\nformat binary_little_endian 1.0\nelement vertex 256\n"
        header += "".join(f"property float {name}\n" for name in fields) + "end_header\n"
        source.write_bytes(header.encode() + points.tobytes())
        splat = lf.io.load(str(source)).splat_data
        output = tmp_output / "ssog_directory"
        progress = []

        def on_progress(value, stage):
            progress.append((value, stage))
            return True

        lf.io.save_ssog(splat, str(output), lod_levels=2, chunk_count_k=1,
                                chunk_min_k=0, kmeans_iterations=2, progress=on_progress)
        assert (output / "lod-meta.json").is_file()
        assert progress
        for path in (output, output / "lod-meta.json"):
            loaded = lf.io.load(str(path)).splat_data
            assert loaded.num_points == n
            actual = loaded.get_means().cpu().numpy()
            expected = numpy.stack([points[name] for name in ("x", "y", "z")], axis=1)
            # The chunk writer reorders points; compare nearest original positions.
            distances = numpy.linalg.norm(actual[:, None, :] - expected[None, :, :], axis=2)
            assert float(distances.min(axis=1).max()) < 0.01
            assert numpy.isfinite(loaded.get_scaling().cpu().numpy()).all()

        import zipfile
        bundle = tmp_output / "scene.ssog"
        lf.io.save_ssog(splat, bundle, lod_levels=2, chunk_count_k=1,
                        chunk_min_k=0, kmeans_iterations=2)
        with zipfile.ZipFile(bundle) as archive:
            assert archive.namelist()[-1] == "lod-meta.json"
            assert all(info.compress_type == (zipfile.ZIP_STORED if info.filename.endswith(".webp") else zipfile.ZIP_DEFLATED)
                       for info in archive.infolist())
            archive.extractall(tmp_output / "unzipped")
        bundled = lf.io.load(bundle).splat_data
        unzipped = lf.io.load(tmp_output / "unzipped").splat_data
        assert bundled.num_points == unzipped.num_points == n
        numpy.testing.assert_array_equal(bundled.get_means().cpu().numpy(), unzipped.get_means().cpu().numpy())
        numpy.testing.assert_array_equal(bundled.get_scaling().cpu().numpy(), unzipped.get_scaling().cpu().numpy())
        with pytest.raises(RuntimeError, match="[Cc]ancel"):
            lf.io.save_ssog(splat, tmp_output / "cancelled.ssog", progress=lambda _value, _stage: False)
        assert not (tmp_output / "cancelled.ssog").exists()

        with pytest.raises(RuntimeError, match="[Cc]ancel"):
            lf.io.save_ssog(splat, str(tmp_output / "cancelled"),
                                    progress=lambda _value, _stage: False)
