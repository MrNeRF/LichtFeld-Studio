# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for error handling in lichtfeld module."""

import pytest


class TestTensorErrors:
    """Tests for Tensor error handling."""

    def test_from_numpy_invalid_type(self, lf, numpy):
        """Test from_numpy rejects unsupported dtypes."""
        # complex128 is likely not supported
        arr = numpy.array([1 + 2j, 3 + 4j], dtype=numpy.complex128)
        with pytest.raises((RuntimeError, TypeError)):
            lf.Tensor.from_numpy(arr)

    def test_reshape_preserves_numel(self, lf, numpy):
        """Test reshape preserves number of elements when valid."""
        arr = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        reshaped = t.reshape([2, 3])
        assert reshaped.shape == (2, 3)
        assert reshaped.numel == 6

    def test_invalid_dimension_access(self, lf, numpy):
        """Test accessing invalid dimension raises error."""
        arr = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        with pytest.raises((RuntimeError, IndexError)):
            t.size(5)  # Only 1 dimension

    def test_transpose_valid(self, lf, numpy):
        """Test transpose with valid dimensions."""
        arr = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        transposed = t.transpose(0, 1)
        assert transposed.shape == (2, 2)
        # Check values are transposed
        expected = arr.T
        numpy.testing.assert_allclose(transposed.numpy(), expected)

    def test_empty_tensor_operations(self, lf, numpy):
        """Test operations on empty tensor return sensible values."""
        arr = numpy.array([], dtype=numpy.float32).reshape(0, 3)
        t = lf.Tensor.from_numpy(arr)

        assert t.shape == (0, 3)
        assert t.numel == 0

    def test_cat_empty_list(self, lf):
        """Test cat with empty tensor list raises ValueError."""
        with pytest.raises(ValueError):
            lf.Tensor.cat([])

    def test_cat_mismatched_shapes(self, lf, numpy):
        """Test cat with incompatible tensor shapes raises ValueError."""
        t1 = lf.Tensor.from_numpy(numpy.array([[1.0, 2.0]], dtype=numpy.float32))
        t2 = lf.Tensor.from_numpy(numpy.array([[1.0, 2.0, 3.0]], dtype=numpy.float32))

        with pytest.raises(ValueError, match="dimension mismatch"):
            lf.Tensor.cat([t1, t2], dim=0)  # cols differ: 2 vs 3


class TestIOErrors:
    """Tests for I/O error handling."""

    def test_load_nonexistent_path(self, lf):
        """Test loading nonexistent file."""
        with pytest.raises(RuntimeError):
            lf.io.load("/nonexistent/path/to/file.ply")

    def test_load_empty_path(self, lf):
        """Test loading empty path."""
        with pytest.raises((RuntimeError, ValueError)):
            lf.io.load("")

    def test_load_directory_as_file(self, lf, tmp_output):
        """Test loading directory as file fails appropriately."""
        # tmp_output is a directory
        with pytest.raises(RuntimeError):
            lf.io.load(str(tmp_output))


class TestRunScriptErrors:
    """Tests for run() function error handling."""

    def test_run_nonexistent_script(self, lf):
        """Test running nonexistent script."""
        with pytest.raises(RuntimeError, match="not found"):
            lf.run("/nonexistent/script.py")

    def test_run_invalid_python(self, lf, tmp_output):
        """Test running invalid Python code."""
        bad_script = tmp_output / "bad_script.py"
        bad_script.write_text("this is not valid python {{{{")

        with pytest.raises(Exception):  # Could be SyntaxError or RuntimeError
            lf.run(str(bad_script))

    def test_run_script_with_exception(self, lf, tmp_output):
        """Test running script that raises exception."""
        error_script = tmp_output / "error_script.py"
        error_script.write_text("raise ValueError('Test error')")

        with pytest.raises(Exception):
            lf.run(str(error_script))


class TestOptimizerErrors:
    """Tests for Optimizer error handling without training context."""

    def test_scale_lr_without_training(self, lf):
        """Test scale_lr fails without active training."""
        opt = lf.Optimizer()
        with pytest.raises(RuntimeError):
            opt.scale_lr(0.5)

    def test_set_lr_without_training(self, lf):
        """Test set_lr fails without active training."""
        opt = lf.Optimizer()
        with pytest.raises(RuntimeError):
            opt.set_lr(0.001)


class TestModelErrors:
    """Tests for Model error handling without training context."""

    def test_clamp_without_training(self, lf):
        """Test clamp fails without active training."""
        model = lf.Model()
        with pytest.raises(RuntimeError):
            model.clamp("opacity", min=0.0, max=1.0)

    def test_scale_without_training(self, lf):
        """Test scale fails without active training."""
        model = lf.Model()
        with pytest.raises(RuntimeError):
            model.scale("scaling", 0.5)


class TestSessionErrors:
    """Tests for Session error handling without training context."""

    def test_pause_without_training(self, lf):
        """Test pause fails without active training."""
        session = lf.session()
        with pytest.raises(RuntimeError):
            session.pause()

    def test_resume_without_training(self, lf):
        """Test resume fails without active training."""
        session = lf.session()
        with pytest.raises(RuntimeError):
            session.resume()

    def test_request_stop_without_training(self, lf):
        """Test request_stop fails without active training."""
        session = lf.session()
        with pytest.raises(RuntimeError):
            session.request_stop()


class TestMat4Errors:
    """Tests for mat4() error handling."""

    def test_mat4_empty(self, lf):
        """Test mat4 with empty list."""
        with pytest.raises((RuntimeError, ValueError)):
            lf.mat4([])

    def test_mat4_wrong_row_count(self, lf):
        """Test mat4 with wrong number of rows."""
        with pytest.raises(RuntimeError, match="4 rows"):
            lf.mat4([[1.0, 0.0, 0.0, 0.0]])

    def test_mat4_wrong_col_count(self, lf):
        """Test mat4 with wrong number of columns."""
        with pytest.raises(RuntimeError, match="4 columns"):
            lf.mat4([
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ])

    def test_mat4_inconsistent_cols(self, lf):
        """Test mat4 with inconsistent column counts."""
        with pytest.raises(RuntimeError):
            lf.mat4([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],  # only 3 columns
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])


class TestIndexingErrors:
    """Tests for tensor indexing error handling."""

    def test_getitem_out_of_bounds(self, lf, numpy):
        """Test indexing out of bounds."""
        arr = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        with pytest.raises((RuntimeError, IndexError)):
            _ = t[100]  # Out of bounds

    def test_getitem_negative_overflow(self, lf, numpy):
        """Test negative indexing overflow."""
        arr = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        with pytest.raises((RuntimeError, IndexError)):
            _ = t[-100]  # Too negative


@pytest.mark.gpu
class TestCreationFactories:
    """Tests for tensor creation factory functions.

    Tensor.zeros/ones allocate on the default device, which is CUDA, so these
    raise ResourceError("Out of memory") on a machine without a GPU. They were
    unmarked and therefore selected by "not gpu" runs, failing wherever no
    device is present.
    """

    def test_zeros_creation(self, lf, numpy):
        """Test zeros factory creates zero-filled tensor."""
        t = lf.Tensor.zeros([3, 4])
        assert t.shape == (3, 4)
        assert t.numel == 12
        numpy.testing.assert_allclose(t.cpu().numpy(), numpy.zeros((3, 4)))

    def test_ones_creation(self, lf, numpy):
        """Test ones factory creates one-filled tensor."""
        t = lf.Tensor.ones([2, 3])
        assert t.shape == (2, 3)
        numpy.testing.assert_allclose(t.cpu().numpy(), numpy.ones((2, 3)))
