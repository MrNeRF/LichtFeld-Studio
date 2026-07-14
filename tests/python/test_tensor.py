# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for lichtfeld.Tensor class."""

import pytest


class TestTensorCreation:
    """Tests for tensor creation and basic properties."""

    def test_from_numpy_float32(self, lf, numpy):
        """Test creating tensor from float32 numpy array."""
        arr = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        assert t.shape == (3,)
        assert t.dtype == "float32"
        assert t.numel == 3
        assert t.ndim == 1

    def test_from_numpy_2d(self, lf, numpy):
        """Test creating 2D tensor from numpy array."""
        arr = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        assert t.shape == (2, 2)
        assert t.ndim == 2
        assert t.numel == 4
        assert t.size(0) == 2
        assert t.size(1) == 2

    def test_from_numpy_int32(self, lf, numpy):
        """Test creating tensor from int32 numpy array."""
        arr = numpy.array([1, 2, 3], dtype=numpy.int32)
        t = lf.Tensor.from_numpy(arr)

        assert t.dtype == "int32"
        assert t.numel == 3

    def test_device_cpu(self, lf, numpy):
        """Test that tensor is on CPU by default."""
        arr = numpy.array([1.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        assert t.device == "cpu"
        assert not t.is_cuda


class TestTensorRuntimePrimitives:
    """Tensor behaviors used by NumPy-free Python and UI workflows."""

    def test_sort_returns_values_and_int64_indices(self, lf):
        values = lf.Tensor.zeros([5], dtype="float32", device="cpu")
        for index, value in enumerate([3.0, 1.0, 4.0, 1.5, 2.0]):
            values[index] = value

        sorted_values, sorted_indices = values.sort(0, False)

        assert sorted_values.dtype == "float32"
        assert sorted_indices.dtype == "int64"
        assert sorted_values.shape == (5,)
        assert sorted_indices.shape == (5,)
        assert [sorted_values[i].item() for i in range(5)] == [
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
        ]
        assert [sorted_indices[i].int_() for i in range(5)] == [1, 3, 4, 0, 2]

    def test_uint8_selection_assignment_from_bool_mask(self, lf):
        mask = lf.Tensor.zeros([5], dtype="bool", device="cpu")
        mask[1] = 1
        mask[3] = 1
        selection = lf.Tensor.zeros([5], dtype="uint8", device="cpu")

        selection[mask] = 2

        assert selection.tolist() == [0, 2, 0, 2, 0]

    def test_tolist_and_count_nonzero(self, lf):
        values = lf.Tensor.zeros([4], dtype="int32", device="cpu")
        values[0] = 2
        values[2] = 5

        assert values.tolist() == [2, 0, 5, 0]
        assert values.count_nonzero() == 2

    def test_index_add_counts_occurrences(self, lf):
        counts = lf.Tensor.zeros([4], dtype="int32", device="cpu")
        indices = lf.Tensor.zeros([4], dtype="int32", device="cpu")
        for index, value in enumerate([0, 2, 2, 3]):
            indices[index] = value

        counts.index_add_(
            0, indices, lf.Tensor.ones([4], dtype="int32", device="cpu")
        )

        assert counts.tolist() == [1, 0, 2, 1]


class TestTensorArithmetic:
    """Tests for tensor arithmetic operations."""

    def test_add_tensors(self, lf, numpy):
        """Test tensor addition."""
        arr1 = numpy.array([1.0, 2.0], dtype=numpy.float32)
        arr2 = numpy.array([3.0, 4.0], dtype=numpy.float32)
        t1 = lf.Tensor.from_numpy(arr1)
        t2 = lf.Tensor.from_numpy(arr2)

        result = t1 + t2
        expected = numpy.array([4.0, 6.0], dtype=numpy.float32)

        numpy.testing.assert_allclose(result.numpy(), expected)

    def test_add_scalar(self, lf, numpy):
        """Test tensor + scalar."""
        arr = numpy.array([1.0, 2.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        result = t + 10.0
        expected = numpy.array([11.0, 12.0], dtype=numpy.float32)

        numpy.testing.assert_allclose(result.numpy(), expected)

    def test_subtract_tensors(self, lf, numpy):
        """Test tensor subtraction."""
        arr1 = numpy.array([5.0, 6.0], dtype=numpy.float32)
        arr2 = numpy.array([1.0, 2.0], dtype=numpy.float32)
        t1 = lf.Tensor.from_numpy(arr1)
        t2 = lf.Tensor.from_numpy(arr2)

        result = t1 - t2
        expected = numpy.array([4.0, 4.0], dtype=numpy.float32)

        numpy.testing.assert_allclose(result.numpy(), expected)

    def test_multiply_tensors(self, lf, numpy):
        """Test tensor multiplication."""
        arr1 = numpy.array([2.0, 3.0], dtype=numpy.float32)
        arr2 = numpy.array([4.0, 5.0], dtype=numpy.float32)
        t1 = lf.Tensor.from_numpy(arr1)
        t2 = lf.Tensor.from_numpy(arr2)

        result = t1 * t2
        expected = numpy.array([8.0, 15.0], dtype=numpy.float32)

        numpy.testing.assert_allclose(result.numpy(), expected)

    def test_multiply_scalar(self, lf, numpy):
        """Test tensor * scalar."""
        arr = numpy.array([2.0, 3.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        result = t * 2.0
        expected = numpy.array([4.0, 6.0], dtype=numpy.float32)

        numpy.testing.assert_allclose(result.numpy(), expected)

    def test_divide_tensors(self, lf, numpy):
        """Test tensor division."""
        arr1 = numpy.array([6.0, 8.0], dtype=numpy.float32)
        arr2 = numpy.array([2.0, 4.0], dtype=numpy.float32)
        t1 = lf.Tensor.from_numpy(arr1)
        t2 = lf.Tensor.from_numpy(arr2)

        result = t1 / t2
        expected = numpy.array([3.0, 2.0], dtype=numpy.float32)

        numpy.testing.assert_allclose(result.numpy(), expected)

    def test_negate(self, lf, numpy):
        """Test tensor negation."""
        arr = numpy.array([1.0, -2.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        result = -t
        expected = numpy.array([-1.0, 2.0], dtype=numpy.float32)

        numpy.testing.assert_allclose(result.numpy(), expected)


class TestTensorReduction:
    """Tests for tensor reduction operations."""

    def test_sum_all(self, lf, numpy):
        """Test sum over all elements."""
        arr = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        total = t.sum_scalar()
        assert abs(total - 10.0) < 1e-6

    def test_mean_all(self, lf, numpy):
        """Test mean over all elements."""
        arr = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        mean = t.mean_scalar()
        assert abs(mean - 2.5) < 1e-6

    def test_max_all(self, lf, numpy):
        """Test max over all elements."""
        arr = numpy.array([[1.0, 5.0], [3.0, 2.0]], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        maximum = t.max_scalar()
        assert abs(maximum - 5.0) < 1e-6

    def test_min_all(self, lf, numpy):
        """Test min over all elements."""
        arr = numpy.array([[1.0, 5.0], [3.0, 2.0]], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        minimum = t.min_scalar()
        assert abs(minimum - 1.0) < 1e-6

    def test_sum_dim(self, lf, numpy):
        """Test sum along specific dimension."""
        arr = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        result = t.sum(dim=0)
        expected = numpy.array([4.0, 6.0], dtype=numpy.float32)

        numpy.testing.assert_allclose(result.numpy(), expected)

    def test_sum_dim_keepdim(self, lf, numpy):
        """Test sum with keepdim=True."""
        arr = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        result = t.sum(dim=1, keepdim=True)
        assert result.shape == (2, 1)


class TestTensorShape:
    """Tests for tensor shape operations."""

    def test_reshape(self, lf, numpy):
        """Test tensor reshape."""
        arr = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        reshaped = t.reshape([2, 3])
        assert reshaped.shape == (2, 3)
        assert reshaped.numel == 6

    def test_squeeze(self, lf, numpy):
        """Test tensor squeeze."""
        arr = numpy.array([[[1.0, 2.0]]], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        assert t.shape == (1, 1, 2)
        squeezed = t.squeeze()
        assert squeezed.shape == (2,)

    def test_unsqueeze(self, lf, numpy):
        """Test tensor unsqueeze."""
        arr = numpy.array([1.0, 2.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        unsqueezed = t.unsqueeze(0)
        assert unsqueezed.shape == (1, 2)

    def test_transpose(self, lf, numpy):
        """Test tensor transpose."""
        arr = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        transposed = t.transpose(0, 1)
        assert transposed.shape == (3, 2)

    def test_flatten(self, lf, numpy):
        """Test tensor flatten."""
        arr = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        flat = t.flatten()
        assert flat.shape == (4,)


class TestTensorNumpy:
    """Tests for NumPy interoperability."""

    def test_numpy_copy(self, lf, numpy):
        """Test numpy() returns copy by default."""
        arr = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        result = t.numpy(copy=True)
        result[0] = 999.0

        # Original tensor should be unchanged
        numpy.testing.assert_allclose(t.numpy(), arr)

    def test_numpy_roundtrip(self, lf, numpy):
        """Test numpy -> tensor -> numpy roundtrip."""
        original = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(original)
        result = t.numpy()

        numpy.testing.assert_allclose(result, original)

    def test_array_protocol(self, lf, numpy):
        """Test __array__ protocol for numpy integration."""
        arr = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        # Should work with numpy.asarray
        as_array = numpy.asarray(t)
        numpy.testing.assert_allclose(as_array, arr)


class TestTensorElementwise:
    """Tests for elementwise operations."""

    def test_sigmoid(self, lf, numpy):
        """Test sigmoid activation."""
        arr = numpy.array([0.0, 1.0, -1.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        result = t.sigmoid()
        expected = 1.0 / (1.0 + numpy.exp(-arr))

        numpy.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_exp(self, lf, numpy):
        """Test exponential."""
        arr = numpy.array([0.0, 1.0, 2.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        result = t.exp()
        expected = numpy.exp(arr)

        numpy.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_log(self, lf, numpy):
        """Test logarithm."""
        arr = numpy.array([1.0, 2.0, 10.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        result = t.log()
        expected = numpy.log(arr)

        numpy.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_sqrt(self, lf, numpy):
        """Test square root."""
        arr = numpy.array([1.0, 4.0, 9.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        result = t.sqrt()
        expected = numpy.sqrt(arr)

        numpy.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_relu(self, lf, numpy):
        """Test ReLU activation."""
        arr = numpy.array([-1.0, 0.0, 1.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        result = t.relu()
        expected = numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32)

        numpy.testing.assert_allclose(result.numpy(), expected)


class TestTensorClone:
    """Tests for tensor cloning and memory operations."""

    def test_clone(self, lf, numpy):
        """Test tensor clone creates independent copy."""
        arr = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)
        t_clone = t.clone()

        # Verify same values
        numpy.testing.assert_allclose(t_clone.numpy(), t.numpy())

    def test_contiguous(self, lf, numpy):
        """Test contiguous returns contiguous tensor."""
        arr = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)

        contig = t.contiguous()
        assert contig.is_contiguous


@pytest.mark.gpu
class TestTensorGPU:
    """Tests for GPU tensor operations."""

    def test_cpu_to_cuda(self, lf, numpy, gpu_available):
        """Test moving tensor from CPU to CUDA."""
        if not gpu_available:
            pytest.skip("GPU not available")

        arr = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
        t_cpu = lf.Tensor.from_numpy(arr)

        t_cuda = t_cpu.cuda()
        assert t_cuda.is_cuda
        assert "cuda" in t_cuda.device

    def test_cuda_to_cpu(self, lf, numpy, gpu_available):
        """Test moving tensor from CUDA to CPU."""
        if not gpu_available:
            pytest.skip("GPU not available")

        arr = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
        t = lf.Tensor.from_numpy(arr)
        t_cuda = t.cuda()
        t_back = t_cuda.cpu()

        assert t_back.device == "cpu"
        numpy.testing.assert_allclose(t_back.numpy(), arr)

    def test_gpu_arithmetic(self, lf, numpy, gpu_available):
        """Test arithmetic on GPU tensors."""
        if not gpu_available:
            pytest.skip("GPU not available")

        arr1 = numpy.array([1.0, 2.0], dtype=numpy.float32)
        arr2 = numpy.array([3.0, 4.0], dtype=numpy.float32)

        t1 = lf.Tensor.from_numpy(arr1).cuda()
        t2 = lf.Tensor.from_numpy(arr2).cuda()

        result = t1 + t2
        expected = numpy.array([4.0, 6.0], dtype=numpy.float32)

        numpy.testing.assert_allclose(result.cpu().numpy(), expected)
