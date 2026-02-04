# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Comprehensive tensor tests comparing lichtfeld against PyTorch."""

import pytest

np = pytest.importorskip("numpy")

# Test shape configurations
SHAPES_1D = [(5,), (100,), (1,)]
SHAPES_2D = [(3, 4), (10, 10), (1, 5), (5, 1)]
SHAPES_3D = [(2, 3, 4), (1, 1, 10), (4, 4, 4)]
SHAPES_ALL = SHAPES_1D + SHAPES_2D + SHAPES_3D
DTYPES_FLOAT = ["float32"]
DTYPES_INT = ["int32", "int64"]
DTYPES_ALL = DTYPES_FLOAT + DTYPES_INT


def assert_tensors_close(lf_tensor, torch_tensor, rtol=1e-5, atol=1e-8):
    """Compare lf tensor against torch tensor."""
    lf_np = lf_tensor.cpu().numpy() if lf_tensor.is_cuda else lf_tensor.numpy()
    torch_np = torch_tensor.detach().cpu().numpy()
    np.testing.assert_allclose(lf_np, torch_np, rtol=rtol, atol=atol)


def assert_shapes_equal(lf_tensor, torch_tensor):
    """Compare tensor shapes."""
    assert lf_tensor.shape == tuple(torch_tensor.shape), \
        f"Shape mismatch: lf={lf_tensor.shape} vs torch={tuple(torch_tensor.shape)}"


def lf_dtype_to_torch(lf_dtype, torch):
    """Convert lf dtype string to torch dtype."""
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    return mapping.get(lf_dtype, torch.float32)


def torch_dtype_to_np(torch_dtype, torch):
    """Convert torch dtype to numpy dtype."""
    mapping = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.bool: np.bool_,
    }
    return mapping.get(torch_dtype, np.float32)


class TestDLPackRoundtrip:
    """Test DLPack interoperability between lf and PyTorch."""

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_lf_to_torch_cuda_float32(self, lf, torch, gpu_available, shape):
        """lf.Tensor (CUDA) -> torch.Tensor via DLPack."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(*shape).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()

        # Export to torch
        t_torch = torch.from_dlpack(t_lf)

        assert tuple(t_torch.shape) == shape
        assert t_torch.device.type == "cuda"
        np.testing.assert_allclose(t_torch.cpu().numpy(), np_data, rtol=1e-6)

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_torch_to_lf_cuda_float32(self, lf, torch, gpu_available, shape):
        """torch.Tensor (CUDA) -> lf.Tensor via DLPack."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(*shape).astype(np.float32)
        t_torch = torch.from_numpy(np_data).cuda()

        # Import to lf
        t_lf = lf.Tensor.from_dlpack(t_torch)

        assert t_lf.shape == shape
        assert t_lf.is_cuda
        np.testing.assert_allclose(t_lf.cpu().numpy(), np_data, rtol=1e-6)

    @pytest.mark.gpu
    def test_roundtrip_lf_torch_lf(self, lf, torch, gpu_available):
        """lf -> torch -> lf roundtrip preserves data."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(10, 5).astype(np.float32)
        t_lf1 = lf.Tensor.from_numpy(np_data).cuda()

        t_torch = torch.from_dlpack(t_lf1)
        t_lf2 = lf.Tensor.from_dlpack(t_torch)

        assert t_lf2.shape == t_lf1.shape
        np.testing.assert_allclose(t_lf2.cpu().numpy(), np_data, rtol=1e-6)

    @pytest.mark.gpu
    def test_roundtrip_torch_lf_torch(self, lf, torch, gpu_available):
        """torch -> lf -> torch roundtrip preserves data."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(8, 4).astype(np.float32)
        t_torch1 = torch.from_numpy(np_data).cuda()

        t_lf = lf.Tensor.from_dlpack(t_torch1)
        t_torch2 = torch.from_dlpack(t_lf)

        assert tuple(t_torch2.shape) == tuple(t_torch1.shape)
        np.testing.assert_allclose(t_torch2.cpu().numpy(), np_data, rtol=1e-6)

    @pytest.mark.parametrize("shape", SHAPES_2D)
    def test_cpu_float32(self, lf, torch, shape):
        """CPU tensor DLPack roundtrip."""
        np_data = np.random.randn(*shape).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data)

        t_torch = torch.from_dlpack(t_lf)
        assert t_torch.device.type == "cpu"
        np.testing.assert_allclose(t_torch.numpy(), np_data, rtol=1e-6)

    @pytest.mark.gpu
    def test_int32_cuda(self, lf, torch, gpu_available):
        """int32 tensor DLPack."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()

        t_torch = torch.from_dlpack(t_lf)
        assert t_torch.dtype == torch.int32
        np.testing.assert_array_equal(t_torch.cpu().numpy(), np_data)

    @pytest.mark.gpu
    def test_int64_cuda(self, lf, torch, gpu_available):
        """int64 tensor DLPack."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()

        t_torch = torch.from_dlpack(t_lf)
        assert t_torch.dtype == torch.int64
        np.testing.assert_array_equal(t_torch.cpu().numpy(), np_data)

    @pytest.mark.gpu
    def test_non_contiguous_transpose(self, lf, torch, gpu_available):
        """Non-contiguous tensor from transpose."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(4, 6).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_lf_t = t_lf.transpose(0, 1)

        # Make contiguous before DLPack export
        t_lf_contig = t_lf_t.contiguous()
        t_torch = torch.from_dlpack(t_lf_contig)

        expected = np_data.T
        np.testing.assert_allclose(t_torch.cpu().numpy(), expected, rtol=1e-6)


class TestCreationFunctions:
    """Test tensor creation functions against PyTorch."""

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_zeros(self, lf, torch, gpu_available, shape):
        """zeros() creates correct zero tensor."""
        if not gpu_available:
            pytest.skip("GPU not available")

        t_lf = lf.Tensor.zeros(shape, device="cuda", dtype="float32")
        t_torch = torch.zeros(shape, device="cuda", dtype=torch.float32)

        assert_shapes_equal(t_lf, t_torch)
        assert_tensors_close(t_lf, t_torch)

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_ones(self, lf, torch, gpu_available, shape):
        """ones() creates correct tensor of ones."""
        if not gpu_available:
            pytest.skip("GPU not available")

        t_lf = lf.Tensor.ones(shape, device="cuda", dtype="float32")
        t_torch = torch.ones(shape, device="cuda", dtype=torch.float32)

        assert_shapes_equal(t_lf, t_torch)
        assert_tensors_close(t_lf, t_torch)

    @pytest.mark.gpu
    @pytest.mark.parametrize("value", [0.0, 1.0, -5.5, 3.14159])
    def test_full(self, lf, torch, gpu_available, value):
        """full() creates tensor with correct fill value."""
        if not gpu_available:
            pytest.skip("GPU not available")

        shape = (4, 5)
        t_lf = lf.Tensor.full(shape, value, device="cuda", dtype="float32")
        t_torch = torch.full(shape, value, device="cuda", dtype=torch.float32)

        assert_tensors_close(t_lf, t_torch)

    @pytest.mark.gpu
    def test_arange_end_only(self, lf, torch, gpu_available):
        """arange(end) matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        # Use start=0 to match our API which requires start,end for device arg
        t_lf = lf.Tensor.arange(0, 10, 1, device="cuda", dtype="float32")
        t_torch = torch.arange(10, device="cuda", dtype=torch.float32)

        assert_tensors_close(t_lf, t_torch)

    @pytest.mark.gpu
    def test_arange_start_end(self, lf, torch, gpu_available):
        """arange(start, end) matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        t_lf = lf.Tensor.arange(5, 15, device="cuda", dtype="float32")
        t_torch = torch.arange(5, 15, device="cuda", dtype=torch.float32)

        assert_tensors_close(t_lf, t_torch)

    @pytest.mark.gpu
    def test_arange_with_step(self, lf, torch, gpu_available):
        """arange(start, end, step) matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        t_lf = lf.Tensor.arange(0, 10, 2, device="cuda", dtype="float32")
        t_torch = torch.arange(0, 10, 2, device="cuda", dtype=torch.float32)

        assert_tensors_close(t_lf, t_torch)

    @pytest.mark.gpu
    def test_linspace(self, lf, torch, gpu_available):
        """linspace matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        t_lf = lf.Tensor.linspace(0, 1, 11, device="cuda", dtype="float32")
        t_torch = torch.linspace(0, 1, 11, device="cuda", dtype=torch.float32)

        assert_tensors_close(t_lf, t_torch)

    @pytest.mark.gpu
    def test_eye_square(self, lf, torch, gpu_available):
        """eye(n) creates identity matrix."""
        if not gpu_available:
            pytest.skip("GPU not available")

        t_lf = lf.Tensor.eye(5, device="cuda", dtype="float32")
        t_torch = torch.eye(5, device="cuda", dtype=torch.float32)

        assert_tensors_close(t_lf, t_torch)

    @pytest.mark.gpu
    def test_eye_rectangular(self, lf, torch, gpu_available):
        """eye(m, n) creates rectangular identity."""
        if not gpu_available:
            pytest.skip("GPU not available")

        t_lf = lf.Tensor.eye(3, 5, device="cuda", dtype="float32")
        t_torch = torch.eye(3, 5, device="cuda", dtype=torch.float32)

        assert_tensors_close(t_lf, t_torch)


class TestUnaryOperations:
    """Test unary operations against PyTorch."""

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_exp(self, lf, torch, gpu_available, shape):
        """exp() matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(*shape).astype(np.float32)
        np_data = np.clip(np_data, -10, 10)  # Avoid overflow
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.exp()
        r_torch = torch.exp(t_torch)

        assert_tensors_close(r_lf, r_torch, rtol=1e-5)

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_log(self, lf, torch, gpu_available, shape):
        """log() matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.abs(np.random.randn(*shape).astype(np.float32)) + 0.1
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.log()
        r_torch = torch.log(t_torch)

        # Slightly looser tolerance for transcendental functions
        assert_tensors_close(r_lf, r_torch, rtol=1e-4)

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_sqrt(self, lf, torch, gpu_available, shape):
        """sqrt() matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.abs(np.random.randn(*shape).astype(np.float32)) + 0.01
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.sqrt()
        r_torch = torch.sqrt(t_torch)

        assert_tensors_close(r_lf, r_torch, rtol=1e-5)

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_sigmoid(self, lf, torch, gpu_available, shape):
        """sigmoid() matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(*shape).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.sigmoid()
        r_torch = torch.sigmoid(t_torch)

        assert_tensors_close(r_lf, r_torch, rtol=1e-5)

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_relu(self, lf, torch, gpu_available, shape):
        """relu() matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(*shape).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.relu()
        r_torch = torch.relu(t_torch)

        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_sin(self, lf, torch, gpu_available):
        """sin() matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(10, 5).astype(np.float32) * 3.14
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.sin()
        r_torch = torch.sin(t_torch)

        # Looser tolerance for transcendental functions
        assert_tensors_close(r_lf, r_torch, rtol=1e-4)

    @pytest.mark.gpu
    def test_cos(self, lf, torch, gpu_available):
        """cos() matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(10, 5).astype(np.float32) * 3.14
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.cos()
        r_torch = torch.cos(t_torch)

        # Looser tolerance for transcendental functions
        assert_tensors_close(r_lf, r_torch, rtol=1e-4)

    @pytest.mark.gpu
    def test_tanh(self, lf, torch, gpu_available):
        """tanh() matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(10, 5).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.tanh()
        r_torch = torch.tanh(t_torch)

        assert_tensors_close(r_lf, r_torch, rtol=1e-5)

    @pytest.mark.gpu
    def test_abs(self, lf, torch, gpu_available):
        """abs() matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(10, 5).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.abs()
        r_torch = torch.abs(t_torch)

        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_neg(self, lf, torch, gpu_available):
        """Negation matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(10, 5).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = -t_lf
        r_torch = -t_torch

        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_floor(self, lf, torch, gpu_available):
        """floor() matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(10, 5).astype(np.float32) * 10
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.floor()
        r_torch = torch.floor(t_torch)

        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_ceil(self, lf, torch, gpu_available):
        """ceil() matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(10, 5).astype(np.float32) * 10
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.ceil()
        r_torch = torch.ceil(t_torch)

        assert_tensors_close(r_lf, r_torch)


class TestBinaryOperations:
    """Test binary operations against PyTorch."""

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_add_tensors(self, lf, torch, gpu_available, shape):
        """Tensor addition matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_a = np.random.randn(*shape).astype(np.float32)
        np_b = np.random.randn(*shape).astype(np.float32)

        t_lf_a = lf.Tensor.from_numpy(np_a).cuda()
        t_lf_b = lf.Tensor.from_numpy(np_b).cuda()
        t_torch_a = torch.from_numpy(np_a).cuda()
        t_torch_b = torch.from_numpy(np_b).cuda()

        r_lf = t_lf_a + t_lf_b
        r_torch = t_torch_a + t_torch_b

        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_sub_tensors(self, lf, torch, gpu_available, shape):
        """Tensor subtraction matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_a = np.random.randn(*shape).astype(np.float32)
        np_b = np.random.randn(*shape).astype(np.float32)

        t_lf_a = lf.Tensor.from_numpy(np_a).cuda()
        t_lf_b = lf.Tensor.from_numpy(np_b).cuda()
        t_torch_a = torch.from_numpy(np_a).cuda()
        t_torch_b = torch.from_numpy(np_b).cuda()

        r_lf = t_lf_a - t_lf_b
        r_torch = t_torch_a - t_torch_b

        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_mul_tensors(self, lf, torch, gpu_available, shape):
        """Tensor multiplication matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_a = np.random.randn(*shape).astype(np.float32)
        np_b = np.random.randn(*shape).astype(np.float32)

        t_lf_a = lf.Tensor.from_numpy(np_a).cuda()
        t_lf_b = lf.Tensor.from_numpy(np_b).cuda()
        t_torch_a = torch.from_numpy(np_a).cuda()
        t_torch_b = torch.from_numpy(np_b).cuda()

        r_lf = t_lf_a * t_lf_b
        r_torch = t_torch_a * t_torch_b

        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_div_tensors(self, lf, torch, gpu_available, shape):
        """Tensor division matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_a = np.random.randn(*shape).astype(np.float32)
        np_b = np.random.randn(*shape).astype(np.float32)
        np_b = np.where(np.abs(np_b) < 0.1, 0.1, np_b)  # Avoid div by ~0

        t_lf_a = lf.Tensor.from_numpy(np_a).cuda()
        t_lf_b = lf.Tensor.from_numpy(np_b).cuda()
        t_torch_a = torch.from_numpy(np_a).cuda()
        t_torch_b = torch.from_numpy(np_b).cuda()

        r_lf = t_lf_a / t_lf_b
        r_torch = t_torch_a / t_torch_b

        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    @pytest.mark.parametrize("scalar", [2.0, 0.5, -1.0, 10.0])
    def test_add_scalar(self, lf, torch, gpu_available, scalar):
        """Tensor + scalar matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(5, 4).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf + scalar
        r_torch = t_torch + scalar

        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    @pytest.mark.parametrize("scalar", [2.0, 0.5, -1.0, 10.0])
    def test_mul_scalar(self, lf, torch, gpu_available, scalar):
        """Tensor * scalar matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(5, 4).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf * scalar
        r_torch = t_torch * scalar

        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_scalar_sub_tensor(self, lf, torch, gpu_available):
        """scalar - tensor matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(5, 4).astype(np.float32)
        scalar = 5.0
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = scalar - t_lf
        r_torch = scalar - t_torch

        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_scalar_div_tensor(self, lf, torch, gpu_available):
        """scalar / tensor matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(5, 4).astype(np.float32)
        np_data = np.where(np.abs(np_data) < 0.1, 0.1, np_data)
        scalar = 10.0
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = scalar / t_lf
        r_torch = scalar / t_torch

        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_eq(self, lf, torch, gpu_available):
        """== comparison matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_a = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        np_b = np.array([[1, 0, 3], [0, 2, 0]], dtype=np.float32)

        t_lf_a = lf.Tensor.from_numpy(np_a).cuda()
        t_lf_b = lf.Tensor.from_numpy(np_b).cuda()
        t_torch_a = torch.from_numpy(np_a).cuda()
        t_torch_b = torch.from_numpy(np_b).cuda()

        r_lf = t_lf_a == t_lf_b
        r_torch = t_torch_a == t_torch_b

        assert r_lf.dtype == "bool"
        np.testing.assert_array_equal(r_lf.cpu().numpy(), r_torch.cpu().numpy())

    @pytest.mark.gpu
    def test_lt(self, lf, torch, gpu_available):
        """< comparison matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_a = np.random.randn(5, 4).astype(np.float32)
        np_b = np.random.randn(5, 4).astype(np.float32)

        t_lf_a = lf.Tensor.from_numpy(np_a).cuda()
        t_lf_b = lf.Tensor.from_numpy(np_b).cuda()
        t_torch_a = torch.from_numpy(np_a).cuda()
        t_torch_b = torch.from_numpy(np_b).cuda()

        r_lf = t_lf_a < t_lf_b
        r_torch = t_torch_a < t_torch_b

        np.testing.assert_array_equal(r_lf.cpu().numpy(), r_torch.cpu().numpy())

    @pytest.mark.gpu
    def test_gt(self, lf, torch, gpu_available):
        """> comparison matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_a = np.random.randn(5, 4).astype(np.float32)
        np_b = np.random.randn(5, 4).astype(np.float32)

        t_lf_a = lf.Tensor.from_numpy(np_a).cuda()
        t_lf_b = lf.Tensor.from_numpy(np_b).cuda()
        t_torch_a = torch.from_numpy(np_a).cuda()
        t_torch_b = torch.from_numpy(np_b).cuda()

        r_lf = t_lf_a > t_lf_b
        r_torch = t_torch_a > t_torch_b

        np.testing.assert_array_equal(r_lf.cpu().numpy(), r_torch.cpu().numpy())


class TestBroadcasting:
    """Test broadcasting behavior matches PyTorch."""

    @pytest.mark.gpu
    def test_same_shape(self, lf, torch, gpu_available):
        """Same shape addition."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_a = np.random.randn(3, 4).astype(np.float32)
        np_b = np.random.randn(3, 4).astype(np.float32)

        t_lf_a = lf.Tensor.from_numpy(np_a).cuda()
        t_lf_b = lf.Tensor.from_numpy(np_b).cuda()
        t_torch_a = torch.from_numpy(np_a).cuda()
        t_torch_b = torch.from_numpy(np_b).cuda()

        r_lf = t_lf_a + t_lf_b
        r_torch = t_torch_a + t_torch_b

        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_broadcast_1d_to_2d(self, lf, torch, gpu_available):
        """(4,) + (3, 4) broadcasts."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_a = np.random.randn(4).astype(np.float32)
        np_b = np.random.randn(3, 4).astype(np.float32)

        t_lf_a = lf.Tensor.from_numpy(np_a).cuda()
        t_lf_b = lf.Tensor.from_numpy(np_b).cuda()
        t_torch_a = torch.from_numpy(np_a).cuda()
        t_torch_b = torch.from_numpy(np_b).cuda()

        r_lf = t_lf_a + t_lf_b
        r_torch = t_torch_a + t_torch_b

        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_broadcast_cross(self, lf, torch, gpu_available):
        """(1, 4) + (3, 1) broadcasts to (3, 4)."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_a = np.random.randn(1, 4).astype(np.float32)
        np_b = np.random.randn(3, 1).astype(np.float32)

        t_lf_a = lf.Tensor.from_numpy(np_a).cuda()
        t_lf_b = lf.Tensor.from_numpy(np_b).cuda()
        t_torch_a = torch.from_numpy(np_a).cuda()
        t_torch_b = torch.from_numpy(np_b).cuda()

        r_lf = t_lf_a + t_lf_b
        r_torch = t_torch_a + t_torch_b

        assert r_lf.shape == (3, 4)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_broadcast_scalar_like(self, lf, torch, gpu_available):
        """(1,) + (5, 3) broadcasts."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_a = np.array([5.0], dtype=np.float32)
        np_b = np.random.randn(5, 3).astype(np.float32)

        t_lf_a = lf.Tensor.from_numpy(np_a).cuda()
        t_lf_b = lf.Tensor.from_numpy(np_b).cuda()
        t_torch_a = torch.from_numpy(np_a).cuda()
        t_torch_b = torch.from_numpy(np_b).cuda()

        r_lf = t_lf_a + t_lf_b
        r_torch = t_torch_a + t_torch_b

        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_broadcast_3d(self, lf, torch, gpu_available):
        """(2, 1, 4) + (3, 4) broadcasts to (2, 3, 4)."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_a = np.random.randn(2, 1, 4).astype(np.float32)
        np_b = np.random.randn(3, 4).astype(np.float32)

        t_lf_a = lf.Tensor.from_numpy(np_a).cuda()
        t_lf_b = lf.Tensor.from_numpy(np_b).cuda()
        t_torch_a = torch.from_numpy(np_a).cuda()
        t_torch_b = torch.from_numpy(np_b).cuda()

        r_lf = t_lf_a + t_lf_b
        r_torch = t_torch_a + t_torch_b

        assert r_lf.shape == (2, 3, 4)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_broadcast_mul(self, lf, torch, gpu_available):
        """Broadcasting with multiplication."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_a = np.random.randn(4, 1).astype(np.float32)
        np_b = np.random.randn(1, 5).astype(np.float32)

        t_lf_a = lf.Tensor.from_numpy(np_a).cuda()
        t_lf_b = lf.Tensor.from_numpy(np_b).cuda()
        t_torch_a = torch.from_numpy(np_a).cuda()
        t_torch_b = torch.from_numpy(np_b).cuda()

        r_lf = t_lf_a * t_lf_b
        r_torch = t_torch_a * t_torch_b

        assert r_lf.shape == (4, 5)
        assert_tensors_close(r_lf, r_torch)


class TestReductions:
    """Test reduction operations against PyTorch."""

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_sum_full(self, lf, torch, gpu_available, shape):
        """Full sum matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(*shape).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.sum_scalar()
        r_torch = t_torch.sum().item()

        assert abs(r_lf - r_torch) < 1e-4 * abs(r_torch) + 1e-5

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_mean_full(self, lf, torch, gpu_available, shape):
        """Full mean matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(*shape).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.mean_scalar()
        r_torch = t_torch.mean().item()

        assert abs(r_lf - r_torch) < 1e-5

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_max_full(self, lf, torch, gpu_available, shape):
        """Full max matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(*shape).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.max_scalar()
        r_torch = t_torch.max().item()

        assert abs(r_lf - r_torch) < 1e-6

    @pytest.mark.gpu
    @pytest.mark.parametrize("shape", SHAPES_ALL)
    def test_min_full(self, lf, torch, gpu_available, shape):
        """Full min matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(*shape).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.min_scalar()
        r_torch = t_torch.min().item()

        assert abs(r_lf - r_torch) < 1e-6

    @pytest.mark.gpu
    def test_sum_dim0(self, lf, torch, gpu_available):
        """Sum along dim 0 matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(4, 5).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.sum(dim=0)
        r_torch = t_torch.sum(dim=0)

        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch, rtol=1e-4)

    @pytest.mark.gpu
    def test_sum_dim1(self, lf, torch, gpu_available):
        """Sum along dim 1 matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(4, 5).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.sum(dim=1)
        r_torch = t_torch.sum(dim=1)

        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch, rtol=1e-4)

    @pytest.mark.gpu
    def test_sum_keepdim(self, lf, torch, gpu_available):
        """Sum with keepdim=True matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(4, 5).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.sum(dim=1, keepdim=True)
        r_torch = t_torch.sum(dim=1, keepdim=True)

        assert r_lf.shape == (4, 1)
        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch, rtol=1e-4)

    @pytest.mark.gpu
    def test_mean_dim(self, lf, torch, gpu_available):
        """Mean along dimension matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(4, 5, 3).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.mean(dim=1)
        r_torch = t_torch.mean(dim=1)

        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch, rtol=1e-4)

    @pytest.mark.gpu
    def test_max_dim(self, lf, torch, gpu_available):
        """Max along dimension matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(4, 5).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.max(dim=0)
        r_torch = t_torch.max(dim=0).values

        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_sum_negative_dim(self, lf, torch, gpu_available):
        """Sum with negative dim matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(4, 5, 3).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.sum(dim=-1)
        r_torch = t_torch.sum(dim=-1)

        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch, rtol=1e-4)


class TestShapeOperations:
    """Test shape manipulation operations against PyTorch."""

    @pytest.mark.gpu
    def test_reshape_flatten(self, lf, torch, gpu_available):
        """reshape to 1D matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(3, 4, 5).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.reshape([-1])
        r_torch = t_torch.reshape([-1])

        assert r_lf.shape == (60,)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_reshape_2d(self, lf, torch, gpu_available):
        """reshape to 2D matches torch."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(12).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.reshape([3, 4])
        r_torch = t_torch.reshape([3, 4])

        assert r_lf.shape == (3, 4)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_reshape_infer_dim(self, lf, torch, gpu_available):
        """reshape with -1 infers dimension."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(24).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.reshape([4, -1])
        r_torch = t_torch.reshape([4, -1])

        assert r_lf.shape == (4, 6)
        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_squeeze_all(self, lf, torch, gpu_available):
        """squeeze() removes all size-1 dims."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(1, 3, 1, 4, 1).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.squeeze()
        r_torch = t_torch.squeeze()

        assert r_lf.shape == (3, 4)
        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_squeeze_dim(self, lf, torch, gpu_available):
        """squeeze(dim) removes specific size-1 dim."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(1, 3, 4).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.squeeze(0)
        r_torch = t_torch.squeeze(0)

        assert r_lf.shape == (3, 4)
        assert_shapes_equal(r_lf, r_torch)

    @pytest.mark.gpu
    def test_unsqueeze(self, lf, torch, gpu_available):
        """unsqueeze adds dimension."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(3, 4).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.unsqueeze(0)
        r_torch = t_torch.unsqueeze(0)

        assert r_lf.shape == (1, 3, 4)
        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_unsqueeze_end(self, lf, torch, gpu_available):
        """unsqueeze(-1) adds at end."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(3, 4).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.unsqueeze(-1)
        r_torch = t_torch.unsqueeze(-1)

        assert r_lf.shape == (3, 4, 1)
        assert_shapes_equal(r_lf, r_torch)

    @pytest.mark.gpu
    def test_transpose_2d(self, lf, torch, gpu_available):
        """transpose swaps dimensions."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(3, 5).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.transpose(0, 1)
        r_torch = t_torch.transpose(0, 1)

        assert r_lf.shape == (5, 3)
        assert_shapes_equal(r_lf, r_torch)
        # Need contiguous for comparison
        assert_tensors_close(r_lf.contiguous(), r_torch.contiguous())

    @pytest.mark.gpu
    def test_transpose_3d(self, lf, torch, gpu_available):
        """transpose on 3D tensor."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(2, 3, 4).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.transpose(1, 2)
        r_torch = t_torch.transpose(1, 2)

        assert r_lf.shape == (2, 4, 3)
        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf.contiguous(), r_torch.contiguous())

    @pytest.mark.gpu
    def test_permute(self, lf, torch, gpu_available):
        """permute reorders dimensions."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(2, 3, 4).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.permute([2, 0, 1])
        r_torch = t_torch.permute([2, 0, 1])

        assert r_lf.shape == (4, 2, 3)
        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf.contiguous(), r_torch.contiguous())

    @pytest.mark.gpu
    def test_flatten_full(self, lf, torch, gpu_available):
        """flatten() flattens entire tensor."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(2, 3, 4).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.flatten()
        r_torch = t_torch.flatten()

        assert r_lf.shape == (24,)
        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_flatten_partial(self, lf, torch, gpu_available):
        """flatten(start, end) flattens range."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(2, 3, 4, 5).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.flatten(1, 2)
        r_torch = t_torch.flatten(1, 2)

        assert r_lf.shape == (2, 12, 5)
        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch)


class TestIndexingSlicing:
    """Test indexing and slicing against PyTorch."""

    @pytest.mark.gpu
    def test_single_index(self, lf, torch, gpu_available):
        """t[0] gets first row."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(5, 4).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf[0]
        r_torch = t_torch[0]

        assert r_lf.shape == (4,)
        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_negative_index(self, lf, torch, gpu_available):
        """t[-1] gets last row."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(5, 4).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf[-1]
        r_torch = t_torch[-1]

        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_slice_range(self, lf, torch, gpu_available):
        """t[1:3] slices rows."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(5, 4).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf[1:3]
        r_torch = t_torch[1:3]

        assert r_lf.shape == (2, 4)
        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_slice_open_end(self, lf, torch, gpu_available):
        """t[2:] slices from index to end."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(5, 4).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf[2:]
        r_torch = t_torch[2:]

        assert r_lf.shape == (3, 4)
        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_slice_open_start(self, lf, torch, gpu_available):
        """t[:3] slices from start."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(5, 4).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf[:3]
        r_torch = t_torch[:3]

        assert r_lf.shape == (3, 4)
        assert_shapes_equal(r_lf, r_torch)
        assert_tensors_close(r_lf, r_torch)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.gpu
    def test_single_element(self, lf, torch, gpu_available):
        """Single element tensor operations."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.array([5.0], dtype=np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        assert t_lf.item() == pytest.approx(5.0)
        assert t_lf.sum_scalar() == pytest.approx(5.0)

        r_lf = t_lf * 2
        r_torch = t_torch * 2

        assert_tensors_close(r_lf, r_torch)

    @pytest.mark.gpu
    def test_large_tensor(self, lf, torch, gpu_available):
        """Large tensor operations."""
        if not gpu_available:
            pytest.skip("GPU not available")

        shape = (1000, 500)
        np_data = np.random.randn(*shape).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf.sum_scalar()
        r_torch = t_torch.sum().item()

        # Allow larger tolerance for accumulated errors
        assert abs(r_lf - r_torch) < 1e-2 * abs(r_torch) + 1e-3

    @pytest.mark.gpu
    def test_nan_handling(self, lf, torch, gpu_available):
        """NaN values handled correctly."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        # Operations should preserve NaN
        r_lf = t_lf * 2
        r_torch = t_torch * 2

        lf_np = r_lf.cpu().numpy()
        torch_np = r_torch.cpu().numpy()

        # Check non-NaN values match
        assert lf_np[0] == pytest.approx(2.0)
        assert lf_np[2] == pytest.approx(6.0)
        # Check NaN preserved
        assert np.isnan(lf_np[1])

    @pytest.mark.gpu
    def test_inf_handling(self, lf, torch, gpu_available):
        """Inf values handled correctly."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.array([1.0, np.inf, -np.inf], dtype=np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf + 1
        r_torch = t_torch + 1

        lf_np = r_lf.cpu().numpy()

        assert lf_np[0] == pytest.approx(2.0)
        assert np.isinf(lf_np[1]) and lf_np[1] > 0
        assert np.isinf(lf_np[2]) and lf_np[2] < 0

    @pytest.mark.gpu
    def test_very_small_values(self, lf, torch, gpu_available):
        """Very small values don't underflow incorrectly."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.array([1e-38, 1e-37, 1e-36], dtype=np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()
        t_torch = torch.from_numpy(np_data).cuda()

        r_lf = t_lf * 1e10
        r_torch = t_torch * 1e10

        assert_tensors_close(r_lf, r_torch, rtol=1e-5)

    @pytest.mark.gpu
    def test_clone_independence(self, lf, torch, gpu_available):
        """Clone creates independent copy."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()

        t_clone = t_lf.clone()

        # Original should be unchanged after clone operations
        original_sum = t_lf.sum_scalar()
        clone_sum = t_clone.sum_scalar()

        assert original_sum == pytest.approx(clone_sum)
        assert original_sum == pytest.approx(6.0)

    @pytest.mark.gpu
    def test_contiguous_from_transpose(self, lf, torch, gpu_available):
        """Contiguous materializes non-contiguous tensor."""
        if not gpu_available:
            pytest.skip("GPU not available")

        np_data = np.random.randn(4, 6).astype(np.float32)
        t_lf = lf.Tensor.from_numpy(np_data).cuda()

        t_t = t_lf.transpose(0, 1)
        # Transpose is typically non-contiguous
        t_contig = t_t.contiguous()

        assert t_contig.is_contiguous
        assert t_contig.shape == (6, 4)

        # Values should match transposed numpy
        expected = np_data.T
        np.testing.assert_allclose(t_contig.cpu().numpy(), expected, rtol=1e-6)
