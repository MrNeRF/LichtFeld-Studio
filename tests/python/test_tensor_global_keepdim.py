# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Global reductions preserve each reduced axis when keepdim is requested."""

import pytest

OPERATIONS = ["sum", "prod", "min", "max", "mean", "std", "var", "argmin", "argmax", "all", "any"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 2, 2)])
@pytest.mark.parametrize("operation", OPERATIONS)
def test_global_reduction_keepdim(lf, numpy, device, shape, operation):
    array = numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape)
    if operation in ("all", "any"):
        array = array.astype(bool)
    tensor = getattr(lf.Tensor.from_numpy(array), device)()
    kwargs = {"ddof": 1} if operation in ("std", "var") else {}
    expected = getattr(numpy, operation)(array, keepdims=True, **kwargs)
    result = getattr(tensor, operation)(dim=None, keepdim=True)
    assert result.shape == (1,) * len(shape)
    numpy.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-6)
    reduced = getattr(tensor, operation)(keepdim=False)
    assert reduced.shape == ()
    numpy.testing.assert_allclose(reduced.numpy(), expected.reshape(()), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("operation", OPERATIONS)
def test_global_keepdim_on_noncontiguous_input(lf, numpy, device, operation):
    array = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
    if operation in ("all", "any"):
        array = array.astype(bool)
    tensor = getattr(lf.Tensor.from_numpy(array), device)().permute((1, 0))
    result = getattr(tensor, operation)(keepdim=True)
    expected = getattr(numpy, operation)(array.T, keepdims=True, **({"ddof": 1} if operation in ("std", "var") else {}))
    assert result.shape == (1, 1)
    numpy.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scalar_global_keepdim_retains_rank_zero(lf, device):
    tensor = lf.Tensor.full((), 2, device=device)
    for operation in ("sum", "mean", "min", "max", "prod", "argmin", "argmax", "all", "any"):
        operand = tensor.to("bool") if operation in ("all", "any") else tensor
        assert getattr(operand, operation)(keepdim=True).shape == ()
