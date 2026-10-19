# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sign propagates NaN without changing finite sign semantics."""

import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("size", [1, 7, 1025])
@pytest.mark.parametrize("chain", [False, True])
def test_sign_propagates_nan(lf, numpy, device, size, chain):
    values = numpy.resize(numpy.array([-numpy.inf, -3, -0., 0., 1e-30, 2, numpy.inf], dtype=numpy.float32), size)
    values[0] = values[size // 2] = values[-1] = numpy.nan
    tensor = getattr(lf.Tensor.from_numpy(values), device)()
    result = tensor.sign()
    if chain:
        result = result + 0
    numpy.testing.assert_array_equal(result.numpy(), numpy.sign(values))
    numpy.testing.assert_array_equal(tensor.numpy(), values)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_sign_noncontiguous_and_integer_controls(lf, numpy, device):
    values = numpy.array([[numpy.nan, -2, 0], [3, numpy.nan, numpy.inf]], dtype=numpy.float32)
    tensor = getattr(lf.Tensor.from_numpy(values), device)().permute((1, 0))
    numpy.testing.assert_array_equal(tensor.sign().numpy(), numpy.sign(values.T))
    integers = numpy.array([-7, 0, 9], dtype=numpy.int32)
    integer_tensor = getattr(lf.Tensor.from_numpy(integers), device)()
    numpy.testing.assert_array_equal(integer_tensor.sign().numpy(), numpy.sign(integers))
