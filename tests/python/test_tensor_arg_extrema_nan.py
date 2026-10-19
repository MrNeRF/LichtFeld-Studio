# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Indexed extrema choose the first NaN, including an initial NaN."""

import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("operation", ["argmin", "argmax"])
@pytest.mark.parametrize("values", [
    [float("nan"), float("nan")], [float("nan")] * 257,
    [float("nan"), 2, float("nan")], [2, float("nan"), float("nan")],
    [3, 3, 3], [-float("inf"), -float("inf"), float("inf")],
])
def test_arg_extrema_first_nan(lf, numpy, device, operation, values):
    array = numpy.array(values, dtype=numpy.float32)
    tensor = getattr(lf.Tensor.from_numpy(array), device)()
    assert getattr(tensor, operation)().int_() == getattr(numpy, operation)(array)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("operation", ["argmin", "argmax"])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("transpose", [False, True])
def test_arg_extrema_nan_axis_and_views(lf, numpy, device, operation, axis, transpose):
    array = numpy.array([[numpy.nan, 2, numpy.nan], [numpy.nan, numpy.nan, 1]], dtype=numpy.float32)
    tensor = getattr(lf.Tensor.from_numpy(array), device)()
    if transpose:
        array = array.T
        tensor = tensor.permute((1, 0))
    numpy.testing.assert_array_equal(getattr(tensor, operation)(dim=axis).numpy(), getattr(numpy, operation)(array, axis=axis))
    assert getattr(tensor, operation)().int_() == getattr(numpy, operation)(array)
