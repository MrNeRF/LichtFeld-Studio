# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Half precision tensors support scalar and nested Python value extraction."""

import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("value", [1.25, -2.5, 0., -0., float("inf"), -float("inf"), float("nan")])
def test_float16_item(lf, numpy, device, value):
    tensor = getattr(lf.Tensor.from_numpy(numpy.array([value], dtype=numpy.float32)), device)().to("float16")
    numpy.testing.assert_equal(tensor.item(), float(numpy.float16(value)))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_float16_nested_and_strided_lists(lf, numpy, device):
    array = numpy.array([[1.25, -2.5, 0.], [numpy.inf, numpy.nan, 65504]], dtype=numpy.float32)
    tensor = getattr(lf.Tensor.from_numpy(array), device)().to("float16")
    numpy.testing.assert_equal(tensor.tolist(), array.astype(numpy.float16).tolist())
    numpy.testing.assert_equal(tensor.permute((1, 0)).tolist(), array.astype(numpy.float16).T.tolist())
    assert lf.Tensor.zeros((0, 3), dtype="float16", device=device).tolist() == []


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_float16_integer_and_boolean_conversion(lf, device):
    tensor = lf.Tensor.full((1,), 2.5, device=device).to("float16")
    assert tensor.int_() == 2
    assert tensor.bool_() is True
    assert lf.Tensor.zeros((1,), device=device).to("float16").bool_() is False
