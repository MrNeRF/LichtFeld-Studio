# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Boolean row assignment follows the same leading axis as row selection."""

import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("shape,mask_values", [
    ((2, 2), [True, False]), ((3, 5), [True, False, True]),
    ((3, 2, 4), [False, True, True]), ((3, 5), [False] * 3),
    ((3, 5), [True] * 3),
])
@pytest.mark.parametrize("tensor_value", [False, True])
def test_row_mask_assignment(lf, numpy, device, shape, mask_values, tensor_value):
    array = numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape)
    tensor = getattr(lf.Tensor.from_numpy(array), device)()
    mask_array = numpy.array(mask_values, dtype=bool)
    mask = getattr(lf.Tensor.from_numpy(mask_array), device)()
    expected = array.copy()
    if tensor_value:
        values = numpy.full(expected[mask_array].shape, 17, dtype=numpy.float32)
        value = lf.Tensor.full(values.shape, 17, device=device)
    else:
        values = value = 17
    expected[mask_array] = values
    tensor[mask] = value
    numpy.testing.assert_array_equal(tensor.numpy(), expected)
    numpy.testing.assert_array_equal(tensor[mask].numpy(), expected[mask_array])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_row_mask_on_transposed_view(lf, numpy, device):
    array = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
    base = getattr(lf.Tensor.from_numpy(array), device)()
    view = base.permute((1, 0))
    mask_array = numpy.array([True, False, True, False])
    mask = getattr(lf.Tensor.from_numpy(mask_array), device)()
    view[mask] = 9
    expected = array.copy()
    expected.T[mask_array] = 9
    numpy.testing.assert_array_equal(base.numpy(), expected)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_same_shape_mask_still_selects_elements(lf, numpy, device):
    array = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
    mask_array = numpy.array([[True, False, True], [False, True, False]])
    tensor = getattr(lf.Tensor.from_numpy(array), device)()
    mask = getattr(lf.Tensor.from_numpy(mask_array), device)()
    tensor[mask] = 9
    array[mask_array] = 9
    numpy.testing.assert_array_equal(tensor.numpy(), array)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_row_mask_view_keeps_its_source(lf, numpy, device):
    source = getattr(lf.Tensor.from_numpy(numpy.array([True, False, True, False])), device)()
    mask = source[:3]
    tensor = lf.Tensor.zeros((3, 5), device=device)
    tensor[mask] = 7
    numpy.testing.assert_array_equal(source.numpy(), [True, False, True, False])
    numpy.testing.assert_array_equal(tensor.numpy(), [[7] * 5, [0] * 5, [7] * 5])
