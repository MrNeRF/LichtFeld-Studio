# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Empty Python slices produce valid empty Tensor views."""

import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 3, 4)])
@pytest.mark.parametrize("key", [slice(None), slice(0, 0), slice(3, 1), slice(99, None), slice(None, -99)])
def test_empty_and_full_slices(lf, numpy, device, shape, key):
    array = numpy.zeros(shape, dtype=numpy.float32)
    tensor = lf.Tensor.zeros(shape, device=device)
    result = tensor[key]
    assert result.shape == array[key].shape
    numpy.testing.assert_array_equal(result.numpy(), array[key])
    tensor[key] = 7
    array[key] = 7
    numpy.testing.assert_array_equal(tensor.numpy(), array)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_empty_slice_on_inner_axes(lf, numpy, device):
    tensor = lf.Tensor.zeros((2, 3, 4), device=device)
    assert tensor[:, 1:1, :].shape == (2, 0, 4)
    assert tensor[2:2, 3:3, 4:4].shape == (0, 0, 0)
    tensor[:, 1:1, :] = 8
    numpy.testing.assert_array_equal(tensor.numpy(), numpy.zeros((2, 3, 4)))
