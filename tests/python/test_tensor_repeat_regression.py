# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tensor tiling must preserve exact repeat counts and values."""

import pytest


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape,repeats", [
    ((1, 2), (2, 3)), ((2, 3), (3, 5)), ((2, 3), (1, 7)),
    ((2, 3), (2, 1, 3)), ((2, 3), (0, 3)), ((0, 3), (2, 3)),
    ((2, 3), (1, 1)),
])
def test_repeat_matches_tile(lf, numpy, device, shape, repeats):
    array = numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape)
    tensor = getattr(lf.Tensor.from_numpy(array), device)()
    result = tensor.repeat(repeats)
    expected = numpy.tile(array, repeats)
    assert result.shape == expected.shape
    numpy.testing.assert_array_equal(result.numpy(), expected)
    numpy.testing.assert_array_equal(tensor.numpy(), array)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_repeat_noncontiguous(lf, numpy, device):
    array = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
    tensor = getattr(lf.Tensor.from_numpy(array), device)().permute((1, 0))
    numpy.testing.assert_array_equal(tensor.repeat((3, 2)).numpy(), numpy.tile(array.T, (3, 2)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_repeat_rejects_negative_counts(lf, device):
    tensor = lf.Tensor.ones((2, 3), device=device)
    with pytest.raises((ValueError, RuntimeError)):
        tensor.repeat((1, -1))
