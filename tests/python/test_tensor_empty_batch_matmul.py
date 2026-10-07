# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Empty batch products return the mathematically shaped empty output."""

import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("left,right", [
    ((0, 3, 4), (0, 4, 5)), ((0, 3, 4), (1, 4, 5)),
    ((1, 3, 4), (0, 4, 5)), ((3, 4), (0, 4, 5)),
    ((0, 3, 4), (4, 5)), ((2, 0, 4), (2, 4, 5)),
    ((2, 3, 4), (2, 4, 0)), ((2, 3, 0), (2, 0, 5)),
])
def test_empty_batch_matmul_matches_numpy(lf, numpy, device, left, right):
    a = lf.Tensor.zeros(left, device=device)
    b = lf.Tensor.zeros(right, device=device)
    result = a.matmul(b)
    expected = numpy.matmul(numpy.zeros(left, dtype=numpy.float32), numpy.zeros(right, dtype=numpy.float32))
    assert result.shape == expected.shape
    assert result.device == a.device
    numpy.testing.assert_array_equal(result.numpy(), expected)
    # An empty operation must leave CUDA usable for subsequent work.
    numpy.testing.assert_array_equal(lf.Tensor.ones((2, 2), device=device).matmul(lf.Tensor.ones((2, 2), device=device)).numpy(), numpy.full((2, 2), 2))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_empty_batch_still_checks_inner_dimensions(lf, device):
    with pytest.raises(RuntimeError):
        lf.Tensor.zeros((0, 3, 4), device=device).matmul(lf.Tensor.zeros((0, 6, 5), device=device))
