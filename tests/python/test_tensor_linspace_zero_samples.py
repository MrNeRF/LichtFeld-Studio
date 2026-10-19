# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tensor.linspace with zero samples returns an empty Tensor."""

import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_linspace_zero_samples_is_empty(lf, numpy, device):
    result = lf.Tensor.linspace(0, 1, 0, device=device)
    expected = numpy.linspace(0, 1, 0, dtype=numpy.float32)

    assert result.shape == (0,)
    numpy.testing.assert_array_equal(result.numpy(), expected)
