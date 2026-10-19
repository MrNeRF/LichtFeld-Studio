# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tensor.arange returns an empty Tensor when the step points away from end."""

import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("start,end,step", [(3, -2, 1), (-2, 3, -1)])
def test_arange_direction_mismatch_is_empty(lf, numpy, device, start, end, step):
    result = lf.Tensor.arange(start, end, step, device=device)
    expected = numpy.arange(start, end, step, dtype=numpy.float32)

    assert result.shape == (0,)
    numpy.testing.assert_array_equal(result.numpy(), expected)
