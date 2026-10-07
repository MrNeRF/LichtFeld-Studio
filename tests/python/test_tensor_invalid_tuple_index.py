# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unsupported tuple indices are rejected without mutating the Tensor."""

import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_invalid_tuple_assignment_does_not_overwrite_tensor(lf, numpy, device):
    original = numpy.arange(4, dtype=numpy.float32).reshape(2, 2)
    tensor = getattr(lf.Tensor.from_numpy(original), device)()

    with pytest.raises(IndexError, match="unsupported tensor index type"):
        tensor[(object(),)] = 9.0

    numpy.testing.assert_array_equal(tensor.numpy(), original)
