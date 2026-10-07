# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tensor row-slice assignment supports NumPy-compatible broadcasting."""

import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_row_slice_assignment_broadcasts_vector(lf, numpy, device):
    array = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
    tensor = getattr(lf.Tensor.from_numpy(array), device)()
    value = lf.Tensor.full((3,), 17, device=device)

    tensor[0:1, :] = value

    expected = array.copy()
    expected[0, :] = 17
    numpy.testing.assert_array_equal(tensor.numpy(), expected)
