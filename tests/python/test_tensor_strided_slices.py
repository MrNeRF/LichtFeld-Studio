# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Positive non-unit Tensor slice steps preserve strided indexing semantics."""

import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_positive_step_slice_getitem_and_setitem(lf, numpy, device):
    array = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
    tensor = getattr(lf.Tensor.from_numpy(array), device)()

    numpy.testing.assert_array_equal(tensor[:, ::2, 1::2].numpy(), array[:, ::2, 1::2])
    tensor[:, ::2, 1::2] = 17
    array[:, ::2, 1::2] = 17
    numpy.testing.assert_array_equal(tensor.numpy(), array)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_negative_step_slice_has_clear_error(lf, device):
    tensor = lf.Tensor.arange(0, 8, 1, device=device)
    with pytest.raises(ValueError, match="negative slice steps are not supported"):
        tensor[::-1]
