# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Empty contiguous NumPy arrays retain multidimensional shapes on import."""

import pytest


@pytest.mark.parametrize("shape", [(0, 2), (0, 2, 3)])
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_from_numpy_accepts_empty_multidimensional_arrays(lf, numpy, shape, device):
    array = numpy.empty(shape, dtype=numpy.float32)

    tensor = getattr(lf.Tensor.from_numpy(array), device)()

    assert tensor.shape == shape
    assert tensor.dtype == "float32"
    assert tensor.numpy().shape == shape
