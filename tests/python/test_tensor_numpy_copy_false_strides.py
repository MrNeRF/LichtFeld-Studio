# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""numpy(copy=False) must not silently copy a strided CPU Tensor view."""

import pytest


def test_numpy_copy_false_rejects_noncontiguous_cpu_view(lf, numpy):
    values = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
    view = lf.Tensor.from_numpy(values).permute((1, 0))

    with pytest.raises(RuntimeError, match=r"numpy\(copy=False\).*non-contiguous"):
        view.numpy(copy=False)


def test_numpy_copy_true_handles_noncontiguous_cpu_view(lf, numpy):
    values = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
    view = lf.Tensor.from_numpy(values).permute((1, 0))

    numpy.testing.assert_array_equal(view.numpy(copy=True), values.T)
