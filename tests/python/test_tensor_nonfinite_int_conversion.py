# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Integer scalar conversion rejects non-finite floating-point values."""

import pytest


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize(
    ("value", "error"),
    [
        (float("nan"), ValueError),
        (float("inf"), OverflowError),
        (float("-inf"), OverflowError),
    ],
)
def test_nonfinite_float_int_conversion_matches_python(lf, numpy, dtype, value, error):
    tensor = lf.Tensor.from_numpy(numpy.array([value], dtype=numpy.float32)).to(dtype)

    with pytest.raises(error):
        tensor.int_()

    assert tensor.bool_() is True
