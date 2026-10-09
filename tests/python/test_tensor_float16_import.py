# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""NumPy half precision imports preserve shapes and stored bits."""

import pytest


def test_float16_numpy_roundtrip_preserves_bits(lf, numpy):
    bits = numpy.array([0, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3d00, 0xc100, 0x7bff, 0x7c00, 0xfc00, 0x7e01], dtype=numpy.uint16)
    array = bits.view(numpy.float16)
    tensor = lf.Tensor.from_numpy(array)
    assert tensor.dtype == "float16"
    numpy.testing.assert_array_equal(tensor.numpy().view(numpy.uint16), bits)
    array[0] = 3
    assert tensor.numpy()[0] == 0


@pytest.mark.parametrize("shape", [(), (2, 3), (2, 2, 3)])
def test_float16_numpy_shapes(lf, numpy, shape):
    array = numpy.arange(int(numpy.prod(shape)), dtype=numpy.float16).reshape(shape)
    tensor = lf.Tensor.from_numpy(array)
    assert tensor.shape == shape
    assert tensor.dtype == "float16"
    numpy.testing.assert_array_equal(tensor.numpy(), array)
