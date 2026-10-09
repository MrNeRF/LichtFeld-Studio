# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""DLPack interchange preserves Tensor bool values and dtype."""


def test_bool_tensor_dlpack_export_preserves_dtype(lf, numpy):
    values = numpy.array([True, False, True, False], dtype=numpy.bool_)

    tensor = lf.Tensor.from_numpy(values)
    exported = numpy.from_dlpack(tensor)
    assert exported.dtype == numpy.bool_
    numpy.testing.assert_array_equal(exported, values)


def test_bool_tensor_dlpack_import(lf, numpy):
    values = numpy.array([True, False, True, False], dtype=numpy.bool_)
    imported = lf.Tensor.from_dlpack(values)
    assert imported.dtype == "bool"
    numpy.testing.assert_array_equal(imported.numpy(), values)
