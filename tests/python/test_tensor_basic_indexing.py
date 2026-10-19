# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Basic indexing follows Python slice, ellipsis and new-axis semantics."""

import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("key", [
    (Ellipsis, 1), (1, Ellipsis), (slice(None), Ellipsis, -1),
    None, Ellipsis, (), (None, slice(None), slice(None)),
    (slice(None), None, Ellipsis, -1), (1, Ellipsis, None),
    (Ellipsis, None), (None, Ellipsis, None),
])
def test_ellipsis_and_new_axis_values(lf, numpy, device, key):
    array = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
    tensor = getattr(lf.Tensor.from_numpy(array), device)()
    result = tensor[key]
    assert result.shape == array[key].shape
    numpy.testing.assert_array_equal(result.numpy(), array[key])
    tensor[key] = 9
    array[key] = 9
    numpy.testing.assert_array_equal(tensor.numpy(), array)
    replacement = numpy.full(array[key].shape, -3, dtype=numpy.float32)
    tensor[key] = getattr(lf.Tensor.from_numpy(replacement), device)()
    array[key] = replacement
    numpy.testing.assert_array_equal(tensor.numpy(), array)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_basic_indexing_preserves_strided_view_aliases(lf, numpy, device):
    array = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
    tensor = getattr(lf.Tensor.from_numpy(array), device)()
    tensor.permute((1, 0))[None, ..., 1] = 17
    array.T[None, ..., 1] = 17
    numpy.testing.assert_array_equal(tensor.numpy(), array)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_invalid_basic_indices_raise(lf, device):
    tensor = lf.Tensor.zeros((2, 3), device=device)
    for key in [(Ellipsis, Ellipsis), (slice(None),) * 3, (2,), (-3,), (object(),)]:
        with pytest.raises((IndexError, RuntimeError)):
            tensor[key]
        with pytest.raises((IndexError, RuntimeError)):
            tensor[key] = 1


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_scalar_ellipsis_and_new_axis(lf, numpy, device):
    tensor = lf.Tensor.full((), 3, device=device)
    assert tensor[...].shape == ()
    assert tensor[None].shape == (1,)
    assert tensor[None, ..., None].shape == (1, 1)
    numpy.testing.assert_array_equal(tensor[None].numpy(), [3])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("key,value_shapes", [
    ((1,), [(3,), (1, 3)]),
    ((1, slice(1, 3)), [(2,), (1, 2)]),
])
def test_integer_index_assignment_accepts_leading_unit_dimensions(lf, numpy, device, key, value_shapes):
    for shape in value_shapes:
        tensor = lf.Tensor.zeros((3, 3), device=device)
        values = getattr(lf.Tensor.from_numpy(numpy.full(shape, 5, dtype=numpy.float32)), device)()
        tensor[key] = values
        expected = numpy.zeros((3, 3), dtype=numpy.float32)
        expected[key] = 5
        numpy.testing.assert_array_equal(tensor.numpy(), expected)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_integer_index_assignment_rejects_mismatching_shape(lf, device):
    tensor = lf.Tensor.zeros((3, 3), device=device)
    with pytest.raises((RuntimeError, ValueError)):
        tensor[1] = lf.Tensor.zeros((2,), device=device)
