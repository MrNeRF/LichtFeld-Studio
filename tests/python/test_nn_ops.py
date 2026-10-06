# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Neural operation bindings on CPU and the selected GPU backend."""

import pytest

from test_tensor_restored import assert_values, device, tensor  # noqa: F401


def test_softmax(lf, numpy, tensor):
    x = tensor([[1, 2, 3], [-2, 0, 2]])
    values = x.cpu().numpy()
    for mask in (None, tensor([0, -2, 1])):
        logits = values if mask is None else values + mask.cpu().numpy()
        exp = numpy.exp(logits - logits.max(axis=-1, keepdims=True))
        result = lf.nn.softmax(x, mask=mask)
        assert result.backend == x.backend
        assert_values(numpy, result, exp / exp.sum(axis=-1, keepdims=True), atol=1e-6)
    assert_values(numpy, lf.nn.softmax(x), lf.nn.softmax(x, None).cpu().numpy())


def test_silu(lf, numpy, tensor):
    values = numpy.array([-3, 0, 2], dtype=numpy.float32)
    x = tensor(values)
    result = lf.nn.silu(input=x)
    assert result.backend == x.backend
    assert_values(numpy, result, values / (1 + numpy.exp(-values)), atol=1e-6)


def test_rms_norm(lf, numpy, tensor):
    values = numpy.array([[1, 2, 3], [2, 4, 6]], dtype=numpy.float32)
    x = tensor(values)
    weight = tensor([1, 2, 3])
    for eps in (1e-6, 0.5):
        expected = values / numpy.sqrt((values ** 2).mean(axis=-1, keepdims=True) + eps) * [1, 2, 3]
        assert_values(numpy, lf.nn.rms_norm(x, weight, eps=eps), expected, rtol=1e-5)
    assert_values(numpy, lf.nn.rms_norm(x, weight), lf.nn.rms_norm(x, weight, 1e-6).cpu().numpy())


def test_residual_scale(lf, numpy, tensor):
    x = tensor([[1, 2], [3, 4]])
    result = lf.nn.residual_scale(x=x, hidden=tensor([[2, 3], [4, 5]]), gamma=tensor([0.5, 2]))
    assert result.backend == x.backend
    assert_values(numpy, result, [[2, 8], [5, 14]])


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_windows(lf, numpy, tensor, dtype):
    values = numpy.arange(2 * 2 * 5 * 3).reshape(2, 2, 5, 3)
    x = tensor(values, dtype)
    windows = lf.nn.window_partition(input=x, window_size=3)
    assert windows.shape == (4, 2, 3, 3)
    assert windows.dtype == dtype and windows.backend == x.backend
    padded = numpy.pad(values, ((0, 0), (0, 0), (0, 1), (0, 0)))
    expected = padded.reshape(2, 2, 2, 3, 3).transpose(0, 2, 1, 3, 4).reshape(4, 2, 3, 3)
    assert_values(numpy, windows, expected)
    result = lf.nn.window_unpartition(windows=windows, window_size=3, original_n=5)
    assert_values(numpy, result, values)


def test_nn_errors(lf, tensor):
    with pytest.raises((RuntimeError, ValueError, lf.Error)):
        lf.nn.window_partition(tensor([1, 2]), window_size=2)
    with pytest.raises((RuntimeError, ValueError, lf.Error)):
        lf.nn.rms_norm(tensor([[1, 2]]), tensor([1, 2, 3]))


def test_model_weight_bytes_without_loading(lf):
    model = lf.nn.RomaV1()
    assert not model.is_loaded
    assert model.weights_bytes == 0
    model.close()
    assert model.weights_bytes == 0
