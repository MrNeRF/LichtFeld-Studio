# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bridge correctness for Tensor.from_numpy, from_dlpack, and Scene.add_splat."""

import pytest


def _noncontiguous_dlpack_producer(numpy):
    """Return an object whose __dlpack__ exports a non-contiguous layout.

    Prefers a transposed NumPy array when that producer can actually emit a
    capsule. Falls back to torch, then skips.
    """
    base = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
    view = base.T
    if hasattr(view, "__dlpack__"):
        try:
            view.__dlpack__()
            return view
        except Exception:
            pass

    torch = pytest.importorskip(
        "torch",
        reason="numpy cannot export a non-contiguous DLPack producer and torch is unavailable",
    )
    return torch.from_numpy(view)


class TestFromNumpy:
    def test_contiguous_roundtrip(self, lf, numpy):
        arr = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
        tensor = lf.Tensor.from_numpy(arr)
        numpy.testing.assert_array_equal(tensor.numpy(), arr)

    def test_strided_slice_raises(self, lf, numpy):
        arr = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
        view = arr[:, ::2]
        with pytest.raises(RuntimeError, match="contiguous"):
            lf.Tensor.from_numpy(view)

    def test_copy_false_raises(self, lf, numpy):
        arr = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
        with pytest.raises(RuntimeError, match="copy"):
            lf.Tensor.from_numpy(arr, copy=False)


class TestFromDlpack:
    def test_contiguous_roundtrip(self, lf, numpy):
        arr = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32)
        if hasattr(arr, "__dlpack__"):
            tensor = lf.Tensor.from_dlpack(arr)
        else:
            tensor = lf.Tensor.from_dlpack(lf.Tensor.from_numpy(arr))
        numpy.testing.assert_array_equal(tensor.numpy(), arr)

    def test_noncontiguous_raises(self, lf, numpy):
        producer = _noncontiguous_dlpack_producer(numpy)
        with pytest.raises(RuntimeError, match="contiguous"):
            lf.Tensor.from_dlpack(producer)

    def test_empty_sliced_view_roundtrip(self, lf, numpy):
        base = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
        view = base[:, ::2][:, :0]
        if not hasattr(view, "__dlpack__"):
            pytest.skip("numpy cannot export an empty sliced DLPack producer")
        try:
            view.__dlpack__()
        except Exception:
            pytest.skip("numpy cannot export an empty sliced DLPack producer")
        tensor = lf.Tensor.from_dlpack(view)
        numpy.testing.assert_array_equal(tensor.numpy(), numpy.ascontiguousarray(view))


class TestAddSplatDtype:
    def test_float16_shn_raises(self, lf):
        scene = lf.get_scene()
        if scene is None or not scene.is_valid():
            pytest.skip("Scene not available")

        n = 1
        means = lf.Tensor.zeros([n, 3], device="cpu", dtype="float32")
        sh0 = lf.Tensor.zeros([n, 1, 3], device="cpu", dtype="float32")
        shN = lf.Tensor.zeros([n, 3, 3], device="cpu", dtype="float16")
        scaling = lf.Tensor.zeros([n, 3], device="cpu", dtype="float32")
        rotation = lf.Tensor.zeros([n, 4], device="cpu", dtype="float32")
        opacity = lf.Tensor.zeros([n, 1], device="cpu", dtype="float32")
        with pytest.raises(RuntimeError, match="shN"):
            scene.add_splat(
                "bad_shn",
                means,
                sh0,
                shN,
                scaling,
                rotation,
                opacity,
                sh_degree=1,
            )
