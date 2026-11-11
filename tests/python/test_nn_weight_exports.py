# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native archive export and pinned checkpoint rejection; no model framework."""
from importlib import import_module
import json
from pathlib import Path
import mmap

import numpy as np
import pytest


@pytest.fixture
def exporter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "src/python"))
    return import_module("_lfs_nn_weights")



def test_shared_lfw_writer_preserves_scalar_shape_alignment_and_float_values(exporter, tmp_path):
    source = {"scalar": np.array(3.5), "matrix": np.array([[1., -2.], [0., 4.]])}
    first, second = tmp_path / "one.lfw", tmp_path / "two.lfw"
    for path in (first, second):
        exporter.write_lfw(path, source, {"model": "fixture"})
    raw = first.read_bytes()
    assert raw == second.read_bytes()
    assert raw[:4] == b"LFW1"
    length = int.from_bytes(raw[4:8], "little")
    header = json.loads(raw[8:8 + length])
    payload = (8 + length + 63) & ~63
    assert header["meta"] == {"model": "fixture"}
    for name, expected in source.items():
        record = header["tensors"][name]
        assert record["offset"] % 64 == 0
        assert record["shape"] == list(expected.shape)
        assert record["dtype"] == "float32"
        begin = payload + record["offset"]
        actual = np.frombuffer(raw[begin:begin + record["length"]], dtype="<f4").reshape(record["shape"])
        np.testing.assert_array_equal(actual, expected.astype(np.float32))


def test_writer_polls_cancellation_during_large_tensor(exporter, tmp_path):
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise InterruptedError("cancel")

    output = tmp_path / "staging.lfw"
    with pytest.raises(InterruptedError):
        exporter.write_lfw(output, {"large": np.zeros(1024 * 1024, dtype=np.float32)}, {},
                           check_cancelled=cancel)
    assert calls == 4
    assert output.stat().st_size < 4 * 1024 * 1024
    # A new attempt can write a complete archive after an interrupted staging write.
    exporter.write_lfw(output, {"scalar": np.array(2.)}, {})
    assert output.read_bytes()[:4] == b"LFW1"



def test_stdlib_tensor_export_matches_array_export_and_rejects_invalid_extent(exporter, tmp_path):
    from _lfs_nn_weights import FloatTensor
    arrays = {"scalar": np.array(3.5, dtype="<f4"),
              "matrix": np.array([[1., -2.], [0., 4.]], dtype="<f4")}
    buffers = {name: FloatTensor(value.shape, value.tobytes()) for name, value in arrays.items()}
    first, second = tmp_path / "arrays.lfw", tmp_path / "buffers.lfw"
    exporter.write_lfw(first, arrays, {})
    exporter.write_lfw(second, buffers, {})
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(ValueError, match="extent"):
        exporter.write_lfw(tmp_path / "invalid.lfw", {"broken": FloatTensor((2,), b"1234")}, {})


def test_cancellation_releases_mapped_input_views(exporter, tmp_path):
    from _lfs_nn_weights import FloatTensor
    source = tmp_path / "mapped.bin"
    source.write_bytes(bytes(4 * 1024 * 1024))
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise InterruptedError("cancel")

    with source.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        view = memoryview(mapped)
        try:
            with pytest.raises(InterruptedError):
                exporter.write_lfw(tmp_path / "staging.lfw", {"mapped": FloatTensor((1024 * 1024,), view)}, {},
                                   check_cancelled=cancel)
        finally:
            view.release()
    # Windows cannot replace/delete a still-mapped source.
    source.unlink()


def test_llmdet_rejects_unverified_input_and_cancellation_preserves_destination(tmp_path):
    from _lfs_llmdet_weights import convert
    source, output = tmp_path / "source.safetensors", tmp_path / "staging.lfw"
    source.write_bytes(b"untrusted input")
    output.write_bytes(b"existing archive")
    with pytest.raises(ValueError, match="pinned"):
        convert(source, output)
    with pytest.raises(InterruptedError):
        convert(source, output, lambda: True)
    with pytest.raises(ValueError, match="differ"):
        convert(source, source)
    alias = tmp_path / "source-alias.safetensors"
    alias.hardlink_to(source)
    with pytest.raises(ValueError, match="differ"):
        convert(source, alias)
    assert output.read_bytes() == b"existing archive"
    assert source.read_bytes() == b"untrusted input"
