# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared writer for native neural-network weight archives."""
from pathlib import Path
import json
import hashlib
from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class FloatTensor:
    shape: tuple[int, ...]
    data: bytes | memoryview


def sha256_file(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def write_lfw(path: Path, tensors: dict, meta: dict[str, object]) -> None:
    blobs: list[tuple[int, memoryview]] = []
    blob = None
    try:
        payload_offset = 0
        entries: dict[str, dict[str, object]] = {}
        for name, value in tensors.items():
            if isinstance(value, FloatTensor):
                shape = value.shape
                blob = memoryview(value.data)
                if any(dimension < 0 for dimension in shape) or len(blob) != prod(shape) * 4:
                    raise ValueError(f"Invalid float32 tensor extent: {name}")
            else:
                # NumPy-based developer exporters share the same archive layout.
                array = value.astype("<f4", order="C", copy=False)
                shape = array.shape
                blob = memoryview(array).cast("B")
            payload_offset = (payload_offset + 63) & ~63
            entries[name] = {
                "dtype": "float32",
                "shape": [int(dimension) for dimension in shape],
                "offset": payload_offset,
                "length": len(blob),
            }
            blobs.append((payload_offset, blob))
            payload_offset += len(blob)

        header = json.dumps(
            {"format": "lfw", "version": 1, "meta": meta, "tensors": entries},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        file_payload_offset = (8 + len(header) + 63) & ~63
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.write(b"LFW1")
            stream.write(len(header).to_bytes(4, "little"))
            stream.write(header)
            stream.write(bytes(file_payload_offset - stream.tell()))
            for offset, blob in blobs:
                stream.write(bytes(file_payload_offset + offset - stream.tell()))
                for begin in range(0, len(blob), 1024 * 1024):
                    stream.write(blob[begin:begin + 1024 * 1024])
    finally:
        # Release mmap-backed views before their owner closes the mapping,
        # including when an exception retains the writer frame.
        for _, view in blobs:
            view.release()
        if blob is not None:
            blob.release()
