# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Convert the pinned official LLMDet Tiny checkpoint, without model frameworks.

The shared cache owns staging, cancellation and atomic publication. This adapter
only reads the authenticated safetensors archive and writes the supplied staging
file through the canonical LFW writer. No downloaded code or pickle is executed.
"""
import json
import mmap
from math import prod
import struct

from _lfs_path_io import Path
from _lfs_nn_weights import FloatTensor, check_cancelled, sha256_file, write_lfw

SOURCE = "https://huggingface.co/iSEE-Laboratory/llmdet_tiny/resolve/d05199165a19320a9236396e20fca0a5e065189c/model.safetensors"
SOURCE_SHA256 = "f0e476be3d65c7c50e13029b0adc4bb5d4e05f3eb6d7bb04856578095e6ad92f"
SOURCE_BYTES = 692015412


def _relative_position_indices():
    # Swin's window is 7x7. The native network reconstructs these constant
    # indices; verify the checkpoint agrees before omitting the integer buffers.
    return tuple(((y // 7 - x // 7 + 6) * 13 + y % 7 - x % 7 + 6)
                 for y in range(49) for x in range(49))


def convert(checkpoint, output, cancel_requested=lambda: False):
    return _convert(checkpoint, output, cancel_requested, authenticate=True)


def _convert_verified(checkpoint, output, cancel_requested=lambda: False):
    """Shared-cache adapter: source authenticated by the native cache owner.

    The cache also checks the deterministic native digest before publication.
    The standalone export entry above authenticates its source on the same open
    handle before using this single safetensors decoder and archive writer.
    """
    return _convert(checkpoint, output, cancel_requested, authenticate=False)


def _convert(checkpoint, output, cancel_requested, *, authenticate):
    check_cancelled(cancel_requested)
    source, target = Path(checkpoint), Path(output)
    if source.resolve() == target.resolve() or (target.exists() and source.samefile(target)):
        raise ValueError("Model conversion output must differ from its source")
    with source.open("rb") as stream:
        if authenticate and sha256_file(stream, cancel_requested) != SOURCE_SHA256:
            raise ValueError("Checkpoint does not match the pinned LLMDet Tiny source")
        check_cancelled(cancel_requested)
        # mmap keeps conversion from duplicating the complete checkpoint in RAM.
        # The mapping and every exported view end before the worker publishes.
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            header_bytes = int.from_bytes(mapped[:8], "little")
            if not 2 <= header_bytes <= len(mapped) - 8:
                raise ValueError("Invalid LLMDet safetensors header")
            header = json.loads(mapped[8:8 + header_bytes])
            payload = 8 + header_bytes
            tensors = {}
            try:
                integer_buffers = 0
                expected_indices = _relative_position_indices()
                for name, record in header.items():
                    check_cancelled(cancel_requested)
                    if name == "__metadata__":
                        continue
                    shape = tuple(record["shape"])
                    begin, end = record["data_offsets"]
                    if not 0 <= begin <= end <= len(mapped) - payload:
                        raise ValueError("Invalid LLMDet tensor extent")
                    if record["dtype"] == "I64" and name.endswith(".relative_position_index"):
                        if shape != (49, 49) or end - begin != 49 * 49 * 8:
                            raise ValueError("Invalid Swin relative position buffer")
                        indices = struct.unpack_from("<2401q", mapped, payload + begin)
                        if indices != expected_indices:
                            raise ValueError("Unexpected Swin relative position indices")
                        integer_buffers += 1
                    elif record["dtype"] == "F32" and end - begin == prod(shape) * 4:
                        tensors[name] = FloatTensor(shape, memoryview(mapped)[payload + begin:payload + end])
                    else:
                        raise ValueError("Unsupported LLMDet tensor representation")
                if len(tensors) != 1015 or integer_buffers != 12:
                    raise ValueError("Unexpected LLMDet Tiny architecture")
                check_cancelled(cancel_requested)
                write_lfw(target, tensors,
                          {"model": "llmdet-tiny", "source_sha256": SOURCE_SHA256},
                          check_cancelled=lambda: check_cancelled(cancel_requested))
            finally:
                for tensor in tensors.values():
                    tensor.data.release()
