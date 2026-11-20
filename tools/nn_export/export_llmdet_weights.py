# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export the pinned official LLMDet Tiny checkpoint without Torch."""
import json
import mmap
from math import prod
import struct

from pathlib import Path
import argparse
from lfw import FloatTensor, sha256_file, write_lfw

SOURCE = "https://huggingface.co/iSEE-Laboratory/llmdet_tiny/resolve/d05199165a19320a9236396e20fca0a5e065189c/model.safetensors"
SOURCE_SHA256 = "f0e476be3d65c7c50e13029b0adc4bb5d4e05f3eb6d7bb04856578095e6ad92f"


def _relative_position_indices():
    # Swin's window is 7x7. The native network reconstructs these constant
    # indices; verify the checkpoint agrees before omitting the integer buffers.
    return tuple(((y // 7 - x // 7 + 6) * 13 + y % 7 - x % 7 + 6)
                 for y in range(49) for x in range(49))


def convert(checkpoint, output):
    source, target = Path(checkpoint), Path(output)
    if source.resolve() == target.resolve() or (target.exists() and source.samefile(target)):
        raise ValueError("Model conversion output must differ from its source")
    with source.open("rb") as stream:
        if sha256_file(stream) != SOURCE_SHA256:
            raise ValueError("Checkpoint does not match the pinned LLMDet Tiny source")
        # mmap keeps conversion from duplicating the complete checkpoint in RAM.
        # The mapping and every exported view end before returning.
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
                write_lfw(target, tensors,
                          {"model": "llmdet-tiny", "source_sha256": SOURCE_SHA256})
            finally:
                for tensor in tensors.values():
                    tensor.data.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help=SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    convert(args.checkpoint, args.out)
    with args.out.open("rb") as stream:
        print(f"wrote {args.out} sha256={sha256_file(stream)}")


if __name__ == "__main__":
    main()
