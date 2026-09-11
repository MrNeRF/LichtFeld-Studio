# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Structural checks for currently constrained IO paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def test_mask_cache_and_eight_bit_staging_assumptions_fail_loudly():
    pipeline = _read("src/io/pipelined_image_loader.cpp")
    nvcodec = _read("src/io/nvcodec_image_loader.cpp")

    mask_start = pipeline.index("} else if (batch[i].is_mask) {")
    mask_end = pipeline.index("} else {", mask_start + 1)
    mask_body = pipeline[mask_start:mask_end]
    assert "pipeline JPEG2000 mask cache must contain eight-bit samples" in mask_body
    assert "reinterpret_cast<const uint16_t*>(raw.data_ptr())" not in mask_body

    encode_start = nvcodec.index("NvCodecImageLoader::encode_grayscale_to_jpeg2k")
    encode_end = nvcodec.index("NvCodecImageLoader::decode_jpeg2k_16bit_from_memory_gpu")
    encode_body = nvcodec[encode_start:encode_end]
    assert "!eight_bit || cuda_stream == nullptr" in encode_body
    assert "eight-bit JPEG2000 staging currently requires" in encode_body
