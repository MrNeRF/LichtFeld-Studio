# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_verified_zero_reference_census_stays_removed() -> None:
    zero_stride = source("src/core/tensor/internal/tensor_zero_stride.hpp")
    for token in ("consumer_name", "Stack", "MaskedFill", "IndexSelect", "Unknown"):
        assert token not in zero_stride
    assert "dense_for_kernel" not in zero_stride
    assert "dense_for_kernel" not in source("src/core/tensor/internal/tensor_impl.hpp")

    joint_device = source("src/training/include/lfs/training/joint_adam_codec.cuh")
    assert "encode_g1g2" not in joint_device
    sh_device = source("src/training/include/lfs/training/sh_value_codec.cuh")
    for token in ("kBlockSizeDevice", "decode_slot", "encode_slot"):
        assert token not in sh_device

    fused_structs = (
        source("src/training/optimizer/adam_optimizer.hpp").split(
            "struct FastGSFusedAdamParam", 1
        )[1].split("};", 1)[0],
        source(
            "src/training/rasterization/fastgs/rasterization/include/fused_adam_types.h"
        ).split("struct FusedAdamParam", 1)[1].split("};", 1)[0],
    )
    for text in fused_structs:
        for token in ("exp_avg_q", "exp_avg_sq_q", "exp_avg_scale", "exp_avg_sq_scale"):
            assert token not in text

    ledger_header = source("src/diagnostics/include/diagnostics/vram_ledger_model.hpp")
    ledger_source = source("src/diagnostics/vram_ledger_model.cpp")
    for token in ("vram_row_kind_name", "include_vulkan_roots"):
        assert token not in ledger_header
    for token in ("vram_row_kind_name", "hooked_exportable_desc", "logical_raster"):
        assert token not in ledger_source

    loader_header = source("src/io/include/io/pipelined_image_loader.hpp")
    loader_source = source("src/io/pipelined_image_loader.cpp")
    for token in (
        "adaptive_prefetch_occupancy",
        "decoded_frames_live",
        "decoded_frames_peak",
        "decoded_frame_capacity",
        "adaptive_peak_occupancy_",
    ):
        assert token not in loader_header
    for token in (
        "after_primary_image_enqueued",
        "vram.audit.io.decoded_frames.",
        "vram.audit.io.prefetch.",
    ):
        assert token not in loader_source

    app_store = source("src/visualizer/include/visualizer/app_store.hpp")
    gui_manager = source("src/visualizer/gui/gui_manager.cpp")
    for token in ("frame_ms", "profiler_snapshot"):
        assert token not in app_store
        assert token not in gui_manager
