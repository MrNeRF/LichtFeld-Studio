/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "lfs/training/ops/fast_services.hpp"
#include <cuda_runtime.h>
namespace lfs::training {
    const lfs::gpu_ops::FastRasterOps& cuda_fast_ops();
    // Scalar snapshot of the CUDA frame held by a saved state. Empty after release.
    struct CudaFastFrameView {
        int n_instances = 0;
        int n_visible = 0;
        std::size_t per_tile_buffers_size = 0;
        std::size_t per_instance_sort_total_size = 0;
        std::uint64_t frame_id = 0;
        cudaStream_t completion_stream = nullptr;
        const unsigned* primitive_work_indices = nullptr;
        // Per-instance primitive indices in depth-sorted tile order.
        const unsigned* sorted_primitive_indices = nullptr;
        std::size_t visible_sort_scratch_size = 0;
        unsigned tile_sort_key_bytes = 0;
        // Per-instance blend region masks and their tile positions from the forward blend.
        const unsigned char* contributor_masks = nullptr;
        const unsigned* contributor_positions = nullptr;
        // Visible rows the backward walk reached; null when touched gating is off.
        const unsigned* primitive_touched = nullptr;
        // Per-tile instance ranges and per-pixel last contributors of the forward blend.
        const uint2* tile_ranges = nullptr;
        const unsigned* last_contributors = nullptr;
    };

    [[nodiscard]] CudaFastFrameView cuda_fast_frame_view(const lfs::gpu_ops::FastSaved& saved) noexcept;

} // namespace lfs::training
