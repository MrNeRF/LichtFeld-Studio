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
    };

    [[nodiscard]] CudaFastFrameView cuda_fast_frame_view(const lfs::gpu_ops::FastSaved& saved) noexcept;

} // namespace lfs::training
