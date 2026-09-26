/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/error.hpp"
#include "core/splat_data.hpp"
#include "lfs/training/ops/raster.hpp"
#include "optimizer/render_output.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training {

    const lfs::gpu_ops::FastRasterOps& cuda_fast_ops();

    [[nodiscard]] lfs::Error fast_raster_error(const lfs::gpu_ops::RasterResult& result);

    // Packs the camera and model the trainer already holds and calls FastRasterOps::forward.
    lfs::gpu_ops::RasterResult fast_render(
        const lfs::gpu_ops::FastRasterOps& ops,
        lfs::gpu_ops::FastSaved& saved,
        lfs::core::Camera& camera,
        lfs::core::SplatData& model,
        lfs::core::Tensor& bg_color,
        int tile_x,
        int tile_y,
        int tile_width,
        int tile_height,
        bool mip_filter,
        const lfs::core::Tensor& bg_image,
        bool render_normal,
        bool render_depth,
        RenderOutput& output);

    // Forward, then release the frame. The returned image aliases the saved cache.
    RenderOutput fast_infer(
        const lfs::gpu_ops::FastRasterOps& ops,
        lfs::gpu_ops::FastSaved& saved,
        lfs::core::Camera& camera,
        lfs::core::SplatData& model,
        lfs::core::Tensor& bg_color,
        bool mip_filter,
        const lfs::core::Tensor& bg_image = {},
        bool render_normal = false);

    // Rasterizer-owned VRAM rows. Image and alpha are the bound outputs.
    void fast_record_vram(
        const lfs::gpu_ops::FastSaved& saved,
        const lfs::core::Tensor& image,
        const lfs::core::Tensor& alpha,
        bool run_gaussian_backward,
        size_t num_primitives);

    // Drops cached output tensors. The live forward frame stays owned by release.
    void fast_release_caches(lfs::gpu_ops::FastSaved& saved) noexcept;

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
