/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda/memory_arena.hpp"
#include "gsplat/Common.h"
#include "gsplat/TileBatch.h"
#include "lfs/training/ops/gsplat_cuda.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace lfs::training {

    // Forward pass context - holds raw pointers needed for backward (arena allocated)
    struct GsplatRasterizeContext {
        // Raw pointers to arena-allocated intermediate buffers
        float* render_colors_ptr = nullptr;  // [C, channels, H, W]
        float* render_alphas_ptr = nullptr;  // [C, 1, H, W]
        int32_t* radii_ptr = nullptr;        // [C, N, 2]
        float* means2d_ptr = nullptr;        // [C, N, 2]
        float* depths_ptr = nullptr;         // [C, N]
        float* colors_ptr = nullptr;         // [C, N, channels]
        float* dirs_ptr = nullptr;           // [C, N, 3]
        int32_t* tile_offsets_ptr = nullptr; // [C, tile_height, tile_width]
        int32_t* last_ids_ptr = nullptr;     // [C, H, W]
        float* compensations_ptr = nullptr;  // [C, N] or nullptr

        // Borrowed from the gsplat owner-held VMM intersection cache (do NOT free).
        // Valid from forward through backward for this owner; released at its cache boundary or destruction.
        int64_t* isect_ids_ptr = nullptr;
        int32_t* flatten_ids_ptr = nullptr;
        int64_t n_isects = 0;
        std::vector<gsplat_lfs::TileBatch> batches;
        int32_t* tiles_per_gauss_ptr = nullptr;
        int32_t n_sort = 0;

        // Saved input tensors (references, not copies)
        lfs::core::Tensor means;     // [N, 3]
        lfs::core::Tensor quats;     // [N, 4]
        lfs::core::Tensor scales;    // [N, 3]
        lfs::core::Tensor opacities; // [N]
        lfs::core::Tensor sh0;       // [N, 1, 3]
        lfs::core::Tensor shN;       // swizzled 1D SH-rest buffer
        lfs::core::Tensor bg_color;  // [3] or [C, 3]

        // Retain camera storage until backward and its readers finish.
        lfs::core::Tensor world_view_transform;
        const float* viewmat_ptr = nullptr; // [C, 4, 4]
        const float* K_ptr = nullptr;       // [C, 3, 3]
        lfs::core::Tensor K_tensor;         // Keeps K_ptr alive

        // Distortion coefficients
        const float* radial_ptr = nullptr;
        const float* tangential_ptr = nullptr;
        const float* thin_prism_ptr = nullptr;
        lfs::core::Tensor radial_cuda;
        lfs::core::Tensor tangential_cuda;
        lfs::core::Tensor thin_prism_cuda;

        // Dimensions
        uint32_t N = 0;
        uint32_t K_sh = 0;
        uint32_t layout_bases = 1;
        uint32_t channels = 0;
        uint32_t tile_width = 0;
        uint32_t tile_height = 0;

        // Settings
        uint32_t sh_degree = 0;
        uint32_t sh_layout_degree = 0;
        uint32_t image_width = 0;
        uint32_t image_height = 0;
        uint32_t tile_size = 0;
        float eps2d = 0.0f;
        float near_plane = 0.0f;
        float far_plane = 0.0f;
        float radius_clip = 0.0f;
        float scaling_modifier = 1.0f;
        bool calc_compensations = false;
        GsplatRenderMode render_mode = GsplatRenderMode::RGB;
        CameraModelType camera_model = PINHOLE;

        // Memory arena frame ID (for releasing arena memory in backward)
        uint64_t frame_id = 0;
        // Forward producer queue; backward joins it before consuming retained storage.
        cudaStream_t stream = nullptr;

        // Tile-based training (0 = full image)
        int render_tile_x_offset = 0;
        int render_tile_y_offset = 0;
        int render_tile_width = 0;
        int render_tile_height = 0;

        // Background image for per-pixel blending (optional, empty = use bg_color)
        lfs::core::Tensor bg_image;
    };

    lfs::gpu_ops::State gsplat_create();
    lfs::gpu_ops::RasterResult gsplat_forward(
        lfs::gpu_ops::GsplatSaved&, const lfs::gpu_ops::SplatInputs&, const core::Tensor& view,
        const core::Tensor& radial, const core::Tensor& tangential,
        const core::Tensor& bg_color, const core::Tensor& bg_image,
        const lfs::gpu_ops::GsplatParams&, const lfs::gpu_ops::RenderOutputs&);
    void gsplat_backward(
        lfs::gpu_ops::GsplatSaved&, const core::Tensor& grad_image, const core::Tensor& grad_alpha,
        const lfs::gpu_ops::GsplatGradients&, core::Tensor& densification,
        const core::Tensor& error_map, const core::Tensor& edge_map,
        core::Tensor& edge_scores, core::Tensor& max_screen_share);
    void gsplat_release(lfs::gpu_ops::GsplatSaved&) noexcept;
    void gsplat_record_vram(
        const lfs::gpu_ops::GsplatSaved&, const core::Tensor& image, const core::Tensor& alpha,
        const core::Tensor& gt_tile, const core::Tensor& bg_tile, const core::Tensor& error_map);
    bool gsplat_release_caches(lfs::gpu_ops::GsplatSaved&) noexcept;
    const GsplatRasterizeContext& cuda_gsplat_frame(const lfs::gpu_ops::GsplatSaved&);
} // namespace lfs::training
