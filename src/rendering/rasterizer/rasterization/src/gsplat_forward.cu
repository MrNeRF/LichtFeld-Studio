/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/memory_arena.hpp"
#include "gsplat_forward.h"
#include "gsplat_fwd/Common.h"
#include "gsplat_fwd/Ops.h"

namespace lfs::rendering {

    namespace {
        constexpr size_t MEMORY_ALIGNMENT = 128;
        constexpr float EPS_2D = 0.3f;
        constexpr float NEAR_PLANE = 0.01f;
        constexpr float FAR_PLANE = 10000.0f;
        constexpr float RADIUS_CLIP = 0.0f;
        constexpr uint32_t TILE_SIZE = 16;
        constexpr uint32_t NUM_CAMERAS = 1;

        constexpr size_t align(const size_t size) {
            return (size + MEMORY_ALIGNMENT - 1) & ~(MEMORY_ALIGNMENT - 1);
        }
    } // namespace

    void gsplat_forward_gut(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* sh_coeffs,
        const uint32_t sh_degree,
        const uint32_t N,
        const uint32_t K,
        const uint32_t image_width,
        const uint32_t image_height,
        const float* viewmat,
        const float* K_intrinsics,
        const GutCameraModel camera_model,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* background,
        const GutRenderMode render_mode,
        const float scaling_modifier,
        const int* transform_indices,
        const bool* node_visibility_mask,
        const int num_visibility_nodes,
        const int* visible_indices,
        const uint32_t visible_count,
        float* render_colors_out,
        float* render_alphas_out,
        float* render_depth_out,
        cudaStream_t stream) {

        // Coordinate with training arena
        auto& arena = core::GlobalArenaManager::instance().get_arena();
        arena.set_rendering_active(true);
        arena.wait_for_training();
        const uint64_t frame_id = arena.begin_frame(true);
        auto arena_allocator = arena.get_allocator(frame_id);

        const uint32_t H = image_height;
        const uint32_t W = image_width;
        const uint32_t tile_height = (H + TILE_SIZE - 1) / TILE_SIZE;
        const uint32_t tile_width = (W + TILE_SIZE - 1) / TILE_SIZE;
        constexpr uint32_t CHANNELS = 3; // RGB only for viewer

        // M = number of gaussians to process (visible_count if filtering, else N)
        const uint32_t M = (visible_count > 0 && visible_indices != nullptr) ? visible_count : N;

        // Calculate and allocate buffer sizes using M (reduced when filtering)
        const size_t radii_size = align(NUM_CAMERAS * M * 2 * sizeof(int32_t));
        const size_t means2d_size = align(NUM_CAMERAS * M * 2 * sizeof(float));
        const size_t depths_size = align(NUM_CAMERAS * M * sizeof(float));
        const size_t dirs_size = align(NUM_CAMERAS * M * 3 * sizeof(float));
        const size_t conics_size = align(NUM_CAMERAS * M * 3 * sizeof(float));
        const size_t tiles_per_gauss_size = align(NUM_CAMERAS * M * sizeof(int32_t));
        const size_t tile_offsets_size = align(NUM_CAMERAS * tile_height * tile_width * sizeof(int32_t));
        const size_t colors_size = align(NUM_CAMERAS * M * CHANNELS * sizeof(float));
        const size_t render_colors_size = align(NUM_CAMERAS * H * W * CHANNELS * sizeof(float));
        const size_t render_alphas_size = align(NUM_CAMERAS * H * W * sizeof(float));
        const size_t last_ids_size = align(NUM_CAMERAS * H * W * sizeof(int32_t));
        const size_t median_depths_size = align(NUM_CAMERAS * H * W * sizeof(float));

        const size_t total_size = radii_size + means2d_size + depths_size + dirs_size +
                                  conics_size + tiles_per_gauss_size + tile_offsets_size +
                                  colors_size + render_colors_size + render_alphas_size + last_ids_size +
                                  median_depths_size;

        char* blob = arena_allocator(total_size);
        char* ptr = blob;

        auto* const radii_ptr = reinterpret_cast<int32_t*>(ptr);
        ptr += radii_size;
        auto* const means2d_ptr = reinterpret_cast<float*>(ptr);
        ptr += means2d_size;
        auto* const depths_ptr = reinterpret_cast<float*>(ptr);
        ptr += depths_size;
        auto* const dirs_ptr = reinterpret_cast<float*>(ptr);
        ptr += dirs_size;
        auto* const conics_ptr = reinterpret_cast<float*>(ptr);
        ptr += conics_size;
        auto* const tiles_per_gauss_ptr = reinterpret_cast<int32_t*>(ptr);
        ptr += tiles_per_gauss_size;
        auto* const tile_offsets_ptr = reinterpret_cast<int32_t*>(ptr);
        ptr += tile_offsets_size;
        auto* const colors_ptr = reinterpret_cast<float*>(ptr);
        ptr += colors_size;
        auto* const render_colors_ptr = reinterpret_cast<float*>(ptr);
        ptr += render_colors_size;
        auto* const render_alphas_ptr = reinterpret_cast<float*>(ptr);
        ptr += render_alphas_size;
        auto* const last_ids_ptr = reinterpret_cast<int32_t*>(ptr);
        ptr += last_ids_size;
        auto* const median_depths_ptr = reinterpret_cast<float*>(ptr);

        gsplat_fwd::RasterizeWithSHResult result{
            .render_colors = render_colors_ptr,
            .render_alphas = render_alphas_ptr,
            .radii = radii_ptr,
            .means2d = means2d_ptr,
            .depths = depths_ptr,
            .colors = colors_ptr,
            .dirs = dirs_ptr,
            .conics = conics_ptr,
            .tiles_per_gauss = tiles_per_gauss_ptr,
            .tile_offsets = tile_offsets_ptr,
            .last_ids = last_ids_ptr,
            .median_depths = (render_depth_out != nullptr) ? median_depths_ptr : nullptr,
            .compensations = nullptr,
            .isect_ids = nullptr,
            .flatten_ids = nullptr,
            .n_isects = 0};

        UnscentedTransformParameters ut_params;

        gsplat_fwd::rasterize_from_world_with_sh_fwd(
            means, quats, scales, opacities, sh_coeffs, sh_degree,
            background, nullptr,
            N, M, NUM_CAMERAS, K, image_width, image_height, TILE_SIZE,
            viewmat, nullptr, K_intrinsics,
            static_cast<CameraModelType>(camera_model),
            EPS_2D, NEAR_PLANE, FAR_PLANE, RADIUS_CLIP,
            scaling_modifier, false,
            static_cast<int>(render_mode),
            ut_params, ShutterType::GLOBAL,
            radial_coeffs, tangential_coeffs, nullptr,
            transform_indices, node_visibility_mask, num_visibility_nodes,
            visible_indices,
            result, stream);

        // Copy outputs
        const size_t color_bytes = H * W * CHANNELS * sizeof(float);
        const size_t alpha_bytes = H * W * sizeof(float);
        cudaMemcpyAsync(render_colors_out, render_colors_ptr, color_bytes, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(render_alphas_out, render_alphas_ptr, alpha_bytes, cudaMemcpyDeviceToDevice, stream);

        if (render_depth_out != nullptr) {
            cudaMemcpyAsync(render_depth_out, median_depths_ptr, alpha_bytes, cudaMemcpyDeviceToDevice, stream);
        }

        // Free intersection buffers allocated by gsplat
        if (result.isect_ids)
            cudaFreeAsync(result.isect_ids, stream);
        if (result.flatten_ids)
            cudaFreeAsync(result.flatten_ids, stream);

        arena.end_frame(frame_id, true);
        arena.set_rendering_active(false);
    }

} // namespace lfs::rendering
