/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "buffer_utils.h"
#include "core/cuda/memory_arena.hpp"
#include "cuda_utils.h"
#include "edge_rasterization_api.h"
#include "edge_rasterization_config.h"
#include "forward.h"
#include "helper_math.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <functional>
#include <stdexcept>
#include <string>

namespace edge_compute::rasterization {

    ForwardResult edge_forward_raw(
        const float* means_ptr,
        const float* scales_raw_ptr,
        const float* rotations_raw_ptr,
        const float* opacities_raw_ptr,
        const float* w2c_ptr,
        int n_primitives,
        int width,
        int height,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        float near_plane,
        float far_plane,
        bool mip_filter,
        const float* pixel_weights,
        float* accum_weights) {
        // Validate inputs using pure CUDA validation
        CHECK_CUDA_PTR(means_ptr, "means_ptr");
        CHECK_CUDA_PTR(scales_raw_ptr, "scales_raw_ptr");
        CHECK_CUDA_PTR(rotations_raw_ptr, "rotations_raw_ptr");
        CHECK_CUDA_PTR(opacities_raw_ptr, "opacities_raw_ptr");
        CHECK_CUDA_PTR(w2c_ptr, "w2c_ptr");

        if (n_primitives <= 0 || width <= 0 || height <= 0) {
            throw std::runtime_error("Invalid dimensions in forward_raw");
        }

        // Calculate grid dimensions
        const dim3 grid(div_round_up(width, config::tile_width),
                        div_round_up(height, config::tile_height), 1);
        const int n_tiles = grid.x * grid.y;

        // Get global arena and begin frame
        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
        uint64_t frame_id = arena.begin_frame();

        // Get arena allocator for this frame
        auto arena_allocator = arena.get_allocator(frame_id);

        // Allocate buffers through arena
        size_t per_primitive_size = required<PerPrimitiveBuffers>(n_primitives);
        size_t per_tile_size = required<PerTileBuffers>(n_tiles);

        char* per_primitive_buffers_blob = arena_allocator(per_primitive_size);
        char* per_tile_buffers_blob = arena_allocator(per_tile_size);

        if (!per_primitive_buffers_blob || !per_tile_buffers_blob) {
            arena.end_frame(frame_id);
            return {frame_id, false, "OUT_OF_MEMORY: Failed to allocate initial buffers from arena"};
        }

        // Create allocation wrappers
        std::function<char*(size_t)> per_primitive_buffers_func =
            [&per_primitive_buffers_blob](size_t size) -> char* {
            // Already allocated, just return the pointer
            return per_primitive_buffers_blob;
        };

        std::function<char*(size_t)> per_tile_buffers_func =
            [&per_tile_buffers_blob](size_t size) -> char* {
            return per_tile_buffers_blob;
        };

        // These will be allocated later based on n_instances
        char* per_instance_buffers_blob = nullptr;

        std::function<char*(size_t)> per_instance_buffers_func =
            [&arena_allocator, &per_instance_buffers_blob](size_t size) -> char* {
            per_instance_buffers_blob = arena_allocator(size);
            if (!per_instance_buffers_blob) {
                // Throw immediately to prevent nullptr from being used
                throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate instance buffers");
            }
            return per_instance_buffers_blob;
        };

        try {
            // Call the actual forward implementation
            const int n_instances = edge_forward(per_primitive_buffers_func,
                                                 per_tile_buffers_func,
                                                 per_instance_buffers_func,
                                                 reinterpret_cast<const float3*>(means_ptr),
                                                 reinterpret_cast<const float3*>(scales_raw_ptr),
                                                 reinterpret_cast<const float4*>(rotations_raw_ptr),
                                                 opacities_raw_ptr,
                                                 reinterpret_cast<const float4*>(w2c_ptr),
                                                 n_primitives,
                                                 width,
                                                 height,
                                                 focal_x,
                                                 focal_y,
                                                 center_x,
                                                 center_y,
                                                 near_plane,
                                                 far_plane,
                                                 mip_filter,
                                                 pixel_weights,
                                                 accum_weights);

            // Verify allocations happened
            if (n_instances > 0 && !per_instance_buffers_blob) {
                arena.end_frame(frame_id);
                return {frame_id, false, "OUT_OF_MEMORY: Instance buffers were not allocated despite n_instances > 0"};
            }

            return {frame_id, true, nullptr};

        } catch (const std::exception& e) {
            // Clean up frame on error and return error context instead of throwing
            arena.end_frame(frame_id);
            static thread_local std::string error_message;
            error_message = e.what();
            return {frame_id, false, error_message.c_str()};
        }
    }

} // namespace edge_compute::rasterization
