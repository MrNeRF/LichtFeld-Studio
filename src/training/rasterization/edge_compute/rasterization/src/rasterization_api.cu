/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "buffer_utils.h"
#include "core/cuda/memory_arena.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "edge_phase_arena.h"
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
        float* accum_weights,
        cudaStream_t stream) {
        if (stream == nullptr) {
            stream = lfs::core::getCurrentCUDAStream();
        }
        // Validate inputs using pure CUDA validation
        LFS_VALIDATE_CUDA_DEVICE_POINTER(means_ptr, "means_ptr");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(scales_raw_ptr, "scales_raw_ptr");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(rotations_raw_ptr, "rotations_raw_ptr");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(opacities_raw_ptr, "opacities_raw_ptr");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(w2c_ptr, "w2c_ptr");

        if (n_primitives <= 0 || width <= 0 || height <= 0) {
            throw std::runtime_error("Invalid dimensions in forward_raw");
        }

        // Calculate grid dimensions
        const dim3 grid(div_round_up(width, config::tile_width),
                        div_round_up(height, config::tile_height), 1);
        const int n_tiles = grid.x * grid.y;

        // Get global arena and begin frame
        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
        uint64_t frame_id = arena.begin_frame(stream);

        // Get arena allocator for this frame
        auto arena_allocator = arena.get_allocator(frame_id);

        try {
            // Workspace queries are part of the allocation transaction: if CUB
            // rejects one, the arena frame must be released through this boundary.
            size_t per_tile_size = required<PerTileBuffers>(n_tiles);

            char* per_tile_buffers_blob = arena_allocator(per_tile_size);

            if (!per_tile_buffers_blob) {
                arena.end_frame(frame_id, stream);
                return {frame_id, false, "OUT_OF_MEMORY: Failed to allocate initial buffers from arena"};
            }

            // Create allocation wrappers
            std::function<char*(size_t)> per_tile_buffers_func =
                [&per_tile_buffers_blob](size_t size) -> char* {
                return per_tile_buffers_blob;
            };

            auto phase_arena = std::make_shared<EdgePhaseArena>(arena_allocator);
            auto phase_allocator = phase_arena->allocator();

            // Call the actual forward implementation
            const int n_instances = edge_forward(
                arena_allocator,
                [phase_arena](size_t size) {
                    phase_arena->begin(size);
                },
                phase_allocator,
                per_tile_buffers_func,
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
                accum_weights,
                stream);

            return {frame_id, true, nullptr};

        } catch (const std::exception& e) {
            // Clean up frame on error and return error context instead of throwing
            arena.end_frame(frame_id, stream);
            static thread_local std::string error_message;
            error_message = e.what();
            return {frame_id, false, error_message.c_str()};
        }
    }

    ForwardResult edge_forward_raw_legacy_for_testing(
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
        float* accum_weights,
        cudaStream_t stream) {
        if (stream == nullptr)
            stream = lfs::core::getCurrentCUDAStream();
        LFS_VALIDATE_CUDA_DEVICE_POINTER(means_ptr, "means_ptr");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(scales_raw_ptr, "scales_raw_ptr");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(rotations_raw_ptr, "rotations_raw_ptr");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(opacities_raw_ptr, "opacities_raw_ptr");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(w2c_ptr, "w2c_ptr");
        if (n_primitives <= 0 || width <= 0 || height <= 0)
            throw std::runtime_error("Invalid dimensions in legacy forward_raw");

        const dim3 grid(div_round_up(width, config::tile_width),
                        div_round_up(height, config::tile_height), 1);
        const int n_tiles = static_cast<int>(grid.x * grid.y);
        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
        const uint64_t frame_id = arena.begin_frame(stream);
        auto arena_allocator = arena.get_allocator(frame_id);
        try {
            const size_t primitive_size = required<PerPrimitiveBuffers>(n_primitives);
            const size_t tile_size = required<PerTileBuffers>(n_tiles);
            char* primitive_blob = arena_allocator(primitive_size);
            char* tile_blob = arena_allocator(tile_size);
            if (!primitive_blob || !tile_blob) {
                arena.end_frame(frame_id, stream);
                return {frame_id, false, "OUT_OF_MEMORY: Failed to allocate legacy EDGE buffers"};
            }
            char* instance_blob = nullptr;
            const int n_instances = edge_forward_legacy(
                [primitive_blob](size_t) { return primitive_blob; },
                [tile_blob](size_t) { return tile_blob; },
                [&arena_allocator, &instance_blob](size_t size) {
                    instance_blob = arena_allocator(size);
                    if (!instance_blob)
                        throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate legacy EDGE sort buffers");
                    return instance_blob;
                },
                reinterpret_cast<const float3*>(means_ptr),
                reinterpret_cast<const float3*>(scales_raw_ptr),
                reinterpret_cast<const float4*>(rotations_raw_ptr),
                opacities_raw_ptr,
                reinterpret_cast<const float4*>(w2c_ptr),
                n_primitives, width, height, focal_x, focal_y, center_x, center_y,
                near_plane, far_plane, mip_filter, pixel_weights, accum_weights, stream);
            if (n_instances > 0 && !instance_blob) {
                arena.end_frame(frame_id, stream);
                return {frame_id, false, "OUT_OF_MEMORY: Legacy EDGE sort buffers were not allocated"};
            }
            return {frame_id, true, nullptr};
        } catch (const std::exception& e) {
            arena.end_frame(frame_id, stream);
            static thread_local std::string error_message;
            error_message = e.what();
            return {frame_id, false, error_message.c_str()};
        }
    }

} // namespace edge_compute::rasterization
