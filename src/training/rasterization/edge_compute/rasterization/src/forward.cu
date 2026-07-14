/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "buffer_utils.h"
#include "edge_rasterization_config.h"
#include "forward.h"
#include "helper_math.h"
#include "kernels_forward.cuh"
#include "utils.h"
#include <cstdint>
#include <cub/cub.cuh>
#include <limits>
#include <stdexcept>

namespace {
    int checked_to_int(uint64_t value, const char* message) {
        if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(message);
        }
        return static_cast<int>(value);
    }
} // namespace

int edge_compute::rasterization::edge_forward(
    std::function<char*(size_t)> per_primitive_buffers_func,
    std::function<char*(size_t)> per_tile_buffers_func,
    std::function<char*(size_t)> per_instance_buffers_func,
    const float3* means,
    const float3* scales_raw,
    const float4* rotations_raw,
    const float* opacities_raw,
    const float4* w2c,
    const int n_primitives,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float near_, // near and far are macros in windows
    const float far_,
    bool mip_filter,
    const float* pixel_weights,
    float* accum_weights,
    cudaStream_t stream) {
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const uint64_t n_tiles_u64 = static_cast<uint64_t>(grid.x) * static_cast<uint64_t>(grid.y);
    const int n_tiles = checked_to_int(n_tiles_u64, "n_tiles exceeds int range");
    const uint n_tiles_u32 = static_cast<uint>(n_tiles);
    const uint depth_bits = static_cast<uint>(packed_instance_depth_bits(n_tiles_u32));
    const int key_end_bit = packed_instance_key_end_bit(n_tiles_u32);

    // Allocate per-tile buffers through arena
    char* per_tile_buffers_blob = per_tile_buffers_func(required<PerTileBuffers>(n_tiles));
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);

    // Initialize tile instance ranges on the frame's stream (see fastgs
    // forward.cu: the old side-stream overlap relied on legacy ordering).
    LFS_CUDA_CHECK_MSG(
        cudaMemsetAsync(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, stream),
        "edge tile-range initialization");

    // Allocate per-primitive buffers through arena
    char* per_primitive_buffers_blob = per_primitive_buffers_func(required<PerPrimitiveBuffers>(n_primitives));
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);

    // Preprocess primitives
    kernels::forward::preprocess_cu<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess, 0, stream>>>(
        means,
        scales_raw,
        rotations_raw,
        opacities_raw,
        w2c,
        per_primitive_buffers.depth_keys,
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.screen_bounds,
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        n_primitives,
        grid.x,
        grid.y,
        static_cast<float>(width),
        static_cast<float>(height),
        fx,
        fy,
        cx,
        cy,
        near_,
        far_,
        depth_bits,
        mip_filter);
    LFS_EDGE_PHASE_CHECK(config::debug, "preprocess");

    LFS_CUDA_CHECK_MSG(
        cub::DeviceScan::InclusiveSum(
            per_primitive_buffers.cub_workspace,
            per_primitive_buffers.cub_workspace_size,
            per_primitive_buffers.n_touched_tiles,
            per_primitive_buffers.offset,
            n_primitives,
            stream),
        "cub::DeviceScan::InclusiveSum (Primitive Offsets)");
    LFS_EDGE_PHASE_CHECK(config::debug, "cub::DeviceScan::InclusiveSum (Primitive Offsets)");

    uint32_t n_instances_u32;
    LFS_CUDA_CHECK_MSG(
        cudaMemcpyAsync(
            &n_instances_u32, per_primitive_buffers.offset + n_primitives - 1,
            sizeof(n_instances_u32), cudaMemcpyDeviceToHost, stream),
        "edge instance-count readback");
    LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(stream), "edge instance-count stream sync");
    LFS_EDGE_PHASE_CHECK(config::debug, "cudaMemcpy(n_instances)");
    const int n_instances = checked_to_int(n_instances_u32, "n_instances exceeds int range");

    const int alloc_instances = std::max(n_instances, 1);
    char* per_instance_buffers_blob = per_instance_buffers_func(required<PerInstanceBuffers>(alloc_instances, key_end_bit));
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(per_instance_buffers_blob, alloc_instances, key_end_bit);

    if (n_instances > 0) {
        kernels::forward::create_instances_cu<<<div_round_up(n_primitives, config::block_size_create_instances), config::block_size_create_instances, 0, stream>>>(
            per_primitive_buffers.n_touched_tiles,
            per_primitive_buffers.offset,
            per_primitive_buffers.depth_keys,
            per_primitive_buffers.screen_bounds,
            per_primitive_buffers.mean2d,
            per_primitive_buffers.conic_opacity,
            per_instance_buffers.keys.Current(),
            per_instance_buffers.primitive_indices.Current(),
            grid.x,
            depth_bits,
            n_primitives);
        LFS_EDGE_PHASE_CHECK(config::debug, "create_instances");

        LFS_CUDA_CHECK_MSG(
            cub::DeviceRadixSort::SortPairs(
                per_instance_buffers.cub_workspace,
                per_instance_buffers.cub_workspace_size,
                per_instance_buffers.keys,
                per_instance_buffers.primitive_indices,
                n_instances, 0, key_end_bit,
                stream),
            "cub::DeviceRadixSort::SortPairs (Tile/Depth)");
        LFS_EDGE_PHASE_CHECK(config::debug, "cub::DeviceRadixSort::SortPairs (Tile/Depth)");
    }

    // Extract instance ranges
    if (n_instances > 0) {
        kernels::forward::extract_instance_ranges_cu<<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges, 0, stream>>>(
            per_instance_buffers.keys.Current(),
            per_tile_buffers.instance_ranges,
            depth_bits,
            n_instances);
        LFS_EDGE_PHASE_CHECK(config::debug, "extract_instance_ranges");
    }

    // Perform blending
    kernels::forward::edge_blend_cu<<<grid, block, 0, stream>>>(
        per_tile_buffers.instance_ranges,
        per_instance_buffers.primitive_indices.Current(),
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        width,
        height,
        grid.x,
        pixel_weights,
        accum_weights);
    LFS_EDGE_PHASE_CHECK(config::debug, "blend");

    return n_instances;
}
