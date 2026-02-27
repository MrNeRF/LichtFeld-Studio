/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"
#include "rasterization_config.h"
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_fp16.h>

namespace fast_lfs::rasterization {

    inline int extract_end_bit(uint n) {
        int leading_zeros = 0;
        if ((n & 0xffff0000u) == 0) {
            leading_zeros += 16;
            n <<= 16;
        }
        if ((n & 0xff000000u) == 0) {
            leading_zeros += 8;
            n <<= 8;
        }
        if ((n & 0xf0000000u) == 0) {
            leading_zeros += 4;
            n <<= 4;
        }
        if ((n & 0xc0000000u) == 0) {
            leading_zeros += 2;
            n <<= 2;
        }
        if ((n & 0x80000000u) == 0) {
            leading_zeros += 1;
        }
        return 32 - leading_zeros;
    }

    struct mat3x3 {
        float m11, m12, m13;
        float m21, m22, m23;
        float m31, m32, m33;
    };

    struct __align__(8) mat3x3_triu {
        float m11, m12, m13, m22, m23, m33;
    };

    template <typename T>
    static void obtain(char*& blob, T*& ptr, std::size_t count, std::size_t alignment) {
        std::size_t offset = reinterpret_cast<std::uintptr_t>(blob) + alignment - 1 & ~(alignment - 1);
        ptr = reinterpret_cast<T*>(offset);
        blob = reinterpret_cast<char*>(ptr + count);
    }

    template <typename T, typename... Args>
    size_t required(size_t P, Args... args) {
        char* size = nullptr;
        T::from_blob(size, P, args...);
        return ((size_t)size) + 128;
    }

    struct PerPrimitiveBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        cub::DoubleBuffer<uint> depth_keys;
        cub::DoubleBuffer<uint> primitive_indices;
        uint* n_touched_tiles;
        uint* offset;
        ushort4* screen_bounds;
        float2* mean2d;
        float4* conic_opacity;
        float3* color;
        uint* n_visible_primitives;
        uint* n_instances;

        static PerPrimitiveBuffers from_blob(char*& blob, int n_primitives) {
            PerPrimitiveBuffers buffers;
            uint* depth_keys_current;
            obtain(blob, depth_keys_current, n_primitives, 128);
            uint* depth_keys_alternate;
            obtain(blob, depth_keys_alternate, n_primitives, 128);
            buffers.depth_keys = cub::DoubleBuffer<uint>(depth_keys_current, depth_keys_alternate);
            uint* primitive_indices_current;
            obtain(blob, primitive_indices_current, n_primitives, 128);
            uint* primitive_indices_alternate;
            obtain(blob, primitive_indices_alternate, n_primitives, 128);
            buffers.primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current, primitive_indices_alternate);
            obtain(blob, buffers.n_touched_tiles, n_primitives, 128);
            obtain(blob, buffers.offset, n_primitives, 128);
            obtain(blob, buffers.screen_bounds, n_primitives, 128);
            obtain(blob, buffers.mean2d, n_primitives, 128);
            obtain(blob, buffers.conic_opacity, n_primitives, 128);
            obtain(blob, buffers.color, n_primitives, 128);
            cub::DeviceScan::ExclusiveSum(
                nullptr, buffers.cub_workspace_size,
                buffers.offset, buffers.offset,
                n_primitives);
            size_t sorting_workspace_size;
            cub::DeviceRadixSort::SortPairs(
                nullptr, sorting_workspace_size,
                buffers.depth_keys, buffers.primitive_indices,
                n_primitives);
            buffers.cub_workspace_size = max(buffers.cub_workspace_size, sorting_workspace_size);
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            obtain(blob, buffers.n_visible_primitives, 1, 128);
            obtain(blob, buffers.n_instances, 1, 128);
            return buffers;
        }
    };

    struct PerInstanceBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        cub::DoubleBuffer<ushort> keys;
        cub::DoubleBuffer<uint> primitive_indices;

        static PerInstanceBuffers from_blob(char*& blob, int n_instances, int end_bit = 16) {
            PerInstanceBuffers buffers;
            ushort* keys_current;
            obtain(blob, keys_current, n_instances, 128);
            ushort* keys_alternate;
            obtain(blob, keys_alternate, n_instances, 128);
            buffers.keys = cub::DoubleBuffer<ushort>(keys_current, keys_alternate);
            uint* primitive_indices_current;
            obtain(blob, primitive_indices_current, n_instances, 128);
            uint* primitive_indices_alternate;
            obtain(blob, primitive_indices_alternate, n_instances, 128);
            buffers.primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current, primitive_indices_alternate);
            cub::DeviceRadixSort::SortPairs(
                nullptr, buffers.cub_workspace_size,
                buffers.keys, buffers.primitive_indices,
                n_instances, 0, end_bit);
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            return buffers;
        }
    };

    struct PerTileBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        uint2* instance_ranges;
        uint* n_buckets;
        uint* bucket_offsets;
        uint* max_n_contributions;
        uint* n_contributions;

        static PerTileBuffers from_blob(char*& blob, int n_tiles) {
            PerTileBuffers buffers;
            obtain(blob, buffers.instance_ranges, n_tiles, 128);
            obtain(blob, buffers.n_buckets, n_tiles, 128);
            obtain(blob, buffers.bucket_offsets, n_tiles, 128);
            obtain(blob, buffers.max_n_contributions, n_tiles, 128);
            obtain(blob, buffers.n_contributions, n_tiles * config::block_size_blend, 128);
            cub::DeviceScan::InclusiveSum(
                nullptr, buffers.cub_workspace_size,
                buffers.n_buckets, buffers.bucket_offsets,
                n_tiles);
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            return buffers;
        }
    };

    struct PerBucketBuffers {
        uint* tile_index;
        uint* checkpoint_uint8; // packed RGBA as 4x uint8 (colors [0,4], transmittance [0,1])

        static PerBucketBuffers from_blob(char*& blob, int n_buckets) {
            PerBucketBuffers buffers;
            obtain(blob, buffers.tile_index, n_buckets, 128);
            obtain(blob, buffers.checkpoint_uint8, n_buckets * config::block_size_blend, 128);
            return buffers;
        }
    };

} // namespace fast_lfs::rasterization
