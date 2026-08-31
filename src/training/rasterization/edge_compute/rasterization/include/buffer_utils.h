/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../../../fastgs/rasterization/include/visibility.cuh"
#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "edge_rasterization_config.h"
#include "helper_math.h"
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace edge_compute::rasterization {

    using InstanceKey = std::uint32_t;

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

    inline int packed_instance_depth_bits(uint n_tiles) {
        const int tile_bits = n_tiles <= 1 ? 0 : extract_end_bit(n_tiles - 1);
        const int depth_bits = 32 - tile_bits;
        return depth_bits > 23 ? 23 : (depth_bits < 0 ? 0 : depth_bits);
    }

    inline int packed_instance_key_end_bit(uint n_tiles) {
        const int tile_bits = n_tiles <= 1 ? 0 : extract_end_bit(n_tiles - 1);
        return tile_bits + packed_instance_depth_bits(n_tiles);
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

    struct VisibilityBuffers {
        uint* visibility_mask;
        uint* block_counts;
        uint* block_offsets;
        uint* visible_indices;
        uint* primitive_work_indices;
        char* cub_workspace;
        size_t cub_workspace_size;

        static VisibilityBuffers from_blob(char*& blob, int n_primitives) {
            VisibilityBuffers buffers{};
            const size_t mask_words = (static_cast<size_t>(n_primitives) + 31u) / 32u;
            const size_t n_blocks = (static_cast<size_t>(n_primitives) +
                                     lfs::rasterization::visibility::kBlockSize - 1u) /
                                    lfs::rasterization::visibility::kBlockSize;
            obtain(blob, buffers.visibility_mask, mask_words, 128);
            obtain(blob, buffers.block_counts, n_blocks, 128);
            obtain(blob, buffers.block_offsets, n_blocks, 128);
            obtain(blob, buffers.visible_indices, n_primitives, 128);
            obtain(blob, buffers.primitive_work_indices, n_primitives, 128);
            LFS_CUDA_CHECK_MSG(
                cub::DeviceScan::InclusiveSum(
                    nullptr, buffers.cub_workspace_size,
                    buffers.block_counts, buffers.block_offsets,
                    static_cast<int>(n_blocks)),
                "EDGE visibility workspace query");
            LFS_ASSERT_MSG(buffers.cub_workspace_size > 0,
                           "EDGE visibility scan returned an empty workspace for nonempty input");
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            return buffers;
        }
    };

    struct PerPrimitiveBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        uint* depth_keys;
        uint* n_touched_tiles;
        uint* offset;
        ushort4* screen_bounds;
        float2* mean2d;
        float4* conic_opacity;

        static PerPrimitiveBuffers from_blob(char*& blob, int n_primitives) {
            PerPrimitiveBuffers buffers{};
            obtain(blob, buffers.depth_keys, n_primitives, 128);
            obtain(blob, buffers.n_touched_tiles, n_primitives, 128);
            obtain(blob, buffers.offset, n_primitives, 128);
            obtain(blob, buffers.screen_bounds, n_primitives, 128);
            obtain(blob, buffers.mean2d, n_primitives, 128);
            obtain(blob, buffers.conic_opacity, n_primitives, 128);
            LFS_CUDA_CHECK_MSG(
                cub::DeviceScan::InclusiveSum(
                    nullptr, buffers.cub_workspace_size,
                    buffers.n_touched_tiles, buffers.offset,
                    n_primitives),
                "cub::DeviceScan::InclusiveSum workspace query");
            LFS_ASSERT_MSG(buffers.cub_workspace_size > 0,
                           "CUB scan returned an empty workspace for nonempty primitive input");
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            return buffers;
        }
    };

    struct PerInstanceBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        cub::DoubleBuffer<InstanceKey> keys;
        cub::DoubleBuffer<uint> primitive_indices;

        static PerInstanceBuffers from_blob(char*& blob, int n_instances, int end_bit = 16) {
            PerInstanceBuffers buffers{};
            InstanceKey* keys_current;
            obtain(blob, keys_current, n_instances, 128);
            InstanceKey* keys_alternate;
            obtain(blob, keys_alternate, n_instances, 128);
            buffers.keys = cub::DoubleBuffer<InstanceKey>(keys_current, keys_alternate);
            uint* primitive_indices_current;
            obtain(blob, primitive_indices_current, n_instances, 128);
            uint* primitive_indices_alternate;
            obtain(blob, primitive_indices_alternate, n_instances, 128);
            buffers.primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current, primitive_indices_alternate);
            LFS_CUDA_CHECK_MSG(
                cub::DeviceRadixSort::SortPairs(
                    nullptr, buffers.cub_workspace_size,
                    buffers.keys, buffers.primitive_indices,
                    n_instances, 0, end_bit),
                "cub::DeviceRadixSort::SortPairs workspace query");
            LFS_ASSERT_MSG(buffers.cub_workspace_size > 0,
                           "CUB radix sort returned an empty workspace for nonempty instance input");
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            return buffers;
        }
    };

    // The sort phase is one exact 256-byte-aligned allocation. The sorted
    // primitive IDs are the only payload EDGE retains after radix sort; keys,
    // the alternate payload, and CUB scratch are phase-local.
    struct EdgeSortWorkspace {
        char* base = nullptr;
        size_t cub_workspace_bytes = 0;
        size_t cub_workspace_offset_bytes = 0;
        int n_instances = 0;

        // Keep the four arrays in the same order and alignment as the legacy
        // PerInstanceBuffers allocator. This also keeps CUB on the same
        // aligned path when the compacted count is not a 128-byte multiple.
        static constexpr size_t buffer_alignment = 128;
        static constexpr size_t alignment = 256;

        static size_t align_offset(size_t bytes, size_t alignment) {
            if (bytes > SIZE_MAX - (alignment - 1))
                throw std::overflow_error("EDGE sort workspace alignment overflow");
            return (bytes + alignment - 1) & ~(alignment - 1);
        }

        static size_t data_bytes(int instances) {
            const size_t bytes = static_cast<size_t>(instances) * sizeof(InstanceKey);
            size_t offset = bytes;
            offset = align_offset(offset, buffer_alignment) + bytes;
            offset = align_offset(offset, buffer_alignment) + bytes;
            offset = align_offset(offset, buffer_alignment) + bytes;
            return offset;
        }

        static size_t aligned_offset(size_t bytes) {
            return align_offset(bytes, alignment);
        }

        void bind(char* allocation, int instances, size_t cub_bytes, size_t cub_offset) {
            LFS_ASSERT_MSG(allocation != nullptr && instances > 0,
                           "EDGE sort workspace requires a nonempty allocation");
            base = allocation;
            n_instances = instances;
            cub_workspace_bytes = cub_bytes;
            cub_workspace_offset_bytes = cub_offset;
            LFS_ASSERT_MSG(reinterpret_cast<std::uintptr_t>(cub_workspace()) % alignment == 0,
                           "EDGE CUB sort workspace must be 256-byte aligned");
        }

        size_t per_buffer_bytes() const noexcept {
            return static_cast<size_t>(n_instances) * sizeof(InstanceKey);
        }

        size_t keys_alternate_offset() const noexcept {
            return align_offset(per_buffer_bytes(), buffer_alignment);
        }

        size_t primitive_indices_current_offset() const noexcept {
            return align_offset(keys_alternate_offset() + per_buffer_bytes(), buffer_alignment);
        }

        size_t primitive_indices_alternate_offset() const noexcept {
            return align_offset(primitive_indices_current_offset() + per_buffer_bytes(), buffer_alignment);
        }

        uint* primitive_indices_current() const noexcept {
            return reinterpret_cast<uint*>(base + primitive_indices_current_offset());
        }
        uint* primitive_indices_alternate() const noexcept {
            return reinterpret_cast<uint*>(base + primitive_indices_alternate_offset());
        }
        InstanceKey* keys_current() const noexcept {
            return reinterpret_cast<InstanceKey*>(base);
        }
        InstanceKey* keys_alternate() const noexcept {
            return reinterpret_cast<InstanceKey*>(base + keys_alternate_offset());
        }
        void* cub_workspace() const noexcept {
            return base + cub_workspace_offset_bytes;
        }
    };

    struct PerTileBuffers {
        uint2* instance_ranges;

        static PerTileBuffers from_blob(char*& blob, int n_tiles) {
            PerTileBuffers buffers;
            obtain(blob, buffers.instance_ranges, n_tiles, 128);
            return buffers;
        }
    };

} // namespace edge_compute::rasterization
