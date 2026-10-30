/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "point_filter_kernels.hpp"
#include "point_tree_api.h"
namespace {
    template <bool Final>
    __global__ void morton_bounds_native(const float* input, float* output, size_t count) {
        __shared__ float values[6][256];
        const unsigned lane = threadIdx.x;
        float lo[3]{INFINITY, INFINITY, INFINITY}, hi[3]{-INFINITY, -INFINITY, -INFINITY};
        for (size_t i = size_t(blockIdx.x) * 256 + lane; i < count; i += size_t(gridDim.x) * 256)
            for (unsigned a = 0; a < 3; ++a) {
                lo[a] = fminf(lo[a], input[i * (Final ? 6 : 3) + a]);
                hi[a] = fmaxf(hi[a], input[i * (Final ? 6 : 3) + (Final ? 3 : 0) + a]);
            }
        for (unsigned a = 0; a < 3; ++a) {
            values[a][lane] = lo[a];
            values[a + 3][lane] = hi[a];
        }
        __syncthreads();
        for (unsigned step = 128; step; step /= 2) {
            if (lane < step)
                for (unsigned a = 0; a < 3; ++a) {
                    values[a][lane] = fminf(values[a][lane], values[a][lane + step]);
                    values[a + 3][lane] = fmaxf(values[a + 3][lane], values[a + 3][lane + step]);
                }
            __syncthreads();
        }
        if (!lane)
            for (unsigned a = 0; a < 6; ++a)
                output[size_t(blockIdx.x) * 6 + a] = values[a][0];
    }
    __global__ void morton_keys_native(const float* points, const float* bounds, uint64_t* keys, int32_t* indices, size_t count) {
#pragma clang fp contract(off)
        for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < count; i += size_t(gridDim.x) * 256) {
            uint32_t axis[3];
            for (unsigned a = 0; a < 3; ++a)
                axis[a] = lfs::core::morton_coordinate(points[i * 3 + a], bounds[a], lfs::core::morton_multiplier(bounds[a + 3] - bounds[a]));
            keys[i] = lfs::core::morton_encode(axis[0], axis[1], axis[2]);
            indices[i] = int32_t(i);
        }
    }
} // namespace
int lfs_hip_morton(LfsHipContext* context, LfsHipView positions, LfsHipView keys, LfsHipView indices, size_t count, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && count <= INT32_MAX, "Invalid HIP Morton count");
        if (!count)
            return;
        const LfsHipView views[]{positions, keys, indices};
        const size_t sizes[]{count * 12, count * 8, count * 4};
        for (unsigned i = 0; i < 3; ++i) {
            validate(context, views[i].buffer, views[i].offset, sizes[i]);
            require(views[i].offset % (i == 1 ? 8 : 4) == 0, "Misaligned HIP Morton storage");
            for (unsigned j = 0; j < i; ++j)
                require(views[i].buffer != views[j].buffer || views[i].offset >= views[j].offset + sizes[j] ||
                            views[j].offset >= views[i].offset + sizes[i],
                        "Overlapping HIP Morton ranges");
        }
        DeviceScope device(context->device);
        auto* output_keys = reinterpret_cast<uint64_t*>(static_cast<char*>(keys.buffer->data) + keys.offset);
        auto* output_indices = reinterpret_cast<int32_t*>(static_cast<char*>(indices.buffer->data) + indices.offset);
        const auto* input = reinterpret_cast<const float*>(static_cast<const char*>(positions.buffer->data) + positions.offset);
        size_t temporary_bytes = 0;
        check(hipcub::DeviceRadixSort::SortPairs(nullptr, temporary_bytes, output_keys, output_keys, output_indices, output_indices, int(count), 0, 63, context->stream));
        const size_t groups = std::min<size_t>((count + 255) / 256, 4096);
        const auto aligned = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
        const size_t bounds_bytes = aligned(groups * 24), key_bytes = aligned(count * 8), index_bytes = aligned(count * 4);
        const size_t buffers = bounds_bytes + 256 + key_bytes + index_bytes;
        require(temporary_bytes <= SIZE_MAX - buffers, "HIP Morton workspace overflow");
        const size_t required = buffers + temporary_bytes;
        ensure_workspace(context, context->sort_workspace, context->sort_workspace_bytes, required);
        auto* memory = static_cast<char*>(context->sort_workspace);
        auto* partials = reinterpret_cast<float*>(memory);
        auto* bounds = reinterpret_cast<float*>(memory + bounds_bytes);
        auto* unsorted_keys = reinterpret_cast<uint64_t*>(memory + bounds_bytes + 256);
        auto* unsorted_indices = reinterpret_cast<int32_t*>(memory + bounds_bytes + 256 + key_bytes);
        morton_bounds_native<false><<<groups, 256, 0, context->stream>>>(input, partials, count);
        morton_bounds_native<true><<<1, 256, 0, context->stream>>>(partials, bounds, groups);
        morton_keys_native<<<groups, 256, 0, context->stream>>>(input, bounds, unsorted_keys, unsorted_indices, count);
        check(hipGetLastError());
        check(hipcub::DeviceRadixSort::SortPairs(memory + buffers, temporary_bytes, unsorted_keys, output_keys, unsorted_indices, output_indices, int(count), 0, 63, context->stream));
    });
}
