/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cmath>
#include <cooperative_groups.h>
#include <cstdio>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/transform_iterator.h>

#include "Common.h"
#include "Intersect.h"
#include "Utils.cuh"

namespace gsplat_lfs {

    namespace cg = cooperative_groups;

    template <typename scalar_t>
    __global__ void intersect_tile_kernel(
        const bool packed,
        const uint32_t C,
        const uint32_t N,
        const uint32_t nnz,
        const int64_t* __restrict__ camera_ids,
        const int64_t* __restrict__ gaussian_ids,
        const scalar_t* __restrict__ means2d,
        const int32_t* __restrict__ radii,
        const scalar_t* __restrict__ depths,
        const int64_t* __restrict__ cum_tiles_per_gauss,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const uint32_t tile_n_bits,
        int32_t* __restrict__ tiles_per_gauss,
        int64_t* __restrict__ isect_ids,
        int32_t* __restrict__ flatten_ids,
        const int64_t max_isects) {
        uint32_t idx = cg::this_grid().thread_rank();
        bool first_pass = cum_tiles_per_gauss == nullptr;
        if (idx >= (packed ? nnz : C * N)) {
            return;
        }

        const float radius_x = radii[idx * 2];
        const float radius_y = radii[idx * 2 + 1];
        if (radius_x <= 0 || radius_y <= 0) {
            if (first_pass) {
                tiles_per_gauss[idx] = 0;
            }
            return;
        }

        vec2 mean2d = glm::make_vec2(means2d + 2 * idx);

        float tile_radius_x = radius_x / static_cast<float>(tile_size);
        float tile_radius_y = radius_y / static_cast<float>(tile_size);
        float tile_x = mean2d.x / static_cast<float>(tile_size);
        float tile_y = mean2d.y / static_cast<float>(tile_size);

        uint2 tile_min, tile_max;
        tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius_x)), tile_width);
        tile_min.y = min(max(0, (uint32_t)floor(tile_y - tile_radius_y)), tile_height);
        tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius_x)), tile_width);
        tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius_y)), tile_height);

        if (first_pass) {
            tiles_per_gauss[idx] = static_cast<int32_t>(
                (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x));
            return;
        }

        int64_t cid;
        if (packed) {
            cid = camera_ids[idx];
        } else {
            cid = idx / N;
        }
        const int64_t cid_enc = cid << (32 + tile_n_bits);

        int32_t depth_i32 = *(int32_t*)&(depths[idx]);
        int64_t depth_id_enc = static_cast<uint32_t>(depth_i32);

        int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
        for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
            for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
                if (max_isects >= 0 && cur_idx >= max_isects) {
                    return;
                }
                int64_t tile_id = i * tile_width + j;
                isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
                flatten_ids[cur_idx] = static_cast<int32_t>(idx);
                ++cur_idx;
            }
        }
    }

    void launch_intersect_tile_kernel(
        const float* means2d,
        const int32_t* radii,
        const float* depths,
        const int64_t* camera_ids,
        const int64_t* gaussian_ids,
        uint32_t C,
        uint32_t N,
        uint32_t nnz,
        bool packed,
        uint32_t tile_size,
        uint32_t tile_width,
        uint32_t tile_height,
        const int64_t* cum_tiles_per_gauss,
        int32_t* tiles_per_gauss,
        int64_t* isect_ids,
        int32_t* flatten_ids,
        cudaStream_t stream,
        int64_t max_isects) {
        int64_t n_elements = packed ? nnz : C * N;

        uint32_t n_tiles = tile_width * tile_height;
        uint32_t tile_n_bits = static_cast<uint32_t>(floor(log2(n_tiles))) + 1;

        if (n_elements == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((n_elements + threads.x - 1) / threads.x);

        intersect_tile_kernel<float><<<grid, threads, 0, stream>>>(
            packed,
            C, N, nnz,
            camera_ids, gaussian_ids,
            means2d, radii, depths,
            cum_tiles_per_gauss,
            tile_size, tile_width, tile_height, tile_n_bits,
            tiles_per_gauss, isect_ids, flatten_ids, max_isects);
        LFS_CUDA_LAUNCH_CHECK(stream, "gsplat.intersect_tile");
    }

    __global__ void fill_isect_sentinels_kernel(
        int64_t* __restrict__ isect_ids,
        int32_t* __restrict__ flatten_ids,
        const int64_t n,
        const int64_t sentinel_key) {
        const int64_t idx = static_cast<int64_t>(cg::this_grid().thread_rank());
        if (idx >= n) {
            return;
        }
        isect_ids[idx] = sentinel_key;
        flatten_ids[idx] = -1;
    }

    void launch_fill_isect_sentinels_kernel(
        int64_t* isect_ids,
        int32_t* flatten_ids,
        int64_t n,
        int64_t sentinel_key,
        cudaStream_t stream) {
        if (n <= 0) {
            return;
        }
        dim3 threads(256);
        dim3 grid(static_cast<unsigned>((n + 255) / 256));
        fill_isect_sentinels_kernel<<<grid, threads, 0, stream>>>(
            isect_ids, flatten_ids, n, sentinel_key);
        LFS_CUDA_LAUNCH_CHECK(stream, "gsplat.fill_isect_sentinels");
    }

    __global__ void intersect_offset_kernel(
        const uint32_t n_keys,
        const int64_t* __restrict__ isect_ids,
        const uint32_t C,
        const uint32_t n_tiles,
        const uint32_t tile_n_bits,
        int32_t* __restrict__ offsets) {
        uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= n_keys)
            return;

        const int64_t tile_mask = (1LL << tile_n_bits) - 1;
        const int64_t isect_id_curr = isect_ids[idx] >> 32;
        const int64_t cid_curr = isect_id_curr >> tile_n_bits;
        const int64_t tid_curr = isect_id_curr & tile_mask;
        const bool sentinel =
            tid_curr >= static_cast<int64_t>(n_tiles) ||
            cid_curr >= static_cast<int64_t>(C);
        const int64_t n_slots = static_cast<int64_t>(C) * static_cast<int64_t>(n_tiles);
        const int64_t id_curr = sentinel ? n_slots
                                         : cid_curr * static_cast<int64_t>(n_tiles) + tid_curr;

        auto fill_range = [&](int64_t begin, int64_t end, int32_t value) {
            if (begin < 0) {
                begin = 0;
            }
            if (end > n_slots) {
                end = n_slots;
            }
            for (int64_t i = begin; i <= end; ++i) {
                offsets[i] = value;
            }
        };

        if (idx == 0) {
            fill_range(0, id_curr, 0);
        }
        if (idx == n_keys - 1 && !sentinel) {
            fill_range(id_curr + 1, n_slots, static_cast<int32_t>(n_keys));
        }

        if (idx > 0) {
            const int64_t isect_id_prev = isect_ids[idx - 1] >> 32;
            if (isect_id_prev == isect_id_curr)
                return;

            const int64_t cid_prev = isect_id_prev >> tile_n_bits;
            const int64_t tid_prev = isect_id_prev & tile_mask;
            const bool prev_sentinel =
                tid_prev >= static_cast<int64_t>(n_tiles) ||
                cid_prev >= static_cast<int64_t>(C);
            const int64_t id_prev = prev_sentinel
                                        ? n_slots
                                        : cid_prev * static_cast<int64_t>(n_tiles) + tid_prev;
            fill_range(id_prev + 1, id_curr, static_cast<int32_t>(idx));
        }
    }

    void launch_intersect_offset_kernel(
        const int64_t* isect_ids,
        uint32_t n_keys,
        uint32_t C,
        uint32_t tile_width,
        uint32_t tile_height,
        int32_t* offsets,
        cudaStream_t stream) {
        const uint32_t n_tiles = tile_width * tile_height;
        const size_t offset_count =
            static_cast<size_t>(C) * static_cast<size_t>(n_tiles) + 1u;
        if (n_keys == 0) {
            LFS_CUDA_CHECK_MSG(
                cudaMemsetAsync(offsets, 0, offset_count * sizeof(int32_t), stream),
                "gsplat empty intersection-offset kernel output");
            return;
        }

        dim3 threads(256);
        dim3 grid((n_keys + threads.x - 1) / threads.x);

        uint32_t tile_n_bits = static_cast<uint32_t>(floor(log2(n_tiles))) + 1;

        intersect_offset_kernel<<<grid, threads, 0, stream>>>(
            n_keys, isect_ids, C, n_tiles, tile_n_bits, offsets);
        LFS_CUDA_LAUNCH_CHECK(stream, "gsplat.intersect_offset");
    }

    void radix_sort_double_buffer(
        int64_t n_isects,
        uint32_t tile_n_bits,
        uint32_t cam_n_bits,
        int64_t* isect_ids,
        int32_t* flatten_ids,
        int64_t* isect_ids_sorted,
        int32_t* flatten_ids_sorted,
        int64_t** keys_out,
        int32_t** vals_out,
        size_t& cub_ws_bytes,
        int64_t& cub_n,
        uint32_t& cub_end_bit,
        cudaStream_t stream) {
        if (n_isects <= 0) {
            if (keys_out) {
                *keys_out = isect_ids;
            }
            if (vals_out) {
                *vals_out = flatten_ids;
            }
            return;
        }

        const uint32_t end_bit = 32 + tile_n_bits + cam_n_bits;
        if (n_isects != cub_n || end_bit != cub_end_bit || cub_ws_bytes == 0) {
            cub::DoubleBuffer<int64_t> query_keys(isect_ids, isect_ids_sorted);
            cub::DoubleBuffer<int32_t> query_vals(flatten_ids, flatten_ids_sorted);
            size_t ws = 0;
            LFS_CUDA_CHECK_MSG(
                cub::DeviceRadixSort::SortPairs(
                    nullptr, ws, query_keys, query_vals, n_isects,
                    0, end_bit, stream),
                "cub::DeviceRadixSort::SortPairs workspace query");
            LFS_ASSERT_MSG(ws > 0,
                           "gsplat CUB radix sort returned an empty workspace");
            cub_ws_bytes = ws;
            cub_n = n_isects;
            cub_end_bit = end_bit;
        }

        cub::DoubleBuffer<int64_t> d_keys(isect_ids, isect_ids_sorted);
        cub::DoubleBuffer<int32_t> d_values(flatten_ids, flatten_ids_sorted);
        void* const workspace = ensure_gsplat_cub_workspace(cub_ws_bytes, stream);
        size_t storage_bytes = cub_ws_bytes;
        LFS_CUDA_CHECK_MSG(
            cub::DeviceRadixSort::SortPairs(
                workspace, storage_bytes, d_keys, d_values, n_isects,
                0, end_bit, stream),
            "cub::DeviceRadixSort::SortPairs");
        if (keys_out) {
            *keys_out = d_keys.Current();
        }
        if (vals_out) {
            *vals_out = d_values.Current();
        }
    }

    void compute_cumsum_gpu(
        const int32_t* input,
        int64_t* output,
        uint32_t n_elements,
        cudaStream_t stream) {
        if (n_elements == 0) {
            return;
        }

        auto cast_op = [] __host__ __device__(int32_t x) { return static_cast<int64_t>(x); };
        auto cast_iter = thrust::make_transform_iterator(input, cast_op);

        run_cub_operation(
            "cub::DeviceScan::InclusiveSum", stream,
            [&](void* workspace, size_t& workspace_bytes) {
                return cub::DeviceScan::InclusiveSum(
                    workspace, workspace_bytes, cast_iter, output, n_elements, stream);
            });
    }

} // namespace gsplat_lfs
