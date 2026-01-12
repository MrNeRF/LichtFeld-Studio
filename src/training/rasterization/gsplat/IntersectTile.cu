/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cmath>
#include <cooperative_groups.h>
#include <cstdio>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

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
        int32_t* __restrict__ flatten_ids) {
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
        cudaStream_t stream) {
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
            tiles_per_gauss, isect_ids, flatten_ids);
    }

    __global__ void intersect_offset_kernel(
        const uint32_t n_isects,
        const int64_t* __restrict__ isect_ids,
        const uint32_t C,
        const uint32_t n_tiles,
        const uint32_t tile_n_bits,
        int32_t* __restrict__ offsets) {
        uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= n_isects)
            return;

        int64_t isect_id_curr = isect_ids[idx] >> 32;
        int64_t cid_curr = isect_id_curr >> tile_n_bits;
        int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
        int64_t id_curr = cid_curr * n_tiles + tid_curr;

        if (idx == 0) {
            for (uint32_t i = 0; i < id_curr + 1; ++i)
                offsets[i] = static_cast<int32_t>(idx);
        }
        if (idx == n_isects - 1) {
            for (uint32_t i = id_curr + 1; i < C * n_tiles; ++i)
                offsets[i] = static_cast<int32_t>(n_isects);
        }

        if (idx > 0) {
            int64_t isect_id_prev = isect_ids[idx - 1] >> 32;
            if (isect_id_prev == isect_id_curr)
                return;

            int64_t cid_prev = isect_id_prev >> tile_n_bits;
            int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
            int64_t id_prev = cid_prev * n_tiles + tid_prev;
            for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
                offsets[i] = static_cast<int32_t>(idx);
        }
    }

    void launch_intersect_offset_kernel(
        const int64_t* isect_ids,
        uint32_t n_isects,
        uint32_t C,
        uint32_t tile_width,
        uint32_t tile_height,
        int32_t* offsets,
        cudaStream_t stream) {
        if (n_isects == 0) {
            cudaMemsetAsync(offsets, 0, C * tile_height * tile_width * sizeof(int32_t), stream);
            return;
        }

        dim3 threads(256);
        dim3 grid((n_isects + threads.x - 1) / threads.x);

        uint32_t n_tiles = tile_width * tile_height;
        uint32_t tile_n_bits = static_cast<uint32_t>(floor(log2(n_tiles))) + 1;

        intersect_offset_kernel<<<grid, threads, 0, stream>>>(
            n_isects, isect_ids, C, n_tiles, tile_n_bits, offsets);
    }

    void radix_sort_double_buffer(
        int64_t n_isects,
        uint32_t tile_n_bits,
        uint32_t cam_n_bits,
        int64_t* isect_ids,
        int32_t* flatten_ids,
        int64_t* isect_ids_sorted,
        int32_t* flatten_ids_sorted,
        cudaStream_t stream) {
        if (n_isects <= 0) {
            return;
        }

        cub::DoubleBuffer<int64_t> d_keys(isect_ids, isect_ids_sorted);
        cub::DoubleBuffer<int32_t> d_values(flatten_ids, flatten_ids_sorted);

        CUB_WRAPPER_LFS(
            cub::DeviceRadixSort::SortPairs,
            d_keys,
            d_values,
            n_isects,
            0,
            32 + tile_n_bits + cam_n_bits,
            stream);

        // Copy results to sorted buffers if needed
        if (d_keys.selector == 0) {
            cudaMemcpyAsync(isect_ids_sorted, isect_ids,
                            n_isects * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream);
        }
        if (d_values.selector == 0) {
            cudaMemcpyAsync(flatten_ids_sorted, flatten_ids,
                            n_isects * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
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

        // Use CUB transform iterator for int32→int64 conversion during scan
        cub::TransformInputIterator<int64_t, cub::CastOp<int64_t>, const int32_t*>
            cast_iter(input, cub::CastOp<int64_t>());

        CUB_WRAPPER_LFS(
            cub::DeviceScan::InclusiveSum,
            cast_iter,
            output,
            n_elements,
            stream);
    }

} // namespace gsplat_lfs
