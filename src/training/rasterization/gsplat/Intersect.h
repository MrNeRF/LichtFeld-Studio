/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace gsplat_lfs {

    void launch_intersect_tile_kernel(
        // inputs
        const float* means2d,        // [C, N, 2] or [nnz, 2]
        const int32_t* radii,        // [C, N, 2] or [nnz, 2]
        const float* depths,         // [C, N] or [nnz]
        const int64_t* camera_ids,   // [nnz] optional (nullptr for dense)
        const int64_t* gaussian_ids, // [nnz] optional (nullptr for dense)
        uint32_t C,
        uint32_t N,   // gaussians per camera (for dense)
        uint32_t nnz, // total non-zeros (for packed)
        bool packed,
        uint32_t tile_size,
        uint32_t tile_width,
        uint32_t tile_height,
        const int64_t* cum_tiles_per_gauss, // [C, N] or [nnz] optional (nullptr for first pass)
        // outputs
        int32_t* tiles_per_gauss, // [C, N] or [nnz] optional (for first pass)
        int64_t* isect_ids,       // [n_isects] optional (for second pass)
        int32_t* flatten_ids,     // [n_isects] optional (for second pass)
        cudaStream_t stream = nullptr);

    void launch_intersect_offset_kernel(
        // inputs
        const int64_t* isect_ids, // [n_isects]
        uint32_t n_isects,
        uint32_t C,
        uint32_t tile_width,
        uint32_t tile_height,
        // outputs
        int32_t* offsets, // [C, tile_height, tile_width]
        cudaStream_t stream = nullptr);

    void radix_sort_double_buffer(
        int64_t n_isects,
        uint32_t tile_n_bits,
        uint32_t cam_n_bits,
        int64_t* isect_ids,
        int32_t* flatten_ids,
        int64_t* isect_ids_sorted,
        int32_t* flatten_ids_sorted,
        cudaStream_t stream = nullptr);

    void compute_cumsum_gpu(
        const int32_t* input,
        int64_t* output,
        uint32_t n_elements,
        cudaStream_t stream = nullptr);

} // namespace gsplat_lfs
