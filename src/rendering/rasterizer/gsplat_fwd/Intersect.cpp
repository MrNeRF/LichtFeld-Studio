/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "Intersect.h"
#include "Common.h"
#include "Ops.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace gsplat_fwd {

    IntersectTileResult intersect_tile(
        const float* means2d,
        const int32_t* radii,
        const float* depths,
        const int32_t* camera_ids,
        const int32_t* gaussian_ids,
        uint32_t C,
        uint32_t N,
        uint32_t tile_size,
        uint32_t tile_width,
        uint32_t tile_height,
        bool sort,
        int32_t* tiles_per_gauss_out,
        cudaStream_t stream) {
        bool packed = (camera_ids != nullptr && gaussian_ids != nullptr);
        uint32_t n_elements = packed ? 0 : C * N; // For non-packed only
        uint32_t nnz = packed ? 0 : 0;            // TODO: For packed mode

        uint32_t n_tiles = tile_width * tile_height;
        uint32_t tile_n_bits = static_cast<uint32_t>(floor(log2(n_tiles))) + 1;
        uint32_t cam_n_bits = static_cast<uint32_t>(floor(log2(C))) + 1;

        IntersectTileResult result = {};
        result.tiles_per_gauss = tiles_per_gauss_out;
        result.isect_ids = nullptr;
        result.flatten_ids = nullptr;
        result.n_isects = 0;

        if (n_elements == 0 && nnz == 0) {
            return result;
        }

        // First pass: compute tiles_per_gauss
        launch_intersect_tile_kernel(
            means2d, radii, depths,
            nullptr, nullptr, // camera_ids, gaussian_ids (dense)
            C, N, nnz, packed,
            tile_size, tile_width, tile_height,
            nullptr, // cum_tiles_per_gauss
            tiles_per_gauss_out,
            nullptr, nullptr, // isect_ids, flatten_ids
            stream);

        // Compute cumsum on CPU for simplicity
        // For performance, this should be done on GPU with CUB
        int32_t* h_tiles_per_gauss = new int32_t[n_elements];
        cudaMemcpyAsync(h_tiles_per_gauss, tiles_per_gauss_out,
                        n_elements * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        int64_t* h_cum_tiles = new int64_t[n_elements];
        int64_t cumsum = 0;
        for (uint32_t i = 0; i < n_elements; ++i) {
            cumsum += h_tiles_per_gauss[i];
            h_cum_tiles[i] = cumsum;
        }
        int64_t n_isects = cumsum;
        result.n_isects = static_cast<int32_t>(n_isects);

        if (n_isects == 0) {
            delete[] h_tiles_per_gauss;
            delete[] h_cum_tiles;
            return result;
        }

        int64_t* d_cum_tiles;
        cudaMalloc(&d_cum_tiles, n_elements * sizeof(int64_t));
        // Sync copy required - h_cum_tiles deleted after return
        cudaMemcpy(d_cum_tiles, h_cum_tiles, n_elements * sizeof(int64_t), cudaMemcpyHostToDevice);

        // Allocate outputs
        cudaMalloc(&result.isect_ids, n_isects * sizeof(int64_t));
        cudaMalloc(&result.flatten_ids, n_isects * sizeof(int32_t));

        // Second pass: compute isect_ids and flatten_ids
        launch_intersect_tile_kernel(
            means2d, radii, depths,
            nullptr, nullptr, // camera_ids, gaussian_ids (dense)
            C, N, nnz, packed,
            tile_size, tile_width, tile_height,
            d_cum_tiles,
            nullptr, // tiles_per_gauss (not needed in second pass)
            result.isect_ids, result.flatten_ids,
            stream);

        // Sort by isect_ids if requested
        if (sort && n_isects > 0) {
            int64_t* isect_ids_sorted;
            int32_t* flatten_ids_sorted;
            cudaMalloc(&isect_ids_sorted, n_isects * sizeof(int64_t));
            cudaMalloc(&flatten_ids_sorted, n_isects * sizeof(int32_t));

            radix_sort_double_buffer(
                n_isects, tile_n_bits, cam_n_bits,
                result.isect_ids, result.flatten_ids,
                isect_ids_sorted, flatten_ids_sorted,
                stream);

            // Swap sorted buffers (radix sort may swap internally)
            cudaFree(result.isect_ids);
            cudaFree(result.flatten_ids);
            result.isect_ids = isect_ids_sorted;
            result.flatten_ids = flatten_ids_sorted;
        }

        cudaFree(d_cum_tiles);
        delete[] h_tiles_per_gauss;
        delete[] h_cum_tiles;

        return result;
    }

    void intersect_offset(
        const int64_t* isect_ids,
        int32_t n_isects,
        uint32_t C,
        uint32_t tile_width,
        uint32_t tile_height,
        int32_t* isect_offsets,
        cudaStream_t stream) {
        if (n_isects == 0) {
            cudaMemsetAsync(isect_offsets, 0,
                            C * tile_height * tile_width * sizeof(int32_t), stream);
            return;
        }

        launch_intersect_offset_kernel(
            isect_ids, n_isects,
            C, tile_width, tile_height,
            isect_offsets, stream);
    }

} // namespace gsplat_fwd
