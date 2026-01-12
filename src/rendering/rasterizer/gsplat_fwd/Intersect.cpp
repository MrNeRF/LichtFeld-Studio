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

    namespace {
        struct IntersectBufferCache {
            int64_t* cum_tiles = nullptr;
            int64_t* isect_ids_sort = nullptr;
            int32_t* flatten_ids_sort = nullptr;
            size_t cum_tiles_capacity = 0;
            size_t sort_capacity = 0;

            void ensure_cum_tiles(size_t n_elements) {
                if (n_elements > cum_tiles_capacity) {
                    if (cum_tiles)
                        cudaFree(cum_tiles);
                    size_t new_cap = n_elements + n_elements / 4; // 25% headroom
                    cudaMalloc(&cum_tiles, new_cap * sizeof(int64_t));
                    cum_tiles_capacity = new_cap;
                }
            }

            void ensure_sort_buffers(size_t n_isects) {
                if (n_isects > sort_capacity) {
                    if (isect_ids_sort)
                        cudaFree(isect_ids_sort);
                    if (flatten_ids_sort)
                        cudaFree(flatten_ids_sort);
                    size_t new_cap = n_isects + n_isects / 4; // 25% headroom
                    cudaMalloc(&isect_ids_sort, new_cap * sizeof(int64_t));
                    cudaMalloc(&flatten_ids_sort, new_cap * sizeof(int32_t));
                    sort_capacity = new_cap;
                }
            }

            ~IntersectBufferCache() {
                if (cum_tiles)
                    cudaFree(cum_tiles);
                if (isect_ids_sort)
                    cudaFree(isect_ids_sort);
                if (flatten_ids_sort)
                    cudaFree(flatten_ids_sort);
            }
        };

        IntersectBufferCache& get_cache() {
            static thread_local IntersectBufferCache cache;
            return cache;
        }
    } // namespace

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

        // GPU-based inclusive scan using CUB (replaces slow CPU cumsum)
        auto& cache = get_cache();
        cache.ensure_cum_tiles(n_elements);
        int64_t* d_cum_tiles = cache.cum_tiles;

        // Compute cumulative sum on GPU with int32→int64 promotion
        compute_cumsum_gpu(tiles_per_gauss_out, d_cum_tiles, n_elements, stream);

        // Get total intersection count (single 8-byte copy instead of full array)
        int64_t n_isects;
        cudaMemcpyAsync(&n_isects, d_cum_tiles + n_elements - 1, sizeof(int64_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        result.n_isects = static_cast<int32_t>(n_isects);

        if (n_isects == 0) {
            return result;
        }

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
            cache.ensure_sort_buffers(n_isects);

            radix_sort_double_buffer(
                n_isects, tile_n_bits, cam_n_bits,
                result.isect_ids, result.flatten_ids,
                cache.isect_ids_sort, cache.flatten_ids_sort,
                stream);

            // Copy sorted results back (sort may have used either buffer)
            cudaMemcpyAsync(result.isect_ids, cache.isect_ids_sort,
                            n_isects * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(result.flatten_ids, cache.flatten_ids_sort,
                            n_isects * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        }

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
