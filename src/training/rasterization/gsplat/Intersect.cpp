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
#include <format>
#include <limits>
#include <string_view>
#include <utility>

namespace gsplat_lfs {

    namespace {
        struct IntersectBufferCache {
            DirectDeviceBuffer cum_tiles;
            // Each id/value pair is one exact block. This avoids paying the
            // allocator's quantization independently for the two arrays.
            DirectDeviceBuffer isect_pair;
            DirectDeviceBuffer sort_pair;
            size_t cum_tiles_capacity = 0;
            size_t isect_capacity = 0;
            size_t sort_capacity = 0;
            cudaEvent_t sort_reuse_event = nullptr;
            bool sort_reuse_event_recorded = false;

            static size_t pair_bytes(const size_t count) {
                return checked_bytes(
                    count, sizeof(int64_t) + sizeof(int32_t),
                    "gsplat intersection id/value pair");
            }

            int64_t* isect_ids() const {
                return isect_pair.as<int64_t>();
            }

            int32_t* flatten_ids() const {
                return reinterpret_cast<int32_t*>(
                    static_cast<std::byte*>(isect_pair.get()) +
                    isect_capacity * sizeof(int64_t));
            }

            int64_t* sorted_isect_ids() const {
                return sort_pair.as<int64_t>();
            }

            int32_t* sorted_flatten_ids() const {
                return reinterpret_cast<int32_t*>(
                    static_cast<std::byte*>(sort_pair.get()) +
                    sort_capacity * sizeof(int64_t));
            }

            void ensure_cum_tiles(size_t n_elements, cudaStream_t stream) {
                if (n_elements <= cum_tiles_capacity && cum_tiles) {
                    cum_tiles.bind_stream(stream);
                    return;
                }
                if (n_elements > cum_tiles_capacity) {
                    const size_t new_cap = n_elements;
                    DirectDeviceBuffer replacement(
                        checked_bytes(new_cap, sizeof(int64_t), "gsplat cumulative tiles"),
                        stream,
                        "rasterizer.gsplat.cumulative_tiles");

                    cum_tiles = std::move(replacement);
                    cum_tiles_capacity = new_cap;
                }
            }

            void ensure_isect_buffers(size_t n_isects, cudaStream_t stream) {
                if (n_isects == 0) {
                    return;
                }
                if (n_isects <= isect_capacity) {
                    isect_pair.bind_stream(stream);
                    return;
                }
                const size_t new_cap = n_isects;
                DirectDeviceBuffer replacement(
                    pair_bytes(new_cap), stream,
                    "rasterizer.gsplat.intersection_id_value_pair");
                isect_pair = std::move(replacement);
                isect_capacity = new_cap;
            }

            void ensure_sort_buffers(size_t n_isects, cudaStream_t stream) {
                if (!sort_reuse_event) {
                    LFS_CUDA_CHECK_MSG(
                        cudaEventCreateWithFlags(&sort_reuse_event, cudaEventDisableTiming),
                        "gsplat sort-cache event creation");
                }
                if (sort_reuse_event_recorded) {
                    LFS_CUDA_CHECK_MSG(
                        cudaStreamWaitEvent(stream, sort_reuse_event, 0),
                        "gsplat sort-cache stream handoff");
                }
                if (n_isects > sort_capacity) {
                    const size_t new_cap = n_isects;
                    DirectDeviceBuffer replacement(
                        pair_bytes(new_cap), stream,
                        "rasterizer.gsplat.sorted_intersection_id_value_pair");
                    sort_pair = std::move(replacement);
                    sort_capacity = new_cap;
                } else {
                    sort_pair.bind_stream(stream);
                }
            }

            void record_sort_use(cudaStream_t stream) {
                LFS_ASSERT(sort_reuse_event != nullptr);
                const cudaError_t status = cudaEventRecord(sort_reuse_event, stream);
                if (status != cudaSuccess) {
                    // Without an event, the only safe recovery is to drain the
                    // stream before allowing another caller to reuse the cache.
                    sort_reuse_event_recorded = false;
                    const cudaError_t sync_status = cudaStreamSynchronize(stream);
                    if (sync_status != cudaSuccess) {
                        lfs::core::ensure_cuda_success(
                            sync_status, "cudaStreamSynchronize(gsplat sort-cache fallback)", {},
                            LFS_SOURCE_SITE_CURRENT(),
                            lfs::core::CudaFailureDisposition::LogOnly);
                    }
                    LFS_ENSURE_CUDA_SUCCESS_MSG(
                        status, "cudaEventRecord(gsplat sort cache)",
                        "fallback=stream synchronization");
                }
                sort_reuse_event_recorded = true;
            }

            bool release() noexcept {
                cum_tiles.reset();
                isect_pair.reset();
                sort_pair.reset();
                cum_tiles_capacity = 0;
                isect_capacity = 0;
                sort_capacity = 0;
                cudaEvent_t event = std::exchange(sort_reuse_event, nullptr);
                sort_reuse_event_recorded = false;
                if (event) {
                    const cudaError_t status = cudaEventDestroy(event);
                    if (status != cudaSuccess) {
                        lfs::core::ensure_cuda_success(
                            status, "cudaEventDestroy(gsplat sort cache)", {},
                            LFS_SOURCE_SITE_CURRENT(),
                            lfs::core::CudaFailureDisposition::LogOnlyNoLatch);
                    }
                }
                // Drop pooled gsplat temporaries so thread exit returns VRAM.
                const bool cub_released = release_gsplat_cub_workspace();
                const bool color_grad_released = release_gsplat_color_grad_workspace();
                return !cum_tiles && !isect_pair && !sort_pair &&
                       sort_reuse_event == nullptr && cub_released &&
                       color_grad_released;
            }

            ~IntersectBufferCache() {
                release();
            }
        };

        IntersectBufferCache& get_cache() {
            static thread_local IntersectBufferCache cache;
            return cache;
        }
    } // namespace

    bool release_intersect_thread_local_cache() noexcept {
        return get_cache().release();
    }

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
        const uint64_t dense_elements = static_cast<uint64_t>(C) * static_cast<uint64_t>(N);
        LFS_ASSERT_MSG(
            packed || dense_elements <= std::numeric_limits<uint32_t>::max(),
            "gsplat dense intersection input exceeds uint32 range");
        uint32_t n_elements = packed ? 0 : static_cast<uint32_t>(dense_elements);
        uint32_t nnz = packed ? 0 : 0; // TODO: For packed mode

        const uint64_t tile_count = static_cast<uint64_t>(tile_width) * tile_height;
        LFS_ASSERT_MSG(tile_count > 0 && tile_count <= std::numeric_limits<uint32_t>::max(),
                       "gsplat tile count is zero or exceeds uint32 range");
        LFS_ASSERT_MSG(C > 0, "gsplat camera count must be nonzero");
        uint32_t n_tiles = static_cast<uint32_t>(tile_count);
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
        cache.ensure_cum_tiles(n_elements, stream);
        int64_t* d_cum_tiles = cache.cum_tiles.as<int64_t>();

        // Compute cumulative sum on GPU with int32→int64 promotion
        compute_cumsum_gpu(tiles_per_gauss_out, d_cum_tiles, n_elements, stream);

        // Get total intersection count (single 8-byte copy instead of full array)
        int64_t n_isects;
        LFS_CUDA_CHECK_MSG(
            cudaMemcpyAsync(&n_isects, d_cum_tiles + n_elements - 1, sizeof(int64_t),
                            cudaMemcpyDeviceToHost, stream),
            "gsplat intersection-count readback");
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(stream), "gsplat intersection-count stream sync");
        LFS_ASSERT_MSG(
            n_isects >= 0 && n_isects <= std::numeric_limits<int32_t>::max(),
            std::format("gsplat intersection count {} exceeds int32 range", n_isects));
        result.n_isects = static_cast<int32_t>(n_isects);

        if (n_isects == 0) {
            return result;
        }

        // Grow-only high-water buffers: ownership stays in the TLS cache.
        // Callers must NOT cudaFree the returned pointers (release is a no-op).
        cache.ensure_isect_buffers(static_cast<size_t>(n_isects), stream);
        int64_t* const isect_ids_ptr = cache.isect_ids();
        int32_t* const flatten_ids_ptr = cache.flatten_ids();

        // Second pass: compute isect_ids and flatten_ids
        launch_intersect_tile_kernel(
            means2d, radii, depths,
            nullptr, nullptr, // camera_ids, gaussian_ids (dense)
            C, N, nnz, packed,
            tile_size, tile_width, tile_height,
            d_cum_tiles,
            nullptr, // tiles_per_gauss (not needed in second pass)
            isect_ids_ptr, flatten_ids_ptr,
            stream);

        // Sort by isect_ids if requested
        if (sort && n_isects > 0) {
            cache.ensure_sort_buffers(static_cast<size_t>(n_isects), stream);

            try {
                radix_sort_double_buffer(
                    n_isects, tile_n_bits, cam_n_bits,
                    isect_ids_ptr, flatten_ids_ptr,
                    cache.sorted_isect_ids(), cache.sorted_flatten_ids(),
                    stream);

                // Copy sorted results back (sort may have used either buffer)
                LFS_CUDA_CHECK_MSG(
                    cudaMemcpyAsync(
                        isect_ids_ptr, cache.sorted_isect_ids(),
                        checked_bytes(static_cast<size_t>(n_isects), sizeof(int64_t),
                                      "gsplat sorted intersection output"),
                        cudaMemcpyDeviceToDevice, stream),
                    "gsplat sorted intersection output copy");
                LFS_CUDA_CHECK_MSG(
                    cudaMemcpyAsync(
                        flatten_ids_ptr, cache.sorted_flatten_ids(),
                        checked_bytes(static_cast<size_t>(n_isects), sizeof(int32_t),
                                      "gsplat sorted flatten output"),
                        cudaMemcpyDeviceToDevice, stream),
                    "gsplat sorted flatten output copy");
            } catch (...) {
                cache.record_sort_use(stream);
                throw;
            }
            cache.record_sort_use(stream);
        }

        result.isect_ids = isect_ids_ptr;
        result.flatten_ids = flatten_ids_ptr;

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
            LFS_CUDA_CHECK_MSG(
                cudaMemsetAsync(isect_offsets, 0,
                                C * tile_height * tile_width * sizeof(int32_t), stream),
                "gsplat empty intersection-offset output");
            return;
        }

        launch_intersect_offset_kernel(
            isect_ids, n_isects,
            C, tile_width, tile_height,
            isect_offsets, stream);
    }

} // namespace gsplat_lfs
