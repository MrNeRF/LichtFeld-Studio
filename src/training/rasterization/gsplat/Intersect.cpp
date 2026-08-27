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
            int64_t* h_n_isects_pinned = nullptr;
            bool h_n_isects_is_pinned = false;
            cudaEvent_t n_isects_ready_event = nullptr;
            size_t cub_sort_ws_bytes = 0;
            int64_t cub_sort_n = 0;
            uint32_t cub_sort_end_bit = 0;
            size_t pending_isect_capacity = 0;

            IntersectBufferCache() {
#if CUDART_VERSION >= 11020
                void* ptr = nullptr;
                if (cudaMallocHost(&ptr, sizeof(int64_t)) == cudaSuccess) {
                    h_n_isects_pinned = static_cast<int64_t*>(ptr);
                    h_n_isects_is_pinned = true;
                    *h_n_isects_pinned = 0;
                }
#endif
                if (!h_n_isects_pinned) {
                    h_n_isects_pinned = new int64_t(0);
                    h_n_isects_is_pinned = false;
                }
                if (cudaEventCreateWithFlags(&n_isects_ready_event, cudaEventDisableTiming) !=
                    cudaSuccess) {
                    n_isects_ready_event = nullptr;
                }
            }

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

            static size_t grow_cap(size_t current, size_t needed) {
                if (needed <= current) {
                    return current;
                }
                size_t grown = current * 2;
                if (grown < needed * 2) {
                    grown = needed * 2;
                }
                if (grown < needed) {
                    grown = needed;
                }
                return grown;
            }

            void ensure_isect_buffers(size_t n_isects, cudaStream_t stream) {
                if (n_isects == 0) {
                    return;
                }
                if (n_isects <= isect_capacity) {
                    isect_pair.bind_stream(stream);
                    return;
                }
                const size_t new_cap = grow_cap(isect_capacity, n_isects);
                DirectDeviceBuffer replacement(
                    pair_bytes(new_cap), stream,
                    "rasterizer.gsplat.intersection_id_value_pair");
                isect_pair = std::move(replacement);
                isect_capacity = new_cap;
                cub_sort_ws_bytes = 0;
                cub_sort_n = 0;
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
                    const size_t new_cap = grow_cap(sort_capacity, n_isects);
                    DirectDeviceBuffer replacement(
                        pair_bytes(new_cap), stream,
                        "rasterizer.gsplat.sorted_intersection_id_value_pair");
                    sort_pair = std::move(replacement);
                    sort_capacity = new_cap;
                    cub_sort_ws_bytes = 0;
                    cub_sort_n = 0;
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
                cub_sort_ws_bytes = 0;
                cub_sort_n = 0;
                cub_sort_end_bit = 0;
                pending_isect_capacity = 0;
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
                // Pinned n_isects slot + event stay for the TLS lifetime so a
                // mid-process release does not break a later forward on this thread.
                const bool cub_released = release_gsplat_cub_workspace();
                const bool color_grad_released = release_gsplat_color_grad_workspace();
                return !cum_tiles && !isect_pair && !sort_pair &&
                       sort_reuse_event == nullptr && cub_released &&
                       color_grad_released;
            }

            ~IntersectBufferCache() {
                release();
                cudaEvent_t count_event = std::exchange(n_isects_ready_event, nullptr);
                if (count_event) {
                    (void)cudaEventDestroy(count_event);
                }
                if (h_n_isects_pinned) {
                    if (h_n_isects_is_pinned) {
                        (void)cudaFreeHost(h_n_isects_pinned);
                    } else {
                        delete h_n_isects_pinned;
                    }
                    h_n_isects_pinned = nullptr;
                }
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
        cudaStream_t stream,
        int32_t* isect_offsets) {
        bool packed = (camera_ids != nullptr && gaussian_ids != nullptr);
        const uint64_t dense_elements = static_cast<uint64_t>(C) * static_cast<uint64_t>(N);
        LFS_ASSERT_MSG(
            packed || dense_elements <= std::numeric_limits<uint32_t>::max(),
            "gsplat dense intersection input exceeds uint32 range");
        uint32_t n_elements = packed ? 0 : static_cast<uint32_t>(dense_elements);
        uint32_t nnz = 0;

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
        result.n_sort = 0;

        if (n_elements == 0 && nnz == 0) {
            return result;
        }

        launch_intersect_tile_kernel(
            means2d, radii, depths,
            nullptr, nullptr,
            C, N, nnz, packed,
            tile_size, tile_width, tile_height,
            nullptr,
            tiles_per_gauss_out,
            nullptr, nullptr,
            stream);

        auto& cache = get_cache();
        cache.ensure_cum_tiles(n_elements, stream);
        int64_t* d_cum_tiles = cache.cum_tiles.as<int64_t>();
        compute_cumsum_gpu(tiles_per_gauss_out, d_cum_tiles, n_elements, stream);

        LFS_ASSERT_MSG(cache.h_n_isects_pinned != nullptr,
                       "gsplat intersection cache missing pinned n_isects slot");

        const int64_t sentinel_key =
            (static_cast<int64_t>(C - 1) << (32 + tile_n_bits)) |
            (static_cast<int64_t>(n_tiles) << 32);

        auto drain_count = [&]() {
            if (cache.n_isects_ready_event) {
                LFS_CUDA_CHECK_MSG(
                    cudaEventSynchronize(cache.n_isects_ready_event),
                    "gsplat n_isects event wait");
            } else {
                LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(stream),
                                   "gsplat intersection-count stream sync");
            }
            const int64_t n = *cache.h_n_isects_pinned;
            LFS_ASSERT_MSG(
                n >= 0 && n <= std::numeric_limits<int32_t>::max(),
                std::format("gsplat intersection count {} exceeds int32 range", n));
            return n;
        };

        auto schedule_proactive_grow = [&](const int64_t n) {
            if (n > 0 && n * 5 > static_cast<int64_t>(cache.isect_capacity) * 4) {
                const size_t want = static_cast<size_t>(n) * 2;
                if (want > cache.isect_capacity) {
                    cache.pending_isect_capacity = want;
                }
            }
        };

        LFS_CUDA_CHECK_MSG(
            cudaMemcpyAsync(cache.h_n_isects_pinned, d_cum_tiles + n_elements - 1,
                            sizeof(int64_t), cudaMemcpyDeviceToHost, stream),
            "gsplat intersection-count readback");
        if (cache.n_isects_ready_event) {
            LFS_CUDA_CHECK_MSG(
                cudaEventRecord(cache.n_isects_ready_event, stream),
                "gsplat n_isects event record");
        }

        if (cache.pending_isect_capacity > cache.isect_capacity) {
            cache.ensure_isect_buffers(cache.pending_isect_capacity, stream);
        }
        cache.pending_isect_capacity = 0;

        auto fill_and_sort_capacity = [&](const size_t cap) {
            cache.ensure_isect_buffers(cap, stream);
            cache.ensure_sort_buffers(cache.isect_capacity, stream);
            const size_t sort_n = cache.isect_capacity;
            launch_fill_isect_sentinels_kernel(
                cache.isect_ids(), cache.flatten_ids(),
                static_cast<int64_t>(sort_n), sentinel_key, stream);
            launch_intersect_tile_kernel(
                means2d, radii, depths,
                nullptr, nullptr,
                C, N, nnz, packed,
                tile_size, tile_width, tile_height,
                d_cum_tiles,
                nullptr,
                cache.isect_ids(), cache.flatten_ids(),
                stream, static_cast<int64_t>(sort_n));

            int64_t* keys_out = cache.isect_ids();
            int32_t* vals_out = cache.flatten_ids();
            if (sort && sort_n > 0) {
                try {
                    radix_sort_double_buffer(
                        static_cast<int64_t>(sort_n), tile_n_bits, cam_n_bits,
                        cache.isect_ids(), cache.flatten_ids(),
                        cache.sorted_isect_ids(), cache.sorted_flatten_ids(),
                        &keys_out, &vals_out,
                        cache.cub_sort_ws_bytes, cache.cub_sort_n, cache.cub_sort_end_bit,
                        stream);
                } catch (...) {
                    cache.record_sort_use(stream);
                    throw;
                }
                cache.record_sort_use(stream);
            }
            result.isect_ids = keys_out;
            result.flatten_ids = vals_out;
            result.n_sort = static_cast<int32_t>(sort_n);
        };

        auto launch_offsets = [&]() {
            if (!isect_offsets) {
                return;
            }
            intersect_offset(
                result.isect_ids, result.n_sort,
                C, tile_width, tile_height,
                isect_offsets, stream);
        };

        if (cache.isect_capacity == 0) {
            const int64_t n_isects = drain_count();
            result.n_isects = static_cast<int32_t>(n_isects);
            if (n_isects == 0) {
                launch_offsets();
                return result;
            }
            fill_and_sort_capacity(static_cast<size_t>(n_isects));
            launch_offsets();
            schedule_proactive_grow(n_isects);
            return result;
        }

        fill_and_sort_capacity(cache.isect_capacity);
        launch_offsets();
        // Count event was recorded before fill; fill/sort/offsets are already
        // queued so this wait does not drain the GPU.
        const int64_t n_isects = drain_count();
        result.n_isects = static_cast<int32_t>(n_isects);
        if (n_isects > static_cast<int64_t>(result.n_sort)) {
            LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(stream),
                               "gsplat intersection overflow drain");
            fill_and_sort_capacity(static_cast<size_t>(n_isects));
            launch_offsets();
        } else {
            schedule_proactive_grow(n_isects);
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
        const uint32_t n_keys = n_isects < 0 ? 0u : static_cast<uint32_t>(n_isects);
        if (n_keys == 0) {
            LFS_CUDA_CHECK_MSG(
                cudaMemsetAsync(isect_offsets, 0,
                                (static_cast<size_t>(C) * tile_height * tile_width + 1u) *
                                    sizeof(int32_t),
                                stream),
                "gsplat empty intersection-offset output");
            return;
        }

        launch_intersect_offset_kernel(
            isect_ids, n_keys,
            C, tile_width, tile_height,
            isect_offsets, stream);
    }

} // namespace gsplat_lfs
