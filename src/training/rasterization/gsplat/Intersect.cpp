/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "Intersect.h"
#include "Common.h"
#include "Ops.h"
#include "core/cuda/vmm_device_buffer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <format>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>

namespace gsplat_lfs {

    namespace {
        struct IntersectBufferCache {
            struct RollingPeak {
                static constexpr size_t kBucketCalls = 256;
                static constexpr size_t kBucketCount = 2;

                std::array<size_t, kBucketCount> maxima{};
                std::array<size_t, kBucketCount> counts{};
                size_t active_bucket = 0;
                size_t samples = 0;

                void observe(const size_t value) {
                    if (counts[active_bucket] == kBucketCalls) {
                        active_bucket = (active_bucket + 1) % kBucketCount;
                        counts[active_bucket] = 0;
                        maxima[active_bucket] = 0;
                    }
                    maxima[active_bucket] = std::max(maxima[active_bucket], value);
                    ++counts[active_bucket];
                    ++samples;
                }

                [[nodiscard]] size_t peak() const noexcept {
                    return std::max(maxima[0], maxima[1]);
                }

                [[nodiscard]] bool has_samples() const noexcept {
                    return samples != 0;
                }
            };

            static constexpr size_t kTrimWindowCalls = 512;
            static constexpr size_t kTrimHysteresisBytes = 64u * 1024u * 1024u;

            lfs::core::VmmDeviceBuffer cum_tiles;
            // Separate regions keep both addresses stable as the count grows.
            std::optional<lfs::core::VmmDeviceBuffer> isect_ids_storage;
            std::optional<lfs::core::VmmDeviceBuffer> isect_values_storage;
            std::optional<lfs::core::VmmDeviceBuffer> sort_ids_storage;
            std::optional<lfs::core::VmmDeviceBuffer> sort_values_storage;
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
            RollingPeak cum_tiles_peak;
            RollingPeak isect_peak;
            size_t call_count = 0;
            bool trim_used_in_window = false;
            cudaStream_t bound_stream = nullptr;
            bool has_bound_stream = false;

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

            [[nodiscard]] size_t granularity_bytes() const noexcept {
                if (cum_tiles) {
                    return cum_tiles.granularity_bytes();
                }
                if (isect_ids_storage) {
                    return isect_ids_storage->granularity_bytes();
                }
                if (isect_values_storage) {
                    return isect_values_storage->granularity_bytes();
                }
                if (sort_ids_storage) {
                    return sort_ids_storage->granularity_bytes();
                }
                if (sort_values_storage) {
                    return sort_values_storage->granularity_bytes();
                }
                return lfs::core::VmmDeviceBuffer::kGranularityBytes;
            }

            size_t pair_reservation_bytes(const size_t element_bytes) const {
                constexpr size_t kMaxIntersectionCount =
                    static_cast<size_t>(std::numeric_limits<int32_t>::max());
                constexpr size_t kPairBytes =
                    kMaxIntersectionCount * (sizeof(int64_t) + sizeof(int32_t));

                size_t free_bytes = 0;
                size_t total_bytes = 0;
                LFS_CUDA_CHECK_MSG(cudaMemGetInfo(&free_bytes, &total_bytes),
                                   "gsplat VMM intersection reservation sizing");
                (void)free_bytes;
                const size_t granularity = granularity_bytes();
                const size_t pair_budget = std::max(
                    granularity, (std::min(total_bytes, kPairBytes) / granularity) * granularity);
                const size_t element_budget = pair_budget / (sizeof(int64_t) + sizeof(int32_t));
                return std::max(
                    granularity,
                    ((element_budget * element_bytes + granularity - 1) / granularity) *
                        granularity);
            }

            static lfs::core::VmmDeviceBuffer create_vmm_buffer(
                const size_t reservation_bytes, const char* label) {
                auto result = lfs::core::VmmDeviceBuffer::create(reservation_bytes, label);
                if (!result) {
                    throw lfs::Exception(std::move(result).error());
                }
                return std::move(*result);
            }

            static void commit_vmm_buffer(lfs::core::VmmDeviceBuffer& buffer,
                                          const size_t bytes) {
                auto result = buffer.commit(bytes);
                if (!result) {
                    throw lfs::Exception(std::move(result).error());
                }
            }

            static void decommit_vmm_buffer(lfs::core::VmmDeviceBuffer& buffer,
                                            const size_t bytes) {
                auto result = buffer.decommit_tail(bytes);
                if (!result) {
                    throw lfs::Exception(std::move(result).error());
                }
            }

            static bool exceeds_hysteresis(const lfs::core::VmmDeviceBuffer& buffer,
                                           const size_t required_bytes) {
                return buffer && buffer.committed_bytes() > required_bytes &&
                       buffer.committed_bytes() - required_bytes >= kTrimHysteresisBytes;
            }

            void update_capacities() {
                cum_tiles_capacity = cum_tiles
                                         ? cum_tiles.committed_bytes() / sizeof(int64_t)
                                         : 0;
                isect_capacity = isect_ids_storage && isect_values_storage
                                     ? std::min(
                                           isect_ids_storage->committed_bytes() / sizeof(int64_t),
                                           isect_values_storage->committed_bytes() / sizeof(int32_t))
                                     : 0;
                sort_capacity = sort_ids_storage && sort_values_storage
                                    ? std::min(
                                          sort_ids_storage->committed_bytes() / sizeof(int64_t),
                                          sort_values_storage->committed_bytes() / sizeof(int32_t))
                                    : 0;
            }

            void trim_to_recent_peak() {
                if (trim_used_in_window || !has_bound_stream) {
                    return;
                }

                const size_t cum_need = cum_tiles_peak.has_samples()
                                            ? checked_bytes(cum_tiles_peak.peak(), sizeof(int64_t),
                                                            "gsplat cumulative tile trim")
                                            : 0;
                const size_t isect_need = isect_peak.has_samples()
                                              ? checked_bytes(isect_peak.peak(), sizeof(int64_t),
                                                              "gsplat intersection trim")
                                              : 0;
                const size_t value_need = isect_peak.has_samples()
                                              ? checked_bytes(isect_peak.peak(), sizeof(int32_t),
                                                              "gsplat flatten-id trim")
                                              : 0;
                const bool should_trim =
                    exceeds_hysteresis(cum_tiles, cum_need) ||
                    (isect_ids_storage && exceeds_hysteresis(*isect_ids_storage, isect_need)) ||
                    (isect_values_storage &&
                     exceeds_hysteresis(*isect_values_storage, value_need)) ||
                    (sort_ids_storage && exceeds_hysteresis(*sort_ids_storage, isect_need)) ||
                    (sort_values_storage &&
                     exceeds_hysteresis(*sort_values_storage, value_need));
                if (!should_trim) {
                    return;
                }

                if (sort_reuse_event && sort_reuse_event_recorded) {
                    LFS_CUDA_CHECK_MSG(
                        cudaEventSynchronize(sort_reuse_event),
                        "gsplat sort-cache trim event wait");
                }
                if (n_isects_ready_event) {
                    LFS_CUDA_CHECK_MSG(
                        cudaEventSynchronize(n_isects_ready_event),
                        "gsplat intersection-count trim event wait");
                }
                LFS_CUDA_CHECK_MSG(
                    cudaStreamSynchronize(bound_stream),
                    "gsplat intersection-cache trim stream wait");

                if (cum_tiles) {
                    decommit_vmm_buffer(cum_tiles, cum_need);
                }
                if (isect_ids_storage) {
                    decommit_vmm_buffer(*isect_ids_storage, isect_need);
                }
                if (isect_values_storage) {
                    decommit_vmm_buffer(*isect_values_storage, value_need);
                }
                if (sort_ids_storage) {
                    decommit_vmm_buffer(*sort_ids_storage, isect_need);
                }
                if (sort_values_storage) {
                    decommit_vmm_buffer(*sort_values_storage, value_need);
                }
                update_capacities();
                cub_sort_ws_bytes = 0;
                cub_sort_n = 0;
                trim_used_in_window = true;
            }

            void begin_call(const size_t n_elements, cudaStream_t stream) {
                ++call_count;
                if (call_count > 1 && (call_count - 1) % kTrimWindowCalls == 0) {
                    trim_used_in_window = false;
                }
                cum_tiles_peak.observe(n_elements);
                if (has_bound_stream && bound_stream != stream) {
                    LFS_CUDA_CHECK_MSG(
                        cudaStreamSynchronize(bound_stream),
                        "gsplat intersection-cache stream handoff");
                }
                trim_to_recent_peak();
                bound_stream = stream;
                has_bound_stream = true;
            }

            void finish_call(const int64_t n_isects) {
                if (n_isects >= 0) {
                    isect_peak.observe(static_cast<size_t>(n_isects));
                }
            }

            int64_t* isect_ids() const {
                return isect_ids_storage ? isect_ids_storage->as<int64_t>() : nullptr;
            }

            int32_t* flatten_ids() const {
                return isect_values_storage ? isect_values_storage->as<int32_t>() : nullptr;
            }

            int64_t* sorted_isect_ids() const {
                return sort_ids_storage ? sort_ids_storage->as<int64_t>() : nullptr;
            }

            int32_t* sorted_flatten_ids() const {
                return sort_values_storage ? sort_values_storage->as<int32_t>() : nullptr;
            }

            void ensure_cum_tiles(size_t n_elements, cudaStream_t stream) {
                (void)stream;
                if (n_elements <= cum_tiles_capacity && cum_tiles) {
                    return;
                }
                if (n_elements > cum_tiles_capacity) {
                    if (!cum_tiles) {
                        cum_tiles = create_vmm_buffer(
                            pair_reservation_bytes(sizeof(int64_t)),
                            "rasterizer.gsplat.cumulative_tiles");
                    }
                    commit_vmm_buffer(
                        cum_tiles,
                        checked_bytes(n_elements, sizeof(int64_t),
                                      "gsplat cumulative tiles"));
                    cum_tiles_capacity = cum_tiles.committed_bytes() / sizeof(int64_t);
                }
            }

            void ensure_isect_buffers(size_t n_isects, cudaStream_t stream) {
                (void)stream;
                if (n_isects == 0) {
                    return;
                }
                if (n_isects <= isect_capacity) {
                    return;
                }
                if (!isect_ids_storage) {
                    isect_ids_storage = create_vmm_buffer(
                        pair_reservation_bytes(sizeof(int64_t)),
                        "rasterizer.gsplat.intersection_ids");
                }
                if (!isect_values_storage) {
                    isect_values_storage = create_vmm_buffer(
                        pair_reservation_bytes(sizeof(int32_t)),
                        "rasterizer.gsplat.intersection_flatten_ids");
                }
                commit_vmm_buffer(
                    *isect_ids_storage,
                    checked_bytes(n_isects, sizeof(int64_t), "gsplat intersection ids"));
                commit_vmm_buffer(
                    *isect_values_storage,
                    checked_bytes(n_isects, sizeof(int32_t),
                                  "gsplat intersection flatten ids"));
                isect_capacity = std::min(
                    isect_ids_storage->committed_bytes() / sizeof(int64_t),
                    isect_values_storage->committed_bytes() / sizeof(int32_t));
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
                    if (!sort_ids_storage) {
                        sort_ids_storage = create_vmm_buffer(
                            pair_reservation_bytes(sizeof(int64_t)),
                            "rasterizer.gsplat.sorted_intersection_ids");
                    }
                    if (!sort_values_storage) {
                        sort_values_storage = create_vmm_buffer(
                            pair_reservation_bytes(sizeof(int32_t)),
                            "rasterizer.gsplat.sorted_intersection_flatten_ids");
                    }
                    commit_vmm_buffer(
                        *sort_ids_storage,
                        checked_bytes(n_isects, sizeof(int64_t),
                                      "gsplat sorted intersection ids"));
                    commit_vmm_buffer(
                        *sort_values_storage,
                        checked_bytes(n_isects, sizeof(int32_t),
                                      "gsplat sorted intersection flatten ids"));
                    sort_capacity = std::min(
                        sort_ids_storage->committed_bytes() / sizeof(int64_t),
                        sort_values_storage->committed_bytes() / sizeof(int32_t));
                    cub_sort_ws_bytes = 0;
                    cub_sort_n = 0;
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
                cum_tiles.release();
                isect_ids_storage.reset();
                isect_values_storage.reset();
                sort_ids_storage.reset();
                sort_values_storage.reset();
                cum_tiles_capacity = 0;
                isect_capacity = 0;
                sort_capacity = 0;
                cum_tiles_peak = {};
                isect_peak = {};
                call_count = 0;
                trim_used_in_window = false;
                bound_stream = nullptr;
                has_bound_stream = false;
                cub_sort_ws_bytes = 0;
                cub_sort_n = 0;
                cub_sort_end_bit = 0;
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
                return !cum_tiles && !isect_ids_storage && !isect_values_storage &&
                       !sort_ids_storage && !sort_values_storage &&
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

        auto& cache = get_cache();
        cache.begin_call(n_elements, stream);

        launch_intersect_tile_kernel(
            means2d, radii, depths,
            nullptr, nullptr,
            C, N, nnz, packed,
            tile_size, tile_width, tile_height,
            nullptr,
            tiles_per_gauss_out,
            nullptr, nullptr,
            stream);

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

        LFS_CUDA_CHECK_MSG(
            cudaMemcpyAsync(cache.h_n_isects_pinned, d_cum_tiles + n_elements - 1,
                            sizeof(int64_t), cudaMemcpyDeviceToHost, stream),
            "gsplat intersection-count readback");
        if (cache.n_isects_ready_event) {
            LFS_CUDA_CHECK_MSG(
                cudaEventRecord(cache.n_isects_ready_event, stream),
                "gsplat n_isects event record");
        }

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
                cache.finish_call(n_isects);
                launch_offsets();
                return result;
            }
            fill_and_sort_capacity(static_cast<size_t>(n_isects));
            launch_offsets();
            cache.finish_call(n_isects);
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
        }

        cache.finish_call(n_isects);
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
