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
#include <limits>
#include <string_view>

namespace gsplat_lfs {

    namespace {
        size_t growth_capacity(const size_t required, const std::string_view allocation) {
            const size_t headroom = required / 4;
            LFS_ASSERT_MSG(
                required <= std::numeric_limits<size_t>::max() - headroom,
                std::format("{} capacity overflow for {} elements", allocation, required));
            return required + headroom;
        }

        void free_device_memory(void* ptr) noexcept {
            if (!ptr) {
                return;
            }
            if (cudaFree(ptr) != cudaSuccess) {
                cudaGetLastError();
            }
        }

        class PendingDeviceAllocation {
        public:
            PendingDeviceAllocation(const size_t bytes, const std::string_view label) {
                checked_cuda_malloc(&ptr_, bytes, label);
            }

            ~PendingDeviceAllocation() {
                free_device_memory(ptr_);
            }

            PendingDeviceAllocation(const PendingDeviceAllocation&) = delete;
            PendingDeviceAllocation& operator=(const PendingDeviceAllocation&) = delete;

            template <typename T>
            T* as() const noexcept {
                return static_cast<T*>(ptr_);
            }

            void* get() const noexcept { return ptr_; }

            void* release() noexcept {
                void* ptr = ptr_;
                ptr_ = nullptr;
                return ptr;
            }

        private:
            void* ptr_ = nullptr;
        };

        struct IntersectBufferCache {
            int64_t* cum_tiles = nullptr;
            int64_t* isect_ids_sort = nullptr;
            int32_t* flatten_ids_sort = nullptr;
            size_t cum_tiles_capacity = 0;
            size_t sort_capacity = 0;
            cudaEvent_t sort_reuse_event = nullptr;
            bool sort_reuse_event_recorded = false;

            void ensure_cum_tiles(size_t n_elements) {
                if (n_elements > cum_tiles_capacity) {
                    const size_t new_cap = growth_capacity(n_elements, "gsplat cumulative tiles");
                    PendingDeviceAllocation replacement(
                        checked_bytes(new_cap, sizeof(int64_t), "gsplat cumulative tiles"),
                        "rasterizer.gsplat.cumulative_tiles");

                    int64_t* old = cum_tiles;
                    cum_tiles = replacement.as<int64_t>();
                    cum_tiles_capacity = new_cap;
                    (void)replacement.release();
                    free_device_memory(old);
                }
            }

            void ensure_sort_buffers(size_t n_isects, cudaStream_t stream) {
                if (!sort_reuse_event) {
                    check_cuda_status(
                        cudaEventCreateWithFlags(&sort_reuse_event, cudaEventDisableTiming),
                        "gsplat sort-cache event creation");
                }
                if (sort_reuse_event_recorded) {
                    check_cuda_status(
                        cudaStreamWaitEvent(stream, sort_reuse_event, 0),
                        "gsplat sort-cache stream handoff");
                }
                if (n_isects > sort_capacity) {
                    const size_t new_cap = growth_capacity(n_isects, "gsplat sort buffers");
                    PendingDeviceAllocation replacement_isect_ids(
                        checked_bytes(new_cap, sizeof(int64_t), "gsplat sorted intersection ids"),
                        "rasterizer.gsplat.sorted_intersection_ids");
                    PendingDeviceAllocation replacement_flatten_ids(
                        checked_bytes(new_cap, sizeof(int32_t), "gsplat sorted flatten ids"),
                        "rasterizer.gsplat.sorted_flatten_ids");

                    int64_t* old_isect_ids = isect_ids_sort;
                    int32_t* old_flatten_ids = flatten_ids_sort;
                    isect_ids_sort = replacement_isect_ids.as<int64_t>();
                    flatten_ids_sort = replacement_flatten_ids.as<int32_t>();
                    sort_capacity = new_cap;
                    (void)replacement_isect_ids.release();
                    (void)replacement_flatten_ids.release();
                    free_device_memory(old_isect_ids);
                    free_device_memory(old_flatten_ids);
                }
            }

            void record_sort_use(cudaStream_t stream) {
                LFS_ASSERT(sort_reuse_event != nullptr);
                const cudaError_t status = cudaEventRecord(sort_reuse_event, stream);
                if (status != cudaSuccess) {
                    // Without an event, the only safe recovery is to drain the
                    // stream before allowing another caller to reuse the cache.
                    sort_reuse_event_recorded = false;
                    (void)cudaStreamSynchronize(stream);
                    check_cuda_status(status, "gsplat sort-cache event record");
                }
                sort_reuse_event_recorded = true;
            }

            ~IntersectBufferCache() {
                free_device_memory(cum_tiles);
                free_device_memory(isect_ids_sort);
                free_device_memory(flatten_ids_sort);
                if (sort_reuse_event) {
                    cudaEventDestroy(sort_reuse_event);
                }
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
        check_cuda_status(cudaGetLastError(), "gsplat tile-count kernel launch");

        // GPU-based inclusive scan using CUB (replaces slow CPU cumsum)
        auto& cache = get_cache();
        cache.ensure_cum_tiles(n_elements);
        int64_t* d_cum_tiles = cache.cum_tiles;

        // Compute cumulative sum on GPU with int32→int64 promotion
        compute_cumsum_gpu(tiles_per_gauss_out, d_cum_tiles, n_elements, stream);

        // Get total intersection count (single 8-byte copy instead of full array)
        int64_t n_isects;
        check_cuda_status(
            cudaMemcpyAsync(&n_isects, d_cum_tiles + n_elements - 1, sizeof(int64_t),
                            cudaMemcpyDeviceToHost, stream),
            "gsplat intersection-count readback");
        check_cuda_status(cudaStreamSynchronize(stream), "gsplat intersection-count stream sync");
        LFS_ASSERT_MSG(
            n_isects >= 0 && n_isects <= std::numeric_limits<int32_t>::max(),
            std::format("gsplat intersection count {} exceeds int32 range", n_isects));
        result.n_isects = static_cast<int32_t>(n_isects);

        if (n_isects == 0) {
            return result;
        }

        PendingDeviceAllocation isect_ids(
            checked_bytes(static_cast<size_t>(n_isects), sizeof(int64_t),
                          "gsplat output intersection ids"),
            "rasterizer.gsplat.intersection_ids");
        PendingDeviceAllocation flatten_ids(
            checked_bytes(static_cast<size_t>(n_isects), sizeof(int32_t),
                          "gsplat output flatten ids"),
            "rasterizer.gsplat.flatten_ids");

        // Second pass: compute isect_ids and flatten_ids
        launch_intersect_tile_kernel(
            means2d, radii, depths,
            nullptr, nullptr, // camera_ids, gaussian_ids (dense)
            C, N, nnz, packed,
            tile_size, tile_width, tile_height,
            d_cum_tiles,
            nullptr, // tiles_per_gauss (not needed in second pass)
            isect_ids.as<int64_t>(), flatten_ids.as<int32_t>(),
            stream);
        check_cuda_status(cudaGetLastError(), "gsplat intersection kernel launch");

        // Sort by isect_ids if requested
        if (sort && n_isects > 0) {
            cache.ensure_sort_buffers(static_cast<size_t>(n_isects), stream);

            try {
                radix_sort_double_buffer(
                    n_isects, tile_n_bits, cam_n_bits,
                    isect_ids.as<int64_t>(), flatten_ids.as<int32_t>(),
                    cache.isect_ids_sort, cache.flatten_ids_sort,
                    stream);

                // Copy sorted results back (sort may have used either buffer)
                check_cuda_status(
                    cudaMemcpyAsync(
                        isect_ids.get(), cache.isect_ids_sort,
                        checked_bytes(static_cast<size_t>(n_isects), sizeof(int64_t),
                                      "gsplat sorted intersection output"),
                        cudaMemcpyDeviceToDevice, stream),
                    "gsplat sorted intersection output copy");
                check_cuda_status(
                    cudaMemcpyAsync(
                        flatten_ids.get(), cache.flatten_ids_sort,
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

        result.isect_ids = static_cast<int64_t*>(isect_ids.release());
        result.flatten_ids = static_cast<int32_t*>(flatten_ids.release());

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

} // namespace gsplat_lfs
