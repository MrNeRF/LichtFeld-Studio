/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "helper_math.h"
#include "rasterization_config.h"
#include "utils.h"
#include <cstdint>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace fast_lfs::rasterization {

    using InstanceKey = std::uint64_t;
    // Tile index above the raw IEEE bits of the (positive) view depth.
    inline constexpr int kInstanceDepthBits = 32;

    enum FastGSForwardStatusFlags : unsigned int {
        kFastGSForwardStatusTileIndexOutOfRange = 1u << 0,
        kFastGSForwardStatusInstanceWriteMismatch = 1u << 1,
        kFastGSForwardStatusPrimitiveIndexOutOfRange = 1u << 2,
        kFastGSForwardStatusTileInstanceRangeOutOfRange = 1u << 3,
        /// instance count exceeded sort-buffer capacity (async path).
        kFastGSForwardStatusSortCapacityOverflow = 1u << 4,
    };

    struct FastGSForwardStatus {
        unsigned int flags;
        unsigned int source_index;
        unsigned int tile_index;
        unsigned int expected_count;
        unsigned int actual_count;
        unsigned int bounds_x;
        unsigned int bounds_y;
        unsigned int bounds_z;
        unsigned int bounds_w;
        std::uint64_t value;
    };

    __device__ __forceinline__ void report_fastgs_status(
        FastGSForwardStatus* __restrict__ status,
        const unsigned int flag,
        const uint source_index,
        const uint tile_index,
        const std::uint64_t value,
        const uint4 bounds,
        const uint expected_count,
        const uint actual_count) {
        if (!status)
            return;

        const unsigned int old_flags = atomicOr(&status->flags, flag);
        if (old_flags == 0) {
            status->source_index = source_index;
            status->tile_index = tile_index;
            status->expected_count = expected_count;
            status->actual_count = actual_count;
            status->bounds_x = bounds.x;
            status->bounds_y = bounds.y;
            status->bounds_z = bounds.z;
            status->bounds_w = bounds.w;
            status->value = value;
        }
    }

    inline std::string describe_fastgs_status_flags(unsigned int flags) {
        std::string result;
        const auto append = [&result](const char* name) {
            if (!result.empty()) {
                result += ", ";
            }
            result += name;
        };

        if ((flags & kFastGSForwardStatusTileIndexOutOfRange) != 0) {
            append("tile index out of range");
        }
        if ((flags & kFastGSForwardStatusInstanceWriteMismatch) != 0) {
            append("instance write mismatch");
        }
        if ((flags & kFastGSForwardStatusPrimitiveIndexOutOfRange) != 0) {
            append("primitive index out of range");
        }
        if ((flags & kFastGSForwardStatusTileInstanceRangeOutOfRange) != 0) {
            append("tile instance range out of range");
        }
        if ((flags & kFastGSForwardStatusSortCapacityOverflow) != 0) {
            append("sort capacity overflow");
        }
        if (result.empty()) {
            result = "unknown status flag " + std::to_string(flags);
        }
        return result;
    }

    inline std::string format_fastgs_forward_status(
        const FastGSForwardStatus& status,
        const char* phase,
        const uint64_t n_primitives,
        const uint64_t n_tiles) {
        if (status.flags == 0) {
            return {};
        }

        return "FastGS " + std::string(phase) +
               " detected invalid rasterization state: " +
               describe_fastgs_status_flags(status.flags) +
               " (flags=" + std::to_string(status.flags) +
               ", source_index=" + std::to_string(status.source_index) +
               ", tile_index=" + std::to_string(status.tile_index) +
               ", expected=" + std::to_string(status.expected_count) +
               ", actual=" + std::to_string(status.actual_count) +
               ", value=" + std::to_string(status.value) +
               ", bounds=[" + std::to_string(status.bounds_x) + "," +
               std::to_string(status.bounds_y) + "," +
               std::to_string(status.bounds_z) + "," +
               std::to_string(status.bounds_w) + "]" +
               ", n_primitives=" + std::to_string(n_primitives) +
               ", n_tiles=" + std::to_string(n_tiles) + ")";
    }

    inline bool try_read_fastgs_forward_status(
        const FastGSForwardStatus* device_status,
        FastGSForwardStatus& status) {
        if (!device_status) {
            return false;
        }
        const cudaError_t err = cudaMemcpy(&status, device_status, sizeof(status), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            lfs::core::ensure_cuda_success(
                err, "cudaMemcpy(FastGS forward status)", {},
                LFS_SOURCE_SITE_CURRENT(),
                lfs::core::CudaFailureDisposition::LogOnly);
            cudaGetLastError();
            return false;
        }
        return true;
    }

    inline void throw_if_fastgs_forward_status(
        const FastGSForwardStatus* device_status,
        const char* phase,
        const uint64_t n_primitives,
        const uint64_t n_tiles) {
        FastGSForwardStatus status{};
        if (!try_read_fastgs_forward_status(device_status, status)) {
            return;
        }
        const std::string message = format_fastgs_forward_status(status, phase, n_primitives, n_tiles);
        if (!message.empty()) {
            throw std::runtime_error(message);
        }
    }

    inline void check_cuda_with_fastgs_status(
        const cudaError_t err,
        const char* name,
        const FastGSForwardStatus* device_status,
        const char* phase,
        const uint64_t n_primitives,
        const uint64_t n_tiles) {
        if (err == cudaSuccess) {
            return;
        }

        std::string message;
        FastGSForwardStatus status{};
        if (try_read_fastgs_forward_status(device_status, status)) {
            message = format_fastgs_forward_status(status, phase, n_primitives, n_tiles);
        }
        if (message.empty()) {
            message = lfs::core::detail::format_cuda_safe(
                "FastGS phase={} (n_primitives={}, n_tiles={})",
                phase ? phase : "<unknown>", n_primitives, n_tiles);
        }

        LFS_ENSURE_CUDA_SUCCESS_MSG(err, name, message);
    }

    inline void sync_fastgs_phase_if_requested(
        const char* name,
        const FastGSForwardStatus* device_status,
        const char* phase,
        const uint64_t n_primitives,
        const uint64_t n_tiles) {
        if (!lfs::core::cuda_sync_debug_enabled()) {
            return;
        }
        check_cuda_with_fastgs_status(
            cudaDeviceSynchronize(), name, device_status, phase, n_primitives, n_tiles);
        throw_if_fastgs_forward_status(device_status, phase, n_primitives, n_tiles);
    }

    inline int extract_end_bit(uint n) {
        int leading_zeros = 0;
        if ((n & 0xffff0000u) == 0) {
            leading_zeros += 16;
            n <<= 16;
        }
        if ((n & 0xff000000u) == 0) {
            leading_zeros += 8;
            n <<= 8;
        }
        if ((n & 0xf0000000u) == 0) {
            leading_zeros += 4;
            n <<= 4;
        }
        if ((n & 0xc0000000u) == 0) {
            leading_zeros += 2;
            n <<= 2;
        }
        if ((n & 0x80000000u) == 0) {
            leading_zeros += 1;
        }
        return 32 - leading_zeros;
    }

    inline int packed_instance_key_end_bit(uint n_tiles) {
        const int tile_bits = n_tiles <= 1 ? 0 : extract_end_bit(n_tiles - 1);
        return tile_bits + kInstanceDepthBits;
    }

    struct mat3x3 {
        float m11, m12, m13;
        float m21, m22, m23;
        float m31, m32, m33;
    };

    struct __align__(8) mat3x3_triu {
        float m11, m12, m13, m22, m23, m33;
    };

    template <typename T>
    static void obtain(char*& blob, T*& ptr, std::size_t count, std::size_t alignment) {
        std::size_t offset = reinterpret_cast<std::uintptr_t>(blob) + alignment - 1 & ~(alignment - 1);
        ptr = reinterpret_cast<T*>(offset);
        blob = reinterpret_cast<char*>(ptr + count);
    }

    template <typename T, typename... Args>
    size_t required(size_t P, Args... args) {
        char* size = nullptr;
        T::from_blob(size, P, args...);
        return ((size_t)size) + 128;
    }

    // A FastGS frame has one persistent prefix and one phase-local slice.  The
    // latter is allocated as one arena block and suballocated here so each phase
    // rewinds over the previous one once its kernels have been enqueued: the
    // forward sort scratch, then the loss scratch between forward and backward,
    // then the backward helpers.  The retained prefix is the sorted primitive
    // list and its contributor storage, which backward still reads.
    class FastGSPhaseArena {
    public:
        enum class Phase : unsigned char { Forward = 1,
                                           Loss = 2,
                                           Backward = 3 };

        FastGSPhaseArena(std::function<char*(size_t)> backing, cudaStream_t stream)
            : backing_(std::move(backing)), stream_(stream), poison_(std::getenv("LFS_FASTGS_PHASE_POISON") != nullptr) {}

        void begin_phase(Phase phase, size_t minimum_bytes) {
            LFS_ASSERT_MSG(phase != Phase::Forward || !started_,
                           "FastGS forward phase may only be begun once");
            if (started_ && minimum_bytes > std::numeric_limits<size_t>::max() - retained_bytes_)
                throw std::overflow_error("FastGS phase slice size overflow");
            const size_t requested = align_size(
                started_ ? retained_bytes_ + minimum_bytes : minimum_bytes);
            if (!started_) {
                base_ = requested > 0 ? backing_(requested) : nullptr;
                if (requested > 0 && !base_)
                    throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS phase slice");
                capacity_ = requested;
                retained_bytes_ = 0;
                started_ = true;
            } else {
                if (requested > capacity_) {
                    char* extension = backing_(requested - capacity_);
                    if (!extension)
                        throw std::runtime_error("OUT_OF_MEMORY: FastGS phase slice extension was not contiguous");
                    if (base_ == nullptr) {
                        base_ = extension;
                    } else if (extension != base_ + capacity_) {
                        throw std::runtime_error("OUT_OF_MEMORY: FastGS phase slice extension was not contiguous");
                    }
                    capacity_ = requested;
                }
                if (poison_ && capacity_ > retained_bytes_) {
                    // 0xff is a quiet NaN for IEEE float/float2/float3/float4
                    // storage and is cheap to enqueue on the frame stream.
                    LFS_CUDA_CHECK_MSG(
                        cudaMemsetAsync(base_ + retained_bytes_, 0xff,
                                        capacity_ - retained_bytes_, stream_),
                        "cudaMemsetAsync(FastGS phase poison)");
                }
            }
            phase_ = phase;
            cursor_ = retained_bytes_;
        }

        std::function<char*(size_t)> allocator(Phase expected_phase) {
            return [this, expected_phase](size_t size) {
                return allocate(size, expected_phase);
            };
        }

        char* retain_prefix(const void* source, size_t size) {
            LFS_DEBUG_ASSERT_MSG(started_ && phase_ == Phase::Forward,
                                 "FastGS retained prefix requires the forward phase");
            const size_t aligned = align_allocation_size(size);
            LFS_ASSERT_MSG(aligned <= cursor_, "FastGS retained prefix exceeds the phase high-water");
            if (size > 0 && source != base_) {
                const auto source_address = reinterpret_cast<std::uintptr_t>(source);
                const auto destination_address = reinterpret_cast<std::uintptr_t>(base_);
                LFS_ASSERT_MSG(source_address >= destination_address &&
                                   source_address - destination_address >= size,
                               "FastGS retained prefix overlaps its source");
                LFS_CUDA_CHECK_MSG(
                    cudaMemcpyAsync(base_, source, size, cudaMemcpyDeviceToDevice, stream_),
                    "cudaMemcpyAsync(FastGS retained sorted indices)");
            }
            retained_bytes_ = aligned;
            cursor_ = aligned;
            return base_;
        }

    private:
        static size_t align_size(size_t size) {
            constexpr size_t alignment = 256;
            if (size > std::numeric_limits<size_t>::max() - alignment + 1)
                throw std::overflow_error("FastGS phase slice size overflow");
            return (size + alignment - 1) & ~(alignment - 1);
        }

        char* allocate(size_t size, Phase expected_phase) {
            LFS_DEBUG_ASSERT_MSG(started_ && phase_ == expected_phase,
                                 "FastGS phased allocation obtained outside its active phase");
            if (size == 0)
                return nullptr;
            const size_t aligned = align_allocation_size(size);
            if (cursor_ > capacity_ || aligned > capacity_ - cursor_) {
                if (cursor_ > std::numeric_limits<size_t>::max() - aligned)
                    throw std::overflow_error("FastGS phase allocation size overflow");
                const size_t required_capacity = align_size(cursor_ + aligned);
                const size_t extension_bytes = required_capacity > capacity_
                                                   ? required_capacity - capacity_
                                                   : 0;
                if (extension_bytes == 0)
                    return nullptr;
                char* extension = backing_(extension_bytes);
                if (!extension)
                    return nullptr;
                if (base_ == nullptr) {
                    base_ = extension;
                } else if (extension != base_ + capacity_) {
                    throw std::runtime_error("OUT_OF_MEMORY: FastGS phase allocation extension was not contiguous");
                }
                capacity_ = required_capacity;
            }
            char* result = base_ + cursor_;
            cursor_ += aligned;
            return result;
        }

        static size_t align_allocation_size(size_t size) {
            // The phase slice also contains CUB workspaces whose contract is
            // 256-byte alignment.  Every suballocation must therefore keep
            // the cursor on that boundary; aligning only the arena growth
            // and the workspace's relative offset is insufficient when an
            // earlier phase buffer has an odd 128-byte-sized footprint.
            constexpr size_t alignment = 256;
            if (size > std::numeric_limits<size_t>::max() - alignment + 1)
                throw std::overflow_error("FastGS phase allocation size overflow");
            return (size + alignment - 1) & ~(alignment - 1);
        }

        std::function<char*(size_t)> backing_;
        cudaStream_t stream_ = nullptr;
        char* base_ = nullptr;
        size_t capacity_ = 0;
        size_t cursor_ = 0;
        size_t retained_bytes_ = 0;
        Phase phase_ = Phase::Forward;
        bool started_ = false;
        bool poison_ = false;
    };

    /// 128-bit packed screen position + pixel-space AABB for warp sub-tile culling.
    /// Layout: float2 mean2d (8B) + ushort4 pixel_bbox (8B) = 16B, one 128-bit load.
    /// pixel_bbox = (x_min, x_max_excl, y_min, y_max_excl) in absolute pixel coords.
    /// Tile-space screen_bounds remain separate for create_instances.
    struct alignas(16) PackedMeanBBox {
        float2 mean2d;
        ushort4 pixel_bbox;
    };
    static_assert(sizeof(PackedMeanBBox) == 16, "PackedMeanBBox must be 128-bit");

    // Projected attributes consumed together by blend and scoring kernels.
    // Each field is one aligned 128-bit load; optional depth and normals stay separate.
    struct alignas(16) ProjectedRecord {
        PackedMeanBBox geometry;
        float4 conic_opacity;
        float4 color;
    };
    static_assert(sizeof(ProjectedRecord) == 48);
    static_assert(alignof(ProjectedRecord) == 16);

    // Single-pass decoupled-lookback header and packed 64-bit per-block visibility scan state.
    // Packed tile word: bits [63:62] = status (0=invalid, 1=aggregate, 2=inclusive), bits [31:0] = vis count.
    struct alignas(16) ScanHeader {
        std::uint32_t next_block;
        std::uint32_t n_visible;
        std::uint32_t n_active_blocks;
        std::uint32_t pad;
        std::uint64_t n_instances;
        std::uint64_t pad_tail;
    };
    static_assert(sizeof(ScanHeader) == 32);

    struct PerPrimitiveBuffers {
        ScanHeader* scan_header;
        std::uint64_t* scan_tiles;
        uint* depth_keys;
        float* depths;
        // Per-visible tile counts and their inclusive scan. The instance total is
        // checked against INT_MAX before the scan, so 32 bits cannot overflow.
        uint* n_touched_tiles;
        uint* offset;
        ushort4* screen_bounds;
        ProjectedRecord* projected;
        FastGSForwardStatus* forward_status;

        static PerPrimitiveBuffers from_persistent_blob(char*& blob, int visible_cap) {
            PerPrimitiveBuffers buffers{};
            obtain(blob, buffers.depths, visible_cap, 128);
            obtain(blob, buffers.projected, visible_cap, 128);
            obtain(blob, buffers.forward_status, 1, 128);
            return buffers;
        }

        static PerPrimitiveBuffers from_phase_allocator_single_pass(
            const std::function<char*(size_t)>& allocator, int n_primitives, int visible_cap) {
            PerPrimitiveBuffers buffers{};
            const size_t n_blocks =
                (static_cast<size_t>(n_primitives) + static_cast<size_t>(config::block_size_preprocess) - 1u) /
                static_cast<size_t>(config::block_size_preprocess);
            const size_t scan_bytes = sizeof(ScanHeader) + n_blocks * sizeof(std::uint64_t);
            char* scan_mem = allocator(scan_bytes);
            buffers.scan_header = reinterpret_cast<ScanHeader*>(scan_mem);
            buffers.scan_tiles = reinterpret_cast<std::uint64_t*>(scan_mem + sizeof(ScanHeader));
            buffers.depth_keys = reinterpret_cast<uint*>(allocator(static_cast<size_t>(visible_cap) * sizeof(uint)));
            buffers.n_touched_tiles = reinterpret_cast<uint*>(
                allocator(static_cast<size_t>(visible_cap) * sizeof(uint)));
            buffers.offset = reinterpret_cast<uint*>(
                allocator(static_cast<size_t>(visible_cap) * sizeof(uint)));
            buffers.screen_bounds = reinterpret_cast<ushort4*>(
                allocator(static_cast<size_t>(visible_cap) * sizeof(ushort4)));
            return buffers;
        }

        static size_t required_persistent(int visible_cap) {
            char* size = nullptr;
            from_persistent_blob(size, visible_cap);
            return reinterpret_cast<size_t>(size) + 128;
        }

        static size_t required_phase_single_pass(int n_primitives, int visible_cap) {
            const auto align256 = [](size_t b) { return (b + 255u) & ~size_t{255u}; };
            const size_t n_blocks =
                (static_cast<size_t>(n_primitives) + static_cast<size_t>(config::block_size_preprocess) - 1u) /
                static_cast<size_t>(config::block_size_preprocess);
            return align256(sizeof(ScanHeader) + n_blocks * sizeof(std::uint64_t)) +
                   align256(static_cast<size_t>(visible_cap) * sizeof(uint)) +
                   align256(static_cast<size_t>(visible_cap) * sizeof(uint)) +
                   align256(static_cast<size_t>(visible_cap) * sizeof(uint)) +
                   align256(static_cast<size_t>(visible_cap) * sizeof(ushort4));
        }
    };

    struct PerTileBuffers {
        uint2* instance_ranges;
        uint* n_contributions;
        float* final_transmittance;
        uint* tile_order;
        uint* tile_counts_in;
        uint* tile_counts_out;
        uint* tile_ids_in;
        char* tile_sort_cub;
        size_t tile_sort_cub_bytes;

        static PerTileBuffers from_blob(char*& blob, int n_tiles) {
            PerTileBuffers buffers{};
            obtain(blob, buffers.instance_ranges, n_tiles, 128);
            obtain(blob, buffers.n_contributions,
                   static_cast<std::size_t>(n_tiles) * static_cast<std::size_t>(config::block_size_blend), 128);
            obtain(blob, buffers.final_transmittance,
                   static_cast<std::size_t>(n_tiles) * static_cast<std::size_t>(config::block_size_blend), 128);
            obtain(blob, buffers.tile_order, n_tiles, 128);
            obtain(blob, buffers.tile_counts_in, n_tiles, 128);
            obtain(blob, buffers.tile_counts_out, n_tiles, 128);
            obtain(blob, buffers.tile_ids_in, n_tiles, 128);
            if (n_tiles > 1) {
                LFS_CUDA_CHECK_MSG(
                    cub::DeviceRadixSort::SortPairsDescending(
                        nullptr, buffers.tile_sort_cub_bytes,
                        buffers.tile_counts_in, buffers.tile_counts_out,
                        buffers.tile_ids_in, buffers.tile_order,
                        n_tiles, 0, 32),
                    "FastGS tile LPT workspace query (n_tiles={})", n_tiles);
                obtain(blob, buffers.tile_sort_cub, buffers.tile_sort_cub_bytes, 128);
            }
            return buffers;
        }
    };

} // namespace fast_lfs::rasterization
