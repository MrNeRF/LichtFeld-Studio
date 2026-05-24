/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"
#include "rasterization_config.h"
#include "utils.h"
#include <cstdint>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fast_lfs::rasterization {

    using InstanceKey = std::uint32_t;

    enum FastGSForwardStatusFlags : unsigned int {
        kFastGSForwardStatusTileIndexOutOfRange = 1u << 0,
        kFastGSForwardStatusInstanceWriteMismatch = 1u << 1,
        kFastGSForwardStatusPrimitiveIndexOutOfRange = 1u << 2,
        kFastGSForwardStatusTileInstanceRangeOutOfRange = 1u << 3,
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

        std::string message = std::string("CUDA error in ") + name + " - " +
                              cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
        FastGSForwardStatus status{};
        if (try_read_fastgs_forward_status(device_status, status)) {
            const std::string status_message = format_fastgs_forward_status(status, phase, n_primitives, n_tiles);
            if (!status_message.empty()) {
                message += "; " + status_message;
            }
        }

        std::cerr << "\n[CUDA ERROR] " << message;
        throw std::runtime_error(message);
    }

    inline bool fastgs_diag_sync_enabled() {
        static const bool enabled = [] {
            const char* raw = std::getenv("LFS_FASTGS_DIAG_SYNC");
            return raw != nullptr &&
                   raw[0] != '\0' &&
                   raw[0] != '0' &&
                   raw[0] != 'f' &&
                   raw[0] != 'F' &&
                   raw[0] != 'n' &&
                   raw[0] != 'N';
        }();
        return enabled;
    }

    inline void sync_fastgs_phase_if_requested(
        const char* name,
        const FastGSForwardStatus* device_status,
        const char* phase,
        const uint64_t n_primitives,
        const uint64_t n_tiles) {
        if (!fastgs_diag_sync_enabled()) {
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

    inline int packed_instance_depth_bits(uint n_tiles) {
        const int tile_bits = n_tiles <= 1 ? 0 : extract_end_bit(n_tiles - 1);
        const int depth_bits = 32 - tile_bits;
        return depth_bits > 23 ? 23 : (depth_bits < 0 ? 0 : depth_bits);
    }

    inline int packed_instance_key_end_bit(uint n_tiles) {
        const int tile_bits = n_tiles <= 1 ? 0 : extract_end_bit(n_tiles - 1);
        return tile_bits + packed_instance_depth_bits(n_tiles);
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

    struct PerPrimitiveBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        uint* depth_keys;
        std::uint64_t* n_touched_tiles;
        std::uint64_t* offset;
        ushort4* screen_bounds;
        float2* mean2d;
        float4* conic_opacity;
        float3* color;
        FastGSForwardStatus* forward_status;

        static PerPrimitiveBuffers from_blob(char*& blob, int n_primitives) {
            PerPrimitiveBuffers buffers{};
            obtain(blob, buffers.depth_keys, n_primitives, 128);
            obtain(blob, buffers.n_touched_tiles, n_primitives, 128);
            obtain(blob, buffers.offset, n_primitives, 128);
            obtain(blob, buffers.screen_bounds, n_primitives, 128);
            obtain(blob, buffers.mean2d, n_primitives, 128);
            obtain(blob, buffers.conic_opacity, n_primitives, 128);
            obtain(blob, buffers.color, n_primitives, 128);
            obtain(blob, buffers.forward_status, 1, 128);
            const cudaError_t scan_err = cub::DeviceScan::InclusiveSum(
                nullptr, buffers.cub_workspace_size,
                buffers.n_touched_tiles, buffers.offset,
                n_primitives);
            if (scan_err != cudaSuccess) {
                int device_count = -1;
                int current_device = -1;
                const cudaError_t count_err = cudaGetDeviceCount(&device_count);
                const cudaError_t device_err = cudaGetDevice(&current_device);
                const std::string message =
                    std::string("CUDA error in cub::DeviceScan::InclusiveSum workspace query at ") +
                    __FILE__ + ":" + std::to_string(__LINE__) +
                    " (n_primitives=" + std::to_string(n_primitives) +
                    ", current_device=" + std::to_string(current_device) +
                    ", device_count=" + std::to_string(device_count) +
                    ", cudaGetDevice=" + cudaGetErrorName(device_err) +
                    ", cudaGetDeviceCount=" + cudaGetErrorName(count_err) +
                    ") - " + cudaGetErrorName(scan_err) + ": " + cudaGetErrorString(scan_err);
                std::cerr << "\n[CUDA ERROR] " << message;
                throw std::runtime_error(message);
            }
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            return buffers;
        }
    };

    struct PerTileBuffers {
        uint2* instance_ranges;
        uint* n_contributions;
        float* final_transmittance;

        static PerTileBuffers from_blob(char*& blob, int n_tiles) {
            PerTileBuffers buffers{};
            obtain(blob, buffers.instance_ranges, n_tiles, 128);
            obtain(blob, buffers.n_contributions,
                   static_cast<std::size_t>(n_tiles) * static_cast<std::size_t>(config::block_size_blend), 128);
            obtain(blob, buffers.final_transmittance,
                   static_cast<std::size_t>(n_tiles) * static_cast<std::size_t>(config::block_size_blend), 128);
            return buffers;
        }
    };

} // namespace fast_lfs::rasterization
