/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/tensor_fwd.hpp"

#include <optional>
namespace lfs::core {
    class MemoryInfo;
    class MemoryPressureCoordinator;

    LFS_CORE_API void gpu_device_barrier(GpuBackend backend);
    // Release cached allocations belonging to one already-live backend.
    LFS_CORE_API void gpu_trim_cached_memory(GpuBackend backend);
    LFS_CORE_API int gpu_device_count(GpuBackend backend);
    // Cold pressure services preserve each backend's native headroom and cache policy.
    LFS_CORE_API MemoryInfo gpu_pressure_memory_info(GpuBackend backend, bool free_query);
    LFS_CORE_API void register_gpu_pressure_clients(MemoryPressureCoordinator& coordinator);
    LFS_CORE_API void record_gpu_pressure_breadcrumb();
    // Known backing storage extent, including reserved tensor capacity. Excludes allocator
    // bookkeeping; borrowed storage and CPU views without extent metadata return nullopt.
    LFS_CORE_API std::optional<std::size_t> reserved_allocation_bytes(const Tensor& tensor);
    // Bytes a GPU allocation of `bytes` occupies on the backend's allocator.
    LFS_CORE_API std::size_t gpu_allocation_bytes(GpuBackend backend, std::size_t bytes);

} // namespace lfs::core
