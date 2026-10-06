// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/export.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/source_site.hpp"
#include <cstddef>
#include <cstdint>
namespace lfs {
    class Error;
}
namespace lfs::core {
    // Physical backing of an allocation. CudaDevice, CudaVmm, VulkanDevice and
    // MetalDevice all draw from the same physical device heap and must never
    // be summed; the two host domains are separate system RAM.
    enum class MemoryDomain : uint8_t {
        CudaDevice,
        CudaVmm,
        VulkanDevice,
        MetalDevice,
        PinnedHost,
        PageableHost,
    };

    LFS_ERROR_API const char* to_string(MemoryDomain domain) noexcept;

    // The device-heap domain a tensor backend allocates from.
    LFS_ERROR_API MemoryDomain device_memory_domain(GpuBackend backend) noexcept;

    // True for domains backed by the physical device heap. A reclaim client for
    // any device-heap domain can relieve a failure in any other device-heap
    // domain; host domains are disjoint.
    LFS_ERROR_API bool is_device_heap(MemoryDomain domain) noexcept;

    // Minimal, allocation-free description of a native allocation failure. All
    // string members are static or interned literals; constructing this record
    // must not allocate so it is safe to build on the raw OOM path.
    struct AllocationFailure {
        MemoryDomain domain = MemoryDomain::CudaDevice;
        size_t requested_bytes = 0;
        size_t alignment = 0;
        int device = 0;
        uintptr_t stream = 0;            // cudaStream_t reinterpreted; 0 outside CUDA
        const char* label = nullptr;     // static/interned allocation label
        const char* operation = nullptr; // static/interned operation name
        long long native_error = 0;      // cudaError_t or VkResult
    };

    // Phase 1 error-architecture adapter: builds the typed lfs::Error for a
    // native allocation failure, preserving domain/bytes/device/stream/label
    // as SmallFields and the native status as NativeError. Declared here so
    // AllocationFailure and lfs::Error stay easy to bridge at every call
    // site; defined in core/error.cpp (which owns Error's representation)
    // rather than in this header, so this header never has to include
    // core/error.hpp.
    [[nodiscard]] LFS_ERROR_API Error to_error(const AllocationFailure& failure, SourceSite site);

} // namespace lfs::core
