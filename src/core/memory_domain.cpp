/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/memory_pressure.hpp"

namespace lfs::core {

    const char* to_string(const MemoryDomain domain) noexcept {
        switch (domain) {
        case MemoryDomain::CudaDevice: return "cuda-device";
        case MemoryDomain::CudaVmm: return "cuda-vmm";
        case MemoryDomain::VulkanDevice: return "vulkan-device";
        case MemoryDomain::MetalDevice: return "metal-device";
        case MemoryDomain::PinnedHost: return "pinned-host";
        case MemoryDomain::PageableHost: return "pageable-host";
        }
        return "unknown";
    }

    bool is_device_heap(const MemoryDomain domain) noexcept {
        return domain == MemoryDomain::CudaDevice ||
               domain == MemoryDomain::CudaVmm ||
               domain == MemoryDomain::VulkanDevice ||
               domain == MemoryDomain::MetalDevice;
    }

    MemoryDomain device_memory_domain(const GpuBackend backend) noexcept {
        switch (backend) {
        case GpuBackend::CUDA: return MemoryDomain::CudaDevice;
        case GpuBackend::Vulkan: return MemoryDomain::VulkanDevice;
        case GpuBackend::Metal: return MemoryDomain::MetalDevice;
        }
        return MemoryDomain::CudaDevice;
    }

} // namespace lfs::core
