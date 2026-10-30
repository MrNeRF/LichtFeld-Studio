/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/memory_domain.hpp"

namespace lfs::core {

    const char* to_string(const MemoryDomain domain) noexcept {
        switch (domain) {
        case MemoryDomain::CudaDevice: return "cuda-device";
        case MemoryDomain::CudaVmm: return "cuda-vmm";
        case MemoryDomain::VulkanDevice: return "vulkan-device";
        case MemoryDomain::MetalDevice: return "metal-device";
        case MemoryDomain::HipDevice: return "hip-device";
        case MemoryDomain::PinnedHost: return "pinned-host";
        case MemoryDomain::PageableHost: return "pageable-host";
        }
        return "unknown";
    }

    bool is_device_heap(const MemoryDomain domain) noexcept {
        return domain == MemoryDomain::CudaDevice ||
               domain == MemoryDomain::CudaVmm ||
               domain == MemoryDomain::VulkanDevice ||
               domain == MemoryDomain::MetalDevice ||
               domain == MemoryDomain::HipDevice;
    }

    MemoryDomain device_memory_domain(const GpuBackend backend) noexcept {
        constexpr std::array domains{MemoryDomain::CudaDevice, MemoryDomain::VulkanDevice,
                                     MemoryDomain::MetalDevice, MemoryDomain::HipDevice};
        static_assert(domains.size() == kGpuBackendCount);
        const auto index = static_cast<size_t>(backend);
        return index < domains.size() ? domains[index] : MemoryDomain::CudaDevice;
    }

} // namespace lfs::core
