/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/gpu_backend_fwd.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lfs::core {
    struct VulkanHardwareDevice {
        std::string name;
        std::string uuid;
        std::string tensor_unavailable_reason;
        std::string rendering_unavailable_reason;
        uint32_t index = 0;
        bool discrete = false;
        size_t total_memory_bytes = 0;
        bool shared_memory = false;
        uint32_t vendor_id = 0;
        bool software = false;

        // Display grouping only; each API keeps its own stable selector UUID.
        std::string physical_id;
    };

    // Discovery only: does not create or freeze a tensor device/context.
    LFS_CORE_API std::vector<VulkanHardwareDevice> enumerate_vulkan_hardware();
    LFS_CORE_API std::vector<VulkanHardwareDevice> enumerate_cuda_hardware();
#ifdef LFS_TENSOR_HIP
    struct HipHardwareDevice : VulkanHardwareDevice {
        std::string architecture, pci_bus_id;
        std::array<uint8_t,8> luid{};
        uint32_t device_node_mask=0;
    };
    LFS_CORE_API std::vector<HipHardwareDevice> enumerate_hip_hardware();
    // OS/driver sample of the selected physical adapter; -1 means unavailable.
    // Intended for the background performance sampler, not the render loop.
    LFS_CORE_API float hip_gpu_utilization();
#endif
    struct GpuDeviceInfo {
        std::string name;
        size_t total_memory_bytes = 0;
        bool supports_process_memory_budget = false;
        size_t process_memory_budget_bytes = 0;
        size_t process_memory_used_bytes = 0;
        std::array<std::uint8_t, 16> uuid{};
        int compute_capability_major = 0;
    };

    // Vulkan queries use the backend's current device.
    LFS_CORE_API std::optional<GpuDeviceInfo> gpu_backend_device_info(GpuBackend backend);
    // Does not change the calling thread's CUDA device.
    LFS_CORE_API std::optional<GpuDeviceInfo> gpu_backend_device_info(GpuBackend backend, int device_index);
} // namespace lfs::core
