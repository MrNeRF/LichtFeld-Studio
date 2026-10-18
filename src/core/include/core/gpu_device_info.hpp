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
    };

    // Discovery only: does not create or freeze a tensor device/context.
    LFS_CORE_API std::vector<VulkanHardwareDevice> enumerate_vulkan_hardware();
    LFS_CORE_API std::vector<VulkanHardwareDevice> enumerate_cuda_hardware();
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
