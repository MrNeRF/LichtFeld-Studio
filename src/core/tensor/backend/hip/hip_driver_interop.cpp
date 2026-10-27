/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../tensor_vulkan_interop.hpp"
namespace lfs::core::internal {
    std::shared_ptr<TensorVulkanInteropBackend> make_hip_vulkan_interop(VulkanInteropDevice device) {
        return make_host_vulkan_interop(device, GpuBackend::HIP);
    }
} // namespace lfs::core::internal
