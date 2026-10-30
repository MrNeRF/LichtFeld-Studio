/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "tensor_vulkan_interop.hpp"
namespace lfs::core::internal {
    std::shared_ptr<TensorVulkanInteropBackend> make_cuda_driver_vulkan_interop(VulkanInteropDevice) {
        throw TensorError("Vulkan tensor interop is unavailable in this build");
    }
    std::shared_ptr<TensorVulkanInteropBackend> make_metal_vulkan_interop(VulkanInteropDevice) {
        throw TensorError("Metal tensor interop is unavailable in this build");
    }
} // namespace lfs::core::internal
