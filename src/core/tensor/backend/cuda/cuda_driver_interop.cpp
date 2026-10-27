/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../tensor_vulkan_interop.hpp"
#include "core/gpu_device_info.hpp"
#include "core/tensor_backend.hpp"
#include <algorithm>
#include <vulkan/vulkan.h>
namespace lfs::core::internal {
    std::shared_ptr<TensorVulkanInteropBackend> make_cuda_driver_vulkan_interop(VulkanInteropDevice device) {
        const auto cuda = gpu_backend_device_info(GpuBackend::CUDA, selected_cuda_device());
        VkPhysicalDeviceIDProperties identity{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext = &identity;
        vkGetPhysicalDeviceProperties2(static_cast<VkPhysicalDevice>(device.physical_device), &properties);
        const bool direct = device.external_memory && device.external_semaphore && cuda &&
                            std::equal(cuda->uuid.begin(), cuda->uuid.end(), identity.deviceUUID);
        return direct ? make_cuda_vulkan_interop(device) : make_host_vulkan_interop(device, GpuBackend::CUDA);
    }
} // namespace lfs::core::internal
