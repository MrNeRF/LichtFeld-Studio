/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/vulkan_queue_sync.hpp"
#include "rendering/rasterizer/vulkan/src/gs_renderer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace lfs::test {

    // A Vulkan 1.3 compute device that meets the VkSplat renderer's contracts:
    // push descriptors, the renderer's push-constant block, synchronization2,
    // timeline semaphores and buffer device addresses. Every supported core
    // feature is enabled, as the shipped shaders use several. create() skips
    // the calling test when no device qualifies, so callers check IsSkipped()
    // and HasFatalFailure() after it. Destroy whatever uses the device first.
    // The queue is registered with the queue sync domain the renderer submits through.
    struct VkSplatTestDevice {
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physical = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkQueue queue = VK_NULL_HANDLE;
        uint32_t family = 0;
        VmaAllocator allocator = VK_NULL_HANDLE;

        VkSplatTestDevice() = default;
        VkSplatTestDevice(const VkSplatTestDevice&) = delete;
        VkSplatTestDevice& operator=(const VkSplatTestDevice&) = delete;

        ~VkSplatTestDevice() {
            if (allocator)
                vmaDestroyAllocator(allocator);
            if (device)
                lfs::rendering::vk_destroy_device_synced(device, nullptr);
            if (instance)
                vkDestroyInstance(instance, nullptr);
        }

        void create() {
            VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            app.apiVersion = VK_API_VERSION_1_3;
            VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            instance_info.pApplicationInfo = &app;
            if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS)
                GTEST_SKIP() << "Vulkan 1.3 is unavailable";
            uint32_t count = 0;
            vkEnumeratePhysicalDevices(instance, &count, nullptr);
            std::vector<VkPhysicalDevice> devices(count);
            vkEnumeratePhysicalDevices(instance, &count, devices.data());
            for (const auto candidate : devices) {
                VkPhysicalDeviceProperties properties{};
                vkGetPhysicalDeviceProperties(candidate, &properties);
                if (properties.apiVersion < VK_API_VERSION_1_3 ||
                    properties.limits.maxPushConstantsSize < sizeof(VulkanGSRendererUniforms))
                    continue;
                uint32_t extension_count = 0;
                vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extension_count, nullptr);
                std::vector<VkExtensionProperties> extensions(extension_count);
                vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extension_count, extensions.data());
                if (std::ranges::none_of(extensions, [](const auto& extension) {
                        return std::strcmp(extension.extensionName, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) == 0;
                    }))
                    continue;
                uint32_t family_count = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
                std::vector<VkQueueFamilyProperties> families(family_count);
                vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
                const auto compute = std::ranges::find_if(families, [](const auto& properties) {
                    return (properties.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
                });
                if (compute != families.end()) {
                    physical = candidate;
                    family = static_cast<uint32_t>(compute - families.begin());
                    break;
                }
            }
            if (physical == VK_NULL_HANDLE)
                GTEST_SKIP() << "No device supports the viewer's Vulkan compute contracts";

            VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            features12.pNext = &features13;
            VkPhysicalDeviceVulkan11Features features11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
            features11.pNext = &features12;
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features.pNext = &features11;
            vkGetPhysicalDeviceFeatures2(physical, &features);
            if (!features13.synchronization2 || !features12.timelineSemaphore || !features12.bufferDeviceAddress)
                GTEST_SKIP() << "Required Vulkan features are unavailable";
            const float priority = 1.0f;
            VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue_info.queueFamilyIndex = family;
            queue_info.queueCount = 1;
            queue_info.pQueuePriorities = &priority;
            const char* extensions[] = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
            VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            device_info.pNext = &features;
            device_info.queueCreateInfoCount = 1;
            device_info.pQueueCreateInfos = &queue_info;
            device_info.enabledExtensionCount = 1;
            device_info.ppEnabledExtensionNames = extensions;
            ASSERT_EQ(vkCreateDevice(physical, &device_info, nullptr, &device), VK_SUCCESS);
            lfs::rendering::vk_get_device_queue_synced(device, family, 0, &queue);
            VmaAllocatorCreateInfo allocator_info{};
            allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
            allocator_info.instance = instance;
            allocator_info.physicalDevice = physical;
            allocator_info.device = device;
            allocator_info.vulkanApiVersion = VK_API_VERSION_1_3;
            ASSERT_EQ(vmaCreateAllocator(&allocator_info, &allocator), VK_SUCCESS);
        }
    };

} // namespace lfs::test
