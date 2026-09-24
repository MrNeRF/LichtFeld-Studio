/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::core {
    struct VulkanFeatureRequirements {
        bool viewer_shaders = false;
        bool window_renderer = false;
    };

    struct VulkanFeatureCheck {
        std::string missing;
        bool shader_float16 = false;
        bool shader_float64 = false;
        bool shader_atomic_float = false;
        [[nodiscard]] bool supported() const { return missing.empty(); }
    };

    struct VulkanDeviceFeatureEnableChain {
        VkPhysicalDeviceFeatures2 features{};
        VkPhysicalDeviceVulkan11Features features11{};
        VkPhysicalDeviceVulkan12Features features12{};
        VkPhysicalDeviceVulkan13Features features13{};

        VulkanDeviceFeatureEnableChain() {
            features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
            features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
            features.pNext = &features11;
            features11.pNext = &features12;
            features12.pNext = &features13;
            features.features.shaderInt64 = VK_TRUE;
            features.features.shaderInt16 = VK_TRUE;
            features11.storageBuffer16BitAccess = VK_TRUE;
            features12.storageBuffer8BitAccess = VK_TRUE;
            features12.timelineSemaphore = VK_TRUE;
            features12.bufferDeviceAddress = VK_TRUE;
            features13.synchronization2 = VK_TRUE;
        }
    };

    struct VulkanPhysicalDeviceEnumeration {
        VkResult count_result = VK_SUCCESS;
        VkResult devices_result = VK_SUCCESS;
        uint32_t observed_count = 0;
        uint32_t destination_capacity = 0;
        std::vector<VkPhysicalDevice> devices;
    };

    inline VulkanPhysicalDeviceEnumeration enumerate_vulkan_physical_devices(VkInstance instance) {
        VulkanPhysicalDeviceEnumeration result;
        uint32_t count = 0;
        result.count_result = vkEnumeratePhysicalDevices(instance, &count, nullptr);
        result.observed_count = count;
        if (result.count_result != VK_SUCCESS || count == 0)
            return result;
        result.destination_capacity = count;
        result.devices.resize(count);
        result.devices_result = vkEnumeratePhysicalDevices(instance, &count, result.devices.data());
        result.observed_count = count;
        if (result.devices_result == VK_SUCCESS)
            result.devices.resize(count);
        else
            result.devices.clear();
        return result;
    }

    inline VkResult create_vulkan_device(VkPhysicalDevice physical_device,
                                         const std::vector<VkDeviceQueueCreateInfo>& queues,
                                         const std::vector<const char*>& extensions,
                                         const void* feature_chain,
                                         const VkPhysicalDeviceFeatures* features,
                                         VkDevice* device) {
        VkDeviceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        info.pNext = feature_chain;
        info.queueCreateInfoCount = static_cast<uint32_t>(queues.size());
        info.pQueueCreateInfos = queues.data();
        info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        info.ppEnabledExtensionNames = extensions.data();
        info.pEnabledFeatures = features;
        return vkCreateDevice(physical_device, &info, nullptr, device);
    }

    inline uint32_t find_vulkan_memory_type(const VkPhysicalDeviceMemoryProperties& memory,
                                            uint32_t type_filter,
                                            VkMemoryPropertyFlags properties) {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if ((type_filter & (1u << i)) &&
                (memory.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }
        return std::numeric_limits<uint32_t>::max();
    }

    // This feature list is shared by every Vulkan backend entry point.
    inline VulkanFeatureCheck check_vulkan_feature_requirements(
        VkPhysicalDevice device, VulkanFeatureRequirements requirements = {}) {
        VkPhysicalDeviceVulkan13Features f13{};
        f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        VkPhysicalDeviceVulkan12Features f12{};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceVulkan11Features f11{};
        f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        VkPhysicalDeviceFeatures2 f{};
        f.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f.pNext = &f11;
        f11.pNext = &f12;
        f12.pNext = &f13;
        vkGetPhysicalDeviceFeatures2(device, &f);

        VkPhysicalDeviceSubgroupSizeControlProperties size{};
        size.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
        VkPhysicalDeviceSubgroupProperties subgroup{};
        subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceFloatControlsProperties floats{};
        floats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
        VkPhysicalDeviceProperties2 properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties.pNext = &subgroup;
        subgroup.pNext = &size;
        size.pNext = &floats;
        vkGetPhysicalDeviceProperties2(device, &properties);

        VulkanFeatureCheck result;
        const auto require = [&result](bool present, std::string_view name) {
            if (!present) {
                if (!result.missing.empty())
                    result.missing += ", ";
                result.missing += name;
            }
        };
        const auto api = properties.properties.apiVersion;
        require(VK_API_VERSION_MAJOR(api) > 1 ||
                    (VK_API_VERSION_MAJOR(api) == 1 && VK_API_VERSION_MINOR(api) >= 3),
                "Vulkan 1.3");
        require(f.features.shaderInt64, "shaderInt64");
        require(f.features.shaderInt16, "shaderInt16");
        require(f11.storageBuffer16BitAccess, "storageBuffer16BitAccess");
        require(f12.storageBuffer8BitAccess, "storageBuffer8BitAccess");
        require(f12.timelineSemaphore, "timelineSemaphore");
        require(f12.bufferDeviceAddress, "bufferDeviceAddress");
        require(f13.synchronization2, "synchronization2");
        require(floats.shaderSignedZeroInfNanPreserveFloat32,
                "shaderSignedZeroInfNanPreserveFloat32");
        const auto basic_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
                               VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
        require((subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
                    (subgroup.supportedOperations & basic_ops) == basic_ops,
                "compute subgroups");
        if (requirements.window_renderer) {
            require(f13.dynamicRendering, "dynamicRendering");
        }
        if (requirements.viewer_shaders) {
            require(f12.shaderFloat16, "shaderFloat16 (macro staging)");
            require(f11.uniformAndStorageBuffer16BitAccess,
                    "uniformAndStorageBuffer16BitAccess (macro staging)");
            require(subgroup.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT, "subgroup VOTE");
            if (subgroup.subgroupSize != 32) {
                require(f13.subgroupSizeControl && size.minSubgroupSize <= 32 &&
                            size.maxSubgroupSize >= 32 &&
                            (size.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT),
                        "32-lane compute subgroups (subgroupSizeControl)");
            }
        }
        result.shader_float16 = f12.shaderFloat16 == VK_TRUE;
        result.shader_float64 = f.features.shaderFloat64 == VK_TRUE;
        return result;
    }

    inline bool vulkan_instance_extension_available(const char* name) {
        uint32_t count = 0;
        if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS)
            return false;
        std::vector<VkExtensionProperties> available(count);
        if (vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data()) != VK_SUCCESS)
            return false;
        return std::ranges::any_of(available, [name](const auto& extension) {
            return std::string_view(extension.extensionName) == name;
        });
    }

    inline VkResult create_vulkan_instance(const VkApplicationInfo& application,
                                           const std::vector<const char*>& extensions,
                                           const std::vector<const char*>& layers,
                                           const void* next,
                                           VkInstanceCreateFlags flags,
                                           VkInstance* instance,
                                           bool portability_enumeration = false) {
        std::vector<const char*> enabled_extensions = extensions;
        if (portability_enumeration &&
            vulkan_instance_extension_available(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) &&
            std::ranges::find_if(enabled_extensions, [](const char* extension) {
                return std::string_view(extension) == VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
            }) == enabled_extensions.end()) {
            enabled_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
        VkInstanceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pNext = next;
        info.flags = flags;
        info.pApplicationInfo = &application;
        info.enabledLayerCount = static_cast<uint32_t>(layers.size());
        info.ppEnabledLayerNames = layers.data();
        info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
        info.ppEnabledExtensionNames = enabled_extensions.data();
        return vkCreateInstance(&info, nullptr, instance);
    }
} // namespace lfs::core
