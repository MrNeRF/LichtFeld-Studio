/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <string>
#include <vulkan/vulkan.h>

namespace lfs::core {
    // These features cover every unconditionally created viewer shader, including macro staging.
    inline std::string missing_viewer_shader_features(VkPhysicalDevice device) {
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f.pNext = &f11;
        f11.pNext = &f12;
        f12.pNext = &f13;
        vkGetPhysicalDeviceFeatures2(device, &f);
        VkPhysicalDeviceSubgroupSizeControlProperties size{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceFloatControlsProperties floats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext = &subgroup;
        subgroup.pNext = &size;
        size.pNext = &floats;
        vkGetPhysicalDeviceProperties2(device, &properties);
        std::string missing;
        const auto require = [&](bool present, const char* name) {
            if (!present) {
                if (!missing.empty())
                    missing += ", ";
                missing += name;
            }
        };
        require(f.features.shaderInt64, "shaderInt64");
        require(f.features.shaderInt16, "shaderInt16");
        require(f12.bufferDeviceAddress, "bufferDeviceAddress");
        require(f12.shaderFloat16, "shaderFloat16 (macro staging)");
        require(f11.storageBuffer16BitAccess, "storageBuffer16BitAccess");
        require(f11.uniformAndStorageBuffer16BitAccess, "uniformAndStorageBuffer16BitAccess (macro staging)");
        require(f12.storageBuffer8BitAccess, "storageBuffer8BitAccess");
        require(floats.shaderSignedZeroInfNanPreserveFloat32, "shaderSignedZeroInfNanPreserveFloat32");
        require(subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT, "compute subgroups");
        require(subgroup.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT, "subgroup BASIC");
        require(subgroup.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT, "subgroup ARITHMETIC");
        require(subgroup.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT, "subgroup BALLOT");
        require(subgroup.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT, "subgroup SHUFFLE");
        require(subgroup.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT, "subgroup VOTE");
        if (subgroup.subgroupSize != 32) {
            require(f13.subgroupSizeControl && size.minSubgroupSize <= 32 && size.maxSubgroupSize >= 32 &&
                        (size.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT),
                    "32-lane compute subgroups (subgroupSizeControl)");
        }
        return missing;
    }
} // namespace lfs::core
