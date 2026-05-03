/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_loader_probe.hpp"

#include "config.h"
#include "vulkan_result.hpp"

#include <format>

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include <vulkan/vulkan.h>
#endif

namespace lfs::vis {

    std::string formatVulkanApiVersion(const uint32_t api_version) {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        return std::format("{}.{}.{}",
                           VK_API_VERSION_MAJOR(api_version),
                           VK_API_VERSION_MINOR(api_version),
                           VK_API_VERSION_PATCH(api_version));
#else
        (void)api_version;
        return "disabled";
#endif
    }

    VulkanLoaderInfo probeVulkanLoader() {
        VulkanLoaderInfo info{};

#ifdef LFS_VULKAN_VIEWER_ENABLED
        info.enabled = true;
        info.api_version = VK_API_VERSION_1_3;

        auto* const proc = vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion");
        if (proc == nullptr) {
            info.loader_available = true;
            return info;
        }

        const auto enumerate_instance_version =
            reinterpret_cast<PFN_vkEnumerateInstanceVersion>(proc);
        const VkResult result = enumerate_instance_version(&info.api_version);
        if (result != VK_SUCCESS) {
            info.error = std::format("vkEnumerateInstanceVersion returned {}", vkResultToString(result));
            return info;
        }

        info.loader_available = true;
#else
        info.enabled = false;
#endif

        return info;
    }

} // namespace lfs::vis
