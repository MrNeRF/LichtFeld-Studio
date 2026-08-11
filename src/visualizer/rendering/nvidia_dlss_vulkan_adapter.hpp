/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <array>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace lfs::vis {
    [[nodiscard]] std::vector<std::string> nvidiaDlssRequiredInstanceExtensions() noexcept;
    [[nodiscard]] std::vector<std::string> nvidiaDlssRequiredDeviceExtensions(
        VkInstance instance,
        VkPhysicalDevice physical_device) noexcept;
    void disableNvidiaDlssVulkanBootstrap() noexcept;
    [[nodiscard]] bool nvidiaDlssVulkanBootstrapReady() noexcept;
    [[nodiscard]] std::array<float, 3> nvidiaDlssRecommendedInputScales() noexcept;
    void registerNvidiaDlssVulkanAdapter();
} // namespace lfs::vis
