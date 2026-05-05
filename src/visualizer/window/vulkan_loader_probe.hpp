/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <string>

namespace lfs::vis {

    struct VulkanLoaderInfo {
        bool enabled = false;
        bool loader_available = false;
        uint32_t api_version = 0;
        std::string error;
    };

    [[nodiscard]] VulkanLoaderInfo probeVulkanLoader();
    [[nodiscard]] std::string formatVulkanApiVersion(uint32_t api_version);

} // namespace lfs::vis
