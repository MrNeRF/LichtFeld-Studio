/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <string>

namespace lfs::vis {

    struct VulkanLoaderInfo {
        bool enabled = false;
        bool loader_available = false;
        std::string error;
    };

    [[nodiscard]] VulkanLoaderInfo probeVulkanLoader();

} // namespace lfs::vis
