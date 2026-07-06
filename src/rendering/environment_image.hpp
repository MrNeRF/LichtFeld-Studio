/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace lfs::rendering {

    struct EnvironmentImage {
        std::filesystem::path path;
        int width = 0;
        int height = 0;
        std::vector<float> pixels; // [height, width, 3]

        [[nodiscard]] bool valid() const {
            return width > 0 && height > 0 &&
                   pixels.size() == static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
        }
    };

    // Loads the environment map as float RGB through a process-wide single-entry
    // cache keyed by the resolved path.
    std::expected<EnvironmentImage, std::string> loadEnvironmentImage(const std::filesystem::path& environment_path);

} // namespace lfs::rendering
