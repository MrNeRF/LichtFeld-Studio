/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <expected>
#include <filesystem>
#include <memory>
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

    // Shared hot-path view of the process-wide single-entry cache. Callers keep
    // the decoded pixels alive without copying the full float RGB image.
    std::expected<std::shared_ptr<const EnvironmentImage>, std::string>
    loadEnvironmentImageShared(const std::filesystem::path& environment_path);

    // Compatibility/value API for callers that need independent ownership.
    std::expected<EnvironmentImage, std::string> loadEnvironmentImage(const std::filesystem::path& environment_path);
    void releaseEnvironmentImageCache();

} // namespace lfs::rendering
