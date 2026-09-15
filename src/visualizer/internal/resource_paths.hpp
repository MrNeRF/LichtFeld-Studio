/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/executable_path.hpp"
#include "core/path_utils.hpp"
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfs::vis {

    inline std::filesystem::path resolveAssetPathUncached(const std::string& asset_name) {
        std::vector<std::filesystem::path> search_paths;

#ifdef LFS_DEV_RMLUI_SOURCE_DIR
        constexpr std::string_view rmlui_prefix = "rmlui/";
        if (asset_name.rfind(rmlui_prefix, 0) == 0) {
            search_paths.push_back(
                lfs::core::utf8_to_path(LFS_DEV_RMLUI_SOURCE_DIR) /
                asset_name.substr(rmlui_prefix.size()));
        }
#endif

        // Primary: Use runtime-detected resource directory
        search_paths.push_back(lfs::core::getAssetsDir() / asset_name);

        // Development fallback: Try build directory
#ifdef VISUALIZER_ASSET_PATH
        search_paths.push_back(std::filesystem::path(VISUALIZER_ASSET_PATH) / asset_name);
#endif

        // Development fallback: Source directory
#ifdef VISUALIZER_SOURCE_ASSET_PATH
        search_paths.push_back(std::filesystem::path(VISUALIZER_SOURCE_ASSET_PATH) / asset_name);
#endif

#ifdef PROJECT_ROOT_PATH
        search_paths.push_back(std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/assets" / asset_name);
        if (asset_name == "fonts/JetBrainsMono-Regular.ttf") {
            search_paths.push_back(std::filesystem::path(PROJECT_ROOT_PATH) /
                                   "src/rendering/resources/assets/JetBrainsMono-Regular.ttf");
        }
#endif

        for (const auto& path : search_paths) {
            if (std::filesystem::exists(path))
                return path;
        }

        // Build error message showing all searched locations
        std::string error_msg = "Cannot find asset: " + asset_name + "\nSearched in:\n";
        for (const auto& path : search_paths) {
            error_msg += "  - " + lfs::core::path_to_utf8(path) + "\n";
        }
        error_msg += "\nExecutable directory: " + lfs::core::path_to_utf8(lfs::core::getExecutableDir());

        throw std::runtime_error(error_msg);
    }

    inline std::filesystem::path getAssetPath(const std::string& asset_name) {
        static std::mutex cache_mutex;
        static std::unordered_map<std::string, std::filesystem::path> cache;
        {
            const std::lock_guard lock(cache_mutex);
            if (const auto it = cache.find(asset_name); it != cache.end())
                return it->second;
        }

        const auto path = resolveAssetPathUncached(asset_name);
        {
            const std::lock_guard lock(cache_mutex);
            cache.emplace(asset_name, path);
        }
        return path;
    }

} // namespace lfs::vis
