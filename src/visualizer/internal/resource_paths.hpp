/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/executable_path.hpp"
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfs::vis {

    inline std::filesystem::path getAssetPath(const std::string& asset_name) {
        std::vector<std::filesystem::path> search_paths;

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
#endif

        // Try each path
        for (const auto& path : search_paths) {
            if (std::filesystem::exists(path)) {
                return path;
            }
        }

        // Build error message showing all searched locations
        std::string error_msg = "Cannot find asset: " + asset_name + "\nSearched in:\n";
        for (const auto& path : search_paths) {
            error_msg += "  - " + path.string() + "\n";
        }
        error_msg += "\nExecutable directory: " + lfs::core::getExecutableDir().string();

        throw std::runtime_error(error_msg);
    }

    inline std::filesystem::path getShaderPath(const std::string& shader_name) {
        std::vector<std::filesystem::path> search_paths;

        // Primary: Use runtime-detected resource directory
        search_paths.push_back(lfs::core::getShadersDir() / shader_name);

#ifdef VISUALIZER_SHADER_PATH
        search_paths.push_back(std::filesystem::path(VISUALIZER_SHADER_PATH) / shader_name);
#endif

#ifdef VISUALIZER_SOURCE_SHADER_PATH
        search_paths.push_back(std::filesystem::path(VISUALIZER_SOURCE_SHADER_PATH) / shader_name);
#endif

#ifdef PROJECT_ROOT_PATH
        search_paths.push_back(std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/resources/shaders" / shader_name);
#endif

        for (const auto& path : search_paths) {
            if (std::filesystem::exists(path)) {
                return path;
            }
        }

        std::string error_msg = "Cannot find shader: " + shader_name + "\nSearched in:\n";
        for (const auto& path : search_paths) {
            error_msg += "  - " + path.string() + "\n";
        }
        throw std::runtime_error(error_msg);
    }

} // namespace lfs::vis
