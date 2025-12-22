/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include <filesystem>
#include <string>

namespace lfs::rendering {

    inline std::filesystem::path getShaderPath(const std::string& shader_name) {
        LOG_TIMER_TRACE("getShaderPath");
        LOG_TRACE("Looking for shader file: {}", shader_name);

        // Primary: Use runtime-detected resource directory
        auto runtime_path = lfs::core::getShadersDir() / shader_name;
        if (std::filesystem::exists(runtime_path)) {
            LOG_DEBUG("Found shader '{}' at runtime path: {}", shader_name, runtime_path.string());
            return runtime_path;
        }
        LOG_TRACE("Shader '{}' not found at runtime path: {}", shader_name, runtime_path.string());

        // Development fallback: Try build directory (set at compile time)
#ifdef SHADER_PATH
        std::filesystem::path build_path = std::filesystem::path(SHADER_PATH) / shader_name;
        if (std::filesystem::exists(build_path)) {
            LOG_DEBUG("Found shader '{}' in build directory: {}", shader_name, build_path.string());
            return build_path;
        }
        LOG_TRACE("Shader '{}' not found in build directory: {}", shader_name, build_path.string());
#endif

        // Development fallback: Source directory
#ifdef RENDERING_SOURCE_SHADER_PATH
        std::filesystem::path source_path = std::filesystem::path(RENDERING_SOURCE_SHADER_PATH) / shader_name;
        if (std::filesystem::exists(source_path)) {
            LOG_DEBUG("Found shader '{}' in source directory: {}", shader_name, source_path.string());
            return source_path;
        }
        LOG_TRACE("Shader '{}' not found in source directory: {}", shader_name, source_path.string());
#endif

        LOG_ERROR("Cannot find shader file: {}", shader_name);
        throw std::runtime_error("Cannot find shader file: " + shader_name);
    }

} // namespace lfs::rendering
