/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

namespace lfs::core {

/**
 * @brief Get the absolute path to the currently running executable.
 * @return Absolute path to the executable file.
 */
inline std::filesystem::path getExecutablePath() {
#ifdef _WIN32
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    return std::filesystem::path(path);
#else
    char path[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
    if (count != -1) {
        return std::filesystem::path(std::string(path, count));
    }
    // Fallback: try to resolve from argv[0] or current path
    return std::filesystem::current_path();
#endif
}

/**
 * @brief Get the directory containing the executable.
 * @return Absolute path to the directory containing the executable.
 */
inline std::filesystem::path getExecutableDir() {
    return getExecutablePath().parent_path();
}

/**
 * @brief Get the application's resource base directory.
 *
 * Resource structure (relative to executable):
 *   bin/LichtFeld-Studio        <- executable
 *   share/LichtFeld-Studio/     <- resources
 *       shaders/
 *       assets/
 *           icon/
 *           fonts/
 *           themes/
 *
 * For development builds, also checks:
 *   build/LichtFeld-Studio      <- executable
 *   build/resources/            <- resources (copied by CMake)
 *
 * @return Absolute path to the resource base directory.
 */
inline std::filesystem::path getResourceBaseDir() {
    auto exe_dir = getExecutableDir();

    // Production layout: exe in bin/, resources in ../share/LichtFeld-Studio/
    auto prod_path = exe_dir.parent_path() / "share" / "LichtFeld-Studio";
    if (std::filesystem::exists(prod_path)) {
        return prod_path;
    }

    // Development layout: resources in same dir as executable under resources/
    auto dev_path = exe_dir / "resources";
    if (std::filesystem::exists(dev_path)) {
        return dev_path;
    }

    // Fallback: resources directly alongside executable
    return exe_dir;
}

/**
 * @brief Get the path to the shaders directory.
 * @return Absolute path to the shaders directory.
 */
inline std::filesystem::path getShadersDir() {
    return getResourceBaseDir() / "shaders";
}

/**
 * @brief Get the path to the assets directory.
 * @return Absolute path to the assets directory.
 */
inline std::filesystem::path getAssetsDir() {
    return getResourceBaseDir() / "assets";
}

/**
 * @brief Get the path to the icons directory.
 * @return Absolute path to the icons directory.
 */
inline std::filesystem::path getIconsDir() {
    return getAssetsDir() / "icon";
}

/**
 * @brief Get the path to the fonts directory.
 * @return Absolute path to the fonts directory.
 */
inline std::filesystem::path getFontsDir() {
    return getAssetsDir() / "fonts";
}

/**
 * @brief Get the path to the themes directory.
 * @return Absolute path to the themes directory.
 */
inline std::filesystem::path getThemesDir() {
    return getAssetsDir() / "themes";
}

} // namespace lfs::core
