/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdlib>
#include <filesystem>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
// clang-format off
#include <windows.h>
#include <shlobj.h>
// clang-format on
#else
#include <pwd.h>
#include <unistd.h>
#endif

namespace lfs::core {

    inline std::filesystem::path user_config_dir() {
#ifdef _WIN32
        wchar_t path[MAX_PATH];
        if (SUCCEEDED(SHGetFolderPathW(nullptr, CSIDL_APPDATA, nullptr, 0, path))) {
            return std::filesystem::path(path) / "LichtFeldStudio";
        }
        if (const char* appdata = std::getenv("APPDATA")) {
            return std::filesystem::path(appdata) / "LichtFeldStudio";
        }
        return std::filesystem::current_path() / "config";
#else
        if (const char* xdg = std::getenv("XDG_CONFIG_HOME")) {
            return std::filesystem::path(xdg) / "LichtFeldStudio";
        }
        const char* home = std::getenv("HOME");
        if (!home) {
            if (const struct passwd* pw = getpwuid(getuid())) {
                home = pw->pw_dir;
            }
        }
        if (home) {
            return std::filesystem::path(home) / ".config" / "LichtFeldStudio";
        }
        return std::filesystem::current_path() / "config";
#endif
    }

} // namespace lfs::core
