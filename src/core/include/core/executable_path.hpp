/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
#endif

namespace lfs::core {

#ifdef _WIN32
    inline constexpr DWORD MAX_PATH_EXTENDED = 32768;
#endif

    inline std::filesystem::path getExecutablePath() {
#ifdef _WIN32
        std::wstring path(MAX_PATH, L'\0');
        DWORD size = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));

        while (size == path.size() && GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
            if (path.size() >= MAX_PATH_EXTENDED) {
                throw std::runtime_error("Executable path exceeds maximum length");
            }
            path.resize(path.size() * 2);
            size = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        }

        if (size == 0) {
            throw std::runtime_error("GetModuleFileNameW failed");
        }

        path.resize(size);
        return std::filesystem::path(path);
#else
        char path[PATH_MAX];
        const ssize_t count = readlink("/proc/self/exe", path, PATH_MAX - 1);
        if (count > 0 && count < PATH_MAX - 1) {
            path[count] = '\0';
            return std::filesystem::path(path);
        }
        throw std::runtime_error("readlink(/proc/self/exe) failed");
#endif
    }

    inline std::filesystem::path getExecutableDir() {
        return getExecutablePath().parent_path();
    }

    // Resource lookup: production (bin/../share/LichtFeld-Studio) or dev (build/resources)
    inline std::filesystem::path getResourceBaseDir() {
        const auto exe_dir = getExecutableDir();

        // Production: exe in bin/, resources in ../share/LichtFeld-Studio/
        if (const auto prod = exe_dir.parent_path() / "share" / "LichtFeld-Studio";
            std::filesystem::exists(prod)) {
            return prod;
        }

        // Development: resources/ alongside executable
        if (const auto dev = exe_dir / "resources"; std::filesystem::exists(dev)) {
            return dev;
        }

        return exe_dir;
    }

    inline std::filesystem::path getShadersDir() { return getResourceBaseDir() / "shaders"; }
    inline std::filesystem::path getAssetsDir() { return getResourceBaseDir() / "assets"; }
    inline std::filesystem::path getIconsDir() { return getAssetsDir() / "icon"; }
    inline std::filesystem::path getFontsDir() { return getAssetsDir() / "fonts"; }
    inline std::filesystem::path getThemesDir() { return getAssetsDir() / "themes"; }
    inline std::filesystem::path getLocalesDir() { return getResourceBaseDir() / "locales"; }

    // Library path lookup: production (bin/../lib) or build directory
    inline std::filesystem::path getLibDir() {
        const auto exe_dir = getExecutableDir();

        // Production: exe in bin/, libs in ../lib/
        if (const auto prod = exe_dir.parent_path() / "lib";
            std::filesystem::exists(prod)) {
            return prod;
        }

        // Development: check for extensions relative to build dir
        if (const auto dev = exe_dir / "lib"; std::filesystem::exists(dev)) {
            return dev;
        }

        return exe_dir;
    }

    // nvImageCodec extensions directory
    inline std::filesystem::path getExtensionsDir() {
        const auto exe_dir = getExecutableDir();

        // Distribution: exe in bin/, extensions in ../extensions/ (sibling directories)
        if (const auto ext = exe_dir.parent_path() / "extensions"; std::filesystem::exists(ext)) {
            return ext;
        }

        // Standard location: lib/extensions/
        const auto lib_dir = getLibDir();
        if (const auto ext = lib_dir / "extensions"; std::filesystem::exists(ext)) {
            return ext;
        }

        // Fallback: extensions/ in exe directory (development)
        if (const auto ext = exe_dir / "extensions"; std::filesystem::exists(ext)) {
            return ext;
        }

        return {};
    }

} // namespace lfs::core
