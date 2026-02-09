/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

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

    // Python module directory (lichtfeld.so/.pyd)
    inline std::filesystem::path getPythonModuleDir() {
        const auto exe_dir = getExecutableDir();

        auto module_exists = [](const std::filesystem::path& dir) {
            std::error_code ec;
            if (!std::filesystem::exists(dir, ec)) {
                return false;
            }

            // Check various naming patterns nanobind might produce.
            for (const auto& name : {
                     "lichtfeld.abi3.so",
                     "lichtfeld.so",
                     "lichtfeld.pyd",
                     "lichtfeld.abi3.pyd",
                     "lichtfeld.cp312-win_amd64.pyd",
                     "lichtfeld.cp311-win_amd64.pyd"}) {
                if (std::filesystem::exists(dir / name, ec)) {
                    return true;
                }
            }

            // Match platform-specific CPython suffixes (e.g., lichtfeld.cpython-312-x86_64-linux-gnu.so).
            for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
                if (ec) {
                    break;
                }
                if (!entry.is_regular_file(ec)) {
                    continue;
                }
                const auto filename = entry.path().filename().string();
                const auto ext = entry.path().extension().string();
                if ((ext == ".so" || ext == ".pyd") && filename.rfind("lichtfeld", 0) == 0) {
                    return true;
                }
            }

            return false;
        };

        // Production: exe in bin/, module in ../lib/python/
        if (const auto prod = exe_dir.parent_path() / "lib" / "python"; module_exists(prod)) {
            return prod;
        }

        // Development: module in src/python/ relative to exe (build dir)
        if (const auto dev = exe_dir / "src" / "python"; module_exists(dev)) {
            return dev;
        }

        // Fallback: module in same directory as exe (Windows dev builds)
        if (module_exists(exe_dir)) {
            return exe_dir;
        }

        return {};
    }

    // Python home directory (for embedded Python)
    inline std::filesystem::path getPythonHome() {
        const auto exe_dir = getExecutableDir();

#ifdef _WIN32
        // Windows Production: exe in bin/, Python stdlib in ../Lib/
        // CPython on Windows expects Lib/ (not lib/python3.12/)
        if (const auto prod = exe_dir.parent_path() / "Lib";
            std::filesystem::exists(prod / "os.py")) {
            return exe_dir.parent_path();
        }

        // Windows Development (vcpkg): python in vcpkg_installed/x64-windows/tools/python3/
        // vcpkg Python on Windows has Lib/ directly in tools/python3/
        const auto vcpkg = exe_dir / "vcpkg_installed" / "x64-windows" / "tools" / "python3";
        if (std::filesystem::exists(vcpkg / "python.exe")) {
            return vcpkg;
        }
#else
        // Linux Production: exe in bin/, Python stdlib in ../lib/python3.12/
        if (const auto prod = exe_dir.parent_path() / "lib" / "python3.12";
            std::filesystem::exists(prod)) {
            return exe_dir.parent_path();
        }

        // Linux Development (vcpkg): python3.12 in vcpkg_installed/x64-linux/tools/python3/
        // but stdlib is in vcpkg_installed/x64-linux/lib/python3.12/
        const auto vcpkg_tools = exe_dir / "vcpkg_installed" / "x64-linux" / "tools" / "python3";
        const auto vcpkg_lib = exe_dir / "vcpkg_installed" / "x64-linux" / "lib" / "python3.12";
        if (std::filesystem::exists(vcpkg_tools / "python3.12") && std::filesystem::exists(vcpkg_lib)) {
            // Return the prefix (parent of lib/)
            return exe_dir / "vcpkg_installed" / "x64-linux";
        }
#endif

        return {};
    }

    // Type stubs directory (lichtfeld/*.pyi)
    inline std::filesystem::path getTypingsDir() {
        const auto exe_dir = getExecutableDir();

        auto stubs_exist = [](const std::filesystem::path& dir) {
            return std::filesystem::exists(dir / "lichtfeld" / "__init__.pyi");
        };

        // Production: exe in bin/, stubs in ../lib/python/lichtfeld/
        if (const auto prod = exe_dir.parent_path() / "lib" / "python"; stubs_exist(prod)) {
            return prod;
        }

        // Development: stubs in src/python/typings/ relative to exe
        if (const auto dev = exe_dir / "src" / "python" / "typings"; stubs_exist(dev)) {
            return dev;
        }

        // Portable build: stubs alongside exe
        if (stubs_exist(exe_dir)) {
            return exe_dir;
        }

        return {};
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
