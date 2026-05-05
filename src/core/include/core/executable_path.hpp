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
            for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
                std::error_code file_ec;
                if (!it->is_regular_file(file_ec) || file_ec) {
                    continue;
                }
                const auto filename = it->path().filename().string();
                const auto ext = it->path().extension().string();
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

    inline bool isVcpkgPythonPrefix(const std::filesystem::path& triplet_dir) {
        std::error_code ec;
#ifdef _WIN32
        return std::filesystem::exists(triplet_dir / "tools" / "python3" / "python.exe", ec);
#else
        return std::filesystem::exists(triplet_dir / "tools" / "python3" / "python3.12", ec) &&
               std::filesystem::exists(triplet_dir / "lib" / "python3.12", ec);
#endif
    }

    inline bool tripletLooksLikeCurrentPlatform(const std::string& name) {
#ifdef _WIN32
        if (name.find("windows") == std::string::npos) {
            return false;
        }
#else
        if (name.find("linux") == std::string::npos) {
            return false;
        }
#endif

#if defined(_M_ARM64) || defined(__aarch64__)
        return name.starts_with("arm64");
#elif defined(_M_X64) || defined(__x86_64__)
        return name.starts_with("x64");
#elif defined(_M_ARM) || defined(__arm__)
        return name.starts_with("arm");
#else
        return true;
#endif
    }

    inline std::filesystem::path preferredVcpkgPrefix([[maybe_unused]] const std::filesystem::path& vcpkg_root) {
#ifdef LFS_PYTHON_EXECUTABLE
        auto configured_prefix = std::filesystem::path(LFS_PYTHON_EXECUTABLE);
        for (int i = 0; i < 3 && !configured_prefix.empty(); ++i) {
            configured_prefix = configured_prefix.parent_path();
        }

        if (!configured_prefix.empty()) {
            if (configured_prefix.parent_path() == vcpkg_root && isVcpkgPythonPrefix(configured_prefix)) {
                return configured_prefix;
            }

            if (const auto mapped_prefix = vcpkg_root / configured_prefix.filename();
                isVcpkgPythonPrefix(mapped_prefix)) {
                return mapped_prefix;
            }
        }
#endif
        return {};
    }

    // Discover the vcpkg triplet directory under vcpkg_installed/ at runtime.
    // Returns empty path if no matching triplet is found.
    inline std::filesystem::path findVcpkgPrefix(const std::filesystem::path& exe_dir) {
        std::error_code ec;
        auto vcpkg_root = exe_dir / "vcpkg_installed";
        if (!std::filesystem::exists(vcpkg_root, ec)) {
            // Per-config build (e.g. build/Release/): vcpkg_installed lives one level up
            vcpkg_root = exe_dir.parent_path() / "vcpkg_installed";
            if (!std::filesystem::exists(vcpkg_root, ec)) {
                return {};
            }
        }

        if (const auto prefix = preferredVcpkgPrefix(vcpkg_root); !prefix.empty()) {
            return prefix;
        }

        std::filesystem::path preferred;
        std::string preferred_name;
        std::filesystem::path fallback;
        std::string fallback_name;

        for (std::filesystem::directory_iterator it(vcpkg_root, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code dir_ec;
            if (!it->is_directory(dir_ec) || dir_ec) {
                continue;
            }
            const auto triplet_dir = it->path();
            const auto name = triplet_dir.filename().string();
            if (name == "vcpkg" || name.starts_with(".")) {
                continue;
            }
            if (!isVcpkgPythonPrefix(triplet_dir)) {
                continue;
            }

            if (fallback.empty() || name < fallback_name) {
                fallback = triplet_dir;
                fallback_name = name;
            }

            if (tripletLooksLikeCurrentPlatform(name) &&
                (preferred.empty() || name < preferred_name)) {
                preferred = triplet_dir;
                preferred_name = name;
            }
        }

        if (!preferred.empty()) {
            return preferred;
        }
        return fallback;
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

        // Windows Development (vcpkg)
        if (const auto prefix = findVcpkgPrefix(exe_dir); !prefix.empty()) {
            return prefix / "tools" / "python3";
        }
#else
        // Linux Production: exe in bin/, Python stdlib in ../lib/python3.12/
        if (const auto prod = exe_dir.parent_path() / "lib" / "python3.12";
            std::filesystem::exists(prod)) {
            return exe_dir.parent_path();
        }

        // Linux Development (vcpkg)
        if (const auto prefix = findVcpkgPrefix(exe_dir); !prefix.empty()) {
            return prefix;
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

    // Embedded Python executable (base interpreter for uv sync/venv and plugin isolation)
    inline std::filesystem::path getEmbeddedPython() {
        const auto exe_dir = getExecutableDir();

#ifdef _WIN32
        if (const auto p = exe_dir / "python.exe"; std::filesystem::exists(p))
            return p;

        if (const auto prefix = findVcpkgPrefix(exe_dir); !prefix.empty()) {
            if (const auto p = prefix / "tools" / "python3" / "python.exe"; std::filesystem::exists(p))
                return p;
        }
#else
        if (const auto p = exe_dir / "python3"; std::filesystem::exists(p))
            return p;

        if (const auto prefix = findVcpkgPrefix(exe_dir); !prefix.empty()) {
            if (const auto p = prefix / "tools" / "python3" / "python3.12"; std::filesystem::exists(p))
                return p;
        }
#endif

#ifdef LFS_PYTHON_EXECUTABLE
        if (const auto p = std::filesystem::path(LFS_PYTHON_EXECUTABLE); std::filesystem::exists(p))
            return p;
#endif

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
