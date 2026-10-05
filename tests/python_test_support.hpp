/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>

namespace lfs::test {
    inline bool containsLichtfeldModule(const std::filesystem::path& dir) {
        std::error_code ec;
        if (!std::filesystem::exists(dir, ec)) {
            return false;
        }

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
    }

    inline std::filesystem::path findPythonModuleDir() {
        std::error_code ec;
        const auto cwd = std::filesystem::current_path(ec);
        const auto project_root = std::filesystem::path(PROJECT_ROOT_PATH);

        for (const auto& candidate : {
                 cwd / "src" / "python",
                 cwd.parent_path() / "src" / "python",
                 project_root / "build" / "src" / "python",
             }) {
            if (containsLichtfeldModule(candidate)) {
                return candidate;
            }
        }

        return {};
    }

    inline void prependPythonPath(const std::filesystem::path& path) {
        const auto value = path.string();
        const char* existing = std::getenv("PYTHONPATH");
#ifdef _WIN32
        const char separator = ';';
#else
        const char separator = ':';
#endif
        const std::string combined =
            existing && *existing ? value + separator + std::string(existing) : value;

#ifdef _WIN32
        _putenv_s("PYTHONPATH", combined.c_str());
#else
        setenv("PYTHONPATH", combined.c_str(), 1);
#endif
    }

} // namespace lfs::test
