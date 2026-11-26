/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "openxr_runtimes.hpp"
#include "core/environment.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#endif

namespace lfs::vis::vr {

    namespace {
        using Path = std::filesystem::path;
        namespace native_fs = lfs::core::native_fs;

        [[nodiscard]] bool isFile(const Path& path) {
            std::error_code error;
            return native_fs::is_regular_file(path, error);
        }

        // Two spellings of one file compare equal.
        [[nodiscard]] Path sameFileKey(const Path& path) {
            std::error_code error;
            const Path canonical = native_fs::weakly_canonical(path, error);
            return error ? path.lexically_normal() : canonical;
        }

        [[nodiscard]] std::string runtimeName(const Path& manifest) {
            std::ifstream file(native_fs::native_path(manifest));
            const auto parsed = nlohmann::json::parse(file, nullptr, false);
            if (parsed.is_object()) {
                const auto runtime = parsed.find("runtime");
                if (runtime != parsed.end() && runtime->is_object()) {
                    const auto name = runtime->find("name");
                    if (name != runtime->end() && name->is_string() && !name->get<std::string>().empty())
                        return name->get<std::string>();
                }
            }
            return lfs::core::path_to_utf8(manifest.stem());
        }

        struct Declared {
            Path active;
            std::vector<Path> available;
        };

#ifdef _WIN32
        // The OpenXR loader reads the same key: the active runtime's manifest,
        // and under AvailableRuntimes one value per installed runtime, named by
        // its manifest and holding 0 while it is offered.
        [[nodiscard]] Declared declaredRuntimes() {
            Declared declared;
            for (const HKEY root : {HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE}) {
                HKEY key = nullptr;
                if (RegOpenKeyExW(root, L"SOFTWARE\\Khronos\\OpenXR\\1", 0, KEY_READ | KEY_WOW64_64KEY, &key) != ERROR_SUCCESS)
                    continue;
                if (declared.active.empty()) {
                    wchar_t value[32768];
                    DWORD size = sizeof(value);
                    if (RegGetValueW(key, nullptr, L"ActiveRuntime", RRF_RT_REG_SZ, nullptr, value, &size) == ERROR_SUCCESS)
                        declared.active = value;
                }
                HKEY available = nullptr;
                if (RegOpenKeyExW(key, L"AvailableRuntimes", 0, KEY_READ | KEY_WOW64_64KEY, &available) == ERROR_SUCCESS) {
                    for (DWORD index = 0;; ++index) {
                        wchar_t name[32768];
                        DWORD name_size = static_cast<DWORD>(std::size(name));
                        DWORD type = 0;
                        DWORD disabled = 0;
                        DWORD data_size = sizeof(disabled);
                        const LSTATUS status = RegEnumValueW(available, index, name, &name_size, nullptr, &type,
                                                             reinterpret_cast<BYTE*>(&disabled), &data_size);
                        if (status == ERROR_NO_MORE_ITEMS)
                            break;
                        if (status == ERROR_SUCCESS && type == REG_DWORD && disabled == 0)
                            declared.available.emplace_back(name);
                    }
                    RegCloseKey(available);
                }
                RegCloseKey(key);
            }
            return declared;
        }
#else
        // The loader takes the first active_runtime file of the XDG config
        // directories; installed runtimes put their manifests under share/.
        [[nodiscard]] Declared declaredRuntimes() {
            Declared declared;
            const auto env = [](const char* const name) { return lfs::core::environment::value(name).value_or(""); };
            const std::string home = env("HOME");
            std::vector<Path> config;
            if (const std::string xdg = env("XDG_CONFIG_HOME"); !xdg.empty())
                config.emplace_back(xdg);
            else if (!home.empty())
                config.emplace_back(Path(home) / ".config");
            std::string dirs = env("XDG_CONFIG_DIRS");
            if (dirs.empty())
                dirs = "/etc/xdg";
            for (std::size_t start = 0; start <= dirs.size();) {
                const std::size_t end = std::min(dirs.find(':', start), dirs.size());
                if (end > start)
                    config.emplace_back(dirs.substr(start, end - start));
                start = end + 1;
            }
            config.emplace_back("/etc");
            for (const Path& directory : config) {
                for (const char* const file : {"active_runtime.x86_64.json", "active_runtime.aarch64.json", "active_runtime.json"}) {
                    const Path candidate = directory / "openxr" / "1" / file;
                    if (declared.active.empty() && isFile(candidate))
                        declared.active = candidate;
                }
            }
            std::vector<Path> shared{"/usr/share/openxr/1", "/usr/local/share/openxr/1"};
            if (!home.empty()) {
                shared.push_back(Path(home) / ".local/share/openxr/1");
                for (const char* const steam : {".local/share/Steam", ".steam/steam"})
                    declared.available.push_back(Path(home) / steam / "steamapps/common/SteamVR/steamxr_linux64.json");
            }
            for (const Path& directory : shared) {
                std::error_code error;
                for (auto entry = native_fs::directory_iterator(directory, error);
                     !error && entry != std::filesystem::directory_iterator(); entry.increment(error)) {
                    if (entry->path().extension() == ".json")
                        declared.available.push_back(entry->path());
                }
            }
            return declared;
        }
#endif
    } // namespace

    OpenXrRuntimes installedOpenXrRuntimes() {
        const Declared declared = declaredRuntimes();
        OpenXrRuntimes runtimes;
        std::vector<Path> seen;
        const auto add = [&](const Path& manifest) {
            if (manifest.empty() || !isFile(manifest))
                return;
            const Path key = sameFileKey(manifest);
            if (std::ranges::find(seen, key) != seen.end())
                return;
            seen.push_back(key);
            runtimes.installed.push_back({runtimeName(manifest), lfs::core::path_to_utf8(manifest)});
        };
        for (const Path& manifest : declared.available)
            add(manifest);
        add(declared.active);
        std::ranges::sort(runtimes.installed, {}, &OpenXrRuntime::name);
        if (!declared.active.empty() && isFile(declared.active))
            runtimes.active = runtimeName(declared.active);
        return runtimes;
    }

    bool useOpenXrRuntime(const std::string& manifest) {
        if (manifest.empty() || !isFile(lfs::core::utf8_to_path(manifest)))
            return false;
        // The loader reads this variable before the system's record.
        return lfs::core::environment::set_value("XR_RUNTIME_JSON", manifest);
    }

} // namespace lfs::vis::vr
