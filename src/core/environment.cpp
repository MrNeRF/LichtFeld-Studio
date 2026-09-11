/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/environment.hpp"
#include "core/path_utils.hpp"

#include <cstdlib>
#include <exception>
#include <utility>

namespace lfs::core::environment {

    std::optional<std::string> value(const char* const name) noexcept {
        if (name == nullptr || *name == '\0') {
            return std::nullopt;
        }
        try {
#ifdef _WIN32
            const auto wide_name = utf8_to_wstring(std::string(name));
            std::wstring wide_value(256, L'\0');
            for (;;) {
                const DWORD length = GetEnvironmentVariableW(
                    wide_name.c_str(), wide_value.data(), static_cast<DWORD>(wide_value.size()));
                if (length == 0) {
                    return std::nullopt;
                }
                if (length < wide_value.size()) {
                    wide_value.resize(length);
                    break;
                }
                wide_value.resize(static_cast<std::size_t>(length) + 1);
            }

            auto result = wstring_to_utf8(wide_value);
            return result.empty() ? std::nullopt : std::optional<std::string>{std::move(result)};
#else
            const char* const raw = std::getenv(name);
            if (!raw || *raw == '\0') {
                return std::nullopt;
            }
            return std::string(raw);
#endif
        } catch (const std::exception&) {
            // LFS-CENSUS-OK(empty-catch): environment lookup must remain noexcept.
            return std::nullopt;
        }
    }

    bool set_value(const std::string_view name, const std::string_view utf8_value) noexcept {
        if (name.empty()) {
            return false;
        }
        try {
#ifdef _WIN32
            const auto wide_name = utf8_to_wstring(std::string(name));
            const auto wide_value = utf8_to_wstring(std::string(utf8_value));
            return _wputenv_s(wide_name.c_str(), wide_value.c_str()) == 0;
#else
            return ::setenv(std::string(name).c_str(), std::string(utf8_value).c_str(), 1) == 0;
#endif
        } catch (const std::exception&) {
            // LFS-CENSUS-OK(empty-catch): environment writes must remain noexcept.
            return false;
        }
    }

} // namespace lfs::core::environment
