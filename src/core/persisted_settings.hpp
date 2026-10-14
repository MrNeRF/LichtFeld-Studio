/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/logger.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <type_traits>

namespace lfs::core::persisted_settings {
    using Json = nlohmann::json;
    using ArrayIdentity = std::map<std::string, std::string>;

    [[nodiscard]] inline bool present(const Json& object, const char* key) {
        if (!object.is_object())
            return false;
        const auto found = object.find(key);
        return found != object.end() && !found->is_null();
    }

    template <typename T>
    T field(const Json& object, const char* key, T fallback) {
        if (!object.is_object())
            return fallback;
        const auto found = object.find(key);
        if (found == object.end())
            return fallback;
        bool valid = false;
        if constexpr (std::is_same_v<T, std::string>)
            valid = found->is_string();
        else if constexpr (std::is_same_v<T, bool>)
            valid = found->is_boolean();
        else if constexpr (std::is_integral_v<T>) {
            if (found->is_number_unsigned())
                valid = found->get<std::uint64_t>() <= static_cast<std::uint64_t>(std::numeric_limits<T>::max());
            else if (found->is_number_integer()) {
                const auto number = found->get<std::int64_t>();
                if constexpr (std::is_signed_v<T>)
                    valid = number >= static_cast<std::int64_t>(std::numeric_limits<T>::min()) &&
                            number <= static_cast<std::int64_t>(std::numeric_limits<T>::max());
                else
                    valid = number >= 0 && static_cast<std::uint64_t>(number) <=
                                               static_cast<std::uint64_t>(std::numeric_limits<T>::max());
            }
        } else if constexpr (std::is_floating_point_v<T>)
            valid = found->is_number();
        if (valid) {
            try {
                const auto value = found->get<T>();
                if constexpr (std::is_floating_point_v<T>) {
                    if (std::isfinite(value))
                        return value;
                } else {
                    return value;
                }
            } catch (const std::exception&) {
                // LFS-CENSUS-OK(empty-catch): an out-of-range field uses its default.
            }
        }
        LOG_WARN("Invalid type or range for settings field '{}'; using its default", key);
        return fallback;
    }

    LFS_CORE_API Json readObject(const std::filesystem::path& path);
    LFS_CORE_API void backupDamaged(const std::filesystem::path& path,
                                    const std::filesystem::path& backup_root = {});
    // Null updates explicitly remove their keys; omitted keys are preserved.
    LFS_CORE_API void mergeKnown(Json& destination, const Json& updates,
                                 bool preserve_unrecognized = false,
                                 const ArrayIdentity& array_identity = {});
    LFS_CORE_API Json mergeForWrite(const std::filesystem::path& path, const Json& updates,
                                    const char* version_key, int current_version,
                                    bool* backed_up = nullptr,
                                    const ArrayIdentity& array_identity = {},
                                    const std::filesystem::path& backup_root = {});
} // namespace lfs::core::persisted_settings
