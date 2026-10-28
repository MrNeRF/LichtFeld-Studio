/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/persisted_settings.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"

#include "core/logger.hpp"
#include "core/user_paths.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace lfs::core::persisted_settings {
    namespace {
        bool compatibleType(const Json& old, const Json& value) {
            if (value.is_number_integer())
                return old.is_number_integer();
            if (value.is_number())
                return old.is_number();
            return old.type() == value.type();
        }

        bool replacesInvalidField(const Json& old, const Json& updates) {
            if (updates.is_null())
                return false;
            if (!compatibleType(old, updates))
                return true;
            if (old.is_object() && updates.is_object())
                for (const auto& [key, value] : updates.items())
                    if (old.contains(key) && replacesInvalidField(old[key], value))
                        return true;
            return false;
        }

        Json identity(const Json& value, const std::string& field_name) {
            if (field_name.empty())
                return value;
            if (!value.is_object() || !value.contains(field_name))
                return nullptr;
            return value[field_name];
        }

        std::filesystem::path defaultBackupRoot() {
            auto paths = UserPaths::resolve();
            if (!paths)
                throw std::runtime_error("settings backup paths could not be resolved");
            return paths.value().backupDir();
        }

        void mergeArray(Json& destination, const Json& updates, bool preserve,
                        const std::string& identity_field, const ArrayIdentity& identities) {
            Json merged = Json::array();
            for (const auto& incoming : updates) {
                const auto id = identity(incoming, identity_field);
                Json item = incoming;
                if (!id.is_null()) {
                    for (const auto& old : destination) {
                        if (identity(old, identity_field) == id) {
                            if (old.is_object() && incoming.is_object()) {
                                item = old;
                                mergeKnown(item, incoming, preserve, identities);
                            }
                            break;
                        }
                    }
                }
                merged.push_back(std::move(item));
            }
            if (preserve) {
                for (const auto& old : destination) {
                    const auto id = identity(old, identity_field);
                    const bool represented = !id.is_null() &&
                                             std::any_of(updates.begin(), updates.end(), [&](const Json& incoming) {
                                                 return identity(incoming, identity_field) == id;
                                             });
                    const bool unknown = identity_field.empty()
                                             ? std::none_of(updates.begin(), updates.end(), [&](const Json& incoming) {
                                                   return incoming.type() == old.type();
                                               })
                                             : !represented;
                    if (unknown && std::find(merged.begin(), merged.end(), old) == merged.end())
                        merged.push_back(old);
                }
            }
            destination = std::move(merged);
        }
    } // namespace

    Json readObject(const std::filesystem::path& path) {
        std::ifstream stream(lfs::core::native_fs::native_path(path), std::ios::binary);
        if (!stream)
            throw std::runtime_error("settings file could not be opened");
        Json result = Json::parse(stream);
        if (!result.is_object())
            throw std::runtime_error("settings root must be an object");
        return result;
    }

    void backupDamaged(const std::filesystem::path& path,
                       const std::filesystem::path& backup_root) {
        auto root = backup_root;
        if (root.empty())
            root = defaultBackupRoot();
        const auto backup = backupFile(path, root, "corrupt-settings");
        if (!backup)
            throw std::runtime_error("damaged settings backup failed: " +
                                     lfs::format_for_developer(backup.error()));
    }

    void mergeKnown(Json& destination, const Json& updates, bool preserve_unrecognized,
                    const ArrayIdentity& array_identity) {
        for (const auto& [key, value] : updates.items()) {
            if (value.is_null()) {
                destination.erase(key);
                continue;
            }
            if (preserve_unrecognized && destination.contains(key) &&
                !compatibleType(destination[key], value))
                continue;
            if (value.is_object()) {
                if (!destination.contains(key) || !destination[key].is_object())
                    destination[key] = Json::object();
                mergeKnown(destination[key], value, preserve_unrecognized, array_identity);
            } else if (destination.contains(key) && destination[key].is_array() && value.is_array()) {
                const auto identity_it = array_identity.find(key);
                if (preserve_unrecognized || identity_it != array_identity.end())
                    mergeArray(destination[key], value, preserve_unrecognized,
                               identity_it == array_identity.end() ? std::string{} : identity_it->second,
                               array_identity);
                else
                    destination[key] = value;
            } else {
                destination[key] = value;
            }
        }
    }

    Json mergeForWrite(const std::filesystem::path& path, const Json& updates,
                       const char* version_key, int current_version, bool* backed_up,
                       const ArrayIdentity& array_identity,
                       const std::filesystem::path& backup_root) {
        if (backed_up)
            *backed_up = false;
        Json existing = Json::object();
        bool already_backed_up = false;
        std::error_code error;
        const bool exists = lfs::core::native_fs::exists(path, error);
        if (error)
            throw std::runtime_error("settings file could not be inspected: " + error.message());
        if (exists) {
            try {
                existing = readObject(path);
            } catch (const std::exception& read_error) {
                LOG_WARN("Could not load settings from '{}': {}", lfs::core::path_to_utf8(path), read_error.what());
                backupDamaged(path, backup_root);
                already_backed_up = true;
                if (backed_up)
                    *backed_up = true;
            }
        }
        const auto version = field<std::uint64_t>(existing, version_key, current_version);
        const bool future = version > static_cast<std::uint64_t>(current_version);
        bool damaged = exists && existing.contains(version_key) &&
                       (!existing[version_key].is_number_integer() ||
                        (!existing[version_key].is_number_unsigned() &&
                         existing[version_key].get<std::int64_t>() < 0));
        for (const auto& [key, value] : updates.items())
            if (existing.contains(key) && key != version_key && !future &&
                replacesInvalidField(existing[key], value))
                damaged = true;
        if (damaged && !already_backed_up) {
            backupDamaged(path, backup_root);
            if (backed_up)
                *backed_up = true;
        }
        for (const auto& [key, value] : updates.items()) {
            if (key == version_key && future)
                continue;
            mergeKnown(existing, Json{{key, value}}, future, array_identity);
        }
        return existing;
    }
} // namespace lfs::core::persisted_settings
