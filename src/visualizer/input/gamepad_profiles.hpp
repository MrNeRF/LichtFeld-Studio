/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "gamepad_navigation.hpp"
#include <set>

namespace lfs::vis::input {

    // Named semantic layouts, independent of keyboard profiles and live connections.
    // Enablement, connection IDs and viewport permission never belong to a profile.
    struct GamepadProfiles {
        nlohmann::json value;

        static nlohmann::json profileSettings(const GamepadNavigationSettings& settings) {
            auto result = settings.json();
            result.erase("enabled");
            return result;
        }

        static bool validDeviceKey(const std::string& key) {
            return key.size() == 32 && key.find_first_not_of("0123456789abcdef") == std::string::npos &&
                   key != std::string(32, '0');
        }

        static GamepadProfiles migrate(const GamepadNavigationSettings& settings) {
            return fromJson({{"schema_version", 1}, {"active", "personal"},
                             {"profiles", {{{"id", "personal"}, {"name", "Personal"},
                                            {"settings", profileSettings(settings)}}}},
                             {"device_profiles", nlohmann::json::object()}});
        }

        static GamepadProfiles fromJson(const nlohmann::json& input) {
            const auto require = [](bool condition) {
                if (!condition) throw std::invalid_argument("Invalid gamepad profile library");
            };
            require(input.is_object() && input.size() == 4 && input.contains("schema_version") &&
                    input["schema_version"].is_number_integer() && input["schema_version"] == 1 &&
                    input.contains("active") && input["active"].is_string() &&
                    input.contains("profiles") && input["profiles"].is_array() &&
                    !input["profiles"].empty() && input["profiles"].size() <= 64 &&
                    input.contains("device_profiles") && input["device_profiles"].is_object() &&
                    input["device_profiles"].size() <= 256);
            auto normalized = input;
            std::set<std::string> ids, names;
            for (auto& profile : normalized["profiles"]) {
                require(profile.is_object() && profile.size() == 3 && profile.contains("id") &&
                        profile["id"].is_string() && profile.contains("name") && profile["name"].is_string() &&
                        profile.contains("settings") && profile["settings"].is_object());
                const auto id = profile["id"].get<std::string>();
                const auto name = profile["name"].get<std::string>();
                require(!id.empty() && id.size() <= 64 &&
                        id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == std::string::npos &&
                        ids.insert(id).second && !name.empty() && name.size() <= 128 &&
                        name.front() != ' ' && name.back() != ' ' &&
                        std::none_of(name.begin(), name.end(), [](unsigned char c) { return c < 32 || c == 127; }) &&
                        names.insert(name).second && !profile["settings"].contains("enabled"));
                profile["settings"] = profileSettings(GamepadNavigationSettings::fromJson(profile["settings"]));
            }
            require(ids.contains(input["active"].get<std::string>()));
            for (auto it = input["device_profiles"].begin(); it != input["device_profiles"].end(); ++it)
                require(validDeviceKey(it.key()) && it.value().is_string() && ids.contains(it.value().get<std::string>()));
            return {std::move(normalized)};
        }

        GamepadNavigationSettings activeSettings(bool enabled) const {
            for (const auto& profile : value["profiles"])
                if (profile["id"] == value["active"]) {
                    auto result = GamepadNavigationSettings::fromJson(profile["settings"]);
                    result.enabled = enabled;
                    return result;
                }
            throw std::logic_error("Missing active gamepad profile");
        }

        void updateActive(const GamepadNavigationSettings& settings) {
            for (auto& profile : value["profiles"])
                if (profile["id"] == value["active"]) profile["settings"] = profileSettings(settings);
        }
    };
} // namespace lfs::vis::input
