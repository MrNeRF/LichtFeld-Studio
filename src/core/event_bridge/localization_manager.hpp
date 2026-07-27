/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "event_bridge.hpp"

#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::event {

    class LFS_BRIDGE_API LocalizationManager {
    public:
        static LocalizationManager& getInstance();

        bool initialize(const std::string& locales_dir);
        const char* get(std::string_view key) const;
        const char* operator[](std::string_view key) const { return get(key); }

        std::vector<std::string> getAvailableLanguages() const;
        std::vector<std::string> getAvailableLanguageNames() const;
        bool setLanguage(const std::string& language_code);
        const std::string& getCurrentLanguage() const { return current_language_; }
        std::string getCurrentLanguageName() const;
        bool reload();

        void setOverride(const std::string& key, const std::string& value);
        void clearOverride(const std::string& key);
        void clearAllOverrides();
        bool hasOverride(const std::string& key) const;

    private:
        LocalizationManager() = default;
        ~LocalizationManager() = default;
        LocalizationManager(const LocalizationManager&) = delete;
        LocalizationManager& operator=(const LocalizationManager&) = delete;

        bool loadLanguage(const std::string& language_code);
        bool parseLocaleFile(const std::string& filepath,
                             std::unordered_map<std::string, std::string>& strings) const;

        std::string locales_dir_;
        std::string current_language_;
        std::unordered_map<std::string, std::string> current_strings_;
        std::unordered_map<std::string, std::string> fallback_strings_;
        mutable std::unordered_set<std::string> warned_missing_keys_;
        std::vector<std::string> available_languages_;
        std::unordered_map<std::string, std::string> language_names_;
        mutable std::unordered_map<std::string, std::string> overrides_;
    };

#define LOC(key) lfs::event::LocalizationManager::getInstance().get(key)

} // namespace lfs::event
