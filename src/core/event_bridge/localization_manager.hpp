/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "event_bridge.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <format>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::event {

    class LFS_BRIDGE_API LocalizationManager {
    public:
        using PluginCatalogToken = std::uint64_t;
        using TranslationMap = std::unordered_map<std::string, std::string>;

        static LocalizationManager& getInstance();

        bool initialize(const std::string& locales_dir);
        void reset();
        // Returns a pointer into thread-local storage. The pointer remains valid
        // until a subsequent get()/getEnglishFallback() call on the same thread
        // reuses that ring-buffer slot; copy it when retaining the value.
        const char* get(std::string_view key) const;
        bool hasKey(std::string_view key) const;
        const char* getEnglishFallback(std::string_view key) const;
        const char* operator[](std::string_view key) const { return get(key); }

        std::vector<std::string> getAvailableLanguages() const;
        std::vector<std::string> getAvailableLanguageNames() const;
        bool setLanguage(const std::string& language_code);
        std::string getCurrentLanguage() const;
        [[nodiscard]] std::uint64_t getCurrentLanguageGeneration() const noexcept {
            return language_generation_.load(std::memory_order_acquire);
        }
        std::string getCurrentLanguageName() const;
        bool reload();

        void setOverride(const std::string& key, const std::string& value);
        void clearOverride(const std::string& key);
        void clearAllOverrides();
        bool hasOverride(const std::string& key) const;

        PluginCatalogToken registerPluginCatalog(std::string_view owner_id,
                                                 std::string_view language_code,
                                                 const TranslationMap& entries,
                                                 std::string* error = nullptr);
        bool unregisterPluginCatalog(PluginCatalogToken token);
        std::size_t unregisterPluginCatalogs(std::string_view owner_id);

    private:
        LocalizationManager() = default;
        ~LocalizationManager() = default;
        LocalizationManager(const LocalizationManager&) = delete;
        LocalizationManager& operator=(const LocalizationManager&) = delete;

        bool loadLanguage(const std::string& language_code);
        bool parseLocaleFile(const std::string& filepath,
                             std::unordered_map<std::string, std::string>& strings) const;
        bool unregisterPluginCatalogLocked(PluginCatalogToken token);

        mutable std::mutex mutex_;
        std::string locales_dir_;
        std::string current_language_;
        std::unordered_map<std::string, std::string> current_strings_;
        std::unordered_map<std::string, std::string> fallback_strings_;
        mutable std::unordered_set<std::string> warned_missing_keys_;
        std::vector<std::string> available_languages_;
        std::unordered_map<std::string, std::string> language_names_;
        mutable std::unordered_map<std::string, std::string> overrides_;

        struct PluginStringEntry {
            PluginCatalogToken token;
            std::string value;
        };

        struct PluginCatalogRecord {
            std::string owner_id;
            std::string language_code;
            std::vector<std::string> keys;
        };

        std::unordered_map<std::string,
                           std::unordered_map<std::string, PluginStringEntry>>
            plugin_strings_by_language_;
        std::unordered_map<PluginCatalogToken, PluginCatalogRecord> plugin_catalogs_;
        PluginCatalogToken next_plugin_catalog_token_ = 1;
        std::atomic<std::uint64_t> language_generation_{0};
    };

    template <typename... Args>
    [[nodiscard]] inline std::string formatLocalized(const std::string_view key, Args&&... args) {
        const char* const localized = LocalizationManager::getInstance().get(key);
        try {
            return std::vformat(localized, std::make_format_args(args...));
        } catch (const std::format_error&) {
            const char* const fallback = LocalizationManager::getInstance().getEnglishFallback(key);
            try {
                return std::vformat(fallback, std::make_format_args(args...));
            } catch (const std::format_error&) {
                return fallback;
            }
        }
    }

#define LOC(key)       lfs::event::LocalizationManager::getInstance().get(key)
#define LOCF(key, ...) lfs::event::formatLocalized(key, __VA_ARGS__)

} // namespace lfs::event
