/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "localization_manager.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace lfs::event {

    namespace {
        constexpr const char* LANGUAGE_NAME_KEY = "_language_name";
        constexpr const char* DEFAULT_LANGUAGE = "en";

        constexpr std::size_t MAX_PLUGIN_OWNER_LENGTH = 128;
        constexpr std::size_t MAX_PLUGIN_KEY_LENGTH = 256;
        constexpr std::size_t MAX_PLUGIN_VALUE_LENGTH = 16 * 1024;
        constexpr std::size_t MAX_PLUGIN_CATALOG_ENTRIES = 4096;

        bool isLowerAsciiAlpha(const char value) {
            return value >= 'a' && value <= 'z';
        }

        bool isAsciiDigit(const char value) {
            return value >= '0' && value <= '9';
        }

        bool isValidPluginOwner(const std::string_view owner) {
            if (owner.empty() || owner.size() > MAX_PLUGIN_OWNER_LENGTH)
                return false;

            bool previous_separator = true;
            for (const char value : owner) {
                const bool separator = value == '-';
                if ((!isLowerAsciiAlpha(value) && !isAsciiDigit(value) && !separator) ||
                    (separator && previous_separator))
                    return false;
                previous_separator = separator;
            }
            return !previous_separator;
        }

        bool isValidLanguageCode(const std::string_view language) {
            if (language.size() < 2 || language.size() > 35 || language.back() == '-')
                return false;

            std::size_t segment_start = 0;
            std::size_t segment_index = 0;
            while (segment_start < language.size()) {
                const std::size_t separator = language.find('-', segment_start);
                const std::size_t segment_end =
                    separator == std::string_view::npos ? language.size() : separator;
                const std::size_t segment_length = segment_end - segment_start;
                if ((segment_index == 0 && (segment_length < 2 || segment_length > 3)) ||
                    (segment_index > 0 && (segment_length < 2 || segment_length > 8)))
                    return false;

                for (std::size_t i = segment_start; i < segment_end; ++i) {
                    const char value = language[i];
                    if (segment_index == 0) {
                        if (!isLowerAsciiAlpha(value))
                            return false;
                    } else if (!isLowerAsciiAlpha(value) && !isAsciiDigit(value)) {
                        return false;
                    }
                }

                if (separator == std::string_view::npos)
                    break;
                segment_start = separator + 1;
                ++segment_index;
            }
            return true;
        }

        bool isValidRelativePluginKey(const std::string_view key) {
            if (key.empty() || key.size() > MAX_PLUGIN_KEY_LENGTH || key.starts_with("plugins."))
                return false;

            bool previous_separator = true;
            for (const char value : key) {
                const bool separator = value == '.';
                const bool valid_character = isLowerAsciiAlpha(value) || isAsciiDigit(value) ||
                                             value == '_' || value == '-' || separator;
                if (!valid_character || (separator && previous_separator))
                    return false;
                previous_separator = separator;
            }
            return !previous_separator;
        }
    } // namespace

    LocalizationManager& LocalizationManager::getInstance() {
        static LocalizationManager instance;
        return instance;
    }

    bool LocalizationManager::initialize(const std::string& locales_dir) {
        const std::lock_guard lock(mutex_);
        locales_dir_ = locales_dir;

        std::error_code ec;
        if (!fs::exists(locales_dir_, ec) || ec || !fs::is_directory(locales_dir_, ec) || ec) {
            LOG_ERROR("Locales directory not found: {}", locales_dir_);
            return false;
        }

        available_languages_.clear();
        language_names_.clear();
        fallback_strings_.clear();
        warned_missing_keys_.clear();

        const fs::path index_path = fs::path(locales_dir_).parent_path() / "locale_index.json";
        std::ifstream index_file;
        if (!lfs::core::open_file_for_read(index_path, index_file)) {
            LOG_ERROR("Locale index not found: {}", lfs::core::path_to_utf8(index_path));
            return false;
        }

        try {
            json index;
            index_file >> index;
            const auto languages = index.at("languages");
            if (!languages.is_array())
                throw std::runtime_error("languages is not an array");
            for (const auto& entry : languages) {
                const std::string code = entry.at("code").get<std::string>();
                available_languages_.push_back(code);
                language_names_[code] = entry.value("name", code);
            }
        } catch (const std::exception& error) {
            LOG_ERROR("Invalid locale index '{}': {}", lfs::core::path_to_utf8(index_path), error.what());
            return false;
        }

        const fs::path fallback_path = fs::path(locales_dir_) / (std::string(DEFAULT_LANGUAGE) + ".json");
        if (!parseLocaleFile(lfs::core::path_to_utf8(fallback_path), fallback_strings_))
            return false;

        if (available_languages_.empty()) {
            LOG_ERROR("No valid locale files found in: {}", locales_dir_);
            return false;
        }

        LOG_INFO("Found {} language(s)", available_languages_.size());

        const bool has_default = std::find(available_languages_.begin(),
                                           available_languages_.end(),
                                           DEFAULT_LANGUAGE) != available_languages_.end();
        const std::string initial_language = has_default ? DEFAULT_LANGUAGE : available_languages_[0];
        if (!loadLanguage(initial_language))
            return false;

        current_language_ = initial_language;
        language_generation_.fetch_add(1, std::memory_order_release);
        LOG_INFO("Language set to: {}", initial_language);
        return true;
    }

    void LocalizationManager::reset() {
        const std::lock_guard lock(mutex_);
        locales_dir_.clear();
        current_language_.clear();
        current_strings_.clear();
        fallback_strings_.clear();
        warned_missing_keys_.clear();
        available_languages_.clear();
        language_names_.clear();
        overrides_.clear();
        plugin_strings_by_language_.clear();
        plugin_catalogs_.clear();
        language_generation_.fetch_add(1, std::memory_order_release);
    }

    const char* LocalizationManager::get(std::string_view key) const {
        thread_local std::array<std::string, 8> result_buffers;
        thread_local size_t next_result_buffer = 0;
        std::string& result = result_buffers[next_result_buffer++ % result_buffers.size()];

        const std::lock_guard lock(mutex_);
        const std::string key_str(key);

        const auto override_it = overrides_.find(key_str);
        if (override_it != overrides_.end()) {
            result = override_it->second;
            return result.c_str();
        }

        const auto plugin_language_it = plugin_strings_by_language_.find(current_language_);
        if (plugin_language_it != plugin_strings_by_language_.end()) {
            const auto plugin_string_it = plugin_language_it->second.find(key_str);
            if (plugin_string_it != plugin_language_it->second.end()) {
                result = plugin_string_it->second.value;
                return result.c_str();
            }
        }

        const auto plugin_fallback_language_it =
            plugin_strings_by_language_.find(DEFAULT_LANGUAGE);
        if (plugin_fallback_language_it != plugin_strings_by_language_.end()) {
            const auto plugin_string_it = plugin_fallback_language_it->second.find(key_str);
            if (plugin_string_it != plugin_fallback_language_it->second.end()) {
                if (current_language_ != DEFAULT_LANGUAGE &&
                    warned_missing_keys_.insert(key_str).second)
                    LOG_WARN("Missing plugin localization key '{}' in '{}'; using English fallback",
                             key_str, current_language_);
                result = plugin_string_it->second.value;
                return result.c_str();
            }
        }

        const auto it = current_strings_.find(key_str);
        if (it != current_strings_.end()) {
            result = it->second;
            return result.c_str();
        }

        const auto fallback_it = fallback_strings_.find(key_str);
        if (fallback_it != fallback_strings_.end()) {
            if (current_language_ != DEFAULT_LANGUAGE && warned_missing_keys_.insert(key_str).second)
                LOG_WARN("Missing localization key '{}' in '{}'; using English fallback",
                         key_str, current_language_);
            result = fallback_it->second;
            return result.c_str();
        }

        if (warned_missing_keys_.insert(key_str).second)
            LOG_WARN("Missing localization key: {}", key_str);
        result.assign(key);
        return result.c_str();
    }

    bool LocalizationManager::hasKey(std::string_view key) const {
        const std::lock_guard lock(mutex_);
        const std::string key_str(key);
        const auto plugin_language_it = plugin_strings_by_language_.find(current_language_);
        const bool has_plugin_language =
            plugin_language_it != plugin_strings_by_language_.end() &&
            plugin_language_it->second.find(key_str) != plugin_language_it->second.end();
        const auto plugin_fallback_it = plugin_strings_by_language_.find(DEFAULT_LANGUAGE);
        const bool has_plugin_fallback =
            plugin_fallback_it != plugin_strings_by_language_.end() &&
            plugin_fallback_it->second.find(key_str) != plugin_fallback_it->second.end();
        return overrides_.find(key_str) != overrides_.end() ||
               has_plugin_language ||
               has_plugin_fallback ||
               current_strings_.find(key_str) != current_strings_.end() ||
               fallback_strings_.find(key_str) != fallback_strings_.end();
    }

    const char* LocalizationManager::getEnglishFallback(std::string_view key) const {
        thread_local std::array<std::string, 8> result_buffers;
        thread_local size_t next_result_buffer = 0;
        std::string& result = result_buffers[next_result_buffer++ % result_buffers.size()];

        const std::lock_guard lock(mutex_);
        const std::string key_str(key);
        const auto plugin_fallback_language_it =
            plugin_strings_by_language_.find(DEFAULT_LANGUAGE);
        if (plugin_fallback_language_it != plugin_strings_by_language_.end()) {
            const auto plugin_string_it = plugin_fallback_language_it->second.find(key_str);
            if (plugin_string_it != plugin_fallback_language_it->second.end()) {
                result = plugin_string_it->second.value;
                return result.c_str();
            }
        }

        const auto fallback_it = fallback_strings_.find(key_str);
        if (fallback_it != fallback_strings_.end()) {
            result = fallback_it->second;
            return result.c_str();
        }

        result.assign(key);
        return result.c_str();
    }

    void LocalizationManager::setOverride(const std::string& key, const std::string& value) {
        const std::lock_guard lock(mutex_);
        overrides_[key] = value;
    }

    void LocalizationManager::clearOverride(const std::string& key) {
        const std::lock_guard lock(mutex_);
        overrides_.erase(key);
    }

    void LocalizationManager::clearAllOverrides() {
        const std::lock_guard lock(mutex_);
        overrides_.clear();
    }

    bool LocalizationManager::hasOverride(const std::string& key) const {
        const std::lock_guard lock(mutex_);
        return overrides_.find(key) != overrides_.end();
    }

    LocalizationManager::PluginCatalogToken LocalizationManager::registerPluginCatalog(
        const std::string_view owner_id,
        const std::string_view language_code,
        const TranslationMap& entries,
        std::string* const error) {
        const auto fail = [&](const std::string& message) {
            if (error)
                *error = message;
            return PluginCatalogToken{0};
        };

        if (!isValidPluginOwner(owner_id))
            return fail("Plugin localization owner must match [a-z0-9]+(?:-[a-z0-9]+)*");
        if (!isValidLanguageCode(language_code))
            return fail("Plugin localization language code is invalid or not lowercase");
        if (entries.empty())
            return fail("Plugin localization catalog must not be empty");
        if (entries.size() > MAX_PLUGIN_CATALOG_ENTRIES)
            return fail("Plugin localization catalog exceeds the 4096-entry limit");

        const std::string namespace_prefix = "plugins." + std::string(owner_id) + ".";
        std::vector<std::pair<std::string, std::string>> normalized_entries;
        normalized_entries.reserve(entries.size());
        for (const auto& [key, value] : entries) {
            if (!isValidRelativePluginKey(key))
                return fail("Invalid relative plugin localization key: " + key);
            if (value.empty())
                return fail("Plugin localization value must not be empty: " + key);
            if (value.size() > MAX_PLUGIN_VALUE_LENGTH)
                return fail("Plugin localization value exceeds the 16 KiB limit: " + key);
            normalized_entries.emplace_back(namespace_prefix + key, value);
        }

        const std::lock_guard lock(mutex_);
        auto& language_entries = plugin_strings_by_language_[std::string(language_code)];
        for (const auto& [key, _] : normalized_entries) {
            if (language_entries.find(key) != language_entries.end())
                return fail("Plugin localization key is already registered: " + key);
        }

        while (next_plugin_catalog_token_ == 0 ||
               plugin_catalogs_.contains(next_plugin_catalog_token_))
            ++next_plugin_catalog_token_;
        const PluginCatalogToken token = next_plugin_catalog_token_++;

        PluginCatalogRecord record{
            .owner_id = std::string(owner_id),
            .language_code = std::string(language_code),
        };
        record.keys.reserve(normalized_entries.size());
        for (auto& [key, value] : normalized_entries) {
            record.keys.push_back(key);
            warned_missing_keys_.erase(key);
            language_entries.emplace(std::move(key), PluginStringEntry{token, std::move(value)});
        }
        plugin_catalogs_.emplace(token, std::move(record));
        if (error)
            error->clear();
        return token;
    }

    bool LocalizationManager::unregisterPluginCatalog(const PluginCatalogToken token) {
        if (token == 0)
            return false;

        const std::lock_guard lock(mutex_);
        return unregisterPluginCatalogLocked(token);
    }

    bool LocalizationManager::unregisterPluginCatalogLocked(const PluginCatalogToken token) {
        const auto catalog_it = plugin_catalogs_.find(token);
        if (catalog_it == plugin_catalogs_.end())
            return false;

        const auto language_it =
            plugin_strings_by_language_.find(catalog_it->second.language_code);
        if (language_it != plugin_strings_by_language_.end()) {
            for (const auto& key : catalog_it->second.keys) {
                const auto entry_it = language_it->second.find(key);
                if (entry_it != language_it->second.end() && entry_it->second.token == token)
                    language_it->second.erase(entry_it);
                warned_missing_keys_.erase(key);
            }
            if (language_it->second.empty())
                plugin_strings_by_language_.erase(language_it);
        }

        plugin_catalogs_.erase(catalog_it);
        return true;
    }

    std::size_t LocalizationManager::unregisterPluginCatalogs(const std::string_view owner_id) {
        const std::lock_guard lock(mutex_);
        std::vector<PluginCatalogToken> tokens;
        for (const auto& [token, catalog] : plugin_catalogs_) {
            if (catalog.owner_id == owner_id)
                tokens.push_back(token);
        }

        for (const PluginCatalogToken token : tokens)
            unregisterPluginCatalogLocked(token);
        return tokens.size();
    }

    std::vector<std::string> LocalizationManager::getAvailableLanguages() const {
        const std::lock_guard lock(mutex_);
        return available_languages_;
    }

    std::vector<std::string> LocalizationManager::getAvailableLanguageNames() const {
        const std::lock_guard lock(mutex_);
        std::vector<std::string> names;
        names.reserve(available_languages_.size());
        for (const auto& lang : available_languages_) {
            const auto it = language_names_.find(lang);
            names.push_back(it != language_names_.end() ? it->second : lang);
        }
        return names;
    }

    bool LocalizationManager::setLanguage(const std::string& language_code) {
        const std::lock_guard lock(mutex_);
        const bool available = std::find(available_languages_.begin(),
                                         available_languages_.end(),
                                         language_code) != available_languages_.end();
        if (!available) {
            LOG_ERROR("Language not available: {}", language_code);
            return false;
        }

        if (!loadLanguage(language_code))
            return false;

        current_language_ = language_code;
        language_generation_.fetch_add(1, std::memory_order_release);
        LOG_INFO("Language set to: {}", language_code);
        return true;
    }

    std::string LocalizationManager::getCurrentLanguageName() const {
        const std::lock_guard lock(mutex_);
        const auto it = language_names_.find(current_language_);
        return (it != language_names_.end()) ? it->second : current_language_;
    }

    std::string LocalizationManager::getCurrentLanguage() const {
        const std::lock_guard lock(mutex_);
        return current_language_;
    }

    bool LocalizationManager::reload() {
        const std::string language_code = getCurrentLanguage();
        return !language_code.empty() && setLanguage(language_code);
    }

    bool LocalizationManager::loadLanguage(const std::string& language_code) {
        if (language_code == DEFAULT_LANGUAGE && !fallback_strings_.empty()) {
            current_strings_.clear();
            LOG_INFO("Loaded {} strings for language: {}", fallback_strings_.size(), language_code);
            return true;
        }

        const std::string filepath = locales_dir_ + "/" + language_code + ".json";

        std::error_code ec;
        if (!fs::exists(filepath, ec) || ec) {
            LOG_ERROR("Locale file not found: {}", filepath);
            return false;
        }

        std::unordered_map<std::string, std::string> new_strings;
        if (!parseLocaleFile(filepath, new_strings))
            return false;

        current_strings_ = std::move(new_strings);
        LOG_INFO("Loaded {} strings for language: {}", current_strings_.size(), language_code);
        return true;
    }

    bool LocalizationManager::parseLocaleFile(const std::string& filepath,
                                              std::unordered_map<std::string, std::string>& strings) const {
        std::ifstream file;
        if (!lfs::core::open_file_for_read(lfs::core::utf8_to_path(filepath), file)) {
            LOG_ERROR("Failed to open locale file: {}", filepath);
            return false;
        }

        try {
            json j;
            file >> j;

            std::function<void(const json&, const std::string&)> parse_recursive;
            parse_recursive = [&](const json& obj, const std::string& prefix) {
                for (auto it = obj.begin(); it != obj.end(); ++it) {
                    const std::string key = prefix.empty() ? it.key() : prefix + "." + it.key();
                    if (it.value().is_string()) {
                        strings[key] = it.value().get<std::string>();
                    } else if (it.value().is_object()) {
                        parse_recursive(it.value(), key);
                    }
                }
            };

            parse_recursive(j, "");
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to parse locale file {}: {}", filepath, e.what());
            return false;
        }
    }

} // namespace lfs::event
