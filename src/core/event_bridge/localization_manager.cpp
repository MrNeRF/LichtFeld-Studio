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

        const fs::path index_path = fs::path(locales_dir_) / "index.json";
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
        return overrides_.find(key_str) != overrides_.end() ||
               current_strings_.find(key_str) != current_strings_.end() ||
               fallback_strings_.find(key_str) != fallback_strings_.end();
    }

    const char* LocalizationManager::getEnglishFallback(std::string_view key) const {
        thread_local std::array<std::string, 8> result_buffers;
        thread_local size_t next_result_buffer = 0;
        std::string& result = result_buffers[next_result_buffer++ % result_buffers.size()];

        const std::lock_guard lock(mutex_);
        const auto fallback_it = fallback_strings_.find(std::string(key));
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
