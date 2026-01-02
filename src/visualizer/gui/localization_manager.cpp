#include "localization_manager.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace lichtfeld {

    namespace {
        constexpr const char* LANGUAGE_NAME_KEY = "_language_name";
        constexpr const char* DEFAULT_LANGUAGE = "en";
    } // namespace

    LocalizationManager& LocalizationManager::getInstance() {
        static LocalizationManager instance;
        return instance;
    }

    bool LocalizationManager::initialize(const std::string& locales_dir) {
        locales_dir_ = locales_dir;

        if (!fs::exists(locales_dir_) || !fs::is_directory(locales_dir_)) {
            LOG_ERROR("Locales directory not found: {}", locales_dir_);
            return false;
        }

        available_languages_.clear();
        language_names_.clear();

        for (const auto& entry : fs::directory_iterator(locales_dir_)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".json")
                continue;

            const std::string lang_code = entry.path().stem().string();
            std::unordered_map<std::string, std::string> test_strings;

            if (!parseLocaleFile(lfs::core::path_to_utf8(entry.path()), test_strings))
                continue;

            available_languages_.push_back(lang_code);

            // Extract language name from parsed strings
            const auto it = test_strings.find(LANGUAGE_NAME_KEY);
            language_names_[lang_code] = (it != test_strings.end()) ? it->second : lang_code;
        }

        if (available_languages_.empty()) {
            LOG_ERROR("No valid locale files found in: {}", locales_dir_);
            return false;
        }

        LOG_INFO("Found {} language(s)", available_languages_.size());

        const bool has_default = std::find(available_languages_.begin(),
                                           available_languages_.end(),
                                           DEFAULT_LANGUAGE) != available_languages_.end();
        return setLanguage(has_default ? DEFAULT_LANGUAGE : available_languages_[0]);
    }

    const char* LocalizationManager::get(std::string_view key) const {
        const std::string key_str(key);
        const auto it = current_strings_.find(key_str);
        if (it != current_strings_.end()) {
            return it->second.c_str();
        }
        LOG_WARN("Missing localization key: {}", key_str);
        return key.data();
    }

    std::vector<std::string> LocalizationManager::getAvailableLanguages() const {
        return available_languages_;
    }

    std::vector<std::string> LocalizationManager::getAvailableLanguageNames() const {
        std::vector<std::string> names;
        names.reserve(available_languages_.size());
        for (const auto& lang : available_languages_) {
            const auto it = language_names_.find(lang);
            names.push_back(it != language_names_.end() ? it->second : lang);
        }
        return names;
    }

    bool LocalizationManager::setLanguage(const std::string& language_code) {
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
        LOG_INFO("Language set to: {}", language_code);
        return true;
    }

    std::string LocalizationManager::getCurrentLanguageName() const {
        const auto it = language_names_.find(current_language_);
        return (it != language_names_.end()) ? it->second : current_language_;
    }

    bool LocalizationManager::reload() {
        return !current_language_.empty() && loadLanguage(current_language_);
    }

    bool LocalizationManager::loadLanguage(const std::string& language_code) {
        const std::string filepath = locales_dir_ + "/" + language_code + ".json";

        if (!fs::exists(filepath)) {
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

} // namespace lichtfeld
