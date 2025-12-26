#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lichtfeld {

// Manages GUI localization with runtime language switching
class LocalizationManager {
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
    std::vector<std::string> available_languages_;
    std::unordered_map<std::string, std::string> language_names_;
};

#define LOC(key) lichtfeld::LocalizationManager::getInstance().get(key)

} // namespace lichtfeld
