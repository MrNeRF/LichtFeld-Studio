/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "preferences.hpp"

#include "core/environment.hpp"
#include "core/error.hpp"
#include "core/logger.hpp"
#include "core/user_paths.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>

namespace lfs::vis {
    namespace {
        using json = nlohmann::json;

        [[nodiscard]] bool disabled() {
            return lfs::core::environment::flag("LFS_SAFE_MODE", false);
        }

        [[nodiscard]] bool knownCameraMode(const std::string& mode) {
            return mode == "orbit" || mode == "trackball" || mode == "fpv" || mode == "drone";
        }
    } // namespace

    struct UserPreferences::Impl {
        std::mutex mutex;
        json values = json::object();
        std::optional<lfs::core::UserPaths> paths;
        std::filesystem::path loaded_path;
        bool loaded = false;
        bool writable = true;
        bool warned = false;

        void loadLocked() {
            if (disabled()) {
                values = json::object();
                paths.reset();
                loaded_path.clear();
                loaded = true;
                writable = false;
                return;
            }

            auto resolved = lfs::core::UserPaths::resolve();
            if (!resolved) {
                if (!warned) {
                    LOG_WARN("Unable to resolve user preferences path: {}",
                             lfs::format_for_developer(resolved.error()));
                    warned = true;
                }
                values = json::object();
                paths.reset();
                loaded = true;
                writable = false;
                return;
            }

            const auto path = resolved->preferencesFile();
            if (loaded && paths && loaded_path == path)
                return;

            paths = *resolved;
            loaded_path = path;
            loaded = true;
            writable = true;
            warned = false;
            values = json::object();

            std::ifstream input(path);
            if (!input)
                return;

            try {
                auto parsed = json::parse(input);
                if (!parsed.is_object())
                    throw std::runtime_error("root value is not an object");
                values = std::move(parsed);
            } catch (const std::exception& error) {
                // Windows does not allow the malformed file to be moved while
                // this reader still holds it open.
                input.close();
                const auto backup = paths->backupCorruptPreferences();
                if (!backup) {
                    writable = false;
                    LOG_WARN("Invalid preferences were not replaced because their backup failed: {}",
                             lfs::format_for_developer(backup.error()));
                } else {
                    const std::string backup_path = *backup
                                                        ? (*backup)->string()
                                                        : std::string("<no source file>");
                    LOG_WARN("Invalid preferences were backed up to '{}'; defaults will be used: {}",
                             backup_path, error.what());
                }
                warned = true;
            }
        }

        void saveLocked() {
            loadLocked();
            if (!paths || !writable)
                return;
            values["schema_version"] = 1;
            if (const auto result = paths->writePreferencesAtomically(values.dump(2) + '\n'); !result)
                LOG_WARN("Unable to save user preferences: {}",
                         lfs::format_for_developer(result.error()));
        }
    };

    UserPreferences& UserPreferences::instance() {
        static UserPreferences preferences;
        return preferences;
    }

    UserPreferences::UserPreferences() : impl_(std::make_unique<Impl>()) {}
    UserPreferences::~UserPreferences() = default;

    void UserPreferences::setThemeName(const std::string& value) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["theme"] = value;
        impl_->saveLocked();
    }
    std::string UserPreferences::themeName() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        return impl_->values.value("theme", std::string{});
    }
    void UserPreferences::setUiScale(const float value) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["ui_scale"] = value <= 0.0f ? json("auto") : json(value);
        impl_->saveLocked();
    }
    float UserPreferences::uiScale() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("ui_scale");
        if (it == impl_->values.end())
            return 0.0f;
        if (it->is_string() && it->get<std::string>() == "auto")
            return 0.0f;
        if (it->is_number()) {
            const float scale = it->get<float>();
            if (std::isfinite(scale) && scale >= 1.0f && scale <= 4.0f)
                return scale;
        }
        return 0.0f;
    }
    void UserPreferences::setLanguage(const std::string& value) {
        if (value.empty())
            return;
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["language"] = value;
        impl_->saveLocked();
    }
    std::string UserPreferences::language() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("language");
        return it != impl_->values.end() && it->is_string() ? it->get<std::string>() : std::string{};
    }
    void UserPreferences::clearLanguage() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values.erase("language");
        impl_->saveLocked();
    }
    void UserPreferences::setCameraNavigation(const std::string& value) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        if (!impl_->values.value("remember_camera_navigation", false))
            return;
        impl_->values["camera_navigation_mode"] = knownCameraMode(value) ? value : "orbit";
        impl_->saveLocked();
    }
    std::string UserPreferences::cameraNavigation() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        if (!impl_->values.value("remember_camera_navigation", false))
            return "orbit";
        const std::string mode = impl_->values.value("camera_navigation_mode", "orbit");
        return knownCameraMode(mode) ? mode : "orbit";
    }
    void UserPreferences::setRememberCameraNavigation(const bool enabled) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["remember_camera_navigation"] = enabled;
        if (!enabled)
            impl_->values.erase("camera_navigation_mode");
        impl_->saveLocked();
    }
    bool UserPreferences::rememberCameraNavigation() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        return impl_->values.value("remember_camera_navigation", false);
    }
    void UserPreferences::setCameraViewSnap(const bool enabled) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        if (!impl_->values.value("remember_camera_view_snap", false))
            return;
        impl_->values["camera_view_snap"] = enabled;
        impl_->saveLocked();
    }
    bool UserPreferences::cameraViewSnap() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        if (!impl_->values.value("remember_camera_view_snap", false))
            return false;
        return impl_->values.value("camera_view_snap", false);
    }
    void UserPreferences::setRememberCameraViewSnap(const bool enabled) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["remember_camera_view_snap"] = enabled;
        if (!enabled)
            impl_->values.erase("camera_view_snap");
        impl_->saveLocked();
    }
    bool UserPreferences::rememberCameraViewSnap() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        return impl_->values.value("remember_camera_view_snap", false);
    }

    void UserPreferences::setMcp(const McpPreferenceState& state) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["mcp"] = {
            {"enabled", state.enabled},
            {"expose_network", state.expose_network},
            {"port", std::clamp(state.port, 1, 65535)},
            {"request_logging", state.request_logging},
        };
        impl_->saveLocked();
    }

    McpPreferenceState UserPreferences::mcp() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        McpPreferenceState result;
        const auto it = impl_->values.find("mcp");
        if (it == impl_->values.end() || !it->is_object())
            return result;
        if (const auto enabled = it->find("enabled");
            enabled != it->end() && enabled->is_boolean())
            result.enabled = enabled->get<bool>();
        if (const auto expose = it->find("expose_network");
            expose != it->end() && expose->is_boolean())
            result.expose_network = expose->get<bool>();
        if (const auto port = it->find("port");
            port != it->end() && port->is_number_integer()) {
            const auto value = port->get<std::int64_t>();
            if (value >= 1 && value <= 65535)
                result.port = static_cast<int>(value);
        }
        if (const auto logging = it->find("request_logging");
            logging != it->end() && logging->is_boolean())
            result.request_logging = logging->get<bool>();
        return result;
    }

    void saveLanguagePreference(const std::string& value) { UserPreferences::instance().setLanguage(value); }
    std::string loadLanguagePreference() { return UserPreferences::instance().language(); }
    void clearLanguagePreference() { UserPreferences::instance().clearLanguage(); }
    void saveCameraNavigationPreference(const std::string& value) { UserPreferences::instance().setCameraNavigation(value); }
    std::string loadCameraNavigationPreference() { return UserPreferences::instance().cameraNavigation(); }
    void setRememberCameraNavigationPreference(const bool enabled) { UserPreferences::instance().setRememberCameraNavigation(enabled); }
    bool rememberCameraNavigationPreference() { return UserPreferences::instance().rememberCameraNavigation(); }
    void saveCameraViewSnapPreference(const bool enabled) { UserPreferences::instance().setCameraViewSnap(enabled); }
    bool loadCameraViewSnapPreference() { return UserPreferences::instance().cameraViewSnap(); }
    void setRememberCameraViewSnapPreference(const bool enabled) { UserPreferences::instance().setRememberCameraViewSnap(enabled); }
    bool rememberCameraViewSnapPreference() { return UserPreferences::instance().rememberCameraViewSnap(); }
    void saveMcpPreferences(const McpPreferenceState& state) { UserPreferences::instance().setMcp(state); }
    McpPreferenceState loadMcpPreferences() { return UserPreferences::instance().mcp(); }

} // namespace lfs::vis
