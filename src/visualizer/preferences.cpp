/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "preferences.hpp"

#include "core/environment.hpp"
#include "core/error.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/user_paths.hpp"
#include "rendering/scene_upscaler_registry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace lfs::vis {
    namespace {
        using json = nlohmann::json;

        [[nodiscard]] bool disabled() {
            return lfs::core::environment::flag("LFS_SAFE_MODE", false);
        }

        [[nodiscard]] bool knownCameraMode(const std::string& mode) {
            return mode == "orbit" || mode == "trackball" || mode == "fpv" || mode == "drone";
        }

        [[nodiscard]] lfs::Error workingDirectoryError(
            const lfs::ErrorCode code,
            std::string user_message,
            std::string detail,
            const std::filesystem::path& path = {}) {
            lfs::SmallFields fields;
            if (!path.empty())
                fields.add("path", lfs::core::path_to_utf8(path));
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .user_message = std::move(user_message),
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
            });
        }

        [[nodiscard]] std::optional<std::filesystem::path> preferenceHomeDirectory() {
#ifdef _WIN32
            if (const auto value = lfs::core::environment::value("USERPROFILE"))
                return lfs::core::utf8_to_path(std::string(*value));
            if (const auto value = lfs::core::environment::value("HOME"))
                return lfs::core::utf8_to_path(std::string(*value));
#else
            if (const auto value = lfs::core::environment::value("HOME"))
                return lfs::core::utf8_to_path(std::string(*value));
#endif
            return std::nullopt;
        }

        [[nodiscard]] std::filesystem::path expandLeadingTilde(
            const std::filesystem::path& candidate) {
            const auto text = lfs::core::path_to_utf8(candidate);
            const bool tilde_home =
                text == "~" || (text.size() >= 2 && text.front() == '~' &&
                                (text[1] == '/' || text[1] == '\\'));
            if (!tilde_home)
                return candidate;
            const auto home = preferenceHomeDirectory();
            if (!home || home->empty())
                return candidate;
            if (text.size() <= 2)
                return *home;
            return *home / lfs::core::utf8_to_path(text.substr(2));
        }

        [[nodiscard]] lfs::Result<std::filesystem::path> validateWritableDirectory(
            const std::filesystem::path& candidate,
            const std::string& preference_key,
            const std::string& display_name,
            const std::string& probe_prefix) {
            if (candidate.empty()) {
                return workingDirectoryError(
                    lfs::ErrorCode::InvalidArgument,
                    display_name + " path is empty.",
                    preference_key + " is empty");
            }
            const auto expanded = expandLeadingTilde(candidate);
            if (!expanded.is_absolute()) {
                return workingDirectoryError(
                    lfs::ErrorCode::InvalidArgument,
                    display_name + " path must be absolute.",
                    preference_key + " is not an absolute path",
                    candidate);
            }
            std::error_code error;
            auto absolute = std::filesystem::absolute(expanded, error);
            if (error || absolute.empty()) {
                return workingDirectoryError(
                    lfs::ErrorCode::InvalidArgument,
                    display_name + " path could not be resolved to an absolute path.",
                    error ? error.message() : "absolute() returned an empty path",
                    expanded);
            }
            absolute = absolute.lexically_normal();
            std::filesystem::create_directories(absolute, error);
            if (error) {
                return workingDirectoryError(
                    lfs::ErrorCode::PermissionDenied,
                    display_name + " could not be created.",
                    error.message(),
                    absolute);
            }
            if (!std::filesystem::is_directory(absolute, error) || error) {
                return workingDirectoryError(
                    lfs::ErrorCode::InvalidArgument,
                    display_name + " path is not a directory.",
                    error ? error.message() : "path exists but is not a directory",
                    absolute);
            }
#ifdef _WIN32
            const auto pid = static_cast<std::uint64_t>(_getpid());
#else
            const auto pid = static_cast<std::uint64_t>(::getpid());
#endif
            const auto probe =
                absolute /
                (probe_prefix + std::to_string(pid) + "-" +
                 std::to_string(
                     std::chrono::steady_clock::now().time_since_epoch().count()));
            {
                std::ofstream output(probe, std::ios::binary | std::ios::trunc);
                if (!output) {
                    return workingDirectoryError(
                        lfs::ErrorCode::PermissionDenied,
                        display_name + " is not writable.",
                        "write probe could not be created",
                        absolute);
                }
                output.put('x');
                output.flush();
                if (!output) {
                    std::error_code ignored;
                    std::filesystem::remove(probe, ignored);
                    return workingDirectoryError(
                        lfs::ErrorCode::PermissionDenied,
                        display_name + " is not writable.",
                        "write probe could not be written",
                        absolute);
                }
            }
            std::filesystem::remove(probe, error);
            if (error) {
                return workingDirectoryError(
                    lfs::ErrorCode::PermissionDenied,
                    display_name + " is not writable.",
                    error.message(),
                    absolute);
            }
            return absolute;
        }

        [[nodiscard]] lfs::Result<std::filesystem::path> validateWritableWorkingDirectory(
            const std::filesystem::path& candidate) {
            return validateWritableDirectory(
                candidate,
                "working_directory",
                "The working folder",
                ".lfs-write-probe-");
        }

        [[nodiscard]] lfs::Result<std::filesystem::path> validateWritableAssetManagerDirectory(
            const std::filesystem::path& candidate) {
            return validateWritableDirectory(
                candidate,
                "asset_manager_directory",
                "The Asset Manager folder",
                ".lfs-asset-write-probe-");
        }

        [[nodiscard]] std::filesystem::path defaultWorkingDirectoryPath() {
            const auto resolved = lfs::core::UserPaths::resolve();
            if (!resolved)
                return {};
            return resolved->rootDir();
        }

        [[nodiscard]] std::filesystem::path defaultAssetManagerDirectoryPath() {
            const auto resolved = lfs::core::UserPaths::resolve();
            if (!resolved)
                return {};
            return resolved->rootDir() / "assets";
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
    void UserPreferences::setSceneGraphSelectionMarkers(const bool enabled) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["scene_graph_selection_markers"] = enabled;
        impl_->saveLocked();
    }
    bool UserPreferences::sceneGraphSelectionMarkers() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        return impl_->values.value("scene_graph_selection_markers", false);
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

    void UserPreferences::setSceneUpscaler(const std::string& backend_id,
                                           const std::string& preset_id) {
        const auto backend = sceneUpscalerBackendFromId(backend_id)
                                 .value_or(SceneUpscalerBackend::Native);
        const auto preset = lfs::vis::sceneUpscalerPreset(backend, preset_id)
                                .value_or(defaultSceneUpscalerPreset(backend));
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["scene_upscaler"] = std::string(sceneUpscalerBackendId(backend));
        auto& presets = impl_->values["scene_upscaler_presets"];
        if (!presets.is_object())
            presets = json::object();
        presets[std::string(sceneUpscalerBackendId(backend))] = std::string(preset.id);
        impl_->saveLocked();
    }

    void UserPreferences::clearSceneUpscaler() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values.erase("scene_upscaler");
        impl_->values.erase("scene_upscaler_presets");
        impl_->saveLocked();
    }

    std::string UserPreferences::sceneUpscaler() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("scene_upscaler");
        if (it == impl_->values.end() || !it->is_string())
            return "native";
        const std::string id = it->get<std::string>();
        return sceneUpscalerBackendFromId(id).has_value() ? id : "native";
    }

    std::string UserPreferences::sceneUpscalerPreset(const std::string& backend_id) {
        const auto backend = sceneUpscalerBackendFromId(backend_id)
                                 .value_or(SceneUpscalerBackend::Native);
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto presets = impl_->values.find("scene_upscaler_presets");
        if (presets != impl_->values.end() && presets->is_object()) {
            const auto preset = presets->find(std::string(sceneUpscalerBackendId(backend)));
            if (preset != presets->end() && preset->is_string()) {
                const std::string id = preset->get<std::string>();
                if (lfs::vis::sceneUpscalerPreset(backend, id).has_value())
                    return id;
            }
        }
        return std::string(defaultSceneUpscalerPreset(backend).id);
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
    void saveSceneGraphSelectionMarkersPreference(const bool enabled) {
        UserPreferences::instance().setSceneGraphSelectionMarkers(enabled);
    }
    bool loadSceneGraphSelectionMarkersPreference() {
        return UserPreferences::instance().sceneGraphSelectionMarkers();
    }
    void saveMcpPreferences(const McpPreferenceState& state) { UserPreferences::instance().setMcp(state); }
    McpPreferenceState loadMcpPreferences() { return UserPreferences::instance().mcp(); }
    void saveSceneUpscalerPreference(const std::string& backend_id,
                                     const std::string& preset_id) {
        UserPreferences::instance().setSceneUpscaler(backend_id, preset_id);
    }
    void clearSceneUpscalerPreference() { UserPreferences::instance().clearSceneUpscaler(); }
    std::string loadSceneUpscalerPreference() {
        return UserPreferences::instance().sceneUpscaler();
    }
    std::string loadSceneUpscalerPresetPreference(const std::string& backend_id) {
        return UserPreferences::instance().sceneUpscalerPreset(backend_id);
    }

    lfs::Status UserPreferences::setWorkingDirectory(const std::filesystem::path& path) {
        auto resolved = validateWritableWorkingDirectory(path);
        if (!resolved)
            return lfs::Status::failure(std::move(resolved).error());
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["working_directory"] = lfs::core::path_to_utf8(*resolved);
        impl_->saveLocked();
        return {};
    }

    std::filesystem::path UserPreferences::workingDirectory() {
        const auto raw = workingDirectoryPreference();
        if (raw.empty())
            return defaultWorkingDirectoryPath();
        return raw;
    }

    std::filesystem::path UserPreferences::workingDirectoryPreference() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("working_directory");
        if (it == impl_->values.end() || !it->is_string())
            return {};
        const std::string stored = it->get<std::string>();
        if (stored.empty())
            return {};
        return lfs::core::utf8_to_path(stored);
    }

    void UserPreferences::clearWorkingDirectory() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["working_directory"] = "";
        impl_->saveLocked();
    }

    lfs::Status UserPreferences::setAssetManagerDirectory(
        const std::filesystem::path& path) {
        auto resolved = validateWritableAssetManagerDirectory(path);
        if (!resolved)
            return lfs::Status::failure(std::move(resolved).error());
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["asset_manager_directory"] = lfs::core::path_to_utf8(*resolved);
        impl_->saveLocked();
        return {};
    }

    std::filesystem::path UserPreferences::assetManagerDirectory() {
        const auto raw = assetManagerDirectoryPreference();
        return raw.empty() ? defaultAssetManagerDirectoryPath() : raw;
    }

    std::filesystem::path UserPreferences::assetManagerDirectoryPreference() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("asset_manager_directory");
        if (it == impl_->values.end() || !it->is_string())
            return {};
        const std::string stored = it->get<std::string>();
        return stored.empty() ? std::filesystem::path{} : lfs::core::utf8_to_path(stored);
    }

    void UserPreferences::clearAssetManagerDirectory() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["asset_manager_directory"] = "";
        impl_->saveLocked();
    }

    lfs::Status setWorkingDirectoryPreference(const std::filesystem::path& path) {
        return UserPreferences::instance().setWorkingDirectory(path);
    }
    std::filesystem::path loadWorkingDirectoryPreference() {
        return UserPreferences::instance().workingDirectory();
    }
    std::filesystem::path workingDirectoryPreferenceRaw() {
        return UserPreferences::instance().workingDirectoryPreference();
    }
    void clearWorkingDirectoryPreference() {
        UserPreferences::instance().clearWorkingDirectory();
    }
    std::filesystem::path defaultWorkingDirectory() {
        return defaultWorkingDirectoryPath();
    }
    std::filesystem::path tempProjectDirectoryPreference() {
        const auto root = UserPreferences::instance().workingDirectory();
        if (root.empty())
            return {};
        return root / "tmp";
    }
    lfs::Status setAssetManagerDirectoryPreference(const std::filesystem::path& path) {
        return UserPreferences::instance().setAssetManagerDirectory(path);
    }
    std::filesystem::path loadAssetManagerDirectoryPreference() {
        return UserPreferences::instance().assetManagerDirectory();
    }
    std::filesystem::path assetManagerDirectoryPreferenceRaw() {
        return UserPreferences::instance().assetManagerDirectoryPreference();
    }
    void clearAssetManagerDirectoryPreference() {
        UserPreferences::instance().clearAssetManagerDirectory();
    }
    std::filesystem::path defaultAssetManagerDirectory() {
        return defaultAssetManagerDirectoryPath();
    }

} // namespace lfs::vis
