/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "preferences.hpp"
#include "core/native_fs.hpp"
#include "input/gamepad_navigation.hpp"
#include "input/gamepad_profiles.hpp"
#include "rendering/rendering_types.hpp"
#include "visualizer/navigation_up_axis.hpp"

#include "core/environment.hpp"
#include "core/error.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/persisted_settings.hpp"
#include "core/user_paths.hpp"
#include "rendering/scene_upscaler_registry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

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

        [[nodiscard]] bool knownProgressBarStyle(std::string_view style) {
            return style == "classic" || style == "miner" || style == "tessle";
        }

        [[nodiscard]] bool knownViewportChromeStyle(std::string_view style) {
            return style == "solid" || style == "translucent" || style == "frosted";
        }

        [[nodiscard]] bool knownViewportToolbarPosition(std::string_view position) {
            return position == "top" || position == "centered" || position == "free";
        }

        [[nodiscard]] bool knownProjectManagerView(std::string_view view) {
            return view == "remember" || view == "gallery" || view == "list";
        }

        void eraseKnownProjectManagerFields(json& project_manager) {
            for (const char* key : {"default_view", "open_at_startup", "remember_state"})
                project_manager.erase(key);
            auto state = project_manager.find("state");
            if (state == project_manager.end() || !state->is_object())
                return;
            for (const char* key : {"view_mode", "bottom_panel_height", "inspector_width",
                                    "inspector_height", "inspector_height_version", "thumbnail_size",
                                    "operations_expanded", "sort_mode", "sort_descending",
                                    "selected_folder_id", "active_filter", "panel_width",
                                    "panel_open", "navigator_width", "thumbnail_sizes"})
                state->erase(key);
            for (const char* nested : {"list_column_overrides", "inspector_sections"}) {
                auto children = state->find(nested);
                if (children == state->end() || !children->is_object())
                    continue;
                for (const char* key : {"name", "gallery", "size", "modified", "folder",
                                        "project", "file"})
                    children->erase(key);
                if (children->empty())
                    state->erase(nested);
            }
            if (state->empty())
                project_manager.erase("state");
        }

        constexpr float kDefaultZoomSpeed = 11.0f;
        constexpr float kDefaultNavigationSpeed = 10.0f;
        constexpr float kMinNavigationSpeed = 1.0f;
        constexpr float kMaxNavigationSpeed = 100.0f;

        [[nodiscard]] float clampNavigationSpeed(const float value, const float fallback) {
            return std::isfinite(value)
                       ? std::clamp(value, kMinNavigationSpeed, kMaxNavigationSpeed)
                       : fallback;
        }

        [[nodiscard]] lfs::Error preferencePathError(
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
                return preferencePathError(
                    lfs::ErrorCode::InvalidArgument,
                    display_name + " path is empty.",
                    preference_key + " is empty");
            }
            const auto expanded = expandLeadingTilde(candidate);
            if (!expanded.is_absolute()) {
                return preferencePathError(
                    lfs::ErrorCode::InvalidArgument,
                    display_name + " path must be absolute.",
                    preference_key + " is not an absolute path",
                    candidate);
            }
            std::error_code error;
            auto absolute = std::filesystem::absolute(expanded, error);
            if (error || absolute.empty()) {
                return preferencePathError(
                    lfs::ErrorCode::InvalidArgument,
                    display_name + " path could not be resolved to an absolute path.",
                    error ? error.message() : "absolute() returned an empty path",
                    expanded);
            }
            absolute = absolute.lexically_normal();
            lfs::core::native_fs::create_directories(absolute, error);
            if (error) {
                return preferencePathError(
                    lfs::ErrorCode::PermissionDenied,
                    display_name + " could not be created.",
                    error.message(),
                    absolute);
            }
            if (!lfs::core::native_fs::is_directory(absolute, error) || error) {
                return preferencePathError(
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
                std::ofstream output(lfs::core::native_fs::native_path(probe), std::ios::binary | std::ios::trunc);
                if (!output) {
                    return preferencePathError(
                        lfs::ErrorCode::PermissionDenied,
                        display_name + " is not writable.",
                        "write probe could not be created",
                        absolute);
                }
                output.put('x');
                output.flush();
                if (!output) {
                    std::error_code ignored;
                    lfs::core::native_fs::remove(probe, ignored);
                    return preferencePathError(
                        lfs::ErrorCode::PermissionDenied,
                        display_name + " is not writable.",
                        "write probe could not be written",
                        absolute);
                }
            }
            lfs::core::native_fs::remove(probe, error);
            if (error) {
                return preferencePathError(
                    lfs::ErrorCode::PermissionDenied,
                    display_name + " is not writable.",
                    error.message(),
                    absolute);
            }
            return absolute;
        }

        [[nodiscard]] std::filesystem::path defaultProjectLocationPath() {
            const auto resolved = lfs::core::UserPaths::resolve();
            if (!resolved)
                return {};
            return resolved->rootDir() / "projects";
        }

        [[nodiscard]] bool containsLichtProject(const std::filesystem::path& directory) {
            std::error_code error;
            if (!lfs::core::native_fs::is_directory(directory, error) || error)
                return false;

            constexpr int max_depth = 4;
            std::filesystem::recursive_directory_iterator iterator(lfs::core::native_fs::traversal_path(directory),
                                                                   std::filesystem::directory_options::skip_permission_denied,
                                                                   error);
            const std::filesystem::recursive_directory_iterator end;
            while (iterator != end) {
                if (error) {
                    error.clear();
                    iterator.increment(error);
                    continue;
                }
                const auto name = lfs::core::path_to_utf8(iterator->path().filename());
                if (iterator->is_directory(error)) {
                    if (name.starts_with('.') || iterator.depth() >= max_depth)
                        iterator.disable_recursion_pending();
                } else if (!error && iterator->is_regular_file(error) &&
                           iterator->path().extension() == ".licht") {
                    return true;
                }
                error.clear();
                iterator.increment(error);
            }
            return false;
        }
    } // namespace

    struct UserPreferences::Impl {
        std::mutex mutex;
        json values = json::object();
        json persisted_values = json::object();
        std::optional<lfs::core::UserPaths> paths;
        std::filesystem::path loaded_path;
        bool loaded = false;
        bool writable = true;
        bool warned = false;
        bool reset_project_manager_on_save = false;

        void saveValuesLocked() {
            if (!paths || !writable)
                return;
            try {
                if (lfs::core::persisted_settings::field<std::uint64_t>(values, "schema_version", 1) <= 1)
                    values["schema_version"] = 1;
                auto merged = lfs::core::persisted_settings::mergeForWrite(
                    paths->preferencesFile(), values, "schema_version", 1,
                    nullptr, {}, paths->backupDir());
                // This map is authoritative: removed panel positions must stay removed.
                if (values.contains("floating_window_geometry"))
                    merged["floating_window_geometry"] = values["floating_window_geometry"];
                if (reset_project_manager_on_save) {
                    auto saved_pm = merged.find("project_manager");
                    if (saved_pm != merged.end() && saved_pm->is_object()) {
                        eraseKnownProjectManagerFields(*saved_pm);
                        if (saved_pm->empty())
                            merged.erase("project_manager");
                    }
                }
                for (auto previous = persisted_values.begin(); previous != persisted_values.end(); ++previous) {
                    if (!values.contains(previous.key()))
                        merged.erase(previous.key());
                }
                if (lfs::core::persisted_settings::field<std::uint64_t>(merged, "schema_version", 1) <= 1) {
                    for (const char* legacy : {"working_directory", "asset_manager_directory"})
                        if (!values.contains(legacy))
                            merged.erase(legacy);
                }
                if (const auto result = paths->writePreferencesAtomically(merged.dump(2) + '\n'); !result)
                    LOG_WARN("Unable to save user preferences: {}",
                             lfs::format_for_developer(result.error()));
                else {
                    values = std::move(merged);
                    persisted_values = values;
                    reset_project_manager_on_save = false;
                }
            } catch (const std::exception& error) {
                LOG_WARN("Unable to save user preferences: {}", error.what());
            }
        }

        void logInvalidFieldsLocked() const {
            using lfs::core::persisted_settings::field;
            for (const char* key : {"theme", "language", "project_location",
                                    "working_directory", "asset_manager_directory",
                                    "legacy_working_directory", "new_project_navigation_up_axis",
                                    "camera_navigation_mode", "status_bar_progress_style",
                                    "viewport_chrome_style", "viewport_toolbar_position",
                                    "scene_upscaler", "marketplace_view"})
                (void)field<std::string>(values, key, {});
            for (const char* key : {"remember_camera_navigation", "remember_camera_view_snap",
                                    "camera_view_snap", "new_project_align_grid", "camera_browse_repeat",
                                    "scene_graph_selection_markers", "marketplace_git_checkout"})
                (void)field<bool>(values, key, false);
            for (const char* key : {"zoom_speed", "navigation_speed", "camera_browse_repeat_rate", "viewport_toolbar_free_y"})
                (void)field<float>(values, key, 0.0f);
            const auto scale = values.find("ui_scale");
            if (scale != values.end() && !scale->is_number() &&
                !(scale->is_string() && scale->get<std::string>() == "auto"))
                LOG_WARN("Invalid type for preference 'ui_scale'; using its default");
            for (const char* key : {"project_manager", "mcp", "scene_upscaler_presets"}) {
                const auto item = values.find(key);
                if (item != values.end() && !item->is_null() && !item->is_object())
                    LOG_WARN("Invalid type for preference '{}'; using its default", key);
            }
            const auto project_manager = values.find("project_manager");
            if (project_manager != values.end() && project_manager->is_object()) {
                (void)field<std::string>(*project_manager, "default_view", {});
                (void)field<bool>(*project_manager, "open_at_startup", true);
                (void)field<bool>(*project_manager, "remember_state", true);
                const auto state = project_manager->find("state");
                if (state != project_manager->end() && !state->is_null() && !state->is_object())
                    LOG_WARN("Invalid type for preference 'project_manager.state'; using its default");
            }
            const auto mcp = values.find("mcp");
            if (mcp != values.end() && mcp->is_object()) {
                (void)field<bool>(*mcp, "enabled", false);
                (void)field<bool>(*mcp, "expose_network", false);
                (void)field<int>(*mcp, "port", 0);
                (void)field<bool>(*mcp, "request_logging", false);
            }
        }

        // Camera browsing preferences were first saved under GT comparison names.
        void migrateRenamedKeysLocked() {
            for (const auto& [from, to] : {std::pair{"gt_comparison_key_repeat", "camera_browse_repeat"},
                                           std::pair{"gt_comparison_repeat_rate", "camera_browse_repeat_rate"}}) {
                if (values.contains(from) && !values.contains(to))
                    values[to] = values[from];
                values.erase(from);
            }
        }

        void migrateProjectLocationLocked() {
            if (!paths || !writable)
                return;
            const bool future = lfs::core::persisted_settings::field<std::uint64_t>(values, "schema_version", 1) > 1;

            const auto working = values.find("working_directory");
            if (working != values.end() && working->is_string() &&
                !working->get<std::string>().empty() &&
                values.find("legacy_working_directory") == values.end()) {
                values["legacy_working_directory"] = working->get<std::string>();
            }
            const auto project = values.find("project_location");
            bool changed = false;
            if (project != values.end() && project->is_string()) {
                if (!future) {
                    changed = values.erase("working_directory") > 0;
                    changed = values.erase("asset_manager_directory") > 0 || changed;
                }
            } else {
                std::string selected;
                const auto asset = values.find("asset_manager_directory");
                if (asset != values.end() && asset->is_string() && !asset->get<std::string>().empty()) {
                    selected = asset->get<std::string>();
                } else {
                    const auto root = paths->rootDir().lexically_normal();
                    if (working != values.end() && working->is_string() && !working->get<std::string>().empty()) {
                        const auto working_path = lfs::core::utf8_to_path(working->get<std::string>());
                        if (working_path.lexically_normal() != root)
                            selected = working->get<std::string>();
                    }
                    if (selected.empty() && containsLichtProject(paths->rootDir() / "assets"))
                        selected = lfs::core::path_to_utf8(paths->rootDir() / "assets");
                }
                values["project_location"] = selected;
                changed = true;
                if (!future) {
                    changed = values.erase("working_directory") > 0 || changed;
                    changed = values.erase("asset_manager_directory") > 0 || changed;
                }
            }
            if (changed && lfs::core::persisted_settings::field<std::uint64_t>(values, "schema_version", 1) <= 1)
                saveValuesLocked();
        }

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
            persisted_values = json::object();
            reset_project_manager_on_save = false;

            std::ifstream input(lfs::core::native_fs::native_path(path));
            if (!input)
                return;

            try {
                auto parsed = json::parse(input);
                if (!parsed.is_object())
                    throw std::runtime_error("root value is not an object");
                values = std::move(parsed);
                persisted_values = values;
                input.close();
                migrateRenamedKeysLocked();
                logInvalidFieldsLocked();
                const auto schema = values.find("schema_version");
                if (schema != values.end() &&
                    (!schema->is_number_integer() ||
                     (!schema->is_number_unsigned() && schema->get<std::int64_t>() < 0))) {
                    const auto backup = paths->backupCorruptPreferences();
                    if (!backup) {
                        writable = false;
                        LOG_WARN("Invalid preference version could not be backed up: {}",
                                 lfs::format_for_developer(backup.error()));
                    } else {
                        LOG_WARN("Invalid preference version was backed up before migration");
                    }
                }
                migrateProjectLocationLocked();
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
            saveValuesLocked();
        }
    };

    UserPreferences& UserPreferences::instance() {
        static UserPreferences preferences;
        return preferences;
    }

    UserPreferences::UserPreferences() : impl_(std::make_unique<Impl>()) {}
    UserPreferences::~UserPreferences() = default;

    namespace {
        lfs::Status invalidRenderingPreference(const std::string_view message, const std::string_view field = {}) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::App,
                .user_message = std::string(message),
                .detail = std::string(field),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
    } // namespace

    std::string renderingDefaultsFromSettings(const RenderSettings& settings) {
        json values = json::object();
        values["show_coord_axes"] = settings.show_coord_axes;
        values["show_pivot"] = settings.show_pivot;
        values["show_grid"] = settings.show_grid;
        values["show_camera_frustums"] = settings.show_camera_frustums;
        values["desaturate_unselected"] = settings.desaturate_unselected;
        values["selection_outline"] = settings.selection_outline;
        values["selection_outline_opacity"] = settings.selection_outline_opacity;
        values["selection_outline_width"] = settings.selection_outline_width;
        values["selection_outline_threshold"] = settings.selection_outline_threshold;
        values["selection_outline_holes"] = settings.selection_outline_holes;
        values["selection_outline_color"] = {settings.selection_outline_color.r, settings.selection_outline_color.g, settings.selection_outline_color.b};
        values["desaturate_cropping"] = settings.desaturate_cropping;
        values["hide_outside_depth_box"] = settings.hide_outside_depth_box;
        values["axes_size"] = settings.axes_size;
        values["grid_opacity"] = settings.grid_opacity;
        values["camera_frustum_scale"] = settings.camera_frustum_scale;
        values["lod_pool_vram_fraction"] = settings.lod_pool_vram_fraction;
        values["lod_page_pool_splats"] = settings.lod_page_pool_splats;
        values["selection_color_preview"] = {settings.selection_color_preview.r, settings.selection_color_preview.g, settings.selection_color_preview.b};
        values["selection_color_center_marker"] = {settings.selection_color_center_marker.r, settings.selection_color_center_marker.g, settings.selection_color_center_marker.b};
        return values.dump();
    }

    lfs::Status applyRenderingDefaults(RenderSettings& settings, const std::string_view serialized) {
        const auto values = json::parse(serialized, nullptr, false);
        if (!values.is_object())
            return invalidRenderingPreference("Rendering defaults must be an object.");
        auto candidate = settings;
        const auto invalid = [](const char* field) {
            return invalidRenderingPreference("Invalid rendering default.", field);
        };
        if (const auto it = values.find("show_coord_axes"); it != values.end()) {
            if (!it->is_boolean())
                return invalid("show_coord_axes");
            candidate.show_coord_axes = it->get<bool>();
        }
        if (const auto it = values.find("show_pivot"); it != values.end()) {
            if (!it->is_boolean())
                return invalid("show_pivot");
            candidate.show_pivot = it->get<bool>();
        }
        if (const auto it = values.find("show_grid"); it != values.end()) {
            if (!it->is_boolean())
                return invalid("show_grid");
            candidate.show_grid = it->get<bool>();
        }
        if (const auto it = values.find("show_camera_frustums"); it != values.end()) {
            if (!it->is_boolean())
                return invalid("show_camera_frustums");
            candidate.show_camera_frustums = it->get<bool>();
        }
        if (const auto it = values.find("selection_outline"); it != values.end()) {
            if (!it->is_boolean())
                return invalid("selection_outline");
            candidate.selection_outline = it->get<bool>();
        }
        if (const auto it = values.find("desaturate_unselected"); it != values.end()) {
            if (!it->is_boolean())
                return invalid("desaturate_unselected");
            candidate.desaturate_unselected = it->get<bool>();
        }
        if (const auto it = values.find("desaturate_cropping"); it != values.end()) {
            if (!it->is_boolean())
                return invalid("desaturate_cropping");
            candidate.desaturate_cropping = it->get<bool>();
        }
        if (const auto it = values.find("hide_outside_depth_box"); it != values.end()) {
            if (!it->is_boolean())
                return invalid("hide_outside_depth_box");
            candidate.hide_outside_depth_box = it->get<bool>();
        }
        if (const auto it = values.find("axes_size"); it != values.end()) {
            if (!it->is_number())
                return invalid("axes_size");
            const double value = it->get<double>();
            if (!std::isfinite(value) || value < 0.5 || value > 10)
                return invalid("axes_size");
            candidate.axes_size = static_cast<float>(value);
        }
        if (const auto it = values.find("grid_opacity"); it != values.end()) {
            if (!it->is_number())
                return invalid("grid_opacity");
            const double value = it->get<double>();
            if (!std::isfinite(value) || value < 0 || value > 1)
                return invalid("grid_opacity");
            candidate.grid_opacity = static_cast<float>(value);
        }
        if (const auto it = values.find("camera_frustum_scale"); it != values.end()) {
            if (!it->is_number())
                return invalid("camera_frustum_scale");
            const double value = it->get<double>();
            if (!std::isfinite(value) || value < 0.01 || value > 10)
                return invalid("camera_frustum_scale");
            candidate.camera_frustum_scale = static_cast<float>(value);
        }
        if (const auto it = values.find("lod_pool_vram_fraction"); it != values.end()) {
            if (!it->is_number())
                return invalid("lod_pool_vram_fraction");
            const double value = it->get<double>();
            if (!std::isfinite(value) || value < 0.05 || value > 0.9)
                return invalid("lod_pool_vram_fraction");
            candidate.lod_pool_vram_fraction = static_cast<float>(value);
        }
        if (const auto it = values.find("lod_page_pool_splats"); it != values.end()) {
            if (!it->is_number())
                return invalid("lod_page_pool_splats");
            const double value = it->get<double>();
            if (!std::isfinite(value) || value < 0 || value > 100'000'000 || std::floor(value) != value)
                return invalid("lod_page_pool_splats");
            candidate.lod_page_pool_splats = static_cast<size_t>(value);
        }
        if (const auto it = values.find("selection_color_preview"); it != values.end()) {
            if (!it->is_array() || it->size() != 3)
                return invalid("selection_color_preview");
            for (size_t i = 0; i < 3; ++i) {
                if (!(*it)[i].is_number())
                    return invalid("selection_color_preview");
                const double value = (*it)[i].get<double>();
                if (!std::isfinite(value) || value < 0 || value > 1)
                    return invalid("selection_color_preview");
                candidate.selection_color_preview[static_cast<int>(i)] = static_cast<float>(value);
            }
        }
        if (const auto it = values.find("selection_color_center_marker"); it != values.end()) {
            if (!it->is_array() || it->size() != 3)
                return invalid("selection_color_center_marker");
            for (size_t i = 0; i < 3; ++i) {
                if (!(*it)[i].is_number())
                    return invalid("selection_color_center_marker");
                const double value = (*it)[i].get<double>();
                if (!std::isfinite(value) || value < 0 || value > 1)
                    return invalid("selection_color_center_marker");
                candidate.selection_color_center_marker[static_cast<int>(i)] = static_cast<float>(value);
            }
        }
        if (const auto it = values.find("selection_outline_color"); it != values.end()) {
            if (!it->is_array() || it->size() != 3)
                return invalid("selection_outline_color");
            for (size_t i = 0; i < 3; ++i) {
                if (!(*it)[i].is_number())
                    return invalid("selection_outline_color");
                const double value = (*it)[i].get<double>();
                if (!std::isfinite(value) || value < 0 || value > 1)
                    return invalid("selection_outline_color");
                candidate.selection_outline_color[static_cast<int>(i)] = static_cast<float>(value);
            }
        }
        if (const auto it = values.find("selection_outline_opacity"); it != values.end()) {
            if (!it->is_number())
                return invalid("selection_outline_opacity");
            const double value = it->get<double>();
            if (!std::isfinite(value) || value < 0 || value > 1)
                return invalid("selection_outline_opacity");
            candidate.selection_outline_opacity = static_cast<float>(value);
        }

        if (const auto it = values.find("selection_outline_width"); it != values.end()) {
            if (!it->is_number())
                return invalid("selection_outline_width");
            const double value = it->get<double>();
            if (!std::isfinite(value) || value < 1 || value > 12)
                return invalid("selection_outline_width");
            candidate.selection_outline_width = static_cast<float>(value);
        }

        if (const auto it = values.find("selection_outline_threshold"); it != values.end()) {
            if (!it->is_number())
                return invalid("selection_outline_threshold");
            const double value = it->get<double>();
            if (!std::isfinite(value) || value < 0.05 || value > 0.95)
                return invalid("selection_outline_threshold");
            candidate.selection_outline_threshold = static_cast<float>(value);
        }

        if (const auto it = values.find("selection_outline_holes"); it != values.end()) {
            if (!it->is_boolean())
                return invalid("selection_outline_holes");
            candidate.selection_outline_holes = it->get<bool>();
        }

        settings = std::move(candidate);
        return {};
    }

    namespace {
        input::GamepadProfiles gamepadProfilesFromValues(const json& values) {
            if (const auto legacy = values.find("gamepad_navigation"); legacy != values.end())
                (void)input::GamepadNavigationSettings::fromJson(*legacy);
            if (const auto profiles = values.find("gamepad_profiles"); profiles != values.end())
                return input::GamepadProfiles::fromJson(*profiles);
            return input::GamepadProfiles::migrate(input::GamepadNavigationSettings::fromJson(
                values.value("gamepad_navigation", json::object())));
        }
    } // namespace

    std::string UserPreferences::gamepadNavigation() {
        std::lock_guard lock(impl_->mutex);
        impl_->loadLocked();
        const auto found = impl_->values.find("gamepad_navigation");
        if (!impl_->values.contains("gamepad_profiles"))
            return found == impl_->values.end() ? "{}" : found->dump();
        const bool enabled = found != impl_->values.end() ? found->value("enabled", false) : false;
        return gamepadProfilesFromValues(impl_->values).activeSettings(enabled).json().dump();
    }

    std::string UserPreferences::gamepadProfiles() {
        std::lock_guard lock(impl_->mutex);
        impl_->loadLocked();
        return gamepadProfilesFromValues(impl_->values).value.dump();
    }

    lfs::Status UserPreferences::setGamepadProfiles(const std::string_view serialized, bool enabled, const std::string_view expected) {
        try {
            auto profiles = input::GamepadProfiles::fromJson(json::parse(serialized));
            std::lock_guard lock(impl_->mutex);
            impl_->loadLocked();
            if (!impl_->paths || !impl_->writable)
                return invalidRenderingPreference("Gamepad preferences are not writable");
            // Never replace unreadable/newer data, including via reset.
            const auto previous = gamepadProfilesFromValues(impl_->values);
            if (!expected.empty() && previous.value.dump() != expected)
                return invalidRenderingPreference("Gamepad profiles changed while editing");
            auto candidate = impl_->values;
            candidate["gamepad_profiles"] = profiles.value;
            candidate["gamepad_navigation"] = profiles.activeSettings(enabled).json();
            auto merged = core::persisted_settings::mergeForWrite(impl_->paths->preferencesFile(), candidate,
                                                                  "schema_version", 1, nullptr, {}, impl_->paths->backupDir());
            merged["gamepad_profiles"] = candidate["gamepad_profiles"];
            merged["gamepad_navigation"] = candidate["gamepad_navigation"];
            if (auto result = impl_->paths->writePreferencesAtomically(merged.dump(2) + '\n'); !result)
                return result;
            impl_->values = merged;
            impl_->persisted_values = std::move(merged);
            return {};
        } catch (const std::exception& error) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::App,
                .user_message = error.what(),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
    }

    lfs::Status UserPreferences::setGamepadNavigation(const std::string_view serialized) {
        try {
            const auto settings = input::GamepadNavigationSettings::fromJson(json::parse(serialized));
            // Read/modify/write under the same mutex: slider/binding edits update
            // only the active profile and cannot resurrect deleted profiles/overrides.
            std::lock_guard lock(impl_->mutex);
            impl_->loadLocked();
            if (!impl_->paths || !impl_->writable)
                return invalidRenderingPreference("Gamepad preferences are not writable");
            auto profiles = gamepadProfilesFromValues(impl_->values);
            profiles.updateActive(settings);
            auto candidate = impl_->values;
            candidate["gamepad_navigation"] = settings.json();
            candidate["gamepad_profiles"] = profiles.value;
            auto merged = core::persisted_settings::mergeForWrite(impl_->paths->preferencesFile(), candidate,
                                                                  "schema_version", 1, nullptr, {}, impl_->paths->backupDir());
            merged["gamepad_navigation"] = candidate["gamepad_navigation"];
            merged["gamepad_profiles"] = candidate["gamepad_profiles"];
            if (auto result = impl_->paths->writePreferencesAtomically(merged.dump(2) + '\n'); !result)
                return result;
            impl_->values = merged;
            impl_->persisted_values = std::move(merged);
            return {};
        } catch (const std::exception& error) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::App,
                .user_message = error.what(),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
    }

    lfs::Status UserPreferences::setRenderingDefaults(const std::string_view serialized) {
        RenderSettings validated;
        if (auto result = applyRenderingDefaults(validated, serialized); !result)
            return result;
        const auto update = json::parse(serialized);
        const auto known = json::parse(renderingDefaultsFromSettings(validated));
        std::lock_guard lock(impl_->mutex);
        impl_->loadLocked();
        auto& saved = impl_->values["rendering_defaults"];
        if (!saved.is_object())
            saved = json::object();
        for (auto it = update.begin(); it != update.end(); ++it)
            if (known.contains(it.key()))
                saved[it.key()] = known[it.key()];
        impl_->saveValuesLocked();
        return {};
    }

    std::string UserPreferences::renderingDefaults() {
        std::lock_guard lock(impl_->mutex);
        impl_->loadLocked();
        RenderSettings defaults;
        const auto stored = impl_->values.find("rendering_defaults");
        if (stored != impl_->values.end() && stored->is_object()) {
            for (auto it = stored->begin(); it != stored->end(); ++it) {
                // Validate independently so one damaged value does not erase valid defaults.
                (void)applyRenderingDefaults(defaults, json{{it.key(), it.value()}}.dump());
            }
        }
        return renderingDefaultsFromSettings(defaults);
    }

    lfs::Status UserPreferences::setViewportVignette(const std::string_view serialized) {
        const auto values = json::parse(serialized, nullptr, false);
        if (!values.is_object())
            return invalidRenderingPreference("Vignette preferences must be an object.");
        if (values.contains("enabled") && !values["enabled"].is_boolean())
            return invalidRenderingPreference("Invalid vignette enabled flag.");
        for (const char* key : {"intensity", "radius", "softness"}) {
            if (!values.contains(key))
                continue;
            if (!values[key].is_number())
                return invalidRenderingPreference("Invalid vignette preference.", key);
            const auto value = values[key].get<double>();
            if (!std::isfinite(value) || value < 0 || value > 1)
                return invalidRenderingPreference("Invalid vignette preference.", key);
        }
        std::lock_guard lock(impl_->mutex);
        impl_->loadLocked();
        auto& saved = impl_->values["viewport_vignette"];
        if (!saved.is_object())
            saved = json::object();
        saved.update(values);
        impl_->saveValuesLocked();
        return {};
    }

    std::string UserPreferences::viewportVignette() {
        std::lock_guard lock(impl_->mutex);
        impl_->loadLocked();
        json result = json::object();
        const auto stored = impl_->values.find("viewport_vignette");
        if (stored != impl_->values.end() && stored->is_object()) {
            if (stored->contains("enabled") && (*stored)["enabled"].is_boolean())
                result["enabled"] = (*stored)["enabled"];
            for (const char* key : {"intensity", "radius", "softness"}) {
                const auto field = stored->find(key);
                if (field != stored->end() && field->is_number()) {
                    const auto value = field->get<double>();
                    if (std::isfinite(value) && value >= 0 && value <= 1)
                        result[key] = value;
                }
            }
        }
        return result.dump();
    }

    void UserPreferences::setSidebarDefaults(const float width, const float scene_ratio,
                                             const std::string& active_tab) {
        if (!std::isfinite(width) || width <= 0 || !std::isfinite(scene_ratio))
            return;
        std::lock_guard lock(impl_->mutex);
        impl_->loadLocked();
        auto& saved = impl_->values["sidebar_defaults"];
        if (!saved.is_object())
            saved = json::object();
        saved["right_panel_width"] = width;
        saved["scene_panel_ratio"] = std::clamp(scene_ratio, 0.01f, 0.99f);
        saved["active_tab"] = active_tab;
        impl_->saveValuesLocked();
    }

    std::string UserPreferences::sidebarDefaults() {
        std::lock_guard lock(impl_->mutex);
        impl_->loadLocked();
        json result = json::object();
        const auto stored = impl_->values.find("sidebar_defaults");
        if (stored != impl_->values.end() && stored->is_object()) {
            for (const char* key : {"right_panel_width", "scene_panel_ratio"}) {
                const auto field = stored->find(key);
                if (field != stored->end() && field->is_number()) {
                    const auto value = field->get<double>();
                    if (std::isfinite(value) && value > 0 &&
                        (std::string_view(key) != "scene_panel_ratio" || value < 1))
                        result[key] = value;
                }
            }
            if (const auto tab = stored->find("active_tab"); tab != stored->end() && tab->is_string())
                result["active_tab"] = *tab;
        }
        return result.dump();
    }

    void UserPreferences::clearSidebarDefaults() {
        std::lock_guard lock(impl_->mutex);
        impl_->loadLocked();
        auto saved = impl_->values.find("sidebar_defaults");
        if (saved != impl_->values.end() && saved->is_object())
            for (const char* key : {"right_panel_width", "scene_panel_ratio", "active_tab"})
                // Explicit nulls replace these fields during the recursive
                // settings merge, without deleting future/unknown children.
                (*saved)[key] = nullptr;
        impl_->saveValuesLocked();
    }

    void UserPreferences::setThemeName(const std::string& value) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["theme"] = value;
        impl_->saveLocked();
    }
    std::string UserPreferences::themeName() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        return lfs::core::persisted_settings::field<std::string>(impl_->values, "theme", {});
    }
    void UserPreferences::setNativeFloatingWindows(bool enabled) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["native_floating_windows"] = enabled;
        impl_->saveLocked();
    }
    bool UserPreferences::nativeFloatingWindows() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("native_floating_windows");
        return it != impl_->values.end() && it->is_boolean() && it->get<bool>();
    }

    namespace {
        [[nodiscard]] VrPreferences clampedVrPreferences(VrPreferences preferences) {
            constexpr VrPreferences defaults;
            const auto clamp = [](const float value, const float low, const float high, const float fallback) {
                return std::isfinite(value) ? std::clamp(value, low, high) : fallback;
            };
            preferences.resolution_scale = clamp(preferences.resolution_scale, 0.25f, 1.5f, defaults.resolution_scale);
            preferences.scene_units_per_meter =
                clamp(preferences.scene_units_per_meter, 0.001f, 1000.0f, defaults.scene_units_per_meter);
            preferences.move_speed = clamp(preferences.move_speed, 0.1f, 10.0f, defaults.move_speed);
            return preferences;
        }
    } // namespace

    void UserPreferences::setVrPreferences(const VrPreferences& preferences, const bool persist) {
        const auto clamped = clampedVrPreferences(preferences);
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["vr_enabled"] = clamped.enabled;
        impl_->values["vr_resolution_scale"] = clamped.resolution_scale;
        impl_->values["vr_scene_units_per_meter"] = clamped.scene_units_per_meter;
        impl_->values["vr_move_speed"] = clamped.move_speed;
        if (persist)
            impl_->saveLocked();
    }

    VrPreferences UserPreferences::vrPreferences() {
        using lfs::core::persisted_settings::field;
        constexpr VrPreferences defaults;
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto& values = impl_->values;
        return clampedVrPreferences({
            .enabled = field<bool>(values, "vr_enabled", defaults.enabled),
            .resolution_scale = field<float>(values, "vr_resolution_scale", defaults.resolution_scale),
            .scene_units_per_meter = field<float>(values, "vr_scene_units_per_meter", defaults.scene_units_per_meter),
            .move_speed = field<float>(values, "vr_move_speed", defaults.move_speed),
        });
    }
    void UserPreferences::setFloatingWindowGeometry(const std::string& value) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto parsed = json::parse(value, nullptr, false);
        if (!parsed.is_object())
            return;
        impl_->values["floating_window_geometry"] = parsed;
        impl_->saveLocked();
    }
    std::string UserPreferences::floatingWindowGeometry() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("floating_window_geometry");
        return it != impl_->values.end() && it->is_object() ? it->dump() : "{}";
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
    void UserPreferences::setZoomSpeed(const float value, const bool persist) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["zoom_speed"] = clampNavigationSpeed(value, kDefaultZoomSpeed);
        if (persist)
            impl_->saveLocked();
    }
    float UserPreferences::zoomSpeed() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("zoom_speed");
        if (it != impl_->values.end() && it->is_number())
            return clampNavigationSpeed(it->get<float>(), kDefaultZoomSpeed);
        return kDefaultZoomSpeed;
    }
    void UserPreferences::setNavigationSpeed(const float value, const bool persist) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["navigation_speed"] = clampNavigationSpeed(value, kDefaultNavigationSpeed);
        if (persist)
            impl_->saveLocked();
    }
    float UserPreferences::navigationSpeed() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("navigation_speed");
        if (it != impl_->values.end() && it->is_number())
            return clampNavigationSpeed(it->get<float>(), kDefaultNavigationSpeed);
        return kDefaultNavigationSpeed;
    }
    void UserPreferences::setCameraBrowseRepeat(const bool enabled) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["camera_browse_repeat"] = enabled;
        impl_->saveLocked();
    }
    bool UserPreferences::cameraBrowseRepeat() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        return lfs::core::persisted_settings::field<bool>(impl_->values, "camera_browse_repeat", true);
    }
    void UserPreferences::setCameraBrowseRepeatRate(const float value, const bool persist) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["camera_browse_repeat_rate"] = std::isfinite(value) ? std::clamp(value, 1.0f, 20.0f) : 5.0f;
        if (persist)
            impl_->saveLocked();
    }
    float UserPreferences::cameraBrowseRepeatRate() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const float value = lfs::core::persisted_settings::field<float>(impl_->values, "camera_browse_repeat_rate", 5.0f);
        return std::isfinite(value) ? std::clamp(value, 1.0f, 20.0f) : 5.0f;
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
    void UserPreferences::setNewProjectNavigation(const std::string_view axis, const bool align_grid) {
        if (!navigationUpAxisFromName(axis))
            throw std::invalid_argument("navigation up axis must be legacy_y or data_z");
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["new_project_navigation_up_axis"] = axis;
        impl_->values["new_project_align_grid"] = align_grid;
        impl_->saveLocked();
    }
    std::string UserPreferences::newProjectNavigationUpAxis() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("new_project_navigation_up_axis");
        if (it == impl_->values.end() || !it->is_string())
            return "legacy_y";
        const auto axis = it->get<std::string>();
        return navigationUpAxisFromName(axis) ? axis : "legacy_y";
    }
    bool UserPreferences::alignNewProjectGrid() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("new_project_align_grid");
        return it != impl_->values.end() && it->is_boolean() ? it->get<bool>() : true;
    }

    void UserPreferences::setCameraNavigation(const std::string& value) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        if (!lfs::core::persisted_settings::field<bool>(impl_->values, "remember_camera_navigation", false))
            return;
        impl_->values["camera_navigation_mode"] = knownCameraMode(value) ? value : "orbit";
        impl_->saveLocked();
    }
    std::string UserPreferences::cameraNavigation() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        if (!lfs::core::persisted_settings::field<bool>(impl_->values, "remember_camera_navigation", false))
            return "orbit";
        const std::string mode = lfs::core::persisted_settings::field<std::string>(impl_->values, "camera_navigation_mode", "orbit");
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
        return lfs::core::persisted_settings::field<bool>(impl_->values, "remember_camera_navigation", false);
    }
    void UserPreferences::setCameraViewSnap(const bool enabled) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        if (!lfs::core::persisted_settings::field<bool>(impl_->values, "remember_camera_view_snap", false))
            return;
        impl_->values["camera_view_snap"] = enabled;
        impl_->saveLocked();
    }
    bool UserPreferences::cameraViewSnap() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        if (!lfs::core::persisted_settings::field<bool>(impl_->values, "remember_camera_view_snap", false))
            return false;
        return lfs::core::persisted_settings::field<bool>(impl_->values, "camera_view_snap", false);
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
        return lfs::core::persisted_settings::field<bool>(impl_->values, "remember_camera_view_snap", false);
    }
    void UserPreferences::setSplatDrag(const SplatDragPreferenceState& state) {
        if ((state.format != "ply" && state.format != "spz") || !state.axes.valid())
            return;
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["splat_drag"] = {{"format", state.format}, {"sh_degree", std::clamp(state.sh_degree, 0, 3)}, {"up_axis", state.axes.up}};
        impl_->saveLocked();
    }
    SplatDragPreferenceState UserPreferences::splatDrag() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        SplatDragPreferenceState state;
        const auto it = impl_->values.find("splat_drag");
        if (it == impl_->values.end() || !it->is_object())
            return state;
        using lfs::core::persisted_settings::field;
        state.format = field<std::string>(*it, "format", "ply");
        if (state.format != "ply" && state.format != "spz")
            state.format = "ply";
        state.sh_degree = std::clamp(field<int>(*it, "sh_degree", 3), 0, 3);
        // Preserve the old checkbox for existing users, while fresh settings
        // retain source axes unless the native drag recognizes Blender.
        if (!it->contains("up_axis") && field<bool>(*it, "blender_axes", false))
            state.axes = SplatDragAxes::blender();
        else
            state.axes = {field<int>(*it, "up_axis", -2)};
        if (!state.axes.valid())
            state.axes = {};
        return state;
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
        return lfs::core::persisted_settings::field<bool>(impl_->values, "scene_graph_selection_markers", false);
    }
    void UserPreferences::setProgressBarStyle(const std::string_view value) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["status_bar_progress_style"] =
            knownProgressBarStyle(value) ? std::string(value) : "classic";
        impl_->saveLocked();
    }
    std::string UserPreferences::progressBarStyle() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("status_bar_progress_style");
        if (it == impl_->values.end() || !it->is_string())
            return "classic";
        const std::string style = it->get<std::string>();
        return knownProgressBarStyle(style) ? style : "classic";
    }
    void UserPreferences::setViewportChromeStyle(const std::string_view value) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["viewport_chrome_style"] =
            knownViewportChromeStyle(value) ? std::string(value) : "translucent";
        impl_->saveLocked();
    }
    std::string UserPreferences::viewportChromeStyle() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("viewport_chrome_style");
        if (it == impl_->values.end() || !it->is_string())
            return "translucent";
        const std::string style = it->get<std::string>();
        return knownViewportChromeStyle(style) ? style : "translucent";
    }
    void UserPreferences::setViewportToolbarPosition(const std::string_view value) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["viewport_toolbar_position"] =
            knownViewportToolbarPosition(value) ? std::string(value) : "centered";
        impl_->saveLocked();
    }
    std::string UserPreferences::viewportToolbarPosition() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("viewport_toolbar_position");
        if (it == impl_->values.end() || !it->is_string())
            return "centered";
        const std::string position = it->get<std::string>();
        return knownViewportToolbarPosition(position) ? position : "centered";
    }
    void UserPreferences::setViewportToolbarFreeY(const float value) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["viewport_toolbar_free_y"] =
            std::clamp(std::isfinite(value) ? value : 0.5f, 0.0f, 1.0f);
        impl_->saveLocked();
    }
    float UserPreferences::viewportToolbarFreeY() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("viewport_toolbar_free_y");
        if (it == impl_->values.end() || !it->is_number())
            return 0.5f;
        const float value = it->get<float>();
        return std::clamp(std::isfinite(value) ? value : 0.5f, 0.0f, 1.0f);
    }

    void UserPreferences::setMarketplaceView(const std::string_view value) {
        if (value != "card" && value != "list")
            throw std::invalid_argument("Unsupported Marketplace view");
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["marketplace_view"] = std::string(value);
        impl_->saveLocked();
    }

    std::string UserPreferences::marketplaceView() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("marketplace_view");
        if (it == impl_->values.end() || !it->is_string())
            return "list";
        const auto value = it->get<std::string>();
        return value == "card" || value == "list" ? value : "list";
    }

    void UserPreferences::setMarketplaceGitCheckout(const bool enabled) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["marketplace_git_checkout"] = enabled;
        impl_->saveLocked();
    }

    bool UserPreferences::marketplaceGitCheckout() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("marketplace_git_checkout");
        return it != impl_->values.end() && it->is_boolean() && it->get<bool>();
    }

    void UserPreferences::resetMarketplacePreferences() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values.erase("marketplace_view");
        impl_->values.erase("marketplace_git_checkout");
        impl_->saveLocked();
    }

    void UserPreferences::setProjectManagerDefaultView(const std::string_view value) {
        if (!knownProjectManagerView(value))
            throw std::invalid_argument("Unsupported Project Manager default view");
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        auto& project_manager = impl_->values["project_manager"];
        if (!project_manager.is_object())
            project_manager = json::object();
        project_manager["default_view"] = std::string(value);
        impl_->saveLocked();
    }

    std::string UserPreferences::projectManagerDefaultView() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto project_manager = impl_->values.find("project_manager");
        if (project_manager == impl_->values.end() || !project_manager->is_object())
            return "remember";
        const auto view = project_manager->find("default_view");
        if (view == project_manager->end() || !view->is_string())
            return "remember";
        const auto value = view->get<std::string>();
        return knownProjectManagerView(value) ? value : "remember";
    }

    void UserPreferences::setOpenProjectManagerAtStartup(const bool enabled) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        auto& project_manager = impl_->values["project_manager"];
        if (!project_manager.is_object())
            project_manager = json::object();
        project_manager["open_at_startup"] = enabled;
        impl_->saveLocked();
    }

    bool UserPreferences::openProjectManagerAtStartup() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto project_manager = impl_->values.find("project_manager");
        if (project_manager == impl_->values.end() || !project_manager->is_object())
            return true;
        const auto open = project_manager->find("open_at_startup");
        return open == project_manager->end() || !open->is_boolean()
                   ? true
                   : open->get<bool>();
    }

    void UserPreferences::setRememberProjectManagerState(const bool enabled) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        auto& project_manager = impl_->values["project_manager"];
        if (!project_manager.is_object())
            project_manager = json::object();
        project_manager["remember_state"] = enabled;
        impl_->saveLocked();
    }

    bool UserPreferences::rememberProjectManagerState() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto project_manager = impl_->values.find("project_manager");
        if (project_manager == impl_->values.end() || !project_manager->is_object())
            return true;
        const auto remember = project_manager->find("remember_state");
        return remember == project_manager->end() || !remember->is_boolean()
                   ? true
                   : remember->get<bool>();
    }

    void UserPreferences::setProjectManagerState(const std::string_view serialized_state) {
        const auto state = json::parse(serialized_state.begin(), serialized_state.end());
        if (!state.is_object())
            throw std::invalid_argument("Project Manager state must be an object");
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        auto& project_manager = impl_->values["project_manager"];
        if (!project_manager.is_object())
            project_manager = json::object();
        if (lfs::core::persisted_settings::field<std::uint64_t>(impl_->values, "schema_version", 1) > 1) {
            auto& stored_state = project_manager["state"];
            if (!stored_state.is_object())
                stored_state = json::object();
            lfs::core::persisted_settings::mergeKnown(stored_state, state);
        } else {
            project_manager["state"] = state;
        }
        impl_->saveLocked();
    }

    std::string UserPreferences::projectManagerState() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto project_manager = impl_->values.find("project_manager");
        if (project_manager == impl_->values.end() || !project_manager->is_object())
            return "{}";
        const auto state = project_manager->find("state");
        return state != project_manager->end() && state->is_object()
                   ? state->dump()
                   : "{}";
    }

    void UserPreferences::resetProjectManagerPreferences() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        auto project_manager = impl_->values.find("project_manager");
        if (lfs::core::persisted_settings::field<std::uint64_t>(impl_->values, "schema_version", 1) > 1) {
            if (project_manager == impl_->values.end() || !project_manager->is_object())
                return;
            impl_->reset_project_manager_on_save = true;
            eraseKnownProjectManagerFields(*project_manager);
        } else {
            impl_->values.erase("project_manager");
        }
        impl_->saveLocked();
    }

    void UserPreferences::setMcp(const McpPreferenceState& state) {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        auto& mcp = impl_->values["mcp"];
        if (!mcp.is_object())
            mcp = json::object();
        mcp["enabled"] = state.enabled;
        mcp["expose_network"] = state.expose_network;
        mcp["port"] = std::clamp(state.port, 1, 65535);
        mcp["request_logging"] = state.request_logging;
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
    void saveZoomSpeedPreference(const float speed, const bool persist) { UserPreferences::instance().setZoomSpeed(speed, persist); }
    float loadZoomSpeedPreference() { return UserPreferences::instance().zoomSpeed(); }
    void saveNavigationSpeedPreference(const float speed, const bool persist) { UserPreferences::instance().setNavigationSpeed(speed, persist); }
    float loadNavigationSpeedPreference() { return UserPreferences::instance().navigationSpeed(); }
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
    void saveProgressBarStylePreference(const std::string_view style) {
        UserPreferences::instance().setProgressBarStyle(style);
    }
    std::string loadProgressBarStylePreference() {
        return UserPreferences::instance().progressBarStyle();
    }
    void saveViewportChromeStylePreference(const std::string_view style) {
        UserPreferences::instance().setViewportChromeStyle(style);
    }
    std::string loadViewportChromeStylePreference() {
        return UserPreferences::instance().viewportChromeStyle();
    }
    void saveViewportToolbarPositionPreference(const std::string_view position) {
        UserPreferences::instance().setViewportToolbarPosition(position);
    }
    std::string loadViewportToolbarPositionPreference() {
        return UserPreferences::instance().viewportToolbarPosition();
    }
    void saveViewportToolbarFreeYPreference(const float value) {
        UserPreferences::instance().setViewportToolbarFreeY(value);
    }
    float loadViewportToolbarFreeYPreference() {
        return UserPreferences::instance().viewportToolbarFreeY();
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

    lfs::Status UserPreferences::setProjectLocation(
        const std::filesystem::path& path) {
        auto resolved = validateWritableDirectory(
            path, "project_location", "The project location",
            ".lfs-project-write-probe-");
        if (!resolved)
            return lfs::Status::failure(std::move(resolved).error());
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["project_location"] =
            lfs::core::path_to_utf8(*resolved);
        impl_->saveLocked();
        return {};
    }

    std::filesystem::path UserPreferences::projectLocation() {
        const auto raw = projectLocationPreference();
        return raw.empty() ? defaultProjectLocationPath() : raw;
    }

    std::filesystem::path UserPreferences::projectLocationPreference() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        const auto it = impl_->values.find("project_location");
        if (it == impl_->values.end() || !it->is_string())
            return {};
        const std::string stored = it->get<std::string>();
        return stored.empty() ? std::filesystem::path{}
                              : lfs::core::utf8_to_path(stored);
    }

    void UserPreferences::clearProjectLocation() {
        std::scoped_lock lock(impl_->mutex);
        impl_->loadLocked();
        impl_->values["project_location"] = "";
        impl_->saveLocked();
    }

    std::filesystem::path workingDirectoryPreferenceRaw() {
        if (disabled())
            return {};
        const auto resolved = lfs::core::UserPaths::resolve();
        if (!resolved)
            return {};
        std::ifstream input(lfs::core::native_fs::native_path(resolved->preferencesFile()));
        if (!input)
            return {};
        try {
            const auto values = json::parse(input);
            auto it = values.find("legacy_working_directory");
            if (it == values.end())
                it = values.find("working_directory");
            if (it == values.end() || !it->is_string() || it->get<std::string>().empty())
                return {};
            return lfs::core::utf8_to_path(it->get<std::string>());
        } catch (const std::exception&) {
            // LFS-CENSUS-OK(empty-catch): malformed legacy preferences are ignored by startup scanning.
            return {};
        }
    }
    lfs::Status setProjectLocationPreference(
        const std::filesystem::path& path) {
        return UserPreferences::instance().setProjectLocation(path);
    }
    std::filesystem::path loadProjectLocationPreference() {
        return UserPreferences::instance().projectLocation();
    }
    std::filesystem::path projectLocationPreferenceRaw() {
        return UserPreferences::instance().projectLocationPreference();
    }
    std::filesystem::path resolvedProjectLocationPreference() {
        const auto raw = projectLocationPreferenceRaw();
        if (raw.empty())
            return {};
        const auto expanded = expandLeadingTilde(raw);
        std::error_code error;
        const auto absolute = std::filesystem::absolute(expanded, error);
        return error || !expanded.is_absolute() ? std::filesystem::path{}
                                                : absolute.lexically_normal();
    }
    lfs::Status probeProjectLocation(const std::filesystem::path& path) {
        if (auto probed = validateWritableDirectory(
                path, "project_location", "The project location",
                ".lfs-project-write-probe-");
            !probed)
            return lfs::Status::failure(std::move(probed).error());
        return {};
    }
    void clearProjectLocationPreference() {
        UserPreferences::instance().clearProjectLocation();
    }
    std::filesystem::path defaultProjectLocation() {
        return defaultProjectLocationPath();
    }
} // namespace lfs::vis
