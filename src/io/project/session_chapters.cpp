/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/session_chapters.hpp"

#include "rendering/coordinate_conventions.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <concepts>
#include <cstring>
#include <exception>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
#include <utility>

namespace lfs::io::project {

    namespace {

        using Json = JsonChapterDom::Json;

        constexpr std::array<std::byte, 8> METR_MAGIC = {
            static_cast<std::byte>('L'),
            static_cast<std::byte>('F'),
            static_cast<std::byte>('M'),
            static_cast<std::byte>('E'),
            static_cast<std::byte>('T'),
            static_cast<std::byte>('R'),
            static_cast<std::byte>('\r'),
            static_cast<std::byte>('\n'),
        };

        lfs::Error session_error(
            const lfs::ErrorCode code,
            std::string detail,
            const std::string_view field) {
            lfs::SmallFields fields;
            fields.add("field", field);
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability =
                    lfs::Retryability::NotRetryable,
                .operation_id = {},
                .user_message =
                    "The project session chapter is invalid.",
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
                .native = std::nullopt,
            });
        }

        template <typename T>
        lfs::Result<T> fail(
            const lfs::ErrorCode code,
            std::string detail,
            const std::string_view field) {
            if constexpr (std::same_as<T, void>) {
                return lfs::Status::failure(
                    session_error(
                        code, std::move(detail), field));
            } else {
                return session_error(
                    code, std::move(detail), field);
            }
        }

        Json vec3(
            const double x,
            const double y,
            const double z) {
            return Json::array({x, y, z});
        }

        Json identity_rotation() {
            return Json::array({
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            });
        }

        Json rotation_matrix(const glm::mat3& rotation) {
            return Json::array({
                rotation[0][0],
                rotation[0][1],
                rotation[0][2],
                rotation[1][0],
                rotation[1][1],
                rotation[1][2],
                rotation[2][0],
                rotation[2][1],
                rotation[2][2],
            });
        }

        Json default_camera_look_at_rotation() {
            // Same look-at as Viewport::CameraMotion (from t toward origin).
            return rotation_matrix(
                lfs::rendering::makeVisualizerLookAtRotation(
                    glm::vec3(-5.657f, 3.0f, -5.657f),
                    glm::vec3(0.0f, 0.0f, 0.0f)));
        }

        Json default_panel_camera(
            const std::string_view panel) {
            const Json look_at =
                default_camera_look_at_rotation();
            return Json{
                {"panel", panel},
                {"R", look_at},
                {"t", vec3(-5.657, 3.0, -5.657)},
                {"pivot", vec3(0.0, 0.0, 0.0)},
                {"home_R", look_at},
                {"home_t", vec3(-5.657, 3.0, -5.657)},
                {"home_pivot", vec3(0.0, 0.0, 0.0)},
                {"home_saved", true},
                {"zoom_speed", 11.0},
                {"max_zoom_speed", 100.0},
                {"rotate_speed", 0.001},
                {"centre_speed", 0.002},
                {"roll_speed", 0.01},
                {"translate_speed", 0.0005},
                {"wasd_speed", 8.0},
                {"max_wasd_speed", 100.0},
                {"ortho_scale", nullptr},
            };
        }

        Json default_render_settings() {
            return Json{
                {"focal_length_mm", 50.0},
                {"scaling_modifier", 1.0},
                {"antialiasing", false},
                {"mip_filter", false},
                {"sh_degree", 3},
                {"render_scale", 1.0},
                {"camera_metrics_mode", 0},
                {"show_crop_box", false},
                {"use_crop_box", false},
                {"show_ellipsoid", false},
                {"use_ellipsoid", false},
                {"desaturate_unselected", false},
                {"desaturate_cropping", false},
                {"hide_outside_depth_box", false},
                {"crop_filter_for_selection", false},
                {"apply_appearance_correction", false},
                {"ppisp_mode", 1},
                {"ppisp_overrides",
                 {
                     {"exposure_offset", 0.0},
                     {"vignette_enabled", true},
                     {"vignette_strength", 1.0},
                     {"wb_temperature", 0.0},
                     {"wb_tint", 0.0},
                     {"color_red_x", 0.0},
                     {"color_red_y", 0.0},
                     {"color_green_x", 0.0},
                     {"color_green_y", 0.0},
                     {"color_blue_x", 0.0},
                     {"color_blue_y", 0.0},
                     {"gamma_multiplier", 1.0},
                     {"gamma_red", 0.0},
                     {"gamma_green", 0.0},
                     {"gamma_blue", 0.0},
                     {"crf_toe", 0.0},
                     {"crf_shoulder", 0.0},
                 }},
                {"background_color", vec3(0.0, 0.0, 0.0)},
                {"environment_mode", 0},
                {"environment_reference_uuid", nullptr},
                {"environment_builtin",
                 "environments/kloofendal_48d_partly_cloudy_puresky_1k.hdr"},
                {"environment_exposure", 0.0},
                {"environment_rotation_degrees", 0.0},
                {"show_coord_axes", false},
                {"axes_size", 2.0},
                {"axes_visibility",
                 Json::array({true, true, true})},
                {"show_grid", true},
                {"grid_plane", 1},
                {"grid_opacity", 0.5},
                {"point_cloud_mode", false},
                {"voxel_size", 0.01},
                {"show_rings", false},
                {"ring_width", 0.01},
                {"show_center_markers", false},
                {"show_camera_frustums", false},
                {"camera_frustum_scale", 0.25},
                {"train_camera_color", vec3(1.0, 1.0, 1.0)},
                {"eval_camera_color", vec3(1.0, 0.0, 0.0)},
                {"show_pivot", false},
                {"split_view_mode", 0},
                {"gt_comparison_mode", 0},
                {"split_position", 0.5},
                {"split_view_offset", 0},
                {"raster_backend", "3dgs"},
                {"equirectangular", false},
                {"orthographic", false},
                {"ortho_scale", 100.0},
                {"depth_view", false},
                {"depth_view_min", 0.0},
                {"depth_view_max", 100.0},
                {"depth_visualization_mode", 0},
                {"selection_color_committed",
                 vec3(0.859, 0.325, 0.325)},
                {"selection_color_preview",
                 vec3(0.0, 0.871, 0.298)},
                {"selection_color_center_marker",
                 vec3(0.0, 0.604, 0.733)},
                {"depth_clip_enabled", false},
                {"depth_clip_far", 100.0},
                {"mesh_wireframe", false},
                {"mesh_wireframe_color", vec3(0.2, 0.2, 0.2)},
                {"mesh_wireframe_width", 1.0},
                {"mesh_light_dir", vec3(0.3, 1.0, 0.5)},
                {"mesh_light_intensity", 0.7},
                {"mesh_ambient", 0.4},
                {"mesh_backface_culling", true},
                {"mesh_shadow_enabled", false},
                {"mesh_shadow_resolution", 2048},
                {"depth_filter_enabled", false},
                {"depth_filter_min",
                 vec3(-50.0, -10000.0, 0.0)},
                {"depth_filter_max",
                 vec3(50.0, 10000.0, 100.0)},
                {"depth_filter_transform",
                 {
                     {"rotation", identity_rotation()},
                     {"translation", vec3(0.0, 0.0, 0.0)},
                 }},
                {"lod_enabled", false},
                {"lod_auto_enable_rad", false},
                {"lod_max_splats", 2'500'000},
                {"lod_render_scale", 1.0},
                {"lod_behind_camera_penalty", 0.2},
                {"lod_cone_foveation", 0.4},
                {"lod_cone_inner_degrees", 90.0},
                {"lod_cone_outer_degrees", 120.0},
                {"lod_page_pool_splats", 0},
                {"lod_pool_vram_fraction", 0.15},
                {"lod_fade_frames", 12},
                {"lod_debug_colors", false},
            };
        }

        Json default_gui_layout() {
            const Json fixed_arrangement{
                {"right_panel_width", 360.0},
                {"scene_panel_ratio", 0.4},
                {"python_console_width", -1.0},
                {"bottom_dock_height", 320.0},
                {"left_dock_width", 320.0},
                {"sequencer_visible", false},
                {"python_console_visible", false},
            };
            return Json{
                {"version", 1},
                {"layouts",
                 Json::array(
                     {{
                         {"areas",
                          Json::array(
                              {{
                                  {"rect_or_split_position",
                                   {
                                       {"kind", "rect"},
                                       {"x", 0.0},
                                       {"y", 0.0},
                                       {"width", 1.0},
                                       {"height", 1.0},
                                   }},
                                  {"active_space", "viewport"},
                                  {"spaces",
                                   Json::array({
                                       {
                                           {"type", "fixed_arrangement"},
                                           {"version", 1},
                                           {"opaque_payload",
                                            fixed_arrangement},
                                       },
                                       {
                                           {"type", "panel_registry"},
                                           {"version", 1},
                                           {"opaque_payload",
                                            {
                                                {"panels", Json::array()},
                                                {"active_tabs",
                                                 Json::object()},
                                            }},
                                       },
                                       {
                                           {"type", "python_console"},
                                           {"version", 1},
                                           {"opaque_payload",
                                            {
                                                {"active_tab", 0},
                                                {"font_scale", 1.0},
                                            }},
                                       },
                                   })},
                              }})},
                         {"active", true},
                     }})},
            };
        }

        Json default_editor() {
            return Json{
                {"version", 2},
                {"open_files", Json::array()},
                {"active_file", nullptr},
                {"vim_mode", false},
                {"contains_embedded_secrets", false},
            };
        }

        Json default_view() {
            return Json{
                {"version", 1},
                {"render_settings",
                 default_render_settings()},
                {"panel_cameras",
                 Json::array({
                     default_panel_camera("primary"),
                     default_panel_camera("secondary"),
                 })},
                {"navigation",
                 {
                     {"mode", "orbit"},
                     {"view_snap", false},
                 }},
                {"split",
                 {
                     {"focused_panel", "left"},
                     {"gt_camera_id", nullptr},
                     {"panel_grid_planes",
                      Json::array({1, 1})},
                 }},
                {"camera_bookmarks", Json::array()},
                {"tools",
                 {
                     {"active_tool_id", "builtin.select"},
                     {"active_submode_id", "centers"},
                     {"selection_submode", "centers"},
                     {"gizmo_operation", "translate"},
                     {"transform_space", "world"},
                     {"pivot_mode", "origin"},
                     {"multi_transform_mode", "individual"},
                     {"crop_shape", "box"},
                     {"crop_operation", "translate"},
                     {"selection",
                      {
                          {"brush_radius", 20.0},
                          {"crop_filter", false},
                          {"depth_filter", false},
                          {"restrict_to_selected_nodes", false},
                      }},
                 }},
                {"sequencer_view",
                 {{"show_camera_path", true}}},
            };
        }

        Json default_sequencer() {
            return Json{
                {"version", 1},
                {"timeline",
                 {
                     {"version", 1},
                     {"clip_duration", 30.0},
                     {"keyframes", Json::array()},
                 }},
                {"ply_sequences", Json::array()},
                {"playhead", 0.0},
                {"loop_mode", "once"},
                {"playback_speed", 1.0},
                {"preferences",
                 {
                     {"snap_to_grid", false},
                     {"snap_interval", 0.5},
                     {"follow_playback", false},
                     {"show_pip_preview", true},
                     {"pip_preview_scale", 1.0},
                     {"show_film_strip", true},
                 }},
            };
        }

        lfs::Result<Json> root_copy(
            const JsonChapterDom& dom,
            const std::string_view field) {
            auto bytes = dom.to_bytes();
            const std::string text(
                reinterpret_cast<const char*>(bytes.data()),
                bytes.size());
            try {
                auto root = Json::parse(text);
                if (!root.is_object()) {
                    return fail<Json>(
                        lfs::ErrorCode::DataLoss,
                        "Session JSON root must be an object",
                        field);
                }
                return root;
            } catch (
                const nlohmann::json::exception& error) {
                return fail<Json>(
                    lfs::ErrorCode::DataLoss,
                    error.what(), field);
            }
        }

        bool number_array(
            const Json& value,
            const std::size_t size) {
            return value.is_array() &&
                   value.size() == size &&
                   std::ranges::all_of(
                       value, [](const Json& item) {
                           return item.is_number() &&
                                  std::isfinite(
                                      item.get<double>());
                       });
        }

        std::string lowercase(std::string value) {
            std::ranges::transform(
                value, value.begin(),
                [](const unsigned char character) {
                    return static_cast<char>(
                        std::tolower(character));
                });
            return value;
        }

        bool contains_imgui_state(
            const Json& value) {
            if (value.is_string()) {
                return lowercase(
                           value.get<std::string>())
                    .contains("imgui");
            }
            if (value.is_array()) {
                return std::ranges::any_of(
                    value, contains_imgui_state);
            }
            if (!value.is_object())
                return false;
            for (const auto& [key, child] :
                 value.items()) {
                if (lowercase(key).contains("imgui") ||
                    contains_imgui_state(child)) {
                    return true;
                }
            }
            return false;
        }

        bool is_user_global_gui_key(
            const std::string_view key) {
            const auto normalized =
                lowercase(std::string(key));
            return std::ranges::find(
                       kUserGlobalGuiFieldKeys,
                       std::string_view(normalized)) !=
                   kUserGlobalGuiFieldKeys.end();
        }

        bool object_contains_user_global_gui_state(
            const Json& value) {
            if (!value.is_object())
                return false;
            for (const auto& item : value.items()) {
                if (is_user_global_gui_key(item.key()))
                    return true;
            }
            return false;
        }

        void strip_user_global_gui_keys(Json& value) {
            if (!value.is_object())
                return;
            for (auto item = value.begin(); item != value.end();) {
                // DPI belongs to the process window contract and is rejected,
                // never silently normalized out of a project document.
                if (is_user_global_gui_key(item.key()) &&
                    item.key() != "dpi" && item.key() != "dpi_scale")
                    item = value.erase(item);
                else
                    ++item;
            }
        }

        bool contains_user_global_gui_state(const Json& value) {
            if (object_contains_user_global_gui_state(value))
                return true;
            bool found = false;
            for_each_fixed_arrangement_payload(value, [&](const Json& payload) {
                found = found || object_contains_user_global_gui_state(payload);
            });
            return found;
        }

        void strip_user_global_gui_state(Json& value) {
            strip_user_global_gui_keys(value);
            for_each_fixed_arrangement_payload(value, [](Json& payload) {
                strip_user_global_gui_keys(payload);
                // Main-window geometry was project-owned in the original
                // GUIL v1 contract. It is user-global now: accept legacy
                // projects, but discard the field before validation and
                // before the retained DOM can participate in a later save.
                payload.erase("window");
            });
        }

        lfs::Result<void> validate_known_gui_space(
            const Json& space) {
            const auto type =
                space["type"].get<std::string>();
            const auto version =
                space["version"].get<std::int64_t>();
            const auto& payload =
                space["opaque_payload"];
            if (type != "fixed_arrangement" &&
                type != "panel_registry" &&
                type != "python_console") {
                return {};
            }
            if (version != 1 || !payload.is_object()) {
                return fail<void>(
                    lfs::ErrorCode::Unsupported,
                    "Known GUIL spaces require version 1 object payloads",
                    "GUIL.layouts.areas.spaces");
            }
            if (type == "fixed_arrangement") {
                if (payload.contains("window")) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "The fixed GUIL arrangement contains user-global main-window state",
                        "GUIL.layouts.areas.spaces.fixed_arrangement.window");
                }
                return {};
            }
            if (type == "panel_registry") {
                if (!payload.contains("panels") ||
                    !payload["panels"].is_array() ||
                    !payload.contains("active_tabs") ||
                    !payload["active_tabs"].is_object()) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "The GUIL panel registry payload is invalid",
                        "GUIL.layouts.areas.spaces.panel_registry");
                }
                if (payload.contains("panel_payloads")) {
                    if (!payload["panel_payloads"].is_object()) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "GUIL panel_payloads must be an object map",
                            "GUIL.layouts.areas.spaces.panel_registry.panel_payloads");
                    }
                    for (const auto& [id, value] :
                         payload["panel_payloads"].items()) {
                        if (!value.is_object()) {
                            return fail<void>(
                                lfs::ErrorCode::DataLoss,
                                "Each GUIL panel payload must be an object",
                                "GUIL.layouts.areas.spaces.panel_registry.panel_payloads");
                        }
                    }
                }
                return {};
            }
            if (!payload.contains("active_tab") ||
                !payload["active_tab"]
                     .is_number_integer() ||
                !payload.contains("font_scale") ||
                !payload["font_scale"].is_number() ||
                !std::isfinite(
                    payload["font_scale"]
                        .get<double>())) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "The GUIL Python console payload is invalid",
                    "GUIL.layouts.areas.spaces.python_console");
            }
            return {};
        }

        lfs::Result<void> validate_gui_area_tree(
            const Json& areas) {
            const bool has_explicit_identity =
                std::ranges::any_of(
                    areas, [](const Json& area) {
                        return area.is_object() &&
                               (area.contains("id") ||
                                area.contains("parent_id"));
                    });
            if (!has_explicit_identity) {
                if (areas.size() != 1) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "GUIL area tree has multiple implicit roots",
                        "GUIL.layouts.areas");
                }
                return {};
            }

            std::map<std::string, std::optional<std::string>> parents;
            for (const auto& area : areas) {
                if (!area.is_object() || !area.contains("id") ||
                    !area["id"].is_string() ||
                    area["id"].get<std::string>().empty() ||
                    !area.contains("parent_id") ||
                    (!area["parent_id"].is_null() &&
                     !area["parent_id"].is_string())) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "Every explicit GUIL area needs an id and nullable parent_id",
                        "GUIL.layouts.areas");
                }
                const auto id = area["id"].get<std::string>();
                std::optional<std::string> parent;
                if (!area["parent_id"].is_null()) {
                    parent = area["parent_id"].get<std::string>();
                    if (parent->empty()) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "GUIL area parent_id cannot be empty",
                            "GUIL.layouts.areas.parent_id");
                    }
                }
                if (!parents.emplace(id, std::move(parent)).second) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "GUIL area ids must be unique",
                        "GUIL.layouts.areas.id");
                }
            }

            std::size_t roots = 0;
            for (const auto& [id, parent] : parents) {
                if (!parent) {
                    ++roots;
                } else if (!parents.contains(*parent)) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        std::format(
                            "GUIL area {} references missing parent {}",
                            id, *parent),
                        "GUIL.layouts.areas.parent_id");
                }
            }
            if (roots != 1) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    std::format(
                        "GUIL area tree requires exactly one root; found {}",
                        roots),
                    "GUIL.layouts.areas.parent_id");
            }

            for (const auto& [origin, ignored] : parents) {
                (void)ignored;
                std::set<std::string> seen;
                auto current = origin;
                while (true) {
                    if (!seen.insert(current).second) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            std::format(
                                "GUIL area tree contains a cycle through {}",
                                current),
                            "GUIL.layouts.areas.parent_id");
                    }
                    const auto& parent = parents.at(current);
                    if (!parent) {
                        break;
                    }
                    current = *parent;
                }
            }
            return {};
        }

        lfs::Result<void> validate_gui(
            const Json& root) {
            if (root.value("version", 0) != 1) {
                return fail<void>(
                    lfs::ErrorCode::Unsupported,
                    "GUIL version must be 1",
                    "GUIL.version");
            }
            if (contains_imgui_state(root)) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "GUIL cannot contain ImGui state",
                    "GUIL");
            }
            if (contains_user_global_gui_state(
                    root)) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "GUIL cannot contain user-global theme, language, UI/DPI scale, or HUD state",
                    "GUIL");
            }
            const auto layouts =
                root.find("layouts");
            if (layouts == root.end() ||
                !layouts->is_array() ||
                layouts->empty()) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "GUIL.layouts must be a non-empty array",
                    "GUIL.layouts");
            }
            for (std::size_t layout_index = 0;
                 layout_index < layouts->size();
                 ++layout_index) {
                const auto& layout =
                    (*layouts)[layout_index];
                if (!layout.is_object() ||
                    !layout.contains("active") ||
                    !layout["active"].is_boolean() ||
                    !layout.contains("areas") ||
                    !layout["areas"].is_array() ||
                    layout["areas"].empty()) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "Each GUIL layout needs active and non-empty areas",
                        "GUIL.layouts");
                }
                if (auto tree = validate_gui_area_tree(
                        layout["areas"]);
                    !tree) {
                    return tree;
                }
                for (const auto& area :
                     layout["areas"]) {
                    if (!area.is_object() ||
                        !area.contains(
                            "rect_or_split_position") ||
                        !area["rect_or_split_position"].is_object() ||
                        !area["rect_or_split_position"].contains("kind") ||
                        !area["rect_or_split_position"]["kind"].is_string() ||
                        !area.contains(
                            "active_space") ||
                        !area["active_space"].is_string() ||
                        !area.contains("spaces") ||
                        !area["spaces"].is_array()) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "Each GUIL area must use the frozen area-tree shape",
                            "GUIL.layouts.areas");
                    }
                    for (const auto& space :
                         area["spaces"]) {
                        if (!space.is_object() ||
                            !space.contains("type") ||
                            !space["type"].is_string() ||
                            !space.contains("version") ||
                            !space["version"]
                                 .is_number_integer() ||
                            space["version"].get<std::int64_t>() < 1 ||
                            !space.contains(
                                "opaque_payload")) {
                            return fail<void>(
                                lfs::ErrorCode::DataLoss,
                                "Each GUIL space needs type, version, and opaque_payload",
                                "GUIL.layouts.areas.spaces");
                        }
                        if (auto known =
                                validate_known_gui_space(
                                    space);
                            !known) {
                            return known;
                        }
                    }
                }
            }
            return {};
        }

        lfs::Result<void> validate_editor(
            const Json& root) {
            if (root.value("version", 0) != 2) {
                return fail<void>(
                    lfs::ErrorCode::Unsupported,
                    "EDTR version must be 2",
                    "EDTR.version");
            }
            if (!root.contains("open_files") ||
                !root["open_files"].is_array() ||
                !root.contains("vim_mode") ||
                !root["vim_mode"].is_boolean() ||
                !root.contains(
                    "contains_embedded_secrets") ||
                !root["contains_embedded_secrets"]
                     .is_boolean()) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "EDTR requires open_files, vim_mode, and the sharing warning flag",
                    "EDTR");
            }
            bool has_embedded = false;
            for (const auto& file :
                 root["open_files"]) {
                if (!file.is_object() ||
                    !file.contains("locator") ||
                    !file["locator"].is_string() ||
                    !file.contains("modified") ||
                    !file["modified"].is_boolean()) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "Every EDTR file needs locator and modified",
                        "EDTR.open_files");
                }
                if (file["modified"].get<bool>()) {
                    has_embedded = true;
                    if (!file.contains(
                            "embedded_buffer") ||
                        !file["embedded_buffer"]
                             .is_string() ||
                        !file.value(
                            "share_warning", false)) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "Modified EDTR buffers must be embedded and explicitly flagged for sharing",
                            "EDTR.open_files.embedded_buffer");
                    }
                }
            }
            if (has_embedded &&
                !root["contains_embedded_secrets"]
                     .get<bool>()) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "EDTR must advertise its embedded-buffer secret-leak surface",
                    "EDTR.contains_embedded_secrets");
            }
            return {};
        }

        lfs::Result<void> validate_view(
            const Json& root) {
            if (root.value("version", 0) != 1) {
                return fail<void>(
                    lfs::ErrorCode::Unsupported,
                    "VIEW version must be 1",
                    "VIEW.version");
            }
            if (!root.contains(
                    "render_settings") ||
                !root["render_settings"].is_object() ||
                !root.contains("panel_cameras") ||
                !root["panel_cameras"].is_array() ||
                root["panel_cameras"].size() != 2 ||
                !root.contains("navigation") ||
                !root["navigation"].is_object() ||
                !root.contains("split") ||
                !root["split"].is_object() ||
                !root.contains(
                    "camera_bookmarks") ||
                !root["camera_bookmarks"].is_array() ||
                !root.contains("tools") ||
                !root["tools"].is_object()) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "VIEW is missing one of its required state groups",
                    "VIEW");
            }
            const auto& settings =
                root["render_settings"];
            if (!settings.contains(
                    "raster_backend") ||
                !settings["raster_backend"]
                     .is_string() ||
                settings.contains("gut")) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "VIEW must store canonical raster_backend and must not serialize the gut mirror",
                    "VIEW.render_settings.raster_backend");
            }
            std::set<std::string> panels;
            for (const auto& camera :
                 root["panel_cameras"]) {
                if (!camera.is_object() ||
                    !camera.contains("panel") ||
                    !camera["panel"].is_string() ||
                    !camera.contains("R") ||
                    !number_array(camera["R"], 9) ||
                    !camera.contains("t") ||
                    !number_array(camera["t"], 3) ||
                    !camera.contains("pivot") ||
                    !number_array(
                        camera["pivot"], 3) ||
                    !camera.contains("home_R") ||
                    !number_array(
                        camera["home_R"], 9) ||
                    !camera.contains("home_t") ||
                    !number_array(
                        camera["home_t"], 3) ||
                    !camera.contains("home_pivot") ||
                    !number_array(
                        camera["home_pivot"], 3)) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "VIEW panel cameras must store R directly plus t/pivot/home",
                        "VIEW.panel_cameras");
                }
                panels.insert(
                    camera["panel"].get<std::string>());
            }
            if (panels !=
                std::set<std::string>{
                    "primary", "secondary"}) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "VIEW needs exactly primary and secondary camera state",
                    "VIEW.panel_cameras.panel");
            }
            return {};
        }

        lfs::Result<void> validate_sequencer(
            const Json& root) {
            if (root.value("version", 0) != 1) {
                return fail<void>(
                    lfs::ErrorCode::Unsupported,
                    "SEQR version must be 1",
                    "SEQR.version");
            }
            if (!root.contains("timeline") ||
                !root["timeline"].is_object() ||
                !root.contains(
                    "ply_sequences") ||
                !root["ply_sequences"].is_array() ||
                !root.contains("playhead") ||
                !root["playhead"].is_number() ||
                !root.contains("loop_mode") ||
                !root["loop_mode"].is_string() ||
                !root.contains("playback_speed") ||
                !root["playback_speed"].is_number() ||
                !root.contains("preferences") ||
                !root["preferences"].is_object()) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "SEQR is missing inline timeline, clips, controller state, or preferences",
                    "SEQR");
            }
            for (const auto& clip :
                 root["ply_sequences"]) {
                if (!clip.is_object() ||
                    !clip.contains("fps") ||
                    !clip["fps"].is_number() ||
                    clip.contains("sequence_fps")) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "SEQR clip FPS is canonical; the UI mirror must not be serialized",
                        "SEQR.ply_sequences.fps");
                }
            }
            return {};
        }

        std::optional<std::pair<
            std::string, Json>>
        array_identity(const Json& value) {
            if (!value.is_object()) {
                return std::nullopt;
            }
            constexpr std::array keys = {
                "uuid",
                "id",
                "locator",
                "panel",
                "type",
            };
            for (const auto* key : keys) {
                const auto found = value.find(key);
                if (found != value.end() &&
                    (found->is_string() ||
                     found->is_number_integer() ||
                     found->is_number_unsigned())) {
                    return std::pair{
                        std::string(key), *found};
                }
            }
            return std::nullopt;
        }

        void merge_json(
            Json& destination,
            const Json& known) {
            if (destination.is_object() &&
                known.is_object()) {
                for (const auto& [key, value] :
                     known.items()) {
                    auto found =
                        destination.find(key);
                    if (found == destination.end()) {
                        destination[key] = value;
                    } else {
                        merge_json(*found, value);
                    }
                }
                return;
            }
            if (destination.is_array() &&
                known.is_array()) {
                for (std::size_t index = 0;
                     index < known.size(); ++index) {
                    const auto identity =
                        array_identity(known[index]);
                    Json* target = nullptr;
                    if (identity) {
                        for (auto& candidate :
                             destination) {
                            if (!candidate.is_object()) {
                                continue;
                            }
                            const auto found =
                                candidate.find(
                                    identity->first);
                            if (found != candidate.end() &&
                                *found ==
                                    identity->second) {
                                target = &candidate;
                                break;
                            }
                        }
                    } else if (
                        index < destination.size()) {
                        target = &destination[index];
                    }
                    if (!target) {
                        destination.push_back(
                            known[index]);
                    } else {
                        merge_json(*target, known[index]);
                    }
                }
                return;
            }
            destination = known;
        }

        template <std::unsigned_integral T>
        void append_le(
            std::vector<std::byte>& output,
            T value) {
            for (std::size_t byte = 0;
                 byte < sizeof(T); ++byte) {
                output.push_back(
                    static_cast<std::byte>(
                        (value >> (byte * 8u)) &
                        static_cast<T>(0xffu)));
            }
        }

        void append_f32(
            std::vector<std::byte>& output,
            const float value) {
            append_le(
                output,
                std::bit_cast<std::uint32_t>(value));
        }

        void append_f64(
            std::vector<std::byte>& output,
            const double value) {
            append_le(
                output,
                std::bit_cast<std::uint64_t>(value));
        }

        template <std::unsigned_integral T>
        lfs::Result<T> read_le(
            const std::span<const std::byte> bytes,
            std::size_t& offset,
            const std::string_view field) {
            if (offset > bytes.size() ||
                bytes.size() - offset < sizeof(T)) {
                return fail<T>(
                    lfs::ErrorCode::DataLoss,
                    "METR payload is truncated",
                    field);
            }
            T value = 0;
            for (std::size_t byte = 0;
                 byte < sizeof(T); ++byte) {
                value |= static_cast<T>(
                             std::to_integer<
                                 std::uint8_t>(
                                 bytes[offset + byte]))
                         << (byte * 8u);
            }
            offset += sizeof(T);
            return value;
        }

        lfs::Result<float> read_f32(
            const std::span<const std::byte> bytes,
            std::size_t& offset,
            const std::string_view field) {
            auto bits =
                read_le<std::uint32_t>(
                    bytes, offset, field);
            if (!bits) {
                return std::move(bits).error();
            }
            return std::bit_cast<float>(*bits);
        }

        lfs::Result<double> read_f64(
            const std::span<const std::byte> bytes,
            std::size_t& offset,
            const std::string_view field) {
            auto bits =
                read_le<std::uint64_t>(
                    bytes, offset, field);
            if (!bits) {
                return std::move(bits).error();
            }
            return std::bit_cast<double>(*bits);
        }

    } // namespace

    JsonChapterDom default_session_chapter_dom(
        const SessionJsonChapterKind kind) {
        Json root;
        switch (kind) {
        case SessionJsonChapterKind::GuiLayout:
            root = default_gui_layout();
            break;
        case SessionJsonChapterKind::Editor:
            root = default_editor();
            break;
        case SessionJsonChapterKind::View:
            root = default_view();
            break;
        case SessionJsonChapterKind::Sequencer:
            root = default_sequencer();
            break;
        }
        const auto text = root.dump(2);
        auto dom = JsonChapterDom::parse(text);
        if (!dom) {
            std::terminate();
        }
        return std::move(*dom);
    }

    lfs::Result<void>
    sanitize_session_chapter_for_load(
        const SessionJsonChapterKind kind,
        JsonChapterDom& dom) {
        if (kind !=
            SessionJsonChapterKind::GuiLayout) {
            return {};
        }
        auto root = root_copy(dom, "GUIL");
        if (!root) {
            return lfs::Status::failure(
                std::move(root).error());
        }
        strip_user_global_gui_state(*root);
        auto sanitized =
            JsonChapterDom::parse(root->dump(2));
        if (!sanitized) {
            return lfs::Status::failure(
                std::move(sanitized).error());
        }
        dom = std::move(*sanitized);
        return {};
    }

    lfs::Result<void>
    validate_session_chapter_dom(
        const SessionJsonChapterKind kind,
        const JsonChapterDom& dom) {
        auto root = root_copy(dom, "session");
        if (!root) {
            return lfs::Status::failure(
                std::move(root).error());
        }
        switch (kind) {
        case SessionJsonChapterKind::GuiLayout:
            return validate_gui(*root);
        case SessionJsonChapterKind::Editor:
            return validate_editor(*root);
        case SessionJsonChapterKind::View:
            return validate_view(*root);
        case SessionJsonChapterKind::Sequencer:
            return validate_sequencer(*root);
        }
        return fail<void>(
            lfs::ErrorCode::ContractViolation,
            "Unknown session JSON chapter kind",
            "session.kind");
    }

    lfs::Result<void>
    merge_session_chapter_known_state(
        JsonChapterDom& destination,
        const Json& known_state) {
        if (!known_state.is_object()) {
            return fail<void>(
                lfs::ErrorCode::InvalidArgument,
                "Known session state must be a JSON object",
                "session.known_state");
        }
        auto root = root_copy(
            destination, "session");
        if (!root) {
            return lfs::Status::failure(
                std::move(root).error());
        }
        merge_json(*root, known_state);
        auto parsed =
            JsonChapterDom::parse(root->dump(2));
        if (!parsed) {
            return lfs::Status::failure(
                std::move(parsed).error());
        }
        destination = std::move(*parsed);
        return {};
    }

    lfs::Result<void>
    MetricsChapter::validate() const {
        if (loss_history.size() >
                MAX_HISTORY_SAMPLES ||
            psnr_history.size() >
                MAX_HISTORY_SAMPLES) {
            return fail<void>(
                lfs::ErrorCode::ResourceExhausted,
                "METR history count exceeds the safety bound",
                "METR.history_count");
        }
        if (!std::isfinite(
                accumulated_training_seconds) ||
            accumulated_training_seconds < 0.0) {
            return fail<void>(
                lfs::ErrorCode::DataLoss,
                "METR accumulated time must be finite and non-negative",
                "METR.accumulated_training_seconds");
        }
        const auto valid_history =
            [](const auto& history) {
                std::int32_t prior = -1;
                for (const auto& sample :
                     history) {
                    if (sample.iteration < 0 ||
                        sample.iteration < prior ||
                        !std::isfinite(sample.value)) {
                        return false;
                    }
                    prior = sample.iteration;
                }
                return true;
            };
        if (!valid_history(loss_history) ||
            !valid_history(psnr_history)) {
            return fail<void>(
                lfs::ErrorCode::DataLoss,
                "METR histories must be finite and ordered by non-negative iteration",
                "METR.history");
        }
        if (last_evaluation &&
            (last_evaluation->iteration < 0 ||
             !std::isfinite(last_evaluation->psnr) ||
             !std::isfinite(last_evaluation->ssim))) {
            return fail<void>(
                lfs::ErrorCode::DataLoss,
                "METR last evaluation is invalid",
                "METR.last_evaluation");
        }
        if (static_cast<std::uint32_t>(finish_reason) >
            static_cast<std::uint32_t>(
                TrainingFinishReason::Error)) {
            return fail<void>(
                lfs::ErrorCode::DataLoss,
                "METR finish reason is invalid",
                "METR.finish_reason");
        }
        return {};
    }

    lfs::Result<std::vector<std::byte>>
    MetricsChapter::to_bytes() const {
        if (auto valid = validate(); !valid) {
            return std::move(valid).error();
        }
        constexpr std::size_t HEADER_BYTES =
            8 + 4 + 4 + 8 + 8 + 8 + 4 + 4 + 4;
        std::vector<std::byte> output;
        output.reserve(
            HEADER_BYTES +
            (loss_history.size() +
             psnr_history.size()) *
                8);
        output.insert(
            output.end(),
            METR_MAGIC.begin(), METR_MAGIC.end());
        append_le(output, VERSION);
        append_le<std::uint32_t>(
            output,
            last_evaluation ? 1u : 0u);
        append_le<std::uint64_t>(
            output, loss_history.size());
        append_le<std::uint64_t>(
            output, psnr_history.size());
        append_f64(
            output,
            accumulated_training_seconds);
        append_le<std::uint32_t>(
            output,
            static_cast<std::uint32_t>(
                last_evaluation
                    ? last_evaluation->iteration
                    : 0));
        append_f32(
            output,
            last_evaluation
                ? last_evaluation->psnr
                : 0.0f);
        append_f32(
            output,
            last_evaluation
                ? last_evaluation->ssim
                : 0.0f);
        const auto append_history =
            [&](const auto& history) {
                for (const auto& sample :
                     history) {
                    append_le<std::uint32_t>(
                        output,
                        static_cast<std::uint32_t>(
                            sample.iteration));
                    append_f32(
                        output, sample.value);
                }
            };
        append_history(loss_history);
        append_history(psnr_history);
        if (finish_reason !=
            TrainingFinishReason::None) {
            append_le<std::uint32_t>(
                output,
                static_cast<std::uint32_t>(
                    finish_reason));
        }
        return output;
    }

    lfs::Result<MetricsChapter>
    MetricsChapter::from_bytes(
        const std::span<const std::byte> bytes) {
        constexpr std::size_t MIN_BYTES =
            8 + 4 + 4 + 8 + 8 + 8 + 4 + 4 + 4;
        if (bytes.size() < MIN_BYTES ||
            !std::equal(
                METR_MAGIC.begin(),
                METR_MAGIC.end(),
                bytes.begin())) {
            return fail<MetricsChapter>(
                lfs::ErrorCode::DataLoss,
                "METR magic/header is invalid",
                "METR.header");
        }
        std::size_t offset = METR_MAGIC.size();
        auto version =
            read_le<std::uint32_t>(
                bytes, offset, "METR.version");
        auto flags =
            read_le<std::uint32_t>(
                bytes, offset, "METR.flags");
        auto loss_count =
            read_le<std::uint64_t>(
                bytes, offset,
                "METR.loss_count");
        auto psnr_count =
            read_le<std::uint64_t>(
                bytes, offset,
                "METR.psnr_count");
        auto accumulated =
            read_f64(
                bytes, offset,
                "METR.accumulated_training_seconds");
        auto last_iteration =
            read_le<std::uint32_t>(
                bytes, offset,
                "METR.last_evaluation.iteration");
        auto last_psnr =
            read_f32(
                bytes, offset,
                "METR.last_evaluation.psnr");
        auto last_ssim =
            read_f32(
                bytes, offset,
                "METR.last_evaluation.ssim");
        if (!version || !flags || !loss_count ||
            !psnr_count || !accumulated ||
            !last_iteration || !last_psnr ||
            !last_ssim) {
            return fail<MetricsChapter>(
                lfs::ErrorCode::DataLoss,
                "METR fixed header is truncated",
                "METR.header");
        }
        if (*version != VERSION) {
            return fail<MetricsChapter>(
                lfs::ErrorCode::Unsupported,
                std::format(
                    "METR version {} is unsupported",
                    *version),
                "METR.version");
        }
        if ((*flags & ~1u) != 0u) {
            return fail<MetricsChapter>(
                lfs::ErrorCode::DataLoss,
                "METR contains unknown flags",
                "METR.flags");
        }
        if (*loss_count > MAX_HISTORY_SAMPLES ||
            *psnr_count > MAX_HISTORY_SAMPLES) {
            return fail<MetricsChapter>(
                lfs::ErrorCode::ResourceExhausted,
                "METR history count exceeds the safety bound",
                "METR.history_count");
        }
        const auto total_count =
            *loss_count + *psnr_count;
        if (total_count >
            (std::numeric_limits<
                 std::size_t>::max() -
             offset) /
                8u) {
            return fail<MetricsChapter>(
                lfs::ErrorCode::DataLoss,
                "METR payload size does not match its sample counts",
                "METR.history");
        }
        const auto history_bytes =
            total_count * 8u;
        const auto history_end =
            offset + history_bytes;
        if (bytes.size() != history_end &&
            bytes.size() != history_end + 4u) {
            return fail<MetricsChapter>(
                lfs::ErrorCode::DataLoss,
                "METR payload size does not match its sample counts",
                "METR.history");
        }
        if (!std::isfinite(*accumulated) ||
            *accumulated < 0.0) {
            return fail<MetricsChapter>(
                lfs::ErrorCode::DataLoss,
                "METR accumulated time must be finite and non-negative",
                "METR.accumulated_training_seconds");
        }
        if (!std::isfinite(*last_psnr) ||
            !std::isfinite(*last_ssim)) {
            return fail<MetricsChapter>(
                lfs::ErrorCode::DataLoss,
                "METR last-evaluation values must be finite",
                "METR.last_evaluation");
        }
        if ((*flags & 1u) != 0u &&
            *last_iteration >
                static_cast<std::uint32_t>(
                    std::numeric_limits<
                        std::int32_t>::max())) {
            return fail<MetricsChapter>(
                lfs::ErrorCode::DataLoss,
                "METR last evaluation iteration overflows int32",
                "METR.last_evaluation.iteration");
        }

        // Validate the entire payload before reserving either history. Hostile
        // counts, non-finite values, bad ordering, and truncation therefore
        // cannot trigger an attacker-directed allocation.
        std::size_t preflight_offset = offset;
        const auto preflight_history =
            [&](const std::uint64_t count)
            -> lfs::Result<void> {
            std::int32_t prior = -1;
            for (std::uint64_t index = 0;
                 index < count; ++index) {
                auto iteration =
                    read_le<std::uint32_t>(
                        bytes, preflight_offset,
                        "METR.history.iteration");
                auto value =
                    read_f32(
                        bytes, preflight_offset,
                        "METR.history.value");
                if (!iteration || !value ||
                    *iteration >
                        static_cast<std::uint32_t>(
                            std::numeric_limits<
                                std::int32_t>::max()) ||
                    static_cast<std::int32_t>(
                        *iteration) < prior ||
                    !std::isfinite(*value)) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "METR history sample is invalid",
                        "METR.history");
                }
                prior = static_cast<std::int32_t>(
                    *iteration);
            }
            return {};
        };
        if (auto valid =
                preflight_history(*loss_count);
            !valid) {
            return std::move(valid).error();
        }
        if (auto valid =
                preflight_history(*psnr_count);
            !valid) {
            return std::move(valid).error();
        }
        if (preflight_offset != history_end) {
            return fail<MetricsChapter>(
                lfs::ErrorCode::DataLoss,
                "METR payload has trailing or misaligned bytes",
                "METR.history");
        }
        std::optional<std::uint32_t>
            finish_reason_bits;
        if (bytes.size() == history_end + 4u) {
            auto parsed_reason =
                read_le<std::uint32_t>(
                    bytes, preflight_offset,
                    "METR.finish_reason");
            if (!parsed_reason ||
                *parsed_reason >
                    static_cast<std::uint32_t>(
                        TrainingFinishReason::
                            Error)) {
                return fail<MetricsChapter>(
                    lfs::ErrorCode::DataLoss,
                    "METR finish reason is invalid",
                    "METR.finish_reason");
            }
            finish_reason_bits = *parsed_reason;
        }

        MetricsChapter result;
        result.accumulated_training_seconds =
            *accumulated;
        if (finish_reason_bits &&
            *finish_reason_bits != 0u) {
            result.finish_reason =
                static_cast<TrainingFinishReason>(
                    *finish_reason_bits);
        }
        if ((*flags & 1u) != 0u) {
            result.last_evaluation =
                LastEvaluationMetrics{
                    .iteration =
                        static_cast<std::int32_t>(
                            *last_iteration),
                    .psnr = *last_psnr,
                    .ssim = *last_ssim,
                };
        }
        const auto read_history =
            [&](const std::uint64_t count,
                std::vector<MetricHistorySample>&
                    destination)
            -> lfs::Result<void> {
            destination.reserve(
                static_cast<std::size_t>(count));
            for (std::uint64_t index = 0;
                 index < count; ++index) {
                auto iteration =
                    read_le<std::uint32_t>(
                        bytes, offset,
                        "METR.history.iteration");
                auto value =
                    read_f32(
                        bytes, offset,
                        "METR.history.value");
                if (!iteration || !value) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "METR history changed after preflight",
                        "METR.history");
                }
                destination.push_back({
                    .iteration =
                        static_cast<std::int32_t>(
                            *iteration),
                    .value = *value,
                });
            }
            return {};
        };
        if (auto read =
                read_history(
                    *loss_count,
                    result.loss_history);
            !read) {
            return std::move(read).error();
        }
        if (auto read =
                read_history(
                    *psnr_count,
                    result.psnr_history);
            !read) {
            return std::move(read).error();
        }
        if (auto valid = result.validate();
            !valid) {
            return std::move(valid).error();
        }
        return result;
    }

} // namespace lfs::io::project
