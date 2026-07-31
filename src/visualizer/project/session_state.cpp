/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "project/session_state.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_layout.hpp"
#include "gui/panel_registry.hpp"
#include "gui/panels/python_console_panel.hpp"
#include "input/input_controller.hpp"
#include "rendering/render_constants.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "tools/selection_tool.hpp"
#include "tools/unified_tool_registry.hpp"
#include "visualizer_impl.hpp"
#include "window/window_manager.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>

namespace lfs::vis::project {

    namespace {

        using Json = SessionJson;

        lfs::Error session_state_error(
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
                    "The project GUI session cannot be restored.",
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
                    session_state_error(
                        code, std::move(detail), field));
            } else {
                return session_state_error(
                    code, std::move(detail), field);
            }
        }

        lfs::Result<Json> chapter_root(
            const lfs::io::JsonChapterDom& dom,
            const std::string_view chapter) {
            try {
                auto root = Json::parse(dom.dump());
                if (!root.is_object()) {
                    return fail<Json>(
                        lfs::ErrorCode::DataLoss,
                        "Session chapter root is not an object",
                        chapter);
                }
                return root;
            } catch (
                const nlohmann::json::exception& error) {
                return fail<Json>(
                    lfs::ErrorCode::DataLoss,
                    error.what(), chapter);
            }
        }

        template <typename T>
        std::optional<T> scalar(
            const Json& object,
            const std::string_view key) {
            if (!object.is_object())
                return std::nullopt;
            const auto found =
                object.find(std::string(key));
            if (found == object.end())
                return std::nullopt;
            try {
                if constexpr (std::same_as<T, bool>) {
                    if (!found->is_boolean())
                        return std::nullopt;
                } else if constexpr (
                    std::same_as<T, std::string>) {
                    if (!found->is_string())
                        return std::nullopt;
                } else if constexpr (
                    std::integral<T>) {
                    if (!found->is_number_integer() &&
                        !found->is_number_unsigned())
                        return std::nullopt;
                } else if constexpr (
                    std::floating_point<T>) {
                    if (!found->is_number())
                        return std::nullopt;
                }
                const T value = found->get<T>();
                if constexpr (std::floating_point<T>) {
                    if (!std::isfinite(value))
                        return std::nullopt;
                }
                return value;
            } catch (
                const nlohmann::json::exception&) {
                return std::nullopt;
            }
        }

        template <typename T>
        lfs::Result<void> assign_required(
            const Json& object,
            const std::string_view key,
            T& destination,
            const std::string_view prefix =
                "VIEW.render_settings") {
            const auto value = scalar<T>(object, key);
            if (!value) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "Required session field is missing or has the wrong type",
                    std::string(prefix) + "." +
                        std::string(key));
            }
            destination = *value;
            return {};
        }

        template <std::size_t Size>
        std::optional<std::array<float, Size>>
        number_array(const Json& value) {
            if (!value.is_array() ||
                value.size() != Size)
                return std::nullopt;
            std::array<float, Size> result{};
            for (std::size_t index = 0;
                 index < Size; ++index) {
                if (!value[index].is_number())
                    return std::nullopt;
                try {
                    result[index] =
                        value[index].get<float>();
                } catch (
                    const nlohmann::json::exception&) {
                    return std::nullopt;
                }
                if (!std::isfinite(result[index]))
                    return std::nullopt;
            }
            return result;
        }

        template <std::size_t Size>
        Json json_array(
            const std::array<float, Size>& values) {
            Json result = Json::array();
            for (const float value : values)
                result.push_back(value);
            return result;
        }

        Json vec3_json(const glm::vec3& value) {
            return Json::array(
                {value.x, value.y, value.z});
        }

        lfs::Result<glm::vec3> required_vec3(
            const Json& object,
            const std::string_view key,
            const std::string_view prefix =
                "VIEW.render_settings") {
            if (!object.is_object()) {
                return fail<glm::vec3>(
                    lfs::ErrorCode::DataLoss,
                    "Expected an object", prefix);
            }
            const auto found =
                object.find(std::string(key));
            if (found == object.end()) {
                return fail<glm::vec3>(
                    lfs::ErrorCode::DataLoss,
                    "Required vector is missing",
                    std::string(prefix) + "." +
                        std::string(key));
            }
            const auto values = number_array<3>(*found);
            if (!values) {
                return fail<glm::vec3>(
                    lfs::ErrorCode::DataLoss,
                    "Required vector must contain three finite numbers",
                    std::string(prefix) + "." +
                        std::string(key));
            }
            return glm::vec3{
                (*values)[0], (*values)[1], (*values)[2]};
        }

        std::array<float, 9> matrix_array(
            const glm::mat3& value) {
            std::array<float, 9> result{};
            std::size_t index = 0;
            for (std::size_t column = 0;
                 column < 3; ++column) {
                for (std::size_t row = 0;
                     row < 3; ++row) {
                    result[index++] =
                        value[column][row];
                }
            }
            return result;
        }

        glm::mat3 array_matrix(
            const std::array<float, 9>& value) {
            glm::mat3 result{1.0f};
            std::size_t index = 0;
            for (std::size_t column = 0;
                 column < 3; ++column) {
                for (std::size_t row = 0;
                     row < 3; ++row) {
                    result[column][row] =
                        value[index++];
                }
            }
            return result;
        }

        std::array<float, 3> vector_array(
            const glm::vec3& value) {
            return {value.x, value.y, value.z};
        }

        glm::vec3 array_vector(
            const std::array<float, 3>& value) {
            return {value[0], value[1], value[2]};
        }

        template <typename T>
        bool assign_optional(
            const Json& object,
            const std::string_view key,
            T& destination) {
            if (const auto value =
                    scalar<T>(object, key)) {
                destination = *value;
                return true;
            }
            return false;
        }

        Json::const_iterator find_required_object(
            const Json& parent,
            const std::string_view key) {
            if (!parent.is_object())
                return parent.end();
            const auto found =
                parent.find(std::string(key));
            if (found == parent.end() ||
                !found->is_object())
                return parent.end();
            return found;
        }

        Json::const_iterator find_required_array(
            const Json& parent,
            const std::string_view key) {
            if (!parent.is_object())
                return parent.end();
            const auto found =
                parent.find(std::string(key));
            if (found == parent.end() ||
                !found->is_array())
                return parent.end();
            return found;
        }

    } // namespace

    SessionJson renderSettingsToProjectJson(
        const RenderSettings& settings) {
        const auto& ppisp = settings.ppisp_overrides;
        const auto rotation =
            settings.depth_filter_transform
                .getRotationMat();
        const bool packaged_environment =
            settings.environment_map_path ==
            lfs::vis::kDefaultEnvironmentMapPath;

        return Json{
            {"focal_length_mm", settings.focal_length_mm},
            {"scaling_modifier", settings.scaling_modifier},
            {"antialiasing", settings.antialiasing},
            {"mip_filter", settings.mip_filter},
            {"sh_degree", settings.sh_degree},
            {"render_scale", settings.render_scale},
            {"camera_metrics_mode",
             static_cast<int>(
                 settings.camera_metrics_mode)},
            {"show_crop_box", settings.show_crop_box},
            {"use_crop_box", settings.use_crop_box},
            {"show_ellipsoid", settings.show_ellipsoid},
            {"use_ellipsoid", settings.use_ellipsoid},
            {"desaturate_unselected",
             settings.desaturate_unselected},
            {"desaturate_cropping",
             settings.desaturate_cropping},
            {"hide_outside_depth_box",
             settings.hide_outside_depth_box},
            {"crop_filter_for_selection",
             settings.crop_filter_for_selection},
            {"apply_appearance_correction",
             settings.apply_appearance_correction},
            {"ppisp_mode",
             static_cast<int>(settings.ppisp_mode)},
            {"ppisp_overrides",
             {
                 {"exposure_offset",
                  ppisp.exposure_offset},
                 {"vignette_enabled",
                  ppisp.vignette_enabled},
                 {"vignette_strength",
                  ppisp.vignette_strength},
                 {"wb_temperature",
                  ppisp.wb_temperature},
                 {"wb_tint", ppisp.wb_tint},
                 {"color_red_x", ppisp.color_red_x},
                 {"color_red_y", ppisp.color_red_y},
                 {"color_green_x",
                  ppisp.color_green_x},
                 {"color_green_y",
                  ppisp.color_green_y},
                 {"color_blue_x",
                  ppisp.color_blue_x},
                 {"color_blue_y",
                  ppisp.color_blue_y},
                 {"gamma_multiplier",
                  ppisp.gamma_multiplier},
                 {"gamma_red", ppisp.gamma_red},
                 {"gamma_green", ppisp.gamma_green},
                 {"gamma_blue", ppisp.gamma_blue},
                 {"crf_toe", ppisp.crf_toe},
                 {"crf_shoulder", ppisp.crf_shoulder},
             }},
            {"background_color",
             vec3_json(settings.background_color)},
            {"environment_mode",
             static_cast<int>(settings.environment_mode)},
            {"environment_reference_uuid", nullptr},
            {"environment_builtin",
             packaged_environment
                 ? Json(settings.environment_map_path)
                 : Json(nullptr)},
            {"environment_exposure",
             settings.environment_exposure},
            {"environment_rotation_degrees",
             settings.environment_rotation_degrees},
            {"show_coord_axes",
             settings.show_coord_axes},
            {"axes_size", settings.axes_size},
            {"axes_visibility",
             Json::array(
                 {settings.axes_visibility[0],
                  settings.axes_visibility[1],
                  settings.axes_visibility[2]})},
            {"show_grid", settings.show_grid},
            {"grid_plane", settings.grid_plane},
            {"grid_opacity", settings.grid_opacity},
            {"point_cloud_mode",
             settings.point_cloud_mode},
            {"voxel_size", settings.voxel_size},
            {"show_rings", settings.show_rings},
            {"ring_width", settings.ring_width},
            {"show_center_markers",
             settings.show_center_markers},
            {"show_camera_frustums",
             settings.show_camera_frustums},
            {"camera_frustum_scale",
             settings.camera_frustum_scale},
            {"train_camera_color",
             vec3_json(settings.train_camera_color)},
            {"eval_camera_color",
             vec3_json(settings.eval_camera_color)},
            {"show_pivot", settings.show_pivot},
            {"split_view_mode",
             static_cast<int>(settings.split_view_mode)},
            {"gt_comparison_mode",
             static_cast<int>(
                 settings.gt_comparison_mode)},
            {"split_position",
             settings.split_position},
            {"split_view_offset",
             settings.split_view_offset},
            {"raster_backend",
             std::string(
                 lfs::rendering::
                     gaussianRasterBackendId(
                         settings.raster_backend))},
            {"equirectangular",
             settings.equirectangular},
            {"orthographic", settings.orthographic},
            {"ortho_scale", settings.ortho_scale},
            {"depth_view", settings.depth_view},
            {"depth_view_min",
             settings.depth_view_min},
            {"depth_view_max",
             settings.depth_view_max},
            {"depth_visualization_mode",
             static_cast<int>(
                 settings.depth_visualization_mode)},
            {"selection_color_committed",
             vec3_json(
                 settings.selection_color_committed)},
            {"selection_color_preview",
             vec3_json(
                 settings.selection_color_preview)},
            {"selection_color_center_marker",
             vec3_json(
                 settings
                     .selection_color_center_marker)},
            {"depth_clip_enabled",
             settings.depth_clip_enabled},
            {"depth_clip_far",
             settings.depth_clip_far},
            {"mesh_wireframe",
             settings.mesh_wireframe},
            {"mesh_wireframe_color",
             vec3_json(
                 settings.mesh_wireframe_color)},
            {"mesh_wireframe_width",
             settings.mesh_wireframe_width},
            {"mesh_light_dir",
             vec3_json(settings.mesh_light_dir)},
            {"mesh_light_intensity",
             settings.mesh_light_intensity},
            {"mesh_ambient", settings.mesh_ambient},
            {"mesh_backface_culling",
             settings.mesh_backface_culling},
            {"mesh_shadow_enabled",
             settings.mesh_shadow_enabled},
            {"mesh_shadow_resolution",
             settings.mesh_shadow_resolution},
            {"depth_filter_enabled",
             settings.depth_filter_enabled},
            {"depth_filter_min",
             vec3_json(settings.depth_filter_min)},
            {"depth_filter_max",
             vec3_json(settings.depth_filter_max)},
            {"depth_filter_transform",
             {
                 {"rotation",
                  json_array(matrix_array(rotation))},
                 {"translation",
                  vec3_json(
                      settings.depth_filter_transform
                          .getTranslation())},
             }},
            {"lod_enabled", settings.lod_enabled},
            {"lod_auto_enable_rad",
             settings.lod_auto_enable_rad},
            {"lod_max_splats",
             settings.lod_max_splats},
            {"lod_render_scale",
             settings.lod_render_scale},
            {"lod_behind_camera_penalty",
             settings.lod_behind_camera_penalty},
            {"lod_cone_foveation",
             settings.lod_cone_foveation},
            {"lod_cone_inner_degrees",
             settings.lod_cone_inner_degrees},
            {"lod_cone_outer_degrees",
             settings.lod_cone_outer_degrees},
            {"lod_page_pool_splats",
             settings.lod_page_pool_splats},
            {"lod_pool_vram_fraction",
             settings.lod_pool_vram_fraction},
            {"lod_fade_frames",
             settings.lod_fade_frames},
            {"lod_debug_colors",
             settings.lod_debug_colors},
        };
    }

    lfs::Result<RenderSettings>
    renderSettingsFromProjectJson(
        const SessionJson& json,
        const RenderSettings& base) {
        if (!json.is_object()) {
            return fail<RenderSettings>(
                lfs::ErrorCode::DataLoss,
                "VIEW.render_settings must be an object",
                "VIEW.render_settings");
        }
        if (json.contains("gut")) {
            return fail<RenderSettings>(
                lfs::ErrorCode::DataLoss,
                "The derived gut compatibility mirror is forbidden in VIEW",
                "VIEW.render_settings.gut");
        }

        RenderSettings settings = base;

#define LFS_SESSION_ASSIGN(Member)                               \
    do {                                                         \
        if (auto status =                                        \
                assign_required(json, #Member, settings.Member); \
            !status) {                                           \
            return std::move(status).error();                    \
        }                                                        \
    } while (false)

        LFS_SESSION_ASSIGN(focal_length_mm);
        LFS_SESSION_ASSIGN(scaling_modifier);
        LFS_SESSION_ASSIGN(antialiasing);
        LFS_SESSION_ASSIGN(mip_filter);
        LFS_SESSION_ASSIGN(sh_degree);
        LFS_SESSION_ASSIGN(render_scale);
        int enum_value = 0;
        if (auto status = assign_required(
                json, "camera_metrics_mode",
                enum_value);
            !status) {
            return std::move(status).error();
        }
        if (enum_value < 0 || enum_value > 2) {
            return fail<RenderSettings>(
                lfs::ErrorCode::DataLoss,
                "Unsupported camera metrics mode",
                "VIEW.render_settings.camera_metrics_mode");
        }
        settings.camera_metrics_mode =
            static_cast<
                RenderSettings::CameraMetricsMode>(
                enum_value);

        LFS_SESSION_ASSIGN(show_crop_box);
        LFS_SESSION_ASSIGN(use_crop_box);
        LFS_SESSION_ASSIGN(show_ellipsoid);
        LFS_SESSION_ASSIGN(use_ellipsoid);
        LFS_SESSION_ASSIGN(desaturate_unselected);
        LFS_SESSION_ASSIGN(desaturate_cropping);
        LFS_SESSION_ASSIGN(hide_outside_depth_box);
        LFS_SESSION_ASSIGN(crop_filter_for_selection);
        LFS_SESSION_ASSIGN(
            apply_appearance_correction);
        if (auto status = assign_required(
                json, "ppisp_mode", enum_value);
            !status) {
            return std::move(status).error();
        }
        if (enum_value < 0 || enum_value > 1) {
            return fail<RenderSettings>(
                lfs::ErrorCode::DataLoss,
                "Unsupported PPISP mode",
                "VIEW.render_settings.ppisp_mode");
        }
        settings.ppisp_mode =
            static_cast<RenderSettings::PPISPMode>(
                enum_value);

        const auto ppisp_it =
            find_required_object(
                json, "ppisp_overrides");
        if (ppisp_it == json.end()) {
            return fail<RenderSettings>(
                lfs::ErrorCode::DataLoss,
                "VIEW PPISP overrides are missing",
                "VIEW.render_settings.ppisp_overrides");
        }
        auto& ppisp = settings.ppisp_overrides;
#define LFS_SESSION_ASSIGN_PPISP(Member)                 \
    do {                                                 \
        if (auto status = assign_required(               \
                *ppisp_it, #Member, ppisp.Member,        \
                "VIEW.render_settings.ppisp_overrides"); \
            !status) {                                   \
            return std::move(status).error();            \
        }                                                \
    } while (false)
        LFS_SESSION_ASSIGN_PPISP(exposure_offset);
        LFS_SESSION_ASSIGN_PPISP(vignette_enabled);
        LFS_SESSION_ASSIGN_PPISP(vignette_strength);
        LFS_SESSION_ASSIGN_PPISP(wb_temperature);
        LFS_SESSION_ASSIGN_PPISP(wb_tint);
        LFS_SESSION_ASSIGN_PPISP(color_red_x);
        LFS_SESSION_ASSIGN_PPISP(color_red_y);
        LFS_SESSION_ASSIGN_PPISP(color_green_x);
        LFS_SESSION_ASSIGN_PPISP(color_green_y);
        LFS_SESSION_ASSIGN_PPISP(color_blue_x);
        LFS_SESSION_ASSIGN_PPISP(color_blue_y);
        LFS_SESSION_ASSIGN_PPISP(gamma_multiplier);
        LFS_SESSION_ASSIGN_PPISP(gamma_red);
        LFS_SESSION_ASSIGN_PPISP(gamma_green);
        LFS_SESSION_ASSIGN_PPISP(gamma_blue);
        LFS_SESSION_ASSIGN_PPISP(crf_toe);
        LFS_SESSION_ASSIGN_PPISP(crf_shoulder);
#undef LFS_SESSION_ASSIGN_PPISP

        auto vec = required_vec3(
            json, "background_color");
        if (!vec)
            return std::move(vec).error();
        settings.background_color = *vec;

        if (auto status = assign_required(
                json, "environment_mode",
                enum_value);
            !status) {
            return std::move(status).error();
        }
        if (enum_value < 0 || enum_value > 1) {
            return fail<RenderSettings>(
                lfs::ErrorCode::DataLoss,
                "Unsupported environment mode",
                "VIEW.render_settings.environment_mode");
        }
        settings.environment_mode =
            static_cast<EnvironmentBackgroundMode>(
                enum_value);
        if (const auto builtin =
                scalar<std::string>(
                    json, "environment_builtin")) {
            settings.environment_map_path = *builtin;
        }
        LFS_SESSION_ASSIGN(environment_exposure);
        LFS_SESSION_ASSIGN(
            environment_rotation_degrees);
        LFS_SESSION_ASSIGN(show_coord_axes);
        LFS_SESSION_ASSIGN(axes_size);

        const auto axes_it =
            find_required_array(
                json, "axes_visibility");
        if (axes_it == json.end() ||
            axes_it->size() != 3 ||
            !std::ranges::all_of(
                *axes_it, [](const Json& item) {
                    return item.is_boolean();
                })) {
            return fail<RenderSettings>(
                lfs::ErrorCode::DataLoss,
                "Axes visibility must contain three booleans",
                "VIEW.render_settings.axes_visibility");
        }
        for (std::size_t index = 0;
             index < 3; ++index) {
            settings.axes_visibility[index] =
                (*axes_it)[index].get<bool>();
        }

        LFS_SESSION_ASSIGN(show_grid);
        LFS_SESSION_ASSIGN(grid_plane);
        LFS_SESSION_ASSIGN(grid_opacity);
        LFS_SESSION_ASSIGN(point_cloud_mode);
        LFS_SESSION_ASSIGN(voxel_size);
        LFS_SESSION_ASSIGN(show_rings);
        LFS_SESSION_ASSIGN(ring_width);
        LFS_SESSION_ASSIGN(show_center_markers);
        LFS_SESSION_ASSIGN(show_camera_frustums);
        LFS_SESSION_ASSIGN(camera_frustum_scale);
        vec = required_vec3(
            json, "train_camera_color");
        if (!vec)
            return std::move(vec).error();
        settings.train_camera_color = *vec;
        vec = required_vec3(
            json, "eval_camera_color");
        if (!vec)
            return std::move(vec).error();
        settings.eval_camera_color = *vec;
        LFS_SESSION_ASSIGN(show_pivot);

        if (auto status = assign_required(
                json, "split_view_mode",
                enum_value);
            !status) {
            return std::move(status).error();
        }
        if (enum_value < 0 || enum_value > 3) {
            return fail<RenderSettings>(
                lfs::ErrorCode::DataLoss,
                "Unsupported split-view mode",
                "VIEW.render_settings.split_view_mode");
        }
        settings.split_view_mode =
            static_cast<SplitViewMode>(enum_value);
        if (auto status = assign_required(
                json, "gt_comparison_mode",
                enum_value);
            !status) {
            return std::move(status).error();
        }
        settings.gt_comparison_mode =
            static_cast<GTComparisonMode>(enum_value);
        sanitizeGTComparisonSettings(settings);
        LFS_SESSION_ASSIGN(split_position);
        LFS_SESSION_ASSIGN(split_view_offset);

        const auto backend =
            scalar<std::string>(
                json, "raster_backend");
        if (!backend ||
            !lfs::rendering::
                isGaussianRasterBackendId(*backend)) {
            return fail<RenderSettings>(
                lfs::ErrorCode::DataLoss,
                "Unsupported raster backend",
                "VIEW.render_settings.raster_backend");
        }
        settings.raster_backend =
            lfs::rendering::
                gaussianRasterBackendFromId(*backend);
        settings.gut =
            lfs::rendering::isGutBackend(
                settings.raster_backend);
        LFS_SESSION_ASSIGN(equirectangular);
        LFS_SESSION_ASSIGN(orthographic);
        LFS_SESSION_ASSIGN(ortho_scale);
        LFS_SESSION_ASSIGN(depth_view);
        LFS_SESSION_ASSIGN(depth_view_min);
        LFS_SESSION_ASSIGN(depth_view_max);
        if (auto status = assign_required(
                json, "depth_visualization_mode",
                enum_value);
            !status) {
            return std::move(status).error();
        }
        settings.depth_visualization_mode =
            static_cast<
                lfs::rendering::
                    DepthVisualizationMode>(
                enum_value);
        sanitizeDepthViewSettings(settings);

        vec = required_vec3(
            json, "selection_color_committed");
        if (!vec)
            return std::move(vec).error();
        settings.selection_color_committed = *vec;
        vec = required_vec3(
            json, "selection_color_preview");
        if (!vec)
            return std::move(vec).error();
        settings.selection_color_preview = *vec;
        vec = required_vec3(
            json,
            "selection_color_center_marker");
        if (!vec)
            return std::move(vec).error();
        settings.selection_color_center_marker = *vec;
        LFS_SESSION_ASSIGN(depth_clip_enabled);
        LFS_SESSION_ASSIGN(depth_clip_far);
        LFS_SESSION_ASSIGN(mesh_wireframe);
        vec = required_vec3(
            json, "mesh_wireframe_color");
        if (!vec)
            return std::move(vec).error();
        settings.mesh_wireframe_color = *vec;
        LFS_SESSION_ASSIGN(mesh_wireframe_width);
        vec = required_vec3(json, "mesh_light_dir");
        if (!vec)
            return std::move(vec).error();
        settings.mesh_light_dir = *vec;
        LFS_SESSION_ASSIGN(mesh_light_intensity);
        LFS_SESSION_ASSIGN(mesh_ambient);
        LFS_SESSION_ASSIGN(mesh_backface_culling);
        LFS_SESSION_ASSIGN(mesh_shadow_enabled);
        LFS_SESSION_ASSIGN(mesh_shadow_resolution);
        LFS_SESSION_ASSIGN(depth_filter_enabled);
        vec = required_vec3(
            json, "depth_filter_min");
        if (!vec)
            return std::move(vec).error();
        settings.depth_filter_min = *vec;
        vec = required_vec3(
            json, "depth_filter_max");
        if (!vec)
            return std::move(vec).error();
        settings.depth_filter_max = *vec;

        const auto transform_it =
            find_required_object(
                json, "depth_filter_transform");
        if (transform_it == json.end()) {
            return fail<RenderSettings>(
                lfs::ErrorCode::DataLoss,
                "Depth-filter transform is missing",
                "VIEW.render_settings.depth_filter_transform");
        }
        const auto rotation_it =
            transform_it->find("rotation");
        const auto rotation =
            rotation_it == transform_it->end()
                ? std::optional<
                      std::array<float, 9>>{}
                : number_array<9>(*rotation_it);
        if (!rotation) {
            return fail<RenderSettings>(
                lfs::ErrorCode::DataLoss,
                "Depth-filter rotation must be a finite 3x3 matrix",
                "VIEW.render_settings.depth_filter_transform.rotation");
        }
        auto translation = required_vec3(
            *transform_it, "translation",
            "VIEW.render_settings.depth_filter_transform");
        if (!translation)
            return std::move(translation).error();
        settings.depth_filter_transform =
            lfs::geometry::EuclideanTransform(
                glm::quat_cast(
                    array_matrix(*rotation)),
                *translation);

        LFS_SESSION_ASSIGN(lod_enabled);
        LFS_SESSION_ASSIGN(lod_auto_enable_rad);
        LFS_SESSION_ASSIGN(lod_max_splats);
        LFS_SESSION_ASSIGN(lod_render_scale);
        LFS_SESSION_ASSIGN(
            lod_behind_camera_penalty);
        LFS_SESSION_ASSIGN(lod_cone_foveation);
        LFS_SESSION_ASSIGN(
            lod_cone_inner_degrees);
        LFS_SESSION_ASSIGN(
            lod_cone_outer_degrees);
        LFS_SESSION_ASSIGN(lod_page_pool_splats);
        LFS_SESSION_ASSIGN(
            lod_pool_vram_fraction);
        LFS_SESSION_ASSIGN(lod_fade_frames);
        LFS_SESSION_ASSIGN(lod_debug_colors);

#undef LFS_SESSION_ASSIGN

        enforceProjectionBackend(settings);
        settings.gut =
            lfs::rendering::isGutBackend(
                settings.raster_backend);
        return settings;
    }

    PanelCameraProjectState
    capturePanelCameraProjectState(
        const Viewport& viewport) {
        const auto& camera = viewport.camera;
        return {
            .rotation = matrix_array(camera.R),
            .translation = vector_array(camera.t),
            .pivot = vector_array(camera.pivot),
            .home_rotation =
                matrix_array(camera.home_R),
            .home_translation =
                vector_array(camera.home_t),
            .home_pivot =
                vector_array(camera.home_pivot),
            .home_saved = camera.home_saved,
            .zoom_speed = camera.zoomSpeed,
            .max_zoom_speed = camera.maxZoomSpeed,
            .rotate_speed = camera.rotateSpeed,
            .centre_speed =
                camera.rotateCenterSpeed,
            .roll_speed = camera.rotateRollSpeed,
            .translate_speed =
                camera.translateSpeed,
            .wasd_speed = camera.wasdSpeed,
            .max_wasd_speed = camera.maxWasdSpeed,
            .ortho_scale =
                viewport.ortho_scale_override,
        };
    }

    void applyPanelCameraProjectState(
        Viewport& viewport,
        const PanelCameraProjectState& state) {
        viewport.setViewMatrix(
            array_matrix(state.rotation),
            array_vector(state.translation));
        auto& camera = viewport.camera;
        camera.pivot = array_vector(state.pivot);
        camera.home_R =
            array_matrix(state.home_rotation);
        camera.home_t =
            array_vector(state.home_translation);
        camera.home_pivot =
            array_vector(state.home_pivot);
        camera.home_saved = state.home_saved;
        camera.zoomSpeed = state.zoom_speed;
        camera.maxZoomSpeed = state.max_zoom_speed;
        camera.rotateSpeed = state.rotate_speed;
        camera.rotateCenterSpeed =
            state.centre_speed;
        camera.rotateRollSpeed = state.roll_speed;
        camera.translateSpeed =
            state.translate_speed;
        camera.wasdSpeed = state.wasd_speed;
        camera.maxWasdSpeed =
            state.max_wasd_speed;
        viewport.ortho_scale_override =
            state.ortho_scale;
        camera.clearTransientMotion();
    }

    SessionJson panelCameraProjectStateToJson(
        const std::string_view panel,
        const PanelCameraProjectState& state) {
        return Json{
            {"panel", panel},
            {"R", json_array(state.rotation)},
            {"t", json_array(state.translation)},
            {"pivot", json_array(state.pivot)},
            {"home_R",
             json_array(state.home_rotation)},
            {"home_t",
             json_array(state.home_translation)},
            {"home_pivot",
             json_array(state.home_pivot)},
            {"home_saved", state.home_saved},
            {"zoom_speed", state.zoom_speed},
            {"max_zoom_speed",
             state.max_zoom_speed},
            {"rotate_speed", state.rotate_speed},
            {"centre_speed", state.centre_speed},
            {"roll_speed", state.roll_speed},
            {"translate_speed",
             state.translate_speed},
            {"wasd_speed", state.wasd_speed},
            {"max_wasd_speed",
             state.max_wasd_speed},
            {"ortho_scale",
             state.ortho_scale
                 ? Json(*state.ortho_scale)
                 : Json(nullptr)},
        };
    }

    lfs::Result<PanelCameraProjectState>
    panelCameraProjectStateFromJson(
        const SessionJson& json) {
        if (!json.is_object()) {
            return fail<PanelCameraProjectState>(
                lfs::ErrorCode::DataLoss,
                "Panel camera state must be an object",
                "VIEW.panel_cameras");
        }

        PanelCameraProjectState state;
        const auto read_array =
            [&](const std::string_view key,
                auto& destination)
            -> lfs::Result<void> {
            const auto found =
                json.find(std::string(key));
            if (found == json.end()) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "Panel camera field is missing",
                    std::string(
                        "VIEW.panel_cameras.") +
                        std::string(key));
            }
            const auto values =
                number_array<
                    std::tuple_size_v<
                        std::remove_cvref_t<
                            decltype(destination)>>>(
                    *found);
            if (!values) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "Panel camera field must contain finite numbers",
                    std::string(
                        "VIEW.panel_cameras.") +
                        std::string(key));
            }
            destination = *values;
            return {};
        };

        if (auto result =
                read_array("R", state.rotation);
            !result)
            return std::move(result).error();
        if (auto result = read_array(
                "t", state.translation);
            !result)
            return std::move(result).error();
        if (auto result =
                read_array("pivot", state.pivot);
            !result)
            return std::move(result).error();
        if (auto result = read_array(
                "home_R", state.home_rotation);
            !result)
            return std::move(result).error();
        if (auto result = read_array(
                "home_t", state.home_translation);
            !result)
            return std::move(result).error();
        if (auto result = read_array(
                "home_pivot", state.home_pivot);
            !result)
            return std::move(result).error();

#define LFS_CAMERA_ASSIGN(Member)             \
    do {                                      \
        if (auto status = assign_required(    \
                json, #Member, state.Member,  \
                "VIEW.panel_cameras");        \
            !status) {                        \
            return std::move(status).error(); \
        }                                     \
    } while (false)
        LFS_CAMERA_ASSIGN(home_saved);
        LFS_CAMERA_ASSIGN(zoom_speed);
        LFS_CAMERA_ASSIGN(max_zoom_speed);
        LFS_CAMERA_ASSIGN(rotate_speed);
        LFS_CAMERA_ASSIGN(centre_speed);
        LFS_CAMERA_ASSIGN(roll_speed);
        LFS_CAMERA_ASSIGN(translate_speed);
        LFS_CAMERA_ASSIGN(wasd_speed);
        LFS_CAMERA_ASSIGN(max_wasd_speed);
#undef LFS_CAMERA_ASSIGN

        const auto ortho =
            json.find("ortho_scale");
        if (ortho == json.end()) {
            return fail<PanelCameraProjectState>(
                lfs::ErrorCode::DataLoss,
                "Panel camera ortho_scale is missing",
                "VIEW.panel_cameras.ortho_scale");
        }
        if (ortho->is_null()) {
            state.ortho_scale.reset();
        } else if (ortho->is_number()) {
            const auto value = ortho->get<float>();
            if (!std::isfinite(value) ||
                value <= 0.0f) {
                return fail<PanelCameraProjectState>(
                    lfs::ErrorCode::DataLoss,
                    "Panel camera ortho scale must be positive and finite",
                    "VIEW.panel_cameras.ortho_scale");
            }
            state.ortho_scale = value;
        } else {
            return fail<PanelCameraProjectState>(
                lfs::ErrorCode::DataLoss,
                "Panel camera ortho scale must be a number or null",
                "VIEW.panel_cameras.ortho_scale");
        }

        constexpr std::array positive_speeds = {
            &PanelCameraProjectState::zoom_speed,
            &PanelCameraProjectState::max_zoom_speed,
            &PanelCameraProjectState::rotate_speed,
            &PanelCameraProjectState::centre_speed,
            &PanelCameraProjectState::roll_speed,
            &PanelCameraProjectState::translate_speed,
            &PanelCameraProjectState::wasd_speed,
            &PanelCameraProjectState::max_wasd_speed,
        };
        if (std::ranges::any_of(
                positive_speeds,
                [&](const auto member) {
                    return state.*member <= 0.0f;
                })) {
            return fail<PanelCameraProjectState>(
                lfs::ErrorCode::DataLoss,
                "Panel camera speeds must be positive",
                "VIEW.panel_cameras");
        }
        return state;
    }

    namespace {

        bool contains_imgui_token(
            const Json& value) {
            if (value.is_string()) {
                auto text = value.get<std::string>();
                std::ranges::transform(
                    text, text.begin(),
                    [](const unsigned char character) {
                        return static_cast<char>(
                            std::tolower(character));
                    });
                return text.find("imgui") !=
                       std::string::npos;
            }
            if (value.is_array()) {
                return std::ranges::any_of(
                    value, contains_imgui_token);
            }
            if (!value.is_object())
                return false;
            for (const auto& [key, child] :
                 value.items()) {
                auto normalized = key;
                std::ranges::transform(
                    normalized, normalized.begin(),
                    [](const unsigned char character) {
                        return static_cast<char>(
                            std::tolower(character));
                    });
                if (normalized.find("imgui") !=
                        std::string::npos ||
                    contains_imgui_token(child)) {
                    return true;
                }
            }
            return false;
        }

        bool contains_gui_global_field(
            const Json& value) {
            if (value.is_array()) {
                return std::ranges::any_of(
                    value,
                    contains_gui_global_field);
            }
            if (!value.is_object())
                return false;
            constexpr std::array<
                std::string_view, 7>
                excluded = {
                    "theme",
                    "language",
                    "scale",
                    "ui_scale",
                    "hud",
                    "vram_hud",
                    "vram_hud_visible",
                };
            for (const auto& [key, child] :
                 value.items()) {
                auto normalized = key;
                std::ranges::transform(
                    normalized, normalized.begin(),
                    [](const unsigned char character) {
                        return static_cast<char>(
                            std::tolower(character));
                    });
                if (std::ranges::find(
                        excluded, normalized) !=
                        excluded.end() ||
                    contains_gui_global_field(child)) {
                    return true;
                }
            }
            return false;
        }

        lfs::Result<void> validate_gui_runtime(
            const Json& root) {
            if (contains_imgui_token(root)) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "GUIL must be framework-agnostic and cannot contain ImGui state",
                    "GUIL");
            }
            if (contains_gui_global_field(root)) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "GUIL contains user-global theme, language, scale, or HUD state",
                    "GUIL");
            }
            const auto layouts =
                find_required_array(root, "layouts");
            if (layouts == root.end())
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "GUIL layouts are missing",
                    "GUIL.layouts");
            bool has_fixed = false;
            bool has_registry = false;
            bool has_console = false;
            for (const auto& layout : *layouts) {
                const auto areas =
                    find_required_array(
                        layout, "areas");
                if (areas == layout.end())
                    continue;
                for (const auto& area : *areas) {
                    const auto spaces =
                        find_required_array(
                            area, "spaces");
                    if (spaces == area.end())
                        continue;
                    for (const auto& space :
                         *spaces) {
                        const auto type =
                            scalar<std::string>(
                                space, "type");
                        if (!type)
                            continue;
                        has_fixed |=
                            *type ==
                            "fixed_arrangement";
                        has_registry |=
                            *type ==
                            "panel_registry";
                        has_console |=
                            *type ==
                            "python_console";
                    }
                }
            }
            if (!has_fixed || !has_registry ||
                !has_console) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "GUIL v1 requires fixed_arrangement, panel_registry, and python_console spaces",
                    "GUIL.layouts.areas.spaces");
            }
            return {};
        }

        lfs::Result<void> validate_editor_runtime(
            const Json& root) {
            const auto files =
                find_required_array(
                    root, "open_files");
            if (files == root.end())
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "EDTR open_files are missing",
                    "EDTR.open_files");
            constexpr std::size_t
                max_embedded_buffer_bytes =
                    64U * 1024U * 1024U;
            for (const auto& file : *files) {
                if (!file.is_object())
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "EDTR file entry is not an object",
                        "EDTR.open_files");
                if (const auto buffer =
                        scalar<std::string>(
                            file,
                            "embedded_buffer");
                    buffer &&
                    buffer->size() >
                        max_embedded_buffer_bytes) {
                    return fail<void>(
                        lfs::ErrorCode::ResourceExhausted,
                        "Embedded editor buffer exceeds the 64 MiB safety bound",
                        "EDTR.open_files.embedded_buffer");
                }
                if (const auto folds =
                        find_required_array(
                            file, "folds");
                    folds != file.end()) {
                    for (const auto& fold : *folds) {
                        if (!fold.is_object())
                            return fail<void>(
                                lfs::ErrorCode::DataLoss,
                                "EDTR fold entry is not an object",
                                "EDTR.open_files.folds");
                    }
                }
            }
            return {};
        }

        lfs::Result<void> validate_view_runtime(
            const Json& root) {
            const auto settings_it =
                find_required_object(
                    root, "render_settings");
            if (settings_it == root.end())
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "VIEW render settings are missing",
                    "VIEW.render_settings");
            auto settings =
                renderSettingsFromProjectJson(
                    *settings_it);
            if (!settings)
                return lfs::Status::failure(
                    std::move(settings).error());

            const auto cameras =
                find_required_array(
                    root, "panel_cameras");
            if (cameras == root.end() ||
                cameras->size() != 2) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "VIEW must contain two panel cameras",
                    "VIEW.panel_cameras");
            }
            for (const auto& camera : *cameras) {
                auto parsed =
                    panelCameraProjectStateFromJson(
                        camera);
                if (!parsed)
                    return lfs::Status::failure(
                        std::move(parsed).error());
            }

            const auto bookmarks =
                find_required_array(
                    root, "camera_bookmarks");
            if (bookmarks == root.end())
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "VIEW camera bookmarks are missing",
                    "VIEW.camera_bookmarks");
            for (const auto& bookmark : *bookmarks) {
                if (!scalar<std::string>(
                        bookmark, "id") ||
                    !scalar<std::string>(
                        bookmark, "name")) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "VIEW camera bookmark needs id and name",
                        "VIEW.camera_bookmarks");
                }
                auto parsed =
                    panelCameraProjectStateFromJson(
                        bookmark);
                if (!parsed)
                    return lfs::Status::failure(
                        std::move(parsed).error());
            }
            return {};
        }

        lfs::Result<void> validate_sequencer_runtime(
            const Json& root) {
            const auto timeline_it =
                find_required_object(
                    root, "timeline");
            if (timeline_it == root.end())
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "SEQR inline timeline is missing",
                    "SEQR.timeline");
            lfs::sequencer::Timeline timeline;
            const auto standard_json =
                nlohmann::json::parse(
                    timeline_it->dump());
            if (!timeline.loadFromJson(
                    standard_json)) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "SEQR inline timeline failed semantic validation",
                    "SEQR.timeline");
            }
            const auto clips =
                find_required_array(
                    root, "ply_sequences");
            if (clips == root.end())
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "SEQR PLY clips are missing",
                    "SEQR.ply_sequences");
            for (const auto& clip : *clips) {
                const auto fps =
                    scalar<float>(clip, "fps");
                const auto frames =
                    find_required_array(
                        clip, "frames");
                if (!fps || *fps < MIN_SEQUENCE_FPS ||
                    *fps > MAX_SEQUENCE_FPS ||
                    frames == clip.end()) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "SEQR PLY clip has invalid FPS or frame list",
                        "SEQR.ply_sequences");
                }
                for (const auto& frame : *frames) {
                    const auto uuid =
                        scalar<std::string>(
                            frame, "node_uuid");
                    if (!scalar<std::string>(
                            frame, "locator") ||
                        !scalar<std::string>(
                            frame, "node_name") ||
                        !uuid ||
                        !lfs::core::Uuid::from_string(
                            *uuid)) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "SEQR PLY frame identity is invalid",
                            "SEQR.ply_sequences.frames");
                    }
                }
            }
            const auto playhead =
                scalar<float>(root, "playhead");
            const auto speed =
                scalar<float>(
                    root, "playback_speed");
            if (!playhead || !speed) {
                return fail<void>(
                    lfs::ErrorCode::DataLoss,
                    "SEQR playhead or playback speed is invalid",
                    "SEQR");
            }
            return {};
        }

    } // namespace

    lfs::Result<PreparedGuiSessionRestore>
    prepareGuiSessionRestore(
        lfs::io::project::ProjectSessionChapters
            chapters) {
        if (auto valid =
                chapters.gui_layout.validate();
            !valid)
            return std::move(valid).error();
        if (auto valid = chapters.editor.validate();
            !valid)
            return std::move(valid).error();
        if (auto valid = chapters.view.validate();
            !valid)
            return std::move(valid).error();
        if (auto valid =
                chapters.sequencer.validate();
            !valid)
            return std::move(valid).error();
        if (auto valid = chapters.metrics.validate();
            !valid)
            return std::move(valid).error();

        auto gui = chapter_root(
            chapters.gui_layout.dom(), "GUIL");
        if (!gui)
            return std::move(gui).error();
        if (auto valid = validate_gui_runtime(*gui);
            !valid)
            return std::move(valid).error();

        auto editor = chapter_root(
            chapters.editor.dom(), "EDTR");
        if (!editor)
            return std::move(editor).error();
        if (auto valid =
                validate_editor_runtime(*editor);
            !valid)
            return std::move(valid).error();

        auto view = chapter_root(
            chapters.view.dom(), "VIEW");
        if (!view)
            return std::move(view).error();
        if (auto valid = validate_view_runtime(*view);
            !valid)
            return std::move(valid).error();

        auto sequencer = chapter_root(
            chapters.sequencer.dom(), "SEQR");
        if (!sequencer)
            return std::move(sequencer).error();
        if (auto valid =
                validate_sequencer_runtime(
                    *sequencer);
            !valid)
            return std::move(valid).error();

        return PreparedGuiSessionRestore{
            .chapters = std::move(chapters)};
    }

    bool pluginPreloadTerminalForGuiPanels(
        const bool start_attempted,
        const std::string_view state) noexcept {
        if (!start_attempted)
            return false;
        return state == "completed" ||
               state == "cancelled" ||
               state == "not_started";
    }

    lfs::Result<void>
    GuiSessionRestoreCoordinator::stage(
        lfs::io::project::ProjectSessionChapters
            chapters) {
        auto prepared =
            prepareGuiSessionRestore(
                std::move(chapters));
        if (!prepared)
            return lfs::Status::failure(
                std::move(prepared).error());
        pending_ = std::move(*prepared);
        return {};
    }

    void GuiSessionRestoreCoordinator::
        onFirstGuiFrame() {
        first_gui_frame_ready_ = true;
    }

    void GuiSessionRestoreCoordinator::onPanelsReady(
        const std::uint64_t
            registration_revision) {
        panels_ready_ = true;
        panels_registration_revision_ =
            registration_revision;
    }

    bool GuiSessionRestoreCoordinator::ready()
        const noexcept {
        return pending_.has_value() &&
               first_gui_frame_ready_ &&
               panels_ready_;
    }

    std::optional<PreparedGuiSessionRestore>
    GuiSessionRestoreCoordinator::takeReady() {
        if (!ready())
            return std::nullopt;
        auto result = std::move(pending_);
        pending_.reset();
        return result;
    }

    void GuiSessionRestoreCoordinator::clear() noexcept {
        pending_.reset();
    }

    namespace {

        std::string panel_space_name(
            const gui::PanelSpace space) {
            switch (space) {
            case gui::PanelSpace::SidePanel:
                return "side_panel";
            case gui::PanelSpace::Floating:
                return "floating";
            case gui::PanelSpace::ViewportOverlay:
                return "viewport_overlay";
            case gui::PanelSpace::MainPanelTab:
                return "main_panel_tab";
            case gui::PanelSpace::SceneHeader:
                return "scene_header";
            case gui::PanelSpace::BottomDock:
                return "bottom_dock";
            case gui::PanelSpace::LeftDock:
                return "left_dock";
            case gui::PanelSpace::StatusBar:
                return "status_bar";
            }
            return "floating";
        }

        std::optional<gui::PanelSpace>
        panel_space_from_name(
            const std::string_view name) {
            if (name == "side_panel")
                return gui::PanelSpace::SidePanel;
            if (name == "floating")
                return gui::PanelSpace::Floating;
            if (name == "viewport_overlay")
                return gui::PanelSpace::ViewportOverlay;
            if (name == "main_panel_tab")
                return gui::PanelSpace::MainPanelTab;
            if (name == "scene_header")
                return gui::PanelSpace::SceneHeader;
            if (name == "bottom_dock")
                return gui::PanelSpace::BottomDock;
            if (name == "left_dock")
                return gui::PanelSpace::LeftDock;
            if (name == "status_bar")
                return gui::PanelSpace::StatusBar;
            return std::nullopt;
        }

        Json finite_or_null(const float value) {
            return std::isfinite(value)
                       ? Json(value)
                       : Json(nullptr);
        }

        std::string selection_submode_name(
            const SelectionSubMode mode) {
            switch (mode) {
            case SelectionSubMode::Centers:
                return "centers";
            case SelectionSubMode::Rectangle:
                return "rectangle";
            case SelectionSubMode::Polygon:
                return "polygon";
            case SelectionSubMode::Lasso:
                return "lasso";
            case SelectionSubMode::Rings:
                return "rings";
            case SelectionSubMode::Color:
                return "color";
            case SelectionSubMode::Box:
                return "box";
            case SelectionSubMode::Sphere:
                return "sphere";
            }
            return "centers";
        }

        std::optional<SelectionSubMode>
        selection_submode_from_name(
            const std::string_view name) {
            if (name == "centers")
                return SelectionSubMode::Centers;
            if (name == "rectangle")
                return SelectionSubMode::Rectangle;
            if (name == "polygon")
                return SelectionSubMode::Polygon;
            if (name == "lasso")
                return SelectionSubMode::Lasso;
            if (name == "rings")
                return SelectionSubMode::Rings;
            if (name == "color")
                return SelectionSubMode::Color;
            if (name == "box")
                return SelectionSubMode::Box;
            if (name == "sphere")
                return SelectionSubMode::Sphere;
            return std::nullopt;
        }

        std::string gizmo_operation_name(
            const gui::GizmoOperation operation) {
            switch (operation) {
            case gui::GizmoOperation::Translate:
                return "translate";
            case gui::GizmoOperation::Rotate:
                return "rotate";
            case gui::GizmoOperation::Scale:
                return "scale";
            }
            return "translate";
        }

        std::string transform_space_name(
            const TransformSpace space) {
            return space == TransformSpace::World
                       ? "world"
                       : "local";
        }

        std::string pivot_mode_name(
            const PivotMode mode) {
            return mode == PivotMode::BoundsCenter
                       ? "bounds_center"
                       : "origin";
        }

        std::string multi_transform_mode_name(
            const gui::MultiTransformMode mode) {
            return mode ==
                           gui::MultiTransformMode::
                               Individual
                       ? "individual"
                       : "selection";
        }

        std::string loop_mode_name(
            const LoopMode mode) {
            switch (mode) {
            case LoopMode::ONCE: return "once";
            case LoopMode::LOOP: return "loop";
            case LoopMode::PING_PONG:
                return "ping_pong";
            }
            return "once";
        }

        Json editor_session_json(
            const editor::PythonEditorSessionState&
                state) {
            Json folds = Json::array();
            for (const auto& fold : state.folds) {
                folds.push_back({
                    {"start_byte", fold.start_byte},
                    {"end_byte", fold.end_byte},
                    {"start_line", fold.start_line},
                    {"end_line", fold.end_line},
                    {"kind", fold.kind},
                    {"collapsed", fold.collapsed},
                });
            }
            return Json{
                {"cursor_byte", state.cursor_byte},
                {"selection_anchor_byte",
                 state.selection_anchor_byte
                     ? Json(
                           *state
                                .selection_anchor_byte)
                     : Json(nullptr)},
                {"scroll",
                 {
                     {"x", state.scroll_x},
                     {"y", state.scroll_y},
                 }},
                {"folds", std::move(folds)},
            };
        }

        Json find_space_payload(
            const Json& root,
            const std::string_view type) {
            const auto layouts =
                find_required_array(root, "layouts");
            if (layouts == root.end())
                return Json::object();
            for (const auto& layout : *layouts) {
                if (const auto active =
                        scalar<bool>(
                            layout, "active");
                    active && !*active)
                    continue;
                const auto areas =
                    find_required_array(
                        layout, "areas");
                if (areas == layout.end())
                    continue;
                for (const auto& area : *areas) {
                    const auto spaces =
                        find_required_array(
                            area, "spaces");
                    if (spaces == area.end())
                        continue;
                    for (const auto& space :
                         *spaces) {
                        const auto space_type =
                            scalar<std::string>(
                                space, "type");
                        if (!space_type ||
                            *space_type != type)
                            continue;
                        const auto payload =
                            space.find(
                                "opaque_payload");
                        if (payload !=
                            space.end()) {
                            return *payload;
                        }
                    }
                }
            }
            return Json::object();
        }

    } // namespace

    lfs::Result<
        lfs::io::project::ProjectSessionChapters>
    captureGuiSession(
        const VisualizerImpl& viewer,
        const lfs::io::project::
            ProjectSessionChapters& retained,
        const std::vector<
            CameraBookmarkProjectState>& bookmarks) {
        auto result = retained;
        const auto* gui_manager =
            viewer.getGuiManager();
        const auto* window_manager =
            viewer.getWindowManager();
        const auto* rendering_manager =
            viewer.getRenderingManager();
        const auto* input_controller =
            viewer.getInputController();
        if (!gui_manager || !window_manager ||
            !rendering_manager ||
            !input_controller) {
            return fail<
                lfs::io::project::
                    ProjectSessionChapters>(
                lfs::ErrorCode::FailedPrecondition,
                "GUI session capture requires initialized GUI, window, renderer, and input owners",
                "session.capture");
        }

        const auto layout =
            gui_manager->panelLayout()
                .captureProjectState();
        const auto window =
            window_manager->captureProjectState();
        const auto& window_states =
            gui_manager->getWindowStates();
        const auto console_visible =
            window_states.contains(
                "python_console") &&
            window_states.at("python_console");
        const Json fixed_payload{
            {"right_panel_width",
             layout.right_panel_width},
            {"scene_panel_ratio",
             layout.scene_panel_ratio},
            {"python_console_width",
             layout.python_console_width},
            {"bottom_dock_height",
             layout.bottom_dock_height},
            {"left_dock_width",
             layout.left_dock_width},
            {"sequencer_visible",
             layout.show_sequencer},
            {"python_console_visible",
             console_visible},
            {"window",
             {
                 {"x", window.x},
                 {"y", window.y},
                 {"width", window.width},
                 {"height", window.height},
                 {"fullscreen",
                  window.fullscreen},
                 {"maximized", window.maximized},
                 {"restore_x", window.restore_x},
                 {"restore_y", window.restore_y},
                 {"restore_width",
                  window.restore_width},
                 {"restore_height",
                  window.restore_height},
             }},
        };

        Json panels = Json::array();
        for (const auto& panel :
             gui::PanelRegistry::instance()
                 .capture_project_state()) {
            panels.push_back({
                {"id", panel.id},
                {"parent_id", panel.parent_id},
                {"space",
                 panel_space_name(panel.space)},
                {"order", panel.order},
                {"enabled", panel.enabled},
                {"float_x",
                 finite_or_null(panel.float_x)},
                {"float_y",
                 finite_or_null(panel.float_y)},
                {"float_user_height",
                 panel.float_user_height},
                {"float_last_bounds_valid",
                 panel.float_last_bounds_valid},
                {"float_last_x",
                 panel.float_last_x},
                {"float_last_y",
                 panel.float_last_y},
                {"float_last_w",
                 panel.float_last_w},
                {"float_last_h",
                 panel.float_last_h},
                {"float_auto_center",
                 panel.float_auto_center},
                {"float_stack_order",
                 panel.float_stack_order},
            });
        }
        const Json registry_payload{
            {"panels", std::move(panels)},
            {"active_tabs",
             {
                 {"main_panel",
                  layout.active_tab_id},
                 {"scene_panel",
                  gui_manager
                      ->scenePanelActiveTab()},
             }},
        };

        Json console_payload{
            {"active_tab", 0},
            {"font_scale", 1.0f},
        };
        if (const auto* console =
                gui::panels::PythonConsoleState::
                    tryGetInstance()) {
            console_payload["active_tab"] =
                console->getActiveTab();
            console_payload["font_scale"] =
                console->getFontScale();
        }

        const Json gui_known{
            {"version", 1},
            {"layouts",
             Json::array({
                 {
                     {"areas",
                      Json::array({
                          {
                              {"rect_or_split_position",
                               {
                                   {"kind", "rect"},
                                   {"x", 0.0f},
                                   {"y", 0.0f},
                                   {"width", 1.0f},
                                   {"height", 1.0f},
                               }},
                              {"active_space",
                               "viewport"},
                              {"spaces",
                               Json::array({
                                   {
                                       {"type",
                                        "fixed_arrangement"},
                                       {"version", 1},
                                       {"opaque_payload",
                                        fixed_payload},
                                   },
                                   {
                                       {"type",
                                        "panel_registry"},
                                       {"version", 1},
                                       {"opaque_payload",
                                        registry_payload},
                                   },
                                   {
                                       {"type",
                                        "python_console"},
                                       {"version", 1},
                                       {"opaque_payload",
                                        console_payload},
                                   },
                               })},
                          },
                      })},
                     {"active", true},
                 },
             })},
        };
        if (auto merged =
                result.gui_layout.merge_known_state(
                    gui_known);
            !merged) {
            return std::move(merged).error();
        }

        if (const auto* console =
                gui::panels::PythonConsoleState::
                    tryGetInstance()) {
            const auto* python_editor =
                console->getEditor();
            const auto active_locator =
                console->getScriptPath().empty()
                    ? std::string(
                          "untitled://python")
                    : lfs::core::path_to_utf8(
                          console->getScriptPath());
            editor::
                PythonEditorWorkspaceSessionState
                    workspace;
            if (python_editor) {
                workspace =
                    python_editor
                        ->captureWorkspaceSessionState(
                            active_locator,
                            console
                                ->isModified());
            }

            Json live_files = Json::array();
            bool has_embedded = false;
            for (const auto& open_file :
                 workspace.open_files) {
                Json file{
                    {"locator",
                     open_file.locator},
                    {"modified",
                     open_file.modified},
                };
                const auto session =
                    editor_session_json(
                        open_file.editor);
                for (const auto& [key, value] :
                     session.items()) {
                    file[key] = value;
                }
                if (open_file.modified) {
                    file["embedded_buffer"] =
                        open_file.text;
                    file["share_warning"] = true;
                    has_embedded = true;
                }
                live_files.push_back(
                    std::move(file));
            }
            Json editor_known{
                {"version", 2},
                {"open_files", live_files},
                {"active_file",
                 workspace.active_file
                     ? Json(
                           *workspace
                                .active_file)
                     : Json(nullptr)},
                {"vim_mode",
                 workspace.vim_mode},
                {"contains_embedded_secrets",
                 has_embedded},
            };
            if (auto merged =
                    result.editor.merge_known_state(
                        editor_known);
                !merged) {
                return std::move(merged).error();
            }
            auto files =
                result.editor.dom().get_json(
                    "open_files");
            if (files && files->is_array()) {
                Json current_files = Json::array();
                for (const auto& live_file :
                     workspace.open_files) {
                    const auto found =
                        std::ranges::find_if(
                            *files,
                            [&](const Json&
                                    entry) {
                                return entry
                                           .is_object() &&
                                       entry.value(
                                           "locator",
                                           std::string{}) ==
                                           live_file
                                               .locator;
                            });
                    if (found == files->end()) {
                        continue;
                    }
                    auto entry = *found;
                    if (!live_file.modified) {
                        entry.erase(
                            "embedded_buffer");
                        entry.erase(
                            "share_warning");
                    }
                    current_files.push_back(
                        std::move(entry));
                }
                if (auto set =
                        result.editor.dom()
                            .set_json(
                                "open_files",
                                std::move(
                                    current_files));
                    !set) {
                    return std::move(set).error();
                }
            }
            if (auto set =
                    result.editor.dom().set(
                        "contains_embedded_secrets",
                        has_embedded);
                !set) {
                return std::move(set).error();
            }
        }

        const auto settings =
            rendering_manager->getSettings();
        Json bookmarks_json = Json::array();
        for (const auto& bookmark : bookmarks) {
            auto item =
                panelCameraProjectStateToJson(
                    "bookmark", bookmark.camera);
            item.erase("panel");
            item["id"] = bookmark.id;
            item["name"] = bookmark.name;
            bookmarks_json.push_back(
                std::move(item));
        }

        const auto primary =
            capturePanelCameraProjectState(
                viewer.getViewport());
        const auto secondary =
            capturePanelCameraProjectState(
                rendering_manager
                    ->projectSecondaryViewport());
        const auto& tool_registry =
            UnifiedToolRegistry::instance();
        const auto& gizmo =
            gui_manager->gizmo();
        const auto* selection_tool =
            viewer.getSelectionTool();
        const auto& sequencer_ui =
            gui_manager->getSequencerUIState();
        auto project_render_settings =
            renderSettingsToProjectJson(settings);
        if (!settings.environment_map_path.empty() &&
            settings.environment_map_path !=
                lfs::vis::kDefaultEnvironmentMapPath) {
            const auto retained_reference =
                result.view.dom().get_json(
                    "render_settings.environment_reference_uuid");
            if (retained_reference &&
                retained_reference->is_string()) {
                project_render_settings
                    ["environment_reference_uuid"] =
                        *retained_reference;
            }
        }
        const Json view_known{
            {"version", 1},
            {"render_settings",
             std::move(project_render_settings)},
            {"panel_cameras",
             Json::array({
                 panelCameraProjectStateToJson(
                     "primary", primary),
                 panelCameraProjectStateToJson(
                     "secondary", secondary),
             })},
            {"navigation",
             {
                 {"mode",
                  InputController::
                      cameraNavigationModeName(
                          input_controller
                              ->cameraNavigationMode())},
                 {"view_snap",
                  input_controller
                      ->cameraViewSnapEnabled()},
             }},
            {"split",
             {
                 {"focused_panel",
                  rendering_manager
                              ->getFocusedSplitPanel() ==
                          SplitViewPanelId::Right
                      ? "right"
                      : "left"},
                 {"gt_camera_id",
                  rendering_manager
                              ->getCurrentCameraId() >=
                          0
                      ? Json(
                            rendering_manager
                                ->getCurrentCameraId())
                      : Json(nullptr)},
                 {"panel_grid_planes",
                  Json::array({
                      rendering_manager
                          ->getGridPlaneForPanel(
                              SplitViewPanelId::
                                  Left),
                      rendering_manager
                          ->getGridPlaneForPanel(
                              SplitViewPanelId::
                                  Right),
                  })},
             }},
            {"camera_bookmarks",
             std::move(bookmarks_json)},
            {"tools",
             {
                 {"active_tool_id",
                  std::string(
                      tool_registry
                          .getActiveTool())},
                 {"selection_submode",
                  selection_submode_name(
                      gizmo
                          .getSelectionSubMode())},
                 {"active_submode_id",
                  std::string(
                      tool_registry
                          .getActiveSubmode())},
                 {"gizmo_operation",
                  gizmo_operation_name(
                      gizmo.getOperation())},
                 {"transform_space",
                  transform_space_name(
                      gizmo
                          .getTransformSpace())},
                 {"pivot_mode",
                  pivot_mode_name(
                      gizmo.getPivotMode())},
                 {"multi_transform_mode",
                  multi_transform_mode_name(
                      gizmo
                          .getMultiTransformMode())},
                 {"crop_shape",
                  gizmo.cropToolShape()},
                 {"crop_operation",
                  gizmo.cropToolOperation()},
                 {"selection",
                  {
                      {"brush_radius",
                       selection_tool
                           ? selection_tool
                                 ->getBrushRadius()
                           : 20.0f},
                      {"crop_filter",
                       selection_tool &&
                           selection_tool
                               ->isCropFilterEnabled()},
                      {"depth_filter",
                       selection_tool &&
                           selection_tool
                               ->isDepthFilterEnabled()},
                      {"restrict_to_selected_nodes",
                       !selection_tool ||
                           selection_tool
                               ->restrictToSelectedNodes()},
                  }},
             }},
            {"sequencer_view",
             {
                 {"show_camera_path",
                  sequencer_ui
                      .show_camera_path},
             }},
        };
        if (auto merged =
                result.view.merge_known_state(
                    view_known);
            !merged) {
            return std::move(merged).error();
        }
        if (auto merged_bookmarks =
                result.view.dom().get_json(
                    "camera_bookmarks");
            merged_bookmarks &&
            merged_bookmarks->is_array()) {
            Json current_bookmarks = Json::array();
            for (auto& entry :
                 *merged_bookmarks) {
                if (!entry.is_object())
                    continue;
                const auto id = entry.value(
                    "id", std::string{});
                if (std::ranges::any_of(
                        bookmarks,
                        [&id](
                            const auto& bookmark) {
                            return bookmark.id == id;
                        })) {
                    current_bookmarks.push_back(
                        std::move(entry));
                }
            }
            if (auto set =
                    result.view.dom().set_json(
                        "camera_bookmarks",
                        std::move(
                            current_bookmarks));
                !set) {
                return std::move(set).error();
            }
        }

        const auto& controller =
            gui_manager->sequencer();
        Json timeline = Json::parse(
            controller.saveToJson().dump());
        Json clips = Json::array();
        if (const auto* clip =
                controller.plySequence()) {
            const auto retained_clips =
                result.sequencer.dom()
                    .get_json("ply_sequences");
            Json frames = Json::array();
            for (const auto& frame :
                 clip->frames) {
                frames.push_back({
                    {"locator",
                     lfs::core::path_to_utf8(
                         frame.path.filename())},
                    {"node_name",
                     frame.node_name},
                    {"node_uuid",
                     frame.node_uuid
                         .to_string()},
                });
            }
            Json saved_clip{
                {"node_name", clip->node_name},
                {"node_uuid",
                 clip->node_uuid.to_string()},
                {"directory_reference_uuid",
                 nullptr},
                {"directory_hint",
                 lfs::core::path_to_utf8(
                     clip->directory
                         .filename())},
                {"frames", std::move(frames)},
                {"fps", clip->fps},
            };
            if (retained_clips &&
                retained_clips->is_array()) {
                const auto retained =
                    std::ranges::find_if(
                        *retained_clips,
                        [&](const Json& item) {
                            return item.is_object() &&
                                   item.value(
                                       "node_uuid",
                                       std::string{}) ==
                                       clip->node_uuid
                                           .to_string();
                        });
                if (retained !=
                    retained_clips->end()) {
                    const auto reference =
                        retained->find(
                            "directory_reference_uuid");
                    if (reference !=
                            retained->end() &&
                        reference->is_string()) {
                        saved_clip
                            ["directory_reference_uuid"] =
                                *reference;
                    }
                }
            }
            clips.push_back(std::move(saved_clip));
        }
        const Json sequencer_known{
            {"version", 1},
            {"timeline", std::move(timeline)},
            {"ply_sequences", std::move(clips)},
            {"playhead", controller.playhead()},
            {"loop_mode",
             loop_mode_name(
                 controller.loopMode())},
            {"playback_speed",
             controller.playbackSpeed()},
            {"preferences",
             {
                 {"snap_to_grid",
                  sequencer_ui.snap_to_grid},
                 {"snap_interval",
                  sequencer_ui.snap_interval},
                 {"follow_playback",
                  sequencer_ui.follow_playback},
                 {"show_pip_preview",
                  sequencer_ui.show_pip_preview},
                 {"pip_preview_scale",
                  sequencer_ui
                      .pip_preview_scale},
                 {"show_film_strip",
                  sequencer_ui.show_film_strip},
             }},
        };
        if (auto merged =
                result.sequencer
                    .merge_known_state(
                        sequencer_known);
            !merged) {
            return std::move(merged).error();
        }
        if (auto merged_clips =
                result.sequencer.dom()
                    .get_json(
                        "ply_sequences");
            merged_clips &&
            merged_clips->is_array()) {
            Json current_clips = Json::array();
            if (const auto* clip =
                    controller.plySequence()) {
                const auto clip_uuid =
                    clip->node_uuid.to_string();
                for (auto& entry :
                     *merged_clips) {
                    if (entry.is_object() &&
                        entry.value(
                            "node_uuid",
                            std::string{}) ==
                            clip_uuid) {
                        current_clips.push_back(
                            std::move(entry));
                    }
                }
            }
            if (auto set =
                    result.sequencer.dom()
                        .set_json(
                            "ply_sequences",
                            std::move(
                                current_clips));
                !set) {
                return std::move(set).error();
            }
        }

        if (const auto* trainer_manager =
                viewer.getTrainerManager()) {
            result.metrics =
                trainer_manager
                    ->captureProjectMetrics();
        }

        auto prepared =
            prepareGuiSessionRestore(result);
        if (!prepared)
            return std::move(prepared).error();
        return result;
    }

    namespace {

        editor::PythonEditorSessionState
        editor_state_from_json(
            const Json& file) {
            editor::PythonEditorSessionState state;
            assign_optional(
                file, "cursor_byte",
                state.cursor_byte);
            const auto anchor =
                file.find(
                    "selection_anchor_byte");
            if (anchor != file.end() &&
                !anchor->is_null() &&
                (anchor->is_number_integer() ||
                 anchor->is_number_unsigned())) {
                try {
                    state.selection_anchor_byte =
                        anchor->get<std::size_t>();
                } catch (
                    const nlohmann::json::exception&) {
                }
            }
            if (const auto scroll =
                    find_required_object(
                        file, "scroll");
                scroll != file.end()) {
                assign_optional(
                    *scroll, "x",
                    state.scroll_x);
                assign_optional(
                    *scroll, "y",
                    state.scroll_y);
            }
            if (const auto folds =
                    find_required_array(
                        file, "folds");
                folds != file.end()) {
                state.folds.reserve(folds->size());
                for (const auto& item : *folds) {
                    editor::
                        PythonEditorSessionFold fold;
                    if (!assign_optional(
                            item, "start_byte",
                            fold.start_byte) ||
                        !assign_optional(
                            item, "end_byte",
                            fold.end_byte) ||
                        !assign_optional(
                            item, "start_line",
                            fold.start_line) ||
                        !assign_optional(
                            item, "end_line",
                            fold.end_line)) {
                        continue;
                    }
                    assign_optional(
                        item, "kind", fold.kind);
                    assign_optional(
                        item, "collapsed",
                        fold.collapsed);
                    state.folds.push_back(
                        std::move(fold));
                }
            }
            return state;
        }

        std::string read_clean_editor_file(
            const std::filesystem::path& path) {
            constexpr std::uintmax_t
                max_editor_file_bytes =
                    64U * 1024U * 1024U;
            std::error_code error;
            const auto size =
                std::filesystem::file_size(
                    path, error);
            if (error ||
                size > max_editor_file_bytes)
                return {};

            std::ifstream stream;
            if (!lfs::core::open_file_for_read(
                    path, std::ios::binary,
                    stream)) {
                return {};
            }
            std::string text(
                static_cast<std::size_t>(size),
                '\0');
            if (!text.empty()) {
                stream.read(
                    text.data(),
                    static_cast<std::streamsize>(
                        text.size()));
                if (!stream)
                    return {};
            }
            return text;
        }

        std::optional<Json> panel_camera_json(
            const Json& root,
            const std::string_view panel) {
            const auto cameras =
                find_required_array(
                    root, "panel_cameras");
            if (cameras == root.end())
                return std::nullopt;
            const auto found =
                std::ranges::find_if(
                    *cameras,
                    [&](const Json& camera) {
                        const auto name =
                            scalar<std::string>(
                                camera, "panel");
                        return name &&
                               *name == panel;
                    });
            if (found == cameras->end())
                return std::nullopt;
            return std::optional<Json>{
                Json(*found)};
        }

        void apply_guil(
            VisualizerImpl& viewer,
            const Json& root) {
            auto* gui_manager =
                viewer.getGuiManager();
            auto* window_manager =
                viewer.getWindowManager();
            if (!gui_manager || !window_manager)
                return;

            const Json fixed =
                find_space_payload(
                    root, "fixed_arrangement");
            gui::PanelLayoutProjectState layout =
                gui_manager->panelLayout()
                    .captureProjectState();
            assign_optional(
                fixed, "right_panel_width",
                layout.right_panel_width);
            assign_optional(
                fixed, "scene_panel_ratio",
                layout.scene_panel_ratio);
            assign_optional(
                fixed, "python_console_width",
                layout.python_console_width);
            assign_optional(
                fixed, "bottom_dock_height",
                layout.bottom_dock_height);
            assign_optional(
                fixed, "left_dock_width",
                layout.left_dock_width);
            assign_optional(
                fixed, "sequencer_visible",
                layout.show_sequencer);

            if (const auto window_json =
                    find_required_object(
                        fixed, "window");
                window_json != fixed.end()) {
                auto state =
                    window_manager
                        ->captureProjectState();
                assign_optional(
                    *window_json, "x", state.x);
                assign_optional(
                    *window_json, "y", state.y);
                assign_optional(
                    *window_json, "width",
                    state.width);
                assign_optional(
                    *window_json, "height",
                    state.height);
                assign_optional(
                    *window_json, "fullscreen",
                    state.fullscreen);
                assign_optional(
                    *window_json, "maximized",
                    state.maximized);
                assign_optional(
                    *window_json, "restore_x",
                    state.restore_x);
                assign_optional(
                    *window_json, "restore_y",
                    state.restore_y);
                assign_optional(
                    *window_json,
                    "restore_width",
                    state.restore_width);
                assign_optional(
                    *window_json,
                    "restore_height",
                    state.restore_height);
                window_manager->applyProjectState(
                    state);
            }

            const Json registry =
                find_space_payload(
                    root, "panel_registry");
            if (const auto active_tabs =
                    find_required_object(
                        registry, "active_tabs");
                active_tabs != registry.end()) {
                assign_optional(
                    *active_tabs, "main_panel",
                    layout.active_tab_id);
                if (const auto scene_tab =
                        scalar<std::string>(
                            *active_tabs,
                            "scene_panel")) {
                    gui_manager
                        ->setScenePanelActiveTab(
                            *scene_tab);
                }
            }
            gui_manager->panelLayout()
                .applyProjectState(layout);

            std::vector<gui::PanelProjectState>
                panels;
            if (const auto panel_array =
                    find_required_array(
                        registry, "panels");
                panel_array != registry.end()) {
                panels.reserve(
                    panel_array->size());
                for (const auto& saved :
                     *panel_array) {
                    const auto id =
                        scalar<std::string>(
                            saved, "id");
                    const auto space_name =
                        scalar<std::string>(
                            saved, "space");
                    const auto space =
                        space_name
                            ? panel_space_from_name(
                                  *space_name)
                            : std::nullopt;
                    if (!id || !space)
                        continue;
                    gui::PanelProjectState state;
                    state.id = *id;
                    assign_optional(
                        saved, "parent_id",
                        state.parent_id);
                    state.space = *space;
                    assign_optional(
                        saved, "order",
                        state.order);
                    assign_optional(
                        saved, "enabled",
                        state.enabled);
                    state.float_x =
                        scalar<float>(
                            saved, "float_x")
                            .value_or(NAN);
                    state.float_y =
                        scalar<float>(
                            saved, "float_y")
                            .value_or(NAN);
                    assign_optional(
                        saved,
                        "float_user_height",
                        state.float_user_height);
                    assign_optional(
                        saved,
                        "float_last_bounds_valid",
                        state
                            .float_last_bounds_valid);
                    assign_optional(
                        saved, "float_last_x",
                        state.float_last_x);
                    assign_optional(
                        saved, "float_last_y",
                        state.float_last_y);
                    assign_optional(
                        saved, "float_last_w",
                        state.float_last_w);
                    assign_optional(
                        saved, "float_last_h",
                        state.float_last_h);
                    assign_optional(
                        saved,
                        "float_auto_center",
                        state.float_auto_center);
                    assign_optional(
                        saved,
                        "float_stack_order",
                        state.float_stack_order);
                    panels.push_back(
                        std::move(state));
                }
            }
            gui::PanelRegistry::instance()
                .apply_project_state(panels);

            if (auto* window_states =
                    gui_manager
                        ->getWindowStates()) {
                bool console_visible =
                    window_states->contains(
                        "python_console") &&
                    window_states->at(
                        "python_console");
                assign_optional(
                    fixed,
                    "python_console_visible",
                    console_visible);
                (*window_states)
                    ["python_console"] =
                        console_visible;
            }

            const Json console =
                find_space_payload(
                    root, "python_console");
            if (auto* console_state =
                    gui::panels::
                        PythonConsoleState::
                            tryGetInstance()) {
                int active_tab =
                    console_state
                        ->getActiveTab();
                float font_scale =
                    console_state
                        ->getFontScale();
                assign_optional(
                    console, "active_tab",
                    active_tab);
                assign_optional(
                    console, "font_scale",
                    font_scale);
                console_state->setActiveTab(
                    std::clamp(active_tab, 0, 1));
                console_state->setFontScale(
                    font_scale);
            }
        }

        void apply_editor(
            const Json& root) {
            auto* console =
                gui::panels::PythonConsoleState::
                    tryGetInstance();
            if (!console)
                return;
            auto* python_editor =
                console->getEditor();
            if (!python_editor)
                return;

            const auto files =
                find_required_array(
                    root, "open_files");
            if (files == root.end())
                return;
            editor::
                PythonEditorWorkspaceSessionState
                    workspace;
            workspace.vim_mode =
                scalar<bool>(
                    root, "vim_mode")
                    .value_or(false);
            workspace.active_file =
                scalar<std::string>(
                    root, "active_file");
            workspace.open_files.reserve(
                files->size());
            for (const auto& file : *files) {
                const auto locator =
                    scalar<std::string>(
                        file, "locator");
                if (!locator)
                    continue;
                const bool modified =
                    scalar<bool>(
                        file, "modified")
                        .value_or(false);
                std::string text;
                if (modified) {
                    text = scalar<std::string>(
                               file,
                               "embedded_buffer")
                               .value_or(
                                   std::string{});
                } else if (
                    !locator->contains("://")) {
                    text =
                        read_clean_editor_file(
                            lfs::core::
                                utf8_to_path(
                                    *locator));
                }
                workspace.open_files.push_back(
                    editor::
                        PythonEditorSessionFile{
                            .locator =
                                *locator,
                            .text =
                                std::move(text),
                            .modified =
                                modified,
                            .editor =
                                editor_state_from_json(
                                    file),
                        });
            }
            if (workspace.active_file &&
                std::ranges::none_of(
                    workspace.open_files,
                    [&](const auto& file) {
                        return file.locator ==
                               *workspace
                                    .active_file;
                    })) {
                workspace.active_file.reset();
            }
            if (!workspace.active_file &&
                !workspace.open_files.empty()) {
                workspace.active_file =
                    workspace.open_files.front()
                        .locator;
            }

            python_editor
                ->restoreWorkspaceSessionState(
                    workspace);
            const auto active =
                workspace.active_file
                    ? std::ranges::find_if(
                          workspace.open_files,
                          [&](const auto& file) {
                              return file.locator ==
                                     *workspace
                                          .active_file;
                          })
                    : workspace.open_files.end();
            if (active ==
                workspace.open_files.end()) {
                console->setScriptPath({});
                console->setModified(false);
                return;
            }
            console->setScriptPath(
                active->locator.contains("://")
                    ? std::filesystem::path{}
                    : lfs::core::utf8_to_path(
                          active->locator));
            console->setModified(
                active->modified);
        }

        void apply_view(
            VisualizerImpl& viewer,
            const Json& root,
            std::vector<
                CameraBookmarkProjectState>&
                bookmarks) {
            auto* rendering =
                viewer.getRenderingManager();
            auto* input =
                viewer.getInputController();
            auto* gui_manager =
                viewer.getGuiManager();
            if (!rendering || !input ||
                !gui_manager)
                return;

            const auto settings_json =
                find_required_object(
                    root, "render_settings");
            if (settings_json == root.end())
                return;
            auto restored =
                renderSettingsFromProjectJson(
                    *settings_json,
                    rendering->getSettings());
            if (!restored)
                return;
            const auto desired_split =
                restored->split_view_mode;
            restored->split_view_mode =
                rendering->getSettings()
                    .split_view_mode;
            restored->gut =
                lfs::rendering::isGutBackend(
                    restored->raster_backend);
            rendering->updateSettings(*restored);

            // The service transition creates/copies secondary panel state;
            // saved cameras therefore apply only after this call.
            rendering->restoreSplitViewMode(
                desired_split,
                viewer.getViewport());

            if (const auto split =
                    find_required_object(
                        root, "split");
                split != root.end()) {
                if (const auto planes =
                        find_required_array(
                            *split,
                            "panel_grid_planes");
                    planes != split->end() &&
                    planes->size() == 2) {
                    if (planes->at(0)
                            .is_number_integer()) {
                        rendering
                            ->setGridPlaneForPanel(
                                SplitViewPanelId::
                                    Left,
                                planes->at(0)
                                    .get<int>());
                    }
                    if (planes->at(1)
                            .is_number_integer()) {
                        rendering
                            ->setGridPlaneForPanel(
                                SplitViewPanelId::
                                    Right,
                                planes->at(1)
                                    .get<int>());
                    }
                }
                const auto focused =
                    scalar<std::string>(
                        *split,
                        "focused_panel");
                rendering->setFocusedSplitPanel(
                    focused &&
                            *focused == "right"
                        ? SplitViewPanelId::Right
                        : SplitViewPanelId::Left);
                const auto camera_id =
                    scalar<int>(
                        *split,
                        "gt_camera_id");
                rendering->setCurrentCameraId(
                    camera_id.value_or(-1));
            }

            if (const auto primary_json =
                    panel_camera_json(
                        root, "primary")) {
                if (auto camera =
                        panelCameraProjectStateFromJson(
                            *primary_json);
                    camera) {
                    applyPanelCameraProjectState(
                        viewer.getViewport(),
                        *camera);
                }
            }
            if (const auto secondary_json =
                    panel_camera_json(
                        root, "secondary")) {
                if (auto camera =
                        panelCameraProjectStateFromJson(
                            *secondary_json);
                    camera) {
                    applyPanelCameraProjectState(
                        rendering
                            ->projectSecondaryViewport(),
                        *camera);
                }
            }

            if (const auto navigation =
                    find_required_object(
                        root, "navigation");
                navigation != root.end()) {
                const auto mode_name =
                    scalar<std::string>(
                        *navigation, "mode");
                const auto mode =
                    mode_name
                        ? InputController::
                              cameraNavigationModeFromName(
                                  *mode_name)
                        : std::nullopt;
                input->restoreProjectNavigation(
                    mode.value_or(
                        InputController::
                            CameraNavigationMode::
                                Orbit),
                    scalar<bool>(
                        *navigation,
                        "view_snap")
                        .value_or(false));
            }

            bookmarks.clear();
            if (const auto saved_bookmarks =
                    find_required_array(
                        root,
                        "camera_bookmarks");
                saved_bookmarks != root.end()) {
                bookmarks.reserve(
                    saved_bookmarks->size());
                for (const auto& saved :
                     *saved_bookmarks) {
                    const auto id =
                        scalar<std::string>(
                            saved, "id");
                    const auto name =
                        scalar<std::string>(
                            saved, "name");
                    auto camera =
                        panelCameraProjectStateFromJson(
                            saved);
                    if (id && name && camera) {
                        bookmarks.push_back({
                            .id = *id,
                            .name = *name,
                            .camera =
                                std::move(*camera),
                        });
                    }
                }
            }

            if (const auto tools =
                    find_required_object(
                        root, "tools");
                tools != root.end()) {
                auto& registry =
                    UnifiedToolRegistry::instance();
                const auto active_tool =
                    scalar<std::string>(
                        *tools,
                        "active_tool_id")
                        .value_or(
                            std::string{});
                if (active_tool.empty()) {
                    registry.clearActiveTool();
                } else if (
                    registry.poll(active_tool)) {
                    registry.setActiveTool(
                        active_tool);
                } else {
                    // The ID remains in retained VIEW, but is not made active
                    // when its plugin is unavailable.
                    registry.clearActiveTool();
                }
                if (const auto submode =
                        scalar<std::string>(
                            *tools,
                            "active_submode_id")) {
                    registry.setActiveSubmode(
                        *submode);
                }

                auto& gizmo =
                    gui_manager->gizmo();
                if (const auto submode =
                        scalar<std::string>(
                            *tools,
                            "selection_submode");
                    submode) {
                    if (const auto parsed =
                            selection_submode_from_name(
                                *submode)) {
                        gizmo.setSelectionSubMode(
                            *parsed);
                    }
                }
                const auto operation =
                    scalar<std::string>(
                        *tools,
                        "gizmo_operation");
                gizmo.setOperation(
                    operation &&
                            *operation == "rotate"
                        ? gui::GizmoOperation::
                              Rotate
                    : operation &&
                            *operation == "scale"
                        ? gui::GizmoOperation::
                              Scale
                        : gui::GizmoOperation::
                              Translate);
                const auto transform =
                    scalar<std::string>(
                        *tools,
                        "transform_space");
                gizmo.setTransformSpace(
                    transform &&
                            *transform == "world"
                        ? TransformSpace::World
                        : TransformSpace::Local);
                const auto pivot =
                    scalar<std::string>(
                        *tools, "pivot_mode");
                gizmo.setPivotMode(
                    pivot &&
                            *pivot ==
                                "bounds_center"
                        ? PivotMode::BoundsCenter
                        : PivotMode::Origin);
                const auto multi =
                    scalar<std::string>(
                        *tools,
                        "multi_transform_mode");
                gizmo.setMultiTransformMode(
                    multi &&
                            *multi == "individual"
                        ? gui::MultiTransformMode::
                              Individual
                        : gui::MultiTransformMode::
                              Selection);
                if (const auto shape =
                        scalar<std::string>(
                            *tools,
                            "crop_shape")) {
                    gizmo.setCropToolShape(
                        *shape);
                }
                if (const auto operation_name =
                        scalar<std::string>(
                            *tools,
                            "crop_operation")) {
                    gizmo.setCropToolOperation(
                        *operation_name);
                }
                if (auto* selection_tool =
                        viewer.getSelectionTool()) {
                    if (const auto selection =
                            find_required_object(
                                *tools,
                                "selection");
                        selection !=
                        tools->end()) {
                        selection_tool
                            ->restoreProjectPreferences(
                                scalar<float>(
                                    *selection,
                                    "brush_radius")
                                    .value_or(
                                        20.0f),
                                scalar<bool>(
                                    *selection,
                                    "crop_filter")
                                    .value_or(
                                        false),
                                scalar<bool>(
                                    *selection,
                                    "depth_filter")
                                    .value_or(
                                        false),
                                scalar<bool>(
                                    *selection,
                                    "restrict_to_selected_nodes")
                                    .value_or(
                                        true));
                    }
                }
            }
            if (const auto sequencer_view =
                    find_required_object(
                        root, "sequencer_view");
                sequencer_view != root.end()) {
                assign_optional(
                    *sequencer_view,
                    "show_camera_path",
                    gui_manager
                        ->getSequencerUIState()
                        .show_camera_path);
            }
            rendering->markDirty(DirtyFlag::ALL);
        }

        void apply_sequencer(
            VisualizerImpl& viewer,
            const Json& root) {
            auto* gui_manager =
                viewer.getGuiManager();
            if (!gui_manager)
                return;
            auto& controller =
                gui_manager->sequencer();
            controller.stop();
            controller.clearPlySequence();

            if (const auto timeline =
                    find_required_object(
                        root, "timeline");
                timeline != root.end()) {
                const auto standard_json =
                    nlohmann::json::parse(
                        timeline->dump());
                (void)controller.loadFromJson(
                    standard_json);
            }

            if (const auto clips =
                    find_required_array(
                        root, "ply_sequences");
                clips != root.end() &&
                !clips->empty()) {
                const auto& clip =
                    clips->front();
                const auto frames =
                    find_required_array(
                        clip, "frames");
                if (frames != clip.end()) {
                    std::vector<
                        std::filesystem::path>
                        paths;
                    std::vector<std::string>
                        names;
                    std::vector<lfs::core::Uuid>
                        uuids;
                    paths.reserve(frames->size());
                    names.reserve(frames->size());
                    uuids.reserve(frames->size());
                    const auto directory_hint =
                        scalar<std::string>(
                            clip,
                            "directory_hint")
                            .value_or(
                                std::string{});
                    const auto directory =
                        lfs::core::utf8_to_path(
                            directory_hint);
                    for (const auto& frame :
                         *frames) {
                        const auto locator =
                            scalar<std::string>(
                                frame,
                                "locator")
                                .value_or(
                                    std::string{});
                        paths.push_back(
                            directory /
                            lfs::core::utf8_to_path(
                                locator));
                        names.push_back(
                            scalar<std::string>(
                                frame,
                                "node_name")
                                .value_or(
                                    std::string{}));
                        const auto uuid_text =
                            scalar<std::string>(
                                frame,
                                "node_uuid")
                                .value_or(
                                    std::string{});
                        uuids.push_back(
                            lfs::core::Uuid::
                                from_string(
                                    uuid_text)
                                    .value_or(
                                        lfs::core::
                                            Uuid{}));
                    }
                    const auto clip_uuid_text =
                        scalar<std::string>(
                            clip, "node_uuid")
                            .value_or(
                                std::string{});
                    controller.setPlySequence(
                        directory,
                        scalar<std::string>(
                            clip, "node_name")
                            .value_or(
                                std::string{}),
                        std::move(paths),
                        std::move(names),
                        scalar<float>(
                            clip, "fps")
                            .value_or(
                                DEFAULT_SEQUENCE_FPS),
                        lfs::core::Uuid::from_string(
                            clip_uuid_text)
                            .value_or(
                                lfs::core::Uuid{}),
                        std::move(uuids));
                }
            }

            const auto loop =
                scalar<std::string>(
                    root, "loop_mode")
                    .value_or("once");
            controller.setLoopMode(
                loop == "loop"
                    ? LoopMode::LOOP
                : loop == "ping_pong"
                    ? LoopMode::PING_PONG
                    : LoopMode::ONCE);
            controller.setPlaybackSpeed(
                scalar<float>(
                    root, "playback_speed")
                    .value_or(1.0f));
            controller.seek(
                scalar<float>(
                    root, "playhead")
                    .value_or(0.0f));
            controller.stop();
            controller.seek(
                scalar<float>(
                    root, "playhead")
                    .value_or(0.0f));

            auto& ui =
                gui_manager
                    ->getSequencerUIState();
            if (const auto preferences =
                    find_required_object(
                        root, "preferences");
                preferences != root.end()) {
                assign_optional(
                    *preferences,
                    "snap_to_grid",
                    ui.snap_to_grid);
                assign_optional(
                    *preferences,
                    "snap_interval",
                    ui.snap_interval);
                assign_optional(
                    *preferences,
                    "follow_playback",
                    ui.follow_playback);
                assign_optional(
                    *preferences,
                    "show_pip_preview",
                    ui.show_pip_preview);
                assign_optional(
                    *preferences,
                    "pip_preview_scale",
                    ui.pip_preview_scale);
                assign_optional(
                    *preferences,
                    "show_film_strip",
                    ui.show_film_strip);
            }
            // Controller values are canonical over the UI mirrors.
            ui.playback_speed =
                controller.playbackSpeed();
            ui.sequence_fps =
                controller.plySequenceFps();
            gui_manager->sequencerUI()
                .syncKeyframesToSceneGraph();
        }

    } // namespace

    void applyGuiSession(
        VisualizerImpl& viewer,
        const PreparedGuiSessionRestore& prepared,
        std::vector<CameraBookmarkProjectState>&
            bookmarks) {
        auto gui = chapter_root(
            prepared.chapters.gui_layout.dom(),
            "GUIL");
        auto editor = chapter_root(
            prepared.chapters.editor.dom(),
            "EDTR");
        auto view = chapter_root(
            prepared.chapters.view.dom(),
            "VIEW");
        auto sequencer = chapter_root(
            prepared.chapters.sequencer.dom(),
            "SEQR");
        // prepareGuiSessionRestore already proved all four roots. These guards
        // keep the event callback noexcept if memory corruption intervenes.
        if (!gui || !editor || !view || !sequencer)
            return;

        // VIEW runs after SCNG/CKPT/PPIS hydration. GUIL then applies only at
        // the panels-ready boundary that delivered this prepared bundle.
        apply_view(viewer, *view, bookmarks);
        apply_guil(viewer, *gui);
        apply_editor(*editor);
        apply_sequencer(viewer, *sequencer);
        if (auto* trainer =
                viewer.getTrainerManager()) {
            trainer->restoreProjectMetrics(
                prepared.chapters.metrics);
        }
        if (auto* rendering =
                viewer.getRenderingManager()) {
            rendering->markDirty(DirtyFlag::ALL);
        }
    }

} // namespace lfs::vis::project
