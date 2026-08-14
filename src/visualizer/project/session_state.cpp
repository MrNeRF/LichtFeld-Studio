/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "project/session_state.hpp"

#include "core/error.hpp"
#include "core/error_bus.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/error_event_bridge.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_layout.hpp"
#include "gui/panel_registry.hpp"
#include "gui/panels/python_console_panel.hpp"
#include "gui/panels/windows_console_utils.hpp"
#include "gui/scene_tree_session.hpp"
#include "gui/ui_context.hpp"
#include "input/input_controller.hpp"
#include "io/video/video_export_options.hpp"
#include "rendering/render_constants.hpp"
#include "rendering/rendering_manager.hpp"
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
#include <format>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
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

        template <typename Owner>
        struct JsonField {
            std::string_view name;
            std::function<Json(const Owner&)> write;
            std::function<lfs::Result<void>(
                const Json&,
                Owner&,
                std::string_view,
                std::string_view)>
                read;
        };

        template <typename Owner, typename Member>
        JsonField<Owner> required_field(
            const std::string_view name,
            Member Owner::*member) {
            return {
                .name = name,
                .write = [member](const Owner& source) { return Json(source.*member); },
                .read = [member](
                            const Json& json,
                            Owner& destination,
                            const std::string_view prefix,
                            const std::string_view field) { return assign_required(
                                                                json,
                                                                field,
                                                                destination.*member,
                                                                prefix); },
            };
        }

        template <typename Owner, typename Member>
        JsonField<Owner> optional_field(
            const std::string_view name,
            Member Owner::*member) {
            return {
                .name = name,
                .write = [member](const Owner& source) { return Json(source.*member); },
                .read = [member](
                            const Json& json,
                            Owner& destination,
                            std::string_view,
                            const std::string_view field) {
                    (void)assign_optional(
                        json, field, destination.*member);
                    return lfs::Result<void>{}; },
            };
        }

        template <typename Owner>
        JsonField<Owner> vec3_field(
            const std::string_view name,
            glm::vec3 Owner::*member) {
            return {
                .name = name,
                .write = [member](const Owner& source) { return vec3_json(source.*member); },
                .read = [member](
                            const Json& json,
                            Owner& destination,
                            const std::string_view prefix,
                            const std::string_view field) {
                    auto value = required_vec3(
                        json, field, prefix);
                    if (!value) {
                        return lfs::Status::failure(
                            std::move(value).error());
                    }
                    destination.*member = *value;
                    return lfs::Result<void>{}; },
            };
        }

        template <typename Owner, typename Enum,
                  typename AfterAssign = std::nullptr_t>
        JsonField<Owner> enum_field(
            const std::string_view name,
            Enum Owner::*member,
            const int minimum,
            const int maximum,
            const std::string_view invalid_detail,
            AfterAssign after_assign = nullptr) {
            return {
                .name = name,
                .write = [member](const Owner& source) { return Json(static_cast<int>(source.*member)); },
                .read = [=](
                            const Json& json,
                            Owner& destination,
                            const std::string_view prefix,
                            const std::string_view field) {
                    int value = 0;
                    if (auto status = assign_required(
                            json, field, value, prefix);
                        !status) {
                        return status;
                    }
                    if (value < minimum || value > maximum) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            std::string(invalid_detail),
                            std::string(prefix) + "." +
                                std::string(field));
                    }
                    destination.*member = static_cast<Enum>(value);
                    if constexpr (!std::same_as<
                                      AfterAssign,
                                      std::nullptr_t>)
                        after_assign(destination);
                    return lfs::Result<void>{}; },
            };
        }

        template <typename Owner, typename Writer, typename Reader>
        JsonField<Owner> custom_field(
            const std::string_view name,
            Writer write,
            Reader read) {
            return {
                .name = name,
                .write = std::move(write),
                .read = std::move(read),
            };
        }

        template <typename Owner>
        Json fields_to_json(
            const Owner& source,
            const std::vector<JsonField<Owner>>& fields) {
            Json result = Json::object();
            for (const auto& field : fields) {
                result[std::string(field.name)] =
                    field.write(source);
            }
            return result;
        }

        template <typename Owner>
        void append_fields(
            Json& result,
            const Owner& source,
            const std::vector<JsonField<Owner>>& fields) {
            for (const auto& field : fields) {
                result[std::string(field.name)] =
                    field.write(source);
            }
        }

        template <typename Owner>
        lfs::Result<void> read_fields(
            const Json& json,
            Owner& destination,
            const std::string_view prefix,
            const std::vector<JsonField<Owner>>& fields) {
            for (const auto& field : fields) {
                if (auto status = field.read(
                        json,
                        destination,
                        prefix,
                        field.name);
                    !status) {
                    return status;
                }
            }
            return {};
        }

        template <typename Owner, std::size_t Size>
        JsonField<Owner> array_field(
            const std::string_view name,
            std::array<float, Size> Owner::*member) {
            return custom_field<Owner>(
                name,
                [member](const Owner& source) {
                    return json_array(source.*member);
                },
                [member](const Json& json,
                         Owner& destination,
                         const std::string_view prefix,
                         const std::string_view field) {
                    const auto found =
                        json.find(std::string(field));
                    if (found == json.end()) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "Panel camera field is missing",
                            std::string(prefix) + "." +
                                std::string(field));
                    }
                    const auto values =
                        number_array<Size>(*found);
                    if (!values) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "Panel camera field must contain finite numbers",
                            std::string(prefix) + "." +
                                std::string(field));
                    }
                    destination.*member = *values;
                    return lfs::Result<void>{};
                });
        }

        template <typename Owner>
        JsonField<Owner> nullable_positive_float_field(
            const std::string_view name,
            std::optional<float> Owner::*member) {
            return custom_field<Owner>(
                name,
                [member](const Owner& source) {
                    const auto& value = source.*member;
                    return value ? Json(*value) : Json(nullptr);
                },
                [member](const Json& json,
                         Owner& destination,
                         const std::string_view prefix,
                         const std::string_view field) {
                    const auto found =
                        json.find(std::string(field));
                    if (found == json.end()) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "Panel camera ortho_scale is missing",
                            std::string(prefix) + "." +
                                std::string(field));
                    }
                    if (found->is_null()) {
                        (destination.*member).reset();
                        return lfs::Result<void>{};
                    }
                    if (!found->is_number()) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "Panel camera ortho scale must be a number or null",
                            std::string(prefix) + "." +
                                std::string(field));
                    }
                    const auto value =
                        scalar<float>(json, field);
                    if (!value || *value <= 0.0f) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "Panel camera ortho scale must be positive and finite",
                            std::string(prefix) + "." +
                                std::string(field));
                    }
                    destination.*member = *value;
                    return lfs::Result<void>{};
                });
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

        const auto& ppisp_override_fields() {
            using Overrides = std::remove_cvref_t<
                decltype(std::declval<RenderSettings>()
                             .ppisp_overrides)>;
            static const std::vector<JsonField<Overrides>> fields{
                required_field("exposure_offset", &Overrides::exposure_offset),
                required_field("vignette_enabled", &Overrides::vignette_enabled),
                required_field("vignette_strength", &Overrides::vignette_strength),
                required_field("wb_temperature", &Overrides::wb_temperature),
                required_field("wb_tint", &Overrides::wb_tint),
                required_field("color_red_x", &Overrides::color_red_x),
                required_field("color_red_y", &Overrides::color_red_y),
                required_field("color_green_x", &Overrides::color_green_x),
                required_field("color_green_y", &Overrides::color_green_y),
                required_field("color_blue_x", &Overrides::color_blue_x),
                required_field("color_blue_y", &Overrides::color_blue_y),
                required_field("gamma_multiplier", &Overrides::gamma_multiplier),
                required_field("gamma_red", &Overrides::gamma_red),
                required_field("gamma_green", &Overrides::gamma_green),
                required_field("gamma_blue", &Overrides::gamma_blue),
                required_field("crf_toe", &Overrides::crf_toe),
                required_field("crf_shoulder", &Overrides::crf_shoulder),
            };
            return fields;
        }

        const auto& render_settings_fields() {
            static const std::vector<JsonField<RenderSettings>> fields{
                required_field("focal_length_mm", &RenderSettings::focal_length_mm),
                required_field("scaling_modifier", &RenderSettings::scaling_modifier),
                required_field("antialiasing", &RenderSettings::antialiasing),
                required_field("mip_filter", &RenderSettings::mip_filter),
                required_field("sh_degree", &RenderSettings::sh_degree),
                required_field("render_scale", &RenderSettings::render_scale),
                enum_field("camera_metrics_mode", &RenderSettings::camera_metrics_mode,
                           0, 2, "Unsupported camera metrics mode"),
                required_field("show_crop_box", &RenderSettings::show_crop_box),
                required_field("use_crop_box", &RenderSettings::use_crop_box),
                required_field("show_ellipsoid", &RenderSettings::show_ellipsoid),
                required_field("use_ellipsoid", &RenderSettings::use_ellipsoid),
                required_field("desaturate_unselected", &RenderSettings::desaturate_unselected),
                required_field("desaturate_cropping", &RenderSettings::desaturate_cropping),
                required_field("hide_outside_depth_box", &RenderSettings::hide_outside_depth_box),
                required_field("crop_filter_for_selection", &RenderSettings::crop_filter_for_selection),
                required_field("apply_appearance_correction", &RenderSettings::apply_appearance_correction),
                enum_field("ppisp_mode", &RenderSettings::ppisp_mode,
                           0, 1, "Unsupported PPISP mode"),
                custom_field<RenderSettings>(
                    "ppisp_overrides",
                    [](const RenderSettings& settings) {
                        return fields_to_json(settings.ppisp_overrides,
                                              ppisp_override_fields());
                    },
                    [](const Json& json, RenderSettings& settings,
                       std::string_view prefix, std::string_view field) {
                        const auto found = find_required_object(json, field);
                        if (found == json.end()) {
                            return fail<void>(
                                lfs::ErrorCode::DataLoss,
                                "VIEW PPISP overrides are missing",
                                std::string(prefix) + "." + std::string(field));
                        }
                        return read_fields(
                            *found, settings.ppisp_overrides,
                            std::string(prefix) + "." + std::string(field),
                            ppisp_override_fields());
                    }),
                vec3_field("background_color", &RenderSettings::background_color),
                enum_field("environment_mode", &RenderSettings::environment_mode,
                           0, 1, "Unsupported environment mode"),
                custom_field<RenderSettings>(
                    "environment_reference_uuid",
                    [](const RenderSettings&) { return Json(nullptr); },
                    [](const Json&, RenderSettings&, std::string_view,
                       std::string_view) { return lfs::Result<void>{}; }),
                custom_field<RenderSettings>(
                    "environment_builtin",
                    [](const RenderSettings& settings) {
                        return settings.environment_map_path ==
                                       lfs::vis::kDefaultEnvironmentMapPath
                                   ? Json(settings.environment_map_path)
                                   : Json(nullptr);
                    },
                    [](const Json& json, RenderSettings& settings,
                       std::string_view, std::string_view field) {
                        if (const auto value = scalar<std::string>(json, field))
                            settings.environment_map_path = *value;
                        return lfs::Result<void>{};
                    }),
                required_field("environment_exposure", &RenderSettings::environment_exposure),
                required_field("environment_rotation_degrees", &RenderSettings::environment_rotation_degrees),
                required_field("show_coord_axes", &RenderSettings::show_coord_axes),
                required_field("axes_size", &RenderSettings::axes_size),
                custom_field<RenderSettings>(
                    "axes_visibility",
                    [](const RenderSettings& settings) {
                        return Json::array({settings.axes_visibility[0],
                                            settings.axes_visibility[1],
                                            settings.axes_visibility[2]});
                    },
                    [](const Json& json, RenderSettings& settings,
                       std::string_view prefix, std::string_view field) {
                        const auto found = find_required_array(json, field);
                        if (found == json.end() || found->size() != 3 ||
                            !std::ranges::all_of(*found, [](const Json& item) {
                                return item.is_boolean();
                            })) {
                            return fail<void>(
                                lfs::ErrorCode::DataLoss,
                                "Axes visibility must contain three booleans",
                                std::string(prefix) + "." + std::string(field));
                        }
                        for (std::size_t index = 0; index < 3; ++index)
                            settings.axes_visibility[index] = (*found)[index].get<bool>();
                        return lfs::Result<void>{};
                    }),
                required_field("show_grid", &RenderSettings::show_grid),
                required_field("grid_plane", &RenderSettings::grid_plane),
                required_field("grid_opacity", &RenderSettings::grid_opacity),
                required_field("point_cloud_mode", &RenderSettings::point_cloud_mode),
                required_field("voxel_size", &RenderSettings::voxel_size),
                required_field("show_rings", &RenderSettings::show_rings),
                required_field("ring_width", &RenderSettings::ring_width),
                required_field("show_center_markers", &RenderSettings::show_center_markers),
                required_field("show_camera_frustums", &RenderSettings::show_camera_frustums),
                required_field("camera_frustum_scale", &RenderSettings::camera_frustum_scale),
                vec3_field("train_camera_color", &RenderSettings::train_camera_color),
                vec3_field("eval_camera_color", &RenderSettings::eval_camera_color),
                required_field("show_pivot", &RenderSettings::show_pivot),
                enum_field("split_view_mode", &RenderSettings::split_view_mode,
                           0, 3, "Unsupported split-view mode"),
                enum_field("gt_comparison_mode", &RenderSettings::gt_comparison_mode,
                           std::numeric_limits<int>::min(),
                           std::numeric_limits<int>::max(),
                           "Unsupported GT comparison mode",
                           [](RenderSettings& settings) {
                               sanitizeGTComparisonSettings(settings);
                           }),
                required_field("split_position", &RenderSettings::split_position),
                required_field("split_view_offset", &RenderSettings::split_view_offset),
                custom_field<RenderSettings>(
                    "raster_backend",
                    [](const RenderSettings& settings) {
                        return Json(std::string(
                            lfs::rendering::gaussianRasterBackendId(
                                settings.raster_backend)));
                    },
                    [](const Json& json, RenderSettings& settings,
                       std::string_view prefix, std::string_view field) {
                        const auto value = scalar<std::string>(json, field);
                        if (!value ||
                            !lfs::rendering::isGaussianRasterBackendId(*value)) {
                            return fail<void>(
                                lfs::ErrorCode::DataLoss,
                                "Unsupported raster backend",
                                std::string(prefix) + "." + std::string(field));
                        }
                        settings.raster_backend =
                            lfs::rendering::gaussianRasterBackendFromId(*value);
                        settings.gut = lfs::rendering::isGutBackend(
                            settings.raster_backend);
                        return lfs::Result<void>{};
                    }),
                required_field("equirectangular", &RenderSettings::equirectangular),
                required_field("orthographic", &RenderSettings::orthographic),
                required_field("ortho_scale", &RenderSettings::ortho_scale),
                required_field("depth_view", &RenderSettings::depth_view),
                required_field("depth_view_min", &RenderSettings::depth_view_min),
                required_field("depth_view_max", &RenderSettings::depth_view_max),
                enum_field("depth_visualization_mode", &RenderSettings::depth_visualization_mode,
                           std::numeric_limits<int>::min(),
                           std::numeric_limits<int>::max(),
                           "Unsupported depth visualization mode",
                           [](RenderSettings& settings) {
                               sanitizeDepthViewSettings(settings);
                           }),
                vec3_field("selection_color_committed", &RenderSettings::selection_color_committed),
                vec3_field("selection_color_preview", &RenderSettings::selection_color_preview),
                vec3_field("selection_color_center_marker", &RenderSettings::selection_color_center_marker),
                required_field("depth_clip_enabled", &RenderSettings::depth_clip_enabled),
                required_field("depth_clip_far", &RenderSettings::depth_clip_far),
                required_field("mesh_wireframe", &RenderSettings::mesh_wireframe),
                vec3_field("mesh_wireframe_color", &RenderSettings::mesh_wireframe_color),
                required_field("mesh_wireframe_width", &RenderSettings::mesh_wireframe_width),
                vec3_field("mesh_light_dir", &RenderSettings::mesh_light_dir),
                required_field("mesh_light_intensity", &RenderSettings::mesh_light_intensity),
                required_field("mesh_ambient", &RenderSettings::mesh_ambient),
                required_field("mesh_backface_culling", &RenderSettings::mesh_backface_culling),
                required_field("mesh_shadow_enabled", &RenderSettings::mesh_shadow_enabled),
                required_field("mesh_shadow_resolution", &RenderSettings::mesh_shadow_resolution),
                required_field("depth_filter_enabled", &RenderSettings::depth_filter_enabled),
                vec3_field("depth_filter_min", &RenderSettings::depth_filter_min),
                vec3_field("depth_filter_max", &RenderSettings::depth_filter_max),
                custom_field<RenderSettings>(
                    "depth_filter_transform",
                    [](const RenderSettings& settings) {
                        return Json{{"rotation", json_array(matrix_array(
                                                     glm::mat3_cast(settings.depth_filter_transform.getRotation())))},
                                    {"translation", vec3_json(
                                                        settings.depth_filter_transform.getTranslation())}};
                    },
                    [](const Json& json, RenderSettings& settings,
                       std::string_view prefix, std::string_view field) {
                        const auto transform = find_required_object(json, field);
                        if (transform == json.end()) {
                            return fail<void>(
                                lfs::ErrorCode::DataLoss,
                                "Depth-filter transform is missing",
                                std::string(prefix) + "." + std::string(field));
                        }
                        const auto rotation_it = transform->find("rotation");
                        const auto rotation = rotation_it == transform->end()
                                                  ? std::optional<std::array<float, 9>>{}
                                                  : number_array<9>(*rotation_it);
                        if (!rotation) {
                            return fail<void>(
                                lfs::ErrorCode::DataLoss,
                                "Depth-filter rotation must be a finite 3x3 matrix",
                                std::string(prefix) + "." + std::string(field) +
                                    ".rotation");
                        }
                        auto translation = required_vec3(
                            *transform, "translation",
                            std::string(prefix) + "." + std::string(field));
                        if (!translation)
                            return lfs::Status::failure(std::move(translation).error());
                        settings.depth_filter_transform =
                            lfs::geometry::EuclideanTransform(
                                glm::quat_cast(array_matrix(*rotation)), *translation);
                        return lfs::Result<void>{};
                    }),
                required_field("lod_enabled", &RenderSettings::lod_enabled),
                required_field("lod_auto_enable_rad", &RenderSettings::lod_auto_enable_rad),
                required_field("lod_max_splats", &RenderSettings::lod_max_splats),
                required_field("lod_render_scale", &RenderSettings::lod_render_scale),
                required_field("lod_behind_camera_penalty", &RenderSettings::lod_behind_camera_penalty),
                required_field("lod_cone_foveation", &RenderSettings::lod_cone_foveation),
                required_field("lod_cone_inner_degrees", &RenderSettings::lod_cone_inner_degrees),
                required_field("lod_cone_outer_degrees", &RenderSettings::lod_cone_outer_degrees),
                required_field("lod_page_pool_splats", &RenderSettings::lod_page_pool_splats),
                required_field("lod_pool_vram_fraction", &RenderSettings::lod_pool_vram_fraction),
                required_field("lod_fade_frames", &RenderSettings::lod_fade_frames),
                required_field("lod_debug_colors", &RenderSettings::lod_debug_colors),
            };
            return fields;
        }

        const auto& panel_camera_fields() {
            static const std::vector<
                JsonField<PanelCameraProjectState>>
                fields{
                    array_field("R", &PanelCameraProjectState::rotation),
                    array_field("t", &PanelCameraProjectState::translation),
                    array_field("pivot", &PanelCameraProjectState::pivot),
                    array_field("home_R", &PanelCameraProjectState::home_rotation),
                    array_field("home_t", &PanelCameraProjectState::home_translation),
                    array_field("home_pivot", &PanelCameraProjectState::home_pivot),
                    required_field("home_saved", &PanelCameraProjectState::home_saved),
                    required_field("zoom_speed", &PanelCameraProjectState::zoom_speed),
                    required_field("max_zoom_speed", &PanelCameraProjectState::max_zoom_speed),
                    required_field("rotate_speed", &PanelCameraProjectState::rotate_speed),
                    required_field("centre_speed", &PanelCameraProjectState::centre_speed),
                    required_field("roll_speed", &PanelCameraProjectState::roll_speed),
                    required_field("translate_speed", &PanelCameraProjectState::translate_speed),
                    required_field("wasd_speed", &PanelCameraProjectState::wasd_speed),
                    required_field("max_wasd_speed", &PanelCameraProjectState::max_wasd_speed),
                    nullable_positive_float_field(
                        "ortho_scale", &PanelCameraProjectState::ortho_scale),
                };
            return fields;
        }

    } // namespace

    SessionJson renderSettingsToProjectJson(
        const RenderSettings& settings) {
        return fields_to_json(
            settings, render_settings_fields());
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
        if (auto status = read_fields(
                json,
                settings,
                "VIEW.render_settings",
                render_settings_fields());
            !status) {
            return std::move(status).error();
        }
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
        Json result{{"panel", panel}};
        append_fields(result, state, panel_camera_fields());
        return result;
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
        if (auto status = read_fields(
                json,
                state,
                "VIEW.panel_cameras",
                panel_camera_fields());
            !status) {
            return std::move(status).error();
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
                        lfs::io::project::
                            kUserGlobalGuiFieldKeys,
                        std::string_view(normalized)) !=
                        lfs::io::project::
                            kUserGlobalGuiFieldKeys
                                .end() ||
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
            chapters,
        const lfs::io::project::ReferencesChapter*
            references,
        const std::filesystem::path& project_root) {
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

        PreparedGuiSessionRestore prepared{
            .chapters = std::move(chapters)};
        if (references) {
            const auto env_uuid_json =
                prepared.chapters.view.dom().get_json(
                    "render_settings.environment_reference_uuid");
            if (env_uuid_json &&
                env_uuid_json->is_string()) {
                if (const auto uuid =
                        lfs::core::Uuid::from_string(
                            env_uuid_json
                                ->get<std::string>());
                    uuid && !uuid->is_nil()) {
                    prepared.environment_map_path =
                        lfs::io::project::
                            resolve_path_reference(
                                *references,
                                project_root,
                                *uuid);
                }
            }
            const auto clips_json =
                prepared.chapters.sequencer.dom()
                    .get_json("ply_sequences");
            if (clips_json && clips_json->is_array() &&
                !clips_json->empty() &&
                (*clips_json)[0].is_object()) {
                const auto& clip = (*clips_json)[0];
                const auto ref = clip.find(
                    "directory_reference_uuid");
                const auto hint =
                    clip.value("directory_hint",
                               std::string{});
                if (ref != clip.end() &&
                    ref->is_string()) {
                    if (const auto uuid =
                            lfs::core::Uuid::
                                from_string(
                                    ref->get<
                                        std::string>());
                        uuid && !uuid->is_nil()) {
                        prepared
                            .ply_sequence_directory =
                            lfs::io::project::
                                resolve_path_reference(
                                    *references,
                                    project_root,
                                    *uuid,
                                    lfs::core::
                                        utf8_to_path(
                                            hint));
                    }
                } else if (!hint.empty()) {
                    prepared.ply_sequence_directory =
                        lfs::core::utf8_to_path(hint);
                }
            }
        }
        return prepared;
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
        prepared->ticket = ++next_ticket_;
        pending_ticket_ = prepared->ticket;
        pending_ = std::move(*prepared);
        return {};
    }

    GuiSessionRestoreTicket GuiSessionRestoreCoordinator::stagePrepared(
        PreparedGuiSessionRestore prepared) {
        prepared.ticket = ++next_ticket_;
        pending_ticket_ = prepared.ticket;
        pending_ = std::move(prepared);
        return pending_ticket_;
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
        pending_ticket_ = 0;
        return result;
    }

    void GuiSessionRestoreCoordinator::clear() noexcept {
        pending_.reset();
        pending_ticket_ = 0;
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

        const auto& fixed_layout_fields() {
            static const std::vector<
                JsonField<gui::PanelLayoutProjectState>>
                fields{
                    optional_field("right_panel_width", &gui::PanelLayoutProjectState::right_panel_width),
                    optional_field("scene_panel_ratio", &gui::PanelLayoutProjectState::scene_panel_ratio),
                    optional_field("python_console_width", &gui::PanelLayoutProjectState::python_console_width),
                    optional_field("bottom_dock_height", &gui::PanelLayoutProjectState::bottom_dock_height),
                    optional_field("left_dock_width", &gui::PanelLayoutProjectState::left_dock_width),
                    optional_field("sequencer_visible", &gui::PanelLayoutProjectState::show_sequencer),
                    optional_field("tab_scroll_offset", &gui::PanelLayoutProjectState::tab_scroll_offset),
                };
            return fields;
        }

        const auto& window_fields() {
            using WindowState =
                WindowManager::ProjectWindowState;
            static const std::vector<JsonField<WindowState>> fields{
                optional_field("x", &WindowState::x),
                optional_field("y", &WindowState::y),
                optional_field("width", &WindowState::width),
                optional_field("height", &WindowState::height),
                optional_field("fullscreen", &WindowState::fullscreen),
                optional_field("maximized", &WindowState::maximized),
                optional_field("restore_x", &WindowState::restore_x),
                optional_field("restore_y", &WindowState::restore_y),
                optional_field("restore_width", &WindowState::restore_width),
                optional_field("restore_height", &WindowState::restore_height),
            };
            return fields;
        }

        const auto& panel_fields() {
            using Panel = gui::PanelProjectState;
            const auto nullable_float = [](
                                            const std::string_view name,
                                            float Panel::*member) {
                return custom_field<Panel>(
                    name,
                    [member](const Panel& panel) {
                        return finite_or_null(panel.*member);
                    },
                    [member](const Json& json,
                             Panel& panel,
                             std::string_view,
                             const std::string_view field) {
                        panel.*member =
                            scalar<float>(json, field).value_or(NAN);
                        return lfs::Result<void>{};
                    });
            };
            static const std::vector<JsonField<Panel>> fields{
                optional_field("id", &Panel::id),
                optional_field("parent_id", &Panel::parent_id),
                custom_field<Panel>(
                    "space",
                    [](const Panel& panel) {
                        return Json(panel_space_name(panel.space));
                    },
                    [](const Json& json,
                       Panel& panel,
                       std::string_view,
                       const std::string_view field) {
                        if (const auto name =
                                scalar<std::string>(json, field)) {
                            if (const auto space =
                                    panel_space_from_name(*name)) {
                                panel.space = *space;
                            }
                        }
                        return lfs::Result<void>{};
                    }),
                optional_field("order", &Panel::order),
                optional_field("enabled", &Panel::enabled),
                nullable_float("float_x", &Panel::float_x),
                nullable_float("float_y", &Panel::float_y),
                optional_field("float_user_height", &Panel::float_user_height),
                optional_field("float_last_bounds_valid", &Panel::float_last_bounds_valid),
                optional_field("float_last_x", &Panel::float_last_x),
                optional_field("float_last_y", &Panel::float_last_y),
                optional_field("float_last_w", &Panel::float_last_w),
                optional_field("float_last_h", &Panel::float_last_h),
                optional_field("float_auto_center", &Panel::float_auto_center),
                optional_field("float_stack_order", &Panel::float_stack_order),
            };
            return fields;
        }

        std::string video_preset_name(
            const lfs::io::video::VideoPreset preset) {
            using lfs::io::video::VideoPreset;
            switch (preset) {
            case VideoPreset::YOUTUBE_1080P:
                return "youtube_1080p";
            case VideoPreset::YOUTUBE_4K:
                return "youtube_4k";
            case VideoPreset::HD_720P:
                return "hd_720p";
            case VideoPreset::TIKTOK:
                return "tiktok";
            case VideoPreset::TIKTOK_HD:
                return "tiktok_hd";
            case VideoPreset::INSTAGRAM_SQUARE:
                return "instagram_square";
            case VideoPreset::INSTAGRAM_PORTRAIT:
                return "instagram_portrait";
            case VideoPreset::CUSTOM:
                return "custom";
            }
            return "youtube_1080p";
        }

        std::optional<lfs::io::video::VideoPreset>
        video_preset_from_name(const std::string_view name) {
            using lfs::io::video::VideoPreset;
            if (name == "youtube_1080p")
                return VideoPreset::YOUTUBE_1080P;
            if (name == "youtube_4k")
                return VideoPreset::YOUTUBE_4K;
            if (name == "hd_720p")
                return VideoPreset::HD_720P;
            if (name == "tiktok")
                return VideoPreset::TIKTOK;
            if (name == "tiktok_hd")
                return VideoPreset::TIKTOK_HD;
            if (name == "instagram_square")
                return VideoPreset::INSTAGRAM_SQUARE;
            if (name == "instagram_portrait")
                return VideoPreset::INSTAGRAM_PORTRAIT;
            if (name == "custom")
                return VideoPreset::CUSTOM;
            return std::nullopt;
        }

        const auto& sequencer_preference_fields() {
            using Preferences = gui::panels::SequencerUIState;
            static const std::vector<JsonField<Preferences>> fields{
                optional_field("snap_to_grid", &Preferences::snap_to_grid),
                optional_field("snap_interval", &Preferences::snap_interval),
                optional_field("follow_playback", &Preferences::follow_playback),
                optional_field("show_pip_preview", &Preferences::show_pip_preview),
                optional_field("pip_preview_scale", &Preferences::pip_preview_scale),
                optional_field("show_film_strip", &Preferences::show_film_strip),
                custom_field<Preferences>(
                    "preset",
                    [](const Preferences& prefs) {
                        return Json(video_preset_name(prefs.preset));
                    },
                    [](const Json& json,
                       Preferences& prefs,
                       std::string_view,
                       const std::string_view field) {
                        if (const auto name =
                                scalar<std::string>(json, field)) {
                            if (const auto parsed =
                                    video_preset_from_name(*name)) {
                                prefs.preset = *parsed;
                            }
                        }
                        return lfs::Result<void>{};
                    }),
                optional_field("custom_width", &Preferences::custom_width),
                optional_field("custom_height", &Preferences::custom_height),
                optional_field("framerate", &Preferences::framerate),
                optional_field("quality", &Preferences::quality),
            };
            return fields;
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
            CameraBookmarkProjectState>& bookmarks,
        lfs::io::project::ReferencesChapter* references,
        const std::filesystem::path& project_root) {
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
        Json fixed_payload =
            fields_to_json(layout, fixed_layout_fields());
        fixed_payload["python_console_visible"] =
            console_visible;
        fixed_payload["system_console_visible"] =
            window_states.contains("system_console") &&
            window_states.at("system_console");
        fixed_payload["tab_strip_scroll"] =
            gui_manager->tabStripScroll();
        {
            const auto tree =
                gui_manager->captureSceneTreeChrome(
                    viewer.getScene());
            Json collapsed = Json::array();
            for (const auto& uuid : tree.collapsed_uuids)
                collapsed.push_back(uuid);
            fixed_payload["scene_tree"] = Json{
                {"collapsed_uuids", std::move(collapsed)},
                {"models_collapsed", tree.models_collapsed},
                {"filter_text", tree.filter_text},
            };
        }
        fixed_payload["window"] =
            fields_to_json(window, window_fields());

        Json panels = Json::array();
        for (const auto& panel :
             gui::PanelRegistry::instance()
                 .capture_project_state()) {
            panels.push_back(
                fields_to_json(panel, panel_fields()));
        }
        Json panel_payloads = Json::object();
        for (const auto& [id, text] :
             gui::PanelRegistry::instance()
                 .capture_panel_payloads()) {
            try {
                auto parsed = Json::parse(text);
                if (parsed.is_object())
                    panel_payloads[id] = std::move(parsed);
            } catch (const nlohmann::json::exception&) {
            }
        }
        Json registry_payload{
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
        if (!panel_payloads.empty())
            registry_payload["panel_payloads"] =
                std::move(panel_payloads);

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
            console_payload["splitter_ratio"] =
                gui::panels::PythonConsoleState::
                    splitterRatio();
        } else {
            console_payload["splitter_ratio"] =
                gui::panels::PythonConsoleState::
                    splitterRatio();
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
            std::optional<lfs::core::Uuid>
                retained_uuid;
            const auto retained_reference =
                result.view.dom().get_json(
                    "render_settings.environment_reference_uuid");
            if (retained_reference &&
                retained_reference->is_string()) {
                retained_uuid =
                    lfs::core::Uuid::from_string(
                        retained_reference
                            ->get<std::string>());
            }
            if (references) {
                auto minted =
                    lfs::io::project::
                        upsert_path_reference(
                            *references,
                            project_root,
                            lfs::core::utf8_to_path(
                                settings
                                    .environment_map_path),
                            "view.environment",
                            "environment_map",
                            retained_uuid);
                if (minted) {
                    project_render_settings
                        ["environment_reference_uuid"] =
                            minted->to_string();
                } else if (retained_uuid) {
                    project_render_settings
                        ["environment_reference_uuid"] =
                            retained_uuid->to_string();
                }
            } else if (retained_uuid) {
                project_render_settings
                    ["environment_reference_uuid"] =
                        retained_uuid->to_string();
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
            std::optional<lfs::core::Uuid>
                retained_uuid;
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
                        retained_uuid =
                            lfs::core::Uuid::
                                from_string(
                                    reference->get<
                                        std::string>());
                    }
                }
            }
            if (references &&
                !clip->directory.empty()) {
                const auto key = std::format(
                    "sequencer.ply_sequence.{}",
                    clip->node_uuid.to_string());
                auto minted =
                    lfs::io::project::
                        upsert_path_reference(
                            *references,
                            project_root,
                            clip->directory,
                            key,
                            "ply_sequence_directory",
                            retained_uuid);
                if (minted) {
                    saved_clip
                        ["directory_reference_uuid"] =
                            minted->to_string();
                    if (!project_root.empty()) {
                        std::error_code error;
                        const auto absolute =
                            std::filesystem::absolute(
                                clip->directory, error)
                                .lexically_normal();
                        const auto root =
                            std::filesystem::absolute(
                                project_root, error)
                                .lexically_normal();
                        if (!error) {
                            const auto relative =
                                absolute
                                    .lexically_relative(
                                        root);
                            if (!relative.empty() &&
                                relative != "." &&
                                !relative
                                     .generic_string()
                                     .starts_with(
                                         "..")) {
                                saved_clip
                                    ["directory_hint"] =
                                        lfs::core::
                                            path_to_utf8(
                                                relative
                                                    .generic_string());
                            }
                        }
                    }
                } else if (retained_uuid) {
                    saved_clip
                        ["directory_reference_uuid"] =
                            retained_uuid->to_string();
                }
            } else if (retained_uuid) {
                saved_clip
                    ["directory_reference_uuid"] =
                        retained_uuid->to_string();
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
             fields_to_json(
                 sequencer_ui,
                 sequencer_preference_fields())},
            {"view",
             {
                 {"zoom",
                  gui_manager->sequencerUI()
                      .timelineZoom()},
                 {"pan",
                  gui_manager->sequencerUI()
                      .timelinePan()},
                 {"selected_keyframe_id",
                  controller.selectedKeyframeId()
                      ? Json(*controller
                                  .selectedKeyframeId())
                      : Json(nullptr)},
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

        struct CleanEditorFileRead {
            std::optional<std::string> text;
            std::string error;
            lfs::ErrorCode code = lfs::ErrorCode::NotFound;
        };

        CleanEditorFileRead read_clean_editor_file(
            const std::filesystem::path& path) {
            constexpr std::uintmax_t
                max_editor_file_bytes =
                    64U * 1024U * 1024U;
            std::error_code error;
            const auto size =
                std::filesystem::file_size(
                    path, error);
            if (error)
                return {.text = std::nullopt,
                        .error =
                            "Could not stat " +
                            lfs::core::path_to_utf8(path),
                        .code = lfs::ErrorCode::NotFound};
            if (size > max_editor_file_bytes)
                return {.text = std::nullopt,
                        .error =
                            "Editor file exceeds the 64 MiB restore limit: " +
                            lfs::core::path_to_utf8(path),
                        .code = lfs::ErrorCode::
                            ResourceExhausted};

            std::ifstream stream;
            if (!lfs::core::open_file_for_read(
                    path, std::ios::binary,
                    stream)) {
                return {.text = std::nullopt,
                        .error =
                            "Could not read editor file: " +
                            lfs::core::path_to_utf8(path),
                        .code = lfs::ErrorCode::NotFound};
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
                    return {.text = std::nullopt,
                            .error =
                                "Could not read editor file: " +
                                lfs::core::path_to_utf8(path),
                            .code = lfs::ErrorCode::DataLoss};
            }
            return {.text = std::move(text),
                    .error = {},
                    .code = lfs::ErrorCode::NotFound};
        }

        void publish_editor_restore_error(
            const lfs::ErrorCode code,
            std::string detail) {
            lfs::ErrorBus::instance().publish(
                lfs::ErrorNotification{
                    .error = lfs::make_error(
                                 lfs::ErrorInit{
                                     .code = code,
                                     .domain = lfs::ErrorDomain::IO,
                                     .severity = lfs::Severity::Error,
                                     .retryability =
                                         lfs::Retryability::
                                             NotRetryable,
                                     .operation_id = {},
                                     .user_message =
                                         "Editor file not restored",
                                     .detail = std::move(detail),
                                     .detection =
                                         LFS_SOURCE_SITE_CURRENT(),
                                     .fields = {},
                                     .native = std::nullopt,
                                 })
                                 .with_context(
                                     gui::error_op::kOpenProject,
                                     LFS_SOURCE_SITE_CURRENT()),
                    .surface = lfs::ErrorSurface::Toast,
                    .actions = {},
                    .operation_id =
                        lfs::OperationId::generate(),
                });
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
            (void)read_fields(
                fixed,
                layout,
                "GUIL.fixed_arrangement",
                fixed_layout_fields());
            if (!fixed.contains("tab_scroll_offset"))
                layout.tab_scroll_offset = 0.0f;

            if (const auto window_json =
                    find_required_object(
                        fixed, "window");
                window_json != fixed.end()) {
                auto state =
                    window_manager
                        ->captureProjectState();
                (void)read_fields(
                    *window_json,
                    state,
                    "GUIL.fixed_arrangement.window",
                    window_fields());
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
            if (const auto tree_json =
                    find_required_object(
                        fixed, "scene_tree");
                tree_json != fixed.end()) {
                gui::SceneTreeSessionChrome tree;
                if (const auto uuids =
                        find_required_array(
                            *tree_json,
                            "collapsed_uuids");
                    uuids != tree_json->end()) {
                    for (const auto& entry : *uuids) {
                        if (entry.is_string())
                            tree.collapsed_uuids.push_back(
                                entry.get<std::string>());
                    }
                }
                assign_optional(
                    *tree_json, "models_collapsed",
                    tree.models_collapsed);
                assign_optional(
                    *tree_json, "filter_text",
                    tree.filter_text);
                gui_manager->applySceneTreeChrome(tree);
            } else {
                gui_manager->resetSceneTreeChrome();
            }

            if (const auto tab_strip =
                    scalar<float>(
                        fixed, "tab_strip_scroll")) {
                gui_manager->setTabStripScroll(*tab_strip);
            } else {
                gui_manager->setTabStripScroll(0.0f);
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
                    (void)read_fields(
                        saved,
                        state,
                        "GUIL.panel_registry.panels",
                        panel_fields());
                    panels.push_back(
                        std::move(state));
                }
            }
            gui::PanelRegistry::instance()
                .apply_project_state(panels);

            std::unordered_map<std::string, std::string>
                panel_payloads;
            if (const auto payloads =
                    find_required_object(
                        registry, "panel_payloads");
                payloads != registry.end()) {
                for (const auto& [id, value] :
                     payloads->items()) {
                    if (value.is_object())
                        panel_payloads.insert_or_assign(
                            id, value.dump());
                }
            }
            gui::PanelRegistry::instance()
                .apply_panel_payloads(panel_payloads);

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
                bool system_console_visible = false;
                assign_optional(
                    fixed,
                    "system_console_visible",
                    system_console_visible);
#ifdef WIN32
                gui::UIContext ctx{
                    .viewer = &viewer,
                    .window_states = window_states,
                    .editor = nullptr,
                    .sequencer_controller = nullptr,
                    .fonts = {}};
                gui::panels::SetSystemConsoleVisible(
                    ctx, system_console_visible);
#else
                (*window_states)["system_console"] =
                    system_console_visible;
#endif
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
                    std::clamp(active_tab, 0, 2));
                console_state->setFontScale(
                    font_scale);
                if (const auto ratio =
                        scalar<float>(
                            console,
                            "splitter_ratio")) {
                    gui::panels::PythonConsoleState::
                        setSplitterRatio(*ratio);
                } else {
                    gui::panels::PythonConsoleState::
                        setSplitterRatio(0.6f);
                }
            } else if (const auto ratio =
                           scalar<float>(
                               console,
                               "splitter_ratio")) {
                gui::panels::PythonConsoleState::
                    setSplitterRatio(*ratio);
            } else {
                gui::panels::PythonConsoleState::
                    setSplitterRatio(0.6f);
            }
        }

        void apply_editor(
            VisualizerImpl& viewer,
            const Json& root) {
            (void)viewer;
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
                    auto read =
                        read_clean_editor_file(
                            lfs::core::utf8_to_path(
                                *locator));
                    if (!read.text) {
                        publish_editor_restore_error(
                            read.code, read.error);
                        continue;
                    }
                    text = std::move(*read.text);
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
                bookmarks,
            const std::optional<
                std::filesystem::path>&
                environment_map_path) {
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
            if (environment_map_path &&
                !environment_map_path->empty()) {
                restored->environment_map_path =
                    lfs::core::path_to_utf8(
                        *environment_map_path);
            }
            const auto desired_split =
                restored->split_view_mode;
            const auto saved_split_offset =
                restored->split_view_offset;
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
            auto split_settings =
                rendering->getSettings();
            split_settings.split_view_offset =
                saved_split_offset;
            rendering->updateSettings(
                split_settings);
            if (auto* selection_tool =
                    viewer.getSelectionTool()) {
                selection_tool
                    ->armPreserveRestoredRenderState();
            }

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

        std::optional<ToolType> builtin_tool_type(
            const std::string_view id) {
            if (id == "builtin.select")
                return ToolType::Selection;
            if (id == "builtin.translate")
                return ToolType::Translate;
            if (id == "builtin.rotate")
                return ToolType::Rotate;
            if (id == "builtin.scale")
                return ToolType::Scale;
            if (id == "builtin.mirror")
                return ToolType::Mirror;
            if (id == "builtin.align")
                return ToolType::Align;
            return std::nullopt;
        }

        void deactivate_tool(VisualizerImpl& viewer) {
            lfs::core::events::tools::SetToolbarTool{
                .tool_mode = static_cast<int>(ToolType::None)}
                .emit();
            viewer.getEditorContext().setActiveTool(
                ToolType::None);
            viewer.getEditorContext().clearActiveOperator();
            UnifiedToolRegistry::instance()
                .clearActiveTool();
        }

        void apply_view_tools(
            VisualizerImpl& viewer,
            const Json& root) {
            auto* gui_manager =
                viewer.getGuiManager();
            auto* rendering =
                viewer.getRenderingManager();
            if (!gui_manager || !rendering) {
                deactivate_tool(viewer);
                return;
            }
            const bool sequencer_visible =
                gui_manager->panelLayout().isShowSequencer();
            const auto finish = [&] {
                viewer.getEditorContext()
                    .armToolRestoreGuard();
                gui_manager->panelLayout()
                    .setShowSequencer(sequencer_visible);
                rendering->markDirty(DirtyFlag::ALL);
            };

            const auto tools =
                find_required_object(root, "tools");
            if (tools == root.end()) {
                deactivate_tool(viewer);
                finish();
                return;
            }

            auto& editor = viewer.getEditorContext();
            auto& registry = UnifiedToolRegistry::instance();
            const auto active_tool =
                scalar<std::string>(*tools,
                                    "active_tool_id")
                    .value_or(std::string{});

            if (active_tool == "builtin.cropbox") {
                const auto operation =
                    scalar<std::string>(*tools,
                                        "crop_operation")
                        .value_or("translate");
                const auto gizmo_type =
                    operation == "rotate" ? "rotate" : operation == "scale" ? "scale"
                                                                            : "translate";
                editor.setActiveOperator(
                    "builtin.cropbox", gizmo_type);
                registry.setActiveTool(
                    "builtin.cropbox");
                gui_manager->gizmo()
                    .setCropToolShape(
                        scalar<std::string>(
                            *tools, "crop_shape")
                            .value_or("box"));
                gui_manager->gizmo()
                    .setCropToolOperation(operation);
                if (!editor.hasSelection() ||
                    editor.isToolsDisabled() ||
                    !gui_manager->gizmo()
                         .ensureCropToolStateForRestore()) {
                    deactivate_tool(viewer);
                }
            } else if (const auto type =
                           builtin_tool_type(active_tool);
                       type && editor.isToolAvailable(*type)) {
                lfs::core::events::tools::SetToolbarTool{
                    .tool_mode = static_cast<int>(*type)}
                    .emit();
                if (editor.getActiveTool() != *type)
                    deactivate_tool(viewer);
            } else {
                deactivate_tool(viewer);
            }

            auto submode = scalar<std::string>(
                *tools, "selection_submode");
            if (!submode)
                submode = scalar<std::string>(
                    *tools, "active_submode_id");
            if (submode) {
                if (const auto parsed =
                        selection_submode_from_name(
                            *submode)) {
                    lfs::core::events::tools::SetSelectionSubMode{
                        .selection_mode =
                            static_cast<int>(*parsed)}
                        .emit();
                }
            }
            auto& gizmo = gui_manager->gizmo();
            const auto operation =
                scalar<std::string>(*tools,
                                    "gizmo_operation");
            gizmo.setOperation(
                operation && *operation == "rotate"
                    ? gui::GizmoOperation::Rotate
                : operation && *operation == "scale"
                    ? gui::GizmoOperation::Scale
                    : gui::GizmoOperation::Translate);
            gizmo.setTransformSpace(
                scalar<std::string>(*tools,
                                    "transform_space")
                            .value_or("local") == "world"
                    ? TransformSpace::World
                    : TransformSpace::Local);
            gizmo.setPivotMode(
                scalar<std::string>(*tools,
                                    "pivot_mode")
                            .value_or("origin") ==
                        "bounds_center"
                    ? PivotMode::BoundsCenter
                    : PivotMode::Origin);
            gizmo.setMultiTransformMode(
                scalar<std::string>(*tools,
                                    "multi_transform_mode")
                            .value_or("selection") ==
                        "individual"
                    ? gui::MultiTransformMode::Individual
                    : gui::MultiTransformMode::Selection);
            if (const auto shape =
                    scalar<std::string>(*tools,
                                        "crop_shape"))
                gizmo.setCropToolShape(*shape);
            if (const auto crop_operation =
                    scalar<std::string>(*tools,
                                        "crop_operation"))
                gizmo.setCropToolOperation(*crop_operation);

            if (auto* selection_tool =
                    viewer.getSelectionTool()) {
                if (const auto selection =
                        find_required_object(*tools,
                                             "selection");
                    selection != tools->end()) {
                    selection_tool->restoreProjectPreferences(
                        scalar<float>(*selection,
                                      "brush_radius")
                            .value_or(20.0f),
                        scalar<bool>(*selection,
                                     "crop_filter")
                            .value_or(false),
                        scalar<bool>(*selection,
                                     "depth_filter")
                            .value_or(false),
                        scalar<bool>(
                            *selection,
                            "restrict_to_selected_nodes")
                            .value_or(true));
                    const auto render_json =
                        find_required_object(
                            root, "render_settings");
                    if (render_json != root.end()) {
                        if (auto settings =
                                renderSettingsFromProjectJson(
                                    *render_json,
                                    rendering->getSettings())) {
                            const float near_plane =
                                std::max(0.0f,
                                         -settings
                                              ->depth_filter_max.z);
                            const float far_plane =
                                std::max(near_plane + 0.01f,
                                         -settings
                                              ->depth_filter_min.z);
                            const float half_width =
                                std::max(
                                    std::abs(settings
                                                 ->depth_filter_min.x),
                                    std::abs(settings
                                                 ->depth_filter_max.x));
                            selection_tool
                                ->setDepthFilterRange(
                                    settings
                                        ->depth_filter_enabled,
                                    near_plane, far_plane,
                                    half_width);
                            auto restored =
                                rendering->getSettings();
                            restored.crop_filter_for_selection =
                                settings->crop_filter_for_selection;
                            restored.depth_filter_enabled =
                                settings->depth_filter_enabled;
                            restored.depth_filter_min =
                                settings->depth_filter_min;
                            restored.depth_filter_max =
                                settings->depth_filter_max;
                            restored.depth_filter_transform =
                                settings->depth_filter_transform;
                            rendering->updateSettings(restored);
                        }
                    }
                }
            }
            finish();
        }

        void apply_sequencer(
            VisualizerImpl& viewer,
            const Json& root,
            const std::optional<
                std::filesystem::path>&
                resolved_directory) {
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
                        resolved_directory &&
                                !resolved_directory
                                     ->empty()
                            ? *resolved_directory
                            : lfs::core::utf8_to_path(
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
                (void)read_fields(
                    *preferences,
                    ui,
                    "SEQR.preferences",
                    sequencer_preference_fields());
            }
            // Controller values are canonical over the UI mirrors.
            ui.playback_speed =
                controller.playbackSpeed();
            ui.sequence_fps =
                controller.plySequenceFps();
            if (auto* rendering =
                    viewer.getRenderingManager()) {
                ui.equirectangular =
                    rendering->getSettings()
                        .equirectangular;
            }
            if (const auto view =
                    find_required_object(
                        root, "view");
                view != root.end()) {
                float zoom = gui_manager
                                 ->sequencerUI()
                                 .timelineZoom();
                float pan = gui_manager
                                ->sequencerUI()
                                .timelinePan();
                assign_optional(*view, "zoom", zoom);
                assign_optional(*view, "pan", pan);
                gui_manager->sequencerUI()
                    .setTimelineView(zoom, pan);
                if (view->contains(
                        "selected_keyframe_id") &&
                    (*view)["selected_keyframe_id"]
                        .is_number_unsigned()) {
                    controller.selectKeyframeById(
                        (*view)["selected_keyframe_id"]
                            .get<std::uint64_t>());
                }
            }
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

        // Panel registration can finish before a newly-visible console has
        // rendered its first frame. Materialize the session owner now so
        // GUIL console settings and EDTR are never consumed by a null owner.
        (void)gui::panels::PythonConsoleState::getInstance();

        // VIEW (except tools), GUIL, EDTR, and SEQR apply at the panels-ready
        // boundary. Functional tool activation waits for hydration + SELM.
        apply_view(
            viewer, *view, bookmarks,
            prepared.environment_map_path);
        apply_guil(viewer, *gui);
        apply_editor(viewer, *editor);
        apply_sequencer(
            viewer, *sequencer,
            prepared.ply_sequence_directory);
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

    void applyGuiSessionTools(
        VisualizerImpl& viewer,
        const PreparedGuiSessionRestore& prepared) {
        auto* gui_manager = viewer.getGuiManager();
        const bool sequencer_visible =
            gui_manager &&
            gui_manager->panelLayout().isShowSequencer();
        auto view = chapter_root(
            prepared.chapters.view.dom(), "VIEW");
        if (!view) {
            deactivate_tool(viewer);
            if (gui_manager) {
                gui_manager->panelLayout()
                    .setShowSequencer(sequencer_visible);
            }
            return;
        }
        apply_view_tools(viewer, *view);
    }

    void applyDefaultGuiLayout(VisualizerImpl& viewer) {
        auto gui = chapter_root(
            lfs::io::project::default_session_chapter_dom(
                lfs::io::project::SessionJsonChapterKind::GuiLayout),
            "GUIL");
        if (gui)
            apply_guil(viewer, *gui);
    }

} // namespace lfs::vis::project
