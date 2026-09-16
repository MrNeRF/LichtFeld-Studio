/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "workspace/workspace_serialization.hpp"

#include "core/error.hpp"
#include "rendering/render_constants.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace lfs::vis {
    namespace {

        using Json = WorkspaceJson;

        lfs::Error serializationError(const lfs::ErrorCode code,
                                      std::string detail,
                                      const std::string_view field) {
            lfs::SmallFields fields;
            fields.add("field", field);
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .user_message = "The viewport workspace state is invalid.",
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
            });
        }

        template <typename T>
        lfs::Result<T> failure(const lfs::ErrorCode code, std::string detail,
                               const std::string_view field) {
            return serializationError(code, std::move(detail), field);
        }

        template <typename T>
        lfs::Result<T> required(const Json& object, const std::string_view key,
                                const std::string_view field) {
            if (!object.is_object())
                return failure<T>(lfs::ErrorCode::DataLoss, "Expected an object", field);
            const auto it = object.find(std::string(key));
            if (it == object.end())
                return failure<T>(lfs::ErrorCode::DataLoss, "Required field is missing", field);
            try {
                return it->get<T>();
            } catch (const Json::exception& error) {
                return failure<T>(lfs::ErrorCode::DataLoss, error.what(), field);
            }
        }

        lfs::Result<std::uint64_t> unsignedId(const Json& object, const std::string_view key,
                                              const std::string_view field) {
            const auto it = object.find(std::string(key));
            if (it == object.end() || !it->is_number_unsigned())
                return failure<std::uint64_t>(lfs::ErrorCode::DataLoss,
                                              "Expected an unsigned integer", field);
            return it->get<std::uint64_t>();
        }

        lfs::Result<std::optional<std::uint64_t>> optionalId(
            const Json& object, const std::string_view key, const std::string_view field) {
            const auto it = object.find(std::string(key));
            if (it == object.end())
                return failure<std::optional<std::uint64_t>>(
                    lfs::ErrorCode::DataLoss, "Required field is missing", field);
            if (it->is_null())
                return std::optional<std::uint64_t>{};
            if (!it->is_number_unsigned())
                return failure<std::optional<std::uint64_t>>(
                    lfs::ErrorCode::DataLoss, "Expected an unsigned integer or null", field);
            return std::optional<std::uint64_t>{it->get<std::uint64_t>()};
        }

        // New optional workspace fields are absent in older files. Keep the
        // distinction between a missing field and a malformed present field
        // while allowing the former to use the model's legacy fallback.
        lfs::Result<std::optional<std::uint64_t>> optionalIdIfPresent(
            const Json& object, const std::string_view key, const std::string_view field) {
            const auto it = object.find(std::string(key));
            if (it == object.end())
                return std::optional<std::uint64_t>{};
            if (it->is_null())
                return std::optional<std::uint64_t>{};
            if (!it->is_number_unsigned())
                return failure<std::optional<std::uint64_t>>(
                    lfs::ErrorCode::DataLoss, "Expected an unsigned integer or null", field);
            return std::optional<std::uint64_t>{it->get<std::uint64_t>()};
        }

        template <std::size_t N>
        lfs::Result<std::array<float, N>> floatArray(const Json& value,
                                                     const std::string_view field) {
            if (!value.is_array() || value.size() != N)
                return failure<std::array<float, N>>(
                    lfs::ErrorCode::DataLoss, "Expected a finite numeric array", field);
            std::array<float, N> result{};
            for (std::size_t i = 0; i < N; ++i) {
                if (!value[i].is_number())
                    return failure<std::array<float, N>>(
                        lfs::ErrorCode::DataLoss, "Expected a finite numeric array", field);
                try {
                    result[i] = value[i].get<float>();
                } catch (const Json::exception& error) {
                    return failure<std::array<float, N>>(
                        lfs::ErrorCode::DataLoss, error.what(), field);
                }
                if (!std::isfinite(result[i]))
                    return failure<std::array<float, N>>(
                        lfs::ErrorCode::DataLoss, "Array contains a non-finite value", field);
            }
            return result;
        }

        template <std::size_t N>
        lfs::Result<std::array<int, N>> intArray(const Json& value,
                                                 const std::string_view field) {
            if (!value.is_array() || value.size() != N)
                return failure<std::array<int, N>>(
                    lfs::ErrorCode::DataLoss, "Expected an integer array", field);
            std::array<int, N> result{};
            for (std::size_t i = 0; i < N; ++i) {
                if (!value[i].is_number_integer() && !value[i].is_number_unsigned())
                    return failure<std::array<int, N>>(
                        lfs::ErrorCode::DataLoss, "Expected an integer array", field);
                try {
                    result[i] = value[i].get<int>();
                } catch (const Json::exception& error) {
                    return failure<std::array<int, N>>(
                        lfs::ErrorCode::DataLoss, error.what(), field);
                }
            }
            return result;
        }

        template <std::size_t N>
        Json floatArray(const std::array<float, N>& values) {
            Json result = Json::array();
            for (const float value : values)
                result.push_back(value);
            return result;
        }

        std::array<float, 9> matrixArray(const glm::mat3& matrix) {
            std::array<float, 9> result{};
            for (int column = 0; column < 3; ++column)
                for (int row = 0; row < 3; ++row)
                    result[static_cast<std::size_t>(column * 3 + row)] = matrix[column][row];
            return result;
        }

        glm::mat3 matrixFromArray(const std::array<float, 9>& values) {
            glm::mat3 result{1.0f};
            for (int column = 0; column < 3; ++column)
                for (int row = 0; row < 3; ++row)
                    result[column][row] = values[static_cast<std::size_t>(column * 3 + row)];
            return result;
        }

        std::array<float, 3> vectorArray(const glm::vec3& value) {
            return {value.x, value.y, value.z};
        }

        glm::vec3 vectorFromArray(const std::array<float, 3>& values) {
            return {values[0], values[1], values[2]};
        }

        Json projectionToJson(const ViewProjectionState& projection) {
            return {
                {"focal_length_mm", projection.focal_length_mm},
                {"orthographic", projection.orthographic},
                {"ortho_scale", projection.ortho_scale},
                {"near_plane", projection.near_plane},
                {"far_plane", projection.far_plane},
                {"equirectangular", projection.equirectangular},
            };
        }

        lfs::Result<ViewProjectionState> projectionFromJson(const Json& json,
                                                            const std::string_view field) {
            if (!json.is_object())
                return failure<ViewProjectionState>(lfs::ErrorCode::DataLoss,
                                                    "Expected an object", field);
            ViewProjectionState projection;
            auto focal = required<float>(json, "focal_length_mm", field);
            auto orthographic = required<bool>(json, "orthographic", field);
            auto scale = required<float>(json, "ortho_scale", field);
            auto near_plane = required<float>(json, "near_plane", field);
            auto far_plane = required<float>(json, "far_plane", field);
            auto equirectangular = required<bool>(json, "equirectangular", field);
            if (!focal || !orthographic || !scale || !near_plane || !far_plane || !equirectangular)
                return serializationError(lfs::ErrorCode::DataLoss,
                                          "Projection field is missing or has the wrong type",
                                          field);
            projection.focal_length_mm = *focal;
            projection.orthographic = *orthographic;
            projection.ortho_scale = *scale;
            projection.near_plane = *near_plane;
            projection.far_plane = *far_plane;
            projection.equirectangular = *equirectangular;
            return projection;
        }

        Json depthToJson(const DepthWindowState& depth) {
            return {
                {"near_plane", depth.near_plane},
                {"far_plane", depth.far_plane},
                {"scale_x", depth.scale_x},
                {"scale_y", depth.scale_y},
                {"offset_x", depth.offset_x},
                {"offset_y", depth.offset_y},
            };
        }

        lfs::Result<DepthWindowState> depthFromJson(const Json& json,
                                                    const std::string_view field) {
            if (!json.is_object())
                return failure<DepthWindowState>(lfs::ErrorCode::DataLoss,
                                                 "Expected an object", field);
            DepthWindowState depth;
            auto near_plane = required<float>(json, "near_plane", field);
            auto far_plane = required<float>(json, "far_plane", field);
            auto scale_x = required<float>(json, "scale_x", field);
            auto scale_y = required<float>(json, "scale_y", field);
            auto offset_x = required<float>(json, "offset_x", field);
            auto offset_y = required<float>(json, "offset_y", field);
            if (!near_plane || !far_plane || !scale_x || !scale_y || !offset_x || !offset_y)
                return serializationError(lfs::ErrorCode::DataLoss,
                                          "Depth-window field is missing or has the wrong type",
                                          field);
            depth.near_plane = *near_plane;
            depth.far_plane = *far_plane;
            depth.scale_x = *scale_x;
            depth.scale_y = *scale_y;
            depth.offset_x = *offset_x;
            depth.offset_y = *offset_y;
            return depth;
        }

        Json viewToJson(const ViewPersistentState& view) {
            return {
                {"id", view.id},
                {"rotation", floatArray(matrixArray(view.rotation))},
                {"translation", floatArray(vectorArray(view.translation))},
                {"pivot", floatArray(vectorArray(view.pivot))},
                {"home_rotation", floatArray(matrixArray(view.home_rotation))},
                {"home_translation", floatArray(vectorArray(view.home_translation))},
                {"home_pivot", floatArray(vectorArray(view.home_pivot))},
                {"home_saved", view.home_saved},
                {"zoom_speed", view.zoom_speed},
                {"max_zoom_speed", view.max_zoom_speed},
                {"rotate_speed", view.rotate_speed},
                {"rotate_center_speed", view.rotate_center_speed},
                {"rotate_roll_speed", view.rotate_roll_speed},
                {"translate_speed", view.translate_speed},
                {"wasd_speed", view.wasd_speed},
                {"max_wasd_speed", view.max_wasd_speed},
                {"ortho_scale_override", view.ortho_scale_override
                                             ? Json(*view.ortho_scale_override)
                                             : Json(nullptr)},
                {"editor_id", view.editor_id},
                {"chrome", view.chrome},
                {"projection", projectionToJson(view.projection)},
                {"depth", depthToJson(view.depth)},
                {"grid_plane", view.grid_plane},
                {"window_size", Json::array({view.window_size.x, view.window_size.y})},
                {"framebuffer_size", Json::array({view.framebuffer_size.x,
                                                  view.framebuffer_size.y})},
                {"state_generation", view.state_generation},
            };
        }

        lfs::Result<ViewPersistentState> viewFromJson(const Json& json,
                                                      const std::string_view field) {
            if (!json.is_object())
                return failure<ViewPersistentState>(lfs::ErrorCode::DataLoss,
                                                    "Expected an object", field);
            ViewPersistentState view;
            auto id = unsignedId(json, "id", field);
            auto rotation = floatArray<9>(json.value("rotation", Json{}), field);
            auto translation = floatArray<3>(json.value("translation", Json{}), field);
            auto pivot = floatArray<3>(json.value("pivot", Json{}), field);
            auto home_rotation = floatArray<9>(json.value("home_rotation", Json{}), field);
            auto home_translation =
                floatArray<3>(json.value("home_translation", Json{}), field);
            auto home_pivot = floatArray<3>(json.value("home_pivot", Json{}), field);
            auto home_saved = required<bool>(json, "home_saved", field);
            auto zoom_speed = required<float>(json, "zoom_speed", field);
            auto max_zoom_speed = required<float>(json, "max_zoom_speed", field);
            auto rotate_speed = required<float>(json, "rotate_speed", field);
            auto rotate_center_speed = required<float>(json, "rotate_center_speed", field);
            auto rotate_roll_speed = required<float>(json, "rotate_roll_speed", field);
            auto translate_speed = required<float>(json, "translate_speed", field);
            auto wasd_speed = required<float>(json, "wasd_speed", field);
            auto max_wasd_speed = required<float>(json, "max_wasd_speed", field);
            auto projection_json = json.find("projection");
            auto depth_json = json.find("depth");
            auto projection = projection_json == json.end()
                                  ? lfs::Result<ViewProjectionState>(serializationError(
                                        lfs::ErrorCode::DataLoss,
                                        "Required field is missing", field))
                                  : projectionFromJson(*projection_json, field);
            auto depth = depth_json == json.end()
                             ? lfs::Result<DepthWindowState>(serializationError(
                                   lfs::ErrorCode::DataLoss,
                                   "Required field is missing", field))
                             : depthFromJson(*depth_json, field);
            auto grid_plane = required<int>(json, "grid_plane", field);
            auto state_generation = unsignedId(json, "state_generation", field);
            const auto ortho = json.find("ortho_scale_override");
            const auto editor = json.find("editor_id");
            const auto chrome = json.find("chrome");
            const auto window = json.find("window_size");
            const auto framebuffer = json.find("framebuffer_size");
            if (!id || !rotation || !translation || !pivot || !home_rotation ||
                !home_translation || !home_pivot || !home_saved || !zoom_speed ||
                !max_zoom_speed || !rotate_speed || !rotate_center_speed || !rotate_roll_speed ||
                !translate_speed || !wasd_speed || !max_wasd_speed || !projection || !depth ||
                !grid_plane || !state_generation || ortho == json.end() || window == json.end() ||
                framebuffer == json.end() ||
                (editor != json.end() && !editor->is_string()) ||
                (chrome != json.end() && !chrome->is_object())) {
                return serializationError(lfs::ErrorCode::DataLoss,
                                          "View field is missing or has the wrong type", field);
            }
            auto window_values = intArray<2>(*window, field);
            auto framebuffer_values = intArray<2>(*framebuffer, field);
            if (!window_values || !framebuffer_values)
                return serializationError(lfs::ErrorCode::DataLoss,
                                          "View size must contain two finite numbers", field);
            if (!ortho->is_null() && !ortho->is_number())
                return serializationError(lfs::ErrorCode::DataLoss,
                                          "Ortho scale override must be a number or null", field);
            view.id = *id;
            view.rotation = matrixFromArray(*rotation);
            view.translation = vectorFromArray(*translation);
            view.pivot = vectorFromArray(*pivot);
            view.home_rotation = matrixFromArray(*home_rotation);
            view.home_translation = vectorFromArray(*home_translation);
            view.home_pivot = vectorFromArray(*home_pivot);
            view.home_saved = *home_saved;
            view.zoom_speed = *zoom_speed;
            view.max_zoom_speed = *max_zoom_speed;
            view.rotate_speed = *rotate_speed;
            view.rotate_center_speed = *rotate_center_speed;
            view.rotate_roll_speed = *rotate_roll_speed;
            view.translate_speed = *translate_speed;
            view.wasd_speed = *wasd_speed;
            view.max_wasd_speed = *max_wasd_speed;
            if (!ortho->is_null()) {
                try {
                    view.ortho_scale_override = ortho->get<float>();
                } catch (const Json::exception& error) {
                    return serializationError(lfs::ErrorCode::DataLoss, error.what(), field);
                }
            }
            if (editor != json.end()) {
                try {
                    view.editor_id = editor->get<std::string>();
                } catch (const Json::exception& error) {
                    return serializationError(lfs::ErrorCode::DataLoss, error.what(), field);
                }
            }
            if (chrome != json.end()) {
                for (const auto& [key, value] : chrome->items()) {
                    if (!value.is_string())
                        return serializationError(lfs::ErrorCode::DataLoss,
                                                  "Chrome values must be strings", field);
                    try {
                        view.chrome.emplace(key, value.get<std::string>());
                    } catch (const Json::exception& error) {
                        return serializationError(lfs::ErrorCode::DataLoss, error.what(), field);
                    }
                }
            }
            view.projection = *projection;
            view.depth = *depth;
            view.grid_plane = *grid_plane;
            view.window_size = {(*window_values)[0], (*window_values)[1]};
            view.framebuffer_size = {(*framebuffer_values)[0], (*framebuffer_values)[1]};
            view.state_generation = *state_generation;
            return view;
        }

        Json layoutNodeToJson(const LayoutNodeState& node) {
            return {
                {"id", node.id},
                {"kind", static_cast<std::uint8_t>(node.kind)},
                {"view_id", node.view_id},
                {"axis", static_cast<std::uint8_t>(node.axis)},
                {"ratio", node.ratio},
                {"first", node.first},
                {"second", node.second},
            };
        }

        lfs::Result<LayoutNodeState> layoutNodeFromJson(const Json& json,
                                                        const std::string_view field) {
            if (!json.is_object())
                return failure<LayoutNodeState>(lfs::ErrorCode::DataLoss,
                                                "Expected an object", field);
            LayoutNodeState node;
            auto id = unsignedId(json, "id", field);
            auto kind = required<std::uint8_t>(json, "kind", field);
            auto view_id = unsignedId(json, "view_id", field);
            auto axis = required<std::uint8_t>(json, "axis", field);
            auto ratio = required<float>(json, "ratio", field);
            auto first = unsignedId(json, "first", field);
            auto second = unsignedId(json, "second", field);
            if (!id || !kind || !view_id || !axis || !ratio || !first || !second)
                return serializationError(lfs::ErrorCode::DataLoss,
                                          "Layout node field is missing or has the wrong type",
                                          field);
            node.id = *id;
            node.kind = static_cast<LayoutNodeKind>(*kind);
            node.view_id = *view_id;
            node.axis = static_cast<SplitAxis>(*axis);
            node.ratio = *ratio;
            node.first = *first;
            node.second = *second;
            return node;
        }

    } // namespace

    lfs::Result<WorkspaceJson> workspaceStateToJson(const WorkspacePersistentState& state) {
        if (auto status = validateWorkspaceState(state); !status)
            return std::move(status).error();
        Json layout{
            {"root", state.layout.root},
            {"next_node_id", state.layout.next_node_id},
            {"generation", state.layout.generation},
            {"focused", state.layout.focused ? Json(*state.layout.focused) : Json(nullptr)},
            {"maximized", state.layout.maximized ? Json(*state.layout.maximized) : Json(nullptr)},
            {"nodes", Json::array()},
        };
        for (const auto& node : state.layout.nodes)
            layout["nodes"].push_back(layoutNodeToJson(node));
        Json views = Json::array();
        for (const auto& view : state.views)
            views.push_back(viewToJson(view));
        return Json{
            {"version", 1},
            {"next_view_id", state.next_view_id},
            {"registry_generation", state.registry_generation},
            {"active_viewport", state.active_viewport ? Json(*state.active_viewport)
                                                      : Json(nullptr)},
            {"layout", std::move(layout)},
            {"views", std::move(views)},
        };
    }

    lfs::Result<WorkspacePersistentState> workspaceStateFromJson(const WorkspaceJson& json) {
        if (!json.is_object())
            return failure<WorkspacePersistentState>(lfs::ErrorCode::DataLoss,
                                                     "Workspace state must be an object",
                                                     "VIEW.workspace");
        auto version = required<int>(json, "version", "VIEW.workspace.version");
        auto next_view_id = unsignedId(json, "next_view_id", "VIEW.workspace.next_view_id");
        auto registry_generation =
            unsignedId(json, "registry_generation", "VIEW.workspace.registry_generation");
        auto active_viewport = optionalIdIfPresent(
            json, "active_viewport", "VIEW.workspace.active_viewport");
        const auto layout_json = json.find("layout");
        const auto views_json = json.find("views");
        const bool unsupported_version = version && *version != 1;
        if (!version || !next_view_id || !registry_generation || !active_viewport ||
            layout_json == json.end() ||
            views_json == json.end() || unsupported_version || !layout_json->is_object() ||
            !views_json->is_array()) {
            return failure<WorkspacePersistentState>(
                unsupported_version ? lfs::ErrorCode::Unsupported : lfs::ErrorCode::DataLoss,
                "Workspace state is missing required fields or has an unsupported version",
                "VIEW.workspace");
        }

        WorkspacePersistentState state;
        state.format_version = 1;
        state.next_view_id = *next_view_id;
        state.registry_generation = *registry_generation;
        state.active_viewport = *active_viewport;
        auto root = unsignedId(*layout_json, "root", "VIEW.workspace.layout.root");
        auto next_node_id = unsignedId(*layout_json, "next_node_id",
                                       "VIEW.workspace.layout.next_node_id");
        auto generation = unsignedId(*layout_json, "generation",
                                     "VIEW.workspace.layout.generation");
        auto focused = optionalId(*layout_json, "focused", "VIEW.workspace.layout.focused");
        auto maximized = optionalId(*layout_json, "maximized",
                                    "VIEW.workspace.layout.maximized");
        const auto nodes_json = layout_json->find("nodes");
        if (!root || !next_node_id || !generation || !focused || !maximized ||
            nodes_json == layout_json->end() || !nodes_json->is_array())
            return failure<WorkspacePersistentState>(lfs::ErrorCode::DataLoss,
                                                     "Workspace layout is malformed",
                                                     "VIEW.workspace.layout");
        state.layout.root = *root;
        state.layout.next_node_id = *next_node_id;
        state.layout.generation = *generation;
        if (*focused)
            state.layout.focused = **focused;
        if (*maximized)
            state.layout.maximized = **maximized;
        state.layout.nodes.reserve(nodes_json->size());
        for (std::size_t index = 0; index < nodes_json->size(); ++index) {
            auto node = layoutNodeFromJson(nodes_json->at(index),
                                           "VIEW.workspace.layout.nodes");
            if (!node)
                return std::move(node).error();
            state.layout.nodes.push_back(std::move(*node));
        }

        state.views.reserve(views_json->size());
        for (std::size_t index = 0; index < views_json->size(); ++index) {
            auto view = viewFromJson(views_json->at(index), "VIEW.workspace.views");
            if (!view)
                return std::move(view).error();
            state.views.push_back(std::move(*view));
        }
        if (auto status = validateWorkspaceState(state); !status)
            return std::move(status).error();
        return state;
    }

    lfs::Status initializeWorkspacePrimaryFromLegacy(ViewportWorkspace& workspace,
                                                     const Viewport& legacy_viewport,
                                                     const RenderSettings& settings) {
        WorkspacePersistentState state = workspace.exportState();
        const ViewId primary = workspace.primaryView();
        if (primary == kInvalidViewId || state.views.empty())
            return lfs::Status::failure(serializationError(
                lfs::ErrorCode::FailedPrecondition,
                "Workspace has no primary view", "VIEW.workspace"));
        // A legacy VIEW has one primary camera and no workspace tree. Start
        // the new workspace in a coherent single-pane layout, retaining the
        // registry high-water mark so retired IDs cannot be reused.
        state.views.erase(
            std::remove_if(state.views.begin(), state.views.end(),
                           [primary](const ViewPersistentState& item) {
                               return item.id != primary;
                           }),
            state.views.end());
        state.layout = WorkspaceLayout::single(primary).exportState();
        state.layout.next_node_id = std::max(state.layout.next_node_id,
                                             workspace.layout().nextNodeId());
        state.layout.generation = workspace.layout().generation();
        state.layout.focused = primary;
        state.layout.maximized.reset();
        state.active_viewport = primary;

        auto view = std::find_if(state.views.begin(), state.views.end(),
                                 [primary](const ViewPersistentState& item) {
                                     return item.id == primary;
                                 });
        if (view == state.views.end())
            return lfs::Status::failure(serializationError(
                lfs::ErrorCode::ContractViolation,
                "Workspace primary view is missing from its registry", "VIEW.workspace.views"));

        const auto& camera = legacy_viewport.camera;
        view->rotation = camera.R;
        view->translation = camera.t;
        view->pivot = camera.pivot;
        view->home_rotation = camera.home_R;
        view->home_translation = camera.home_t;
        view->home_pivot = camera.home_pivot;
        view->home_saved = camera.home_saved;
        view->zoom_speed = camera.zoomSpeed;
        view->max_zoom_speed = camera.maxZoomSpeed;
        view->rotate_speed = camera.rotateSpeed;
        view->rotate_center_speed = camera.rotateCenterSpeed;
        view->rotate_roll_speed = camera.rotateRollSpeed;
        view->translate_speed = camera.translateSpeed;
        view->wasd_speed = camera.wasdSpeed;
        view->max_wasd_speed = camera.maxWasdSpeed;
        view->ortho_scale_override = legacy_viewport.ortho_scale_override;
        view->projection.focal_length_mm = settings.focal_length_mm;
        view->projection.orthographic = settings.orthographic;
        view->projection.ortho_scale =
            legacy_viewport.ortho_scale_override.value_or(settings.ortho_scale);
        view->projection.near_plane = lfs::rendering::DEFAULT_NEAR_PLANE;
        view->projection.far_plane = settings.depth_clip_enabled
                                         ? settings.depth_clip_far
                                         : lfs::rendering::DEFAULT_FAR_PLANE;
        view->projection.equirectangular = settings.equirectangular;
        const bool legacy_depth_default =
            !settings.depth_filter_enabled &&
            settings.depth_filter_min.z == 0.0f &&
            settings.depth_filter_max.z == 100.0f;
        view->depth.near_plane = legacy_depth_default ? 0.0f
                                                      : -settings.depth_filter_max.z;
        view->depth.far_plane = legacy_depth_default ? 100.0f
                                                     : -settings.depth_filter_min.z;
        view->depth.scale_x = settings.depth_filter_scale_x;
        view->depth.scale_y = settings.depth_filter_scale_y;
        view->depth.offset_x = settings.depth_filter_offset_x;
        view->depth.offset_y = settings.depth_filter_offset_y;
        view->grid_plane = settings.grid_plane;
        view->window_size = legacy_viewport.windowSize;
        view->framebuffer_size = legacy_viewport.frameBufferSize;
        ++view->state_generation;
        return workspace.importState(state);
    }

} // namespace lfs::vis
