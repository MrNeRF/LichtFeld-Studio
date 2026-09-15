/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/mcp_workspace_tools.hpp"

#include "app/mcp_app_utils.hpp"
#include "mcp/mcp_tools.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "visualizer/gui/gui_manager.hpp"
#include "visualizer/gui/panel_registry.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/visualizer.hpp"
#include "visualizer/visualizer_impl.hpp"
#include "visualizer/workspace/view_registry.hpp"
#include "visualizer/workspace/viewport_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <utility>

#include <glm/glm.hpp>

namespace lfs::app {
    namespace {

        using json = nlohmann::json;
        using mcp::McpResourceContent;
        using mcp::McpTool;
        using vis::ViewId;

        constexpr std::string_view kWorkspaceUnavailable =
            "Viewport workspace is unavailable";

        json vec3_json(const glm::vec3& value) {
            return json::array({value.x, value.y, value.z});
        }

        std::expected<ViewId, std::string> view_id_arg(const json& args) {
            if (!args.contains("view_id") ||
                (!args["view_id"].is_number_integer() && !args["view_id"].is_number_unsigned()))
                return std::unexpected("Field 'view_id' must be a non-negative integer");
            const auto& value = args["view_id"];
            if (!value.is_number_unsigned() && value.get<std::int64_t>() <= 0)
                return std::unexpected("Field 'view_id' must be non-zero");
            if (value.is_number_unsigned() && value.get<std::uint64_t>() == 0)
                return std::unexpected("Field 'view_id' must be non-zero");
            const auto id = value.get<ViewId>();
            if (id == vis::kInvalidViewId)
                return std::unexpected("Field 'view_id' must be non-zero");
            return id;
        }

        std::expected<float, std::string> finite_float_arg(const json& args,
                                                           const char* name,
                                                           const float fallback = 0.0f) {
            if (!args.contains(name))
                return fallback;
            if (!args[name].is_number())
                return std::unexpected(std::string("Field '") + name + "' must be a number");
            const float value = args[name].get<float>();
            if (!std::isfinite(value))
                return std::unexpected(std::string("Field '") + name + "' must be finite");
            return value;
        }

        std::expected<glm::vec3, std::string> vec3_arg(const json& args,
                                                       const char* name,
                                                       const glm::vec3& fallback = {}) {
            if (!args.contains(name))
                return fallback;
            const auto& value = args[name];
            if (!value.is_array() || value.size() != 3)
                return std::unexpected(std::string("Field '") + name + "' must be a 3-element array");
            glm::vec3 result;
            for (std::size_t i = 0; i < 3; ++i) {
                if (!value[i].is_number())
                    return std::unexpected(std::string("Field '") + name + "' must contain numbers");
                result[i] = value[i].get<float>();
                if (!std::isfinite(result[i]))
                    return std::unexpected(std::string("Field '") + name + "' must contain finite numbers");
            }
            return result;
        }

        json workspace_state_json(const vis::Visualizer& viewer, const vis::ViewportWorkspace& workspace) {
            const auto* impl = dynamic_cast<const vis::VisualizerImpl*>(&viewer);
            const auto* gui = impl ? impl->getGuiManager() : nullptr;
            const auto* rendering = impl ? impl->getRenderingManager() : nullptr;
            const auto snapshot = gui ? gui->workspaceSnapshot() : vis::WorkspaceFrameSnapshot{};
            const auto persistent = workspace.exportState();

            json result{
                {"success", true},
                {"active_viewport", workspace.activeViewport() ? json(*workspace.activeViewport()) : json(nullptr)},
                {"focused", persistent.layout.focused ? json(*persistent.layout.focused) : json(nullptr)},
                {"maximized", persistent.layout.maximized ? json(*persistent.layout.maximized) : json(nullptr)},
                {"primary", workspace.primaryView()},
                {"layout_generation", persistent.layout.generation},
                {"registry_generation", persistent.registry_generation},
                {"views", json::array()},
                {"splitters", json::array()},
            };

            for (const auto& record : persistent.views) {
                json view{
                    {"id", record.id},
                    {"editor_id", record.editor_id},
                    {"eye", vec3_json(record.translation)},
                    {"target", vec3_json(record.pivot)},
                    {"up", vec3_json(record.rotation[1])},
                    {"orthographic", record.projection.orthographic},
                    {"focal_length_mm", record.projection.focal_length_mm},
                    {"ortho_scale", record.projection.ortho_scale},
                    {"near_plane", record.projection.near_plane},
                    {"far_plane", record.projection.far_plane},
                    {"grid_plane", record.grid_plane},
                    {"visible", false},
                    // A valid camera need not have a published image (empty scene or deferred frame).
                    {"published", false},
                    {"fresh", false},
                    {"source", "none"},
                    // Matches the renderer's tagged sceneOutputKey(ViewId)
                    // without exposing Vulkan handles across MCP.
                    {"output_key", std::string("scene:") + std::to_string(record.id)},
                    {"color_image_generation", 0},
                    {"color_external_generation", 0},
                    {"depth_external_generation", 0},
                    {"color_has_image", false},
                    {"depth_has_image", false},
                    {"render_size", {0, 0}},
                    {"framebuffer_size", {0, 0}},
                    {"matches_viewport_extent", false},
                    {"render_diagnostic", ""},
                };
                for (const auto& area : snapshot.areas) {
                    if (area.id != record.id)
                        continue;
                    view["visible"] = true;
                    view["area_rect"] = {area.rect.x, area.rect.y, area.rect.width, area.rect.height};
                    view["rect"] = {area.content_rect.x, area.content_rect.y,
                                    area.content_rect.width, area.content_rect.height};
                    break;
                }
                glm::ivec2 output_extent{0};
                for (const auto& pane : snapshot.panes) {
                    if (pane.id != record.id)
                        continue;
                    view["visible"] = true;
                    view["rect"] = {pane.rect.x, pane.rect.y, pane.rect.width, pane.rect.height};
                    output_extent = pane.framebuffer_size;
                    view["framebuffer_size"] = {output_extent.x, output_extent.y};
                    break;
                }
                if (rendering && record.editor_id == "viewport") {
                    if (const auto frame = rendering->getWorkspaceVulkanFrame(record.id)) {
                        const auto source = frame->source == vis::RenderingManager::WorkspaceVulkanFrame::Source::Gaussian
                                                ? "gaussian"
                                            : frame->source == vis::RenderingManager::WorkspaceVulkanFrame::Source::PointCloud
                                                ? "point_cloud"
                                                : "none";
                        view["published"] = true;
                        view["fresh"] = frame->fresh;
                        view["source"] = source;
                        view["color_image_generation"] = frame->color.image_generation;
                        view["color_external_generation"] = frame->color.external_image_generation;
                        view["depth_external_generation"] = frame->geometry.depth_blit.external_image_generation;
                        view["color_has_image"] = static_cast<bool>(frame->color.image) ||
                                                  frame->color.external_image != VK_NULL_HANDLE;
                        view["depth_has_image"] = static_cast<bool>(frame->geometry.depth_blit.depth) ||
                                                  frame->geometry.depth_blit.external_image != VK_NULL_HANDLE;
                        view["render_size"] = {frame->color.size.x, frame->color.size.y};
                        view["matches_viewport_extent"] = frame->color.matches_viewport_extent &&
                                                          frame->pane.framebuffer_size == output_extent;
                        view["render_diagnostic"] = frame->diagnostic;
                    }
                }
                result["views"].push_back(std::move(view));
            }

            for (const auto& node : persistent.layout.nodes) {
                if (node.kind != vis::LayoutNodeKind::Split)
                    continue;
                result["splitters"].push_back({
                    {"id", node.id},
                    {"axis", node.axis == vis::SplitAxis::Horizontal ? "horizontal" : "vertical"},
                    {"ratio", node.ratio},
                });
            }
            return result;
        }

        json workspace_result(const vis::Visualizer& viewer, vis::ViewportWorkspace* workspace) {
            return workspace ? workspace_state_json(viewer, *workspace)
                             : json{{"error", kWorkspaceUnavailable}};
        }

        void mark_workspace_dirty(vis::Visualizer& viewer);

        template <typename F>
        json mutate_workspace(vis::Visualizer* viewer, F&& fn) {
            return post_and_wait(viewer, [viewer, fn = std::forward<F>(fn)]() mutable -> json {
                auto* const workspace = viewer->getViewportWorkspace();
                if (!workspace)
                    return json{{"error", kWorkspaceUnavailable}};
                if (auto result = fn(*workspace); !result)
                    return json{{"error", std::string(result.error().user_message())}};
                mark_workspace_dirty(*viewer);
                return workspace_state_json(*viewer, *workspace);
            });
        }

        mcp::McpToolMetadata query_metadata() {
            return {.category = "workspace", .kind = "query", .runtime = "gui", .thread_affinity = "gui_thread"};
        }

        mcp::McpToolMetadata command_metadata() {
            return {.category = "workspace", .kind = "command", .runtime = "gui", .thread_affinity = "gui_thread"};
        }

        json view_id_property() {
            return json{{"type", "integer"}, {"minimum", 1}, {"description", "Stable non-zero viewport view ID"}};
        }

        void mark_workspace_dirty(vis::Visualizer& viewer) {
            if (auto* renderer = viewer.getRenderingManager())
                renderer->markDirty(vis::DirtyFlag::CAMERA | vis::DirtyFlag::VIEWPORT);
        }

    } // namespace

    void register_gui_workspace_tools(mcp::ToolRegistry& registry,
                                      vis::Visualizer* viewer) {
        registry.register_tool(
            McpTool{
                .name = "workspace_get",
                .description = "Get stable viewport view IDs, layout, cameras and visible pane rectangles",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = query_metadata()},
            [viewer](const json&) -> json {
                return post_and_wait(viewer, [viewer]() -> json {
                    return workspace_result(*viewer, viewer->getViewportWorkspace());
                });
            });

        registry.register_tool(
            McpTool{
                .name = "workspace_set_editor",
                .description = "Choose a workspace area's editor: viewport or an instantiable panel from ui/panels",
                .input_schema = {.type = "object",
                                 .properties = json{{"view_id", view_id_property()},
                                                    {"editor_id", json{{"type", "string"}}}},
                                 .required = {"view_id", "editor_id"}},
                .metadata = command_metadata()},
            [viewer](const json& args) -> json {
                auto id = view_id_arg(args);
                if (!id)
                    return json{{"error", id.error()}};
                if (!args.contains("editor_id") || !args["editor_id"].is_string())
                    return json{{"error", "Field 'editor_id' must be a string"}};
                const auto editor = args["editor_id"].get<std::string>();
                return post_and_wait(viewer, [viewer, id = *id, editor]() -> json {
                    if (editor != "viewport") {
                        const auto prototype = vis::gui::PanelRegistry::instance().get_panel_instance(editor);
                        if (!prototype || !prototype->supportsAreaInstances())
                            return json{{"error", "Unknown area editor: " + editor}};
                    }
                    auto* workspace = viewer->getViewportWorkspace();
                    if (!workspace)
                        return json{{"error", kWorkspaceUnavailable}};
                    if (auto result = workspace->setAreaEditor(id, editor); !result)
                        return json{{"error", std::string(result.error().user_message())}};
                    mark_workspace_dirty(*viewer);
                    return workspace_state_json(*viewer, *workspace);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "workspace_set_layout",
                .description = "Set the workspace layout preset while retaining surviving stable view IDs",
                .input_schema = {.type = "object",
                                 .properties = json{{"layout", json{{"type", "string"},
                                                                    {"enum", {"single", "horizontal", "vertical", "quad"}}}}},
                                 .required = {"layout"}},
                .metadata = command_metadata()},
            [viewer](const json& args) -> json {
                if (!args.contains("layout") || !args["layout"].is_string())
                    return json{{"error", "Field 'layout' must be a string"}};
                const auto layout = args["layout"].get<std::string>();
                if (layout != "single" && layout != "horizontal" && layout != "vertical" && layout != "quad")
                    return json{{"error", "Layout must be single, horizontal, vertical or quad"}};
                return mutate_workspace(viewer, [layout](vis::ViewportWorkspace& workspace) -> lfs::Status {
                    const auto preset = layout == "single"       ? vis::LayoutPreset::Single
                                        : layout == "horizontal" ? vis::LayoutPreset::DualHorizontal
                                        : layout == "vertical"   ? vis::LayoutPreset::DualVertical
                                                                 : vis::LayoutPreset::Quad;
                    auto result = workspace.setPreset(preset);
                    return result;
                });
            });

        registry.register_tool(
            McpTool{
                .name = "workspace_split",
                .description = "Split a stable viewport view and return the new view state",
                .input_schema = {.type = "object",
                                 .properties = json{{"view_id", view_id_property()},
                                                    {"axis", json{{"type", "string"}, {"enum", {"horizontal", "vertical"}}}},
                                                    {"ratio", json{{"type", "number"}, {"exclusiveMinimum", 0.0}, {"exclusiveMaximum", 1.0}}}},
                                 .required = {"view_id"}},
                .metadata = command_metadata()},
            [viewer](const json& args) -> json {
                auto id = view_id_arg(args);
                if (!id)
                    return json{{"error", id.error()}};
                const auto axis_name = args.value("axis", "horizontal");
                if (axis_name != "horizontal" && axis_name != "vertical")
                    return json{{"error", "Split axis must be horizontal or vertical"}};
                auto ratio = finite_float_arg(args, "ratio", 0.5f);
                if (!ratio)
                    return json{{"error", ratio.error()}};
                return post_and_wait(viewer, [viewer, id = *id, axis_name, ratio = *ratio]() -> json {
                    auto* const workspace = viewer->getViewportWorkspace();
                    if (!workspace)
                        return json{{"error", kWorkspaceUnavailable}};
                    auto result = workspace->split(id,
                                                   axis_name == "horizontal" ? vis::SplitAxis::Horizontal
                                                                             : vis::SplitAxis::Vertical,
                                                   ratio);
                    if (!result)
                        return json{{"error", std::string(result.error().user_message())}};
                    mark_workspace_dirty(*viewer);
                    auto state = workspace_state_json(*viewer, *workspace);
                    state["created_view_id"] = *result;
                    return state;
                });
            });

        registry.register_tool(
            McpTool{.name = "workspace_close", .description = "Close a stable viewport view", .input_schema = {.type = "object", .properties = json{{"view_id", view_id_property()}}, .required = {"view_id"}}, .metadata = command_metadata()},
            [viewer](const json& args) -> json {
                auto id = view_id_arg(args);
                if (!id)
                    return json{{"error", id.error()}};
                return mutate_workspace(viewer, [id = *id](vis::ViewportWorkspace& workspace) {
                    return workspace.close(id);
                });
            });

        registry.register_tool(
            McpTool{.name = "workspace_focus", .description = "Focus a stable viewport view", .input_schema = {.type = "object", .properties = json{{"view_id", view_id_property()}}, .required = {"view_id"}}, .metadata = command_metadata()},
            [viewer](const json& args) -> json {
                auto id = view_id_arg(args);
                if (!id)
                    return json{{"error", id.error()}};
                return mutate_workspace(viewer, [id = *id](vis::ViewportWorkspace& workspace) {
                    return workspace.focus(id);
                });
            });

        registry.register_tool(
            McpTool{.name = "workspace_maximize", .description = "Maximize a stable viewport view, or restore the layout", .input_schema = {.type = "object", .properties = json{{"view_id", json{{"type", json::array({"integer", "null"})}, {"minimum", 1}, {"description", "Stable view ID; omit or use null to restore"}}}}, .required = {}}, .metadata = command_metadata()},
            [viewer](const json& args) -> json {
                if (!args.contains("view_id") || args["view_id"].is_null())
                    return mutate_workspace(viewer, [](vis::ViewportWorkspace& workspace) { return workspace.restoreMaximized(); });
                auto id = view_id_arg(args);
                if (!id)
                    return json{{"error", id.error()}};
                return mutate_workspace(viewer, [id = *id](vis::ViewportWorkspace& workspace) { return workspace.maximize(id); });
            });

        registry.register_tool(
            McpTool{.name = "workspace_resize", .description = "Resize an existing workspace splitter", .input_schema = {.type = "object", .properties = json{{"split_id", json{{"type", "integer"}, {"minimum", 1}}}, {"ratio", json{{"type", "number"}, {"exclusiveMinimum", 0.0}, {"exclusiveMaximum", 1.0}}}}, .required = {"split_id", "ratio"}}, .metadata = command_metadata()},
            [viewer](const json& args) -> json {
                if (!args.contains("split_id") ||
                    ((!args["split_id"].is_number_integer() && !args["split_id"].is_number_unsigned()) ||
                     (args["split_id"].is_number_integer() && args["split_id"].get<std::int64_t>() <= 0) ||
                     (args["split_id"].is_number_unsigned() && args["split_id"].get<std::uint64_t>() == 0)))
                    return json{{"error", "Field 'split_id' must be a non-zero integer"}};
                auto ratio = finite_float_arg(args, "ratio");
                if (!ratio)
                    return json{{"error", ratio.error()}};
                const auto split = args["split_id"].get<vis::LayoutNodeId>();
                return mutate_workspace(viewer, [split, ratio = *ratio](vis::ViewportWorkspace& workspace) { return workspace.resize(split, ratio); });
            });

        registry.register_tool(
            McpTool{.name = "workspace_set_camera", .description = "Set a stable viewport camera by eye, target and up vectors", .input_schema = {.type = "object", .properties = json{{"view_id", view_id_property()}, {"eye", json{{"type", "array"}, {"minItems", 3}, {"maxItems", 3}}}, {"target", json{{"type", "array"}, {"minItems", 3}, {"maxItems", 3}}}, {"up", json{{"type", "array"}, {"minItems", 3}, {"maxItems", 3}}}}, .required = {"view_id", "eye", "target"}}, .metadata = command_metadata()},
            [viewer](const json& args) -> json {
                auto id = view_id_arg(args);
                if (!id)
                    return json{{"error", id.error()}};
                auto eye = vec3_arg(args, "eye");
                if (!eye)
                    return json{{"error", eye.error()}};
                auto target = vec3_arg(args, "target");
                if (!target)
                    return json{{"error", target.error()}};
                auto up = vec3_arg(args, "up", {0.0f, 1.0f, 0.0f});
                if (!up)
                    return json{{"error", up.error()}};
                return mutate_workspace(viewer, [id = *id, eye = *eye, target = *target, up = *up](vis::ViewportWorkspace& workspace) -> lfs::Status {
                    auto* record = workspace.findView(id);
                    if (!record)
                        return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{.code = lfs::ErrorCode::NotFound, .domain = lfs::ErrorDomain::App, .severity = lfs::Severity::Error, .retryability = lfs::Retryability::NotRetryable, .user_message = "Unknown viewport view ID", .detail = "Unknown viewport view ID", .detection = LFS_SOURCE_SITE_CURRENT()}));
                    const auto rotation = lfs::rendering::tryMakeVisualizerLookAtRotation(eye, target, up);
                    if (!rotation)
                        return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{.code = lfs::ErrorCode::InvalidArgument, .domain = lfs::ErrorDomain::App, .severity = lfs::Severity::Error, .retryability = lfs::Retryability::NotRetryable, .user_message = "Camera eye, target and up must define a finite nondegenerate view", .detail = "Camera eye, target and up must define a finite nondegenerate view", .detection = LFS_SOURCE_SITE_CURRENT()}));
                    vis::resetTransientNavigation(record->camera);
                    record->camera.setViewMatrix(*rotation, eye);
                    record->camera.camera.setPivot(target);
                    if (auto status = workspace.bumpViewState(id); !status)
                        return status;
                    return {};
                });
            });

        registry.register_tool(
            McpTool{.name = "workspace_set_projection", .description = "Set projection for a stable viewport view", .input_schema = {.type = "object", .properties = json{{"view_id", view_id_property()}, {"orthographic", json{{"type", "boolean"}}}, {"focal_length_mm", json{{"type", "number"}}}, {"ortho_scale", json{{"type", "number"}}}}, .required = {"view_id", "orthographic"}}, .metadata = command_metadata()},
            [viewer](const json& args) -> json {
                auto id = view_id_arg(args);
                if (!id)
                    return json{{"error", id.error()}};
                if (!args.contains("orthographic") || !args["orthographic"].is_boolean())
                    return json{{"error", "Field 'orthographic' must be a boolean"}};
                auto focal = finite_float_arg(args, "focal_length_mm");
                if (!focal)
                    return json{{"error", focal.error()}};
                auto ortho = finite_float_arg(args, "ortho_scale");
                if (!ortho)
                    return json{{"error", ortho.error()}};
                const bool orthographic = args["orthographic"].get<bool>();
                const bool has_focal = args.contains("focal_length_mm");
                const bool has_ortho = args.contains("ortho_scale");
                return mutate_workspace(viewer, [id = *id, orthographic, focal, ortho, has_focal, has_ortho](vis::ViewportWorkspace& workspace) -> lfs::Status {
                    const auto* record = workspace.findView(id);
                    if (!record)
                        return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{.code = lfs::ErrorCode::NotFound, .domain = lfs::ErrorDomain::App, .severity = lfs::Severity::Error, .retryability = lfs::Retryability::NotRetryable, .user_message = "Unknown viewport view ID", .detail = "Unknown viewport view ID", .detection = LFS_SOURCE_SITE_CURRENT()}));
                    auto projection = record->projection;
                    projection.orthographic = orthographic;
                    if (has_focal)
                        projection.focal_length_mm = *focal;
                    if (has_ortho)
                        projection.ortho_scale = *ortho;
                    auto result = workspace.setViewProjection(id, projection);
                    return result;
                });
            });
    }

    void register_gui_workspace_resources(mcp::ResourceRegistry& registry,
                                          vis::Visualizer* viewer) {
        registry.register_resource(
            mcp::McpResource{.uri = "lichtfeld://workspace/state", .name = "Workspace State", .description = "Current stable viewport views, layout, cameras and pane rectangles", .mime_type = "application/json"},
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer, [viewer, uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto* const workspace = viewer->getViewportWorkspace();
                    if (!workspace)
                        return std::unexpected(std::string(kWorkspaceUnavailable));
                    return single_json_resource(uri, workspace_state_json(*viewer, *workspace));
                });
            });
    }

} // namespace lfs::app
