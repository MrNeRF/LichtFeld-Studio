/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/mcp_screen_tools.hpp"
#include "app/mcp_app_utils.hpp"
#include "app/view_info_json.hpp"

#include "mcp/mcp_tools.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/render_constants.hpp"
#include "screen/screen.hpp"
#include "screen/view3d_space.hpp"
#include "visualizer/gui/gui_manager.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/visualizer_impl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::app {

    namespace {

        using json = nlohmann::json;
        using mcp::McpResource;
        using mcp::McpResourceContent;
        using mcp::McpTool;
        using mcp::ResourceRegistry;
        using mcp::ToolRegistry;

        vis::screen::AreaId area_id(const json& args, const char* key) {
            const int value = args.value(key, 0);
            return vis::screen::AreaId{value > 0 ? static_cast<std::uint32_t>(value) : 0};
        }

        json area_json(vis::VisualizerImpl& impl, const vis::screen::AreaId id) {
            const auto& screen = impl.screens().screen();
            const auto* area = screen.area(id);
            const auto rect = impl.areaRect(id);
            const bool is_view = area && area->editor == vis::screen::editors::kView3D;
            return json{
                {"id", id.value},
                {"editor", area ? area->editor : std::string{}},
                {"x", rect.x},
                {"y", rect.y},
                {"width", rect.w},
                {"height", rect.h},
                {"is_view", is_view},
                {"active_view", is_view && screen.activeView() == id},
                {"maximized", screen.maximized() == id},
            };
        }

        json screen_state_json(vis::VisualizerImpl& impl) {
            json areas = json::array();
            json editors = json::array();
            impl.screens().read([&](const vis::screen::Screen& screen) {
                for (const auto id : screen.areas())
                    areas.push_back(area_json(impl, id));
                for (const auto& type : screen.registry().list()) {
                    editors.push_back(json{
                        {"id", type.id},
                        {"label", type.label},
                        {"multi_instance", type.multi_instance},
                    });
                }
            });
            return json{
                {"success", true},
                {"active_view", impl.screens().screen().activeView().value},
                {"maximized", impl.screens().screen().maximized().value},
                {"areas", std::move(areas)},
                {"editors", std::move(editors)},
            };
        }

        void notify_screen_changed(vis::VisualizerImpl* impl) {
            if (auto* rendering = impl->getRenderingManager())
                rendering->markDirty(vis::DirtyFlag::ALL);
        }

        vis::ViewInfo view_info_from_space(const vis::screen::View3DSpace& space) {
            vis::ViewInfo info;
            const auto& cam = space.camera.camera;
            for (int c = 0; c < 3; ++c)
                for (int r = 0; r < 3; ++r)
                    info.rotation[static_cast<std::size_t>(c * 3 + r)] = cam.R[c][r];
            info.translation = {cam.t.x, cam.t.y, cam.t.z};
            info.pivot = {cam.pivot.x, cam.pivot.y, cam.pivot.z};
            info.width = space.camera.windowSize.x;
            info.height = space.camera.windowSize.y;
            info.fov = lfs::rendering::focalLengthToVFov(space.settings.focal_length_mm);
            info.orthographic = space.settings.orthographic;
            info.ortho_scale = space.settings.ortho_scale;
            return info;
        }

        vis::screen::AreaId resolve_view(vis::VisualizerImpl& impl, const json& args) {
            if (args.contains("view") && !args["view"].is_null())
                return area_id(args, "view");
            return impl.screens().screen().activeView();
        }

        std::expected<glm::vec3, std::string> vec3_arg(const json& args, const char* key) {
            if (!args.contains(key) || !args[key].is_array() || args[key].size() != 3)
                return std::unexpected(std::string("must be a 3-element number array"));
            const glm::vec3 result(args[key][0].get<float>(), args[key][1].get<float>(), args[key][2].get<float>());
            if (!std::isfinite(result.x) || !std::isfinite(result.y) || !std::isfinite(result.z))
                return std::unexpected(std::string("must contain finite numbers"));
            return result;
        }

    } // namespace

    void register_gui_screen_tools(ToolRegistry& registry, vis::Visualizer* viewer) {
        auto* const impl = dynamic_cast<vis::VisualizerImpl*>(viewer);
        if (!impl)
            return;

        const auto area_schema = json{{"type", "integer"}, {"description", "Area id"}};
        const auto view_schema = json{{"type", "integer"}, {"description", "3D view area id"}};
        const auto editor_schema = json{{"type", "string"}, {"description", "Editor type id"}};

        registry.register_tool(
            McpTool{
                .name = "screen.get",
                .description = "Read the current editor screen: areas, editors, and the active 3D view",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "query",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [impl](const json&) -> json {
                return post_and_wait(impl, [impl]() { return screen_state_json(*impl); });
            });

        registry.register_tool(
            McpTool{
                .name = "screen.split",
                .description = "Split an area vertically (side by side) or horizontally (stacked)",
                .input_schema =
                    {.type = "object",
                     .properties = json{
                         {"area", area_schema},
                         {"direction", json{{"type", "string"},
                                            {"enum", json::array({"vertical", "horizontal"})},
                                            {"description", "vertical = columns, horizontal = rows"}}},
                         {"factor", json{{"type", "number"}, {"description", "Size of the new half in (0, 1)"}}}},
                     .required = {"area"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [impl](const json& args) -> json {
                const auto direction = args.value("direction", std::string("vertical"));
                if (direction != "vertical" && direction != "horizontal")
                    return mcp::invalid_argument_result("direction must be 'vertical' or 'horizontal'", "direction");
                const float factor = args.value("factor", 0.5f);
                if (!(factor > 0.0f && factor < 1.0f) || !std::isfinite(factor))
                    return mcp::invalid_argument_result("factor must be in (0, 1)", "factor");
                const auto axis = direction == "vertical" ? vis::screen::SplitAxis::Columns
                                                          : vis::screen::SplitAxis::Rows;
                const auto id = area_id(args, "area");
                return post_and_wait(impl, [impl, id, axis, factor]() -> json {
                    vis::screen::AreaId added;
                    impl->screens().edit([&](vis::screen::Screen& s) { added = s.split(id, axis, factor); });
                    if (!added.valid())
                        return json{{"error", "Split failed"}};
                    notify_screen_changed(impl);
                    auto result = screen_state_json(*impl);
                    result["area"] = added.value;
                    return result;
                });
            });

        registry.register_tool(
            McpTool{
                .name = "screen.join",
                .description = "Join two neighbouring areas, keeping `keep` and absorbing `remove`",
                .input_schema = {.type = "object",
                                 .properties = json{{"keep", area_schema}, {"remove", area_schema}},
                                 .required = {"keep", "remove"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [impl](const json& args) -> json {
                const auto keep = area_id(args, "keep");
                const auto remove = area_id(args, "remove");
                return post_and_wait(impl, [impl, keep, remove]() -> json {
                    bool done = false;
                    impl->screens().edit([&](vis::screen::Screen& s) { done = s.join(keep, remove); });
                    if (!done)
                        return json{{"error", "Join failed"}};
                    notify_screen_changed(impl);
                    return screen_state_json(*impl);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "screen.close",
                .description = "Close an area, keeping at least one 3D view",
                .input_schema = {.type = "object", .properties = json{{"area", area_schema}}, .required = {"area"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [impl](const json& args) -> json {
                const auto id = area_id(args, "area");
                return post_and_wait(impl, [impl, id]() -> json {
                    bool done = false;
                    impl->screens().edit([&](vis::screen::Screen& s) { done = s.close(id); });
                    if (!done)
                        return json{{"error", "Close failed"}};
                    notify_screen_changed(impl);
                    return screen_state_json(*impl);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "screen.swap",
                .description = "Swap the editors of two areas",
                .input_schema = {.type = "object",
                                 .properties = json{{"a", area_schema}, {"b", area_schema}},
                                 .required = {"a", "b"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [impl](const json& args) -> json {
                const auto a = area_id(args, "a");
                const auto b = area_id(args, "b");
                return post_and_wait(impl, [impl, a, b]() -> json {
                    bool done = false;
                    impl->screens().edit([&](vis::screen::Screen& s) { done = s.swap(a, b); });
                    if (!done)
                        return json{{"error", "Swap failed"}};
                    notify_screen_changed(impl);
                    return screen_state_json(*impl);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "screen.set_editor",
                .description = "Show an editor in an area",
                .input_schema = {.type = "object",
                                 .properties = json{{"area", area_schema}, {"editor", editor_schema}},
                                 .required = {"area", "editor"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [impl](const json& args) -> json {
                const auto id = area_id(args, "area");
                const auto editor = args.value("editor", std::string{});
                return post_and_wait(impl, [impl, id, editor]() -> json {
                    bool done = false;
                    impl->screens().edit([&](vis::screen::Screen& s) { done = s.setEditor(id, editor); });
                    if (!done)
                        return json{{"error", "set_editor failed"}};
                    notify_screen_changed(impl);
                    return screen_state_json(*impl);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "screen.maximize",
                .description = "Maximize an area, or restore it if it is already maximized",
                .input_schema = {.type = "object", .properties = json{{"area", area_schema}}, .required = {"area"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [impl](const json& args) -> json {
                const auto id = area_id(args, "area");
                return post_and_wait(impl, [impl, id]() -> json {
                    bool done = false;
                    impl->screens().edit([&](vis::screen::Screen& s) { done = s.toggleMaximized(id); });
                    if (!done)
                        return json{{"error", "maximize failed"}};
                    notify_screen_changed(impl);
                    return screen_state_json(*impl);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "screen.reset",
                .description = "Reset the screen to the default layout",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [impl](const json&) -> json {
                return post_and_wait(impl, [impl]() -> json {
                    impl->screens().resetToDefault();
                    notify_screen_changed(impl);
                    return screen_state_json(*impl);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "view.command",
                .description = "Run a view command on a 3D view (display, axis, overlay, frame_all, area:quad, ...)",
                .input_schema = {.type = "object",
                                 .properties = json{{"view", view_schema},
                                                    {"command", json{{"type", "string"},
                                                                     {"description", "View command string"}}}},
                                 .required = {"command"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [impl](const json& args) -> json {
                const auto command = args.value("command", std::string{});
                return post_and_wait(impl, [impl, args, command]() -> json {
                    const auto view = resolve_view(*impl, args);
                    if (!impl->runViewCommand(view.value, command))
                        return json{{"error", "Unknown or failed view command"}};
                    return json{{"success", true}, {"view", view.value}, {"command", command}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "view.get_camera",
                .description = "Get a 3D view's camera. Defaults to the active view.",
                .input_schema = {.type = "object", .properties = json{{"view", view_schema}}, .required = {}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "query",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [impl](const json& args) -> json {
                return post_and_wait(impl, [impl, args]() -> json {
                    const auto view = resolve_view(*impl, args);
                    const auto* space = impl->screens().view3D(view);
                    if (!space)
                        return json{{"error", "Not a 3D view"}};
                    auto result = view_info_json(view_info_from_space(*space));
                    result["view"] = view.value;
                    return result;
                });
            });

        registry.register_tool(
            McpTool{
                .name = "view.set_camera",
                .description = "Set a 3D view's camera by eye/target/up. Defaults to the active view.",
                .input_schema =
                    {.type = "object",
                     .properties = json{
                         {"view", view_schema},
                         {"eye", number_array_schema(3, "Camera eye position [x,y,z]")},
                         {"target", number_array_schema(3, "Camera target/pivot position [x,y,z]")},
                         {"up", number_array_schema(3, "Optional up vector [x,y,z], defaults to [0,1,0]")}},
                     .required = {"eye", "target"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [impl](const json& args) -> json {
                auto eye = vec3_arg(args, "eye");
                if (!eye)
                    return mcp::invalid_argument_result(eye.error(), "eye");
                auto target = vec3_arg(args, "target");
                if (!target)
                    return mcp::invalid_argument_result(target.error(), "target");
                glm::vec3 up(0.0f, 1.0f, 0.0f);
                if (args.contains("up") && !args["up"].is_null()) {
                    auto parsed_up = vec3_arg(args, "up");
                    if (!parsed_up)
                        return mcp::invalid_argument_result(parsed_up.error(), "up");
                    up = *parsed_up;
                }
                return post_and_wait(impl, [impl, args, eye = *eye, target = *target, up]() -> json {
                    const auto id = resolve_view(*impl, args);
                    auto* space = impl->screens().view3D(id);
                    if (!space)
                        return json{{"error", "Not a 3D view"}};
                    const auto rotation = lfs::rendering::tryMakeVisualizerLookAtRotation(eye, target, up);
                    if (!rotation)
                        return json{{"error", "eye, target and up must form a valid look-at"}};
                    space->camera.setViewMatrix(*rotation, eye);
                    space->camera.camera.setPivot(target);
                    notify_screen_changed(impl);
                    auto result = view_info_json(view_info_from_space(*space));
                    result["view"] = id.value;
                    return result;
                });
            });

        registry.register_tool(
            McpTool{
                .name = "view.get_settings",
                .description = "Get a 3D view's ViewSettings. Defaults to the active view.",
                .input_schema = {.type = "object", .properties = json{{"view", view_schema}}, .required = {}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "query",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [impl](const json& args) -> json {
                return post_and_wait(impl, [impl, args]() -> json {
                    const auto view = resolve_view(*impl, args);
                    const auto* space = impl->screens().view3D(view);
                    if (!space)
                        return json{{"error", "Not a 3D view"}};
                    return json{{"success", true},
                                {"view", view.value},
                                {"settings", vis::screen::viewSettingsToJson(space->settings)}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "view.set_settings",
                .description = "Update ViewSettings fields on a 3D view. Unknown fields are rejected.",
                .input_schema = {.type = "object",
                                 .properties = json{{"view", view_schema},
                                                    {"settings", json{{"type", "object"},
                                                                      {"description", "ViewSettings fields to patch"}}}},
                                 .required = {"settings"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [impl](const json& args) -> json {
                if (!args.contains("settings") || !args["settings"].is_object())
                    return mcp::invalid_argument_result("settings must be an object", "settings");
                const auto& patch = args["settings"];
                const auto known = vis::screen::viewSettingsToJson(vis::ViewSettings{});
                for (const auto& [key, value] : patch.items()) {
                    if (!known.contains(key))
                        return mcp::invalid_argument_result("Unknown view setting: " + key, "settings");
                }
                return post_and_wait(impl, [impl, args, patch]() -> json {
                    const auto view = resolve_view(*impl, args);
                    auto* space = impl->screens().view3D(view);
                    if (!space)
                        return json{{"error", "Not a 3D view"}};
                    auto restored = vis::screen::viewSettingsFromJson(patch, space->settings);
                    if (!restored)
                        return json{{"error", "Invalid view settings"}};
                    space->settings = *restored;
                    notify_screen_changed(impl);
                    return json{{"success", true},
                                {"view", view.value},
                                {"settings", vis::screen::viewSettingsToJson(space->settings)}};
                });
            });
    }

    void register_gui_screen_resources(ResourceRegistry& registry, vis::Visualizer* viewer) {
        auto* const impl = dynamic_cast<vis::VisualizerImpl*>(viewer);
        if (!impl)
            return;

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://ui/screen",
                .name = "Editor Screen",
                .description = "Current editor areas, editors, and the active 3D view",
                .mime_type = "application/json"},
            [impl](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(impl, [impl, uri]() {
                    return single_json_resource(uri, screen_state_json(*impl));
                });
            });
    }

} // namespace lfs::app
