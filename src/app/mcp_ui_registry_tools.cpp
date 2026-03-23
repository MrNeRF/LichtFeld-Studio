/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/mcp_ui_registry_tools.hpp"
#include "app/mcp_app_utils.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "python/python_runtime.hpp"
#include "visualizer/core/editor_context.hpp"
#include "visualizer/gui/panel_registry.hpp"
#include "visualizer/gui/rml_menu_bar.hpp"
#include "visualizer/operator/operator_properties.hpp"
#include "visualizer/operator/operator_registry.hpp"
#include "visualizer/tools/unified_tool_registry.hpp"
#include "visualizer/visualizer.hpp"
#include "visualizer/visualizer_impl.hpp"

#include <array>
#include <atomic>
#include <cctype>
#include <expected>
#include <future>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace lfs::app {

    namespace {

        using json = nlohmann::json;
        using mcp::McpResourceContent;

        constexpr std::array kPanelSpaces = {
            vis::gui::PanelSpace::SidePanel,
            vis::gui::PanelSpace::Floating,
            vis::gui::PanelSpace::ViewportOverlay,
            vis::gui::PanelSpace::MainPanelTab,
            vis::gui::PanelSpace::SceneHeader,
            vis::gui::PanelSpace::StatusBar,
        };

        constexpr std::array kSelectionSubmodes = {
            std::pair<std::string_view, int>{"centers", 0},
            std::pair<std::string_view, int>{"rectangle", 1},
            std::pair<std::string_view, int>{"polygon", 2},
            std::pair<std::string_view, int>{"lasso", 3},
            std::pair<std::string_view, int>{"rings", 4},
        };

        struct CollectedMenuItem {
            int type = static_cast<int>(vis::gui::MenuItemDesc::Type::Separator);
            std::string label;
            std::string display_label;
            std::string operator_id;
            std::string shortcut;
            bool enabled = true;
            bool selected = false;
            int callback_index = -1;
        };

        struct MenuActionDescriptor {
            int action_index = -1;
            std::string type;
            std::string label;
            std::string display_label;
            std::vector<std::string> path;
            std::vector<std::string> display_path;
            std::string operator_id;
            std::string shortcut;
            bool enabled = true;
            bool selected = false;
            int callback_index = -1;
        };

        struct BuiltMenuContent {
            json items = json::array();
            json actions = json::array();
            std::vector<MenuActionDescriptor> action_descriptors;
            size_t flat_item_count = 0;
        };

        const char* tool_source_to_string(const vis::ToolSource source) {
            switch (source) {
            case vis::ToolSource::CPP:
                return "cpp";
            case vis::ToolSource::PYTHON:
                return "python";
            }
            return "unknown";
        }

        const char* panel_space_to_string(const vis::gui::PanelSpace space) {
            switch (space) {
            case vis::gui::PanelSpace::SidePanel:
                return "side_panel";
            case vis::gui::PanelSpace::Floating:
                return "floating";
            case vis::gui::PanelSpace::ViewportOverlay:
                return "viewport_overlay";
            case vis::gui::PanelSpace::MainPanelTab:
                return "main_panel_tab";
            case vis::gui::PanelSpace::SceneHeader:
                return "scene_header";
            case vis::gui::PanelSpace::StatusBar:
                return "status_bar";
            }
            return "unknown";
        }

        std::optional<vis::gui::PanelSpace> panel_space_from_string(const std::string_view value) {
            if (value == "side_panel")
                return vis::gui::PanelSpace::SidePanel;
            if (value == "floating")
                return vis::gui::PanelSpace::Floating;
            if (value == "viewport_overlay")
                return vis::gui::PanelSpace::ViewportOverlay;
            if (value == "main_panel_tab")
                return vis::gui::PanelSpace::MainPanelTab;
            if (value == "scene_header")
                return vis::gui::PanelSpace::SceneHeader;
            if (value == "status_bar")
                return vis::gui::PanelSpace::StatusBar;
            return std::nullopt;
        }

        const char* builtin_tool_id(const vis::ToolType tool) {
            switch (tool) {
            case vis::ToolType::None:
                return "";
            case vis::ToolType::Selection:
                return "builtin.select";
            case vis::ToolType::Translate:
                return "builtin.translate";
            case vis::ToolType::Rotate:
                return "builtin.rotate";
            case vis::ToolType::Scale:
                return "builtin.scale";
            case vis::ToolType::Mirror:
                return "builtin.mirror";
            case vis::ToolType::Brush:
                return "builtin.brush";
            case vis::ToolType::Align:
                return "builtin.align";
            }
            return "";
        }

        bool looks_like_localization_key(const std::string_view value) {
            if (value.empty() || value.find('.') == std::string_view::npos)
                return false;

            bool has_dot = false;
            bool segment_has_content = false;
            for (const unsigned char ch : value) {
                if (ch == '.') {
                    if (!segment_has_content)
                        return false;
                    has_dot = true;
                    segment_has_content = false;
                    continue;
                }

                if (!(std::islower(ch) || std::isdigit(ch) || ch == '_' || ch == '-'))
                    return false;

                segment_has_content = true;
            }

            return has_dot && segment_has_content;
        }

        vis::VisualizerImpl* as_visualizer_impl(vis::Visualizer* viewer);

        std::string effective_active_tool_id(vis::Visualizer* viewer) {
            const std::string registry_active_tool_id = vis::UnifiedToolRegistry::instance().getActiveTool();
            auto* const impl = as_visualizer_impl(viewer);
            if (!impl)
                return registry_active_tool_id;

            const auto& editor = impl->getEditorContext();
            if (editor.hasActiveOperator())
                return editor.getActiveOperator();

            const std::string editor_tool_id = builtin_tool_id(editor.getActiveTool());
            if (!editor_tool_id.empty())
                return editor_tool_id;

            return registry_active_tool_id;
        }

        std::string display_menu_label(const std::string_view value) {
            if (value.empty()) {
                return {};
            }

            if (!looks_like_localization_key(value)) {
                return std::string(value);
            }

            const std::string key(value);
            return LOC(key.c_str());
        }

        const char* menu_item_type_to_string(const int type) {
            switch (static_cast<vis::gui::MenuItemDesc::Type>(type)) {
            case vis::gui::MenuItemDesc::Type::Operator:
                return "operator";
            case vis::gui::MenuItemDesc::Type::Separator:
                return "separator";
            case vis::gui::MenuItemDesc::Type::SubMenuBegin:
                return "submenu";
            case vis::gui::MenuItemDesc::Type::SubMenuEnd:
                return "submenu_end";
            case vis::gui::MenuItemDesc::Type::Toggle:
                return "toggle";
            case vis::gui::MenuItemDesc::Type::ShortcutItem:
                return "shortcut_item";
            case vis::gui::MenuItemDesc::Type::Item:
                return "item";
            }
            return "unknown";
        }

        std::vector<python::MenuBarEntry> menu_bar_entries() {
            return python::get_menu_bar_entries();
        }

        std::expected<python::MenuBarEntry, std::string> find_menu_bar_entry(const std::string& menu_id) {
            for (const auto& entry : menu_bar_entries()) {
                if (entry.idname == menu_id) {
                    return entry;
                }
            }
            return std::unexpected("Menu is not registered: " + menu_id);
        }

        std::vector<CollectedMenuItem> collect_menu_items(const std::string& menu_id) {
            std::vector<CollectedMenuItem> items;
            python::collect_menu_content(
                menu_id,
                [](const python::MenuItemInfo* item, void* user_data) {
                    if (!item || !user_data) {
                        return;
                    }

                    auto* output = static_cast<std::vector<CollectedMenuItem>*>(user_data);
                    CollectedMenuItem collected;
                    collected.type = item->type;
                    collected.label = item->label ? item->label : "";
                    collected.display_label = display_menu_label(collected.label);
                    collected.operator_id = item->operator_id ? item->operator_id : "";
                    collected.shortcut = item->shortcut ? item->shortcut : "";
                    collected.enabled = item->enabled;
                    collected.selected = item->selected;
                    collected.callback_index = item->callback_index;
                    output->push_back(std::move(collected));
                },
                &items);
            return items;
        }

        json menu_action_json(const MenuActionDescriptor& action) {
            json payload{
                {"action_index", action.action_index},
                {"type", action.type},
                {"label", action.label},
                {"display_label", action.display_label},
                {"path", action.path},
                {"display_path", action.display_path},
                {"enabled", action.enabled},
                {"selected", action.selected},
            };

            if (!action.operator_id.empty()) {
                payload["operator_id"] = action.operator_id;
            }
            if (!action.shortcut.empty()) {
                payload["shortcut"] = action.shortcut;
            }
            if (action.callback_index >= 0) {
                payload["callback_index"] = action.callback_index;
            }

            return payload;
        }

        json build_menu_tree(const std::vector<CollectedMenuItem>& items,
                             size_t& cursor,
                             std::vector<std::string>& path,
                             std::vector<std::string>& display_path,
                             std::vector<MenuActionDescriptor>& actions) {
            json nodes = json::array();

            while (cursor < items.size()) {
                const auto& item = items[cursor++];
                switch (static_cast<vis::gui::MenuItemDesc::Type>(item.type)) {
                case vis::gui::MenuItemDesc::Type::Separator:
                    nodes.push_back(json{{"type", "separator"}});
                    break;
                case vis::gui::MenuItemDesc::Type::SubMenuEnd:
                    return nodes;
                case vis::gui::MenuItemDesc::Type::SubMenuBegin: {
                    path.push_back(item.label);
                    display_path.push_back(item.display_label);
                    auto children = build_menu_tree(items, cursor, path, display_path, actions);
                    nodes.push_back(json{
                        {"type", "submenu"},
                        {"label", item.label},
                        {"display_label", item.display_label},
                        {"path", path},
                        {"display_path", display_path},
                        {"items", std::move(children)},
                    });
                    path.pop_back();
                    display_path.pop_back();
                    break;
                }
                case vis::gui::MenuItemDesc::Type::Operator:
                case vis::gui::MenuItemDesc::Type::Toggle:
                case vis::gui::MenuItemDesc::Type::ShortcutItem:
                case vis::gui::MenuItemDesc::Type::Item: {
                    MenuActionDescriptor action;
                    action.action_index = static_cast<int>(actions.size());
                    action.type = menu_item_type_to_string(item.type);
                    action.label = item.label;
                    action.display_label = item.display_label;
                    action.path = path;
                    action.path.push_back(item.label);
                    action.display_path = display_path;
                    action.display_path.push_back(item.display_label);
                    action.operator_id = item.operator_id;
                    action.shortcut = item.shortcut;
                    action.enabled = item.enabled;
                    action.selected = item.selected;
                    action.callback_index = item.callback_index;
                    actions.push_back(action);

                    nodes.push_back(menu_action_json(action));
                    break;
                }
                }
            }

            return nodes;
        }

        BuiltMenuContent build_menu_content(const std::string& menu_id) {
            auto items = collect_menu_items(menu_id);

            BuiltMenuContent content;
            content.flat_item_count = items.size();

            std::vector<std::string> path;
            std::vector<std::string> display_path;
            size_t cursor = 0;
            content.items = build_menu_tree(items, cursor, path, display_path, content.action_descriptors);

            for (const auto& action : content.action_descriptors) {
                content.actions.push_back(menu_action_json(action));
            }

            return content;
        }

        json menu_descriptor_json(const python::MenuBarEntry& entry,
                                  const BuiltMenuContent& content,
                                  const bool include_items) {
            json payload{
                {"id", entry.idname},
                {"label", entry.label},
                {"display_label", display_menu_label(entry.label)},
                {"order", entry.order},
                {"flat_item_count", content.flat_item_count},
                {"action_count", content.action_descriptors.size()},
            };

            if (include_items) {
                payload["items"] = content.items;
                payload["actions"] = content.actions;
            }

            return payload;
        }

        std::expected<std::vector<std::string>, std::string> parse_menu_path(const json& value) {
            if (!value.is_array()) {
                return std::unexpected("Field 'path' must be an array of labels");
            }

            std::vector<std::string> path;
            path.reserve(value.size());
            for (const auto& segment : value) {
                if (!segment.is_string()) {
                    return std::unexpected("Field 'path' must contain only strings");
                }
                path.push_back(segment.get<std::string>());
            }

            if (path.empty()) {
                return std::unexpected("Field 'path' must contain at least one label");
            }

            return path;
        }

        std::expected<MenuActionDescriptor, std::string> resolve_menu_action(
            const std::vector<MenuActionDescriptor>& actions,
            const json& args) {
            const bool has_action_index =
                args.contains("action_index") && !args["action_index"].is_null();
            const bool has_callback_index =
                args.contains("callback_index") && !args["callback_index"].is_null();
            const bool has_operator_id =
                args.contains("operator_id") && !args["operator_id"].is_null();
            const bool has_path = args.contains("path") && !args["path"].is_null();

            const int selector_count = static_cast<int>(has_action_index) +
                                       static_cast<int>(has_callback_index) +
                                       static_cast<int>(has_operator_id) +
                                       static_cast<int>(has_path);
            if (selector_count != 1) {
                return std::unexpected(
                    "Specify exactly one of 'action_index', 'callback_index', 'operator_id', or 'path'");
            }

            if (has_action_index) {
                if (!args["action_index"].is_number_integer()) {
                    return std::unexpected("Field 'action_index' must be an integer");
                }
                const int action_index = args["action_index"].get<int>();
                if (action_index < 0 || action_index >= static_cast<int>(actions.size())) {
                    return std::unexpected("Menu action index is out of range");
                }
                return actions[static_cast<size_t>(action_index)];
            }

            if (has_callback_index) {
                if (!args["callback_index"].is_number_integer()) {
                    return std::unexpected("Field 'callback_index' must be an integer");
                }
                const int callback_index = args["callback_index"].get<int>();
                for (const auto& action : actions) {
                    if (action.callback_index == callback_index) {
                        return action;
                    }
                }
                return std::unexpected("Menu callback index is not available: " +
                                       std::to_string(callback_index));
            }

            if (has_operator_id) {
                if (!args["operator_id"].is_string()) {
                    return std::unexpected("Field 'operator_id' must be a string");
                }

                const std::string operator_id = args["operator_id"].get<std::string>();
                const MenuActionDescriptor* match = nullptr;
                for (const auto& action : actions) {
                    if (action.operator_id != operator_id) {
                        continue;
                    }
                    if (match) {
                        return std::unexpected(
                            "Menu operator id is ambiguous within this menu; use 'action_index' or 'path'");
                    }
                    match = &action;
                }

                if (!match) {
                    return std::unexpected("Menu operator id is not available in this menu: " +
                                           operator_id);
                }

                return *match;
            }

            auto requested_path = parse_menu_path(args["path"]);
            if (!requested_path) {
                return std::unexpected(requested_path.error());
            }

            const MenuActionDescriptor* match = nullptr;
            for (const auto& action : actions) {
                if (action.path != *requested_path && action.display_path != *requested_path) {
                    continue;
                }
                if (match) {
                    return std::unexpected(
                        "Menu path is ambiguous; use 'action_index' for deterministic invocation");
                }
                match = &action;
            }

            if (!match) {
                return std::unexpected("Menu path is not available in this menu");
            }

            return *match;
        }

        json panel_options_json(const uint32_t options) {
            json values = json::array();
            if ((options & static_cast<uint32_t>(vis::gui::PanelOption::DEFAULT_CLOSED)) != 0) {
                values.push_back("default_closed");
            }
            if ((options & static_cast<uint32_t>(vis::gui::PanelOption::HIDE_HEADER)) != 0) {
                values.push_back("hide_header");
            }
            if ((options & static_cast<uint32_t>(vis::gui::PanelOption::SELF_MANAGED)) != 0) {
                values.push_back("self_managed");
            }
            return values;
        }

        json poll_dependencies_json(const vis::gui::PollDependency deps) {
            json values = json::array();
            if ((deps & vis::gui::PollDependency::SELECTION) != vis::gui::PollDependency::NONE) {
                values.push_back("selection");
            }
            if ((deps & vis::gui::PollDependency::TRAINING) != vis::gui::PollDependency::NONE) {
                values.push_back("training");
            }
            if ((deps & vis::gui::PollDependency::SCENE) != vis::gui::PollDependency::NONE) {
                values.push_back("scene");
            }
            return values;
        }

        vis::VisualizerImpl* as_visualizer_impl(vis::Visualizer* viewer) {
            return dynamic_cast<vis::VisualizerImpl*>(viewer);
        }

        json current_ui_state_json(vis::Visualizer* viewer) {
            const std::string active_tool_id = vis::UnifiedToolRegistry::instance().getActiveTool();
            const std::string active_submode_id = vis::UnifiedToolRegistry::instance().getActiveSubmode();
            const auto menus = menu_bar_entries();

            json panel_counts = json::object();
            size_t panel_count = 0;
            auto& panels = vis::gui::PanelRegistry::instance();
            for (const auto space : kPanelSpaces) {
                const auto summaries = panels.get_panels_for_space(space);
                panel_counts[panel_space_to_string(space)] = summaries.size();
                panel_count += summaries.size();
            }

            json payload{
                {"registry_active_tool_id", active_tool_id},
                {"effective_active_tool_id", effective_active_tool_id(viewer)},
                {"active_submode_id", active_submode_id},
                {"panel_count", panel_count},
                {"panel_counts", std::move(panel_counts)},
                {"menu_count", menus.size()},
            };

            if (auto* impl = as_visualizer_impl(viewer)) {
                const auto& editor = impl->getEditorContext();
                payload["editor_active_tool_id"] = builtin_tool_id(editor.getActiveTool());
                payload["active_operator_id"] = editor.getActiveOperator();
                payload["gizmo_type"] = editor.getGizmoType();
                payload["has_active_operator"] = editor.hasActiveOperator();
            }

            return payload;
        }

        json tool_descriptor_json(const vis::ToolDescriptor& tool,
                                  const std::string& active_tool_id,
                                  const std::string& active_submode_id,
                                  const bool include_poll) {
            json submodes = json::array();
            for (const auto& submode : tool.submodes) {
                submodes.push_back(json{
                    {"id", submode.id},
                    {"label", submode.label},
                    {"icon", submode.icon},
                    {"active", submode.id == active_submode_id},
                });
            }

            json payload{
                {"id", tool.id},
                {"label", tool.label},
                {"icon", tool.icon},
                {"shortcut", tool.shortcut},
                {"group", tool.group},
                {"order", tool.order},
                {"source", tool_source_to_string(tool.source)},
                {"active", tool.id == active_tool_id},
                {"submodes", std::move(submodes)},
            };

            if (include_poll) {
                payload["poll"] = !tool.poll_fn || tool.poll_fn();
            }

            return payload;
        }

        json panel_summary_json(const vis::gui::PanelSummary& panel) {
            return json{
                {"id", panel.id},
                {"label", panel.label},
                {"space", panel_space_to_string(panel.space)},
                {"order", panel.order},
                {"enabled", panel.enabled},
            };
        }

        json panel_details_json(const vis::gui::PanelDetails& panel) {
            return json{
                {"id", panel.id},
                {"label", panel.label},
                {"parent_id", panel.parent_id},
                {"space", panel_space_to_string(panel.space)},
                {"order", panel.order},
                {"enabled", panel.enabled},
                {"options", panel_options_json(panel.options)},
                {"poll_dependencies", poll_dependencies_json(panel.poll_dependencies)},
                {"is_native", panel.is_native},
                {"initial_width", panel.initial_width},
                {"initial_height", panel.initial_height},
            };
        }

        std::expected<json, std::string> list_tools_payload(vis::Visualizer* viewer,
                                                            const bool include_poll) {
            const std::string active_tool_id = effective_active_tool_id(viewer);
            const std::string active_submode_id = vis::UnifiedToolRegistry::instance().getActiveSubmode();

            json tools = json::array();
            for (const auto* tool : vis::UnifiedToolRegistry::instance().getAllTools()) {
                if (!tool) {
                    continue;
                }
                tools.push_back(tool_descriptor_json(*tool, active_tool_id, active_submode_id, include_poll));
            }

            return json{
                {"count", static_cast<int64_t>(tools.size())},
                {"tools", std::move(tools)},
                {"state", current_ui_state_json(viewer)},
            };
        }

        std::expected<json, std::string> describe_tool_payload(vis::Visualizer* viewer,
                                                               const std::string& tool_id,
                                                               const bool include_poll) {
            const std::string active_tool_id = effective_active_tool_id(viewer);
            const std::string active_submode_id = vis::UnifiedToolRegistry::instance().getActiveSubmode();

            for (const auto* tool : vis::UnifiedToolRegistry::instance().getAllTools()) {
                if (!tool || tool->id != tool_id) {
                    continue;
                }

                return json{
                    {"tool", tool_descriptor_json(*tool, active_tool_id, active_submode_id, include_poll)},
                    {"state", current_ui_state_json(viewer)},
                };
            }

            return std::unexpected("Tool is not registered: " + tool_id);
        }

        std::expected<json, std::string> list_panels_payload() {
            json panels = json::array();
            auto& registry = vis::gui::PanelRegistry::instance();

            for (const auto space : kPanelSpaces) {
                for (const auto& panel : registry.get_panels_for_space(space)) {
                    if (auto details = registry.get_panel(panel.id)) {
                        panels.push_back(panel_details_json(*details));
                    } else {
                        panels.push_back(panel_summary_json(panel));
                    }
                }
            }

            return json{
                {"count", static_cast<int64_t>(panels.size())},
                {"panels", std::move(panels)},
            };
        }

        std::expected<json, std::string> describe_panel_payload(const std::string& panel_id) {
            auto details = vis::gui::PanelRegistry::instance().get_panel(panel_id);
            if (!details) {
                return std::unexpected("Panel is not registered: " + panel_id);
            }

            return json{{"panel", panel_details_json(*details)}};
        }

        std::expected<json, std::string> list_menus_payload(vis::Visualizer* viewer,
                                                            const bool include_items) {
            json menus = json::array();
            for (const auto& entry : menu_bar_entries()) {
                const auto content = build_menu_content(entry.idname);
                menus.push_back(menu_descriptor_json(entry, content, include_items));
            }

            return json{
                {"count", static_cast<int64_t>(menus.size())},
                {"menus", std::move(menus)},
                {"state", current_ui_state_json(viewer)},
            };
        }

        std::expected<json, std::string> describe_menu_payload(vis::Visualizer* viewer,
                                                               const std::string& menu_id) {
            auto entry = find_menu_bar_entry(menu_id);
            if (!entry) {
                return std::unexpected(entry.error());
            }

            return json{
                {"menu", menu_descriptor_json(*entry, build_menu_content(menu_id), true)},
                {"state", current_ui_state_json(viewer)},
            };
        }

        std::expected<void, std::string> set_selection_submode(const std::string_view submode_id) {
            for (const auto& [name, selection_mode] : kSelectionSubmodes) {
                if (name != submode_id) {
                    continue;
                }

                lfs::core::events::tools::SetSelectionSubMode{.selection_mode = selection_mode}.emit();
                return {};
            }

            return std::unexpected("Unknown selection submode: " + std::string(submode_id));
        }

    } // namespace

    void register_generic_gui_ui_tools(mcp::ToolRegistry& registry,
                                       vis::Visualizer* viewer) {
        registry.register_tool(
            mcp::McpTool{
                .name = "ui.tool.list",
                .description = "List interactive UI tools, including active state and selection submode",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"include_poll", json{{"type", "boolean"}, {"description", "Include current poll/availability state (default: true)"}}}},
                    .required = {}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "query",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json& args) -> json {
                const bool include_poll = args.value("include_poll", true);
                return post_and_wait(viewer, [viewer, include_poll]() -> json {
                    auto payload = list_tools_payload(viewer, include_poll);
                    if (!payload) {
                        return json{{"error", payload.error()}};
                    }
                    (*payload)["success"] = true;
                    return *payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "ui.tool.describe",
                .description = "Describe one interactive UI tool",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"tool_id", json{{"type", "string"}, {"description", "Registered UI tool id, for example 'builtin.select'"}}},
                        {"include_poll", json{{"type", "boolean"}, {"description", "Include current poll/availability state (default: true)"}}}},
                    .required = {"tool_id"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "query",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json& args) -> json {
                const std::string tool_id = args["tool_id"].get<std::string>();
                const bool include_poll = args.value("include_poll", true);
                return post_and_wait(viewer, [viewer, tool_id, include_poll]() -> json {
                    auto payload = describe_tool_payload(viewer, tool_id, include_poll);
                    if (!payload) {
                        return json{{"error", payload.error()}};
                    }
                    (*payload)["success"] = true;
                    return *payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "ui.tool.invoke",
                .description = "Invoke a registered interactive UI tool, typically activating it in the editor",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"tool_id", json{{"type", "string"}, {"description", "Registered UI tool id, for example 'builtin.select'"}}},
                        {"submode_id", json{{"type", "string"}, {"description", "Optional selection submode to apply before invoking, for example 'rectangle'"}}}},
                    .required = {"tool_id"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json& args) -> json {
                const std::string tool_id = args["tool_id"].get<std::string>();
                const std::optional<std::string> submode_id =
                    args.contains("submode_id") && !args["submode_id"].is_null()
                        ? std::optional(args["submode_id"].get<std::string>())
                        : std::nullopt;

                return post_and_wait(viewer, [viewer, tool_id, submode_id]() -> json {
                    auto description = describe_tool_payload(viewer, tool_id, true);
                    if (!description) {
                        return json{{"error", description.error()}};
                    }

                    if (submode_id) {
                        if (auto result = set_selection_submode(*submode_id); !result) {
                            return json{{"error", result.error()}};
                        }
                    }

                    const bool available_before = (*description)["tool"].value("poll", false);
                    vis::UnifiedToolRegistry::instance().invoke(tool_id);

                    auto payload = current_ui_state_json(viewer);
                    payload["success"] = true;
                    payload["tool_id"] = tool_id;
                    payload["available_before"] = available_before;
                    if (submode_id) {
                        payload["requested_submode_id"] = *submode_id;
                    }
                    return payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "ui.tool.clear_active",
                .description = "Clear the active built-in tool by switching the editor toolbar to none",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"clear_submode", json{{"type", "boolean"}, {"description", "Also clear the tracked active submode (default: false)"}}}},
                    .required = {}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json& args) -> json {
                const bool clear_submode = args.value("clear_submode", false);
                return post_and_wait(viewer, [viewer, clear_submode]() -> json {
                    lfs::core::events::tools::SetToolbarTool{
                        .tool_mode = static_cast<int>(vis::ToolType::None)}
                        .emit();

                    if (clear_submode) {
                        vis::UnifiedToolRegistry::instance().clearActiveSubmode();
                    }

                    auto payload = current_ui_state_json(viewer);
                    payload["success"] = true;
                    return payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "ui.tool.set_submode",
                .description = "Set the active selection submode tracked by the editor and unified tool registry",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"submode_id", json{{"type", "string"}, {"enum", json::array({"centers", "rectangle", "polygon", "lasso", "rings"})}, {"description", "Selection submode id"}}}},
                    .required = {"submode_id"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json& args) -> json {
                const std::string submode_id = args["submode_id"].get<std::string>();
                return post_and_wait(viewer, [viewer, submode_id]() -> json {
                    if (auto result = set_selection_submode(submode_id); !result) {
                        return json{{"error", result.error()}};
                    }

                    auto payload = current_ui_state_json(viewer);
                    payload["success"] = true;
                    return payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "ui.menu.list",
                .description = "List menu-bar menus and optionally include their full item trees and actionable entries",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"include_items", json{{"type", "boolean"}, {"description", "Include full menu item trees and flat action lists (default: false)"}}}},
                    .required = {}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "query",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json& args) -> json {
                const bool include_items = args.value("include_items", false);
                return post_and_wait(viewer, [viewer, include_items]() -> json {
                    auto payload = list_menus_payload(viewer, include_items);
                    if (!payload) {
                        return json{{"error", payload.error()}};
                    }
                    (*payload)["success"] = true;
                    return *payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "ui.menu.describe",
                .description = "Describe one top-level menu, including its nested item tree and flat action list",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"menu_id", json{{"type", "string"}, {"description", "Registered top-level menu id"}}}},
                    .required = {"menu_id"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "query",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json& args) -> json {
                const std::string menu_id = args["menu_id"].get<std::string>();
                return post_and_wait(viewer, [viewer, menu_id]() -> json {
                    auto payload = describe_menu_payload(viewer, menu_id);
                    if (!payload) {
                        return json{{"error", payload.error()}};
                    }
                    (*payload)["success"] = true;
                    return *payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "ui.menu.invoke",
                .description = "Invoke one menu action by action index, callback index, operator id, or label path",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"menu_id", json{{"type", "string"}, {"description", "Registered top-level menu id"}}},
                        {"action_index", json{{"type", "integer"}, {"description", "Deterministic action index from ui.menu.describe"}}},
                        {"callback_index", json{{"type", "integer"}, {"description", "Callback index from ui.menu.describe for callback-backed items"}}},
                        {"operator_id", json{{"type", "string"}, {"description", "Operator id present in this menu"}}},
                        {"path", json{{"type", "array"}, {"items", json{{"type", "string"}}}, {"description", "Full raw or display label path, for example ['View', 'Theme', 'Nord']"}}}},
                    .required = {"menu_id"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                    .destructive = true,
                    .long_running = true,
                }},
            [viewer](const json& args) -> json {
                const std::string menu_id = args["menu_id"].get<std::string>();
                return post_and_wait(viewer, [viewer, menu_id, args]() -> json {
                    auto entry = find_menu_bar_entry(menu_id);
                    if (!entry) {
                        return json{{"error", entry.error()}};
                    }

                    const auto content = build_menu_content(menu_id);
                    auto action = resolve_menu_action(content.action_descriptors, args);
                    if (!action) {
                        return json{{"error", action.error()}};
                    }
                    if (!action->enabled) {
                        return json{{"error", "Menu action is currently disabled"},
                                    {"menu_id", menu_id},
                                    {"action", menu_action_json(*action)}};
                    }

                    if (!action->operator_id.empty()) {
                        const auto* descriptor =
                            vis::op::operators().getDescriptor(action->operator_id);
                        if (!descriptor) {
                            return json{{"error", "Operator is not registered: " + action->operator_id},
                                        {"menu_id", menu_id},
                                        {"action", menu_action_json(*action)}};
                        }

                        vis::op::OperatorProperties props;
                        const auto invocation = vis::op::operators().invoke(action->operator_id, &props);
                        if (invocation.is_running_modal()) {
                            auto payload = current_ui_state_json(viewer);
                            payload["success"] = true;
                            payload["menu_id"] = menu_id;
                            payload["status"] = "running_modal";
                            payload["action"] = menu_action_json(*action);
                            return payload;
                        }
                        if (!invocation.is_finished()) {
                            return json{
                                {"error", action->display_label.empty()
                                              ? "Menu action could not be performed"
                                              : action->display_label + " could not be performed"},
                                {"menu_id", menu_id},
                                {"action", menu_action_json(*action)},
                            };
                        }

                        auto payload = current_ui_state_json(viewer);
                        payload["success"] = true;
                        payload["menu_id"] = menu_id;
                        payload["status"] = "finished";
                        payload["action"] = menu_action_json(*action);
                        return payload;
                    }

                    if (action->callback_index < 0) {
                        return json{{"error", "Menu action is not invokable"},
                                    {"menu_id", menu_id},
                                    {"action", menu_action_json(*action)}};
                    }

                    python::execute_menu_callback(menu_id, action->callback_index);
                    auto payload = current_ui_state_json(viewer);
                    payload["success"] = true;
                    payload["menu_id"] = menu_id;
                    payload["status"] = "invoked_callback";
                    payload["action"] = menu_action_json(*action);
                    return payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "ui.panel.list",
                .description = "List registered UI panels across all panel spaces",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "query",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json&) -> json {
                return post_and_wait(viewer, []() -> json {
                    auto payload = list_panels_payload();
                    if (!payload) {
                        return json{{"error", payload.error()}};
                    }
                    (*payload)["success"] = true;
                    return *payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "ui.panel.describe",
                .description = "Describe a registered UI panel",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"panel_id", json{{"type", "string"}, {"description", "Registered panel id"}}}},
                    .required = {"panel_id"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "query",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json& args) -> json {
                const std::string panel_id = args["panel_id"].get<std::string>();
                return post_and_wait(viewer, [panel_id]() -> json {
                    auto payload = describe_panel_payload(panel_id);
                    if (!payload) {
                        return json{{"error", payload.error()}};
                    }
                    (*payload)["success"] = true;
                    return *payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "ui.panel.update",
                .description = "Update a panel's enabled state or layout metadata",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"panel_id", json{{"type", "string"}, {"description", "Registered panel id"}}},
                        {"enabled", json{{"type", "boolean"}, {"description", "Enable or disable the panel"}}},
                        {"label", json{{"type", "string"}, {"description", "Override the display label"}}},
                        {"order", json{{"type", "integer"}, {"description", "Change the panel sort order"}}},
                        {"space", json{{"type", "string"}, {"enum", json::array({"side_panel", "floating", "viewport_overlay", "main_panel_tab", "scene_header", "status_bar"})}, {"description", "Move the panel to a different panel space"}}},
                        {"parent_id", json{{"type", "string"}, {"description", "Set or clear the parent panel id (use empty string to clear)"}}}},
                    .required = {"panel_id"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                    .destructive = true,
                }},
            [viewer](const json& args) -> json {
                const std::string panel_id = args["panel_id"].get<std::string>();
                const bool has_enabled = args.contains("enabled");
                const bool enabled = has_enabled ? args["enabled"].get<bool>() : false;
                const bool has_label = args.contains("label");
                const std::string label = has_label ? args["label"].get<std::string>() : std::string{};
                const bool has_order = args.contains("order");
                const int order = has_order ? args["order"].get<int>() : 0;
                const bool has_space = args.contains("space") && !args["space"].is_null();
                std::optional<vis::gui::PanelSpace> space;
                if (has_space) {
                    space = panel_space_from_string(args["space"].get<std::string>());
                    if (!space) {
                        return json{{"error", "Unknown panel space: " + args["space"].get<std::string>()}};
                    }
                }

                const bool has_parent_id = args.contains("parent_id");
                const std::string parent_id =
                    has_parent_id && !args["parent_id"].is_null()
                        ? args["parent_id"].get<std::string>()
                        : std::string{};

                return post_and_wait(viewer, [panel_id, has_enabled, enabled, has_label, label, has_order, order, has_space, space, has_parent_id, parent_id]() -> json {
                    auto& panels = vis::gui::PanelRegistry::instance();
                    if (!panels.get_panel(panel_id)) {
                        return json{{"error", "Panel is not registered: " + panel_id}};
                    }

                    if (has_enabled) {
                        panels.set_panel_enabled(panel_id, enabled);
                    }
                    if (has_label && !panels.set_panel_label(panel_id, label)) {
                        return json{{"error", "Failed to update panel label: " + panel_id}};
                    }
                    if (has_order && !panels.set_panel_order(panel_id, order)) {
                        return json{{"error", "Failed to update panel order: " + panel_id}};
                    }
                    if (has_space && !panels.set_panel_space(panel_id, *space)) {
                        return json{{"error", "Failed to update panel space: " + panel_id}};
                    }
                    if (has_parent_id && !panels.set_panel_parent(panel_id, parent_id)) {
                        return json{{"error", "Failed to update panel parent: " + panel_id}};
                    }

                    auto payload = describe_panel_payload(panel_id);
                    if (!payload) {
                        return json{{"error", payload.error()}};
                    }
                    (*payload)["success"] = true;
                    return *payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "ui.operator.activate",
                .description = "Set the editor's active operator and optional gizmo type",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"operator_id", json{{"type", "string"}, {"description", "Operator id to mark active in the editor UI"}}},
                        {"gizmo_type", json{{"type", "string"}, {"description", "Optional gizmo type to associate with the active operator"}}}},
                    .required = {"operator_id"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json& args) -> json {
                const std::string operator_id = args["operator_id"].get<std::string>();
                const std::string gizmo_type = args.value("gizmo_type", "");
                return post_and_wait(viewer, [viewer, operator_id, gizmo_type]() -> json {
                    auto* impl = as_visualizer_impl(viewer);
                    if (!impl) {
                        return json{{"error", "Visualizer implementation is unavailable"}};
                    }

                    impl->getEditorContext().setActiveOperator(operator_id, gizmo_type);
                    auto payload = current_ui_state_json(viewer);
                    payload["success"] = true;
                    payload["operator_id"] = operator_id;
                    return payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "ui.operator.clear_active",
                .description = "Clear the editor's active operator and gizmo state",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = mcp::McpToolMetadata{
                    .category = "ui",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json&) -> json {
                return post_and_wait(viewer, [viewer]() -> json {
                    auto* impl = as_visualizer_impl(viewer);
                    if (!impl) {
                        return json{{"error", "Visualizer implementation is unavailable"}};
                    }

                    impl->getEditorContext().clearActiveOperator();
                    auto payload = current_ui_state_json(viewer);
                    payload["success"] = true;
                    return payload;
                });
            });
    }

    void register_generic_gui_ui_resources(mcp::ResourceRegistry& registry,
                                           vis::Visualizer* viewer) {
        registry.register_resource(
            mcp::McpResource{
                .uri = "lichtfeld://ui/menus",
                .name = "UI Menus",
                .description = "Top-level menus, nested item trees, and flat action lists",
                .mime_type = "application/json"},
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer, [viewer, uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto payload = list_menus_payload(viewer, true);
                    if (!payload) {
                        return std::unexpected(payload.error());
                    }
                    return single_json_resource(uri, *payload);
                });
            });

        registry.register_resource_prefix(
            "lichtfeld://ui/menus/",
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                constexpr std::string_view prefix = "lichtfeld://ui/menus/";
                const auto menu_id = uri.substr(prefix.size());
                if (menu_id.empty()) {
                    return std::unexpected("Menu URI must include an id");
                }

                return post_and_wait(viewer, [viewer, uri, menu_id]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto payload = describe_menu_payload(viewer, menu_id);
                    if (!payload) {
                        return std::unexpected(payload.error());
                    }
                    return single_json_resource(uri, *payload);
                });
            });

        registry.register_resource(
            mcp::McpResource{
                .uri = "lichtfeld://ui/tools",
                .name = "UI Tools",
                .description = "Interactive UI tools with active state and availability",
                .mime_type = "application/json"},
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer, [viewer, uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto payload = list_tools_payload(viewer, true);
                    if (!payload) {
                        return std::unexpected(payload.error());
                    }
                    return single_json_resource(uri, *payload);
                });
            });

        registry.register_resource_prefix(
            "lichtfeld://ui/tools/",
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                constexpr std::string_view prefix = "lichtfeld://ui/tools/";
                const auto tool_id = uri.substr(prefix.size());
                if (tool_id.empty()) {
                    return std::unexpected("Tool URI must include an id");
                }

                return post_and_wait(viewer, [viewer, uri, tool_id]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto payload = describe_tool_payload(viewer, tool_id, true);
                    if (!payload) {
                        return std::unexpected(payload.error());
                    }
                    return single_json_resource(uri, *payload);
                });
            });

        registry.register_resource(
            mcp::McpResource{
                .uri = "lichtfeld://ui/panels",
                .name = "UI Panels",
                .description = "Registered UI panels across all spaces",
                .mime_type = "application/json"},
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer, [uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto payload = list_panels_payload();
                    if (!payload) {
                        return std::unexpected(payload.error());
                    }
                    return single_json_resource(uri, *payload);
                });
            });

        registry.register_resource_prefix(
            "lichtfeld://ui/panels/",
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                constexpr std::string_view prefix = "lichtfeld://ui/panels/";
                const auto panel_id = uri.substr(prefix.size());
                if (panel_id.empty()) {
                    return std::unexpected("Panel URI must include an id");
                }

                return post_and_wait(viewer, [uri, panel_id]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto payload = describe_panel_payload(panel_id);
                    if (!payload) {
                        return std::unexpected(payload.error());
                    }
                    return single_json_resource(uri, *payload);
                });
            });

        registry.register_resource(
            mcp::McpResource{
                .uri = "lichtfeld://ui/state",
                .name = "UI State",
                .description = "Current active tool, operator, gizmo, and panel counts",
                .mime_type = "application/json"},
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer, [viewer, uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    return single_json_resource(uri, current_ui_state_json(viewer));
                });
            });
    }

} // namespace lfs::app
