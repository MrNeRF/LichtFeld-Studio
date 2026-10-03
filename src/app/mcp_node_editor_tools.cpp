/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/mcp_node_tools.hpp"
#include "mcp_node_utils.hpp"
#include "visualizer/gui/gui_manager.hpp"
#include "visualizer/gui/rmlui/elements/node_canvas_element.hpp"
#include "visualizer/gui/screen_host.hpp"
#include "visualizer/rendering/rendering_manager.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

namespace lfs::app {
    namespace {
        using namespace node_mcp;

        json editView(vis::VisualizerImpl& viewer, const json& args, const std::string_view operation) {
            if (operation == "open" || operation == "close") {
                viewer.screens().edit([&](auto& screen) {
                    if (operation == "open")
                        screen.openEditor("node_editor");
                    else
                        screen.closeEditor("node_editor");
                });
                if (auto* rendering = viewer.getRenderingManager())
                    rendering->markDirty(vis::DirtyFlag::ALL, vis::FrameReason::Mcp, "node editor");
                return editor(viewer);
            }
            if (!editor(viewer).value("open", false))
                return failure("Open the Node Editor with nodes_editor_open first", "editor");
            auto* canvas = viewer.getGuiManager()->screenHost().nodeCanvas();
            if (!canvas)
                return failure("The Node Editor is not initialized; wait for the first rendered frame", "editor");
            canvas->refresh();
            if (operation == "show") {
                auto& scene = *viewer.getSceneManager();
                const auto host = target(scene, args);
                if (!host)
                    return failure("Unknown splat, mesh or point cloud UUID", "target");
                auto& manager = scene.modifierManager();
                auto* item = modifier(manager, *host, args);
                if (!item && args.contains("tree")) {
                    auto* stack = manager.stack(*host);
                    if (stack) {
                        const auto* graph = tree(scene.modifierManager(), args);
                        const auto found = std::ranges::find(stack->modifiers, graph ? graph->uuid : std::string{}, &vis::Modifier::tree_uuid);
                        if (found != stack->modifiers.end())
                            item = &*found;
                    }
                }
                if (!item)
                    return failure("Specify a modifier UUID or a tree UUID used by this target's stack", "modifier");
                scene.selectNode(scene.getScene().getNodeByUuid(*host)->id);
                if (!canvas->showModifier(item->uuid))
                    return failure("Unable to show the modifier", "modifier");
            } else if (operation == "arrange") {
                canvas->arrange();
                canvas->refresh();
            } else if (operation == "frame") {
                canvas->headerAction("frame", 0.0f, 0.0f);
            } else if (operation == "view") {
                const auto state = canvas->viewState();
                const auto pan = args.value("pan", state["pan"]);
                const float zoom = args.value("zoom", state["zoom"].get<float>());
                if (!pan.is_array() || pan.size() != 2 || !pan[0].is_number() || !pan[1].is_number() ||
                    !std::isfinite(pan[0].get<float>()) || !std::isfinite(pan[1].get<float>()) ||
                    !std::isfinite(zoom) || zoom < 0.3f || zoom > 2.5f)
                    return failure("pan must be finite [x,y]; zoom must be between 0.3 and 2.5", "view");
                canvas->setView({pan[0].get<float>(), pan[1].get<float>()}, zoom);
            } else if (operation == "preview_selection") {
                canvas->setPreviewSelection(args.at("enabled").get<bool>());
            } else if (operation == "select") {
                const auto nodes = args.value("nodes", std::unordered_set<std::string>{});
                std::optional<lfs::nodes::Link> link;
                if (args.contains("links") && !args["links"].empty()) {
                    if (args["links"].size() > 1)
                        return failure("The editor supports one selected link at a time", "links");
                    const auto& item = args["links"][0];
                    if (!item.contains("from_node") || !item.contains("from_socket") || !item.contains("to_node") || !item.contains("to_socket"))
                        return failure("Each link needs from_node, from_socket, to_node and to_socket", "links");
                    link = lfs::nodes::Link{item["from_node"], item["from_socket"], item["to_node"], item["to_socket"]};
                }
                if (!canvas->selectNodes(nodes, link))
                    return failure("A selected node or link is not in the shown graph", "nodes");
            }
            return editor(viewer);
        }
    } // namespace

    nlohmann::json node_evaluation_job(vis::VisualizerImpl& viewer) {
        auto& manager = viewer.getSceneManager()->modifierManager();
        const auto progress = manager.progress();
        return {{"id", "nodes.evaluate"}, {"job_id", "nodes.evaluate"}, {"label", "Node evaluation"}, {"kind", "nodes"}, {"available", true}, {"active", progress.busy}, {"status", progress.busy ? "running" : "idle"}, {"cancel_supported", false}, {"progress", node_mcp::progress(progress)}, {"resource_uri", "lichtfeld://runtime/jobs/nodes.evaluate"}, {"results", node_mcp::stacks(*viewer.getSceneManager())}};
    }

    void register_gui_node_editor_tools(mcp::ToolRegistry& registry, vis::Visualizer* viewer) {
        using namespace node_mcp;
        auto* impl = dynamic_cast<vis::VisualizerImpl*>(viewer);
        if (!impl)
            return;
        for (const std::string operation : {"open", "close", "show", "select", "arrange", "frame", "view", "preview_selection"}) {
            json properties = json::object();
            std::vector<std::string> required;
            if (operation == "show") {
                properties = {{"target", stringSchema()}, {"modifier", stringSchema()}, {"tree", stringSchema()}};
                required = {"target"};
            } else if (operation == "select") {
                properties = {{"nodes", {{"type", "array"}, {"items", stringSchema()}}}, {"links", {{"type", "array"}, {"items", {{"type", "object"}}}}}};
            } else if (operation == "view") {
                properties = {{"pan", pointSchema()}, {"zoom", {{"type", "number"}, {"minimum", 0.3}, {"maximum", 2.5}}}};
            } else if (operation == "preview_selection") {
                properties = {{"enabled", boolSchema()}};
                required = {"enabled"};
            }
            add(registry, impl, "nodes.editor_" + operation, operation + " the Node Editor; returns the resulting editor state", properties, required,
                [operation](auto& viewer, const json& args) { return editView(viewer, args, operation); });
        }
        registry.register_tool(mcp::McpTool{
                                   .name = "nodes.evaluate",
                                   .description = "Submit dirty modifier stacks to the evaluation worker. Optionally wait without blocking the viewer; job id nodes.evaluate.",
                                   .input_schema = {.type = "object", .properties = {{"target", stringSchema()}, {"wait", boolSchema()}, {"timeout_ms", {{"type", "integer"}, {"minimum", 0}, {"maximum", 120000}}}}, .required = {"target"}},
                                   .metadata = {.category = "nodes", .kind = "command", .runtime = "gui", .thread_affinity = "gui_thread", .long_running = true}},
                               [impl](const json& args) {
                                   const auto poll = [&]() {
                                       return post_and_wait(impl, [&]() {
                                           auto& scene = *impl->getSceneManager();
                                           const auto host = target(scene, args);
                                           if (!host)
                                               return failure("Unknown splat, mesh or point cloud UUID", "target");
                                           scene.modifierManager().tick();
                                           auto result = stack(scene, *host);
                                           result["job_id"] = "nodes.evaluate";
                                           return result;
                                       });
                                   };
                                   auto result = poll();
                                   if (result.contains("error") || !args.value("wait", false))
                                       return result;
                                   const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::clamp(args.value("timeout_ms", 30000), 0, 120000));
                                   while (result["progress"].value("busy", false) && std::chrono::steady_clock::now() < deadline) {
                                       std::this_thread::sleep_for(std::chrono::milliseconds(10));
                                       result = poll();
                                       if (result.contains("error"))
                                           return result;
                                   }
                                   result["timed_out"] = result["progress"].value("busy", false);
                                   return result;
                               });
    }
} // namespace lfs::app
