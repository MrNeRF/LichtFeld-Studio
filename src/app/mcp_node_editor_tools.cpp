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

        json treeResult(const lfs::nodes::NodeTree& tree) {
            return {{"success", true}, {"tree", tree.to_json()}};
        }

        struct ViewportNodeContext {
            vis::ModifierManager* manager = nullptr;
            lfs::nodes::NodeTree* tree = nullptr;
            lfs::nodes::Node* node = nullptr;
            core::Uuid host;
        };

        std::expected<ViewportNodeContext, json>
        viewportNode(vis::VisualizerImpl& viewer, const std::string_view name) {
            const auto state = editor(viewer);
            if (!state.value("open", false))
                return std::unexpected(failure("Open the Node Editor first", "node"));
            auto& manager = viewer.getSceneManager()->modifierManager();
            auto* graph = tree(manager, state);
            const auto host = core::Uuid::from_string(state.value("target", ""));
            auto* node = graph ? graph->find_node(name) : nullptr;
            if (!graph || !host || !viewer.getSceneManager()->getScene().getNodeByUuid(*host))
                return std::unexpected(failure("Show a modifier for a scene node first", "node"));
            if (!node)
                return std::unexpected(failure("Unknown node name in the shown graph", "node"));
            manager.setViewportNodeSelection(*host, graph->uuid, node->name, true);
            return ViewportNodeContext{&manager, graph, node, *host};
        }

        json matrixJson(const glm::mat4& value) {
            json result = json::array();
            for (int row = 0; row < 4; ++row) {
                json values = json::array();
                for (int column = 0; column < 4; ++column)
                    values.push_back(value[column][row]);
                result.push_back(std::move(values));
            }
            return result;
        }

        json vectorJson(const glm::vec3& value) {
            return {value.x, value.y, value.z};
        }

        json gizmoState(vis::VisualizerImpl& viewer) {
            const auto state = viewer.getSceneManager()->modifierManager().viewportNodeGizmo();
            if (!state)
                return failure("Select Box Selection, Ellipsoid Selection or Transform Geometry in the open Node Editor", "node");
            const auto kind = state->kind == vis::NodeViewportGizmoKind::Box
                                  ? "box"
                              : state->kind == vis::NodeViewportGizmoKind::Ellipsoid ? "ellipsoid"
                                                                                     : "transform";
            return {{"success", true},
                    {"node", state->node},
                    {"target", state->host.to_string()},
                    {"kind", kind},
                    {"editable", state->editable},
                    {"local", {{"matrix", matrixJson(state->local_transform)}, {"translation", vectorJson(state->local_translation)}, {"rotation", vectorJson(state->local_rotation)}, {"scale", vectorJson(state->local_scale)}, {"falloff", state->falloff}}},
                    {"world", {{"matrix", matrixJson(state->world_transform)}, {"translation", vectorJson(glm::vec3(state->world_transform[3]))}}}};
        }

        json editView(vis::VisualizerImpl& viewer, const json& args, const std::string_view operation) {
            if (operation == "open" || operation == "close") {
                viewer.screens().edit([&](auto& screen) {
                    if (operation == "open")
                        screen.openEditor("node_editor");
                    else
                        screen.closeEditor("node_editor");
                });
                viewer.getSceneManager()->modifierManager().setViewportEditorVisible(operation == "open");
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
                std::optional<std::unordered_set<std::string>> nodes;
                if (args.contains("nodes"))
                    nodes = args["nodes"].get<std::unordered_set<std::string>>();
                if (!canvas->arrange(args.value("selection_only", true), nodes))
                    return failure("An arranged node is not in the shown graph", "nodes");
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
            } else if (operation == "arrange") {
                properties = {{"selection_only", {{"type", "boolean"}, {"default", true}, {"description", "Arrange selected nodes, or all nodes if none are selected. False arranges the whole graph."}}},
                              {"nodes", {{"type", "array"}, {"items", stringSchema()}, {"description", "Explicit node identifiers to arrange; overrides selection_only. Their bounding-box centre stays fixed."}}}};
            } else if (operation == "view") {
                properties = {{"pan", pointSchema()}, {"zoom", {{"type", "number"}, {"minimum", 0.3}, {"maximum", 2.5}}}};
            } else if (operation == "preview_selection") {
                properties = {{"enabled", boolSchema()}};
                required = {"enabled"};
            }
            add(registry, impl, "nodes.editor_" + operation, operation + " the Node Editor; returns the resulting editor state", properties, required,
                [operation](auto& viewer, const json& args) { return editView(viewer, args, operation); });
        }
        add(registry, impl, "nodes.gizmo_get", "Return the selected node's viewport gizmo in host-local and visualizer-world space", {}, {}, [](auto& viewer, const auto&) { return gizmoState(viewer); }, true);
        add(registry, impl, "nodes.paint_stroke",
            "Add one undoable Paint Selection stroke using world [x,y,z,radius?,value?] or screen [x,y,radius?,value?] samples",
            {{"node", stringSchema()},
             {"space", {{"type", "string"}, {"enum", {"world", "screen"}}, {"default", "world"}}},
             {"samples", {{"type", "array"}, {"items", {{"type", "array"}, {"items", {{"type", "number"}}}, {"minItems", 2}, {"maxItems", 5}}}, {"minItems", 1}}},
             {"erase", boolSchema()}},
            {"node", "samples"}, [](auto& viewer, const json& args) {
                auto context = viewportNode(viewer, args.at("node").get<std::string>());
                if (!context)
                    return context.error();
                if (context->node->type_id != "lfs.paint_selection")
                    return failure("node must be a Paint Selection node", "node");
                const bool screen = args.value("space", "world") == "screen";
                const bool erase = args.value("erase", false);
                std::vector<vis::PaintStrokeSample> samples;
                for (const auto& raw : args.at("samples")) {
                    const std::size_t minimum = screen ? 2 : 3;
                    if (!raw.is_array() || raw.size() < minimum ||
                        std::ranges::any_of(raw, [](const auto& item) {
                            return !item.is_number() || !std::isfinite(item.template get<float>());
                        }))
                        return failure("Each sample must contain finite numeric coordinates", "samples");
                    glm::vec3 position;
                    const std::size_t radius_index = screen ? 2 : 3;
                    const std::size_t value_index = radius_index + 1;
                    float radius = raw.size() > radius_index
                                       ? raw[radius_index].get<float>()
                                   : screen ? context->manager->paintRadius()
                                            : 0.1f;
                    if (screen) {
                        auto* service = viewer.getSceneManager()->getSelectionService();
                        const auto picked = service ? service->pickAtScreen(raw[0].get<float>(), raw[1].get<float>())
                                                    : std::expected<vis::ViewportGaussianPick, vis::ViewportPickError>(std::unexpected(vis::ViewportPickError{"Selection service unavailable"}));
                        if (!picked)
                            return failure(picked.error().message, "samples");
                        position = picked->world_position;
                        const auto world_radius = service->worldRadiusAtScreen(
                            raw[0].get<float>(), raw[1].get<float>(), position, radius);
                        if (!world_radius)
                            return failure(world_radius.error().message, "samples");
                        radius = *world_radius;
                    } else {
                        position = {raw[0].get<float>(), raw[1].get<float>(), raw[2].get<float>()};
                    }
                    const float value = erase                      ? 0.0f
                                        : raw.size() > value_index ? raw[value_index].get<float>()
                                                                   : 1.0f;
                    samples.push_back({position, radius, value});
                }
                const auto result = context->manager->addPaintStroke(samples, true);
                if (!result)
                    return failure(result.error().message, "samples");
                return treeResult(*context->tree);
            });
        add(registry, impl, "nodes.paint_clear", "Clear every stroke from a Paint Selection node as one undo step",
            {{"node", stringSchema()}}, {"node"}, [](auto& viewer, const json& args) {
                auto context = viewportNode(viewer, args.at("node").get<std::string>());
                if (!context)
                    return context.error();
                const auto result = context->manager->clearPaintStrokes();
                if (!result)
                    return failure(result.error().message, "node");
                return treeResult(*context->tree);
            });
        add(registry, impl, "nodes.paint_mode",
            "Turn viewport painting on or off for a Paint Selection node, optionally setting the brush radius in screen pixels",
            {{"node", stringSchema()}, {"enabled", boolSchema()}, {"radius", {{"type", "number"}, {"minimum", 2}, {"maximum", 256}}}},
            {"node", "enabled"}, [](auto& viewer, const json& args) {
                auto context = viewportNode(viewer, args.at("node").get<std::string>());
                if (!context)
                    return context.error();
                if (!context->manager->setPaintMode(args.at("enabled").get<bool>()))
                    return failure("node must be a Paint Selection node", "node");
                if (args.contains("radius")) {
                    const float radius = args["radius"].get<float>();
                    if (!std::isfinite(radius) || radius < 2.0f || radius > 256.0f)
                        return failure("radius must be between 2 and 256 pixels", "radius");
                    context->manager->adjustPaintRadius(radius / context->manager->paintRadius());
                }
                if (auto* rendering = viewer.getRenderingManager())
                    rendering->markDirty(vis::DirtyFlag::ALL, vis::FrameReason::Mcp, "node paint mode");
                return json{{"success", true},
                            {"paint_mode", context->manager->paintModeActive()},
                            {"radius", context->manager->paintRadius()}};
            });
        add(registry, impl, "nodes.pick_colour",
            "Pick a Gaussian's stored base colour at a viewport screen coordinate and write one unlinked colour input",
            {{"node", stringSchema()}, {"input", stringSchema()}, {"screen_x", {{"type", "number"}}}, {"screen_y", {{"type", "number"}}}},
            {"node", "input", "screen_x", "screen_y"}, [](auto& viewer, const json& args) {
                auto context = viewportNode(viewer, args.at("node").get<std::string>());
                if (!context)
                    return context.error();
                if (!context->manager->beginColourPick(context->node->name,
                                                       args.at("input").get<std::string>()))
                    return failure("input must be an unlinked colour input, or Hue on HSV Range", "input");
                auto* service = viewer.getSceneManager()->getSelectionService();
                const auto picked = service ? service->pickAtScreen(args.at("screen_x").get<float>(),
                                                                    args.at("screen_y").get<float>())
                                            : std::expected<vis::ViewportGaussianPick, vis::ViewportPickError>(std::unexpected(vis::ViewportPickError{"Selection service unavailable"}));
                if (!picked) {
                    context->manager->cancelViewportMode();
                    return failure(picked.error().message, "screen_x");
                }
                const auto result = context->manager->applyPickedColour(picked->colour);
                if (!result)
                    return failure(result.error().message, "input");
                auto response = treeResult(*context->tree);
                response["picked"] = {picked->colour.x, picked->colour.y, picked->colour.z};
                response["stored_payload"] = picked->colour_from_stored_payload;
                return response;
            });
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
