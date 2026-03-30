/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/include/app/mcp_sequencer_tools.hpp"

#include "app/include/app/mcp_app_utils.hpp"
#include "mcp/mcp_tools.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "sequencer/keyframe.hpp"
#include "visualizer/ipc/view_context.hpp"
#include "visualizer/sequencer/sequencer_controller.hpp"
#include "visualizer/visualizer.hpp"

#include <cassert>
#include <optional>
#include <string>

#include <glm/glm.hpp>

namespace lfs::app {

    namespace {

        using json = nlohmann::json;
        using mcp::McpTool;

        json vec3_to_json(const glm::vec3& value) {
            return json::array({value.x, value.y, value.z});
        }

        json view_info_json(const vis::ViewInfo& info) {
            const json rotation = json::array({
                json::array({info.rotation[0], info.rotation[1], info.rotation[2]}),
                json::array({info.rotation[3], info.rotation[4], info.rotation[5]}),
                json::array({info.rotation[6], info.rotation[7], info.rotation[8]}),
            });
            const glm::mat3 rotation_matrix(
                glm::vec3(info.rotation[0], info.rotation[3], info.rotation[6]),
                glm::vec3(info.rotation[1], info.rotation[4], info.rotation[7]),
                glm::vec3(info.rotation[2], info.rotation[5], info.rotation[8]));
            const glm::vec3 forward = rendering::cameraForward(rotation_matrix);

            return json{
                {"success", true},
                {"camera", {
                               {"eye", json::array({info.translation[0], info.translation[1], info.translation[2]})},
                               {"target", json::array({info.pivot[0], info.pivot[1], info.pivot[2]})},
                               {"pivot", json::array({info.pivot[0], info.pivot[1], info.pivot[2]})},
                               {"up", json::array({info.rotation[1], info.rotation[4], info.rotation[7]})},
                               {"forward", json::array({forward.x, forward.y, forward.z})},
                               {"rotation_matrix", rotation},
                               {"width", info.width},
                               {"height", info.height},
                               {"fov_degrees", info.fov},
                           }},
            };
        }

        std::expected<std::optional<glm::vec3>, std::string> optional_vec3_arg(const json& args, const char* key) {
            if (!args.contains(key) || args[key].is_null())
                return std::optional<glm::vec3>{};

            const auto& value = args[key];
            if (!value.is_array() || value.size() != 3)
                return std::unexpected(std::string("Field '") + key + "' must be a 3-element array");

            return glm::vec3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
        }

        const char* keyframe_easing_name(const uint8_t easing) {
            switch (easing) {
            case 0: return "linear";
            case 1: return "ease_in";
            case 2: return "ease_out";
            case 3: return "ease_in_out";
            default: return "unknown";
            }
        }

        json keyframe_json(const sequencer::Keyframe& keyframe, const int64_t visible_index) {
            return json{
                {"name", "Keyframe " + std::to_string(visible_index + 1)},
                {"id", keyframe.id},
                {"time", keyframe.time},
                {"position", vec3_to_json(keyframe.position)},
                {"rotation_quat", json::array({keyframe.rotation.w, keyframe.rotation.x, keyframe.rotation.y, keyframe.rotation.z})},
                {"focal_length_mm", keyframe.focal_length_mm},
                {"easing", keyframe.easing},
                {"easing_name", keyframe_easing_name(static_cast<uint8_t>(keyframe.easing))},
            };
        }

        std::expected<vis::SequencerController*, std::string> ensure_ready_controller(const SequencerToolBackend& backend) {
            if (backend.ensure_ready) {
                auto ready = backend.ensure_ready();
                if (!ready)
                    return std::unexpected(ready.error());
            }

            if (!backend.controller)
                return std::unexpected("Sequencer controller backend is unavailable");

            auto* const controller = backend.controller();
            if (!controller)
                return std::unexpected("Sequencer controller not initialized");

            return controller;
        }

        std::expected<std::optional<size_t>, std::string> resolve_optional_sequencer_keyframe_command_index(
            const json& args,
            const SequencerToolBackend& backend) {
            auto controller = ensure_ready_controller(backend);
            if (!controller)
                return std::unexpected(controller.error());

            const auto& timeline = (*controller)->timeline();
            if (args.contains("keyframe_id")) {
                const auto keyframe_id = args["keyframe_id"].get<sequencer::KeyframeId>();
                if (const auto keyframe_index = timeline.findKeyframeIndex(keyframe_id); keyframe_index.has_value()) {
                    const auto* const keyframe = timeline.getKeyframe(*keyframe_index);
                    if (keyframe && !keyframe->is_loop_point)
                        return std::optional<size_t>(*keyframe_index);
                }
                return std::unexpected("Keyframe id not found: " + std::to_string(keyframe_id));
            }

            return std::optional<size_t>{};
        }

        std::expected<size_t, std::string> resolve_required_sequencer_keyframe_command_index(
            const json& args,
            const SequencerToolBackend& backend) {
            auto resolved = resolve_optional_sequencer_keyframe_command_index(args, backend);
            if (!resolved)
                return std::unexpected(resolved.error());
            if (!resolved->has_value())
                return std::unexpected("Missing keyframe_id");
            return **resolved;
        }

        json sequencer_state_json(const SequencerToolBackend& backend, const vis::SequencerController& controller) {
            const auto& timeline = controller.timeline();
            const bool has_any_timeline_state =
                timeline.realKeyframeCount() > 0 || timeline.hasAnimationClip();
            json keyframe_list = json::array();
            int64_t visible_index = 0;
            for (const auto& keyframe : timeline.keyframes()) {
                if (keyframe.is_loop_point)
                    continue;
                keyframe_list.push_back(keyframe_json(keyframe, visible_index));
                ++visible_index;
            }

            const auto* const ui_state = backend.ui_state ? backend.ui_state() : nullptr;
            json result = json{
                {"success", true},
                {"visible", backend.is_visible ? backend.is_visible() : false},
                {"has_keyframes", has_any_timeline_state},
                {"playback_speed", controller.playbackSpeed()},
                {"show_camera_path", ui_state ? ui_state->show_camera_path : true},
                {"follow_playback", ui_state ? ui_state->follow_playback : false},
                {"keyframe_count", keyframe_list.size()},
                {"keyframes", keyframe_list},
            };
            result["selected_keyframe_id"] = controller.selectedKeyframeId()
                                                 ? json(*controller.selectedKeyframeId())
                                                 : json(nullptr);
            return result;
        }

        void apply_camera_args(
            const std::optional<glm::vec3>& eye,
            const std::optional<glm::vec3>& target,
            const std::optional<glm::vec3>& up,
            const std::optional<float>& fov) {
            if (eye && target) {
                const glm::vec3 up_value = up.value_or(glm::vec3(0.0f, 1.0f, 0.0f));
                vis::apply_set_view(vis::SetViewParams{
                    .eye = {eye->x, eye->y, eye->z},
                    .target = {target->x, target->y, target->z},
                    .up = {up_value.x, up_value.y, up_value.z},
                });
            }
            if (fov)
                vis::apply_set_fov(*fov);
        }

    } // namespace

    void register_gui_sequencer_tools(
        mcp::ToolRegistry& registry,
        vis::Visualizer* viewer,
        SequencerToolBackend backend) {
        assert(viewer);

        registry.register_tool(
            McpTool{
                .name = "sequencer.get",
                .description = "Inspect sequencer visibility, selected keyframe, and stable keyframe IDs",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer, backend](const json&) -> json {
                return post_and_wait(viewer, [backend]() -> json {
                    auto controller = ensure_ready_controller(backend);
                    if (!controller)
                        return json{{"error", controller.error()}};
                    return sequencer_state_json(backend, **controller);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "sequencer.add_keyframe",
                .description = "Add a keyframe at the current viewport camera, optionally setting the camera first",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"eye", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional camera eye position [x,y,z]"}}},
                        {"target", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional camera target [x,y,z]"}}},
                        {"up", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional camera up vector [x,y,z]"}}},
                        {"fov_degrees", json{{"type", "number"}, {"description", "Optional camera FOV override"}}},
                        {"show_sequencer", json{{"type", "boolean"}, {"description", "Show the sequencer panel before operating (default: true)"}}}},
                    .required = {}}},
            [viewer, backend](const json& args) -> json {
                auto eye = optional_vec3_arg(args, "eye");
                if (!eye)
                    return json{{"error", eye.error()}};
                auto target = optional_vec3_arg(args, "target");
                if (!target)
                    return json{{"error", target.error()}};
                auto up = optional_vec3_arg(args, "up");
                if (!up)
                    return json{{"error", up.error()}};
                if (eye->has_value() != target->has_value())
                    return json{{"error", "Fields 'eye' and 'target' must either both be provided or both be omitted"}};

                const std::optional<float> fov = args.contains("fov_degrees")
                                                     ? std::optional<float>(args["fov_degrees"].get<float>())
                                                     : std::nullopt;
                const bool show_sequencer = args.value("show_sequencer", true);

                return post_and_wait(viewer, [backend, eye = *eye, target = *target, up = *up, fov, show_sequencer]() -> json {
                    auto controller = ensure_ready_controller(backend);
                    if (!controller)
                        return json{{"error", controller.error()}};

                    if (show_sequencer && backend.set_visible)
                        backend.set_visible(true);
                    apply_camera_args(eye, target, up, fov);
                    if (backend.add_keyframe)
                        backend.add_keyframe();
                    return sequencer_state_json(backend, **controller);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "sequencer.update_keyframe",
                .description = "Update the selected keyframe, optionally selecting it and/or setting the viewport camera first",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"keyframe_id", json{{"type", "integer"}, {"description", "Stable keyframe id to select before updating"}}},
                        {"eye", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional camera eye position [x,y,z]"}}},
                        {"target", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional camera target [x,y,z]"}}},
                        {"up", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional camera up vector [x,y,z]"}}},
                        {"fov_degrees", json{{"type", "number"}, {"description", "Optional camera FOV override"}}},
                        {"show_sequencer", json{{"type", "boolean"}, {"description", "Show the sequencer panel before operating (default: true)"}}}},
                    .required = {}}},
            [viewer, backend](const json& args) -> json {
                auto eye = optional_vec3_arg(args, "eye");
                if (!eye)
                    return json{{"error", eye.error()}};
                auto target = optional_vec3_arg(args, "target");
                if (!target)
                    return json{{"error", target.error()}};
                auto up = optional_vec3_arg(args, "up");
                if (!up)
                    return json{{"error", up.error()}};
                if (eye->has_value() != target->has_value())
                    return json{{"error", "Fields 'eye' and 'target' must either both be provided or both be omitted"}};

                const std::optional<float> fov = args.contains("fov_degrees")
                                                     ? std::optional<float>(args["fov_degrees"].get<float>())
                                                     : std::nullopt;
                const bool show_sequencer = args.value("show_sequencer", true);

                return post_and_wait(viewer, [backend, args, eye = *eye, target = *target, up = *up, fov, show_sequencer]() -> json {
                    auto controller = ensure_ready_controller(backend);
                    if (!controller)
                        return json{{"error", controller.error()}};

                    const auto keyframe_index = resolve_optional_sequencer_keyframe_command_index(args, backend);
                    if (!keyframe_index)
                        return json{{"error", keyframe_index.error()}};

                    if (show_sequencer && backend.set_visible)
                        backend.set_visible(true);
                    if (keyframe_index->has_value() && backend.select_keyframe)
                        backend.select_keyframe(**keyframe_index);
                    apply_camera_args(eye, target, up, fov);
                    if (backend.update_selected_keyframe)
                        backend.update_selected_keyframe();
                    return sequencer_state_json(backend, **controller);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "sequencer.select_keyframe",
                .description = "Select a keyframe in the shared sequencer timeline",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"keyframe_id", json{{"type", "integer"}, {"description", "Stable keyframe id"}}},
                        {"show_sequencer", json{{"type", "boolean"}, {"description", "Show the sequencer panel before operating (default: true)"}}}},
                    .required = {"keyframe_id"}}},
            [viewer, backend](const json& args) -> json {
                const bool show_sequencer = args.value("show_sequencer", true);

                return post_and_wait(viewer, [backend, args, show_sequencer]() -> json {
                    auto controller = ensure_ready_controller(backend);
                    if (!controller)
                        return json{{"error", controller.error()}};

                    const auto keyframe_index = resolve_required_sequencer_keyframe_command_index(args, backend);
                    if (!keyframe_index)
                        return json{{"error", keyframe_index.error()}};
                    if (show_sequencer && backend.set_visible)
                        backend.set_visible(true);
                    if (backend.select_keyframe)
                        backend.select_keyframe(*keyframe_index);
                    return sequencer_state_json(backend, **controller);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "sequencer.go_to_keyframe",
                .description = "Move the viewport camera to a sequencer keyframe",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"keyframe_id", json{{"type", "integer"}, {"description", "Stable keyframe id"}}},
                        {"show_sequencer", json{{"type", "boolean"}, {"description", "Show the sequencer panel before operating (default: true)"}}}},
                    .required = {"keyframe_id"}}},
            [viewer, backend](const json& args) -> json {
                const bool show_sequencer = args.value("show_sequencer", true);

                return post_and_wait(viewer, [backend, args, show_sequencer]() -> json {
                    auto controller = ensure_ready_controller(backend);
                    if (!controller)
                        return json{{"error", controller.error()}};

                    const auto keyframe_index = resolve_required_sequencer_keyframe_command_index(args, backend);
                    if (!keyframe_index)
                        return json{{"error", keyframe_index.error()}};
                    if (show_sequencer && backend.set_visible)
                        backend.set_visible(true);
                    if (backend.go_to_keyframe)
                        backend.go_to_keyframe(*keyframe_index);

                    json result = sequencer_state_json(backend, **controller);
                    const auto info = vis::get_current_view_info();
                    if (info)
                        result["camera"] = view_info_json(*info)["camera"];
                    return result;
                });
            });

        registry.register_tool(
            McpTool{
                .name = "sequencer.delete_keyframe",
                .description = "Delete a keyframe from the shared sequencer timeline",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"keyframe_id", json{{"type", "integer"}, {"description", "Stable keyframe id"}}}},
                    .required = {"keyframe_id"}}},
            [viewer, backend](const json& args) -> json {
                return post_and_wait(viewer, [backend, args]() -> json {
                    auto controller = ensure_ready_controller(backend);
                    if (!controller)
                        return json{{"error", controller.error()}};

                    const auto keyframe_index = resolve_required_sequencer_keyframe_command_index(args, backend);
                    if (!keyframe_index)
                        return json{{"error", keyframe_index.error()}};

                    if (backend.delete_keyframe)
                        backend.delete_keyframe(*keyframe_index);
                    return sequencer_state_json(backend, **controller);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "sequencer.set_easing",
                .description = "Set the easing mode for a keyframe",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"keyframe_id", json{{"type", "integer"}, {"description", "Stable keyframe id"}}},
                        {"easing", json{{"oneOf", json::array({json{{"type", "integer"}},
                                                               json{{"type", "string"}, {"enum", json::array({"linear", "ease_in", "ease_out", "ease_in_out"})}}})},
                                        {"description", "Easing mode as integer or name"}}}},
                    .required = {"keyframe_id", "easing"}}},
            [viewer, backend](const json& args) -> json {
                int easing = 0;
                if (args["easing"].is_string()) {
                    const std::string value = args["easing"].get<std::string>();
                    if (value == "linear")
                        easing = 0;
                    else if (value == "ease_in")
                        easing = 1;
                    else if (value == "ease_out")
                        easing = 2;
                    else if (value == "ease_in_out")
                        easing = 3;
                    else
                        return json{{"error", "Unsupported easing mode: " + value}};
                } else {
                    easing = args["easing"].get<int>();
                }

                return post_and_wait(viewer, [backend, args, easing]() -> json {
                    auto controller = ensure_ready_controller(backend);
                    if (!controller)
                        return json{{"error", controller.error()}};

                    const auto keyframe_index = resolve_required_sequencer_keyframe_command_index(args, backend);
                    if (!keyframe_index)
                        return json{{"error", keyframe_index.error()}};

                    if (backend.set_keyframe_easing)
                        backend.set_keyframe_easing(*keyframe_index, easing);
                    return sequencer_state_json(backend, **controller);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "sequencer.play_pause",
                .description = "Toggle sequencer playback",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer, backend](const json&) -> json {
                return post_and_wait(viewer, [backend]() -> json {
                    auto controller = ensure_ready_controller(backend);
                    if (!controller)
                        return json{{"error", controller.error()}};

                    if (backend.play_pause)
                        backend.play_pause();
                    json result = sequencer_state_json(backend, **controller);
                    result["toggled"] = true;
                    return result;
                });
            });

        registry.register_tool(
            McpTool{
                .name = "sequencer.clear",
                .description = "Clear all sequencer keyframes",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer, backend](const json&) -> json {
                return post_and_wait(viewer, [backend]() -> json {
                    auto controller = ensure_ready_controller(backend);
                    if (!controller)
                        return json{{"error", controller.error()}};

                    if (backend.clear)
                        backend.clear();
                    return sequencer_state_json(backend, **controller);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "sequencer.save_path",
                .description = "Save the sequencer camera path to JSON",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Destination JSON path"}}}},
                    .required = {"path"}}},
            [viewer, backend](const json& args) -> json {
                const std::string path = args["path"].get<std::string>();

                return post_and_wait(viewer, [backend, path]() -> json {
                    auto controller = ensure_ready_controller(backend);
                    if (!controller)
                        return json{{"error", controller.error()}};

                    const bool saved = backend.save_path && backend.save_path(path);
                    if (!saved)
                        return json{{"error", "Failed to save camera path"}};

                    json result = sequencer_state_json(backend, **controller);
                    result["path"] = path;
                    return result;
                });
            });

        registry.register_tool(
            McpTool{
                .name = "sequencer.load_path",
                .description = "Load the sequencer camera path from JSON",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Source JSON path"}}},
                        {"show_sequencer", json{{"type", "boolean"}, {"description", "Show the sequencer panel before operating (default: true)"}}}},
                    .required = {"path"}}},
            [viewer, backend](const json& args) -> json {
                const std::string path = args["path"].get<std::string>();
                const bool show_sequencer = args.value("show_sequencer", true);

                return post_and_wait(viewer, [backend, path, show_sequencer]() -> json {
                    auto controller = ensure_ready_controller(backend);
                    if (!controller)
                        return json{{"error", controller.error()}};

                    if (show_sequencer && backend.set_visible)
                        backend.set_visible(true);

                    const bool loaded = backend.load_path && backend.load_path(path);
                    if (!loaded)
                        return json{{"error", "Failed to load camera path"}};

                    json result = sequencer_state_json(backend, **controller);
                    result["path"] = path;
                    return result;
                });
            });

        registry.register_tool(
            McpTool{
                .name = "sequencer.set_playback_speed",
                .description = "Set the sequencer playback speed multiplier",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"speed", json{{"type", "number"}, {"description", "Playback speed multiplier"}}}},
                    .required = {"speed"}}},
            [viewer, backend](const json& args) -> json {
                const float speed = args["speed"].get<float>();

                return post_and_wait(viewer, [backend, speed]() -> json {
                    auto controller = ensure_ready_controller(backend);
                    if (!controller)
                        return json{{"error", controller.error()}};

                    if (backend.set_playback_speed)
                        backend.set_playback_speed(speed);
                    return sequencer_state_json(backend, **controller);
                });
            });
    }

} // namespace lfs::app
