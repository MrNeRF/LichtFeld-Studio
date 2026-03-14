/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#define GLM_ENABLE_EXPERIMENTAL

#include "app/mcp_gui_tools.hpp"
#include "app/mcp_app_utils.hpp"
#include "app/mcp_event_handlers.hpp"
#include "app/mcp_operator_tools.hpp"
#include "app/mcp_runtime_tools.hpp"
#include "app/mcp_sequencer_tools.hpp"
#include "app/mcp_ui_registry_tools.hpp"

#include "core/event_bridge/command_center_bridge.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor.hpp"
#include "io/exporter.hpp"
#include "mcp/llm_client.hpp"
#include "mcp/mcp_tools.hpp"
#include "mcp/render_capture_utils.hpp"
#include "mcp/shared_scene_tools.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"
#include "rendering/gs_rasterizer_tensor.hpp"
#include "sequencer/keyframe.hpp"
#include "visualizer/gui/html_viewer_export.hpp"
#include "visualizer/gui/panels/python_console_panel.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "visualizer/ipc/view_context.hpp"
#include "visualizer/operation/undo_entry.hpp"
#include "visualizer/operation/undo_history.hpp"
#include "visualizer/operator/operator_properties.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/visualizer.hpp"
#include "visualizer/visualizer_impl.hpp"

#include <glad/glad.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <deque>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

namespace lfs::app {

    using json = nlohmann::json;
    using mcp::McpResource;
    using mcp::McpResourceContent;
    using mcp::McpTool;
    using mcp::ResourceRegistry;
    using mcp::ToolRegistry;

    namespace {

        using TransformComponents = vis::cap::TransformComponents;

        const core::SceneNode* find_first_visible_splat_node(const core::Scene& scene) {
            for (const auto* node : scene.getNodes()) {
                if (node->type == core::NodeType::SPLAT && node->model &&
                    static_cast<bool>(node->visible))
                    return node;
            }
            return nullptr;
        }

        std::expected<int64_t, std::string> count_visible_model_gaussians(const core::Scene& scene) {
            int64_t total = 0;
            bool has_model = false;
            for (const auto* node : scene.getVisibleNodes()) {
                if (!node)
                    continue;
                has_model = true;
                total += static_cast<int64_t>(node->gaussian_count);
            }

            if (!has_model)
                return std::unexpected("No model loaded");

            return total;
        }

        std::expected<std::string, std::string> render_scene_to_base64(
            core::Scene& scene,
            int camera_index = 0,
            int width = 0,
            int height = 0) {

            auto* model = scene.getTrainingModel();
            if (!model) {
                const auto* node = find_first_visible_splat_node(scene);
                if (node)
                    model = node->model.get();
            }
            if (!model)
                return std::unexpected("No model to render");

            auto cameras = scene.getAllCameras();
            if (cameras.empty())
                return std::unexpected("No cameras available");

            if (camera_index < 0 || camera_index >= static_cast<int>(cameras.size()))
                camera_index = 0;

            auto& camera = cameras[camera_index];
            if (!camera)
                return std::unexpected("Failed to get camera");

            core::Tensor bg = core::Tensor::zeros({3}, core::Device::CUDA);

            try {
                auto [image, alpha] = rendering::rasterize_tensor(*camera, *model, bg);
                return mcp::encode_render_tensor_to_base64(std::move(image), width, height);
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Render failed: ") + e.what());
            }
        }

        std::expected<std::string, std::string> capture_live_viewport_to_base64(
            vis::Visualizer* viewer,
            int width = 0,
            int height = 0) {
            auto* const viewer_impl = dynamic_cast<vis::VisualizerImpl*>(viewer);
            if (!viewer_impl)
                return std::unexpected("Live viewport capture requires a GUI visualizer");

            auto* const gui_manager = viewer_impl->getGuiManager();
            auto* const window_manager = viewer_impl->getWindowManager();
            if (!gui_manager || !window_manager)
                return std::unexpected("Viewport capture is not initialized");

            const glm::vec2 viewport_pos = gui_manager->getViewportPos();
            const glm::vec2 viewport_size = gui_manager->getViewportSize();
            const glm::ivec2 window_size = window_manager->getWindowSize();
            const glm::ivec2 framebuffer_size = window_manager->getFramebufferSize();
            if (window_size.x <= 0 || window_size.y <= 0 ||
                framebuffer_size.x <= 0 || framebuffer_size.y <= 0) {
                return std::unexpected("Window framebuffer size is unavailable");
            }

            const float scale_x = static_cast<float>(framebuffer_size.x) / static_cast<float>(window_size.x);
            const float scale_y = static_cast<float>(framebuffer_size.y) / static_cast<float>(window_size.y);

            const int capture_x = std::max(0, static_cast<int>(viewport_pos.x * scale_x));
            const int capture_y_top = std::max(0, static_cast<int>(viewport_pos.y * scale_y));
            const int capture_width = std::min(
                framebuffer_size.x - capture_x,
                std::max(1, static_cast<int>(viewport_size.x * scale_x)));
            const int capture_height = std::min(
                framebuffer_size.y - capture_y_top,
                std::max(1, static_cast<int>(viewport_size.y * scale_y)));
            if (capture_width <= 0 || capture_height <= 0)
                return std::unexpected("Viewport capture size is empty");

            const int capture_y_bottom = framebuffer_size.y - capture_y_top - capture_height;
            if (capture_y_bottom < 0)
                return std::unexpected("Viewport capture bounds are invalid");

            std::vector<uint8_t> pixels(static_cast<size_t>(capture_width) * capture_height * 4u);

            GLint previous_read_fbo = 0;
            GLint previous_read_buffer = GL_BACK;
            GLint previous_pack_alignment = 4;
            glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previous_read_fbo);
            glGetIntegerv(GL_READ_BUFFER, &previous_read_buffer);
            glGetIntegerv(GL_PACK_ALIGNMENT, &previous_pack_alignment);

            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            glReadBuffer(GL_FRONT);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glFinish();
            glReadPixels(
                capture_x,
                capture_y_bottom,
                capture_width,
                capture_height,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                pixels.data());
            const GLenum read_error = glGetError();

            glPixelStorei(GL_PACK_ALIGNMENT, previous_pack_alignment);
            glReadBuffer(static_cast<GLenum>(previous_read_buffer));
            glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(previous_read_fbo));

            if (read_error != GL_NO_ERROR)
                return std::unexpected("glReadPixels failed while capturing the live viewport");

            const size_t row_bytes = static_cast<size_t>(capture_width) * 4u;
            std::vector<uint8_t> flipped(pixels.size());
            for (int row = 0; row < capture_height; ++row) {
                const size_t src_offset = static_cast<size_t>(capture_height - 1 - row) * row_bytes;
                const size_t dst_offset = static_cast<size_t>(row) * row_bytes;
                std::copy_n(pixels.data() + src_offset, row_bytes, flipped.data() + dst_offset);
            }

            return mcp::encode_pixels_to_base64(
                flipped.data(),
                capture_width,
                capture_height,
                4,
                width,
                height);
        }

        json selection_state_json(core::Scene& scene, const int max_indices = 100000) {
            auto mask = scene.getSelectionMask();
            if (!mask)
                return json{{"selected_count", 0}, {"indices", json::array()}, {"truncated", false}};

            auto mask_vec = mask->to_vector_uint8();

            int64_t count = 0;
            std::vector<int64_t> indices;
            for (size_t i = 0; i < mask_vec.size(); ++i) {
                if (mask_vec[i] == 0)
                    continue;
                ++count;
                if (static_cast<int>(indices.size()) < max_indices)
                    indices.push_back(static_cast<int64_t>(i));
            }

            return json{
                {"selected_count", count},
                {"indices", indices},
                {"truncated", count > static_cast<int64_t>(indices.size())}};
        }

        json vec3_to_json(const glm::vec3& value) {
            return json::array({value.x, value.y, value.z});
        }

        json mat4_to_json(const glm::mat4& value) {
            return json::array({
                json::array({value[0][0], value[1][0], value[2][0], value[3][0]}),
                json::array({value[0][1], value[1][1], value[2][1], value[3][1]}),
                json::array({value[0][2], value[1][2], value[2][2], value[3][2]}),
                json::array({value[0][3], value[1][3], value[2][3], value[3][3]}),
            });
        }

        const char* node_type_to_string(const core::NodeType type) {
            switch (type) {
            case core::NodeType::SPLAT:
                return "splat";
            case core::NodeType::POINTCLOUD:
                return "pointcloud";
            case core::NodeType::GROUP:
                return "group";
            case core::NodeType::CROPBOX:
                return "crop_box";
            case core::NodeType::ELLIPSOID:
                return "ellipsoid";
            case core::NodeType::DATASET:
                return "dataset";
            case core::NodeType::CAMERA_GROUP:
                return "camera_group";
            case core::NodeType::CAMERA:
                return "camera";
            case core::NodeType::IMAGE_GROUP:
                return "image_group";
            case core::NodeType::IMAGE:
                return "image";
            case core::NodeType::MESH:
                return "mesh";
            case core::NodeType::KEYFRAME_GROUP:
                return "keyframe_group";
            case core::NodeType::KEYFRAME:
                return "keyframe";
            }
            return "unknown";
        }

        TransformComponents decompose_transform(const glm::mat4& matrix) {
            return vis::cap::decomposeTransform(matrix);
        }

        glm::mat4 compose_transform(const TransformComponents& components) {
            return vis::cap::composeTransform(components);
        }

        int64_t selected_gaussian_count(const core::Scene& scene) {
            const auto mask = scene.getSelectionMask();
            if (!mask || !mask->is_valid())
                return 0;
            return static_cast<int64_t>(mask->count_nonzero());
        }

        json node_summary_json(const core::Scene& scene, const core::SceneNode& node) {
            json result{
                {"name", node.name},
                {"type", node_type_to_string(node.type)},
                {"visible", static_cast<bool>(node.visible)},
                {"locked", static_cast<bool>(node.locked)},
                {"gaussian_count", node.gaussian_count},
            };

            if (node.parent_id != core::NULL_NODE) {
                if (const auto* const parent = scene.getNodeById(node.parent_id)) {
                    result["parent"] = parent->name;
                }
            }

            if (node.type == core::NodeType::SPLAT || node.type == core::NodeType::POINTCLOUD) {
                result["has_crop_box"] = scene.getCropBoxForSplat(node.id) != core::NULL_NODE;
                result["has_ellipsoid"] = scene.getEllipsoidForSplat(node.id) != core::NULL_NODE;
            }

            return result;
        }

        json transform_info_json(const core::Scene& scene, const core::SceneNode& node) {
            const glm::mat4 local = scene.getNodeTransform(node.name);
            const glm::mat4 world = scene.getWorldTransform(node.id);
            const auto local_components = decompose_transform(local);
            const auto world_components = decompose_transform(world);

            return json{
                {"name", node.name},
                {"type", node_type_to_string(node.type)},
                {"local", json{
                              {"translation", vec3_to_json(local_components.translation)},
                              {"rotation", vec3_to_json(local_components.rotation)},
                              {"scale", vec3_to_json(local_components.scale)},
                              {"matrix", mat4_to_json(local)},
                          }},
                {"world", json{
                              {"translation", vec3_to_json(world_components.translation)},
                              {"rotation", vec3_to_json(world_components.rotation)},
                              {"scale", vec3_to_json(world_components.scale)},
                              {"matrix", mat4_to_json(world)},
                          }},
            };
        }

        json selection_result_json(const vis::SceneManager& scene_manager, const vis::SelectionResult& result) {
            if (!result.success)
                return json{{"error", result.error}};

            return json{
                {"success", true},
                {"affected_count", static_cast<int64_t>(result.affected_count)},
                {"selected_count", selected_gaussian_count(scene_manager.getScene())},
            };
        }

        std::optional<std::string> optional_string_arg(const json& args, const char* key) {
            if (!args.contains(key) || args[key].is_null())
                return std::nullopt;
            return args[key].get<std::string>();
        }

        std::expected<std::optional<glm::vec3>, std::string> optional_vec3_arg(const json& args, const char* key) {
            if (!args.contains(key) || args[key].is_null())
                return std::optional<glm::vec3>{};

            const auto& value = args[key];
            if (!value.is_array() || value.size() != 3)
                return std::unexpected(std::string("Field '") + key + "' must be a 3-element array");

            return glm::vec3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
        }

        void show_python_console() {
            core::events::cmd::ShowWindow{.window_name = "python_console", .show = true}.emit();
        }

        json text_payload_json(std::string text, const int max_chars, const bool tail = true) {
            if (max_chars >= 0 && static_cast<int>(text.size()) > max_chars) {
                const size_t keep = static_cast<size_t>(max_chars);
                if (tail) {
                    text = text.substr(text.size() - keep);
                } else {
                    text.resize(keep);
                }
                return json{
                    {"text", std::move(text)},
                    {"truncated", true},
                };
            }

            return json{
                {"text", std::move(text)},
                {"truncated", false},
            };
        }

        struct EditorOutputObservation {
            std::string text;
            int64_t total_chars = 0;
            bool running = false;
            bool completed = false;
            bool timed_out = false;
            bool saw_output = false;
        };

        json editor_output_json(const std::string& text, const int max_chars, const bool tail = true) {
            auto payload = text_payload_json(text, max_chars, tail);
            payload["total_chars"] = static_cast<int64_t>(text.size());
            return payload;
        }

        EditorOutputObservation observe_editor_output(vis::gui::panels::PythonConsoleState& console,
                                                      const bool wait_for_completion,
                                                      const bool wait_for_output,
                                                      const int timeout_ms) {
            using Clock = std::chrono::steady_clock;

            constexpr auto POLL_INTERVAL = std::chrono::milliseconds(25);
            const int bounded_timeout_ms = std::max(timeout_ms, 0);
            const auto deadline = Clock::now() + std::chrono::milliseconds(bounded_timeout_ms);

            EditorOutputObservation observation;
            while (true) {
                observation.text = console.getOutputText();
                observation.total_chars = static_cast<int64_t>(observation.text.size());
                observation.running = console.isScriptRunning();
                observation.completed = !observation.running;
                observation.saw_output = !observation.text.empty();

                const bool output_ready = !wait_for_output || observation.saw_output || observation.completed;
                const bool completion_ready = !wait_for_completion || observation.completed;
                if (output_ready && completion_ready) {
                    observation.timed_out = false;
                    return observation;
                }

                if (bounded_timeout_ms == 0 || Clock::now() >= deadline) {
                    observation.timed_out = true;
                    return observation;
                }

                std::this_thread::sleep_for(POLL_INTERVAL);
            }
        }

        json editor_output_response_json(vis::gui::panels::PythonConsoleState& console,
                                         const bool wait_for_completion,
                                         const bool wait_for_output,
                                         const int timeout_ms,
                                         const int output_max_chars,
                                         const bool output_tail) {
            auto observation = observe_editor_output(
                console, wait_for_completion, wait_for_output, timeout_ms);

            json response{
                {"running", observation.running},
                {"completed", observation.completed},
                {"timed_out", observation.timed_out},
                {"saw_output", observation.saw_output},
            };
            response["output"] = editor_output_json(observation.text, output_max_chars, output_tail);
            return response;
        }

        std::expected<std::vector<std::string>, std::string> resolve_transform_targets(
            const vis::SceneManager& scene_manager,
            const std::optional<std::string>& requested_node) {
            return vis::cap::resolveTransformTargets(scene_manager, requested_node);
        }

        std::expected<std::vector<std::string>, std::string> resolve_editable_transform_targets(
            const vis::SceneManager& scene_manager,
            const std::optional<std::string>& requested_node) {
            auto resolved = vis::cap::resolveEditableTransformSelection(
                scene_manager, requested_node, vis::cap::TransformTargetPolicy::AllowEditableSubset);
            if (!resolved)
                return std::unexpected(resolved.error());
            return resolved->node_names;
        }

        std::expected<core::NodeId, std::string> resolve_cropbox_parent_id(
            const vis::SceneManager& scene_manager,
            const std::optional<std::string>& requested_node) {
            return vis::cap::resolveCropBoxParentId(scene_manager, requested_node);
        }

        std::expected<core::NodeId, std::string> resolve_cropbox_id(
            const vis::SceneManager& scene_manager,
            const std::optional<std::string>& requested_node) {
            return vis::cap::resolveCropBoxId(scene_manager, requested_node);
        }

        json crop_box_info_json(const vis::SceneManager& scene_manager,
                                const vis::RenderingManager* rendering_manager,
                                const core::NodeId cropbox_id) {
            const auto& scene = scene_manager.getScene();
            const auto* const node = scene.getNodeById(cropbox_id);
            assert(node && node->cropbox);

            const auto components = decompose_transform(scene.getNodeTransform(node->name));
            json crop_box{
                {"node", node->name},
                {"type", node_type_to_string(node->type)},
                {"min", vec3_to_json(node->cropbox->min)},
                {"max", vec3_to_json(node->cropbox->max)},
                {"inverse", node->cropbox->inverse},
                {"enabled", node->cropbox->enabled},
                {"translation", vec3_to_json(components.translation)},
                {"rotation", vec3_to_json(components.rotation)},
                {"scale", vec3_to_json(components.scale)},
                {"local_matrix", mat4_to_json(scene.getNodeTransform(node->name))},
                {"world_matrix", mat4_to_json(scene.getWorldTransform(cropbox_id))},
            };

            if (node->parent_id != core::NULL_NODE) {
                if (const auto* const parent = scene.getNodeById(node->parent_id)) {
                    crop_box["parent"] = parent->name;
                }
            }

            if (rendering_manager) {
                const auto settings = rendering_manager->getSettings();
                crop_box["show"] = settings.show_crop_box;
                crop_box["use"] = settings.use_crop_box;
            }

            return json{{"success", true}, {"crop_box", crop_box}};
        }

        std::expected<core::NodeId, std::string> ensure_cropbox(
            vis::SceneManager& scene_manager,
            vis::RenderingManager* rendering_manager,
            const core::NodeId parent_id) {
            return vis::cap::ensureCropBox(scene_manager, rendering_manager, parent_id);
        }

        std::expected<void, std::string> fit_cropbox_to_parent(
            vis::SceneManager& scene_manager,
            vis::RenderingManager* rendering_manager,
            const core::NodeId cropbox_id,
            const bool use_percentile) {
            return vis::cap::fitCropBoxToParent(scene_manager, rendering_manager, cropbox_id, use_percentile);
        }

        std::expected<void, std::string> reset_cropbox(
            vis::SceneManager& scene_manager,
            vis::RenderingManager* rendering_manager,
            const core::NodeId cropbox_id) {
            return vis::cap::resetCropBox(scene_manager, rendering_manager, cropbox_id);
        }

        json view_info_json(const vis::ViewInfo& info) {
            const json rotation = json::array({
                json::array({info.rotation[0], info.rotation[1], info.rotation[2]}),
                json::array({info.rotation[3], info.rotation[4], info.rotation[5]}),
                json::array({info.rotation[6], info.rotation[7], info.rotation[8]}),
            });

            return json{
                {"success", true},
                {"camera", {
                               {"eye", json::array({info.translation[0], info.translation[1], info.translation[2]})},
                               {"target", json::array({info.pivot[0], info.pivot[1], info.pivot[2]})},
                               {"pivot", json::array({info.pivot[0], info.pivot[1], info.pivot[2]})},
                               {"up", json::array({info.rotation[1], info.rotation[4], info.rotation[7]})},
                               {"forward", json::array({info.rotation[2], info.rotation[5], info.rotation[8]})},
                               {"rotation_matrix", rotation},
                               {"width", info.width},
                               {"height", info.height},
                               {"fov_degrees", info.fov},
                           }},
            };
        }

        json render_settings_json(const vis::RenderSettingsProxy& settings) {
            return json{
                {"success", true},
                {"settings", {
                                 {"focal_length_mm", settings.focal_length_mm},
                                 {"scaling_modifier", settings.scaling_modifier},
                                 {"antialiasing", settings.antialiasing},
                                 {"mip_filter", settings.mip_filter},
                                 {"sh_degree", settings.sh_degree},
                                 {"render_scale", settings.render_scale},
                                 {"show_crop_box", settings.show_crop_box},
                                 {"use_crop_box", settings.use_crop_box},
                                 {"show_ellipsoid", settings.show_ellipsoid},
                                 {"use_ellipsoid", settings.use_ellipsoid},
                                 {"desaturate_unselected", settings.desaturate_unselected},
                                 {"desaturate_cropping", settings.desaturate_cropping},
                                 {"crop_filter_for_selection", settings.crop_filter_for_selection},
                                 {"apply_appearance_correction", settings.apply_appearance_correction},
                                 {"ppisp_mode", settings.ppisp_mode},
                                 {"ppisp", {
                                               {"exposure_offset", settings.ppisp.exposure_offset},
                                               {"vignette_enabled", settings.ppisp.vignette_enabled},
                                               {"vignette_strength", settings.ppisp.vignette_strength},
                                               {"wb_temperature", settings.ppisp.wb_temperature},
                                               {"wb_tint", settings.ppisp.wb_tint},
                                               {"color_red_x", settings.ppisp.color_red_x},
                                               {"color_red_y", settings.ppisp.color_red_y},
                                               {"color_green_x", settings.ppisp.color_green_x},
                                               {"color_green_y", settings.ppisp.color_green_y},
                                               {"color_blue_x", settings.ppisp.color_blue_x},
                                               {"color_blue_y", settings.ppisp.color_blue_y},
                                               {"gamma_multiplier", settings.ppisp.gamma_multiplier},
                                               {"gamma_red", settings.ppisp.gamma_red},
                                               {"gamma_green", settings.ppisp.gamma_green},
                                               {"gamma_blue", settings.ppisp.gamma_blue},
                                               {"crf_toe", settings.ppisp.crf_toe},
                                               {"crf_shoulder", settings.ppisp.crf_shoulder},
                                           }},
                                 {"background_color", json::array({settings.background_color[0], settings.background_color[1], settings.background_color[2]})},
                                 {"show_coord_axes", settings.show_coord_axes},
                                 {"axes_size", settings.axes_size},
                                 {"axes_visibility", json::array({settings.axes_visibility[0], settings.axes_visibility[1], settings.axes_visibility[2]})},
                                 {"show_grid", settings.show_grid},
                                 {"grid_plane", settings.grid_plane},
                                 {"grid_opacity", settings.grid_opacity},
                                 {"point_cloud_mode", settings.point_cloud_mode},
                                 {"voxel_size", settings.voxel_size},
                                 {"show_rings", settings.show_rings},
                                 {"ring_width", settings.ring_width},
                                 {"show_center_markers", settings.show_center_markers},
                                 {"show_camera_frustums", settings.show_camera_frustums},
                                 {"camera_frustum_scale", settings.camera_frustum_scale},
                                 {"train_camera_color", json::array({settings.train_camera_color[0], settings.train_camera_color[1], settings.train_camera_color[2]})},
                                 {"eval_camera_color", json::array({settings.eval_camera_color[0], settings.eval_camera_color[1], settings.eval_camera_color[2]})},
                                 {"show_pivot", settings.show_pivot},
                                 {"split_view_mode", settings.split_view_mode},
                                 {"split_position", settings.split_position},
                                 {"gut", settings.gut},
                                 {"equirectangular", settings.equirectangular},
                                 {"orthographic", settings.orthographic},
                                 {"ortho_scale", settings.ortho_scale},
                                 {"selection_color_committed", json::array({settings.selection_color_committed[0], settings.selection_color_committed[1], settings.selection_color_committed[2]})},
                                 {"selection_color_preview", json::array({settings.selection_color_preview[0], settings.selection_color_preview[1], settings.selection_color_preview[2]})},
                                 {"selection_color_center_marker", json::array({settings.selection_color_center_marker[0], settings.selection_color_center_marker[1], settings.selection_color_center_marker[2]})},
                                 {"depth_clip_enabled", settings.depth_clip_enabled},
                                 {"depth_clip_far", settings.depth_clip_far},
                                 {"mesh_wireframe", settings.mesh_wireframe},
                                 {"mesh_wireframe_color", json::array({settings.mesh_wireframe_color[0], settings.mesh_wireframe_color[1], settings.mesh_wireframe_color[2]})},
                                 {"mesh_wireframe_width", settings.mesh_wireframe_width},
                                 {"mesh_light_dir", json::array({settings.mesh_light_dir[0], settings.mesh_light_dir[1], settings.mesh_light_dir[2]})},
                                 {"mesh_light_intensity", settings.mesh_light_intensity},
                                 {"mesh_ambient", settings.mesh_ambient},
                                 {"mesh_backface_culling", settings.mesh_backface_culling},
                                 {"mesh_shadow_enabled", settings.mesh_shadow_enabled},
                                 {"mesh_shadow_resolution", settings.mesh_shadow_resolution},
                                 {"depth_filter_enabled", settings.depth_filter_enabled},
                                 {"depth_filter_min", json::array({settings.depth_filter_min[0], settings.depth_filter_min[1], settings.depth_filter_min[2]})},
                                 {"depth_filter_max", json::array({settings.depth_filter_max[0], settings.depth_filter_max[1], settings.depth_filter_max[2]})},
                             }},
            };
        }

        std::expected<void, std::string> apply_render_settings_patch(const json& args, vis::RenderSettingsProxy& settings) {
            bool touched = false;

            const auto set_bool = [&args, &touched](const char* key, bool& field) {
                if (args.contains(key)) {
                    field = args[key].get<bool>();
                    touched = true;
                }
            };

            const auto set_int = [&args, &touched](const char* key, int& field) {
                if (args.contains(key)) {
                    field = args[key].get<int>();
                    touched = true;
                }
            };

            const auto set_float = [&args, &touched](const char* key, float& field) {
                if (args.contains(key)) {
                    field = args[key].get<float>();
                    touched = true;
                }
            };

            const auto set_vec3 = [&args, &touched](const char* key,
                                                    std::array<float, 3>& field) -> std::expected<void, std::string> {
                if (!args.contains(key))
                    return {};
                const auto vec = optional_vec3_arg(args, key);
                if (!vec)
                    return std::unexpected(vec.error());
                if (!vec->has_value())
                    return std::unexpected(std::string("Field '") + key + "' must be provided");
                field = {(**vec).x, (**vec).y, (**vec).z};
                touched = true;
                return {};
            };

            const auto set_bool3 = [&args, &touched](const char* key,
                                                     std::array<bool, 3>& field) -> std::expected<void, std::string> {
                if (!args.contains(key))
                    return {};
                const auto& value = args[key];
                if (!value.is_array() || value.size() != 3)
                    return std::unexpected(std::string("Field '") + key + "' must be a 3-element array");
                field = {value[0].get<bool>(), value[1].get<bool>(), value[2].get<bool>()};
                touched = true;
                return {};
            };

            set_float("focal_length_mm", settings.focal_length_mm);
            set_float("scaling_modifier", settings.scaling_modifier);
            set_bool("antialiasing", settings.antialiasing);
            set_bool("mip_filter", settings.mip_filter);
            set_int("sh_degree", settings.sh_degree);
            set_float("render_scale", settings.render_scale);
            set_bool("show_crop_box", settings.show_crop_box);
            set_bool("use_crop_box", settings.use_crop_box);
            set_bool("show_ellipsoid", settings.show_ellipsoid);
            set_bool("use_ellipsoid", settings.use_ellipsoid);
            set_bool("desaturate_unselected", settings.desaturate_unselected);
            set_bool("desaturate_cropping", settings.desaturate_cropping);
            set_bool("crop_filter_for_selection", settings.crop_filter_for_selection);
            set_bool("apply_appearance_correction", settings.apply_appearance_correction);
            set_int("ppisp_mode", settings.ppisp_mode);
            set_bool("show_coord_axes", settings.show_coord_axes);
            set_float("axes_size", settings.axes_size);
            set_bool("show_grid", settings.show_grid);
            set_int("grid_plane", settings.grid_plane);
            set_float("grid_opacity", settings.grid_opacity);
            set_bool("point_cloud_mode", settings.point_cloud_mode);
            set_float("voxel_size", settings.voxel_size);
            set_bool("show_rings", settings.show_rings);
            set_float("ring_width", settings.ring_width);
            set_bool("show_center_markers", settings.show_center_markers);
            set_bool("show_camera_frustums", settings.show_camera_frustums);
            set_float("camera_frustum_scale", settings.camera_frustum_scale);
            set_bool("show_pivot", settings.show_pivot);
            set_int("split_view_mode", settings.split_view_mode);
            set_float("split_position", settings.split_position);
            set_bool("gut", settings.gut);
            set_bool("equirectangular", settings.equirectangular);
            set_bool("orthographic", settings.orthographic);
            set_float("ortho_scale", settings.ortho_scale);
            set_bool("depth_clip_enabled", settings.depth_clip_enabled);
            set_float("depth_clip_far", settings.depth_clip_far);
            set_bool("mesh_wireframe", settings.mesh_wireframe);
            set_float("mesh_wireframe_width", settings.mesh_wireframe_width);
            set_float("mesh_light_intensity", settings.mesh_light_intensity);
            set_float("mesh_ambient", settings.mesh_ambient);
            set_bool("mesh_backface_culling", settings.mesh_backface_culling);
            set_bool("mesh_shadow_enabled", settings.mesh_shadow_enabled);
            set_int("mesh_shadow_resolution", settings.mesh_shadow_resolution);
            set_bool("depth_filter_enabled", settings.depth_filter_enabled);
            if (auto result = set_vec3("background_color", settings.background_color); !result)
                return result;
            if (auto result = set_bool3("axes_visibility", settings.axes_visibility); !result)
                return result;
            if (auto result = set_vec3("train_camera_color", settings.train_camera_color); !result)
                return result;
            if (auto result = set_vec3("eval_camera_color", settings.eval_camera_color); !result)
                return result;
            if (auto result = set_vec3("selection_color_committed", settings.selection_color_committed); !result)
                return result;
            if (auto result = set_vec3("selection_color_preview", settings.selection_color_preview); !result)
                return result;
            if (auto result = set_vec3("selection_color_center_marker", settings.selection_color_center_marker); !result)
                return result;
            if (auto result = set_vec3("mesh_wireframe_color", settings.mesh_wireframe_color); !result)
                return result;
            if (auto result = set_vec3("mesh_light_dir", settings.mesh_light_dir); !result)
                return result;
            if (auto result = set_vec3("depth_filter_min", settings.depth_filter_min); !result)
                return result;
            if (auto result = set_vec3("depth_filter_max", settings.depth_filter_max); !result)
                return result;

            if (args.contains("ppisp_exposure")) {
                settings.ppisp.exposure_offset = args["ppisp_exposure"].get<float>();
                touched = true;
            }

            if (args.contains("ppisp")) {
                const auto& ppisp = args["ppisp"];
                if (!ppisp.is_object())
                    return std::unexpected("Field 'ppisp' must be an object");

                const auto set_ppisp_bool = [&ppisp, &touched](const char* key, bool& field) {
                    if (ppisp.contains(key)) {
                        field = ppisp[key].get<bool>();
                        touched = true;
                    }
                };
                const auto set_ppisp_float = [&ppisp, &touched](const char* key, float& field) {
                    if (ppisp.contains(key)) {
                        field = ppisp[key].get<float>();
                        touched = true;
                    }
                };

                set_ppisp_float("exposure_offset", settings.ppisp.exposure_offset);
                set_ppisp_bool("vignette_enabled", settings.ppisp.vignette_enabled);
                set_ppisp_float("vignette_strength", settings.ppisp.vignette_strength);
                set_ppisp_float("wb_temperature", settings.ppisp.wb_temperature);
                set_ppisp_float("wb_tint", settings.ppisp.wb_tint);
                set_ppisp_float("color_red_x", settings.ppisp.color_red_x);
                set_ppisp_float("color_red_y", settings.ppisp.color_red_y);
                set_ppisp_float("color_green_x", settings.ppisp.color_green_x);
                set_ppisp_float("color_green_y", settings.ppisp.color_green_y);
                set_ppisp_float("color_blue_x", settings.ppisp.color_blue_x);
                set_ppisp_float("color_blue_y", settings.ppisp.color_blue_y);
                set_ppisp_float("gamma_multiplier", settings.ppisp.gamma_multiplier);
                set_ppisp_float("gamma_red", settings.ppisp.gamma_red);
                set_ppisp_float("gamma_green", settings.ppisp.gamma_green);
                set_ppisp_float("gamma_blue", settings.ppisp.gamma_blue);
                set_ppisp_float("crf_toe", settings.ppisp.crf_toe);
                set_ppisp_float("crf_shoulder", settings.ppisp.crf_shoulder);
            }

            if (!touched)
                return std::unexpected("No render settings fields were provided");

            return {};
        }

        json history_json() {
            auto& history = vis::op::undoHistory();
            return json{
                {"success", true},
                {"can_undo", history.canUndo()},
                {"can_redo", history.canRedo()},
                {"undo_count", static_cast<int64_t>(history.undoCount())},
                {"redo_count", static_cast<int64_t>(history.redoCount())},
                {"undo_name", history.undoName()},
                {"redo_name", history.redoName()},
            };
        }

        std::expected<void, std::string> prepare_delete_operator(vis::Visualizer& viewer,
                                                                 const json& /*args*/,
                                                                 vis::op::OperatorProperties& props) {
            auto* const scene_manager = viewer.getSceneManager();
            if (!scene_manager)
                return std::unexpected("Scene manager not initialized");

            if (const auto name = props.get<std::string>("name"); name && !name->empty()) {
                if (!scene_manager->getScene().getNode(*name))
                    return std::unexpected("Node not found: " + *name);
                return {};
            }

            if (!scene_manager->hasSelectedNode())
                return std::unexpected("No node specified and no node selected");

            return {};
        }

        json delete_operator_result(vis::Visualizer& /*viewer*/,
                                    const json& args,
                                    const vis::op::OperatorProperties& props,
                                    const vis::op::OperatorReturnValue& /*result*/) {
            const auto removed_nodes = props.get<std::vector<std::string>>("resolved_node_names").value_or(std::vector<std::string>{});
            const bool keep_children = props.get_or<bool>("keep_children", false);

            json payload{
                {"success", true},
                {"removed_count", removed_nodes.size()},
                {"removed_nodes", removed_nodes},
                {"keep_children", keep_children},
            };
            if (args.contains("name") && args["name"].is_string())
                payload["removed"] = args["name"].get<std::string>();
            return payload;
        }

        std::expected<void, std::string> prepare_transform_operator(vis::Visualizer& viewer,
                                                                    const json& /*args*/,
                                                                    vis::op::OperatorProperties& props) {
            auto* const scene_manager = viewer.getSceneManager();
            if (!scene_manager)
                return std::unexpected("Scene manager not initialized");

            std::optional<std::string> requested_node;
            if (const auto node = props.get<std::string>("node"); node && !node->empty())
                requested_node = *node;

            auto targets = resolve_editable_transform_targets(*scene_manager, requested_node);
            if (!targets)
                return std::unexpected(targets.error());

            props.set("resolved_node_names", *targets);
            return {};
        }

        std::expected<void, std::string> prepare_transform_set_operator(vis::Visualizer& viewer,
                                                                        const json& args,
                                                                        vis::op::OperatorProperties& props) {
            if (!props.has("translation") && !props.has("rotation") && !props.has("scale")) {
                return std::unexpected("At least one of translation, rotation, or scale must be provided");
            }
            return prepare_transform_operator(viewer, args, props);
        }

        json transform_operator_result(vis::Visualizer& viewer,
                                       const json& /*args*/,
                                       const vis::op::OperatorProperties& props,
                                       const vis::op::OperatorReturnValue& /*result*/) {
            auto* const scene_manager = viewer.getSceneManager();
            if (!scene_manager)
                return json{{"error", "Scene manager not initialized"}};

            const auto resolved_nodes = props.get<std::vector<std::string>>("resolved_node_names").value_or(std::vector<std::string>{});

            json nodes = json::array();
            for (const auto& name : resolved_nodes) {
                if (const auto* const node = scene_manager->getScene().getNode(name))
                    nodes.push_back(transform_info_json(scene_manager->getScene(), *node));
            }

            return json{{"success", true}, {"count", nodes.size()}, {"nodes", nodes}};
        }

        std::expected<void, std::string> prepare_scene_select_node_operator(
            vis::Visualizer& viewer,
            const json& /*args*/,
            vis::op::OperatorProperties& props) {
            auto* const scene_manager = viewer.getSceneManager();
            if (!scene_manager)
                return std::unexpected("Scene manager not initialized");

            const auto name = props.get<std::string>("name");
            if (!name || name->empty())
                return std::unexpected("Field 'name' must be provided");
            if (!scene_manager->getScene().getNode(*name))
                return std::unexpected("Node not found: " + *name);

            const auto mode = props.get_or<std::string>("mode", "replace");
            if (mode != "replace" && mode != "add")
                return std::unexpected("Unsupported node selection mode: " + mode);

            return {};
        }

        json scene_select_node_result(vis::Visualizer& viewer,
                                      const json& /*args*/,
                                      const vis::op::OperatorProperties& /*props*/,
                                      const vis::op::OperatorReturnValue& /*result*/) {
            auto* const scene_manager = viewer.getSceneManager();
            if (!scene_manager)
                return json{{"error", "Scene manager not initialized"}};

            const auto& scene = scene_manager->getScene();
            json nodes = json::array();
            for (const auto& selected_name : scene_manager->getSelectedNodeNames()) {
                if (const auto* const node = scene.getNode(selected_name))
                    nodes.push_back(node_summary_json(scene, *node));
            }

            return json{{"success", true}, {"count", nodes.size()}, {"nodes", nodes}};
        }

        std::expected<void, std::string> prepare_crop_box_add_operator(
            vis::Visualizer& viewer,
            const json& /*args*/,
            vis::op::OperatorProperties& props) {
            auto* const scene_manager = viewer.getSceneManager();
            if (!scene_manager)
                return std::unexpected("Scene manager not initialized");

            auto parent_id = vis::cap::resolveCropBoxParentId(
                *scene_manager, props.get<std::string>("node"));
            if (!parent_id)
                return std::unexpected(parent_id.error());
            return {};
        }

        std::expected<void, std::string> prepare_crop_box_set_operator(
            vis::Visualizer& viewer,
            const json& /*args*/,
            vis::op::OperatorProperties& props) {
            auto* const scene_manager = viewer.getSceneManager();
            if (!scene_manager)
                return std::unexpected("Scene manager not initialized");

            auto cropbox_id = vis::cap::resolveCropBoxId(*scene_manager, props.get<std::string>("node"));
            if (!cropbox_id)
                return std::unexpected(cropbox_id.error());

            if (!props.has("min") && !props.has("max") &&
                !props.has("translation") && !props.has("rotation") && !props.has("scale") &&
                !props.has("inverse") && !props.has("enabled") && !props.has("show") && !props.has("use")) {
                return std::unexpected("No crop box fields were provided");
            }
            return {};
        }

        std::expected<void, std::string> prepare_crop_box_target_operator(
            vis::Visualizer& viewer,
            const json& /*args*/,
            vis::op::OperatorProperties& props) {
            auto* const scene_manager = viewer.getSceneManager();
            if (!scene_manager)
                return std::unexpected("Scene manager not initialized");

            auto cropbox_id = vis::cap::resolveCropBoxId(*scene_manager, props.get<std::string>("node"));
            if (!cropbox_id)
                return std::unexpected(cropbox_id.error());
            return {};
        }

        json crop_box_operator_result(vis::Visualizer& viewer,
                                      const json& /*args*/,
                                      const vis::op::OperatorProperties& props,
                                      const vis::op::OperatorReturnValue& /*result*/) {
            auto* const scene_manager = viewer.getSceneManager();
            auto* const rendering_manager = viewer.getRenderingManager();
            if (!scene_manager)
                return json{{"error", "Scene manager not initialized"}};

            const auto cropbox_id = props.get<core::NodeId>("resolved_cropbox_id");
            if (!cropbox_id)
                return json{{"error", "Crop box result did not resolve a target"}};

            return crop_box_info_json(*scene_manager, rendering_manager, *cropbox_id);
        }

        json ellipsoid_info_json(const vis::SceneManager& scene_manager,
                                 const vis::RenderingManager* rendering_manager,
                                 const core::NodeId ellipsoid_id);

        std::expected<void, std::string> prepare_ellipsoid_add_operator(
            vis::Visualizer& viewer,
            const json& /*args*/,
            vis::op::OperatorProperties& props) {
            auto* const scene_manager = viewer.getSceneManager();
            if (!scene_manager)
                return std::unexpected("Scene manager not initialized");

            auto parent_id = vis::cap::resolveEllipsoidParentId(
                *scene_manager, props.get<std::string>("node"));
            if (!parent_id)
                return std::unexpected(parent_id.error());
            return {};
        }

        std::expected<void, std::string> prepare_ellipsoid_set_operator(
            vis::Visualizer& viewer,
            const json& /*args*/,
            vis::op::OperatorProperties& props) {
            auto* const scene_manager = viewer.getSceneManager();
            if (!scene_manager)
                return std::unexpected("Scene manager not initialized");

            auto ellipsoid_id = vis::cap::resolveEllipsoidId(*scene_manager, props.get<std::string>("node"));
            if (!ellipsoid_id)
                return std::unexpected(ellipsoid_id.error());

            if (!props.has("radii") && !props.has("translation") && !props.has("rotation") &&
                !props.has("scale") && !props.has("inverse") && !props.has("enabled") &&
                !props.has("show") && !props.has("use")) {
                return std::unexpected("No ellipsoid fields were provided");
            }
            return {};
        }

        std::expected<void, std::string> prepare_ellipsoid_target_operator(
            vis::Visualizer& viewer,
            const json& /*args*/,
            vis::op::OperatorProperties& props) {
            auto* const scene_manager = viewer.getSceneManager();
            if (!scene_manager)
                return std::unexpected("Scene manager not initialized");

            auto ellipsoid_id = vis::cap::resolveEllipsoidId(*scene_manager, props.get<std::string>("node"));
            if (!ellipsoid_id)
                return std::unexpected(ellipsoid_id.error());
            return {};
        }

        json ellipsoid_operator_result(vis::Visualizer& viewer,
                                       const json& /*args*/,
                                       const vis::op::OperatorProperties& props,
                                       const vis::op::OperatorReturnValue& /*result*/) {
            auto* const scene_manager = viewer.getSceneManager();
            auto* const rendering_manager = viewer.getRenderingManager();
            if (!scene_manager)
                return json{{"error", "Scene manager not initialized"}};

            const auto ellipsoid_id = props.get<core::NodeId>("resolved_ellipsoid_id");
            if (!ellipsoid_id)
                return json{{"error", "Ellipsoid result did not resolve a target"}};

            return ellipsoid_info_json(*scene_manager, rendering_manager, *ellipsoid_id);
        }

        json camera_node_json(const core::Scene& scene, const core::SceneNode& node) {
            assert(node.camera);

            std::vector<float> position = node.camera->cam_position().to(core::Device::CPU).to(core::DataType::Float32).contiguous().to_vector();
            if (position.size() < 3)
                position = {0.0f, 0.0f, 0.0f};

            const auto [focal_x, focal_y, center_x, center_y] = node.camera->get_intrinsics();

            json camera{
                {"name", node.name},
                {"uid", node.camera_uid},
                {"camera_id", node.camera->camera_id()},
                {"image_name", node.camera->image_name()},
                {"image_path", core::path_to_utf8(node.camera->image_path())},
                {"mask_path", core::path_to_utf8(node.camera->mask_path())},
                {"camera_width", node.camera->camera_width()},
                {"camera_height", node.camera->camera_height()},
                {"image_width", node.camera->image_width()},
                {"image_height", node.camera->image_height()},
                {"focal_x", focal_x},
                {"focal_y", focal_y},
                {"center_x", center_x},
                {"center_y", center_y},
                {"fov_x_radians", node.camera->FoVx()},
                {"fov_y_radians", node.camera->FoVy()},
                {"position", json::array({position[0], position[1], position[2]})},
                {"training_enabled", node.training_enabled},
                {"visible", static_cast<bool>(node.visible)},
            };

            if (node.parent_id != core::NULL_NODE) {
                if (const auto* const parent = scene.getNodeById(node.parent_id))
                    camera["parent"] = parent->name;
            }

            return camera;
        }

        json dataset_info_json(const vis::SceneManager& scene_manager) {
            const auto info = scene_manager.getSceneInfo();
            const auto& scene = scene_manager.getScene();

            int64_t total_cameras = 0;
            int64_t active_cameras = 0;
            int64_t masked_cameras = 0;
            json camera_groups = json::array();
            for (const auto* const node : scene.getNodes()) {
                if (!node)
                    continue;
                if (node->type == core::NodeType::CAMERA && node->camera) {
                    ++total_cameras;
                    if (node->training_enabled)
                        ++active_cameras;
                    if (node->camera->has_mask())
                        ++masked_cameras;
                } else if (node->type == core::NodeType::CAMERA_GROUP) {
                    camera_groups.push_back(node_summary_json(scene, *node));
                }
            }

            return json{
                {"success", true},
                {"dataset", {
                                {"path", core::path_to_utf8(scene_manager.getDatasetPath())},
                                {"source_type", info.source_type},
                                {"source_path", core::path_to_utf8(info.source_path)},
                                {"has_model", info.has_model},
                                {"num_gaussians", static_cast<int64_t>(info.num_gaussians)},
                                {"num_nodes", static_cast<int64_t>(info.num_nodes)},
                                {"camera_count", total_cameras},
                                {"active_camera_count", active_cameras},
                                {"masked_camera_count", masked_cameras},
                                {"camera_groups", camera_groups},
                            }},
            };
        }

        json ellipsoid_info_json(const vis::SceneManager& scene_manager,
                                 const vis::RenderingManager* rendering_manager,
                                 const core::NodeId ellipsoid_id) {
            const auto& scene = scene_manager.getScene();
            const auto* const node = scene.getNodeById(ellipsoid_id);
            assert(node && node->ellipsoid);

            const auto components = decompose_transform(scene.getNodeTransform(node->name));
            json ellipsoid{
                {"node", node->name},
                {"type", node_type_to_string(node->type)},
                {"radii", vec3_to_json(node->ellipsoid->radii)},
                {"inverse", node->ellipsoid->inverse},
                {"enabled", node->ellipsoid->enabled},
                {"translation", vec3_to_json(components.translation)},
                {"rotation", vec3_to_json(components.rotation)},
                {"scale", vec3_to_json(components.scale)},
                {"local_matrix", mat4_to_json(scene.getNodeTransform(node->name))},
                {"world_matrix", mat4_to_json(scene.getWorldTransform(ellipsoid_id))},
            };

            if (node->parent_id != core::NULL_NODE) {
                if (const auto* const parent = scene.getNodeById(node->parent_id))
                    ellipsoid["parent"] = parent->name;
            }

            if (rendering_manager) {
                const auto settings = rendering_manager->getSettings();
                ellipsoid["show"] = settings.show_ellipsoid;
                ellipsoid["use"] = settings.use_ellipsoid;
            }

            return json{{"success", true}, {"ellipsoid", ellipsoid}};
        }

        std::expected<core::NodeId, std::string> resolve_ellipsoid_parent_id(
            const vis::SceneManager& scene_manager,
            const std::optional<std::string>& requested_node) {
            return vis::cap::resolveEllipsoidParentId(scene_manager, requested_node);
        }

        std::expected<core::NodeId, std::string> resolve_ellipsoid_id(
            const vis::SceneManager& scene_manager,
            const std::optional<std::string>& requested_node) {
            return vis::cap::resolveEllipsoidId(scene_manager, requested_node);
        }

        std::expected<core::NodeId, std::string> ensure_ellipsoid(
            vis::SceneManager& scene_manager,
            vis::RenderingManager* rendering_manager,
            const core::NodeId parent_id) {
            return vis::cap::ensureEllipsoid(scene_manager, rendering_manager, parent_id);
        }

        std::expected<void, std::string> fit_ellipsoid_to_parent(
            vis::SceneManager& scene_manager,
            vis::RenderingManager* rendering_manager,
            const core::NodeId ellipsoid_id,
            const bool use_percentile) {
            return vis::cap::fitEllipsoidToParent(scene_manager, rendering_manager, ellipsoid_id, use_percentile);
        }

        std::expected<void, std::string> reset_ellipsoid(
            vis::SceneManager& scene_manager,
            vis::RenderingManager* rendering_manager,
            const core::NodeId ellipsoid_id) {
            return vis::cap::resetEllipsoid(scene_manager, rendering_manager, ellipsoid_id);
        }

        void collect_exportable_splats(const core::Scene& scene,
                                       const core::SceneNode& node,
                                       std::vector<std::string>& out,
                                       std::unordered_set<std::string>& seen) {
            if (node.type == core::NodeType::SPLAT && node.model) {
                if (seen.insert(node.name).second)
                    out.push_back(node.name);
                return;
            }

            for (const auto child_id : node.children) {
                const auto* const child = scene.getNodeById(child_id);
                if (child)
                    collect_exportable_splats(scene, *child, out, seen);
            }
        }

        std::expected<std::vector<std::string>, std::string> resolve_export_nodes(
            const vis::SceneManager& scene_manager,
            const json& args) {
            const auto& scene = scene_manager.getScene();

            std::vector<std::string> requested;
            if (args.contains("nodes")) {
                const auto& nodes = args["nodes"];
                if (!nodes.is_array())
                    return std::unexpected("Field 'nodes' must be an array of node names");
                requested.reserve(nodes.size());
                for (const auto& item : nodes)
                    requested.push_back(item.get<std::string>());
            } else if (const auto node = optional_string_arg(args, "node")) {
                requested.push_back(*node);
            } else {
                requested = scene_manager.getSelectedNodeNames();
                if (requested.empty()) {
                    const auto& training_name = scene.getTrainingModelNodeName();
                    if (!training_name.empty())
                        requested.push_back(training_name);
                }
                if (requested.empty()) {
                    const auto* node = find_first_visible_splat_node(scene);
                    if (node)
                        requested.push_back(node->name);
                }
            }

            if (requested.empty())
                return std::unexpected("No exportable node specified and no suitable selection found");

            std::vector<std::string> export_nodes;
            std::unordered_set<std::string> seen;
            for (const auto& name : requested) {
                const auto* const node = scene.getNode(name);
                if (!node)
                    return std::unexpected("Node not found: " + name);
                collect_exportable_splats(scene, *node, export_nodes, seen);
            }

            if (export_nodes.empty())
                return std::unexpected("The requested node set does not contain any splat nodes");

            return export_nodes;
        }

        void truncate_sh_degree(core::SplatData& splat, const int target_degree) {
            if (target_degree >= splat.get_max_sh_degree())
                return;

            if (target_degree == 0) {
                splat.shN() = core::Tensor{};
            } else {
                const size_t keep_coeffs = static_cast<size_t>((target_degree + 1) * (target_degree + 1) - 1);
                auto& sh_n = splat.shN();
                if (sh_n.is_valid() && sh_n.ndim() >= 2 && sh_n.shape()[1] > keep_coeffs) {
                    if (sh_n.ndim() == 3) {
                        sh_n = sh_n.slice(1, 0, static_cast<int64_t>(keep_coeffs)).contiguous();
                    } else {
                        constexpr size_t channels = 3;
                        sh_n = sh_n.slice(1, 0, static_cast<int64_t>(keep_coeffs * channels)).contiguous();
                    }
                }
            }

            splat.set_max_sh_degree(target_degree);
            splat.set_active_sh_degree(target_degree);
        }

        std::expected<void, std::string> export_scene_nodes(const vis::SceneManager& scene_manager,
                                                            const std::vector<std::string>& node_names,
                                                            const core::ExportFormat format,
                                                            const std::filesystem::path& path,
                                                            const int sh_degree) {
            const auto& scene = scene_manager.getScene();
            std::vector<std::pair<const core::SplatData*, glm::mat4>> splats;
            splats.reserve(node_names.size());
            for (const auto& name : node_names) {
                const auto* const node = scene.getNode(name);
                if (node && node->type == core::NodeType::SPLAT && node->model) {
                    splats.emplace_back(node->model.get(), scene.getWorldTransform(node->id));
                }
            }

            if (splats.empty())
                return std::unexpected("The requested node set does not contain any splat nodes");

            auto merged = core::Scene::mergeSplatsWithTransforms(splats);
            if (!merged)
                return std::unexpected("Failed to merge scene nodes for export");

            truncate_sh_degree(*merged, sh_degree);

            switch (format) {
            case core::ExportFormat::PLY: {
                if (auto result = io::save_ply(*merged, io::PlySaveOptions{.output_path = path, .binary = true}); !result)
                    return std::unexpected(result.error().message);
                break;
            }
            case core::ExportFormat::SOG: {
                if (auto result = io::save_sog(*merged, io::SogSaveOptions{.output_path = path, .kmeans_iterations = 10}); !result)
                    return std::unexpected(result.error().message);
                break;
            }
            case core::ExportFormat::SPZ: {
                if (auto result = io::save_spz(*merged, io::SpzSaveOptions{.output_path = path}); !result)
                    return std::unexpected(result.error().message);
                break;
            }
            case core::ExportFormat::HTML_VIEWER: {
                if (auto result = vis::gui::export_html_viewer(*merged, vis::gui::HtmlViewerExportOptions{.output_path = path}); !result)
                    return std::unexpected(result.error());
                break;
            }
            }

            return {};
        }

        std::expected<std::string, std::string> resolve_gaussian_node_name(
            const vis::SceneManager& scene_manager,
            const std::optional<std::string>& requested_node) {
            const auto& scene = scene_manager.getScene();
            if (requested_node) {
                const auto* const node = scene.getNode(*requested_node);
                if (!node)
                    return std::unexpected("Node not found: " + *requested_node);
                if (!node->model)
                    return std::unexpected("Node does not contain gaussian data: " + *requested_node);
                return *requested_node;
            }

            const auto selected_name = scene_manager.getSelectedNodeName();
            if (!selected_name.empty()) {
                const auto* const node = scene.getNode(selected_name);
                if (node && node->model)
                    return selected_name;
            }

            const auto& training_name = scene.getTrainingModelNodeName();
            if (!training_name.empty()) {
                const auto* const node = scene.getNode(training_name);
                if (node && node->model)
                    return training_name;
            }

            const auto* fallback = find_first_visible_splat_node(scene);
            if (fallback)
                return fallback->name;

            return std::unexpected("No gaussian node specified and no suitable selected/training node is available");
        }

        core::Tensor* resolve_gaussian_field(core::SplatData& splat_data, std::string_view field_name) {
            if (field_name == "means")
                return &splat_data.means_raw();
            if (field_name == "scales" || field_name == "scaling" || field_name == "scaling_raw")
                return &splat_data.scaling_raw();
            if (field_name == "rotations" || field_name == "rotation" || field_name == "rotation_raw")
                return &splat_data.rotation_raw();
            if (field_name == "opacities" || field_name == "opacity" || field_name == "opacity_raw")
                return &splat_data.opacity_raw();
            if (field_name == "sh0")
                return &splat_data.sh0_raw();
            if (field_name == "shN")
                return &splat_data.shN_raw();
            return nullptr;
        }

        const core::Tensor* resolve_gaussian_field(const core::SplatData& splat_data, std::string_view field_name) {
            return resolve_gaussian_field(const_cast<core::SplatData&>(splat_data), field_name);
        }

        json tensor_payload_json(const core::Tensor& tensor) {
            json shape = json::array();
            for (const auto dim : tensor.shape().dims())
                shape.push_back(static_cast<int64_t>(dim));

            const auto cpu_tensor = tensor.to(core::Device::CPU).to(core::DataType::Float32).contiguous();
            return json{
                {"shape", shape},
                {"values", cpu_tensor.to_vector()},
            };
        }

        std::expected<std::vector<int>, std::string> parse_int_array(const json& value, const char* field_name) {
            if (!value.is_array())
                return std::unexpected(std::string("Field '") + field_name + "' must be an array of integers");

            std::vector<int> result;
            result.reserve(value.size());
            for (const auto& item : value)
                result.push_back(item.get<int>());
            return result;
        }

        std::expected<std::vector<float>, std::string> parse_float_array(const json& value, const char* field_name) {
            if (!value.is_array())
                return std::unexpected(std::string("Field '") + field_name + "' must be an array of numbers");

            std::vector<float> result;
            result.reserve(value.size());
            for (const auto& item : value)
                result.push_back(item.get<float>());
            return result;
        }

        size_t product_of_tail_dims(const core::Tensor& tensor) {
            size_t product = 1;
            const auto& shape = tensor.shape();
            for (size_t i = 1; i < shape.rank(); ++i)
                product *= shape[i];
            return product;
        }

        class EventSubscriptionRegistry {
        public:
            static EventSubscriptionRegistry& instance() {
                static EventSubscriptionRegistry registry;
                return registry;
            }

            std::expected<int64_t, std::string> subscribe(const std::vector<std::string>& types, const size_t max_queue) {
                std::unordered_set<std::string> supported;
                for (const auto type : kMcpSubscriptionEventTypes)
                    supported.insert(std::string(type));

                for (const auto& type : types) {
                    if (type != "*" && !supported.contains(type))
                        return std::unexpected("Unsupported event type: " + type);
                }

                std::lock_guard lock(mutex_);
                const int64_t id = next_id_++;
                Subscription sub;
                sub.max_queue = std::max<size_t>(1, max_queue);
                for (const auto& type : types)
                    sub.types.insert(type);
                subscriptions_.emplace(id, std::move(sub));
                return id;
            }

            json poll(const int64_t id, const size_t max_events, const bool clear) {
                std::lock_guard lock(mutex_);
                const auto it = subscriptions_.find(id);
                if (it == subscriptions_.end())
                    return json{{"error", "Unknown subscription id"}};

                auto& sub = it->second;
                const size_t count = std::min(max_events, sub.queue.size());
                json events = json::array();
                for (size_t i = 0; i < count; ++i)
                    events.push_back(sub.queue[i]);
                if (clear) {
                    for (size_t i = 0; i < count; ++i)
                        sub.queue.pop_front();
                }

                return json{
                    {"success", true},
                    {"subscription_id", id},
                    {"available", static_cast<int64_t>(sub.queue.size())},
                    {"returned", static_cast<int64_t>(events.size())},
                    {"dropped", static_cast<int64_t>(sub.dropped)},
                    {"events", events},
                };
            }

            bool unsubscribe(const int64_t id) {
                std::lock_guard lock(mutex_);
                return subscriptions_.erase(id) > 0;
            }

            json list() {
                std::lock_guard lock(mutex_);
                json subscriptions = json::array();
                for (const auto& [id, sub] : subscriptions_) {
                    json types = json::array();
                    for (const auto& type : sub.types)
                        types.push_back(type);
                    subscriptions.push_back({
                        {"subscription_id", id},
                        {"types", types},
                        {"queued", static_cast<int64_t>(sub.queue.size())},
                        {"dropped", static_cast<int64_t>(sub.dropped)},
                        {"max_queue", static_cast<int64_t>(sub.max_queue)},
                    });
                }

                return json{
                    {"success", true},
                    {"supported_types", mcp_subscription_event_types_json()},
                    {"subscriptions", subscriptions},
                };
            }

        private:
            struct Subscription {
                std::unordered_set<std::string> types;
                std::deque<json> queue;
                size_t dropped = 0;
                size_t max_queue = 256;
            };

            std::mutex mutex_;
            std::unordered_map<int64_t, Subscription> subscriptions_;
            int64_t next_id_ = 1;
            event::ScopedHandler handlers_;

            EventSubscriptionRegistry() {
                register_mcp_event_handlers(
                    handlers_,
                    McpEventStreamKind::SubscriptionQueue,
                    [this](const std::string& type, json payload) {
                        publish(type, std::move(payload));
                    });
            }

            void publish(const std::string& type, json payload) {
                const auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::system_clock::now().time_since_epoch())
                                              .count();

                std::lock_guard lock(mutex_);
                for (auto& [_, sub] : subscriptions_) {
                    if (!sub.types.empty() && !sub.types.contains("*") && !sub.types.contains(type))
                        continue;

                    if (sub.queue.size() >= sub.max_queue) {
                        sub.queue.pop_front();
                        ++sub.dropped;
                    }

                    sub.queue.push_back(json{
                        {"type", type},
                        {"timestamp_ms", timestamp_ms},
                        {"data", payload},
                    });
                }
            }
        };

    } // namespace

    void register_gui_scene_tools(vis::Visualizer* viewer) {
        assert(viewer);
        auto& registry = ToolRegistry::instance();

        register_generic_gui_operator_tools(registry, viewer);
        register_generic_gui_runtime_tools(registry, viewer);
        register_generic_gui_ui_tools(registry, viewer);

        auto* const viewer_impl = dynamic_cast<vis::VisualizerImpl*>(viewer);
        assert(viewer_impl);
        if (!viewer_impl) {
            LOG_ERROR("GUI-native MCP scene tools require a GUI VisualizerImpl");
            return;
        }

        // --- Scene operations (posted to GUI thread) ---

        registry.register_tool(
            McpTool{
                .name = "scene.load_ply",
                .description = "Load a PLY file for viewing",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Path to PLY file"}}}},
                    .required = {"path"}}},
            [viewer](const json& args) -> json {
                std::filesystem::path path = args["path"].get<std::string>();

                auto result = post_and_wait(viewer, [viewer, path]() {
                    return viewer->loadPLY(path);
                });

                if (!result)
                    return json{{"error", result.error()}};

                return json{{"success", true}, {"path", path.string()}};
            });

        mcp::register_shared_scene_tools(mcp::SharedSceneToolBackend{
            .runtime = "gui",
            .thread_affinity = "gui_thread",
            .load_dataset =
                [viewer](const std::filesystem::path& path,
                         const core::param::TrainingParameters& params) {
                    auto immediate_params = params;
                    immediate_params.dataset.data_path.clear();
                    return post_and_wait(viewer, [viewer, params = std::move(immediate_params), path]() {
                        viewer->setParameters(params);
                        return viewer->loadDataset(path);
                    });
                },
            .load_checkpoint =
                [viewer](const std::filesystem::path& path) {
                    return post_and_wait(viewer, [viewer, path]() {
                        return viewer->loadCheckpointForTraining(path);
                    });
                },
            .save_checkpoint =
                [viewer](const std::optional<std::filesystem::path>& path)
                -> std::expected<std::filesystem::path, std::string> {
                return post_and_wait(viewer, [viewer, path]() {
                    return viewer->saveCheckpoint(path);
                });
            },
            .save_ply =
                [viewer](const std::filesystem::path& path) {
                    return post_and_wait(viewer, [viewer, path]() -> std::expected<void, std::string> {
                        auto& scene = viewer->getScene();
                        auto* model = scene.getTrainingModel();
                        if (!model)
                            return std::unexpected("No model to save");

                        io::PlySaveOptions options{.output_path = path, .binary = true};
                        auto result = io::save_ply(*model, options);
                        if (!result)
                            return std::unexpected(result.error().message);
                        return {};
                    });
                },
            .start_training =
                [viewer]() {
                    return post_and_wait(viewer, [viewer]() {
                        return viewer->startTraining();
                    });
                },
            .render_capture =
                [viewer](std::optional<int> camera_index, int width, int height) {
                    return post_and_wait(viewer, [viewer, camera_index, width, height]() {
                        if (camera_index)
                            return render_scene_to_base64(viewer->getScene(), *camera_index, width, height);
                        return capture_live_viewport_to_base64(viewer, width, height);
                    });
                },
            .gaussian_count =
                [viewer]() -> std::expected<int64_t, std::string> {
                return post_and_wait(viewer, [viewer]() -> std::expected<int64_t, std::string> {
                    return count_visible_model_gaussians(viewer->getScene());
                });
            }});

        registry.register_tool(
            McpTool{
                .name = "camera.get",
                .description = "Get the current interactive viewport camera state",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, []() -> json {
                    const auto info = vis::get_current_view_info();
                    if (!info)
                        return json{{"error", "Viewport camera bridge is not available"}};
                    return view_info_json(*info);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "camera.set_view",
                .description = "Set the interactive viewport camera by eye/target/up, with optional FOV override",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"eye", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Camera eye position [x,y,z]"}}},
                        {"target", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Camera target/pivot position [x,y,z]"}}},
                        {"up", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional up vector [x,y,z], defaults to [0,1,0]"}}},
                        {"fov_degrees", json{{"type", "number"}, {"description", "Optional vertical field of view in degrees"}}}},
                    .required = {"eye", "target"}}},
            [viewer_impl](const json& args) -> json {
                auto eye = optional_vec3_arg(args, "eye");
                if (!eye)
                    return json{{"error", eye.error()}};
                auto target = optional_vec3_arg(args, "target");
                if (!target)
                    return json{{"error", target.error()}};
                auto up = optional_vec3_arg(args, "up");
                if (!up)
                    return json{{"error", up.error()}};
                if (!eye->has_value() || !target->has_value())
                    return json{{"error", "Fields 'eye' and 'target' must be provided"}};

                const glm::vec3 up_value = up->value_or(glm::vec3(0.0f, 1.0f, 0.0f));
                const std::optional<float> fov = args.contains("fov_degrees")
                                                     ? std::optional<float>(args["fov_degrees"].get<float>())
                                                     : std::nullopt;

                return post_and_wait(viewer_impl, [eye = **eye, target = **target, up_value, fov]() -> json {
                    vis::apply_set_view(vis::SetViewParams{
                        .eye = {eye.x, eye.y, eye.z},
                        .target = {target.x, target.y, target.z},
                        .up = {up_value.x, up_value.y, up_value.z},
                    });
                    if (fov)
                        vis::apply_set_fov(*fov);

                    const auto info = vis::get_current_view_info();
                    if (!info)
                        return json{{"success", true}};
                    return view_info_json(*info);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "camera.reset",
                .description = "Reset the interactive viewport camera to its saved home position",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, []() -> json {
                    core::events::cmd::ResetCamera{}.emit();
                    const auto info = vis::get_current_view_info();
                    if (!info)
                        return json{{"success", true}};
                    return view_info_json(*info);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "camera.list",
                .description = "List dataset camera nodes currently available in the shared GUI scene",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, [viewer_impl]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    const auto& scene = scene_manager->getScene();
                    json cameras = json::array();
                    for (const auto* const node : scene.getNodes()) {
                        if (node && node->type == core::NodeType::CAMERA && node->camera)
                            cameras.push_back(camera_node_json(scene, *node));
                    }

                    return json{{"success", true}, {"count", cameras.size()}, {"cameras", cameras}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "camera.go_to_dataset_camera",
                .description = "Move the interactive viewport to a dataset camera by UID or camera node name",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"uid", json{{"type", "integer"}, {"description", "Dataset camera UID"}}},
                        {"node", json{{"type", "string"}, {"description", "Dataset camera node name"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const std::optional<int> uid = args.contains("uid")
                                                   ? std::optional<int>(args["uid"].get<int>())
                                                   : std::nullopt;
                const auto node_name = optional_string_arg(args, "node");
                if (!uid && !node_name)
                    return json{{"error", "Either 'uid' or 'node' must be provided"}};

                return post_and_wait(viewer_impl, [viewer_impl, uid, node_name]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    const auto& scene = scene_manager->getScene();
                    const core::SceneNode* target = nullptr;
                    if (node_name) {
                        target = scene.getNode(*node_name);
                        if (!target || target->type != core::NodeType::CAMERA || !target->camera)
                            return json{{"error", "Camera node not found: " + *node_name}};
                    } else {
                        for (const auto* const node : scene.getNodes()) {
                            if (node && node->type == core::NodeType::CAMERA && node->camera &&
                                node->camera_uid == *uid) {
                                target = node;
                                break;
                            }
                        }
                        if (!target)
                            return json{{"error", "Camera UID not found: " + std::to_string(*uid)}};
                    }

                    core::events::cmd::GoToCamView{.cam_id = target->camera_uid}.emit();
                    return json{{"success", true}, {"camera", camera_node_json(scene, *target)}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "history.get",
                .description = "Inspect the shared undo/redo history state",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, []() -> json {
                    return history_json();
                });
            });

        register_gui_operator_tool(
            registry, viewer_impl,
            GuiOperatorToolBinding{
                .tool_name = "history.undo",
                .operator_id = vis::op::BuiltinOp::Undo,
                .category = "history",
                .description = "Undo the most recent shared scene operation",
                .prepare =
                    [](vis::Visualizer& /*viewer*/, const json& /*args*/,
                       vis::op::OperatorProperties& /*props*/) -> std::expected<void, std::string> {
                    if (!vis::op::undoHistory().canUndo())
                        return std::unexpected("Nothing to undo");
                    return {};
                },
                .on_success =
                    [](vis::Visualizer& /*viewer*/, const json& /*args*/,
                       const vis::op::OperatorProperties& /*props*/,
                       const vis::op::OperatorReturnValue& /*result*/) {
                        auto payload = history_json();
                        payload["performed"] = "undo";
                        return payload;
                    },
            });

        register_gui_operator_tool(
            registry, viewer_impl,
            GuiOperatorToolBinding{
                .tool_name = "history.redo",
                .operator_id = vis::op::BuiltinOp::Redo,
                .category = "history",
                .description = "Redo the next shared scene operation",
                .prepare =
                    [](vis::Visualizer& /*viewer*/, const json& /*args*/,
                       vis::op::OperatorProperties& /*props*/) -> std::expected<void, std::string> {
                    if (!vis::op::undoHistory().canRedo())
                        return std::unexpected("Nothing to redo");
                    return {};
                },
                .on_success =
                    [](vis::Visualizer& /*viewer*/, const json& /*args*/,
                       const vis::op::OperatorProperties& /*props*/,
                       const vis::op::OperatorReturnValue& /*result*/) {
                        auto payload = history_json();
                        payload["performed"] = "redo";
                        return payload;
                    },
            });

        registry.register_tool(
            McpTool{
                .name = "render.settings.get",
                .description = "Read the current viewport render settings",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, []() -> json {
                    const auto settings = vis::get_render_settings();
                    if (!settings)
                        return json{{"error", "Render settings bridge is not available"}};
                    return render_settings_json(*settings);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "render.settings.set",
                .description = "Update one or more viewport render settings",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"focal_length_mm", json{{"type", "number"}}},
                        {"render_scale", json{{"type", "number"}}},
                        {"background_color", json{{"type", "array"}, {"items", json{{"type", "number"}}}}},
                        {"antialiasing", json{{"type", "boolean"}}},
                        {"show_grid", json{{"type", "boolean"}}},
                        {"show_camera_frustums", json{{"type", "boolean"}}},
                        {"point_cloud_mode", json{{"type", "boolean"}}},
                        {"show_crop_box", json{{"type", "boolean"}}},
                        {"use_crop_box", json{{"type", "boolean"}}},
                        {"show_ellipsoid", json{{"type", "boolean"}}},
                        {"use_ellipsoid", json{{"type", "boolean"}}},
                        {"ppisp_exposure", json{{"type", "number"}}},
                        {"ppisp", json{{"type", "object"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                return post_and_wait(viewer_impl, [args]() -> json {
                    auto settings = vis::get_render_settings();
                    if (!settings)
                        return json{{"error", "Render settings bridge is not available"}};

                    if (auto result = apply_render_settings_patch(args, *settings); !result)
                        return json{{"error", result.error()}};

                    vis::update_render_settings(*settings);
                    const auto updated = vis::get_render_settings();
                    if (!updated)
                        return json{{"success", true}};
                    return render_settings_json(*updated);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "dataset.get_info",
                .description = "Inspect the current dataset/training scene metadata",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, [viewer_impl]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    return dataset_info_json(*scene_manager);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.set_node_visibility",
                .description = "Show or hide a scene node using the same command path as the UI",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"name", json{{"type", "string"}, {"description", "Node name"}}},
                        {"visible", json{{"type", "boolean"}, {"description", "Whether the node should be visible"}}}},
                    .required = {"name", "visible"}}},
            [viewer_impl](const json& args) -> json {
                const std::string name = args["name"].get<std::string>();
                const bool visible = args["visible"].get<bool>();

                return post_and_wait(viewer_impl, [viewer_impl, name, visible]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    const auto& scene = scene_manager->getScene();
                    if (!scene.getNode(name))
                        return json{{"error", "Node not found: " + name}};

                    core::events::cmd::SetPLYVisibility{.name = name, .visible = visible}.emit();
                    if (const auto* const node = scene.getNode(name))
                        return json{{"success", true}, {"node", node_summary_json(scene, *node)}};
                    return json{{"success", true}, {"name", name}, {"visible", visible}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.set_node_locked",
                .description = "Lock or unlock a scene node for editing",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"name", json{{"type", "string"}, {"description", "Node name"}}},
                        {"locked", json{{"type", "boolean"}, {"description", "Whether the node should be locked"}}}},
                    .required = {"name", "locked"}}},
            [viewer_impl](const json& args) -> json {
                const std::string name = args["name"].get<std::string>();
                const bool locked = args["locked"].get<bool>();

                return post_and_wait(viewer_impl, [viewer_impl, name, locked]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    const auto& scene = scene_manager->getScene();
                    if (!scene.getNode(name))
                        return json{{"error", "Node not found: " + name}};

                    core::events::cmd::SetNodeLocked{.name = name, .locked = locked}.emit();
                    if (const auto* const node = scene.getNode(name))
                        return json{{"success", true}, {"node", node_summary_json(scene, *node)}};
                    return json{{"success", true}, {"name", name}, {"locked", locked}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.rename_node",
                .description = "Rename a scene node",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"old_name", json{{"type", "string"}, {"description", "Current node name"}}},
                        {"new_name", json{{"type", "string"}, {"description", "New node name"}}}},
                    .required = {"old_name", "new_name"}}},
            [viewer_impl](const json& args) -> json {
                const std::string old_name = args["old_name"].get<std::string>();
                const std::string new_name = args["new_name"].get<std::string>();

                return post_and_wait(viewer_impl, [viewer_impl, old_name, new_name]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    const auto& scene = scene_manager->getScene();
                    if (!scene.getNode(old_name))
                        return json{{"error", "Node not found: " + old_name}};

                    core::events::cmd::RenamePLY{.old_name = old_name, .new_name = new_name}.emit();
                    if (const auto* const node = scene.getNode(new_name))
                        return json{{"success", true}, {"node", node_summary_json(scene, *node)}};
                    return json{{"error", "Rename did not produce a node named: " + new_name}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.reparent_node",
                .description = "Reparent a scene node under another node or move it to the root",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"name", json{{"type", "string"}, {"description", "Node to move"}}},
                        {"parent", json{{"type", "string"}, {"description", "New parent node; omit or null for root"}}}},
                    .required = {"name"}}},
            [viewer_impl](const json& args) -> json {
                const std::string name = args["name"].get<std::string>();
                const auto parent = optional_string_arg(args, "parent");

                return post_and_wait(viewer_impl, [viewer_impl, name, parent]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    const auto& scene = scene_manager->getScene();
                    if (!scene.getNode(name))
                        return json{{"error", "Node not found: " + name}};
                    if (parent && !scene.getNode(*parent))
                        return json{{"error", "Parent node not found: " + *parent}};

                    core::events::cmd::ReparentNode{.node_name = name, .new_parent_name = parent.value_or("")}.emit();
                    if (const auto* const node = scene.getNode(name))
                        return json{{"success", true}, {"node", node_summary_json(scene, *node)}};
                    return json{{"error", "Node disappeared after reparent: " + name}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.add_group",
                .description = "Create a new empty group node",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"name", json{{"type", "string"}, {"description", "Requested group name"}}},
                        {"parent", json{{"type", "string"}, {"description", "Optional parent node name"}}}},
                    .required = {"name"}}},
            [viewer_impl](const json& args) -> json {
                const std::string name = args["name"].get<std::string>();
                const auto parent = optional_string_arg(args, "parent");

                return post_and_wait(viewer_impl, [viewer_impl, name, parent]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    const auto& scene = scene_manager->getScene();
                    if (parent && !scene.getNode(*parent))
                        return json{{"error", "Parent node not found: " + *parent}};

                    std::unordered_set<std::string> before;
                    for (const auto* const node : scene.getNodes()) {
                        if (node)
                            before.insert(node->name);
                    }

                    core::events::cmd::AddGroup{.name = name, .parent_name = parent.value_or("")}.emit();

                    for (const auto* const node : scene.getNodes()) {
                        if (node && node->type == core::NodeType::GROUP && !before.contains(node->name))
                            return json{{"success", true}, {"node", node_summary_json(scene, *node)}};
                    }

                    return json{{"error", "Group creation did not add a new group node"}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.duplicate_node",
                .description = "Duplicate a scene node and its descendants",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"name", json{{"type", "string"}, {"description", "Node to duplicate"}}}},
                    .required = {"name"}}},
            [viewer_impl](const json& args) -> json {
                const std::string name = args["name"].get<std::string>();

                return post_and_wait(viewer_impl, [viewer_impl, name]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    const auto& scene = scene_manager->getScene();
                    if (!scene.getNode(name))
                        return json{{"error", "Node not found: " + name}};

                    std::unordered_set<std::string> before;
                    for (const auto* const node : scene.getNodes()) {
                        if (node)
                            before.insert(node->name);
                    }

                    core::events::cmd::DuplicateNode{.name = name}.emit();

                    json nodes = json::array();
                    for (const auto* const node : scene.getNodes()) {
                        if (node && !before.contains(node->name))
                            nodes.push_back(node_summary_json(scene, *node));
                    }

                    if (nodes.empty())
                        return json{{"error", "Node duplication did not add any nodes"}};
                    return json{{"success", true}, {"count", nodes.size()}, {"nodes", nodes}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.merge_group",
                .description = "Merge the splat children of a group into a single splat node",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"name", json{{"type", "string"}, {"description", "Group node to merge"}}}},
                    .required = {"name"}}},
            [viewer_impl](const json& args) -> json {
                const std::string name = args["name"].get<std::string>();

                return post_and_wait(viewer_impl, [viewer_impl, name]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    const auto& scene = scene_manager->getScene();
                    const auto* const group = scene.getNode(name);
                    if (!group)
                        return json{{"error", "Node not found: " + name}};
                    if (group->type != core::NodeType::GROUP)
                        return json{{"error", "Node is not a group: " + name}};

                    std::unordered_set<std::string> before;
                    for (const auto* const node : scene.getNodes()) {
                        if (node)
                            before.insert(node->name);
                    }

                    core::events::cmd::MergeGroup{.name = name}.emit();

                    for (const auto* const node : scene.getNodes()) {
                        if (node && !before.contains(node->name))
                            return json{{"success", true}, {"node", node_summary_json(scene, *node)}};
                    }

                    return json{{"error", "Group merge did not create a merged node"}};
                });
            });

        register_gui_operator_tool(
            registry, viewer_impl,
            GuiOperatorToolBinding{
                .tool_name = "scene.delete_node",
                .operator_id = vis::op::BuiltinOp::Delete,
                .category = "scene",
                .description = "Delete a scene node, optionally keeping its children attached to the parent",
                .destructive = true,
                .prepare = prepare_delete_operator,
                .on_success = delete_operator_result,
            });

        registry.register_tool(
            McpTool{
                .name = "scene.export_ply",
                .description = "Start an asynchronous export of one or more scene nodes to PLY",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Destination file path"}}},
                        {"node", json{{"type", "string"}, {"description", "Optional node name"}}},
                        {"nodes", json{{"type", "array"}, {"items", json{{"type", "string"}}}, {"description", "Optional list of node names"}}},
                        {"sh_degree", json{{"type", "integer"}, {"description", "Optional SH degree to keep in the export"}}}},
                    .required = {"path"}}},
            [viewer_impl](const json& args) -> json {
                const std::filesystem::path path = args["path"].get<std::string>();
                const int sh_degree = args.value("sh_degree", 3);

                return post_and_wait(viewer_impl, [viewer_impl, args, path, sh_degree]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto node_names = resolve_export_nodes(*scene_manager, args);
                    if (!node_names)
                        return json{{"error", node_names.error()}};

                    if (auto result = export_scene_nodes(*scene_manager, *node_names, core::ExportFormat::PLY, path, sh_degree); !result)
                        return json{{"error", result.error()}};

                    return json{
                        {"success", true},
                        {"started", false},
                        {"completed", true},
                        {"format", "ply"},
                        {"path", core::path_to_utf8(path)},
                        {"nodes", *node_names},
                    };
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.export_sog",
                .description = "Start an asynchronous export of one or more scene nodes to SOG",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Destination file path"}}},
                        {"node", json{{"type", "string"}, {"description", "Optional node name"}}},
                        {"nodes", json{{"type", "array"}, {"items", json{{"type", "string"}}}, {"description", "Optional list of node names"}}},
                        {"sh_degree", json{{"type", "integer"}, {"description", "Optional SH degree to keep in the export"}}}},
                    .required = {"path"}}},
            [viewer_impl](const json& args) -> json {
                const std::filesystem::path path = args["path"].get<std::string>();
                const int sh_degree = args.value("sh_degree", 3);

                return post_and_wait(viewer_impl, [viewer_impl, args, path, sh_degree]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto node_names = resolve_export_nodes(*scene_manager, args);
                    if (!node_names)
                        return json{{"error", node_names.error()}};

                    if (auto result = export_scene_nodes(*scene_manager, *node_names, core::ExportFormat::SOG, path, sh_degree); !result)
                        return json{{"error", result.error()}};

                    return json{
                        {"success", true},
                        {"started", false},
                        {"completed", true},
                        {"format", "sog"},
                        {"path", core::path_to_utf8(path)},
                        {"nodes", *node_names},
                    };
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.export_spz",
                .description = "Start an asynchronous export of one or more scene nodes to SPZ",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Destination file path"}}},
                        {"node", json{{"type", "string"}, {"description", "Optional node name"}}},
                        {"nodes", json{{"type", "array"}, {"items", json{{"type", "string"}}}, {"description", "Optional list of node names"}}},
                        {"sh_degree", json{{"type", "integer"}, {"description", "Optional SH degree to keep in the export"}}}},
                    .required = {"path"}}},
            [viewer_impl](const json& args) -> json {
                const std::filesystem::path path = args["path"].get<std::string>();
                const int sh_degree = args.value("sh_degree", 3);

                return post_and_wait(viewer_impl, [viewer_impl, args, path, sh_degree]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto node_names = resolve_export_nodes(*scene_manager, args);
                    if (!node_names)
                        return json{{"error", node_names.error()}};

                    if (auto result = export_scene_nodes(*scene_manager, *node_names, core::ExportFormat::SPZ, path, sh_degree); !result)
                        return json{{"error", result.error()}};

                    return json{
                        {"success", true},
                        {"started", false},
                        {"completed", true},
                        {"format", "spz"},
                        {"path", core::path_to_utf8(path)},
                        {"nodes", *node_names},
                    };
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.export_html",
                .description = "Start an asynchronous export of one or more scene nodes to the standalone HTML viewer",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Destination file path"}}},
                        {"node", json{{"type", "string"}, {"description", "Optional node name"}}},
                        {"nodes", json{{"type", "array"}, {"items", json{{"type", "string"}}}, {"description", "Optional list of node names"}}},
                        {"sh_degree", json{{"type", "integer"}, {"description", "Optional SH degree to keep in the export"}}}},
                    .required = {"path"}}},
            [viewer_impl](const json& args) -> json {
                const std::filesystem::path path = args["path"].get<std::string>();
                const int sh_degree = args.value("sh_degree", 3);

                return post_and_wait(viewer_impl, [viewer_impl, args, path, sh_degree]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto node_names = resolve_export_nodes(*scene_manager, args);
                    if (!node_names)
                        return json{{"error", node_names.error()}};

                    if (auto result = export_scene_nodes(*scene_manager, *node_names, core::ExportFormat::HTML_VIEWER, path, sh_degree); !result)
                        return json{{"error", result.error()}};

                    return json{
                        {"success", true},
                        {"started", false},
                        {"completed", true},
                        {"format", "html"},
                        {"path", core::path_to_utf8(path)},
                        {"nodes", *node_names},
                    };
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.export_status",
                .description = "Report the export execution mode for scene exports",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, []() -> json {
                    return json{
                        {"success", true},
                        {"active", false},
                        {"mode", "synchronous"},
                        {"stage", "idle"},
                    };
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.export_cancel",
                .description = "Scene exports run synchronously and cannot be cancelled once started",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, []() -> json {
                    return json{{"error", "Scene exports run synchronously and cannot be cancelled"}};
                });
            });

        // --- Selection tools ---

        registry.register_tool(
            McpTool{
                .name = "selection.rect",
                .description = "Select Gaussians inside a screen rectangle",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"x0", json{{"type", "number"}, {"description", "Left edge X coordinate"}}},
                        {"y0", json{{"type", "number"}, {"description", "Top edge Y coordinate"}}},
                        {"x1", json{{"type", "number"}, {"description", "Right edge X coordinate"}}},
                        {"y1", json{{"type", "number"}, {"description", "Bottom edge Y coordinate"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add", "remove"})}, {"description", "Selection mode (default: replace)"}}}},
                    .required = {"x0", "y0", "x1", "y1"}}},
            [viewer_impl](const json& args) -> json {
                const float x0 = args["x0"].get<float>();
                const float y0 = args["y0"].get<float>();
                const float x1 = args["x1"].get<float>();
                const float y1 = args["y1"].get<float>();
                const std::string mode = args.value("mode", "replace");
                const int camera_index = args.value("camera_index", 0);

                return post_and_wait(viewer_impl, [viewer_impl, x0, y0, x1, y1, mode, camera_index]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    return selection_result_json(*scene_manager,
                                                 scene_manager->selectRect(x0, y0, x1, y1, mode, camera_index));
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.polygon",
                .description = "Select Gaussians inside a screen polygon",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"points", json{{"type", "array"}, {"items", json{{"type", "array"}, {"items", json{{"type", "number"}}}}}, {"description", "Polygon vertices [[x0,y0], [x1,y1], ...]"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add", "remove"})}, {"description", "Selection mode (default: replace)"}}}},
                    .required = {"points"}}},
            [viewer_impl](const json& args) -> json {
                const auto& points = args["points"];
                const size_t num_vertices = points.size();
                if (num_vertices < 3)
                    return json{{"error", "Polygon requires at least 3 vertices"}};

                std::vector<float> vertex_data;
                vertex_data.reserve(num_vertices * 2);
                for (const auto& pt : points) {
                    vertex_data.push_back(pt[0].get<float>());
                    vertex_data.push_back(pt[1].get<float>());
                }

                const std::string mode = args.value("mode", "replace");
                const int camera_index = args.value("camera_index", 0);

                return post_and_wait(viewer_impl, [viewer_impl, vertex_data = std::move(vertex_data), mode, camera_index]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    return selection_result_json(*scene_manager,
                                                 scene_manager->selectPolygon(vertex_data, mode, camera_index));
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.lasso",
                .description = "Select Gaussians inside a screen-space lasso path",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"points", json{{"type", "array"}, {"items", json{{"type", "array"}, {"items", json{{"type", "number"}}}}}, {"description", "Lasso points [[x0,y0], [x1,y1], ...]"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add", "remove"})}, {"description", "Selection mode (default: replace)"}}}},
                    .required = {"points"}}},
            [viewer_impl](const json& args) -> json {
                const auto& points = args["points"];
                const size_t num_vertices = points.size();
                if (num_vertices < 3)
                    return json{{"error", "Lasso requires at least 3 points"}};

                std::vector<float> vertex_data;
                vertex_data.reserve(num_vertices * 2);
                for (const auto& pt : points) {
                    vertex_data.push_back(pt[0].get<float>());
                    vertex_data.push_back(pt[1].get<float>());
                }

                const std::string mode = args.value("mode", "replace");
                const int camera_index = args.value("camera_index", 0);

                return post_and_wait(viewer_impl, [viewer_impl, vertex_data = std::move(vertex_data), mode, camera_index]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    return selection_result_json(*scene_manager,
                                                 scene_manager->selectLasso(vertex_data, mode, camera_index));
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.ring",
                .description = "Select the front-most Gaussian under a screen point using ring-mode picking",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"x", json{{"type", "number"}, {"description", "X coordinate"}}},
                        {"y", json{{"type", "number"}, {"description", "Y coordinate"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add", "remove"})}, {"description", "Selection mode (default: replace)"}}}},
                    .required = {"x", "y"}}},
            [viewer_impl](const json& args) -> json {
                const float x = args["x"].get<float>();
                const float y = args["y"].get<float>();
                const std::string mode = args.value("mode", "replace");
                const int camera_index = args.value("camera_index", 0);

                return post_and_wait(viewer_impl, [viewer_impl, x, y, mode, camera_index]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    return selection_result_json(*scene_manager,
                                                 scene_manager->selectRing(x, y, mode, camera_index));
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.brush",
                .description = "Select Gaussians near a screen point using a brush radius",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"x", json{{"type", "number"}, {"description", "X coordinate"}}},
                        {"y", json{{"type", "number"}, {"description", "Y coordinate"}}},
                        {"radius", json{{"type", "number"}, {"description", "Selection radius in pixels (default: 20)"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add", "remove"})}, {"description", "Selection mode (default: replace)"}}}},
                    .required = {"x", "y"}}},
            [viewer_impl](const json& args) -> json {
                const float x = args["x"].get<float>();
                const float y = args["y"].get<float>();
                const float radius = args.value("radius", 20.0f);
                const std::string mode = args.value("mode", "replace");
                const int camera_index = args.value("camera_index", 0);

                return post_and_wait(viewer_impl, [viewer_impl, x, y, radius, mode, camera_index]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    return selection_result_json(*scene_manager,
                                                 scene_manager->selectBrush(x, y, radius, mode, camera_index));
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.click",
                .description = "Alias for selection.brush",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"x", json{{"type", "number"}, {"description", "X coordinate"}}},
                        {"y", json{{"type", "number"}, {"description", "Y coordinate"}}},
                        {"radius", json{{"type", "number"}, {"description", "Selection radius in pixels (default: 20)"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add", "remove"})}, {"description", "Selection mode (default: replace)"}}}},
                    .required = {"x", "y"}}},
            [viewer_impl](const json& args) -> json {
                const float x = args["x"].get<float>();
                const float y = args["y"].get<float>();
                const float radius = args.value("radius", 20.0f);
                const std::string mode = args.value("mode", "replace");
                const int camera_index = args.value("camera_index", 0);

                return post_and_wait(viewer_impl, [viewer_impl, x, y, radius, mode, camera_index]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    return selection_result_json(*scene_manager,
                                                 scene_manager->selectBrush(x, y, radius, mode, camera_index));
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.get",
                .description = "Get current selection (returns selected Gaussian indices)",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"max_indices", json{{"type", "integer"}, {"description", "Maximum indices to return (default: 100000)"}}}},
                    .required = {}}},
            [viewer](const json& args) -> json {
                const int max_indices = args.value("max_indices", 100000);

                return post_and_wait(viewer, [viewer, max_indices]() -> json {
                    const auto selection = vis::cap::getSelectionSnapshot(viewer->getScene(), max_indices);
                    return json{
                        {"success", true},
                        {"selected_count", selection.selected_count},
                        {"indices", selection.indices},
                        {"truncated", selection.truncated},
                    };
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.clear",
                .description = "Clear all selection",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, [viewer_impl]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};
                    if (auto result = vis::cap::clearGaussianSelection(*scene_manager); !result)
                        return json{{"error", result.error()}};
                    return json{{"success", true}, {"selected_count", 0}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.list_nodes",
                .description = "List scene nodes visible to the shared GUI scene",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"include_hidden", json{{"type", "boolean"}, {"description", "Include hidden nodes (default: true)"}}},
                        {"include_auxiliary", json{{"type", "boolean"}, {"description", "Include helper nodes like crop boxes, ellipsoids, cameras, and keyframes (default: true)"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const bool include_hidden = args.value("include_hidden", true);
                const bool include_auxiliary = args.value("include_auxiliary", true);

                return post_and_wait(viewer_impl, [viewer_impl, include_hidden, include_auxiliary]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    const auto& scene = scene_manager->getScene();
                    json nodes = json::array();
                    for (const auto* const node : scene.getNodes()) {
                        if (!node)
                            continue;
                        if (!include_hidden && !static_cast<bool>(node->visible))
                            continue;
                        if (!include_auxiliary) {
                            switch (node->type) {
                            case core::NodeType::CROPBOX:
                            case core::NodeType::ELLIPSOID:
                            case core::NodeType::CAMERA_GROUP:
                            case core::NodeType::CAMERA:
                            case core::NodeType::IMAGE_GROUP:
                            case core::NodeType::IMAGE:
                            case core::NodeType::KEYFRAME_GROUP:
                            case core::NodeType::KEYFRAME:
                                continue;
                            default:
                                break;
                            }
                        }
                        nodes.push_back(node_summary_json(scene, *node));
                    }

                    return json{{"success", true}, {"count", nodes.size()}, {"nodes", nodes}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.get_selected_nodes",
                .description = "Get the current shared node selection from the GUI scene",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, [viewer_impl]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    const auto& scene = scene_manager->getScene();
                    json nodes = json::array();
                    for (const auto& name : scene_manager->getSelectedNodeNames()) {
                        if (const auto* const node = scene.getNode(name))
                            nodes.push_back(node_summary_json(scene, *node));
                    }

                    return json{{"success", true}, {"count", nodes.size()}, {"nodes", nodes}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "scene.select_node",
                .description = "Change the shared GUI node selection",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"name", json{{"type", "string"}, {"description", "Node name to select"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add"})}, {"description", "Selection update mode (default: replace)"}}}},
                    .required = {"name"}}},
            [viewer_impl](const json& args) -> json {
                const std::string name = args["name"].get<std::string>();
                const std::string mode = args.value("mode", "replace");

                return post_and_wait(viewer_impl, [viewer_impl, name, mode]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    if (auto result = vis::cap::selectNode(*scene_manager, name, mode); !result)
                        return json{{"error", result.error()}};

                    json nodes = json::array();
                    for (const auto& selected_name : scene_manager->getSelectedNodeNames()) {
                        if (const auto* const node = scene_manager->getScene().getNode(selected_name))
                            nodes.push_back(node_summary_json(scene_manager->getScene(), *node));
                    }

                    return json{{"success", true}, {"count", nodes.size()}, {"nodes", nodes}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "transform.get",
                .description = "Inspect local and world transforms for a node or the current shared node selection",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional node name; defaults to the current selected node(s)"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");

                return post_and_wait(viewer_impl, [viewer_impl, requested_node]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto targets = resolve_transform_targets(*scene_manager, requested_node);
                    if (!targets)
                        return json{{"error", targets.error()}};

                    json nodes = json::array();
                    for (const auto& name : *targets) {
                        if (const auto* const node = scene_manager->getScene().getNode(name))
                            nodes.push_back(transform_info_json(scene_manager->getScene(), *node));
                    }

                    return json{{"success", true}, {"count", nodes.size()}, {"nodes", nodes}};
                });
            });

        register_gui_operator_tool(
            registry, viewer_impl,
            GuiOperatorToolBinding{
                .tool_name = "transform.set",
                .operator_id = vis::op::BuiltinOp::TransformSet,
                .category = "transform",
                .description = "Set absolute local transform components for a node or the current shared node selection",
                .prepare = prepare_transform_set_operator,
                .on_success = transform_operator_result,
            });

        register_gui_operator_tool(
            registry, viewer_impl,
            GuiOperatorToolBinding{
                .tool_name = "transform.translate",
                .operator_id = vis::op::BuiltinOp::TransformTranslate,
                .category = "transform",
                .description = "Translate a node or the current shared node selection",
                .required = {"value"},
                .prepare = prepare_transform_operator,
                .on_success = transform_operator_result,
            });

        register_gui_operator_tool(
            registry, viewer_impl,
            GuiOperatorToolBinding{
                .tool_name = "transform.rotate",
                .operator_id = vis::op::BuiltinOp::TransformRotate,
                .category = "transform",
                .description = "Rotate a node or the current shared node selection by XYZ Euler deltas in radians",
                .required = {"value"},
                .prepare = prepare_transform_operator,
                .on_success = transform_operator_result,
            });

        register_gui_operator_tool(
            registry, viewer_impl,
            GuiOperatorToolBinding{
                .tool_name = "transform.scale",
                .operator_id = vis::op::BuiltinOp::TransformScale,
                .category = "transform",
                .description = "Scale a node or the current shared node selection by XYZ factors",
                .required = {"value"},
                .prepare = prepare_transform_operator,
                .on_success = transform_operator_result,
            });

        registry.register_tool(
            McpTool{
                .name = "crop_box.add",
                .description = "Add or reuse a crop box for a node in the shared GUI scene",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional splat or pointcloud node name; defaults to the current selected node"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");

                return post_and_wait(viewer_impl, [viewer_impl, requested_node]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto parent_id = resolve_cropbox_parent_id(*scene_manager, requested_node);
                    if (!parent_id)
                        return json{{"error", parent_id.error()}};

                    auto cropbox_id = ensure_cropbox(*scene_manager, rendering_manager, *parent_id);
                    if (!cropbox_id)
                        return json{{"error", cropbox_id.error()}};

                    return crop_box_info_json(*scene_manager, rendering_manager, *cropbox_id);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "crop_box.get",
                .description = "Inspect a crop box in the shared GUI scene",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional crop box node or parent node name; defaults to the current selected crop box"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");

                return post_and_wait(viewer_impl, [viewer_impl, requested_node]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto cropbox_id = resolve_cropbox_id(*scene_manager, requested_node);
                    if (!cropbox_id)
                        return json{{"error", cropbox_id.error()}};

                    return crop_box_info_json(*scene_manager, rendering_manager, *cropbox_id);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "crop_box.set",
                .description = "Update crop box bounds, transform, or render toggles in the shared GUI scene",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional crop box node or parent node name; defaults to the current selected crop box"}}},
                        {"min", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional local minimum bounds"}}},
                        {"max", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional local maximum bounds"}}},
                        {"translation", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional local XYZ translation"}}},
                        {"rotation", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional local XYZ Euler rotation in radians"}}},
                        {"scale", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional local XYZ scale"}}},
                        {"inverse", json{{"type", "boolean"}, {"description", "Invert the crop volume"}}},
                        {"enabled", json{{"type", "boolean"}, {"description", "Enable crop filtering for this crop box"}}},
                        {"show", json{{"type", "boolean"}, {"description", "Show crop boxes in the viewport"}}},
                        {"use", json{{"type", "boolean"}, {"description", "Use crop box filtering in rendering"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");
                auto min_bounds = optional_vec3_arg(args, "min");
                if (!min_bounds)
                    return json{{"error", min_bounds.error()}};
                auto max_bounds = optional_vec3_arg(args, "max");
                if (!max_bounds)
                    return json{{"error", max_bounds.error()}};
                auto translation = optional_vec3_arg(args, "translation");
                if (!translation)
                    return json{{"error", translation.error()}};
                auto rotation = optional_vec3_arg(args, "rotation");
                if (!rotation)
                    return json{{"error", rotation.error()}};
                auto scale = optional_vec3_arg(args, "scale");
                if (!scale)
                    return json{{"error", scale.error()}};

                const bool has_inverse = args.contains("inverse");
                const bool has_enabled = args.contains("enabled");
                const bool has_show = args.contains("show");
                const bool has_use = args.contains("use");

                if (!min_bounds->has_value() && !max_bounds->has_value() &&
                    !translation->has_value() && !rotation->has_value() && !scale->has_value() &&
                    !has_inverse && !has_enabled && !has_show && !has_use) {
                    return json{{"error", "No crop box fields were provided"}};
                }

                const bool inverse = has_inverse ? args["inverse"].get<bool>() : false;
                const bool enabled = has_enabled ? args["enabled"].get<bool>() : false;
                const bool show = has_show ? args["show"].get<bool>() : false;
                const bool use = has_use ? args["use"].get<bool>() : false;

                return post_and_wait(viewer_impl, [viewer_impl, requested_node, min_bounds = *min_bounds, max_bounds = *max_bounds, translation = *translation, rotation = *rotation, scale = *scale, has_inverse, inverse, has_enabled, enabled, has_show, show, has_use, use]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto cropbox_id = resolve_cropbox_id(*scene_manager, requested_node);
                    if (!cropbox_id)
                        return json{{"error", cropbox_id.error()}};

                    vis::cap::CropBoxUpdate update;
                    update.min_bounds = min_bounds;
                    update.max_bounds = max_bounds;
                    update.translation = translation;
                    update.rotation = rotation;
                    update.scale = scale;
                    update.has_inverse = has_inverse;
                    update.inverse = inverse;
                    update.has_enabled = has_enabled;
                    update.enabled = enabled;
                    update.has_show = has_show;
                    update.show = show;
                    update.has_use = has_use;
                    update.use = use;

                    if (auto result = vis::cap::updateCropBox(
                            *scene_manager, rendering_manager, *cropbox_id, update);
                        !result) {
                        return json{{"error", result.error()}};
                    }

                    return crop_box_info_json(*scene_manager, rendering_manager, *cropbox_id);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "crop_box.fit",
                .description = "Fit a crop box to its parent node bounds",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional crop box node or parent node name; defaults to the current selected crop box"}}},
                        {"use_percentile", json{{"type", "boolean"}, {"description", "Use percentile bounds instead of strict min/max (default: true)"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");
                const bool use_percentile = args.value("use_percentile", true);

                return post_and_wait(viewer_impl, [viewer_impl, requested_node, use_percentile]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto cropbox_id = resolve_cropbox_id(*scene_manager, requested_node);
                    if (!cropbox_id)
                        return json{{"error", cropbox_id.error()}};

                    auto result = fit_cropbox_to_parent(*scene_manager, rendering_manager, *cropbox_id, use_percentile);
                    if (!result)
                        return json{{"error", result.error()}};

                    return crop_box_info_json(*scene_manager, rendering_manager, *cropbox_id);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "crop_box.reset",
                .description = "Reset a crop box to default bounds and identity transform",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional crop box node or parent node name; defaults to the current selected crop box"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");

                return post_and_wait(viewer_impl, [viewer_impl, requested_node]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto cropbox_id = resolve_cropbox_id(*scene_manager, requested_node);
                    if (!cropbox_id)
                        return json{{"error", cropbox_id.error()}};

                    auto result = reset_cropbox(*scene_manager, rendering_manager, *cropbox_id);
                    if (!result)
                        return json{{"error", result.error()}};

                    return crop_box_info_json(*scene_manager, rendering_manager, *cropbox_id);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "ellipsoid.add",
                .description = "Add or reuse an ellipsoid helper for a node in the shared GUI scene",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional splat or pointcloud node name; defaults to the current selected node"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");

                return post_and_wait(viewer_impl, [viewer_impl, requested_node]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto parent_id = resolve_ellipsoid_parent_id(*scene_manager, requested_node);
                    if (!parent_id)
                        return json{{"error", parent_id.error()}};

                    auto ellipsoid_id = ensure_ellipsoid(*scene_manager, rendering_manager, *parent_id);
                    if (!ellipsoid_id)
                        return json{{"error", ellipsoid_id.error()}};

                    return ellipsoid_info_json(*scene_manager, rendering_manager, *ellipsoid_id);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "ellipsoid.get",
                .description = "Inspect an ellipsoid helper in the shared GUI scene",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional ellipsoid node or parent node name; defaults to the current selected ellipsoid"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");

                return post_and_wait(viewer_impl, [viewer_impl, requested_node]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto ellipsoid_id = resolve_ellipsoid_id(*scene_manager, requested_node);
                    if (!ellipsoid_id)
                        return json{{"error", ellipsoid_id.error()}};

                    return ellipsoid_info_json(*scene_manager, rendering_manager, *ellipsoid_id);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "ellipsoid.set",
                .description = "Update ellipsoid radii, transform, or render toggles in the shared GUI scene",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional ellipsoid node or parent node name; defaults to the current selected ellipsoid"}}},
                        {"radii", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional ellipsoid radii"}}},
                        {"translation", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional local XYZ translation"}}},
                        {"rotation", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional local XYZ Euler rotation in radians"}}},
                        {"scale", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Optional local XYZ scale"}}},
                        {"inverse", json{{"type", "boolean"}, {"description", "Invert the ellipsoid selection volume"}}},
                        {"enabled", json{{"type", "boolean"}, {"description", "Enable ellipsoid filtering for this helper"}}},
                        {"show", json{{"type", "boolean"}, {"description", "Show ellipsoids in the viewport"}}},
                        {"use", json{{"type", "boolean"}, {"description", "Use ellipsoid filtering in rendering"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");
                auto radii = optional_vec3_arg(args, "radii");
                if (!radii)
                    return json{{"error", radii.error()}};
                auto translation = optional_vec3_arg(args, "translation");
                if (!translation)
                    return json{{"error", translation.error()}};
                auto rotation = optional_vec3_arg(args, "rotation");
                if (!rotation)
                    return json{{"error", rotation.error()}};
                auto scale = optional_vec3_arg(args, "scale");
                if (!scale)
                    return json{{"error", scale.error()}};

                const bool has_inverse = args.contains("inverse");
                const bool has_enabled = args.contains("enabled");
                const bool has_show = args.contains("show");
                const bool has_use = args.contains("use");

                if (!radii->has_value() && !translation->has_value() && !rotation->has_value() &&
                    !scale->has_value() && !has_inverse && !has_enabled && !has_show && !has_use) {
                    return json{{"error", "No ellipsoid fields were provided"}};
                }

                const bool inverse = has_inverse ? args["inverse"].get<bool>() : false;
                const bool enabled = has_enabled ? args["enabled"].get<bool>() : false;
                const bool show = has_show ? args["show"].get<bool>() : false;
                const bool use = has_use ? args["use"].get<bool>() : false;

                return post_and_wait(viewer_impl, [viewer_impl, requested_node, radii = *radii, translation = *translation, rotation = *rotation, scale = *scale, has_inverse, inverse, has_enabled, enabled, has_show, show, has_use, use]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto ellipsoid_id = resolve_ellipsoid_id(*scene_manager, requested_node);
                    if (!ellipsoid_id)
                        return json{{"error", ellipsoid_id.error()}};

                    vis::cap::EllipsoidUpdate update;
                    update.radii = radii;
                    update.translation = translation;
                    update.rotation = rotation;
                    update.scale = scale;
                    update.has_inverse = has_inverse;
                    update.inverse = inverse;
                    update.has_enabled = has_enabled;
                    update.enabled = enabled;
                    update.has_show = has_show;
                    update.show = show;
                    update.has_use = has_use;
                    update.use = use;

                    if (auto result = vis::cap::updateEllipsoid(
                            *scene_manager, rendering_manager, *ellipsoid_id, update);
                        !result) {
                        return json{{"error", result.error()}};
                    }

                    return ellipsoid_info_json(*scene_manager, rendering_manager, *ellipsoid_id);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "ellipsoid.fit",
                .description = "Fit an ellipsoid helper to its parent node bounds",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional ellipsoid node or parent node name; defaults to the current selected ellipsoid"}}},
                        {"use_percentile", json{{"type", "boolean"}, {"description", "Use percentile bounds instead of strict min/max (default: true)"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");
                const bool use_percentile = args.value("use_percentile", true);

                return post_and_wait(viewer_impl, [viewer_impl, requested_node, use_percentile]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto ellipsoid_id = resolve_ellipsoid_id(*scene_manager, requested_node);
                    if (!ellipsoid_id)
                        return json{{"error", ellipsoid_id.error()}};

                    if (auto result = fit_ellipsoid_to_parent(*scene_manager, rendering_manager, *ellipsoid_id, use_percentile); !result)
                        return json{{"error", result.error()}};

                    return ellipsoid_info_json(*scene_manager, rendering_manager, *ellipsoid_id);
                });
            });

        registry.register_tool(
            McpTool{
                .name = "ellipsoid.reset",
                .description = "Reset an ellipsoid helper to default radii and identity transform",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional ellipsoid node or parent node name; defaults to the current selected ellipsoid"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");

                return post_and_wait(viewer_impl, [viewer_impl, requested_node]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto ellipsoid_id = resolve_ellipsoid_id(*scene_manager, requested_node);
                    if (!ellipsoid_id)
                        return json{{"error", ellipsoid_id.error()}};

                    if (auto result = reset_ellipsoid(*scene_manager, rendering_manager, *ellipsoid_id); !result)
                        return json{{"error", result.error()}};

                    return ellipsoid_info_json(*scene_manager, rendering_manager, *ellipsoid_id);
                });
            });

        // --- Editor tools ---

        registry.register_tool(
            McpTool{
                .name = "editor.set_code",
                .description = "Populate the visible Python editor with code in the integrated Python console",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"code", json{{"type", "string"}, {"description", "Python code to place into the visible editor"}}},
                        {"show_console", json{{"type", "boolean"}, {"description", "Show the Python console window (default: true)"}}}},
                    .required = {"code"}}},
            [viewer_impl](const json& args) -> json {
                const std::string code = args["code"].get<std::string>();
                const bool show_console_window = args.value("show_console", true);

                return post_and_wait(viewer_impl, [code, show_console_window]() -> json {
                    if (show_console_window)
                        show_python_console();

                    auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                    auto* const editor = console.getEditor();
                    if (!editor)
                        return json{{"error", "Python editor not initialized"}};

                    console.setEditorText(code);
                    console.focusEditor();
                    console.setModified(true);

                    return json{
                        {"success", true},
                        {"chars", static_cast<int64_t>(code.size())},
                        {"running", console.isScriptRunning()},
                    };
                });
            });

        registry.register_tool(
            McpTool{
                .name = "editor.run",
                .description = "Run code through the integrated Python console, optionally wait for completion, and return the latest captured output",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"code", json{{"type", "string"}, {"description", "Optional Python code to set and run; defaults to current editor contents"}}},
                        {"show_console", json{{"type", "boolean"}, {"description", "Show the Python console window (default: true)"}}},
                        {"wait_for_completion", json{{"type", "boolean"}, {"description", "Wait for the script to finish before returning when possible (default: true)"}}},
                        {"wait_for_output", json{{"type", "boolean"}, {"description", "Wait until output appears or the script finishes before returning (default: true)"}}},
                        {"timeout_ms", json{{"type", "integer"}, {"description", "Maximum time to wait for completion/output before returning a partial snapshot (default: 2000)"}}},
                        {"output_max_chars", json{{"type", "integer"}, {"description", "Maximum output characters to include in the response (default: 20000)"}}},
                        {"output_tail", json{{"type", "boolean"}, {"description", "Return the newest output when truncating (default: true)"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const auto code = optional_string_arg(args, "code");
                const bool show_console_window = args.value("show_console", true);
                const bool wait_for_completion = args.value("wait_for_completion", true);
                const bool wait_for_output = args.value("wait_for_output", true);
                const int timeout_ms = args.value("timeout_ms", 2000);
                const int output_max_chars = args.value("output_max_chars", 20000);
                const bool output_tail = args.value("output_tail", true);

                auto response = post_and_wait(viewer_impl, [code, show_console_window]() -> json {
                    if (show_console_window)
                        show_python_console();

                    auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                    auto* const editor = console.getEditor();
                    if (!editor)
                        return json{{"error", "Python editor not initialized"}};
                    if (console.isScriptRunning())
                        return json{{"error", "A script is already running"}};

                    std::string code_to_run;
                    if (code) {
                        console.setEditorText(*code);
                        console.focusEditor();
                        console.setModified(true);
                        code_to_run = *code;
                    } else {
                        code_to_run = console.getEditorTextStripped();
                    }

                    if (code_to_run.empty())
                        return json{{"error", "Editor is empty"}};

                    console.runScriptAsync(code_to_run);
                    return json{
                        {"success", true},
                        {"chars", static_cast<int64_t>(code_to_run.size())},
                        {"running", console.isScriptRunning()},
                    };
                });

                if (response.contains("error"))
                    return response;

                auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                auto observed = editor_output_response_json(
                    console,
                    wait_for_completion,
                    wait_for_output,
                    timeout_ms,
                    output_max_chars,
                    output_tail);
                for (auto& [key, value] : observed.items()) {
                    response[key] = std::move(value);
                }

                response["wait_for_completion"] = wait_for_completion;
                response["wait_for_output"] = wait_for_output;
                response["timeout_ms"] = timeout_ms;
                return response;
            });

        registry.register_tool(
            McpTool{
                .name = "editor.get_code",
                .description = "Read the current contents of the visible integrated Python editor",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"max_chars", json{{"type", "integer"}, {"description", "Maximum characters to return; defaults to no limit"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const int max_chars = args.value("max_chars", -1);

                return post_and_wait(viewer_impl, [max_chars]() -> json {
                    auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                    auto* const editor = console.getEditor();
                    if (!editor)
                        return json{{"error", "Python editor not initialized"}};

                    std::string code = console.getEditorText();
                    auto result = text_payload_json(code, max_chars, false);
                    result["success"] = true;
                    result["total_chars"] = static_cast<int64_t>(code.size());
                    result["modified"] = console.isModified();
                    if (!console.getScriptPath().empty())
                        result["path"] = console.getScriptPath().string();
                    return result;
                });
            });

        registry.register_tool(
            McpTool{
                .name = "editor.get_output",
                .description = "Read captured output from the integrated Python console output terminal",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"max_chars", json{{"type", "integer"}, {"description", "Maximum characters to return (default: 20000)"}}},
                        {"tail", json{{"type", "boolean"}, {"description", "Return the most recent output when truncated (default: true)"}}}},
                    .required = {}}},
            [viewer_impl](const json& args) -> json {
                const int max_chars = args.value("max_chars", 20000);
                const bool tail = args.value("tail", true);

                return post_and_wait(viewer_impl, [max_chars, tail]() -> json {
                    auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                    auto* const output = console.getOutputTerminal();
                    if (!output)
                        return json{{"error", "Python output terminal not initialized"}};

                    std::string text = console.getOutputText();
                    auto result = editor_output_json(text, max_chars, tail);
                    result["success"] = true;
                    result["running"] = console.isScriptRunning();
                    return result;
                });
            });

        registry.register_tool(
            McpTool{
                .name = "editor.wait",
                .description = "Wait for the currently running editor script to finish or emit output, then return the latest output snapshot",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"wait_for_completion", json{{"type", "boolean"}, {"description", "Wait for the script to finish before returning when possible (default: true)"}}},
                        {"wait_for_output", json{{"type", "boolean"}, {"description", "Wait until output appears or the script finishes before returning (default: true)"}}},
                        {"timeout_ms", json{{"type", "integer"}, {"description", "Maximum time to wait before returning a partial snapshot (default: 2000)"}}},
                        {"output_max_chars", json{{"type", "integer"}, {"description", "Maximum output characters to include in the response (default: 20000)"}}},
                        {"output_tail", json{{"type", "boolean"}, {"description", "Return the newest output when truncating (default: true)"}}}},
                    .required = {}}},
            [](const json& args) -> json {
                const bool wait_for_completion = args.value("wait_for_completion", true);
                const bool wait_for_output = args.value("wait_for_output", true);
                const int timeout_ms = args.value("timeout_ms", 2000);
                const int output_max_chars = args.value("output_max_chars", 20000);
                const bool output_tail = args.value("output_tail", true);

                auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                json response{
                    {"success", true},
                    {"wait_for_completion", wait_for_completion},
                    {"wait_for_output", wait_for_output},
                    {"timeout_ms", timeout_ms},
                };

                auto observed = editor_output_response_json(
                    console,
                    wait_for_completion,
                    wait_for_output,
                    timeout_ms,
                    output_max_chars,
                    output_tail);
                for (auto& [key, value] : observed.items()) {
                    response[key] = std::move(value);
                }
                return response;
            });

        registry.register_tool(
            McpTool{
                .name = "editor.is_running",
                .description = "Check whether the integrated Python console is currently running a script",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, []() -> json {
                    auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                    return json{
                        {"success", true},
                        {"running", console.isScriptRunning()},
                        {"modified", console.isModified()},
                    };
                });
            });

        registry.register_tool(
            McpTool{
                .name = "editor.interrupt",
                .description = "Interrupt the currently running script in the integrated Python console",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [viewer_impl](const json&) -> json {
                return post_and_wait(viewer_impl, []() -> json {
                    auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                    const bool was_running = console.isScriptRunning();
                    if (was_running)
                        console.interruptScript();

                    return json{
                        {"success", true},
                        {"was_running", was_running},
                        {"running", console.isScriptRunning()},
                    };
                });
            });

        // --- Event, Tensor, and Sequencer tools ---

        registry.register_tool(
            McpTool{
                .name = "events.subscribe",
                .description = "Create a polling subscription for shared scene/training/render events",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"types", json{{"type", "array"}, {"items", json{{"type", "string"}}}, {"description", "Event types to receive; omit or use ['*'] for all supported types"}}},
                        {"max_queue", json{{"type", "integer"}, {"description", "Maximum queued events to retain before dropping oldest events (default: 256)"}}}},
                    .required = {}}},
            [](const json& args) -> json {
                std::vector<std::string> types;
                if (args.contains("types")) {
                    const auto& value = args["types"];
                    if (!value.is_array())
                        return json{{"error", "Field 'types' must be an array of strings"}};
                    types.reserve(value.size());
                    for (const auto& item : value)
                        types.push_back(item.get<std::string>());
                }
                if (types.empty())
                    types.push_back("*");

                const size_t max_queue = static_cast<size_t>(args.value("max_queue", 256));
                auto subscription_id = EventSubscriptionRegistry::instance().subscribe(types, max_queue);
                if (!subscription_id)
                    return json{{"error", subscription_id.error()}};

                return json{
                    {"success", true},
                    {"subscription_id", *subscription_id},
                    {"types", types},
                    {"max_queue", static_cast<int64_t>(max_queue)},
                    {"supported_types", mcp_subscription_event_types_json()},
                };
            });

        registry.register_tool(
            McpTool{
                .name = "events.poll",
                .description = "Poll queued events for a subscription created with events.subscribe",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"subscription_id", json{{"type", "integer"}, {"description", "Subscription identifier"}}},
                        {"max_events", json{{"type", "integer"}, {"description", "Maximum queued events to return (default: 100)"}}},
                        {"clear", json{{"type", "boolean"}, {"description", "Remove returned events from the queue (default: true)"}}}},
                    .required = {"subscription_id"}}},
            [](const json& args) -> json {
                const int64_t subscription_id = args["subscription_id"].get<int64_t>();
                const size_t max_events = static_cast<size_t>(args.value("max_events", 100));
                const bool clear = args.value("clear", true);
                return EventSubscriptionRegistry::instance().poll(subscription_id, max_events, clear);
            });

        registry.register_tool(
            McpTool{
                .name = "events.unsubscribe",
                .description = "Remove an events polling subscription",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"subscription_id", json{{"type", "integer"}, {"description", "Subscription identifier"}}}},
                    .required = {"subscription_id"}}},
            [](const json& args) -> json {
                const int64_t subscription_id = args["subscription_id"].get<int64_t>();
                return json{
                    {"success", EventSubscriptionRegistry::instance().unsubscribe(subscription_id)},
                    {"subscription_id", subscription_id},
                };
            });

        registry.register_tool(
            McpTool{
                .name = "events.list",
                .description = "List active event subscriptions and supported event types",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [](const json&) -> json {
                return EventSubscriptionRegistry::instance().list();
            });

        registry.register_tool(
            McpTool{
                .name = "gaussians.read",
                .description = "Read raw gaussian parameter tensors for a node and index set",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional gaussian node name; defaults to the selected or training node"}}},
                        {"fields", json{{"type", "array"}, {"items", json{{"type", "string"}}}, {"description", "Field names to read (means, scales/scaling_raw, rotations/rotation_raw, opacities/opacity_raw, sh0, shN)"}}},
                        {"indices", json{{"type", "array"}, {"items", json{{"type", "integer"}}}, {"description", "Optional gaussian indices to read"}}},
                        {"limit", json{{"type", "integer"}, {"description", "When indices are omitted, read the first N rows (default: 256)"}}}},
                    .required = {"fields"}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");
                if (!args.contains("fields") || !args["fields"].is_array())
                    return json{{"error", "Field 'fields' must be an array of field names"}};

                std::vector<std::string> fields;
                fields.reserve(args["fields"].size());
                for (const auto& item : args["fields"])
                    fields.push_back(item.get<std::string>());

                std::optional<std::vector<int>> indices;
                if (args.contains("indices")) {
                    auto parsed = parse_int_array(args["indices"], "indices");
                    if (!parsed)
                        return json{{"error", parsed.error()}};
                    indices = std::move(*parsed);
                }
                const int limit = args.value("limit", 256);

                return post_and_wait(viewer_impl, [viewer_impl, requested_node, fields = std::move(fields), indices = std::move(indices), limit]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto node_name = resolve_gaussian_node_name(*scene_manager, requested_node);
                    if (!node_name)
                        return json{{"error", node_name.error()}};

                    const auto& scene = scene_manager->getScene();
                    const auto* const node = scene.getNode(*node_name);
                    if (!node || !node->model)
                        return json{{"error", "Gaussian node not found: " + *node_name}};

                    std::vector<int> resolved_indices = indices.value_or(std::vector<int>{});
                    if (!indices) {
                        const int max_count = std::max(0, std::min(limit, static_cast<int>(node->model->size())));
                        resolved_indices.reserve(static_cast<size_t>(max_count));
                        for (int i = 0; i < max_count; ++i)
                            resolved_indices.push_back(i);
                    }

                    for (const int index : resolved_indices) {
                        if (index < 0 || static_cast<size_t>(index) >= node->model->size())
                            return json{{"error", "Gaussian index out of range: " + std::to_string(index)}};
                    }

                    const auto index_tensor = core::Tensor::from_vector(
                        resolved_indices,
                        {resolved_indices.size()},
                        node->model->means_raw().device());

                    json field_payloads = json::object();
                    for (const auto& field_name : fields) {
                        const auto* const field = resolve_gaussian_field(*node->model, field_name);
                        if (!field)
                            return json{{"error", "Unsupported gaussian field: " + field_name}};

                        field_payloads[field_name] = tensor_payload_json(field->index_select(0, index_tensor));
                    }

                    return json{
                        {"success", true},
                        {"node", *node_name},
                        {"count", static_cast<int64_t>(resolved_indices.size())},
                        {"indices", resolved_indices},
                        {"fields", field_payloads},
                    };
                });
            });

        registry.register_tool(
            McpTool{
                .name = "gaussians.write",
                .description = "Write raw gaussian parameter tensor rows for a node and index set",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"node", json{{"type", "string"}, {"description", "Optional gaussian node name; defaults to the selected or training node"}}},
                        {"field", json{{"type", "string"}, {"description", "Field name to update (means, scales/scaling_raw, rotations/rotation_raw, opacities/opacity_raw, sh0, shN)"}}},
                        {"indices", json{{"type", "array"}, {"items", json{{"type", "integer"}}}, {"description", "Gaussian row indices to update"}}},
                        {"values", json{{"type", "array"}, {"items", json{{"type", "number"}}}, {"description", "Flat row-major values for the selected tensor slice"}}}},
                    .required = {"field", "indices", "values"}}},
            [viewer_impl](const json& args) -> json {
                const auto requested_node = optional_string_arg(args, "node");
                const std::string field_name = args["field"].get<std::string>();
                auto indices = parse_int_array(args["indices"], "indices");
                if (!indices)
                    return json{{"error", indices.error()}};
                auto values = parse_float_array(args["values"], "values");
                if (!values)
                    return json{{"error", values.error()}};

                return post_and_wait(viewer_impl, [viewer_impl, requested_node, field_name, indices = std::move(*indices), values = std::move(*values)]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    auto* const rendering_manager = viewer_impl->getRenderingManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto node_name = resolve_gaussian_node_name(*scene_manager, requested_node);
                    if (!node_name)
                        return json{{"error", node_name.error()}};

                    auto& scene = scene_manager->getScene();
                    auto* const node = scene.getMutableNode(*node_name);
                    if (!node || !node->model)
                        return json{{"error", "Gaussian node not found: " + *node_name}};

                    auto* const field = resolve_gaussian_field(*node->model, field_name);
                    if (!field)
                        return json{{"error", "Unsupported gaussian field: " + field_name}};

                    for (const int index : indices) {
                        if (index < 0 || static_cast<size_t>(index) >= node->model->size())
                            return json{{"error", "Gaussian index out of range: " + std::to_string(index)}};
                    }

                    const size_t row_width = product_of_tail_dims(*field);
                    const size_t expected_values = row_width * indices.size();
                    if (values.size() != expected_values) {
                        return json{{"error", "Field slice expects " + std::to_string(expected_values) +
                                                  " values but received " + std::to_string(values.size())}};
                    }

                    const auto& field_shape = field->shape();
                    if (field_shape.rank() == 0)
                        return json{{"error", "Gaussian tensor field has invalid rank"}};
                    auto shape_dims = field_shape.dims();
                    shape_dims[0] = indices.size();

                    const auto index_tensor = core::Tensor::from_vector(indices, {indices.size()}, field->device());
                    const auto src_tensor = core::Tensor::from_vector(values, core::TensorShape(shape_dims), field->device());
                    field->index_copy_(0, index_tensor, src_tensor);

                    scene.notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
                    if (rendering_manager)
                        rendering_manager->markDirty(vis::DirtyFlag::SPLATS | vis::DirtyFlag::OVERLAY);

                    return json{
                        {"success", true},
                        {"node", *node_name},
                        {"field", field_name},
                        {"updated_count", static_cast<int64_t>(indices.size())},
                    };
                });
            });

        register_gui_sequencer_tools(
            registry,
            viewer,
            SequencerToolBackend{
                .ensure_ready =
                    [viewer_impl]() -> std::expected<void, std::string> {
                        if (!viewer_impl)
                            return std::unexpected("Sequencer tools require a GUI visualizer");
                        if (!viewer_impl->getSceneManager())
                            return std::unexpected("Scene manager not initialized");
                        if (!viewer_impl->getGuiManager())
                            return std::unexpected("GUI manager not initialized");
                        return {};
                    },
                .controller =
                    [viewer_impl]() -> vis::SequencerController* {
                        if (!viewer_impl)
                            return nullptr;
                        auto* const gui_manager = viewer_impl->getGuiManager();
                        return gui_manager ? &gui_manager->sequencer() : nullptr;
                    },
                .is_visible = []() { return python::is_sequencer_visible(); },
                .set_visible = [](const bool visible) { python::set_sequencer_visible(visible); },
                .ui_state = []() { return python::get_sequencer_ui_state(); },
                .add_keyframe = []() { core::events::cmd::SequencerAddKeyframe{}.emit(); },
                .update_selected_keyframe = []() { core::events::cmd::SequencerUpdateKeyframe{}.emit(); },
                .select_keyframe = [](const size_t index) { core::events::cmd::SequencerSelectKeyframe{.keyframe_index = index}.emit(); },
                .go_to_keyframe = [](const size_t index) { core::events::cmd::SequencerGoToKeyframe{.keyframe_index = index}.emit(); },
                .delete_keyframe = [](const size_t index) { core::events::cmd::SequencerDeleteKeyframe{.keyframe_index = index}.emit(); },
                .set_keyframe_easing =
                    [](const size_t index, const int easing) {
                        core::events::cmd::SequencerSetKeyframeEasing{.keyframe_index = index, .easing_type = easing}.emit();
                    },
                .play_pause = []() { core::events::cmd::SequencerPlayPause{}.emit(); },
                .clear = []() { python::clear_keyframes(); },
                .save_path = [](const std::string& path) { return python::save_camera_path(path); },
                .load_path = [](const std::string& path) { return python::load_camera_path(path); },
                .set_playback_speed = [](const float speed) { python::set_playback_speed(speed); },
            });

        // --- Plugin tools ---

        registry.register_tool(
            McpTool{
                .name = "plugin.invoke",
                .description = "Invoke a plugin capability by name. Use plugin.list to see available capabilities.",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"capability", json{{"type", "string"}, {"description", "Capability name (e.g., 'selection.by_text')"}}},
                        {"args", json{{"type", "object"}, {"description", "Arguments to pass to the capability"}}}},
                    .required = {"capability"}}},
            [viewer](const json& args) -> json {
                const auto capability = args.value("capability", "");
                if (capability.empty())
                    return json{{"error", "Missing capability name"}};

                const std::string args_json = args.contains("args") ? args["args"].dump() : "{}";

                return post_and_wait(viewer, [viewer, capability, args_json]() -> json {
                    python::SceneContextGuard ctx(&viewer->getScene());
                    auto result = python::invoke_capability(capability, args_json);
                    if (!result.success)
                        return json{{"success", false}, {"error", result.error}};

                    if (auto* const rendering_manager = viewer->getRenderingManager())
                        rendering_manager->markDirty(vis::DirtyFlag::ALL);

                    try {
                        return json::parse(result.result_json);
                    } catch (const std::exception& e) {
                        LOG_WARN("Failed to parse capability result: {}", e.what());
                        return json{{"success", true}, {"raw_result", result.result_json}};
                    }
                });
            });

        registry.register_tool(
            McpTool{
                .name = "plugin.list",
                .description = "List all registered plugin capabilities",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [](const json&) -> json {
                auto capabilities = python::list_capabilities();
                json result = json::array();
                for (const auto& cap : capabilities) {
                    result.push_back({{"name", cap.name}, {"description", cap.description}, {"plugin", cap.plugin_name}});
                }
                return json{{"success", true}, {"capabilities", result}};
            });

        // --- LLM-powered tools ---

        registry.register_tool(
            McpTool{
                .name = "training.ask_advisor",
                .description = "Ask an LLM for training advice based on current state and render",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"problem", json{{"type", "string"}, {"description", "Description of the problem or question"}}},
                        {"include_render", json{{"type", "boolean"}, {"description", "Include current render in request (default: true)"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index for render (default: 0)"}}}},
                    .required = {}}},
            [viewer](const json& args) -> json {
                auto api_key = mcp::LLMClient::load_api_key_from_env();
                if (!api_key)
                    return json{{"error", api_key.error()}};

                mcp::LLMClient client;
                client.set_api_key(*api_key);

                auto* cc = event::command_center();
                if (!cc)
                    return json{{"error", "Training system not initialized"}};

                auto snapshot = cc->snapshot();

                std::string base64_render;
                bool include_render = args.value("include_render", true);
                if (include_render) {
                    int camera_index = args.value("camera_index", 0);
                    auto render_result = post_and_wait(viewer, [viewer, camera_index]() {
                        return render_scene_to_base64(viewer->getScene(), camera_index);
                    });
                    if (render_result)
                        base64_render = *render_result;
                }

                std::string problem = args.value("problem", "");

                auto result = mcp::ask_training_advisor(
                    client,
                    snapshot.iteration,
                    snapshot.loss,
                    snapshot.num_gaussians,
                    base64_render,
                    problem);

                if (!result)
                    return json{{"error", result.error()}};

                json response;
                response["success"] = result->success;
                response["advice"] = result->content;
                response["model"] = result->model;
                response["input_tokens"] = result->input_tokens;
                response["output_tokens"] = result->output_tokens;
                if (!result->success)
                    response["error"] = result->error;
                return response;
            });

        registry.register_tool(
            McpTool{
                .name = "selection.by_description",
                .description = "Select Gaussians by natural language description using LLM vision",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"description", json{{"type", "string"}, {"description", "Natural language description of what to select (e.g., 'the bicycle wheel')"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index for rendering (default: 0)"}}}},
                    .required = {"description"}}},
            [viewer_impl](const json& args) -> json {
                auto api_key = mcp::LLMClient::load_api_key_from_env();
                if (!api_key)
                    return json{{"error", api_key.error()}};

                const int camera_index = args.value("camera_index", 0);
                const std::string description = args["description"].get<std::string>();

                auto render_result = post_and_wait(viewer_impl, [viewer_impl, camera_index]() {
                    return render_scene_to_base64(viewer_impl->getScene(), camera_index);
                });
                if (!render_result)
                    return json{{"error", render_result.error()}};

                mcp::LLMClient client;
                client.set_api_key(*api_key);

                mcp::LLMRequest request;
                request.prompt = "Look at this 3D scene render. I need you to identify the bounding box for: \"" + description + "\"\n\n"
                                                                                                                                 "Return ONLY a JSON object with the bounding box coordinates in pixel space:\n"
                                                                                                                                 "{\"x0\": <left>, \"y0\": <top>, \"x1\": <right>, \"y1\": <bottom>}\n\n"
                                                                                                                                 "The coordinates should be integers representing pixel positions. "
                                                                                                                                 "If you cannot identify the object, return: {\"error\": \"Object not found\"}";
                request.attachments.push_back(mcp::ImageAttachment{.base64_data = *render_result, .media_type = "image/png"});
                request.temperature = 0.0f;
                request.max_tokens = 256;

                auto response = client.complete(request);
                if (!response)
                    return json{{"error", response.error()}};

                if (!response->success)
                    return json{{"error", response->error}};

                json bbox;
                try {
                    auto content = response->content;
                    auto json_start = content.find('{');
                    auto json_end = content.rfind('}');
                    if (json_start == std::string::npos || json_end == std::string::npos)
                        return json{{"error", "LLM response did not contain valid JSON"}};
                    bbox = json::parse(content.substr(json_start, json_end - json_start + 1));
                } catch (const std::exception& e) {
                    return json{{"error", std::string("Failed to parse LLM response: ") + e.what()}};
                }

                if (bbox.contains("error"))
                    return json{{"error", bbox["error"].get<std::string>()}};

                if (!bbox.contains("x0") || !bbox.contains("y0") || !bbox.contains("x1") || !bbox.contains("y1"))
                    return json{{"error", "LLM response missing bounding box coordinates"}};

                const float x0 = bbox["x0"].get<float>();
                const float y0 = bbox["y0"].get<float>();
                const float x1 = bbox["x1"].get<float>();
                const float y1 = bbox["y1"].get<float>();

                return post_and_wait(viewer_impl, [viewer_impl, x0, y0, x1, y1, camera_index, bbox, description]() -> json {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return json{{"error", "Scene manager not initialized"}};

                    auto result = selection_result_json(*scene_manager,
                                                        scene_manager->selectRect(x0, y0, x1, y1, "replace", camera_index));
                    if (!result.value("success", false))
                        return result;
                    result["bounding_box"] = bbox;
                    result["description"] = description;
                    return result;
                });
            });

        LOG_INFO("Registered GUI-native MCP scene tools");
    }

    void register_gui_scene_resources(vis::Visualizer* viewer) {
        assert(viewer);
        auto* const viewer_impl = viewer;
        auto& registry = ResourceRegistry::instance();

        register_generic_gui_operator_resources(registry, viewer);
        register_generic_gui_runtime_resources(registry, viewer);
        register_generic_gui_ui_resources(registry, viewer);

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://editor/code",
                .name = "Editor Code",
                .description = "Current contents of the integrated Python editor",
                .mime_type = "text/x-python"},
            [viewer_impl](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer_impl, [uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                    auto* const editor = console.getEditor();
                    if (!editor)
                        return std::unexpected("Python editor not initialized");

                    return std::vector<McpResourceContent>{
                        McpResourceContent{
                            .uri = uri,
                            .mime_type = "text/x-python",
                            .content = console.getEditorText()}};
                });
            });

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://editor/output",
                .name = "Editor Output",
                .description = "Captured output from the integrated Python editor console",
                .mime_type = "text/plain"},
            [viewer_impl](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer_impl, [uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                    auto* const output = console.getOutputTerminal();
                    if (!output)
                        return std::unexpected("Python output terminal not initialized");

                    return std::vector<McpResourceContent>{
                        McpResourceContent{
                            .uri = uri,
                            .mime_type = "text/plain",
                            .content = console.getOutputText()}};
                });
            });

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://editor/state",
                .name = "Editor State",
                .description = "Current state of the integrated Python editor and console",
                .mime_type = "application/json"},
            [viewer_impl](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer_impl, [uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                    auto* const editor = console.getEditor();
                    auto* const output = console.getOutputTerminal();
                    if (!editor || !output)
                        return std::unexpected("Python editor resources are not initialized");

                    json payload{
                        {"running", console.isScriptRunning()},
                        {"modified", console.isModified()},
                        {"output_total_chars", static_cast<int64_t>(console.getOutputText().size())},
                    };
                    if (!console.getScriptPath().empty())
                        payload["path"] = console.getScriptPath().string();

                    return single_json_resource(uri, std::move(payload));
                });
            });

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://render/current",
                .name = "Current Render",
                .description = "Base64-encoded PNG capture of the current GUI viewport",
                .mime_type = "image/png"},
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                auto result = post_and_wait(viewer, [viewer]() {
                    return capture_live_viewport_to_base64(viewer);
                });
                if (!result)
                    return std::unexpected(result.error());

                return single_blob_resource(uri, "image/png", *result);
            });

        registry.register_resource_prefix(
            "lichtfeld://render/",
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                if (uri != "lichtfeld://render/current")
                    return std::unexpected("Unknown resource URI: " + uri);

                auto result = post_and_wait(viewer, [viewer]() {
                    return capture_live_viewport_to_base64(viewer);
                });
                if (!result)
                    return std::unexpected(result.error());

                return single_blob_resource(uri, "image/png", *result);
            });

        registry.register_resource_prefix(
            "lichtfeld://render/camera/",
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                constexpr std::string_view camera_prefix = "lichtfeld://render/camera/";
                int camera_index = 0;
                const auto idx_str = uri.substr(camera_prefix.size());
                try {
                    camera_index = std::stoi(idx_str);
                } catch (...) {
                    return std::unexpected("Invalid camera resource URI: " + uri);
                }

                auto result = post_and_wait(viewer, [viewer, camera_index]() {
                    return render_scene_to_base64(viewer->getScene(), camera_index);
                });
                if (!result)
                    return std::unexpected(result.error());

                return single_blob_resource(uri, "image/png", *result);
            });

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://gaussians/stats",
                .name = "Gaussian Statistics",
                .description = "Statistics about the current GUI scene Gaussian model",
                .mime_type = "application/json"},
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer, [viewer, uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    json payload;
                    auto& scene = viewer->getScene();
                    payload["count"] = scene.getTotalGaussianCount();

                    const auto selection = selection_state_json(scene, 0);
                    payload["selected_count"] = selection.value("selected_count", 0);

                    if (auto* cc = event::command_center()) {
                        auto snapshot = cc->snapshot();
                        payload["is_refining"] = snapshot.is_refining;
                    }

                    return single_json_resource(uri, std::move(payload));
                });
            });

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://selection/current",
                .name = "Current Selection",
                .description = "Current Gaussian selection from the shared GUI scene",
                .mime_type = "application/json"},
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer, [viewer, uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto payload = selection_state_json(viewer->getScene());
                    payload["success"] = true;
                    return single_json_resource(uri, std::move(payload));
                });
            });

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://scene/nodes",
                .name = "Scene Nodes",
                .description = "All nodes from the shared GUI scene",
                .mime_type = "application/json"},
            [viewer_impl](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer_impl, [viewer_impl, uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return std::unexpected("Scene manager not initialized");

                    const auto& scene = scene_manager->getScene();
                    json nodes = json::array();
                    for (const auto* const node : scene.getNodes()) {
                        if (!node)
                            continue;
                        nodes.push_back(node_summary_json(scene, *node));
                    }

                    return single_json_resource(uri, json{{"count", nodes.size()}, {"nodes", std::move(nodes)}});
                });
            });

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://scene/selected_nodes",
                .name = "Selected Scene Nodes",
                .description = "Currently selected nodes from the shared GUI scene",
                .mime_type = "application/json"},
            [viewer_impl](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer_impl, [viewer_impl, uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto* const scene_manager = viewer_impl->getSceneManager();
                    if (!scene_manager)
                        return std::unexpected("Scene manager not initialized");

                    const auto& scene = scene_manager->getScene();
                    json nodes = json::array();
                    for (const auto& name : scene_manager->getSelectedNodeNames()) {
                        if (const auto* const node = scene.getNode(name))
                            nodes.push_back(node_summary_json(scene, *node));
                    }

                    return single_json_resource(uri, json{{"count", nodes.size()}, {"nodes", std::move(nodes)}});
                });
            });

        LOG_INFO("Registered GUI-native MCP scene resources");
    }

} // namespace lfs::app
