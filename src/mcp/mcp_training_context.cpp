/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcp_training_context.hpp"
#include "llm_client.hpp"
#include "mcp_tools.hpp"
#include "render_capture_utils.hpp"
#include "shared_scene_tools.hpp"

#include "core/checkpoint_format.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/logger.hpp"
#include "io/exporter.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"
#include "rendering/gs_rasterizer_tensor.hpp"
#include "rendering/rasterizer/rasterization/include/forward.h"
#include "rendering/rasterizer/rasterization/include/rasterization_api_tensor.h"
#include "training/checkpoint.hpp"
#include "training/dataset.hpp"
#include "training/training_setup.hpp"
#include "visualizer/selection/selection_group_mask.hpp"

#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <limits>
#include <sstream>

namespace lfs::mcp {

    namespace {
        struct ScreenRect {
            float x0;
            float y0;
            float x1;
            float y1;
        };

        [[nodiscard]] ScreenRect normalize_screen_rect(const float x0, const float y0, const float x1, const float y1) {
            return {
                .x0 = std::min(x0, x1),
                .y0 = std::min(y0, y1),
                .x1 = std::max(x0, x1),
                .y1 = std::max(y0, y1),
            };
        }

        [[nodiscard]] std::vector<float> close_screen_polygon(std::vector<float> vertices) {
            if (vertices.size() >= 6 &&
                (vertices[0] != vertices[vertices.size() - 2] ||
                 vertices[1] != vertices[vertices.size() - 1])) {
                vertices.push_back(vertices[0]);
                vertices.push_back(vertices[1]);
            }
            return vertices;
        }

        core::Tensor ensure_cuda_bool_mask(const core::Tensor& mask) {
            auto result = (mask.dtype() == core::DataType::Bool) ? mask : mask.to(core::DataType::Bool);
            if (result.device() != core::Device::CUDA) {
                result = result.cuda();
            }
            return result;
        }

        int64_t count_selected(const core::Tensor& mask) {
            if (!mask.is_valid()) {
                return 0;
            }
            const auto bool_mask = (mask.dtype() == core::DataType::Bool) ? mask : mask.to(core::DataType::Bool);
            return static_cast<int64_t>(bool_mask.sum_scalar());
        }

        std::expected<int64_t, std::string> count_visible_model_gaussians(const core::Scene& scene) {
            int64_t total = 0;
            bool has_model = false;
            for (const auto* node : scene.getVisibleNodes()) {
                if (!node)
                    continue;
                has_model = true;
                total += static_cast<int64_t>(node->gaussian_count.load(std::memory_order_acquire));
            }

            if (!has_model)
                return std::unexpected("No model loaded");

            return total;
        }

        core::Tensor& reset_cuda_bool_scratch(core::Tensor& buffer, const size_t size) {
            const bool needs_realloc = !buffer.is_valid() ||
                                       buffer.device() != core::Device::CUDA ||
                                       buffer.dtype() != core::DataType::Bool ||
                                       buffer.numel() != size;
            if (needs_realloc) {
                buffer = core::Tensor::zeros({size}, core::Device::CUDA, core::DataType::Bool);
                return buffer;
            }

            buffer.zero_();
            return buffer;
        }

        core::Tensor& reset_cuda_uint8_scratch(core::Tensor& buffer, const size_t size) {
            const bool needs_realloc = !buffer.is_valid() ||
                                       buffer.device() != core::Device::CUDA ||
                                       buffer.dtype() != core::DataType::UInt8 ||
                                       buffer.numel() != size;
            if (needs_realloc) {
                buffer = core::Tensor::zeros({size}, core::Device::CUDA, core::DataType::UInt8);
                return buffer;
            }

            buffer.zero_();
            return buffer;
        }

        core::Tensor& acquire_selection_output_buffer(std::array<core::Tensor, 2>& buffers,
                                                      size_t& next_index,
                                                      const size_t size) {
            auto& buffer = reset_cuda_uint8_scratch(buffers[next_index], size);
            next_index = (next_index + 1) % buffers.size();
            return buffer;
        }

        core::Tensor& upload_polygon_vertices_to_cuda(const std::vector<float>& vertices,
                                                      core::Tensor& device_buffer) {
            const size_t num_vertices = vertices.size() / 2;
            const bool needs_realloc = !device_buffer.is_valid() ||
                                       device_buffer.device() != core::Device::CUDA ||
                                       device_buffer.dtype() != core::DataType::Float32 ||
                                       device_buffer.shape().rank() != 2 ||
                                       device_buffer.size(0) != num_vertices ||
                                       device_buffer.size(1) != 2;
            if (needs_realloc) {
                device_buffer = core::Tensor::empty({num_vertices, size_t{2}},
                                                    core::Device::CUDA,
                                                    core::DataType::Float32);
            }

            auto host_view = core::Tensor::from_blob(const_cast<float*>(vertices.data()),
                                                     {num_vertices, size_t{2}},
                                                     core::Device::CPU,
                                                     core::DataType::Float32);
            device_buffer.copy_from(host_view);
            return device_buffer;
        }

        json invoke_plugin_capability(core::Scene* scene,
                                      const std::string& capability,
                                      const std::string& args_json) {
            python::SceneContextGuard ctx(scene);
            auto result = python::invoke_capability(capability, args_json);
            if (!result.success) {
                return json{{"success", false}, {"error", result.error}};
            }

            try {
                return json::parse(result.result_json);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to parse capability result: {}", e.what());
                return json{{"success", true}, {"raw_result", result.result_json}};
            }
        }

        std::expected<int, std::string> pick_headless_ring_gaussian(
            const core::Camera& camera,
            const core::SplatData& model,
            const float x,
            const float y) {

            core::Tensor bg = core::Tensor::zeros({3}, core::Device::CUDA);
            unsigned long long* hovered_depth_id_device = nullptr;
            unsigned long long* hovered_depth_id_host = nullptr;

            const auto cleanup = [&]() {
                if (hovered_depth_id_device) {
                    cudaFree(hovered_depth_id_device);
                    hovered_depth_id_device = nullptr;
                }
                if (hovered_depth_id_host) {
                    cudaFreeHost(hovered_depth_id_host);
                    hovered_depth_id_host = nullptr;
                }
            };

            if (cudaMalloc(&hovered_depth_id_device, sizeof(unsigned long long)) != cudaSuccess) {
                return std::unexpected("Failed to allocate ring hover buffer");
            }
            if (cudaMallocHost(&hovered_depth_id_host, sizeof(unsigned long long)) != cudaSuccess) {
                cleanup();
                return std::unexpected("Failed to allocate ring hover readback buffer");
            }
            if (cudaMemset(hovered_depth_id_device, 0xFF, sizeof(unsigned long long)) != cudaSuccess) {
                cleanup();
                return std::unexpected("Failed to reset ring hover buffer");
            }

            try {
                auto [image, alpha] = rendering::rasterize_tensor(
                    camera,
                    model,
                    bg,
                    -1,      // sh_degree_override
                    false,   // show_rings
                    0.01f,   // ring_width
                    nullptr, // model_transforms
                    nullptr, // transform_indices
                    nullptr, // selection_mask
                    nullptr, // screen_positions_out
                    true,    // cursor_active
                    x,
                    y,
                    0.0f,    // cursor_radius
                    true,    // preview_selection_add_mode
                    nullptr, // preview_selection_out
                    false,   // cursor_saturation_preview
                    0.0f,    // cursor_saturation_amount
                    false,   // show_center_markers
                    nullptr, // crop_box_transform
                    nullptr, // crop_box_min
                    nullptr, // crop_box_max
                    false,   // crop_inverse
                    false,   // crop_desaturate
                    -1,      // crop_parent_node_index
                    nullptr, // ellipsoid_transform
                    nullptr, // ellipsoid_radii
                    false,   // ellipsoid_inverse
                    false,   // ellipsoid_desaturate
                    -1,      // ellipsoid_parent_node_index
                    nullptr, // depth_filter_transform
                    nullptr, // depth_filter_min
                    nullptr, // depth_filter_max
                    false,   // view_volume_cull
                    nullptr, // deleted_mask
                    hovered_depth_id_device,
                    -1); // highlight_gaussian_id
                (void)image;
                (void)alpha;
            } catch (const std::exception& e) {
                cleanup();
                return std::unexpected(std::string("Ring pick render failed: ") + e.what());
            }

            if (cudaMemcpy(hovered_depth_id_host, hovered_depth_id_device,
                           sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) {
                cleanup();
                return std::unexpected("Failed to read back ring hover result");
            }

            constexpr auto NO_HOVERED_RESULT = std::numeric_limits<unsigned long long>::max();
            const unsigned long long packed = *hovered_depth_id_host;
            cleanup();

            if (packed == NO_HOVERED_RESULT) {
                return std::unexpected("No hovered gaussian");
            }

            return static_cast<int>(packed & 0xFFFFFFFFu);
        }

        std::expected<std::pair<core::SplatData*, std::shared_ptr<core::Camera>>, std::string>
        resolve_model_and_camera(const std::shared_ptr<core::Scene>& scene,
                                 int camera_index) {
            if (!scene) {
                return std::unexpected("No scene loaded");
            }

            auto* model = scene->getTrainingModel();
            if (!model) {
                return std::unexpected("No model loaded");
            }

            auto cameras = scene->getAllCameras();
            if (cameras.empty()) {
                return std::unexpected("No cameras available");
            }

            if (camera_index < 0 || camera_index >= static_cast<int>(cameras.size())) {
                camera_index = 0;
            }

            auto camera = cameras[camera_index];
            if (!camera) {
                return std::unexpected("Failed to get camera");
            }

            return std::pair{model, std::move(camera)};
        }

        std::expected<core::Tensor, std::string> compute_screen_positions_for_scene(
            const std::shared_ptr<core::Scene>& scene,
            const int camera_index) {
            auto resolved = resolve_model_and_camera(scene, camera_index);
            if (!resolved) {
                return std::unexpected(resolved.error());
            }

            const auto [model, camera] = *resolved;
            core::Tensor bg = core::Tensor::zeros({3}, core::Device::CUDA);
            core::Tensor screen_positions;

            try {
                auto [image, alpha] = rendering::rasterize_tensor(
                    *camera,
                    *model,
                    bg,
                    -1,      // sh_degree_override
                    false,   // show_rings
                    0.01f,   // ring_width
                    nullptr, // model_transforms
                    nullptr, // transform_indices
                    nullptr, // selection_mask
                    &screen_positions);
                (void)image;
                (void)alpha;

                return screen_positions;
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Screen position computation failed: ") + e.what());
            }
        }

        std::expected<std::string, std::string> render_to_base64_for_scene(
            const std::shared_ptr<core::Scene>& scene,
            const int camera_index,
            const int width,
            const int height) {
            auto resolved = resolve_model_and_camera(scene, camera_index);
            if (!resolved) {
                return std::unexpected(resolved.error());
            }

            const auto [model, camera] = *resolved;
            core::Tensor bg = core::Tensor::zeros({3}, core::Device::CUDA);

            try {
                auto [image, alpha] = rendering::rasterize_tensor(*camera, *model, bg);
                (void)alpha;
                return encode_render_tensor_to_base64(std::move(image), width, height);
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Render failed: ") + e.what());
            }
        }

        std::expected<int64_t, std::string> apply_headless_selection(
            core::Scene& scene,
            core::Tensor& locked_groups_device_mask,
            std::array<core::Tensor, 2>& selection_output_buffers,
            size_t& selection_output_buffer_index,
            const core::Tensor& raw_selection,
            const std::string& mode) {

            auto selection_mask = ensure_cuda_bool_mask(raw_selection);
            if (!selection_mask.is_valid()) {
                return std::unexpected("Invalid selection mask");
            }

            auto locked_groups = vis::selection::upload_locked_group_mask(scene, locked_groups_device_mask);
            if (!locked_groups) {
                return std::unexpected(locked_groups.error());
            }

            const auto existing_mask = scene.getSelectionMask();
            const core::Tensor empty_mask;
            const auto& existing_ref = (existing_mask && existing_mask->is_valid()) ? *existing_mask : empty_mask;
            const auto transform_indices = scene.getTransformIndices();
            const bool add_mode = (mode != "remove");
            const bool replace_mode = (mode == "replace");
            auto& output_mask = acquire_selection_output_buffer(
                selection_output_buffers, selection_output_buffer_index, selection_mask.numel());

            rendering::apply_selection_group_tensor_mask(
                selection_mask,
                existing_ref,
                output_mask,
                scene.getActiveSelectionGroup(),
                *locked_groups,
                add_mode,
                transform_indices.get(),
                {},
                replace_mode);

            auto new_selection = std::make_shared<core::Tensor>(output_mask.clone());
            const int64_t count = count_selected(*new_selection);
            scene.setSelectionMask(new_selection);
            return count;
        }
    } // namespace

    TrainingContext& TrainingContext::instance() {
        static TrainingContext inst;
        return inst;
    }

    TrainingContext::~TrainingContext() {
        shutdown();
    }

    std::expected<void, std::string> TrainingContext::load_dataset(
        const std::filesystem::path& path,
        const core::param::TrainingParameters& params) {

        std::lock_guard lock(mutex_);

        stop_training_locked();

        params_ = params;
        params_.dataset.data_path = path;

        scene_ = std::make_shared<core::Scene>();

        if (auto result = training::loadTrainingDataIntoScene(params_, *scene_); !result) {
            scene_.reset();
            return std::unexpected(result.error());
        }

        if (auto result = training::initializeTrainingModel(params_, *scene_); !result) {
            scene_.reset();
            return std::unexpected(result.error());
        }

        trainer_ = std::make_shared<training::Trainer>(*scene_);

        if (auto result = trainer_->initialize(params_); !result) {
            trainer_.reset();
            scene_.reset();
            return std::unexpected(result.error());
        }

        LOG_INFO("MCP: Loaded dataset from {}", core::path_to_utf8(path));
        return {};
    }

    std::expected<void, std::string> TrainingContext::load_checkpoint(
        const std::filesystem::path& path) {

        std::lock_guard lock(mutex_);

        stop_training_locked();

        auto header_result = core::load_checkpoint_header(path);
        if (!header_result) {
            return std::unexpected(header_result.error());
        }

        auto params_result = core::load_checkpoint_params(path);
        if (!params_result) {
            return std::unexpected(params_result.error());
        }
        params_ = std::move(*params_result);

        auto splat_result = core::load_checkpoint_splat_data(path);
        if (!splat_result) {
            return std::unexpected(splat_result.error());
        }

        scene_ = std::make_shared<core::Scene>();
        scene_->setTrainingModel(
            std::make_unique<core::SplatData>(std::move(*splat_result)),
            "checkpoint");

        trainer_ = std::make_shared<training::Trainer>(*scene_);

        if (auto result = trainer_->initialize(params_); !result) {
            trainer_.reset();
            scene_.reset();
            return std::unexpected(result.error());
        }

        LOG_INFO("MCP: Loaded checkpoint from {}", core::path_to_utf8(path));
        return {};
    }

    std::expected<void, std::string> TrainingContext::save_checkpoint(
        const std::filesystem::path& path) {

        std::lock_guard lock(mutex_);

        if (!trainer_) {
            return std::unexpected("No training session to save");
        }

        auto result = training::save_checkpoint(
            path,
            trainer_->get_current_iteration(),
            trainer_->get_strategy(),
            params_,
            nullptr);

        if (!result) {
            return std::unexpected(result.error());
        }

        LOG_INFO("MCP: Saved checkpoint to {}", core::path_to_utf8(path));
        return {};
    }

    std::expected<void, std::string> TrainingContext::save_ply(
        const std::filesystem::path& path) {

        std::lock_guard lock(mutex_);

        if (!scene_) {
            return std::unexpected("No scene to save");
        }

        auto* model = scene_->getTrainingModel();
        if (!model) {
            return std::unexpected("No model to save");
        }

        io::PlySaveOptions options{.output_path = path, .binary = true};
        auto result = io::save_ply(*model, options);
        if (!result) {
            return std::unexpected(result.error().message);
        }

        LOG_INFO("MCP: Saved PLY to {}", core::path_to_utf8(path));
        return {};
    }

    std::expected<std::string, std::string> TrainingContext::render_to_base64(
        int camera_index,
        int width,
        int height) {
        return render_to_base64_for_scene(scene(), camera_index, width, height);
    }

    std::expected<core::Tensor, std::string> TrainingContext::compute_screen_positions(
        int camera_index) {
        return compute_screen_positions_for_scene(scene(), camera_index);
    }

    std::expected<void, std::string> TrainingContext::start_training() {
        std::lock_guard lock(mutex_);

        if (!trainer_) {
            return std::unexpected("No trainer initialized");
        }

        if (training_thread_) {
            return std::unexpected("Training already running");
        }

        auto trainer = trainer_;
        training_thread_ = std::make_unique<std::jthread>([trainer](std::stop_token stop) {
            auto result = trainer->train(stop);
            if (!result) {
                LOG_ERROR("Training error: {}", result.error());
            }
        });

        LOG_INFO("MCP: Training started");
        return {};
    }

    void TrainingContext::stop_training() {
        std::lock_guard lock(mutex_);
        stop_training_locked();
    }

    void TrainingContext::stop_training_locked() {
        if (training_thread_) {
            training_thread_->request_stop();
            training_thread_.reset();
        }
    }

    void TrainingContext::pause_training() {
        auto trainer = this->trainer();
        if (trainer) {
            trainer->request_pause();
        }
    }

    void TrainingContext::resume_training() {
        auto trainer = this->trainer();
        if (trainer) {
            trainer->request_resume();
        }
    }

    void TrainingContext::shutdown() {
        std::lock_guard lock(mutex_);
        stop_training_locked();
        trainer_.reset();
        scene_.reset();
    }

    void register_scene_tools() {
        register_shared_scene_tools(SharedSceneToolBackend{
            .runtime = "headless",
            .thread_affinity = "training_context",
            .load_dataset =
                [](const std::filesystem::path& path,
                   const core::param::TrainingParameters& params) {
                    return TrainingContext::instance().load_dataset(path, params);
                },
            .load_checkpoint =
                [](const std::filesystem::path& path) {
                    return TrainingContext::instance().load_checkpoint(path);
                },
            .save_checkpoint =
                [](const std::optional<std::filesystem::path>& path)
                -> std::expected<std::filesystem::path, std::string> {
                auto& ctx = TrainingContext::instance();
                auto trainer = ctx.trainer();
                if (!trainer)
                    return std::unexpected("No training session to save");

                const bool training_active = ctx.is_training();
                if (training_active) {
                    if (path) {
                        return std::unexpected(
                            "Custom checkpoint output paths are not supported while training is active");
                    }
                    return std::unexpected(
                        "Cannot report checkpoint save success while training is active; "
                        "use the async training checkpoint action or stop training first");
                }

                if (path) {
                    if (auto result = ctx.save_checkpoint(*path); !result)
                        return std::unexpected(result.error());
                    return *path;
                }

                if (auto result = trainer->save_checkpoint(trainer->get_current_iteration()); !result)
                    return std::unexpected(result.error());
                return trainer->get_output_path();
            },
            .save_ply =
                [](const std::filesystem::path& path) {
                    return TrainingContext::instance().save_ply(path);
                },
            .start_training =
                []() {
                    return TrainingContext::instance().start_training();
                },
            .render_capture =
                [](std::optional<int> camera_index, int width, int height)
                -> std::expected<std::string, std::string> {
                if (!camera_index) {
                    return std::unexpected(
                        "camera_index is required in the training runtime; "
                        "live viewport capture is only available in the GUI runtime");
                }
                return TrainingContext::instance().render_to_base64(*camera_index, width, height);
            },
            .gaussian_count =
                []() -> std::expected<int64_t, std::string> {
                auto scene = TrainingContext::instance().scene();
                if (!scene)
                    return std::unexpected("No scene loaded");
                return count_visible_model_gaussians(*scene);
            }});

        auto& registry = ToolRegistry::instance();

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
            [](const json& args) -> json {
                auto api_key = LLMClient::load_api_key_from_env();
                if (!api_key) {
                    return json{{"error", api_key.error()}};
                }

                LLMClient client;
                client.set_api_key(*api_key);

                auto* cc = event::command_center();
                if (!cc) {
                    return json{{"error", "Training system not initialized"}};
                }

                auto snapshot = cc->snapshot();

                std::string base64_render;
                bool include_render = args.value("include_render", true);
                if (include_render) {
                    int camera_index = args.value("camera_index", 0);
                    auto render_result = TrainingContext::instance().render_to_base64(camera_index);
                    if (render_result) {
                        base64_render = *render_result;
                    }
                }

                std::string problem = args.value("problem", "");

                auto result = ask_training_advisor(
                    client,
                    snapshot.iteration,
                    snapshot.loss,
                    snapshot.num_gaussians,
                    base64_render,
                    problem);

                if (!result) {
                    return json{{"error", result.error()}};
                }

                json response;
                response["success"] = result->success;
                response["advice"] = result->content;
                response["model"] = result->model;
                response["input_tokens"] = result->input_tokens;
                response["output_tokens"] = result->output_tokens;
                if (!result->success) {
                    response["error"] = result->error;
                }
                return response;
            });

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
            [](const json& args) -> json {
                const float x0 = args["x0"].get<float>();
                const float y0 = args["y0"].get<float>();
                const float x1 = args["x1"].get<float>();
                const float y1 = args["y1"].get<float>();
                const std::string mode = args.value("mode", "replace");
                const int camera_index = args.value("camera_index", 0);

                auto& ctx = TrainingContext::instance();
                auto scene = ctx.scene();
                if (!scene) {
                    return json{{"error", "No scene loaded"}};
                }

                auto screen_pos_result = compute_screen_positions_for_scene(scene, camera_index);
                if (!screen_pos_result) {
                    return json{{"error", screen_pos_result.error()}};
                }

                const auto& screen_positions = *screen_pos_result;
                const auto N = static_cast<size_t>(screen_positions.shape()[0]);
                const auto rect = normalize_screen_rect(x0, y0, x1, y1);
                return ctx.with_selection_workspace([&](TrainingContext::SelectionWorkspace& workspace) -> json {
                    auto& selection = reset_cuda_bool_scratch(workspace.selection_scratch_buffer, N);

                    if (mode == "replace") {
                        rendering::rect_select_tensor(screen_positions, rect.x0, rect.y0, rect.x1, rect.y1, selection);
                    } else {
                        const bool add_mode = (mode == "add");
                        rendering::rect_select_mode_tensor(screen_positions,
                                                           rect.x0,
                                                           rect.y0,
                                                           rect.x1,
                                                           rect.y1,
                                                           selection,
                                                           add_mode);
                    }

                    auto result = apply_headless_selection(*scene,
                                                           workspace.locked_groups_device_mask,
                                                           workspace.selection_output_buffers,
                                                           workspace.selection_output_buffer_index,
                                                           selection,
                                                           mode);
                    if (!result) {
                        return json{{"error", result.error()}};
                    }

                    return json{{"success", true}, {"selected_count", *result}};
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
            [](const json& args) -> json {
                auto& ctx = TrainingContext::instance();

                int camera_index = args.value("camera_index", 0);
                const auto& points = args["points"];
                const size_t num_vertices = points.size();
                if (num_vertices < 3) {
                    return json{{"error", "Polygon requires at least 3 vertices"}};
                }

                std::vector<float> vertex_data;
                vertex_data.reserve(num_vertices * 2);
                for (const auto& pt : points) {
                    vertex_data.push_back(pt[0].get<float>());
                    vertex_data.push_back(pt[1].get<float>());
                }
                vertex_data = close_screen_polygon(std::move(vertex_data));

                const std::string mode = args.value("mode", "replace");

                auto scene = ctx.scene();
                if (!scene) {
                    return json{{"error", "No scene loaded"}};
                }

                auto screen_pos_result = compute_screen_positions_for_scene(scene, camera_index);
                if (!screen_pos_result) {
                    return json{{"error", screen_pos_result.error()}};
                }

                const auto& screen_positions = *screen_pos_result;
                const auto N = static_cast<size_t>(screen_positions.shape()[0]);
                return ctx.with_selection_workspace([&](TrainingContext::SelectionWorkspace& workspace) -> json {
                    auto& polygon_vertices = upload_polygon_vertices_to_cuda(
                        vertex_data,
                        workspace.selection_polygon_vertex_buffer);
                    auto& selection = reset_cuda_bool_scratch(workspace.selection_scratch_buffer, N);

                    if (mode == "replace") {
                        rendering::polygon_select_tensor(screen_positions, polygon_vertices, selection);
                    } else {
                        const bool add_mode = (mode == "add");
                        rendering::polygon_select_mode_tensor(screen_positions, polygon_vertices, selection, add_mode);
                    }

                    auto result = apply_headless_selection(*scene,
                                                           workspace.locked_groups_device_mask,
                                                           workspace.selection_output_buffers,
                                                           workspace.selection_output_buffer_index,
                                                           selection,
                                                           mode);
                    if (!result) {
                        return json{{"error", result.error()}};
                    }

                    return json{{"success", true}, {"selected_count", *result}};
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
            [](const json& args) -> json {
                auto& ctx = TrainingContext::instance();

                const int camera_index = args.value("camera_index", 0);
                const auto& points = args["points"];
                const size_t num_vertices = points.size();
                if (num_vertices < 3) {
                    return json{{"error", "Lasso requires at least 3 points"}};
                }

                std::vector<float> vertex_data;
                vertex_data.reserve(num_vertices * 2);
                for (const auto& pt : points) {
                    vertex_data.push_back(pt[0].get<float>());
                    vertex_data.push_back(pt[1].get<float>());
                }

                const std::string mode = args.value("mode", "replace");

                auto scene = ctx.scene();
                if (!scene) {
                    return json{{"error", "No scene loaded"}};
                }

                auto screen_pos_result = compute_screen_positions_for_scene(scene, camera_index);
                if (!screen_pos_result) {
                    return json{{"error", screen_pos_result.error()}};
                }

                const auto& screen_positions = *screen_pos_result;
                const auto N = static_cast<size_t>(screen_positions.shape()[0]);
                return ctx.with_selection_workspace([&](TrainingContext::SelectionWorkspace& workspace) -> json {
                    auto& lasso_vertices = upload_polygon_vertices_to_cuda(
                        vertex_data,
                        workspace.selection_polygon_vertex_buffer);
                    auto& selection = reset_cuda_bool_scratch(workspace.selection_scratch_buffer, N);

                    if (mode == "replace") {
                        rendering::polygon_select_tensor(screen_positions, lasso_vertices, selection);
                    } else {
                        const bool add_mode = (mode == "add");
                        rendering::polygon_select_mode_tensor(screen_positions, lasso_vertices, selection, add_mode);
                    }

                    auto result = apply_headless_selection(*scene,
                                                           workspace.locked_groups_device_mask,
                                                           workspace.selection_output_buffers,
                                                           workspace.selection_output_buffer_index,
                                                           selection,
                                                           mode);
                    if (!result) {
                        return json{{"error", result.error()}};
                    }

                    return json{{"success", true}, {"selected_count", *result}};
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
            [](const json& args) -> json {
                auto& ctx = TrainingContext::instance();

                const float x = args["x"].get<float>();
                const float y = args["y"].get<float>();
                int camera_index = args.value("camera_index", 0);
                const std::string mode = args.value("mode", "replace");

                auto scene = ctx.scene();
                if (!scene) {
                    return json{{"error", "No scene loaded"}};
                }

                auto resolved = resolve_model_and_camera(scene, camera_index);
                if (!resolved) {
                    return json{{"error", resolved.error()}};
                }

                const auto [model, camera] = *resolved;
                auto hovered_id = pick_headless_ring_gaussian(*camera, *model, x, y);
                if (!hovered_id) {
                    return json{{"error", hovered_id.error()}};
                }

                const size_t total = scene->getTotalGaussianCount();
                return ctx.with_selection_workspace([&](TrainingContext::SelectionWorkspace& workspace) -> json {
                    auto& selection = reset_cuda_bool_scratch(workspace.selection_scratch_buffer, total);
                    rendering::set_selection_element(selection.ptr<bool>(), *hovered_id, true);

                    auto result = apply_headless_selection(*scene,
                                                           workspace.locked_groups_device_mask,
                                                           workspace.selection_output_buffers,
                                                           workspace.selection_output_buffer_index,
                                                           selection,
                                                           mode);
                    if (!result) {
                        return json{{"error", result.error()}};
                    }

                    return json{{"success", true}, {"selected_count", *result}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.click",
                .description = "Select Gaussians near a screen point (brush selection)",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"x", json{{"type", "number"}, {"description", "X coordinate"}}},
                        {"y", json{{"type", "number"}, {"description", "Y coordinate"}}},
                        {"radius", json{{"type", "number"}, {"description", "Selection radius in pixels (default: 20)"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add", "remove"})}, {"description", "Selection mode (default: replace)"}}}},
                    .required = {"x", "y"}}},
            [](const json& args) -> json {
                auto& ctx = TrainingContext::instance();

                const float x = args["x"].get<float>();
                const float y = args["y"].get<float>();
                const float radius = args.value("radius", 20.0f);
                const int camera_index = args.value("camera_index", 0);
                const std::string mode = args.value("mode", "replace");

                auto scene = ctx.scene();
                if (!scene) {
                    return json{{"error", "No scene loaded"}};
                }

                auto screen_pos_result = compute_screen_positions_for_scene(scene, camera_index);
                if (!screen_pos_result) {
                    return json{{"error", screen_pos_result.error()}};
                }

                const auto& screen_positions = *screen_pos_result;
                const auto N = static_cast<size_t>(screen_positions.shape()[0]);
                return ctx.with_selection_workspace([&](TrainingContext::SelectionWorkspace& workspace) -> json {
                    auto& selection = reset_cuda_bool_scratch(workspace.selection_scratch_buffer, N);
                    rendering::brush_select_tensor(screen_positions, x, y, radius, selection);

                    auto result = apply_headless_selection(*scene,
                                                           workspace.locked_groups_device_mask,
                                                           workspace.selection_output_buffers,
                                                           workspace.selection_output_buffer_index,
                                                           selection,
                                                           mode);
                    if (!result) {
                        return json{{"error", result.error()}};
                    }

                    return json{{"success", true}, {"selected_count", *result}};
                });
            });

        registry.register_tool(
            McpTool{
                .name = "selection.get",
                .description = "Get current selection (returns selected Gaussian indices)",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [](const json&) -> json {
                auto scene = TrainingContext::instance().scene();
                if (!scene) {
                    return json{{"error", "No scene loaded"}};
                }

                auto mask = scene->getSelectionMask();
                if (!mask) {
                    return json{{"success", true}, {"selected_count", 0}, {"indices", json::array()}};
                }

                auto mask_vec = mask->to_vector_uint8();

                std::vector<int64_t> indices;
                for (size_t i = 0; i < mask_vec.size(); ++i) {
                    if (mask_vec[i] > 0) {
                        indices.push_back(static_cast<int64_t>(i));
                    }
                }

                return json{{"success", true}, {"selected_count", indices.size()}, {"indices", indices}};
            });

        registry.register_tool(
            McpTool{
                .name = "selection.clear",
                .description = "Clear all selection",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [](const json&) -> json {
                auto scene = TrainingContext::instance().scene();
                if (!scene) {
                    return json{{"error", "No scene loaded"}};
                }

                auto* model = scene->getTrainingModel();
                if (!model) {
                    return json{{"error", "No model loaded"}};
                }

                const auto N = model->size();
                auto empty_mask = std::make_shared<core::Tensor>(
                    core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8));
                scene->setSelectionMask(empty_mask);

                return json{{"success", true}};
            });

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
            [](const json& args) -> json {
                const auto capability = args.value("capability", "");
                if (capability.empty()) {
                    return json{{"error", "Missing capability name"}};
                }

                const std::string args_json = args.contains("args") ? args["args"].dump() : "{}";
                auto scene = TrainingContext::instance().scene();
                if (!scene) {
                    return json{{"error", "No scene loaded"}};
                }

                return invoke_plugin_capability(scene.get(), capability, args_json);
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
            [](const json& args) -> json {
                auto api_key = LLMClient::load_api_key_from_env();
                if (!api_key) {
                    return json{{"error", api_key.error()}};
                }

                auto& ctx = TrainingContext::instance();
                auto scene = ctx.scene();
                if (!scene) {
                    return json{{"error", "No scene loaded"}};
                }

                int camera_index = args.value("camera_index", 0);
                auto render_result = render_to_base64_for_scene(scene, camera_index, 0, 0);
                if (!render_result) {
                    return json{{"error", render_result.error()}};
                }

                LLMClient client;
                client.set_api_key(*api_key);

                const std::string description = args["description"].get<std::string>();

                LLMRequest request;
                request.prompt = "Look at this 3D scene render. I need you to identify the bounding box for: \"" + description + "\"\n\n"
                                                                                                                                 "Return ONLY a JSON object with the bounding box coordinates in pixel space:\n"
                                                                                                                                 "{\"x0\": <left>, \"y0\": <top>, \"x1\": <right>, \"y1\": <bottom>}\n\n"
                                                                                                                                 "The coordinates should be integers representing pixel positions. "
                                                                                                                                 "If you cannot identify the object, return: {\"error\": \"Object not found\"}";
                request.attachments.push_back(ImageAttachment{.base64_data = *render_result, .media_type = "image/png"});
                request.temperature = 0.0f;
                request.max_tokens = 256;

                auto response = client.complete(request);
                if (!response) {
                    return json{{"error", response.error()}};
                }

                if (!response->success) {
                    return json{{"error", response->error}};
                }

                json bbox;
                try {
                    auto content = response->content;
                    auto json_start = content.find('{');
                    auto json_end = content.rfind('}');
                    if (json_start == std::string::npos || json_end == std::string::npos) {
                        return json{{"error", "LLM response did not contain valid JSON"}};
                    }
                    bbox = json::parse(content.substr(json_start, json_end - json_start + 1));
                } catch (const std::exception& e) {
                    return json{{"error", std::string("Failed to parse LLM response: ") + e.what()}};
                }

                if (bbox.contains("error")) {
                    return json{{"error", bbox["error"].get<std::string>()}};
                }

                if (!bbox.contains("x0") || !bbox.contains("y0") || !bbox.contains("x1") || !bbox.contains("y1")) {
                    return json{{"error", "LLM response missing bounding box coordinates"}};
                }

                const float x0 = bbox["x0"].get<float>();
                const float y0 = bbox["y0"].get<float>();
                const float x1 = bbox["x1"].get<float>();
                const float y1 = bbox["y1"].get<float>();
                const auto rect = normalize_screen_rect(x0, y0, x1, y1);
                auto screen_pos_result = compute_screen_positions_for_scene(scene, camera_index);
                if (!screen_pos_result) {
                    return json{{"error", screen_pos_result.error()}};
                }

                const auto& screen_positions = *screen_pos_result;
                const auto N = static_cast<size_t>(screen_positions.shape()[0]);
                return ctx.with_selection_workspace([&](TrainingContext::SelectionWorkspace& workspace) -> json {
                    auto& selection = reset_cuda_bool_scratch(workspace.selection_scratch_buffer, N);
                    rendering::rect_select_tensor(screen_positions, rect.x0, rect.y0, rect.x1, rect.y1, selection);

                    auto selection_result = apply_headless_selection(*scene,
                                                                     workspace.locked_groups_device_mask,
                                                                     workspace.selection_output_buffers,
                                                                     workspace.selection_output_buffer_index,
                                                                     selection,
                                                                     "replace");
                    if (!selection_result) {
                        return json{{"error", selection_result.error()}};
                    }

                    json json_response;
                    json_response["success"] = true;
                    json_response["selected_count"] = *selection_result;
                    json_response["bounding_box"] = bbox;
                    json_response["description"] = description;
                    return json_response;
                });
            });
    }

} // namespace lfs::mcp
