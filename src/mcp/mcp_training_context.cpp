/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcp_training_context.hpp"
#include "llm_client.hpp"
#include "mcp_tools.hpp"
#include "shared_scene_tools.hpp"

#include "core/checkpoint_format.hpp"
#include "core/error.hpp"
#include "core/error_reporter.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/guarded_task.hpp"
#include "core/logger.hpp"
#include "io/exporter.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"
#include "rendering/selection_ops.hpp"
#include "training/checkpoint.hpp"
#include "training/dataset.hpp"
#include "training/training_setup.hpp"
#include "visualizer/selection/selection_group_mask.hpp"

#include <algorithm>
#include <cassert>
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
            (void)camera;
            (void)model;
            (void)x;
            (void)y;
            return std::unexpected(
                "Headless CUDA ring-pick rendering has been removed; use the live Vulkan selection path");
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
            return std::unexpected(
                "Headless CUDA screen-position rendering has been removed; use the live Vulkan selection path");
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
            (void)width;
            (void)height;
            return std::unexpected(
                "Headless CUDA scene rendering has been removed; use live Vulkan viewport capture");
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

            const uint8_t group_id = scene.getActiveSelectionGroup();
            const auto existing_mask = scene.getSelectionMask();
            const core::Tensor empty_mask;
            const core::Tensor* const existing_ptr =
                (existing_mask && existing_mask->is_valid() && existing_mask->numel() == selection_mask.numel())
                    ? existing_mask.get()
                    : nullptr;
            if (mode == "intersect") {
                if (existing_ptr) {
                    auto active_group = existing_ptr->eq(group_id);
                    if (active_group.device() != core::Device::CUDA) {
                        active_group = active_group.cuda();
                    }
                    selection_mask = selection_mask.logical_and(active_group);
                } else {
                    selection_mask =
                        core::Tensor::zeros({selection_mask.numel()}, core::Device::CUDA, core::DataType::Bool);
                }
            }

            const auto& existing_ref = existing_ptr ? *existing_ptr : empty_mask;
            const auto transform_indices = scene.getTransformIndices();
            const bool add_mode = (mode != "remove");
            const bool replace_mode = (mode == "replace" || mode == "intersect");
            auto& output_mask = acquire_selection_output_buffer(
                selection_output_buffers, selection_output_buffer_index, selection_mask.numel());

            rendering::apply_selection_group_tensor_mask(
                selection_mask,
                existing_ref,
                output_mask,
                group_id,
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

        if (training_thread_ && training_active_.load(std::memory_order_acquire)) {
            return std::unexpected("Training already running");
        }

        // A naturally completed jthread remains joinable until its owner reaps
        // it. Join that finished generation before installing the next one.
        training_thread_.reset();

        auto trainer = trainer_;
        training_active_.store(true, std::memory_order_release);
        try {
            last_training_error_.clear();
            training_thread_ = std::make_unique<std::jthread>([this, trainer](std::stop_token stop) {
                lfs::core::run_guarded<void>(
                    lfs::core::TaskContext{
                        .name = "mcp.training-worker",
                        .domain = lfs::ErrorDomain::Training,
                        .operation_id = lfs::OperationId::generate(),
                        .site = LFS_SOURCE_SITE_CURRENT(),
                    },
                    [trainer, stop]() -> lfs::Result<void> {
                        return trainer->train(stop);
                    },
                    [this](lfs::Result<void>&& result) {
                        if (!result) {
                            last_training_error_.set(result.error());
                            lfs::core::ErrorReporter::get().report(result.error(), lfs::core::ReportChannel::OwnerLog);
                        }
                        training_active_.store(false, std::memory_order_release);
                    });
            });
        } catch (const std::exception& e) {
            training_active_.store(false, std::memory_order_release);
            std::string message = std::string("Failed to start training thread: ") + e.what();
            last_training_error_.set(lfs::make_legacy_error(message, lfs::LegacyErrorContext{
                                                                         .code = lfs::ErrorCode::FailedPrecondition,
                                                                         .domain = lfs::ErrorDomain::Training,
                                                                         .operation = "mcp.training.start",
                                                                         .source = LFS_SOURCE_SITE_CURRENT(),
                                                                         .operation_id = lfs::OperationId::generate(),
                                                                     }));
            return std::unexpected(std::move(message));
        }

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
        training_active_.store(false, std::memory_order_release);
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
            },
            .last_training_error =
                []() -> std::optional<lfs::Error> {
                return TrainingContext::instance().last_training_error();
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
                .name = "selection.ring",
                .description = "Select the front-most Gaussian under a screen point using ring-mode picking",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"x", json{{"type", "number"}, {"description", "X coordinate"}}},
                        {"y", json{{"type", "number"}, {"description", "Y coordinate"}}},
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"mode", json{{"type", "string"}, {"enum", json::array({"replace", "add", "remove", "intersect"})}, {"description", "Selection mode (default: replace)"}}}},
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
    }

} // namespace lfs::mcp
