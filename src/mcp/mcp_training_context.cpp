/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcp_training_context.hpp"
#include "llm_client.hpp"
#include "mcp_tools.hpp"
#include "selection_client.hpp"

#include "core/checkpoint_format.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "io/exporter.hpp"
#include "python/runner.hpp"
#include "rendering/gs_rasterizer_tensor.hpp"
#include "rendering/rasterizer/rasterization/include/rasterization_api_tensor.h"
#include "training/checkpoint.hpp"
#include "training/dataset.hpp"
#include "training/training_setup.hpp"

#include <fstream>
#include <sstream>

namespace lfs::mcp {

    namespace {
        constexpr char BASE64_CHARS[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        std::string base64_encode(const std::vector<uint8_t>& data) {
            std::string result;
            result.reserve(((data.size() + 2) / 3) * 4);

            for (size_t i = 0; i < data.size(); i += 3) {
                const uint32_t b0 = data[i];
                const uint32_t b1 = (i + 1 < data.size()) ? data[i + 1] : 0;
                const uint32_t b2 = (i + 2 < data.size()) ? data[i + 2] : 0;

                result += BASE64_CHARS[(b0 >> 2) & 0x3F];
                result += BASE64_CHARS[((b0 << 4) | (b1 >> 4)) & 0x3F];

                if (i + 1 < data.size()) {
                    result += BASE64_CHARS[((b1 << 2) | (b2 >> 6)) & 0x3F];
                } else {
                    result += '=';
                }

                if (i + 2 < data.size()) {
                    result += BASE64_CHARS[b2 & 0x3F];
                } else {
                    result += '=';
                }
            }
            return result;
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

        stop_training();

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

        trainer_ = std::make_unique<training::Trainer>(*scene_);

        if (auto result = trainer_->initialize(params_); !result) {
            trainer_.reset();
            scene_.reset();
            return std::unexpected(result.error());
        }

        LOG_INFO("MCP: Loaded dataset from {}", path.string());
        return {};
    }

    std::expected<void, std::string> TrainingContext::load_checkpoint(
        const std::filesystem::path& path) {

        std::lock_guard lock(mutex_);

        stop_training();

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

        trainer_ = std::make_unique<training::Trainer>(*scene_);

        if (auto result = trainer_->initialize(params_); !result) {
            trainer_.reset();
            scene_.reset();
            return std::unexpected(result.error());
        }

        LOG_INFO("MCP: Loaded checkpoint from {}", path.string());
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

        LOG_INFO("MCP: Saved checkpoint to {}", path.string());
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

        LOG_INFO("MCP: Saved PLY to {}", path.string());
        return {};
    }

    std::expected<std::string, std::string> TrainingContext::render_to_base64(
        int camera_index,
        int width,
        int height) {

        std::lock_guard lock(mutex_);

        if (!scene_) {
            return std::unexpected("No scene loaded");
        }

        auto* model = scene_->getTrainingModel();
        if (!model) {
            return std::unexpected("No model to render");
        }

        auto cameras = scene_->getAllCameras();
        if (cameras.empty()) {
            return std::unexpected("No cameras available");
        }

        if (camera_index < 0 || camera_index >= static_cast<int>(cameras.size())) {
            camera_index = 0;
        }

        auto& camera = cameras[camera_index];
        if (!camera) {
            return std::unexpected("Failed to get camera");
        }

        core::Tensor bg = core::Tensor::zeros({3}, core::Device::CUDA);

        try {
            auto [image, alpha] = rendering::rasterize_tensor(*camera, *model, bg);

            std::ostringstream oss;
            oss << "mcp_render_" << std::this_thread::get_id() << ".png";
            auto temp_path = std::filesystem::temp_directory_path() / oss.str();
            core::save_image(temp_path, image);

            std::ifstream file(temp_path, std::ios::binary | std::ios::ate);
            if (!file) {
                return std::unexpected("Failed to read rendered image");
            }

            const auto size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<uint8_t> buffer(size);
            if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
                return std::unexpected("Failed to read image data");
            }

            std::filesystem::remove(temp_path);

            return base64_encode(buffer);
        } catch (const std::exception& e) {
            return std::unexpected(std::string("Render failed: ") + e.what());
        }
    }

    std::expected<core::Tensor, std::string> TrainingContext::compute_screen_positions(
        int camera_index) {

        std::lock_guard lock(mutex_);

        if (!scene_) {
            return std::unexpected("No scene loaded");
        }

        auto* model = scene_->getTrainingModel();
        if (!model) {
            return std::unexpected("No model loaded");
        }

        auto cameras = scene_->getAllCameras();
        if (cameras.empty()) {
            return std::unexpected("No cameras available");
        }

        if (camera_index < 0 || camera_index >= static_cast<int>(cameras.size())) {
            camera_index = 0;
        }

        auto& camera = cameras[camera_index];
        if (!camera) {
            return std::unexpected("Failed to get camera");
        }

        core::Tensor bg = core::Tensor::zeros({3}, core::Device::CUDA);
        core::Tensor screen_positions;

        try {
            auto [image, alpha] = rendering::rasterize_tensor(
                *camera, *model, bg,
                false,   // show_rings
                0.01f,   // ring_width
                nullptr, // model_transforms
                nullptr, // transform_indices
                nullptr, // selection_mask
                &screen_positions);

            return screen_positions;
        } catch (const std::exception& e) {
            return std::unexpected(std::string("Screen position computation failed: ") + e.what());
        }
    }

    std::expected<void, std::string> TrainingContext::start_training() {
        std::lock_guard lock(mutex_);

        if (!trainer_) {
            return std::unexpected("No trainer initialized");
        }

        if (training_thread_) {
            return std::unexpected("Training already running");
        }

        training_thread_ = std::make_unique<std::jthread>([this](std::stop_token stop) {
            auto result = trainer_->train(stop);
            if (!result) {
                LOG_ERROR("Training error: {}", result.error());
            }
        });

        LOG_INFO("MCP: Training started");
        return {};
    }

    void TrainingContext::stop_training() {
        if (training_thread_) {
            training_thread_->request_stop();
            training_thread_.reset();
        }
    }

    void TrainingContext::pause_training() {
        if (trainer_) {
            trainer_->request_pause();
        }
    }

    void TrainingContext::resume_training() {
        if (trainer_) {
            trainer_->request_resume();
        }
    }

    void TrainingContext::shutdown() {
        stop_training();
        trainer_.reset();
        scene_.reset();
    }

    void register_scene_tools() {
        auto& registry = ToolRegistry::instance();

        registry.register_tool(
            McpTool{
                .name = "scene.load_dataset",
                .description = "Load a COLMAP dataset for training",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Path to COLMAP dataset directory"}}},
                        {"images_folder", json{{"type", "string"}, {"description", "Images subfolder (default: images)"}}},
                        {"max_iterations", json{{"type", "integer"}, {"description", "Maximum training iterations (default: 30000)"}}},
                        {"strategy", json{{"type", "string"}, {"enum", json::array({"mcmc", "default"})}, {"description", "Training strategy"}}}},
                    .required = {"path"}}},
            [](const json& args) -> json {
                std::filesystem::path path = args["path"].get<std::string>();

                core::param::TrainingParameters params;
                params.dataset.data_path = path;

                if (args.contains("images_folder")) {
                    params.dataset.images = args["images_folder"].get<std::string>();
                }
                if (args.contains("max_iterations")) {
                    params.optimization.iterations = args["max_iterations"].get<size_t>();
                }
                if (args.contains("strategy")) {
                    params.optimization.strategy = args["strategy"].get<std::string>();
                }

                auto result = TrainingContext::instance().load_dataset(path, params);
                if (!result) {
                    return json{{"error", result.error()}};
                }

                json response;
                response["success"] = true;
                response["path"] = path.string();

                auto scene = TrainingContext::instance().scene();
                if (scene) {
                    response["num_gaussians"] = scene->getTotalGaussianCount();
                }

                return response;
            });

        registry.register_tool(
            McpTool{
                .name = "scene.load_checkpoint",
                .description = "Load a training checkpoint (.resume file)",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Path to checkpoint file"}}}},
                    .required = {"path"}}},
            [](const json& args) -> json {
                std::filesystem::path path = args["path"].get<std::string>();

                auto result = TrainingContext::instance().load_checkpoint(path);
                if (!result) {
                    return json{{"error", result.error()}};
                }

                json response;
                response["success"] = true;
                response["path"] = path.string();

                return response;
            });

        registry.register_tool(
            McpTool{
                .name = "scene.save_checkpoint",
                .description = "Save current training state to checkpoint file",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Path to save checkpoint"}}}},
                    .required = {"path"}}},
            [](const json& args) -> json {
                std::filesystem::path path = args["path"].get<std::string>();

                auto result = TrainingContext::instance().save_checkpoint(path);
                if (!result) {
                    return json{{"error", result.error()}};
                }

                return json{{"success", true}, {"path", path.string()}};
            });

        registry.register_tool(
            McpTool{
                .name = "scene.save_ply",
                .description = "Save current model as PLY file",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Path to save PLY file"}}}},
                    .required = {"path"}}},
            [](const json& args) -> json {
                std::filesystem::path path = args["path"].get<std::string>();

                auto result = TrainingContext::instance().save_ply(path);
                if (!result) {
                    return json{{"error", result.error()}};
                }

                return json{{"success", true}, {"path", path.string()}};
            });

        registry.register_tool(
            McpTool{
                .name = "training.start",
                .description = "Start training in background",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [](const json&) -> json {
                auto result = TrainingContext::instance().start_training();
                if (!result) {
                    return json{{"error", result.error()}};
                }
                return json{{"success", true}, {"message", "Training started"}};
            });

        registry.register_tool(
            McpTool{
                .name = "render.capture",
                .description = "Render current scene and return as base64 PNG",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"camera_index", json{{"type", "integer"}, {"description", "Camera index (default: 0)"}}},
                        {"width", json{{"type", "integer"}, {"description", "Output width (default: camera native)"}}},
                        {"height", json{{"type", "integer"}, {"description", "Output height (default: camera native)"}}}},
                    .required = {}}},
            [](const json& args) -> json {
                int camera_index = args.contains("camera_index") ? args["camera_index"].get<int>() : 0;
                int width = args.contains("width") ? args["width"].get<int>() : 0;
                int height = args.contains("height") ? args["height"].get<int>() : 0;

                auto result = TrainingContext::instance().render_to_base64(camera_index, width, height);
                if (!result) {
                    return json{{"error", result.error()}};
                }

                json response;
                response["success"] = true;
                response["mime_type"] = "image/png";
                response["data"] = *result;
                return response;
            });

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

                SelectionClient client;
                if (client.is_gui_running()) {
                    auto result = client.select_rect(x0, y0, x1, y1, mode, camera_index);
                    if (!result) {
                        return json{{"error", result.error()}};
                    }
                    return json{{"success", true}, {"via_gui", true}};
                }

                auto& ctx = TrainingContext::instance();
                auto screen_pos_result = ctx.compute_screen_positions(camera_index);
                if (!screen_pos_result) {
                    return json{{"error", screen_pos_result.error()}};
                }

                auto scene = ctx.scene();
                if (!scene) {
                    return json{{"error", "No scene loaded"}};
                }

                const auto& screen_positions = *screen_pos_result;
                const auto N = static_cast<size_t>(screen_positions.shape()[0]);

                core::Tensor selection = core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8);

                if (mode == "replace") {
                    rendering::rect_select_tensor(screen_positions, x0, y0, x1, y1, selection);
                } else {
                    bool add_mode = (mode == "add");
                    rendering::rect_select_mode_tensor(screen_positions, x0, y0, x1, y1, selection, add_mode);
                }

                auto existing_mask = scene->getSelectionMask();
                if (!existing_mask) {
                    existing_mask = std::make_shared<core::Tensor>(
                        core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8));
                }

                core::Tensor output_mask = core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8);
                uint32_t locked_groups[8] = {0};

                rendering::apply_selection_group_tensor(
                    selection, *existing_mask, output_mask,
                    1, locked_groups, true);

                scene->setSelectionMask(std::make_shared<core::Tensor>(std::move(output_mask)));

                int64_t count = 0;
                auto mask_vec = scene->getSelectionMask()->to_vector_uint8();
                for (auto v : mask_vec) {
                    if (v > 0)
                        count++;
                }

                return json{{"success", true}, {"selected_count", count}};
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
                auto screen_pos_result = ctx.compute_screen_positions(camera_index);
                if (!screen_pos_result) {
                    return json{{"error", screen_pos_result.error()}};
                }

                auto scene = ctx.scene();
                if (!scene) {
                    return json{{"error", "No scene loaded"}};
                }

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

                core::Tensor polygon_vertices = core::Tensor::from_vector(
                    vertex_data,
                    {num_vertices, 2},
                    core::Device::CUDA);

                const std::string mode = args.value("mode", "replace");
                const auto& screen_positions = *screen_pos_result;
                const auto N = static_cast<size_t>(screen_positions.shape()[0]);

                core::Tensor selection = core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8);

                if (mode == "replace") {
                    rendering::polygon_select_tensor(screen_positions, polygon_vertices, selection);
                } else {
                    bool add_mode = (mode == "add");
                    rendering::polygon_select_mode_tensor(screen_positions, polygon_vertices, selection, add_mode);
                }

                auto existing_mask = scene->getSelectionMask();
                if (!existing_mask) {
                    existing_mask = std::make_shared<core::Tensor>(
                        core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8));
                }

                core::Tensor output_mask = core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8);
                uint32_t locked_groups[8] = {0};

                rendering::apply_selection_group_tensor(
                    selection, *existing_mask, output_mask,
                    1, locked_groups, true);

                scene->setSelectionMask(std::make_shared<core::Tensor>(std::move(output_mask)));

                int64_t count = 0;
                auto mask_vec = scene->getSelectionMask()->to_vector_uint8();
                for (auto v : mask_vec) {
                    if (v > 0)
                        count++;
                }

                return json{{"success", true}, {"selected_count", count}};
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

                int camera_index = args.value("camera_index", 0);
                auto screen_pos_result = ctx.compute_screen_positions(camera_index);
                if (!screen_pos_result) {
                    return json{{"error", screen_pos_result.error()}};
                }

                auto scene = ctx.scene();
                if (!scene) {
                    return json{{"error", "No scene loaded"}};
                }

                const float x = args["x"].get<float>();
                const float y = args["y"].get<float>();
                const float radius = args.value("radius", 20.0f);

                const auto& screen_positions = *screen_pos_result;
                const auto N = static_cast<size_t>(screen_positions.shape()[0]);

                core::Tensor selection = core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8);
                rendering::brush_select_tensor(screen_positions, x, y, radius, selection);

                auto existing_mask = scene->getSelectionMask();
                if (!existing_mask) {
                    existing_mask = std::make_shared<core::Tensor>(
                        core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8));
                }

                const std::string mode = args.value("mode", "replace");
                core::Tensor output_mask = core::Tensor::zeros({N}, core::Device::CUDA, core::DataType::UInt8);
                uint32_t locked_groups[8] = {0};

                bool add_mode = (mode != "remove");
                rendering::apply_selection_group_tensor(
                    selection, *existing_mask, output_mask,
                    1, locked_groups, add_mode);

                scene->setSelectionMask(std::make_shared<core::Tensor>(std::move(output_mask)));

                int64_t count = 0;
                auto mask_vec = scene->getSelectionMask()->to_vector_uint8();
                for (auto v : mask_vec) {
                    if (v > 0)
                        count++;
                }

                return json{{"success", true}, {"selected_count", count}};
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

                SelectionClient client;
                if (!client.is_gui_running()) {
                    return json{{"error", "GUI not running"}};
                }

                const std::string args_json = args.contains("args") ? args["args"].dump() : "{}";
                auto result = client.invoke_capability(capability, args_json);
                if (!result) {
                    return json{{"error", result.error()}};
                }

                if (!result->success) {
                    return json{{"success", false}, {"error", result->error}};
                }

                try {
                    return json::parse(result->result_json);
                } catch (...) {
                    return json{{"success", true}};
                }
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

                int camera_index = args.value("camera_index", 0);
                auto render_result = ctx.render_to_base64(camera_index);
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

                // Try to send selection to GUI if running
                SelectionClient selection_client;
                if (selection_client.is_gui_running()) {
                    auto sel_result = selection_client.select_rect(x0, y0, x1, y1, "replace", camera_index);
                    if (!sel_result) {
                        return json{{"error", sel_result.error()}};
                    }
                    json gui_response;
                    gui_response["success"] = true;
                    gui_response["via_gui"] = true;
                    gui_response["bounding_box"] = bbox;
                    gui_response["description"] = description;
                    return gui_response;
                }

                // Fall back to headless selection
                auto screen_pos_result = ctx.compute_screen_positions(camera_index);
                if (!screen_pos_result) {
                    return json{{"error", screen_pos_result.error()}};
                }

                auto scene = ctx.scene();
                if (!scene) {
                    return json{{"error", "No scene loaded"}};
                }

                const auto& screen_positions = *screen_pos_result;
                const auto N = static_cast<size_t>(screen_positions.shape()[0]);

                core::Tensor selection = core::Tensor::zeros({static_cast<size_t>(N)}, core::Device::CUDA, core::DataType::UInt8);
                rendering::rect_select_tensor(screen_positions, x0, y0, x1, y1, selection);

                auto existing_mask = scene->getSelectionMask();
                if (!existing_mask) {
                    existing_mask = std::make_shared<core::Tensor>(
                        core::Tensor::zeros({static_cast<size_t>(N)}, core::Device::CUDA, core::DataType::UInt8));
                }

                core::Tensor output_mask = core::Tensor::zeros({static_cast<size_t>(N)}, core::Device::CUDA, core::DataType::UInt8);
                uint32_t locked_groups[8] = {0};

                rendering::apply_selection_group_tensor(
                    selection, *existing_mask, output_mask,
                    1, locked_groups, true);

                scene->setSelectionMask(std::make_shared<core::Tensor>(std::move(output_mask)));

                int64_t count = 0;
                auto mask_vec = scene->getSelectionMask()->to_vector_uint8();
                for (auto v : mask_vec) {
                    if (v > 0)
                        count++;
                }

                json result;
                result["success"] = true;
                result["selected_count"] = count;
                result["bounding_box"] = bbox;
                result["description"] = description;
                return result;
            });
    }

} // namespace lfs::mcp
