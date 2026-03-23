/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "shared_scene_tools.hpp"

#include "core/path_utils.hpp"
#include "mcp_tools.hpp"

#include <utility>

namespace lfs::mcp {

    namespace {

        constexpr std::string_view NO_MODEL_LOADED_ERROR = "No model loaded";

        McpToolMetadata command_metadata(const SharedSceneToolBackend& backend,
                                         std::string category,
                                         const bool destructive = false,
                                         const bool long_running = false) {
            return McpToolMetadata{
                .category = std::move(category),
                .kind = "command",
                .runtime = backend.runtime,
                .thread_affinity = backend.thread_affinity,
                .destructive = destructive,
                .long_running = long_running,
            };
        }

        McpToolMetadata query_metadata(const SharedSceneToolBackend& backend,
                                       std::string category) {
            return McpToolMetadata{
                .category = std::move(category),
                .kind = "query",
                .runtime = backend.runtime,
                .thread_affinity = backend.thread_affinity,
            };
        }

    } // namespace

    void register_shared_scene_tools(const SharedSceneToolBackend& backend) {
        auto& registry = ToolRegistry::instance();

        registry.register_tool(
            McpTool{
                .name = "scene.load_dataset",
                .description = "Load a COLMAP dataset for training/viewing",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Path to COLMAP dataset directory"}}},
                        {"images_folder", json{{"type", "string"}, {"description", "Images subfolder (default: images)"}}},
                        {"max_iterations", json{{"type", "integer"}, {"description", "Maximum training iterations (default: 30000)"}}},
                        {"strategy", json{{"type", "string"}, {"enum", json::array({"mcmc", "default"})}, {"description", "Training strategy"}}}},
                    .required = {"path"}},
                .metadata = command_metadata(backend, "scene", true)},
            [backend](const json& args) -> json {
                std::filesystem::path path = args["path"].get<std::string>();

                core::param::TrainingParameters params;
                params.dataset.data_path = path;
                if (args.contains("images_folder"))
                    params.dataset.images = args["images_folder"].get<std::string>();
                if (args.contains("max_iterations"))
                    params.optimization.iterations = args["max_iterations"].get<size_t>();
                if (args.contains("strategy"))
                    params.optimization.strategy = args["strategy"].get<std::string>();

                auto result = backend.load_dataset(path, params);
                if (!result)
                    return json{{"error", result.error()}};

                json response{
                    {"success", true},
                    {"path", core::path_to_utf8(path)},
                };
                if (backend.gaussian_count) {
                    if (const auto count = backend.gaussian_count(); count) {
                        response["num_gaussians"] = *count;
                    } else if (count.error() == NO_MODEL_LOADED_ERROR) {
                        response["num_gaussians"] = 0;
                    }
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
                    .required = {"path"}},
                .metadata = command_metadata(backend, "scene", true)},
            [backend](const json& args) -> json {
                std::filesystem::path path = args["path"].get<std::string>();
                auto result = backend.load_checkpoint(path);
                if (!result)
                    return json{{"error", result.error()}};
                return json{{"success", true}, {"path", core::path_to_utf8(path)}};
            });

        registry.register_tool(
            McpTool{
                .name = "scene.save_checkpoint",
                .description = "Save current training state. The path is a base directory; checkpoints are saved as checkpoints/checkpoint.resume inside it. Omit path to use the current output path.",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Base output directory; checkpoint files are written to <path>/checkpoints/checkpoint.resume"}}}},
                    .required = {}},
                .metadata = command_metadata(backend, "scene", false, true)},
            [backend](const json& args) -> json {
                const std::optional<std::filesystem::path> requested_path =
                    args.contains("path")
                        ? std::optional<std::filesystem::path>(args["path"].get<std::string>())
                        : std::nullopt;

                auto result = backend.save_checkpoint(requested_path);
                if (!result)
                    return json{{"error", result.error()}};

                return json{
                    {"success", true},
                    {"path", core::path_to_utf8(*result)},
                    {"output_path", core::path_to_utf8(*result)},
                    {"used_default_path", !requested_path.has_value()},
                };
            });

        registry.register_tool(
            McpTool{
                .name = "scene.save_ply",
                .description = "Save current model as a PLY file",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"path", json{{"type", "string"}, {"description", "Path to save PLY file"}}}},
                    .required = {"path"}},
                .metadata = command_metadata(backend, "scene", false, true)},
            [backend](const json& args) -> json {
                std::filesystem::path path = args["path"].get<std::string>();
                auto result = backend.save_ply(path);
                if (!result)
                    return json{{"error", result.error()}};
                return json{{"success", true}, {"path", core::path_to_utf8(path)}};
            });

        registry.register_tool(
            McpTool{
                .name = "training.start",
                .description = "Start training in the current runtime",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = command_metadata(backend, "training", false, true)},
            [backend](const json&) -> json {
                auto result = backend.start_training();
                if (!result)
                    return json{{"error", result.error()}};
                return json{{"success", true}, {"message", "Training started"}};
            });

        registry.register_tool(
            McpTool{
                .name = "render.capture",
                .description = "Capture the current scene. Omit camera_index to grab the live viewport; pass camera_index to render from a dataset camera.",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"camera_index", json{{"type", "integer"}, {"description", "Dataset camera index; omit to capture the live viewport"}}},
                        {"width", json{{"type", "integer"}, {"description", "Optional output width; preserves aspect ratio when height is omitted"}}},
                        {"height", json{{"type", "integer"}, {"description", "Optional output height; preserves aspect ratio when width is omitted"}}}},
                    .required = {}},
                .metadata = query_metadata(backend, "render")},
            [backend](const json& args) -> json {
                const std::optional<int> camera_index =
                    args.contains("camera_index")
                        ? std::optional<int>(args["camera_index"].get<int>())
                        : std::nullopt;
                const int width = args.value("width", 0);
                const int height = args.value("height", 0);

                auto result = backend.render_capture(camera_index, width, height);
                if (!result)
                    return json{{"error", result.error()}};

                return json{
                    {"success", true},
                    {"mime_type", "image/png"},
                    {"data", *result},
                };
            });
    }

} // namespace lfs::mcp
