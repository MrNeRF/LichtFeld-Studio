/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/path_utils.hpp"
#include "mcp/mcp_tools.hpp"
#include "mcp/shared_scene_tools.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <optional>
#include <string>

namespace {

    using json = nlohmann::json;

    constexpr std::array<const char*, 6> kSharedSceneToolNames = {
        "scene.load_dataset",
        "scene.load_checkpoint",
        "scene.save_ply",
        "training.start",
        "render.capture",
        "training.get_last_error",
    };

    class ScopedSharedSceneToolRegistration {
    public:
        ScopedSharedSceneToolRegistration() {
            for (const auto* name : kSharedSceneToolNames) {
                lfs::mcp::ToolRegistry::instance().unregister_tool(name);
            }
        }

        ~ScopedSharedSceneToolRegistration() {
            for (const auto* name : kSharedSceneToolNames) {
                lfs::mcp::ToolRegistry::instance().unregister_tool(name);
            }
        }
    };

    struct FakeSharedSceneBackend {
        std::filesystem::path loaded_path;
        lfs::core::param::TrainingParameters loaded_params;
        bool load_dataset_called = false;
        std::optional<bool> start_overwrite;
        int capture_calls = 0;

        lfs::mcp::SharedSceneToolBackend backend() {
            return lfs::mcp::SharedSceneToolBackend{
                .runtime = "test",
                .thread_affinity = "any",
                .load_dataset =
                    [this](const std::filesystem::path& path,
                           const lfs::core::param::TrainingParameters& params)
                    -> std::expected<void, std::string> {
                    load_dataset_called = true;
                    loaded_path = path;
                    loaded_params = params;
                    return {};
                },
                .load_checkpoint =
                    [](const std::filesystem::path&) -> std::expected<void, std::string> {
                    return {};
                },
                .save_ply =
                    [](const std::filesystem::path&, bool) -> std::expected<void, std::string> {
                    return {};
                },
                .start_training = [this](const bool overwrite) -> std::expected<void, std::string> {
                    start_overwrite = overwrite;
                    return {};
                },
                .render_capture = [this](int, int, bool) -> lfs::Result<std::string> {
                    ++capture_calls;
                    return std::string{};
                },
                .gaussian_count = []() -> std::expected<int64_t, std::string> { return 0; },
            };
        }
    };

} // namespace

TEST(McpSharedSceneToolsTest, LoadDatasetDefaultsOutputPathAndStrategyAlias) {
    ScopedSharedSceneToolRegistration cleanup;
    FakeSharedSceneBackend backend;
    lfs::mcp::register_shared_scene_tools(backend.backend());

    const std::filesystem::path dataset_path = "/tmp/mcp_dataset";
    const auto result = lfs::mcp::ToolRegistry::instance().call_tool(
        "scene.load_dataset",
        json{{"path", dataset_path.string()}, {"strategy", "default"}});

    ASSERT_TRUE(result["success"].get<bool>());
    ASSERT_TRUE(backend.load_dataset_called);
    EXPECT_EQ(backend.loaded_path, dataset_path);
    EXPECT_EQ(backend.loaded_params.dataset.data_path, dataset_path);
    EXPECT_EQ(backend.loaded_params.dataset.output_path,
              lfs::core::param::default_dataset_output_path(dataset_path));
    EXPECT_EQ(backend.loaded_params.optimization.strategy,
              std::string(lfs::core::param::kStrategyMRNF));
    EXPECT_EQ(result["output_path"].get<std::string>(),
              lfs::core::path_to_utf8(lfs::core::param::default_dataset_output_path(dataset_path)));
    EXPECT_EQ(result["strategy"].get<std::string>(), std::string(lfs::core::param::kStrategyMRNF));
}

TEST(McpSharedSceneToolsTest, LoadDatasetHonorsExplicitOutputPathAndCanonicalStrategy) {
    ScopedSharedSceneToolRegistration cleanup;
    FakeSharedSceneBackend backend;
    lfs::mcp::register_shared_scene_tools(backend.backend());

    const std::filesystem::path dataset_path = "/tmp/mcp_dataset/transforms.json";
    const std::filesystem::path output_path = "/tmp/custom_output";
    const auto result = lfs::mcp::ToolRegistry::instance().call_tool(
        "scene.load_dataset",
        json{
            {"path", dataset_path.string()},
            {"output_path", output_path.string()},
            {"strategy", "igs+"},
            {"images_folder", "images_8"},
            {"max_iterations", 42},
        });

    ASSERT_TRUE(result["success"].get<bool>());
    ASSERT_TRUE(backend.load_dataset_called);
    EXPECT_EQ(backend.loaded_params.dataset.output_path, output_path);
    EXPECT_EQ(backend.loaded_params.dataset.images, "images_8");
    EXPECT_EQ(backend.loaded_params.optimization.iterations, 42u);
    EXPECT_EQ(backend.loaded_params.optimization.strategy,
              std::string(lfs::core::param::kStrategyIGSPlus));
    EXPECT_EQ(result["output_path"].get<std::string>(), output_path.string());
    EXPECT_EQ(result["strategy"].get<std::string>(), std::string(lfs::core::param::kStrategyIGSPlus));
}

TEST(McpSharedSceneToolsTest, LoadDatasetAppliesMrnfDefaultsWhenStrategyOmitted) {
    ScopedSharedSceneToolRegistration cleanup;
    FakeSharedSceneBackend backend;
    lfs::mcp::register_shared_scene_tools(backend.backend());

    const std::filesystem::path dataset_path = "/tmp/mcp_dataset";
    const auto result = lfs::mcp::ToolRegistry::instance().call_tool(
        "scene.load_dataset",
        json{{"path", dataset_path.string()}});

    ASSERT_TRUE(result["success"].get<bool>());
    ASSERT_TRUE(backend.load_dataset_called);
    const auto expected = lfs::core::param::OptimizationParameters::mrnf_defaults();
    EXPECT_EQ(backend.loaded_params.optimization.max_cap, expected.max_cap);
    EXPECT_FLOAT_EQ(backend.loaded_params.optimization.opacity_reg, expected.opacity_reg);
    EXPECT_EQ(backend.loaded_params.optimization.refine_every, expected.refine_every);
}

TEST(McpSharedSceneToolsTest, LoadDatasetAppliesIgsPlusStrategyDefaults) {
    ScopedSharedSceneToolRegistration cleanup;
    FakeSharedSceneBackend backend;
    lfs::mcp::register_shared_scene_tools(backend.backend());

    const std::filesystem::path dataset_path = "/tmp/mcp_dataset";
    const auto result = lfs::mcp::ToolRegistry::instance().call_tool(
        "scene.load_dataset",
        json{{"path", dataset_path.string()}, {"strategy", "igs+"}});

    ASSERT_TRUE(result["success"].get<bool>());
    ASSERT_TRUE(backend.load_dataset_called);
    const auto expected = lfs::core::param::OptimizationParameters::igs_plus_defaults();
    EXPECT_EQ(backend.loaded_params.optimization.max_cap, expected.max_cap);
    EXPECT_EQ(backend.loaded_params.optimization.refine_every, expected.refine_every);
}

TEST(McpSharedSceneToolsTest, LoadDatasetPreservesMaxIterationsWithMcmcDefaults) {
    ScopedSharedSceneToolRegistration cleanup;
    FakeSharedSceneBackend backend;
    lfs::mcp::register_shared_scene_tools(backend.backend());

    const std::filesystem::path dataset_path = "/tmp/mcp_dataset";
    const auto result = lfs::mcp::ToolRegistry::instance().call_tool(
        "scene.load_dataset",
        json{
            {"path", dataset_path.string()},
            {"strategy", "mcmc"},
            {"max_iterations", 12345},
        });

    ASSERT_TRUE(result["success"].get<bool>());
    ASSERT_TRUE(backend.load_dataset_called);
    EXPECT_EQ(backend.loaded_params.optimization.iterations, 12345u);
    const auto expected = lfs::core::param::OptimizationParameters::mcmc_defaults();
    EXPECT_EQ(backend.loaded_params.optimization.max_cap, expected.max_cap);
    EXPECT_EQ(backend.loaded_params.optimization.refine_every, expected.refine_every);
    EXPECT_FLOAT_EQ(backend.loaded_params.optimization.opacity_reg, expected.opacity_reg);
}

TEST(McpSharedSceneToolsTest, GetLastErrorReturnsEnvelopeWhenLatched) {
    ScopedSharedSceneToolRegistration cleanup;
    FakeSharedSceneBackend fake;
    auto backend = fake.backend();
    backend.last_training_error = []() -> std::optional<lfs::Error> {
        return lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::ResourceExhausted,
            .domain = lfs::ErrorDomain::CUDA,
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
    };
    lfs::mcp::register_shared_scene_tools(backend);

    const auto result =
        lfs::mcp::ToolRegistry::instance().call_tool("training.get_last_error", json::object());

    ASSERT_TRUE(result["success"].get<bool>());
    EXPECT_EQ(result["last_error"]["code"], "ResourceExhausted");
    EXPECT_EQ(result["last_error"]["domain"], "CUDA");
    EXPECT_EQ(result["last_error_message"], result["last_error"]["message"]);
}

TEST(McpSharedSceneToolsTest, GetLastErrorReturnsNullWhenNoFailureLatched) {
    ScopedSharedSceneToolRegistration cleanup;
    FakeSharedSceneBackend fake;
    auto backend = fake.backend();
    backend.last_training_error = []() -> std::optional<lfs::Error> { return std::nullopt; };
    lfs::mcp::register_shared_scene_tools(backend);

    const auto result =
        lfs::mcp::ToolRegistry::instance().call_tool("training.get_last_error", json::object());

    ASSERT_TRUE(result["success"].get<bool>());
    EXPECT_TRUE(result["last_error"].is_null());
    EXPECT_TRUE(result["last_error_message"].is_null());
}

TEST(McpSharedSceneToolsTest, LoadDatasetRejectsInvalidParametersBeforeReachingTheBackend) {
    ScopedSharedSceneToolRegistration cleanup;
    FakeSharedSceneBackend backend;
    lfs::mcp::register_shared_scene_tools(backend.backend());

    const auto result = lfs::mcp::ToolRegistry::instance().call_tool(
        "scene.load_dataset",
        json{{"path", "/tmp/mcp_dataset"}, {"strategy", "igs+"}, {"max_iterations", 0}});

    ASSERT_TRUE(result.contains("error_message")) << result.dump();
    EXPECT_NE(result["error_message"].get<std::string>().find("iterations"), std::string::npos)
        << result.dump();
    EXPECT_FALSE(backend.load_dataset_called);
}

TEST(McpSharedSceneToolsTest, TrainingStartForwardsOverwriteConsent) {
    ScopedSharedSceneToolRegistration cleanup;
    FakeSharedSceneBackend backend;
    lfs::mcp::register_shared_scene_tools(backend.backend());
    auto& registry = lfs::mcp::ToolRegistry::instance();

    ASSERT_TRUE(registry.call_tool("training.start", json::object())["success"].get<bool>());
    EXPECT_EQ(backend.start_overwrite, false);

    ASSERT_TRUE(registry.call_tool("training.start", json{{"overwrite", true}})["success"].get<bool>());
    EXPECT_EQ(backend.start_overwrite, true);
}

TEST(McpSharedSceneToolsTest, RenderCaptureRejectsSizesOutsideTheCaptureLimit) {
    ScopedSharedSceneToolRegistration cleanup;
    FakeSharedSceneBackend backend;
    lfs::mcp::register_shared_scene_tools(backend.backend());
    auto& registry = lfs::mcp::ToolRegistry::instance();

    for (const auto& args : {json{{"width", 0}}, json{{"width", -1}}, json{{"height", 65536}},
                             json{{"width", 65536}, {"height", 65536}},
                             json{{"width", lfs::mcp::MAX_CAPTURE_DIMENSION + 1}}}) {
        const auto result = registry.call_tool("render.capture", args);
        EXPECT_EQ(result["error"]["code"], "InvalidArgument") << args.dump() << " -> " << result.dump();
    }
    EXPECT_EQ(backend.capture_calls, 0) << "an out-of-range capture size reached the backend";

    EXPECT_EQ(registry.call_tool("render.capture", json{{"width", lfs::mcp::MAX_CAPTURE_DIMENSION}})["success"], true);
    EXPECT_EQ(backend.capture_calls, 1);
}

TEST(McpSharedSceneToolsTest, RenderCaptureKeepsTheBackendErrorCode) {
    ScopedSharedSceneToolRegistration cleanup;
    FakeSharedSceneBackend fake;
    auto backend = fake.backend();
    backend.render_capture = [](int, int, bool) -> lfs::Result<std::string> {
        return lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::InvalidArgument,
            .domain = lfs::ErrorDomain::MCP,
            .user_message = "Capture size 20603x16384 exceeds the 16384 pixel limit per side",
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
    };
    lfs::mcp::register_shared_scene_tools(backend);

    const auto result = lfs::mcp::ToolRegistry::instance().call_tool("render.capture", json{{"height", 16384}});
    EXPECT_EQ(result["error"]["code"], "InvalidArgument") << result.dump();
    EXPECT_EQ(result["error_message"], "Capture size 20603x16384 exceeds the 16384 pixel limit per side");
}

TEST(McpSharedSceneToolsTest, RenderCaptureDoesNotAdvertiseCameraIndex) {
    ScopedSharedSceneToolRegistration cleanup;
    FakeSharedSceneBackend backend;
    lfs::mcp::register_shared_scene_tools(backend.backend());

    const auto tools = lfs::mcp::ToolRegistry::instance().list_tools();
    const auto capture = std::ranges::find_if(tools, [](const auto& tool) { return tool.name == "render.capture"; });
    ASSERT_NE(capture, tools.end());
    EXPECT_FALSE(capture->input_schema.properties.contains("camera_index"))
        << "render.capture advertises camera_index, which no backend can render";
}
