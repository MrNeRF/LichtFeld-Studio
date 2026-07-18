/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/path_utils.hpp"
#include "mcp/mcp_tools.hpp"
#include "mcp/shared_scene_tools.hpp"

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
        "scene.save_checkpoint",
        "scene.save_ply",
        "training.start",
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
                .save_checkpoint =
                    [](const std::optional<std::filesystem::path>& path)
                    -> std::expected<std::filesystem::path, std::string> {
                    return path.value_or(std::filesystem::path{"/tmp/output"});
                },
                .save_ply =
                    [](const std::filesystem::path&) -> std::expected<void, std::string> {
                    return {};
                },
                .start_training = []() -> std::expected<void, std::string> { return {}; },
                .render_capture = [](std::optional<int>, int, int) -> std::expected<std::string, std::string> {
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
