/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/events.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "core/tensor.hpp"
#include "gui/gui_focus_state.hpp"
#include "input/input_controller.hpp"
#include "internal/viewport.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "tools/tool_base.hpp"

#include <filesystem>
#include <gtest/gtest.h>

namespace lfs::vis {

    namespace {
        class InputControllerDatasetLoadTest : public ::testing::Test {
        protected:
            void SetUp() override {
                lfs::event::EventBridge::instance().clear_all();
                services().clear();
                gui::guiFocusState().reset();
            }

            void TearDown() override {
                gui::guiFocusState().reset();
                services().clear();
                lfs::event::EventBridge::instance().clear_all();
            }
        };

        std::shared_ptr<core::PointCloud> makePointCloud(const std::vector<float>& positions) {
            auto means = core::Tensor::from_vector(
                positions,
                {positions.size() / 3, size_t{3}},
                core::Device::CPU);
            auto colors = core::Tensor::from_vector(
                std::vector<float>(positions.size(), 1.0f),
                {positions.size() / 3, size_t{3}},
                core::Device::CPU);
            return std::make_shared<core::PointCloud>(std::move(means), std::move(colors));
        }
    } // namespace

    // Dataset load deliberately resets to the home pose instead of framing the
    // scene (b8f9d6b8 "revert cam to home pos at dataloading").
    TEST_F(InputControllerDatasetLoadTest, DatasetLoadResetsCameraToHome) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        SceneManager scene_manager;
        scene_manager.getScene().addPointCloud(
            "points",
            makePointCloud({
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                2.0f,
                4.0f,
            }));

        ToolContext tool_context(nullptr, &scene_manager, &viewport, nullptr);
        controller.setToolContext(&tool_context);

        viewport.camera.home_t = glm::vec3(123.0f, 456.0f, 789.0f);
        viewport.camera.home_pivot = glm::vec3(10.0f, 20.0f, 30.0f);
        viewport.camera.home_R = glm::mat3(1.0f);
        viewport.camera.t = glm::vec3(-3.0f, 4.0f, 12.0f);
        viewport.camera.pivot = glm::vec3(1.0f, 1.0f, 1.0f);
        viewport.camera.R = glm::mat3(1.0f);

        core::events::state::DatasetLoadCompleted{
            .path = {},
            .success = true,
            .error = std::nullopt,
            .num_images = 0,
            .num_points = 2,
        }
            .emit();

        EXPECT_EQ(viewport.camera.t, viewport.camera.home_t);
        EXPECT_EQ(viewport.camera.pivot, viewport.camera.home_pivot);
        EXPECT_EQ(viewport.camera.home_t, glm::vec3(123.0f, 456.0f, 789.0f));
        EXPECT_EQ(viewport.camera.home_pivot, glm::vec3(10.0f, 20.0f, 30.0f));
    }

    TEST_F(InputControllerDatasetLoadTest, DroppedHdrUpdatesEnvironmentRenderSettings) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        RenderingManager rendering_manager;
        services().set(&rendering_manager);

        const auto drop_path = std::filesystem::temp_directory_path() / "drag_drop_environment.hdr";
        controller.handleFileDrop({lfs::core::path_to_utf8(drop_path)});

        const auto settings = rendering_manager.getSettings();
        EXPECT_EQ(settings.environment_mode, EnvironmentBackgroundMode::Equirectangular);
        EXPECT_EQ(lfs::core::utf8_to_path(settings.environment_map_path), drop_path);
    }

    TEST_F(InputControllerDatasetLoadTest, SingleDroppedVideoShowsVideoExtractor) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);

        std::optional<std::filesystem::path> requested_video_path;
        core::events::cmd::ShowVideoExtractor::when([&](const auto& e) {
            requested_video_path = e.video_path;
        });

        const auto drop_path = std::filesystem::temp_directory_path() / "drag_drop_video.mp4";
        controller.handleFileDrop({lfs::core::path_to_utf8(drop_path)});

        ASSERT_TRUE(requested_video_path.has_value());
        EXPECT_EQ(*requested_video_path, drop_path);
    }

    TEST_F(InputControllerDatasetLoadTest, SingleDroppedVideoExtensionIsCaseInsensitive) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);

        std::optional<std::filesystem::path> requested_video_path;
        core::events::cmd::ShowVideoExtractor::when([&](const auto& e) {
            requested_video_path = e.video_path;
        });

        const auto drop_path = std::filesystem::temp_directory_path() / "drag_drop_video.MP4";
        controller.handleFileDrop({lfs::core::path_to_utf8(drop_path)});

        ASSERT_TRUE(requested_video_path.has_value());
        EXPECT_EQ(*requested_video_path, drop_path);
    }

    TEST_F(InputControllerDatasetLoadTest, VideoMixedWithOtherDropDoesNotShowVideoExtractor) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        RenderingManager rendering_manager;
        services().set(&rendering_manager);

        bool video_extractor_requested = false;
        core::events::cmd::ShowVideoExtractor::when([&](const auto&) {
            video_extractor_requested = true;
        });

        const auto video_path = std::filesystem::temp_directory_path() / "drag_drop_video.mp4";
        const auto hdr_path = std::filesystem::temp_directory_path() / "drag_drop_environment.hdr";
        controller.handleFileDrop({
            lfs::core::path_to_utf8(video_path),
            lfs::core::path_to_utf8(hdr_path),
        });

        EXPECT_FALSE(video_extractor_requested);
        const auto settings = rendering_manager.getSettings();
        EXPECT_EQ(settings.environment_mode, EnvironmentBackgroundMode::Equirectangular);
        EXPECT_EQ(lfs::core::utf8_to_path(settings.environment_map_path), hdr_path);
    }

    TEST_F(InputControllerDatasetLoadTest, SingleDroppedPlyStillUsesSplatLoader) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);

        std::optional<core::events::cmd::LoadFile> load_file;
        bool video_extractor_requested = false;
        core::events::cmd::LoadFile::when([&](const auto& e) {
            load_file = e;
        });
        core::events::cmd::ShowVideoExtractor::when([&](const auto&) {
            video_extractor_requested = true;
        });

        const auto drop_path = std::filesystem::temp_directory_path() / "drag_drop_splat.ply";
        controller.handleFileDrop({lfs::core::path_to_utf8(drop_path)});

        ASSERT_TRUE(load_file.has_value());
        EXPECT_EQ(load_file->path, drop_path);
        EXPECT_FALSE(load_file->is_dataset);
        EXPECT_FALSE(video_extractor_requested);
    }

    TEST_F(InputControllerDatasetLoadTest, UnrecognizedSingleDropStillReportsFailure) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);

        std::optional<core::events::state::FileDropFailed> failure;
        core::events::state::FileDropFailed::when([&](const auto& e) {
            failure = e;
        });

        const auto drop_path = std::filesystem::temp_directory_path() / "drag_drop_unknown.not_video";
        controller.handleFileDrop({lfs::core::path_to_utf8(drop_path)});

        ASSERT_TRUE(failure.has_value());
        ASSERT_EQ(failure->files.size(), 1u);
        EXPECT_EQ(failure->files.front(), lfs::core::path_to_utf8(drop_path));
        EXPECT_NE(failure->error.find(".mp4"), std::string::npos);
    }

} // namespace lfs::vis
