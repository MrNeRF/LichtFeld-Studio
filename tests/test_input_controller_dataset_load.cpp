/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

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
#include <imgui.h>

namespace lfs::vis {

    namespace {
        class InputControllerDatasetLoadTest : public ::testing::Test {
        protected:
            void SetUp() override {
                services().clear();
                gui::guiFocusState().reset();

                IMGUI_CHECKVERSION();
                ImGui::CreateContext();
            }

            void TearDown() override {
                ImGui::DestroyContext();

                gui::guiFocusState().reset();
                services().clear();
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

} // namespace lfs::vis
