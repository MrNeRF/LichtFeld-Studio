/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/events.hpp"
#include "core/point_cloud.hpp"
#include "core/tensor.hpp"
#include "gui/gui_focus_state.hpp"
#include "input/input_controller.hpp"
#include "internal/viewport.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "scene/scene_manager.hpp"
#include "tools/tool_base.hpp"

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

    TEST_F(InputControllerDatasetLoadTest, DatasetLoadFramesSceneAndUpdatesHomeWhenBoundsAvailable) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        SceneManager scene_manager;
        scene_manager.getScene().addPointCloud(
            "points",
            makePointCloud({
                0.0f, 0.0f, 0.0f,
                0.0f, 2.0f, 4.0f,
            }));

        ToolContext tool_context(nullptr, &scene_manager, &viewport, nullptr);
        controller.setToolContext(&tool_context);

        viewport.camera.home_t = glm::vec3(123.0f, 456.0f, 789.0f);
        viewport.camera.home_pivot = glm::vec3(10.0f, 20.0f, 30.0f);
        viewport.camera.home_R = glm::mat3(1.0f);
        viewport.camera.t = viewport.camera.home_t;
        viewport.camera.pivot = viewport.camera.home_pivot;
        viewport.camera.R = viewport.camera.home_R;

        core::events::state::DatasetLoadCompleted{
            .path = {},
            .success = true,
            .error = std::nullopt,
            .num_images = 0,
            .num_points = 2,
        }
            .emit();

        const glm::vec3 expected_pivot =
            lfs::rendering::visualizerWorldPointFromDataWorld(glm::vec3(0.0f, 1.0f, 2.0f));
        EXPECT_NEAR(viewport.camera.getPivot().x, expected_pivot.x, 1e-5f);
        EXPECT_NEAR(viewport.camera.getPivot().y, expected_pivot.y, 1e-5f);
        EXPECT_NEAR(viewport.camera.getPivot().z, expected_pivot.z, 1e-5f);

        EXPECT_NE(viewport.camera.home_t, glm::vec3(123.0f, 456.0f, 789.0f));
        EXPECT_NE(viewport.camera.home_pivot, glm::vec3(10.0f, 20.0f, 30.0f));
        EXPECT_EQ(viewport.camera.home_t, viewport.camera.t);
        EXPECT_EQ(viewport.camera.home_pivot, viewport.camera.pivot);
    }

} // namespace lfs::vis
