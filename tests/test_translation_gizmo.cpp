/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/translation_gizmo.hpp"
#include "rendering/render_constants.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

namespace lfs::vis::gui {
    namespace {
        TEST(TranslationGizmo, OrthographicAxisDragIgnoresEdgeOnPlanesAndPassiveViews) {
            NativeOverlayDrawList draw_list;
            TranslationGizmoConfig config;
            config.id = 901;
            config.viewport_size = {800, 600};
            config.view = glm::lookAt(glm::vec3(0, 0, 5), glm::vec3(0), glm::vec3(0, 1, 0));
            config.projection = lfs::rendering::createProjectionMatrixFromFocal({800, 600}, 35, true, 220);
            config.draw_list = &draw_list;
            config.input = {{475, 300}, true, true};
            auto result = drawTranslationGizmo(config);
            EXPECT_TRUE(result.active);
            EXPECT_EQ(result.active_handle, TranslationGizmoHandle::X);

            auto passive = config;
            passive.id = -1;
            passive.viewport_pos = {800, 0};
            passive.input = {{-1, -1}, false, false};
            passive.input_enabled = false;
            EXPECT_FALSE(drawTranslationGizmo(passive).active);

            config.input = {{510, 300}, true, false};
            result = drawTranslationGizmo(config);
            EXPECT_TRUE(result.active);
            EXPECT_TRUE(result.changed);
            EXPECT_NEAR(result.delta_translation.x, 35.0f / 220.0f, 1.0e-5f);
            EXPECT_NEAR(result.delta_translation.y, 0.0f, 1.0e-6f);
            EXPECT_NEAR(result.delta_translation.z, 0.0f, 1.0e-6f);
            // Repeating the draw at the same pointer applies no second edit.
            EXPECT_FALSE(drawTranslationGizmo(config).changed);
            config.input.mouse_left_down = false;
            EXPECT_FALSE(drawTranslationGizmo(config).active);
            clearLineRendererCommands();
        }
    } // namespace
} // namespace lfs::vis::gui
