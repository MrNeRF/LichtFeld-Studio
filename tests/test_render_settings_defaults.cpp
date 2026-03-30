/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/ipc/view_context.hpp"
#include "visualizer/rendering/rendering_types.hpp"

#include <gtest/gtest.h>

TEST(RenderSettingsDefaults, CameraFrustumsAreDisabledByDefault) {
    const lfs::vis::RenderSettings render_settings;
    const lfs::vis::RenderSettingsProxy proxy_settings;

    EXPECT_FALSE(render_settings.show_camera_frustums);
    EXPECT_FALSE(proxy_settings.show_camera_frustums);
    EXPECT_FLOAT_EQ(render_settings.camera_frustum_scale, 0.25f);
    EXPECT_FLOAT_EQ(proxy_settings.camera_frustum_scale, 0.25f);
}
