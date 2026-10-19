/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/point_cloud.hpp"
#include "core/tensor.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "window/vulkan_context.hpp"

#include <SDL3/SDL.h>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    TEST(ViewportPointCloudExport, PointCloudWithoutGaussiansExportsTheCloud) {
        if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
            GTEST_SKIP() << SDL_GetError();
        struct Quit {
            ~Quit() { SDL_QuitSubSystem(SDL_INIT_VIDEO); }
        } quit;
        SDL_Window* const window = SDL_CreateWindow(
            "Point cloud export", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
        if (!window)
            GTEST_SKIP() << SDL_GetError();
        const std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window_owner(window, SDL_DestroyWindow);
        lfs::vis::VulkanContext context;
        if (!context.init(window, 64, 64))
            GTEST_SKIP() << context.lastError();

        lfs::vis::RenderingManager manager;
        auto settings = manager.getSettings();
        settings.background_color = glm::vec3(0.0f);
        settings.voxel_size = 0.05f;
        settings.show_grid = true;
        settings.show_camera_frustums = true;
        settings.equirectangular = true;
        manager.updateSettings(settings);

        // Forward red point, plus a green point behind the camera. Panorama export would
        // draw the green point; perspective export, matching splat export, must not.
        auto means = Tensor::from_vector(std::vector<float>{0.0f, 0.0f, 2.0f, 0.0f, 0.0f, -2.0f},
                                         {size_t{2}, size_t{3}}, Device::CPU)
                         .to(DataType::Float32);
        auto colors = Tensor::from_vector(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f},
                                          {size_t{2}, size_t{3}}, Device::CPU)
                          .to(DataType::Float32);
        lfs::vis::SceneManager scene;
        scene.getScene().addPointCloud(
            "points", std::make_shared<lfs::core::PointCloud>(std::move(means), std::move(colors)));

        const Viewport viewport(64, 64);
        manager.renderVulkanFrame({.viewport = viewport, .settings = settings, .vulkan_context = &context});

        constexpr int kSize = 64;
        for (const bool transparent : {false, true}) {
            SCOPED_TRACE(transparent);
            lfs::vis::RenderingManager::ExportImageRequest request;
            request.focal_length_mm = 35.0f;
            request.width = kSize;
            request.height = kSize;
            request.mode = transparent ? lfs::vis::ExportPostProcessMode::Transparent
                                       : lfs::vis::ExportPostProcessMode::Opaque;
            auto image = manager.renderExportImage(&scene, request);
            ASSERT_TRUE(image) << image.error();
            const int channels = static_cast<int>(image->size(2));
            ASSERT_EQ(channels, transparent ? 4 : 3);
            const auto* pixels = image->ptr<std::uint8_t>();
            const auto* center = pixels + (32 * kSize + 32) * channels;
            EXPECT_GE(center[0], 250);
            EXPECT_EQ(center[1], 0);
            EXPECT_EQ(center[2], 0);
            if (transparent)
                EXPECT_EQ(center[3], 255);
            const auto* corner = pixels;
            EXPECT_EQ(corner[0] + corner[1] + corner[2], 0);
            if (transparent)
                EXPECT_EQ(corner[3], 0);
            int colored = 0;
            int red = 0;
            for (int i = 0; i < kSize * kSize; ++i) {
                const auto* pixel = pixels + i * channels;
                const bool background = pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0 &&
                                        (channels == 3 || pixel[3] == 0);
                colored += !background;
                red += pixel[0] >= 250 && pixel[1] == 0 && pixel[2] == 0;
            }
            EXPECT_EQ(red, colored);
            EXPECT_GE(red, 8);
            EXPECT_LE(red, 200);
        }
    }

} // namespace
