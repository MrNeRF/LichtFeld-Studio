/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rml_preview_texture.hpp"

#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <thread>

namespace {

    TEST(PreviewTextureTest, ParsesPreviewRequests) {
        const auto project = lfs::vis::gui::parsePreviewTextureRequest(
            "preview://kind=licht&thumb=256&rev=7&w=640&h=480&path=%2Ftmp%2Fa+b%20c.licht");
        ASSERT_TRUE(project);
        EXPECT_EQ(project->path, std::filesystem::path("/tmp/a b c.licht"));
        EXPECT_EQ(project->max_size, 256);
        EXPECT_TRUE(project->embedded_project_preview);
        EXPECT_EQ(project->layoutDimensions(), (Rml::Vector2i{640, 480}));

        const auto image = lfs::vis::gui::parsePreviewTextureRequest(
            "preview://cam=3&thumb=0&pmw=900&mw=1600&path=x.png");
        ASSERT_TRUE(image);
        EXPECT_EQ(image->max_size, 900);
        EXPECT_FALSE(image->embedded_project_preview);
        EXPECT_EQ(image->layoutDimensions(), (Rml::Vector2i{1, 1}));
        EXPECT_FALSE(lfs::vis::gui::parsePreviewTextureRequest("preview://thumb=256"));
        EXPECT_FALSE(lfs::vis::gui::parsePreviewTextureRequest("file://path=x.png"));
    }

    // Catches a renderer that never sees decoded pixels or a panel cache that never redraws.
    TEST(PreviewTextureTest, LoadsPixelsAndAdvancesGeneration) {
        lfs::vis::gui::PreviewTextureLoader loader;
        const lfs::vis::gui::PreviewTextureRequest request{
            .path = std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/assets/icon/align.png",
            .max_size = 64};
        const Rml::String source = "preview://thumb=64&path=align.png";
        loader.load(Rml::TextureHandle{1}, source, request);
        bool uploaded = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!uploaded && std::chrono::steady_clock::now() < deadline) {
            loader.uploadReady([&](Rml::TextureHandle handle, const Rml::String& loaded_source,
                                   lfs::vis::gui::PreviewTexturePixels& pixels) {
                EXPECT_EQ(handle, Rml::TextureHandle{1});
                EXPECT_EQ(loaded_source, source);
                EXPECT_GT(pixels.width, 0);
                EXPECT_GT(pixels.height, 0);
                EXPECT_EQ(pixels.pixels.size(), std::size_t(pixels.width) * pixels.height * 4);
                uploaded = true;
                return true;
            });
            if (!uploaded)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        EXPECT_TRUE(uploaded);
        EXPECT_EQ(loader.generation(), 1u);
    }

} // namespace
