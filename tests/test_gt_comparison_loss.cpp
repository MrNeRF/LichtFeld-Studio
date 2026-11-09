/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "project/session_state.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/viewport_request_builder.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace lfs::vis {
    TEST(GTComparisonLoss, LossRequestsUseRgbWithoutChangingTheDepthPreference) {
        Viewport viewport{64, 64};
        FrameContext frame{.viewport = viewport};
        frame.settings.depth_view = true;
        frame.settings.split_view_mode = SplitViewMode::GTComparison;
        EXPECT_TRUE(buildViewportRenderRequest(frame, {64, 64}).depth_view);
        frame.settings.gt_comparison_mode = GTComparisonMode::Loss;
        EXPECT_FALSE(buildViewportRenderRequest(frame, {64, 64}).depth_view);
        EXPECT_TRUE(frame.settings.depth_view);
        frame.settings.split_view_mode = SplitViewMode::Disabled;
        EXPECT_TRUE(buildViewportRenderRequest(frame, {64, 64}).depth_view);
    }

    TEST(GTComparisonLoss, MatchingPixelsAreBlackAndLargerErrorsAreBrighter) {
        const glm::vec3 reference{0.4f, 0.5f, 0.6f};
        EXPECT_EQ(gtLossHeatmapColor(reference, reference), glm::vec3(0.0f));
        const auto small = gtLossHeatmapColor(reference, reference + glm::vec3(0.01f));
        const auto large = gtLossHeatmapColor(reference, reference + glm::vec3(0.5f));
        EXPECT_GT(small.r + small.g + small.b, 0.0f);
        EXPECT_GT(large.r + large.g + large.b, small.r + small.g + small.b);
        EXPECT_EQ(gtLossHeatmapColor(reference, reference + glm::vec3(0.5f)),
                  gtLossHeatmapColor(reference + glm::vec3(0.5f), reference));
    }

    TEST(GTComparisonLoss, InvalidPixelsAreHighlighted) {
        for (const float invalid : {std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::quiet_NaN()}) {
            EXPECT_EQ(gtLossHeatmapColor(glm::vec3(0.0f), glm::vec3(invalid)),
                      glm::vec3(1.0f, 1.0f, 0.75f));
        }
    }

    TEST(GTComparisonLoss, UsesRgbReferenceAndSurvivesProjectRoundTrip) {
        EXPECT_TRUE(gtComparisonUsesRGBReference(GTComparisonMode::Loss));
        EXPECT_TRUE(gtComparisonUsesRGBReference(GTComparisonMode::RGB));
        EXPECT_FALSE(gtComparisonUsesRGBReference(GTComparisonMode::Normal));
        EXPECT_FALSE(gtComparisonUsesRGBReference(GTComparisonMode::Depth));
        RenderSettings settings;
        settings.gt_comparison_mode = GTComparisonMode::Loss;
        sanitizeGTComparisonSettings(settings);
        EXPECT_EQ(settings.gt_comparison_mode, GTComparisonMode::Loss);
        const auto restored = project::renderSettingsFromProjectJson(project::renderSettingsToProjectJson(settings));
        ASSERT_TRUE(restored);
        EXPECT_EQ(restored->gt_comparison_mode, GTComparisonMode::Loss);
    }

    TEST(GTComparisonLoss, CaptureCoversBothPanelsWithoutDividerAndPreservesLetterbox) {
        constexpr int width = 69, height = 13;
        constexpr int content_width = 65, content_height = 9;
        const auto image = [](const float value) {
            return std::make_shared<lfs::core::Tensor>(lfs::core::Tensor::full(
                {3, content_height, content_width}, value, lfs::core::Device::CPU));
        };
        VulkanSplitViewParams params;
        params.loss_visualization = true;
        params.content_rect = {2, 2, content_width, content_height};
        params.background = {0.1f, 0.2f, 0.3f};
        params.left.image = image(0.4f);
        params.right.image = image(0.5f);
        // Pool-aligned UV values must be replaced with tight CPU sampling.
        params.left.uv_scale = params.right.uv_scale = {0.6f, 0.6f};
        const auto expected = gtLossHeatmapColor(glm::vec3(0.4f), glm::vec3(128.0f / 255.0f));
        for (const float split : {0.0f, 0.5f, 1.0f}) {
            params.split_position = split;
            const auto capture = RenderingManager::composeSplitViewCpu(params, {width, height});
            ASSERT_TRUE(capture);
            const auto* pixels = capture->ptr<float>();
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const auto color = (x >= 2 && x < width - 2 && y >= 2 && y < height - 2)
                                           ? expected
                                           : params.background;
                    for (int c = 0; c < 3; ++c)
                        EXPECT_NEAR(pixels[(y * width + x) * 3 + c], color[c], 1e-5f);
                }
            }
        }
        params.right.image = params.left.image;
        const auto matching = RenderingManager::composeSplitViewCpu(params, {width, height});
        ASSERT_TRUE(matching);
        EXPECT_FLOAT_EQ(matching->ptr<float>()[(height / 2 * width + width / 2) * 3], 0.0f);
    }

    TEST(GTComparisonLoss, CaptureRoundsAndClampsTexelsBeforeBilinearSampling) {
        constexpr int width = 9, height = 7;
        auto reference = std::make_shared<lfs::core::Tensor>(lfs::core::Tensor::empty(
            {3, height, width}, lfs::core::Device::CPU));
        auto rendered = std::make_shared<lfs::core::Tensor>(lfs::core::Tensor::empty(
            {3, height, width}, lfs::core::Device::CPU, lfs::core::DataType::UInt8));
        constexpr float values[] = {0.5f, -0.5f, 1.5f, 0.1f};
        constexpr std::uint8_t bytes[] = {128, 0, 255, 26};
        for (int i = 0; i < width * height * 3; ++i) {
            reference->ptr<float>()[i] = values[i % 4];
            rendered->ptr<std::uint8_t>()[i] = bytes[i % 4];
        }
        VulkanSplitViewParams params;
        params.loss_visualization = true;
        params.content_rect = {0, 0, 21, 13};
        params.left.image = reference;
        params.right.image = rendered;
        const auto capture = RenderingManager::composeSplitViewCpu(params, {21, 13});
        ASSERT_TRUE(capture);
        for (int i = 0; i < 21 * 13 * 3; ++i)
            EXPECT_FLOAT_EQ(capture->ptr<float>()[i], 0.0f);
        EXPECT_FLOAT_EQ(reference->ptr<float>()[0], 0.5f);
    }

    TEST(GTComparisonLoss, CaptureIgnoresEnvironmentAndAlphaWhenComparingRgb) {
        constexpr int width = 9, height = 7;
        auto image = std::make_shared<lfs::core::Tensor>(lfs::core::Tensor::full(
            {4, height, width}, 0.5f, lfs::core::Device::CPU));
        std::fill_n(image->ptr<float>() + 3 * width * height, width * height, 0.0f);
        VulkanSplitViewParams params;
        params.loss_visualization = true;
        params.content_rect = {0, 0, width, height};
        params.left.image = params.right.image = image;
        params.right.environment.enabled = true;
        const auto capture = RenderingManager::composeSplitViewCpu(params, {width, height});
        ASSERT_TRUE(capture);
        for (int i = 0; i < width * height * 3; ++i)
            EXPECT_FLOAT_EQ(capture->ptr<float>()[i], 0.0f);
    }

    TEST(GTComparisonLoss, LossModeDoesNotExposeDraggableDivider) {
        RenderingManager manager;
        auto settings = manager.getSettings();
        settings.split_view_mode = SplitViewMode::GTComparison;
        settings.gt_comparison_mode = GTComparisonMode::RGB;
        manager.updateSettings(settings);
        EXPECT_TRUE(manager.getSplitDividerScreenX({10.0f, 20.0f}, {800.0f, 600.0f}));
        settings.gt_comparison_mode = GTComparisonMode::Loss;
        manager.updateSettings(settings);
        EXPECT_FALSE(manager.getSplitDividerScreenX({10.0f, 20.0f}, {800.0f, 600.0f}));
    }
} // namespace lfs::vis
