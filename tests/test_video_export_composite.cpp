/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "rendering/rendering.hpp"
#include "rendering/video_composite_utils.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::rendering::FrameMetadata;
    using lfs::rendering::FramePanelMetadata;
    using lfs::rendering::MeshLayer;
    using lfs::rendering::RenderingEngine;
    using lfs::rendering::VideoCompositeFrameRequest;

    constexpr int HEIGHT = 1;
    constexpr float FAR_PLANE = 100.0f;

    [[nodiscard]] Tensor makeTensor(
        const std::vector<float>& values,
        const lfs::core::TensorShape& shape) {
        return Tensor::from_vector(values, shape, Device::CUDA);
    }

    [[nodiscard]] VideoCompositeFrameRequest makeRequest(
        const int width,
        const bool orthographic,
        const MeshLayer* mesh = nullptr,
        const glm::vec3 background = glm::vec3(0.0f)) {
        return VideoCompositeFrameRequest{
            .viewport = {.size = {width, HEIGHT}, .orthographic = orthographic},
            .frame_view =
                {.size = {width, HEIGHT},
                 .far_plane = FAR_PLANE,
                 .orthographic = orthographic,
                 .background_color = background},
            .background_color = background,
            .prerendered_meshes = mesh,
        };
    }

    void expectFramePixels(
        const lfs::rendering::Result<Tensor>& result,
        const std::vector<float>& expected) {
        ASSERT_TRUE(result.has_value()) << result.error();
        const auto actual = result->cpu().to_vector();
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_FLOAT_EQ(actual[i], expected[i]) << "channel-major value " << i;
        }
    }

    class VideoExportCompositePathTest : public testing::TestWithParam<bool> {
    protected:
        void SetUp() override {
            engine_ = RenderingEngine::create();
            ASSERT_NE(engine_, nullptr);
            const auto initialized = engine_->initialize();
            ASSERT_TRUE(initialized.has_value()) << initialized.error();
        }

        [[nodiscard]] bool orthographic() const {
            return GetParam();
        }

        [[nodiscard]] std::optional<lfs::rendering::GpuFrame> materialize(
            const std::vector<float>& color,
            const int width,
            FrameMetadata metadata) {
            auto image = std::make_shared<Tensor>(
                makeTensor(color, {3, static_cast<size_t>(HEIGHT), static_cast<size_t>(width)}));
            auto frame = engine_->materializeGpuFrame(image, metadata, {width, HEIGHT});
            EXPECT_TRUE(frame.has_value()) << frame.error();
            if (!frame) {
                return std::nullopt;
            }
            return *frame;
        }

        std::unique_ptr<RenderingEngine> engine_;
    };

    TEST(VideoCompositeDepthTest, MeshFragmentMustBeOpaqueAndCloser) {
        using lfs::rendering::detail::meshFragmentPassesDepthTest;

        EXPECT_TRUE(meshFragmentPassesDepthTest(1.0f, 2.0f, 5.0f));
        EXPECT_FALSE(meshFragmentPassesDepthTest(0.0f, 2.0f, 5.0f));
        EXPECT_FALSE(meshFragmentPassesDepthTest(1.0f, -2.0f, 5.0f));
        EXPECT_FALSE(meshFragmentPassesDepthTest(1.0f, 5.0f, 2.0f));
        EXPECT_FALSE(meshFragmentPassesDepthTest(1.0f, 5.0f, 5.0f));
    }

    TEST_P(VideoExportCompositePathTest, MeshOnlyUsesBackgroundOutsideTheMesh) {
        constexpr int WIDTH = 2;
        MeshLayer mesh{
            .rgba = makeTensor(
                {1.0f, 0.0f,
                 0.0f, 1.0f,
                 0.0f, 0.0f,
                 1.0f, 0.0f},
                {4, HEIGHT, WIDTH}),
            .view_depth = makeTensor({1.0f, 1.0f}, {HEIGHT, WIDTH}),
        };

        expectFramePixels(
            engine_->renderVideoCompositeFrame(
                std::nullopt,
                makeRequest(WIDTH, orthographic(), &mesh, {0.1f, 0.2f, 0.3f})),
            {1.0f, 0.1f,
             0.0f, 0.2f,
             0.0f, 0.3f});
    }

    TEST_P(VideoExportCompositePathTest, SplatOnlyPreservesThePrimaryFrame) {
        constexpr int WIDTH = 2;
        const std::vector<float> colors{
            1.0f, 0.0f,
            0.0f, 1.0f,
            0.0f, 0.5f};
        FrameMetadata metadata{
            .valid = true,
            .far_plane = FAR_PLANE,
            .orthographic = orthographic(),
        };
        const auto primary = materialize(colors, WIDTH, std::move(metadata));
        ASSERT_TRUE(primary.has_value());

        expectFramePixels(
            engine_->renderVideoCompositeFrame(
                primary,
                makeRequest(WIDTH, orthographic())),
            colors);
    }

    TEST_P(VideoExportCompositePathTest, SplatAndMeshUseLinearViewDepth) {
        constexpr int WIDTH = 4;
        const std::vector<float> primary_colors{
            1.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 0.0f};
        auto primary_depth = std::make_shared<Tensor>(
            makeTensor({2.0f, 4.0f, 6.0f, 8.0f}, {1, HEIGHT, WIDTH}));
        FrameMetadata metadata{
            .depth_panels = {FramePanelMetadata{
                .depth = std::move(primary_depth),
                .start_position = 0.0f,
                .end_position = 1.0f}},
            .depth_panel_count = 1,
            .valid = true,
            .far_plane = FAR_PLANE,
            .orthographic = orthographic(),
        };
        const auto primary = materialize(primary_colors, WIDTH, std::move(metadata));
        ASSERT_TRUE(primary.has_value());

        MeshLayer mesh{
            .rgba = makeTensor(
                {0.0f, 1.0f, 1.0f, 1.0f,
                 1.0f, 0.0f, 0.0f, 0.0f,
                 1.0f, 1.0f, 0.0f, 1.0f,
                 1.0f, 1.0f, 0.0f, 1.0f},
                {4, HEIGHT, WIDTH}),
            .view_depth = makeTensor({1.0f, 5.0f, 0.5f, 8.0f}, {HEIGHT, WIDTH}),
        };

        expectFramePixels(
            engine_->renderVideoCompositeFrame(
                primary,
                makeRequest(WIDTH, orthographic(), &mesh)),
            {0.0f, 0.0f, 0.0f, 1.0f,
             1.0f, 1.0f, 0.0f, 1.0f,
             1.0f, 0.0f, 1.0f, 0.0f});
    }

    TEST_P(VideoExportCompositePathTest, SplitViewUsesEachPanelsDepth) {
        constexpr int WIDTH = 4;
        const std::vector<float> primary_colors{
            1.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 1.0f};
        auto left_depth = std::make_shared<Tensor>(
            makeTensor({2.0f, 2.0f}, {1, HEIGHT, 2}));
        auto right_depth = std::make_shared<Tensor>(
            makeTensor({8.0f, 8.0f}, {1, HEIGHT, 2}));
        FrameMetadata metadata{
            .depth_panels =
                {FramePanelMetadata{
                     .depth = std::move(left_depth),
                     .start_position = 0.0f,
                     .end_position = 0.5f},
                 FramePanelMetadata{
                     .depth = std::move(right_depth),
                     .start_position = 0.5f,
                     .end_position = 1.0f}},
            .depth_panel_count = 2,
            .valid = true,
            .far_plane = FAR_PLANE,
            .orthographic = orthographic(),
        };
        const auto primary = materialize(primary_colors, WIDTH, std::move(metadata));
        ASSERT_TRUE(primary.has_value());

        MeshLayer mesh{
            .rgba = makeTensor(
                {0.0f, 0.0f, 0.0f, 0.0f,
                 1.0f, 1.0f, 1.0f, 1.0f,
                 0.0f, 0.0f, 0.0f, 0.0f,
                 1.0f, 1.0f, 1.0f, 1.0f},
                {4, HEIGHT, WIDTH}),
            .view_depth = makeTensor({5.0f, 5.0f, 5.0f, 5.0f}, {HEIGHT, WIDTH}),
        };

        expectFramePixels(
            engine_->renderVideoCompositeFrame(
                primary,
                makeRequest(WIDTH, orthographic(), &mesh)),
            {1.0f, 1.0f, 0.0f, 0.0f,
             0.0f, 0.0f, 1.0f, 1.0f,
             0.0f, 0.0f, 0.0f, 0.0f});
    }

    INSTANTIATE_TEST_SUITE_P(
        PerspectiveAndOrthographic,
        VideoExportCompositePathTest,
        testing::Values(false, true),
        [](const testing::TestParamInfo<bool>& info) {
            return info.param ? std::string("Orthographic") : std::string("Perspective");
        });

} // namespace
