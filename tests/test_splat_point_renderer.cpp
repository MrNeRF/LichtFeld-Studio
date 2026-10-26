/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "point_cloud_renderer.hpp"
#include "splat_contract_test_utils.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace lfs::rendering;
    using namespace lfs::test::splat;

    class SplatPointRendererContracts : public testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (backend_unavailable_or_cuda(GetParam()))
                GTEST_SKIP();
            scope_ = std::make_unique<GpuBackendScope>(GetParam());
        }
        std::unique_ptr<GpuBackendScope> scope_;
    };

    TEST_P(SplatPointRendererContracts, PixelsDepthCropAndSelectionMatchPointSemantics) {
        constexpr uint32_t width = 64, height = 64;
        const auto pixel_position = [](const uint32_t x, const uint32_t y, const float depth) {
            return std::array{2.0f * (float(x) + 0.5f) / float(width) - 1.0f,
                              2.0f * (float(y) + 0.5f) / float(height) - 1.0f,
                              -depth};
        };
        std::vector<float> positions, colors;
        for (const auto point : {pixel_position(16, 16, 2.0f), pixel_position(16, 16, 4.0f),
                                 pixel_position(44, 16, 3.0f), pixel_position(16, 44, 2.0f)})
            positions.insert(positions.end(), point.begin(), point.end());
        const std::array<std::array<float, 3>, 4> point_colors{{
            {0.0f, 1.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f},
            {1.0f, 1.0f, 0.0f},
        }};
        for (const auto& color : point_colors)
            colors.insert(colors.end(), color.begin(), color.end());
        const std::array<uint8_t, 4> selection{0, 0, 1, 0};
        std::array<std::array<float, 4>, 257> palette{};
        palette[1] = {1.0f, 0.0f, 0.0f, 1.0f};
        palette[256] = {0.0f, 1.0f, 0.0f, 1.0f};

        auto position_tensor = Tensor::from_blob(positions.data(), {4, 3}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto color_tensor = Tensor::from_blob(colors.data(), {4, 3}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto selection_tensor = upload(selection);
        PointParameters parameters;
        parameters.view = parameters.view_projection = parameters.crop_to_local = identity();
        // Orthographic screen x/y with monotonically increasing OpenGL depth.
        parameters.view_projection[10] = -0.1f;
        parameters.view_projection[14] = 0.0f;
        parameters.crop_min = {-1.0f, -1.0f, -10.0f, 0.0f};
        parameters.crop_max = {1.0f, 0.0f, -1.0f, 0.0f};
        parameters.voxel_focal_ortho = {0.125f, 1.0f, float(height), 0.0f};
        parameters.counts = {0, 0, 1u | 8u | 32u, 64};

        SplatPointRenderer renderer(GetParam());
        const SplatPointInputs inputs{
            .positions = &position_tensor,
            .colors = &color_tensor,
            .selection = &selection_tensor,
            .selection_palette = std::as_bytes(std::span(palette)),
        };
        const std::array<float, 4> background{0.1f, 0.2f, 0.3f, 0.0f};
        const auto rendered = renderer.render(inputs, parameters, width, height, background);
        ASSERT_TRUE(rendered) << rendered.error().detail();

        const auto rgba = renderer.color().to(Device::CPU);
        const auto depth = renderer.linear_depth().to(Device::CPU);
        const auto channel = [&](const uint32_t x, const uint32_t y, const uint32_t c) {
            return rgba.ptr<uint8_t>()[(size_t(y) * width + x) * 4 + c];
        };
        const auto linear = [&](const uint32_t x, const uint32_t y) {
            return depth.ptr<float>()[size_t(y) * width + x];
        };

        // The nearer coincident point wins and publishes linear view depth.
        EXPECT_NEAR(channel(16, 16, 0), 0, 1);
        EXPECT_NEAR(channel(16, 16, 1), 255, 1);
        EXPECT_NEAR(channel(16, 16, 2), 0, 1);
        EXPECT_NEAR(linear(16, 16), 2.0f, 1.0e-6f);
        // Radius four: the horizontal edge is inside, while the square corner
        // is outside the native point-coordinate disc.
        EXPECT_EQ(channel(19, 16, 1), 255);
        EXPECT_NEAR(channel(19, 19, 0), 26, 1);
        EXPECT_NEAR(channel(19, 19, 1), 51, 1);
        EXPECT_EQ(channel(19, 19, 3), 0);
        EXPECT_FLOAT_EQ(linear(19, 19), -1.0f);
        // Group one blends 75% red over the source blue.
        EXPECT_NEAR(channel(44, 16, 0), 191, 1);
        EXPECT_NEAR(channel(44, 16, 1), 0, 1);
        EXPECT_NEAR(channel(44, 16, 2), 64, 1);
        EXPECT_NEAR(linear(44, 16), 3.0f, 1.0e-6f);
        // The lower point falls outside the active crop volume.
        EXPECT_NEAR(channel(16, 44, 0), 26, 1);
        EXPECT_NEAR(channel(16, 44, 1), 51, 1);
        EXPECT_NEAR(channel(16, 44, 2), 77, 1);
        EXPECT_EQ(channel(16, 44, 3), 0);
        EXPECT_FLOAT_EQ(linear(16, 44), -1.0f);
    }

    TEST_P(SplatPointRendererContracts, QueuedFramesKeepTheirOwnCameraAndCropParameters) {
        constexpr uint32_t width = 32, height = 32;
        std::array<float, 3> position{0.0f, 0.0f, -2.0f}, color{0.0f, 1.0f, 0.0f};
        auto positions = Tensor::from_blob(position.data(), {1, 3}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto colors = Tensor::from_blob(color.data(), {1, 3}, Device::CPU, DataType::Float32).to(Device::GPU);
        const SplatPointInputs inputs{.positions = &positions, .colors = &colors};
        SplatPointRenderer renderer(GetParam());
        std::vector<Tensor> frames, depths;
        for (uint32_t frame = 0; frame < 12; ++frame) {
            PointParameters parameters;
            parameters.view = parameters.view_projection = parameters.crop_to_local = identity();
            parameters.view_projection[10] = -0.1f;
            parameters.view_projection[12] = (frame % 2) ? 0.5f : -0.5f;
            parameters.crop_min = {-1.0f, -1.0f, -3.0f, 0.0f};
            parameters.crop_max = {1.0f, 1.0f, -1.0f, 0.0f};
            parameters.voxel_focal_ortho = {0.125f, 1.0f, float(height), 0.0f};
            parameters.counts = {0, 0, 1u | 8u, 32};
            if (frame % 3 == 2)
                parameters.crop_min[0] = 0.25f;
            const auto result = renderer.render(inputs, parameters, width, height, {0, 0, 0, 0});
            ASSERT_TRUE(result) << result.error().detail();
            frames.push_back(renderer.color().clone());
            depths.push_back(renderer.linear_depth().clone());
        }
        for (uint32_t frame = 0; frame < frames.size(); ++frame) {
            SCOPED_TRACE(frame);
            const auto rgba = frames[frame].to(Device::CPU);
            const auto depth = depths[frame].to(Device::CPU);
            const uint32_t x = (frame % 2) ? 24 : 8;
            const auto pixel = size_t(16) * width + x;
            const bool cropped = frame % 3 == 2;
            EXPECT_EQ(rgba.ptr<uint8_t>()[pixel * 4 + 1], cropped ? 0 : 255);
            EXPECT_FLOAT_EQ(depth.ptr<float>()[pixel], cropped ? -1.0f : 2.0f);
            const auto other = size_t(16) * width + (32 - x);
            EXPECT_EQ(rgba.ptr<uint8_t>()[other * 4 + 1], 0);
            EXPECT_FLOAT_EQ(depth.ptr<float>()[other], -1.0f);
        }
    }

    TEST_P(SplatPointRendererContracts, CropPolygonHidesOrDimsConcaveNotchAndOutsideDepth) {
        const std::vector<float> xyz{-.5f, -.5f, -2.f, .5f, .5f, -2.f, .5f, -.5f, -2.f, -.5f, .5f, -4.f};
        auto positions = Tensor::from_vector(xyz, {4, 3}, Device::GPU);
        auto colors = Tensor::from_vector({1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f}, {4, 3}, Device::GPU);
        const std::array<std::array<float, 2>, 6> polygon{{{-1, -1}, {1, -1}, {1, 0}, {0, 0}, {0, 1}, {-1, 1}}};
        const SplatPointInputs inputs{.positions = &positions, .colors = &colors, .crop_polygon = std::as_bytes(std::span(polygon))};
        PointParameters params;
        params.view = params.view_projection = params.crop_to_local = identity();
        params.view_projection[10] = -.1f;
        params.crop_to_local[14] = 2.f;
        params.crop_min = {.5f, 6.f, 0, 0};
        params.voxel_focal_ortho = {.0625f, 1, 64, 0};
        SplatPointRenderer renderer(GetParam());
        for (const bool inverse : {false, true}) {
            for (const bool dim : {false, true}) {
                params.counts = {0, 0, 4097u | 8u | (inverse ? 2u : 0u) | (dim ? 4u : 0u), 64};
                const auto rendered = renderer.render(inputs, params, 64, 64, {0, 0, 0, 0});
                ASSERT_TRUE(rendered) << rendered.error().detail();
                const auto rgba = renderer.color().cpu();
                const auto depth = renderer.linear_depth().cpu();
                const std::array<size_t, 4> pixels{16 * 64 + 16, 48 * 64 + 48, 16 * 64 + 48, 48 * 64 + 16};
                for (size_t i = 0; i < 4; ++i) {
                    const bool outside = ((i == 0 || i == 2) == inverse);
                    const bool visible = !outside || dim;
                    EXPECT_EQ(rgba.ptr<uint8_t>()[4 * pixels[i] + 3], visible ? 255 : 0);
                    EXPECT_NEAR(rgba.ptr<uint8_t>()[4 * pixels[i]],
                                !visible ? 0.f : outside ? 255.f * (.25f + .75f * .299f)
                                                         : 255.f,
                                1.f);
                    EXPECT_FLOAT_EQ(depth.ptr<float>()[pixels[i]], visible ? (i == 3 ? 4.f : 2.f) : -1.f);
                }
            }
        }
        // Reusing the renderer after a polygon frame must not keep its crop enabled.
        params.counts[2] = 8u;
        ASSERT_TRUE(renderer.render(inputs, params, 64, 64, {0, 0, 0, 0}));
        const auto depth = renderer.linear_depth().cpu();
        EXPECT_FLOAT_EQ(depth.ptr<float>()[48 * 64 + 48], 2.f);
        EXPECT_FLOAT_EQ(depth.ptr<float>()[48 * 64 + 16], 4.f);
    }

    INSTANTIATE_TEST_SUITE_P(Backends, SplatPointRendererContracts,
                             testing::ValuesIn(kCompiledGpuBackends),
                             [](const auto& info) { return std::string(gpu_backend_name(info.param)); });
} // namespace
