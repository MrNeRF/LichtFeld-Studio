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

    INSTANTIATE_TEST_SUITE_P(Backends, SplatPointRendererContracts,
                             testing::ValuesIn(kCompiledGpuBackends),
                             [](const auto& info) { return std::string(gpu_backend_name(info.param)); });
} // namespace
