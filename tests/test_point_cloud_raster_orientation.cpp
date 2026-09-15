/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "rendering/point_cloud_raster.cuh"
#include "rendering/render_constants.hpp"
#include "rendering/rendering.hpp"
#include "visualizer/rendering/viewport_artifact_service.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <glm/gtc/type_ptr.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

namespace {

    using lfs::core::Device;
    using lfs::core::Tensor;

    enum class Projection { Perspective,
                            Orthographic,
                            Equirectangular };

    class PointCloudRasterOrientationTest : public ::testing::TestWithParam<std::tuple<Projection, bool>> {
    protected:
        void SetUp() override {
            int device_count = 0;
            const auto status = cudaGetDeviceCount(&device_count);
            if (status != cudaSuccess || device_count == 0) {
                GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
            }
        }
    };

    TEST_P(PointCloudRasterOrientationTest, ColorAndDepthUseTopDownRows) {
        const auto [projection_type, transparent] = GetParam();
        constexpr int width = 129;
        constexpr int height = 65;
        constexpr int pixels = width * height;
        const size_t channels = transparent ? 4 : 3;
        const bool orthographic = projection_type == Projection::Orthographic;
        const bool equirectangular = projection_type == Projection::Equirectangular;
        constexpr float ortho_scale = height / 4.0f;

        auto positions = Tensor::from_vector({0.0f, 1.0f, -2.0f,
                                              0.0f, -1.0f, -3.0f,
                                              1.0f, 0.0f, -4.0f,
                                              -1.0f, 0.0f, -5.0f},
                                             {4, 3}, Device::CUDA);
        auto colors = Tensor::from_vector({1.0f, 0.0f, 0.0f,
                                           0.0f, 1.0f, 0.0f,
                                           0.0f, 0.0f, 1.0f,
                                           1.0f, 1.0f, 0.0f},
                                          {4, 3}, Device::CUDA);
        auto image = Tensor::empty({channels, height, width}, Device::CUDA);
        auto depth = Tensor::empty({1, height, width}, Device::CUDA);
        const glm::mat4 view(1.0f);
        const auto projection = lfs::rendering::createProjectionMatrix(
            {width, height}, 90.0f, orthographic, ortho_scale, 0.1f, 100.0f);

        lfs::rendering::pcraster::LaunchParams params{};
        params.positions = positions.ptr<float>();
        params.colors = colors.ptr<float>();
        params.n_points = 4;
        std::copy_n(glm::value_ptr(view), 16, params.view);
        std::copy_n(glm::value_ptr(projection), 16, params.view_proj);
        params.width = width;
        params.height = height;
        params.channels = static_cast<int>(channels);
        params.equirectangular = equirectangular;
        params.orthographic = orthographic;
        params.ortho_scale = ortho_scale;
        params.focal_y = height / 2.0f;
        params.voxel_size = 0.001f;
        params.scaling_modifier = 1.0f;
        params.far_plane = 100.0f;
        params.bg_a = 1.0f;
        params.transparent_background = transparent;
        params.image = image.ptr<float>();
        params.depth = depth.ptr<float>();
        params.stream = image.stream();

        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(lfs::rendering::pcraster::launchPointCloudRaster(params), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(params.stream), cudaSuccess);
        ASSERT_EQ(image.shape(), (lfs::core::TensorShape{channels, height, width}));
        ASSERT_EQ(depth.shape(), (lfs::core::TensorShape{1, height, width}));
        const auto image_cpu = image.cpu();
        const auto depth_cpu = depth.cpu();
        const float* const rgb = image_cpu.ptr<float>();
        const float* const depths = depth_cpu.ptr<float>();

        const std::array<glm::ivec2, 4> hit_pixels = equirectangular
                                                         ? std::array<glm::ivec2, 4>{{{64, 23}, {64, 39}, {69, 32}, {60, 32}}}
                                                     : orthographic
                                                         ? std::array<glm::ivec2, 4>{{{64, 16}, {64, 48}, {80, 32}, {48, 32}}}
                                                         : std::array<glm::ivec2, 4>{{{64, 16}, {64, 43}, {72, 32}, {58, 32}}};
        const std::array<glm::vec3, 4> expected_colors{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}}};
        EXPECT_LT(hit_pixels[0].y, height / 2);
        EXPECT_GT(hit_pixels[1].y, height / 2);
        EXPECT_GT(hit_pixels[2].x, width / 2);
        EXPECT_LT(hit_pixels[3].x, width / 2);
        for (size_t point = 0; point < hit_pixels.size(); ++point) {
            SCOPED_TRACE(point);
            const auto hit = hit_pixels[point];
            const int pixel = hit.y * width + hit.x;
            const float view_depth = static_cast<float>(point + 2);
            const float expected_depth = equirectangular ? std::sqrt(view_depth * view_depth + 1.0f) : view_depth;
            for (int channel = 0; channel < 3; ++channel) {
                EXPECT_FLOAT_EQ(rgb[channel * pixels + pixel], expected_colors[point][channel]);
            }
            EXPECT_GT(depths[pixel], 0.0f);
            EXPECT_NEAR(depths[pixel], expected_depth, 1e-5f);
            if (transparent) {
                EXPECT_FLOAT_EQ(rgb[3 * pixels + pixel], 1.0f);
            }
        }
        if (transparent) {
            EXPECT_FLOAT_EQ(rgb[3 * pixels], 0.0f);
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        Projections, PointCloudRasterOrientationTest,
        ::testing::Combine(::testing::Values(Projection::Perspective, Projection::Orthographic, Projection::Equirectangular),
                           ::testing::Bool()),
        [](const ::testing::TestParamInfo<PointCloudRasterOrientationTest::ParamType>& info) {
            const auto projection = std::get<0>(info.param);
            const bool transparent = std::get<1>(info.param);
            const char* name = projection == Projection::Perspective ? "Perspective" : projection == Projection::Orthographic ? "Orthographic"
                                                                                                                              : "Equirectangular";
            return std::string(name) + (transparent ? "Rgba" : "Rgb");
        });

    TEST(ViewportArtifactServiceTest, PointCloudDepthUsesTopDownRows) {
        int device_count = 0;
        const auto status = cudaGetDeviceCount(&device_count);
        if (status != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
        }
        auto depth = Tensor::from_vector({2.0f, 3.0f, 4.0f, 5.0f}, {1, 2, 2}, Device::CUDA);
        lfs::rendering::FrameMetadata metadata{};
        metadata.valid = true;
        metadata.depth_panel_count = 1;
        metadata.depth_panels[0].depth = std::make_shared<Tensor>(std::move(depth));
        lfs::vis::ViewportArtifactService artifacts;
        artifacts.updateFromImageOutput({}, metadata, {2, 2}, true);
        EXPECT_FLOAT_EQ(artifacts.sampleLinearDepthAt(0, 0, {2, 2}, std::nullopt), 2.0f);
        EXPECT_FLOAT_EQ(artifacts.sampleLinearDepthAt(1, 1, {2, 2}, std::nullopt), 5.0f);
    }

} // namespace
