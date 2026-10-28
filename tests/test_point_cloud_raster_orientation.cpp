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
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

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

    // Catches a CUDA point-cloud render (video export, Python rendering) that ignores a crop polygon or crops by its
    // bounds: of an L outline only the point in its arm stays; the points in its notch and past its slab go.
    TEST(PointCloudRasterTest, CropPolygonKeepsOnlyPointsInsideThePrism) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        constexpr int width = 64;
        constexpr int height = 32;
        // Red in the bottom arm, green in the notch, blue over the upper arm but past the slab.
        auto positions = Tensor::from_vector({0.5f, -0.5f, -5.0f, 0.5f, 0.5f, -5.0f, -0.5f, 0.5f, -8.0f}, {3, 3}, Device::CUDA);
        auto colors = Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {3, 3}, Device::CUDA);
        const std::vector<glm::vec2> outline{{-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 1.0f}};
        const glm::mat4 view(1.0f);
        const auto projection = lfs::rendering::createProjectionMatrix({width, height}, 60.0f, false, 1.0f, 0.1f, 100.0f);
        const auto seen = [&](const bool inverse) {
            auto image = Tensor::empty({3, height, width}, Device::CUDA);
            auto depth = Tensor::empty({1, height, width}, Device::CUDA);
            lfs::rendering::pcraster::LaunchParams params{};
            params.positions = positions.ptr<float>();
            params.colors = colors.ptr<float>();
            params.n_points = 3;
            std::copy_n(glm::value_ptr(view), 16, params.view);
            std::copy_n(glm::value_ptr(projection), 16, params.view_proj);
            params.has_crop_polygon = true;
            // The prism sits at z = -5 with |z| <= 1.
            std::copy_n(glm::value_ptr(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 5.0f))), 16,
                        params.crop_polygon.to_local);
            for (size_t i = 0; i < outline.size(); ++i) {
                params.crop_polygon.vertices[2 * i] = outline[i].x;
                params.crop_polygon.vertices[2 * i + 1] = outline[i].y;
            }
            params.crop_polygon.count = static_cast<int>(outline.size());
            params.crop_polygon.half_depth = 1.0f;
            params.crop_polygon.inverse = inverse;
            params.width = width;
            params.height = height;
            params.channels = 3;
            params.focal_y = height / 2.0f;
            params.voxel_size = 0.05f;
            params.scaling_modifier = 1.0f;
            params.far_plane = 100.0f;
            params.bg_a = 1.0f;
            params.image = image.ptr<float>();
            params.depth = depth.ptr<float>();
            params.stream = image.stream();
            EXPECT_EQ(lfs::rendering::pcraster::launchPointCloudRaster(params), cudaSuccess);
            EXPECT_EQ(cudaStreamSynchronize(params.stream), cudaSuccess);
            const auto rgb = image.cpu().to_vector();
            std::array<bool, 3> visible{};
            for (int channel = 0; channel < 3; ++channel)
                visible[channel] = std::any_of(rgb.begin() + channel * width * height,
                                               rgb.begin() + (channel + 1) * width * height,
                                               [](const float value) { return value > 0.5f; });
            return visible;
        };
        EXPECT_EQ(seen(false), (std::array<bool, 3>{true, false, false}));
        EXPECT_EQ(seen(true), (std::array<bool, 3>{false, true, true}));
    }

    TEST(PointCloudRasterTest, ClippedTargetMatchesFullRenderColumns) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        constexpr int width = 64;
        constexpr int height = 32;
        constexpr int origin_x = 40;
        std::vector<float> means;
        std::vector<float> rgb;
        for (int i = 0; i < 200; ++i) {
            means.insert(means.end(), {static_cast<float>(i % 20) * 0.2f - 2.0f,
                                       static_cast<float>(i / 20) * 0.2f - 1.0f,
                                       -3.0f - static_cast<float>(i % 7) * 0.1f});
            rgb.insert(rgb.end(), {static_cast<float>(i % 5) / 4.0f, static_cast<float>(i % 3) / 2.0f, 0.5f});
        }
        auto positions = Tensor::from_vector(means, {200, 3}, Device::CUDA);
        auto colors = Tensor::from_vector(rgb, {200, 3}, Device::CUDA);
        const glm::mat4 view(1.0f);
        const auto projection = lfs::rendering::createProjectionMatrix(
            {width, height}, 60.0f, false, 1.0f, 0.1f, 100.0f);
        const auto render = [&](const int target_width, const int origin) {
            auto image = Tensor::empty({3, height, static_cast<size_t>(target_width)}, Device::CUDA);
            auto depth = Tensor::empty({1, height, static_cast<size_t>(target_width)}, Device::CUDA);
            lfs::rendering::pcraster::LaunchParams params{};
            params.positions = positions.ptr<float>();
            params.colors = colors.ptr<float>();
            params.n_points = 200;
            std::copy_n(glm::value_ptr(view), 16, params.view);
            std::copy_n(glm::value_ptr(projection), 16, params.view_proj);
            params.width = target_width;
            params.height = height;
            params.camera_width = width;
            params.camera_height = height;
            params.origin_x = origin;
            params.channels = 3;
            params.focal_y = height / 2.0f;
            params.voxel_size = 0.05f;
            params.scaling_modifier = 1.0f;
            params.far_plane = 100.0f;
            params.bg_a = 1.0f;
            params.image = image.ptr<float>();
            params.depth = depth.ptr<float>();
            params.stream = image.stream();
            EXPECT_EQ(lfs::rendering::pcraster::launchPointCloudRaster(params), cudaSuccess);
            EXPECT_EQ(cudaStreamSynchronize(params.stream), cudaSuccess);
            return image.cpu().to_vector();
        };
        const auto full = render(width, 0);
        const auto clipped = render(width - origin_x, origin_x);
        int covered = 0;
        for (int channel = 0; channel < 3; ++channel) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width - origin_x; ++x) {
                    const float expected = full[(channel * height + y) * width + origin_x + x];
                    ASSERT_EQ(clipped[(channel * height + y) * (width - origin_x) + x], expected)
                        << "channel " << channel << " x " << x << " y " << y;
                    covered += channel == 2 && expected > 0.0f;
                }
            }
        }
        EXPECT_GT(covered, 0);
    }

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
