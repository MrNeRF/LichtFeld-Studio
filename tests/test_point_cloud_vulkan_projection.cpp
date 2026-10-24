/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "rendering/point_cloud_vulkan_renderer.hpp"

#include <SDL3/SDL.h>
#include <array>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {
    using Renderer = lfs::vis::PointCloudVulkanRenderer;
    using lfs::core::Device;
    using lfs::core::Tensor;

    class PointCloudVulkanProjectionTest : public ::testing::Test {
    protected:
        SDL_Window* window = nullptr;
        std::unique_ptr<lfs::vis::VulkanContext> context;
        std::unique_ptr<Renderer> renderer;
        Tensor positions;
        Tensor colors;
        Renderer::RenderRequest request;

        void SetUp() override {
            if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
                GTEST_SKIP() << SDL_GetError();
            window = SDL_CreateWindow("Point projection test", 256, 128, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
            if (!window)
                GTEST_SKIP() << SDL_GetError();
            context = std::make_unique<lfs::vis::VulkanContext>();
            if (!context->init(window, 256, 128))
                GTEST_SKIP() << context->lastError();
            renderer = std::make_unique<Renderer>();
            request.size = {256, 128};
            request.focal_y = 64.0f;
            request.voxel_size = 0.05f;
            request.ortho_scale = 8.0f;
            projection(false);
        }

        void TearDown() override {
            renderer.reset();
            context.reset();
            if (window)
                SDL_DestroyWindow(window);
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }

        void projection(bool orthographic) {
            request.orthographic = orthographic;
            glm::mat4 flip(1.0f);
            flip[1][1] = -1.0f;
            request.view_projection = flip * (orthographic ? glm::ortho(-8.0f, 8.0f, -4.0f, 4.0f, 0.1f, 100.0f) : glm::perspective(glm::radians(90.0f), 2.0f, 0.1f, 100.0f)) *
                                      request.view;
        }

        void points(const std::vector<float>& xyz, const std::vector<float>& rgb) {
            positions = Tensor::from_vector(xyz, {xyz.size() / 3, std::size_t{3}}, Device::CPU);
            colors = Tensor::from_vector(rgb, {rgb.size() / 3, std::size_t{3}}, Device::CPU);
            request.positions = &positions;
            request.colors = &colors;
            ++request.positions_revision;
            ++request.colors_revision;
        }

        void expectDepth(glm::ivec2 pixel, float expected) {
            Renderer::DepthSampleRequest sample{.pixel = pixel, .source_size = request.size};
            const auto cold = renderer->sampleDepthAtPixel(*context, sample);
            ASSERT_TRUE(cold);
            EXPECT_TRUE(std::isfinite(*cold));
            EXPECT_NEAR(*cold, expected, 2e-4f);
            sample.nonblocking = true;
            auto cached = renderer->sampleDepthAtPixel(*context, sample);
            ASSERT_TRUE(cached);
            if (*cached == Renderer::kDepthSamplePending) {
                ASSERT_TRUE(context->deviceWaitIdle());
                cached = renderer->sampleDepthAtPixel(*context, sample);
            }
            ASSERT_TRUE(cached);
            EXPECT_TRUE(std::isfinite(*cached));
            EXPECT_FLOAT_EQ(*cached, *cold);
        }

        void expectColor(const Tensor& image, glm::ivec2 pixel, glm::vec3 expected) {
            const auto* rgb = image.ptr<float>() + 3 * (pixel.y * request.size.x + pixel.x);
            for (int c = 0; c < 3; ++c) {
                EXPECT_TRUE(std::isfinite(rgb[c]));
                EXPECT_NEAR(rgb[c], expected[c], 1.0f / 255.0f);
            }
        }
    };

    TEST_F(PointCloudVulkanProjectionTest, PanoramaIncludesRearHemisphereAndRadialDepth) {
        points({0, 1, -3, 0, -1, -3, 3, 0, 0, -3, 0, 0, 3, 0, 3, -3, 0, 3},
               {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1});
        request.equirectangular = true;
        const std::array<glm::ivec2, 6> pixels{{{128, 50}, {128, 77}, {192, 64}, {64, 64}, {224, 64}, {32, 64}}};
        const std::array<glm::vec3, 6> rgb{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}}};
        const std::array<float, 6> depths{std::sqrt(10.0f), std::sqrt(10.0f), 3, 3, std::sqrt(18.0f), std::sqrt(18.0f)};
        for (const bool orthographic : {false, true}) {
            SCOPED_TRACE(orthographic);
            projection(orthographic);
            ASSERT_TRUE(renderer->render(*context, request));
            const auto image = renderer->readOutputImage(*context);
            ASSERT_TRUE(image);
            for (std::size_t i = 0; i < pixels.size(); ++i) {
                SCOPED_TRACE(i);
                expectColor(**image, pixels[i], rgb[i]);
                expectDepth(pixels[i], depths[i]);
            }
        }
    }

    TEST_F(PointCloudVulkanProjectionTest, PanoramaOcclusionAndDepthModesUseDistance) {
        // The nearer rear point must win, just as a nearer front point does.
        points({3, 0, 3, 6, 0, 6, 0, 0, -3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
        request.equirectangular = true;
        request.depth_view_min = 1.0f;
        request.depth_view_max = 10.0f;
        for (int mode = 0; mode < 3; ++mode) {
            SCOPED_TRACE(mode);
            request.depth_view = mode > 0;
            request.depth_visualization_mode = mode == 2 ? lfs::rendering::DepthVisualizationMode::Grayscale
                                                         : lfs::rendering::DepthVisualizationMode::Palette;
            ASSERT_TRUE(renderer->render(*context, request));
            const auto image = renderer->readOutputImage(*context);
            ASSERT_TRUE(image);
            const auto* rgb = (*image)->ptr<float>() + 3 * (64 * 256 + 224);
            EXPECT_GT(rgb[0] + rgb[1] + rgb[2], 0.05f);
            if (mode == 0)
                expectColor(**image, {224, 64}, {1, 0, 0});
            if (mode == 2) {
                EXPECT_FLOAT_EQ(rgb[0], rgb[1]);
                EXPECT_FLOAT_EQ(rgb[1], rgb[2]);
            }
            expectDepth({224, 64}, std::sqrt(18.0f));
            expectDepth({128, 64}, 3.0f);
        }
    }

    // Catches a live point-cloud crop polygon that the Vulkan pass ignores or crops by its bounds: of an L outline the
    // points in its arms stay; the point in its notch and the point past its slab go, and inverse swaps them.
    TEST_F(PointCloudVulkanProjectionTest, CropPolygonKeepsOnlyPointsInsideThePrism) {
        // Two arm points at z = -4, the notch point, and a point over the upper arm but past the slab.
        points({0.5f, -0.5f, -4.0f, -0.5f, 0.5f, -4.0f, 0.5f, 0.5f, -4.0f, -0.5f, -0.5f, -7.0f},
               {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0});
        projection(true);
        const std::vector<glm::vec2> outline{{-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 1.0f}};
        const std::array<glm::ivec2, 4> pixels{{{136, 72}, {120, 56}, {136, 56}, {120, 72}}};
        const std::array<glm::vec3, 4> rgb{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}}};
        for (const bool inverse : {false, true}) {
            SCOPED_TRACE(inverse ? "inverse" : "normal");
            request.crop_polygon = Renderer::CropPolygon{
                .to_local = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 4.0f)),
                .vertices = outline,
                .half_depth = 1.0f,
                .inverse = inverse};
            ASSERT_TRUE(renderer->render(*context, request));
            const auto image = renderer->readOutputImage(*context);
            ASSERT_TRUE(image);
            for (std::size_t i = 0; i < pixels.size(); ++i) {
                const bool kept = (i < 2) != inverse;
                expectColor(**image, pixels[i], kept ? rgb[i] : glm::vec3(0.0f));
            }
        }
    }

    TEST_F(PointCloudVulkanProjectionTest, PerspectiveAndOrthographicKeepColorAndDepth) {
        points({0, 1, -4, 0, -1, -4, 1, 0, -4, -1, 0, -4, 3, 0, 3},
               {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1});
        const std::array<glm::ivec2, 4> pixels{{{128, 48}, {128, 80}, {144, 64}, {112, 64}}};
        const std::array<glm::vec3, 4> rgb{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}}};
        for (const bool orthographic : {false, true}) {
            projection(orthographic);
            ASSERT_TRUE(renderer->render(*context, request));
            const auto image = renderer->readOutputImage(*context);
            ASSERT_TRUE(image);
            for (std::size_t i = 0; i < pixels.size(); ++i) {
                expectColor(**image, pixels[i], rgb[i]);
                expectDepth(pixels[i], 4.0f);
            }
            expectDepth({224, 64}, -1.0f);
        }
    }

    TEST_F(PointCloudVulkanProjectionTest, PanoramaUsesViewAndModelTransforms) {
        points({3, 0, 3}, {1, 0, 0});
        const auto model = glm::translate(glm::mat4(1.0f), glm::vec3(1, 2, 3));
        const std::vector<glm::mat4> transforms{model};
        request.model_transforms = &transforms;
        request.view = glm::inverse(model);
        projection(false);
        request.equirectangular = true;
        ASSERT_TRUE(renderer->render(*context, request));
        const auto image = renderer->readOutputImage(*context);
        ASSERT_TRUE(image);
        expectColor(**image, {224, 64}, {1, 0, 0});
        expectDepth({224, 64}, std::sqrt(18.0f));
    }

    TEST_F(PointCloudVulkanProjectionTest, PanoramaPolesAreFiniteAndCameraOriginIsHidden) {
        points({0, 5, 0, 0, -5, 0, 0, 0, 0}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
        request.equirectangular = true;
        ASSERT_TRUE(renderer->render(*context, request));
        const auto image = renderer->readOutputImage(*context);
        ASSERT_TRUE(image);
        expectColor(**image, {128, 0}, {1, 0, 0});
        expectColor(**image, {128, 127}, {0, 1, 0});
        expectDepth({128, 0}, 5.0f);
        expectDepth({128, 127}, 5.0f);
        expectColor(**image, {128, 64}, {0, 0, 0});
        expectDepth({128, 64}, -1.0f);
    }

    TEST_F(PointCloudVulkanProjectionTest, ProjectionToggleInvalidatesCachedDepthAndRestoresImage) {
        points({1, 0, -3, 3, 0, 3}, {1, 0, 0, 0, 1, 0});
        ASSERT_TRUE(renderer->render(*context, request));
        const auto before = renderer->readOutputImage(*context);
        ASSERT_TRUE(before);
        expectDepth({224, 64}, -1.0f);
        request.equirectangular = true;
        ASSERT_TRUE(renderer->render(*context, request));
        expectDepth({224, 64}, std::sqrt(18.0f));
        request.equirectangular = false;
        ASSERT_TRUE(renderer->render(*context, request));
        const auto after = renderer->readOutputImage(*context);
        ASSERT_TRUE(after);
        expectDepth({224, 64}, -1.0f);
        ASSERT_EQ((*before)->numel(), (*after)->numel());
        for (std::size_t i = 0; i < (*before)->numel(); ++i)
            ASSERT_FLOAT_EQ((*before)->ptr<float>()[i], (*after)->ptr<float>()[i]);
    }
} // namespace
