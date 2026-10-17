/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/coordinate_conventions.hpp"
#include "rendering/passes/vulkan_viewport_pass.hpp"
#include "rendering/render_constants.hpp"
#include "window/vulkan_context.hpp"

#include <SDL3/SDL.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {
    using lfs::vis::VulkanContext;

    class ViewportGridTest : public ::testing::Test {
    protected:
        SDL_Window* window = nullptr;
        std::unique_ptr<VulkanContext> context;
        std::unique_ptr<lfs::vis::VulkanViewportPass> pass;

        void SetUp() override {
            if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
                GTEST_SKIP() << SDL_GetError();
            window = SDL_CreateWindow("Viewport grid test", 320, 240, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
            if (!window)
                GTEST_SKIP() << SDL_GetError();
            context = std::make_unique<VulkanContext>();
            if (!context->init(window, 320, 240))
                GTEST_SKIP() << context->lastError();
            pass = std::make_unique<lfs::vis::VulkanViewportPass>();
            ASSERT_TRUE(pass->init(*context));
        }

        void TearDown() override {
            if (context)
                static_cast<void>(context->waitForSubmittedFrames());
            pass.reset();
            context.reset();
            if (window)
                SDL_DestroyWindow(window);
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }

        // Draws only the XZ grid, seen from `eye` looking at `target` while orbiting `pivot`.
        // A positive `ortho_scale` (pixels per world unit) switches to an orthographic view.
        VulkanContext::WindowCapture render(const glm::vec3& eye, const glm::vec3& target, const glm::vec3& pivot,
                                            const float ortho_scale = 0.0f, const float focal_mm = 35.0f) {
            VulkanContext::Frame frame{};
            const VkClearValue clear{};
            EXPECT_TRUE(context->beginFrame(clear, frame)) << context->lastError();
            const glm::ivec2 size{static_cast<int>(frame.extent.width), static_cast<int>(frame.extent.height)};
            lfs::vis::VulkanViewportGridOverlay grid{};
            grid.viewport_size = glm::vec2(size);
            grid.render_size = size;
            grid.view = lfs::rendering::makeViewMatrix(lfs::rendering::makeVisualizerLookAtRotation(eye, target), eye);
            grid.orthographic = ortho_scale > 0.0f;
            grid.projection = lfs::rendering::createProjectionMatrixFromFocal(size, focal_mm, grid.orthographic,
                                                                              grid.orthographic ? ortho_scale : 1.0f);
            grid.view_projection = grid.projection * grid.view;
            grid.view_position = eye;
            grid.pivot = pivot;
            grid.plane = 1;
            grid.opacity = 1.0f;
            lfs::vis::VulkanViewportPassParams params;
            params.frame_slot = frame.frame_slot;
            params.viewport_size = glm::vec2(size);
            params.grid_enabled = true;
            params.grid_overlays = {grid};
            pass->prepare(*context, params);
            pass->record(frame.command_buffer, frame.extent, params);
            auto capture = context->captureAndEndActiveFrameRgba();
            EXPECT_TRUE(capture.has_value()) << (capture ? "" : capture.error());
            return capture ? std::move(*capture) : VulkanContext::WindowCapture{};
        }
    };

    const std::uint8_t* pixel(const VulkanContext::WindowCapture& image, const int x, const int y) {
        return image.rgba.data() + (static_cast<size_t>(y) * image.width + x) * 4;
    }

    // Summed brightness in a box given in fractions of the image, skipping the centre row and column.
    long energy(const VulkanContext::WindowCapture& image, const float x0, const float y0, const float x1, const float y1) {
        long sum = 0;
        for (int y = static_cast<int>(y0 * image.height); y < static_cast<int>(y1 * image.height); ++y) {
            for (int x = static_cast<int>(x0 * image.width); x < static_cast<int>(x1 * image.width); ++x) {
                if (std::abs(x - image.width / 2) < 5 || std::abs(y - image.height / 2) < 5)
                    continue;
                const auto* p = pixel(image, x, y);
                sum += (p[0] + p[1] + p[2]) / 3;
            }
        }
        return sum;
    }

    bool isRed(const std::uint8_t* p, const int min_red) {
        return p[0] > min_red && p[0] > 2 * p[1] && p[0] > 2 * p[2];
    }

    // Median height of the red X axis over the columns it crosses; 0 when it is not drawn.
    int redAxisThickness(const VulkanContext::WindowCapture& image) {
        std::vector<int> red_per_column;
        for (int x = 0; x < image.width; ++x) {
            int red = 0;
            for (int y = 0; y < image.height; ++y)
                red += isRed(pixel(image, x, y), 60);
            if (red > 0)
                red_per_column.push_back(red);
        }
        if (red_per_column.size() <= static_cast<size_t>(image.width / 2))
            return 0;
        std::ranges::nth_element(red_per_column, red_per_column.begin() + red_per_column.size() / 2);
        return red_per_column[red_per_column.size() / 2];
    }

    // RMS distance in pixels between the red X axis and a straight line fitted through it.
    double redAxisWobble(const VulkanContext::WindowCapture& image) {
        std::vector<glm::dvec2> centres;
        for (int x = 0; x < image.width; ++x) {
            double sum = 0.0;
            int count = 0;
            for (int y = 0; y < image.height; ++y)
                if (isRed(pixel(image, x, y), 60)) {
                    sum += y;
                    ++count;
                }
            if (count > 0)
                centres.emplace_back(x, sum / count);
        }
        if (centres.size() <= static_cast<size_t>(image.width / 2))
            return 1e9;
        glm::dvec2 mean{0.0};
        for (const auto& c : centres)
            mean += c / static_cast<double>(centres.size());
        double sxx = 0.0, sxy = 0.0;
        for (const auto& c : centres) {
            sxx += (c.x - mean.x) * (c.x - mean.x);
            sxy += (c.x - mean.x) * (c.y - mean.y);
        }
        const double slope = sxy / sxx;
        double squared = 0.0;
        for (const auto& c : centres) {
            const double residual = c.y - (mean.y + slope * (c.x - mean.x));
            squared += residual * residual;
        }
        return std::sqrt(squared / static_cast<double>(centres.size()));
    }

    // Catches pixel rays rebuilt as the difference of two points near a distant camera: the float
    // difference loses the direction and the axis breaks into a staircase of offset segments.
    TEST_F(ViewportGridTest, AxisStaysStraightFarFromTheOrigin) {
        const auto image = render({30000.0f, 9000.0f, 30000.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 0.0f, 400.0f);
        ASSERT_GT(image.width, 0);
        EXPECT_LT(redAxisWobble(image), 0.5);
    }

    TEST_F(ViewportGridTest, AxesStayThinCloseToTheOrigin) {
        const int thickness = redAxisThickness(render({0.2f, 0.15f, 0.2f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}));
        EXPECT_GT(thickness, 0);
        EXPECT_LE(thickness, 5);
    }

    TEST_F(ViewportGridTest, GridStaysVisibleFarFromTheOrigin) {
        const auto image = render({700.0f, 450.0f, 700.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f});
        ASSERT_GT(image.width, 0);
        EXPECT_GT(energy(image, 0.25f, 0.25f, 0.75f, 0.75f), 5000);
    }

    TEST_F(ViewportGridTest, GridFadesAwayFromThePivot) {
        // Looking straight down at the origin with the pivot off to +X: the two halves of the
        // view mirror each other, so only the floor around the pivot can tell them apart.
        const auto image = render({0.0f, 12.0f, 0.01f}, {0.0f, 0.0f, 0.0f}, {20.0f, 0.0f, 0.0f});
        ASSERT_GT(image.width, 0);
        const long toward_pivot = energy(image, 0.5f, 0.0f, 1.0f, 1.0f);
        const long away_from_pivot = energy(image, 0.0f, 0.0f, 0.5f, 1.0f);
        ASSERT_GT(toward_pivot, 0);
        EXPECT_LT(away_from_pivot * 2, toward_pivot);
    }

    TEST_F(ViewportGridTest, OrthographicViewsKeepTheGridAtAnyZoomAndAngle) {
        const glm::vec3 origin{0.0f, 0.0f, 0.0f};
        // Zoomed far in (3000 pixels per unit), the axes stay a few pixels wide.
        const int thickness = redAxisThickness(render({6.0f, 4.0f, 6.0f}, origin, origin, 3000.0f));
        EXPECT_GT(thickness, 0);
        EXPECT_LE(thickness, 5);
        // Zoomed far out (half a pixel per unit), the grid is still drawn.
        EXPECT_GT(energy(render({6.0f, 4.0f, 6.0f}, origin, origin, 0.5f), 0.25f, 0.25f, 0.75f, 0.75f), 5000);
        // Parallel rays have no horizon: a view 4 degrees above the plane keeps the axes at full strength.
        const auto low = render({7.05f, 0.7f, 7.05f}, origin, origin, 40.0f);
        int strongest_red = 0;
        for (int y = 0; y < low.height; ++y)
            for (int x = 0; x < low.width; ++x)
                if (const auto* p = pixel(low, x, y); isRed(p, 0))
                    strongest_red = std::max(strongest_red, static_cast<int>(p[0]));
        EXPECT_GT(strongest_red, 180);
    }
} // namespace
