/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "rendering/cuda_vulkan_interop.hpp"
#include "rendering/vksplat_viewport_renderer.hpp"
#include "rendering/vulkan_external_tensor.hpp"
#include "window/vulkan_context.hpp"
#include <SDL3/SDL.h>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

TEST(HiGSExactDepth, MedianSurvivesHalfPrecisionBatchBoundary) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
        GTEST_SKIP() << SDL_GetError();
    struct VideoGuard {
        ~VideoGuard() { SDL_QuitSubSystem(SDL_INIT_VIDEO); }
    } video_guard;
    std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window(
        SDL_CreateWindow("HiGS exact depth regression", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN),
        SDL_DestroyWindow);
    ASSERT_TRUE(window) << SDL_GetError();
    lfs::vis::VulkanContext context;
    ASSERT_TRUE(context.init(window.get(), 64, 64)) << context.lastError();
    lfs::rendering::setExpectedVulkanDeviceUuid(context.deviceUUID());
    using lfs::core::Device;
    using lfs::core::Tensor;
    constexpr size_t count = 1025;
    for (const bool overlays : {false, true}) {
        SCOPED_TRACE(overlays);
        for (const glm::ivec2 size : {glm::ivec2(64, 64), glm::ivec2(65, 67)}) {
            SCOPED_TRACE(::testing::Message() << size.x << 'x' << size.y);
            lfs::rendering::ViewportRenderRequest request;
            request.frame_view.size = size;
            request.sh_degree = 0;
            request.overlay.cursor = {.enabled = overlays, .cursor = {-1000.0f, -1000.0f}, .radius = 1.0f};
            request.require_exact_depth = true;
            const auto k = request.frame_view.getCameraIntrinsics();
            std::vector<float> means(count * 3), rotations(count * 4, 0.0f), opacity(count);
            for (size_t i = 0; i < count; ++i) {
                // Only the first and last sources cover the queried pixel. The
                // first 1024-source batch leaves exact T=.50005, rounded to .5
                // in the FP16 color partial. The next batch must set depth to 6.
                const bool filler = i > 0 && i + 1 < count;
                const float z = i == 0 ? 4.0f : filler ? 4.1f
                                                       : 6.0f;
                const float pixel_x = static_cast<float>(size.x / 2) + (filler ? 16.5f : 0.5f);
                const float pixel_y = static_cast<float>(size.y / 2) + 0.5f;
                means[i * 3] = (pixel_x - k.center_x) * z / k.focal_x;
                means[i * 3 + 1] = (k.center_y - pixel_y) * z / k.focal_y;
                means[i * 3 + 2] = -z;
                rotations[i * 4] = 1.0f;
                const float alpha = i == 0 ? 0.49995f : 0.01f;
                opacity[i] = std::log(alpha / (1.0f - alpha));
            }
            const auto external = [&](Tensor values) {
                auto allocation = lfs::vis::makeVulkanExternalTensor(context, values.shape(), values.dtype(), values.numel(), "depth-fixture");
                if (!allocation)
                    throw std::runtime_error(allocation.error());
                allocation->copy_from(values);
                return std::move(*allocation);
            };
            lfs::core::SplatData model(
                0,
                external(Tensor::from_vector(means, {count, size_t{3}}, Device::CPU)),
                external(Tensor::zeros({count, size_t{1}, size_t{3}}, Device::CPU)),
                Tensor::zeros({count, size_t{0}, size_t{3}}, Device::CUDA),
                external(Tensor::full({count, size_t{3}}, std::log(0.005f), Device::CPU)),
                external(Tensor::from_vector(rotations, {count, size_t{4}}, Device::CPU)),
                external(Tensor::from_vector(opacity, {count, size_t{1}}, Device::CPU)), 1.0f);
            lfs::vis::VksplatViewportRenderer renderer;
            request.require_exact_depth = false;
            for (int frame = 0; frame < 2; ++frame) {
                auto rendered = renderer.render(context, model, request, false,
                                                lfs::vis::VksplatViewportRenderer::OutputSlot::Main, false, true);
                ASSERT_TRUE(rendered) << rendered.error();
            }
            auto approximate = renderer.sampleDepthAtPixel(context, {.pixel = size / 2, .source_size = size});
            ASSERT_TRUE(approximate) << approximate.error();
            EXPECT_NEAR(*approximate, 4.0f, 1e-4f); // Confirms the HiGS approximate path is active.
            auto color = renderer.readOutputImageRgb8(context, lfs::vis::VksplatViewportRenderer::OutputSlot::Main);
            ASSERT_TRUE(color) << color.error();
            const auto* data = (*color)->ptr<uint8_t>();
            const std::vector<uint8_t> reference(data, data + (*color)->numel());
            request.require_exact_depth = true;
            for (const bool masked : {false, true}) {
                SCOPED_TRACE(masked);
                if (masked) {
                    const size_t columns = (size.x + 3) / 4;
                    const size_t tiles = columns * ((size.y + 3) / 4);
                    request.exact_depth_sample_mask.assign((tiles + 31) / 32, 0);
                    const size_t bit = size_t((size.y / 2) / 4) * columns + (size.x / 2) / 4;
                    request.exact_depth_sample_mask[bit / 32] |= 1u << (bit % 32);
                }
                // Deterministic export skips the cold legacy warmup but still uses
                // HiGS; depth capture would bypass HiGS and hide this regression.
                for (int frame = 0; frame < 2; ++frame) {
                    auto rendered = renderer.render(context, model, request, false,
                                                    lfs::vis::VksplatViewportRenderer::OutputSlot::Main, false, true);
                    ASSERT_TRUE(rendered) << rendered.error();
                    auto depth = renderer.sampleDepthAtPixel(context, {.pixel = size / 2, .source_size = size});
                    ASSERT_TRUE(depth) << depth.error();
                    EXPECT_NEAR(*depth, 6.0f, 1e-4f);
                    auto exact_color = renderer.readOutputImageRgb8(context, lfs::vis::VksplatViewportRenderer::OutputSlot::Main);
                    ASSERT_TRUE(exact_color) << exact_color.error();
                    const auto* pixels = (*exact_color)->ptr<uint8_t>();
                    EXPECT_EQ(std::vector<uint8_t>(pixels, pixels + (*exact_color)->numel()), reference);
                }
            }
        }
    }
}
