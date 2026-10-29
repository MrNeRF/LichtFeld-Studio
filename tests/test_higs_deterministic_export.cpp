/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/scene_renderer_factory.hpp"
#include "rendering/vksplat_viewport_renderer.hpp"
#include "rendering/vulkan_scene_renderer_factory.hpp"
#include "scene/scene_render_state.hpp"
#include "screen/screen_service.hpp"
#include "vulkan_external_tensor_test.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_graphics_context.hpp"
#include <SDL3/SDL.h>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace lfs::vis {
    // Observe real dispatches without adding counters to the production frame.
    struct HiGSExportTestAccess {
        inline static thread_local VksplatViewportRenderer* observed = nullptr;
        inline static thread_local size_t stable_binds = 0;
        inline static thread_local size_t macro_binds = 0;

        static VKAPI_ATTR void VKAPI_CALL bind(VkCommandBuffer command,
                                               VkPipelineBindPoint point, VkPipeline pipeline) {
            if (observed) {
                const auto& renderer = observed->renderer_;
                stable_binds += pipeline == renderer.pipeline_prepare_stable_depth_sort.pipeline;
                macro_binds += pipeline == renderer.pipeline_macro_coverage.pipeline;
            }
            vkCmdBindPipeline(command, point, pipeline);
        }
        static VksplatViewportRenderer& install(RenderingManager& manager, GraphicsContext& graphics) {
            manager.last_graphics_context_ = &graphics;
            manager.scene_renderer_ = createSceneRenderer(graphics);
            observed = vksplatRendererOf(*manager.scene_renderer_);
            if (!observed)
                throw std::runtime_error("the scene renderer is not a VkSplat renderer");
            auto dispatch = rendering::VulkanDispatch::real();
            dispatch.cmd_bind_pipeline = &bind;
            observed->renderer_.setVulkanDispatch(dispatch);
            return *observed;
        }
        static void reset() { stable_binds = macro_binds = 0; }
        // A target of its own, apart from the manager's preview target.
        static RenderTargetId target(RenderingManager& manager) { return manager.render_targets_.allocate(); }
    };
} // namespace lfs::vis

TEST(HiGSDeterministicExport, FloatOutputIsRepeatableAndViewportSkipsPresort) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
        GTEST_SKIP() << SDL_GetError();
    struct VideoGuard {
        ~VideoGuard() { SDL_QuitSubSystem(SDL_INIT_VIDEO); }
    } video_guard;
    std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window(
        SDL_CreateWindow("HiGS export regression", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN),
        SDL_DestroyWindow);
    ASSERT_TRUE(window) << SDL_GetError();
    lfs::vis::VulkanGraphicsContext graphics;
    ASSERT_TRUE(graphics.initialize(window.get(), 64, 64)) << graphics.lastError();
    auto& context = graphics.vulkanContext();

    using lfs::core::Device;
    using lfs::core::Tensor;
    using Access = lfs::vis::HiGSExportTestAccess;
    // Many workgroups and multiple macro batches, with identical radial keys.
    // Different colours make tie order observable in alpha compositing.
    constexpr size_t count = 8193;
    std::vector<float> means(count * 3, 0.0f), rotations(count * 4, 0.0f), sh0(count * 3);
    for (size_t i = 0; i < count; ++i) {
        means[3 * i + 2] = -4.0f;
        rotations[4 * i] = 1.0f;
        for (size_t c = 0; c < 3; ++c)
            sh0[3 * i + c] = ((i / 64) % 3 == c ? 0.45f : -0.45f) / 0.28209479177387814f;
    }
    const auto external = [&](Tensor values) {
        auto allocation = lfs::test::makeVulkanExternalTensor(
            context, values.shape(), values.dtype(), values.numel(), "export-fixture");
        if (!allocation)
            throw std::runtime_error(allocation.error());
        allocation->copy_from(values);
        return std::move(*allocation);
    };
    lfs::core::SplatData model(
        0,
        external(Tensor::from_vector(means, {count, size_t{3}}, Device::CPU)),
        external(Tensor::from_vector(sh0, {count, size_t{1}, size_t{3}}, Device::CPU)),
        Tensor::zeros({count, size_t{0}, size_t{3}}, Device::GPU),
        external(Tensor::full({count, size_t{3}}, std::log(0.1f), Device::CPU)),
        external(Tensor::from_vector(rotations, {count, size_t{4}}, Device::CPU)),
        external(Tensor::full({count, size_t{1}}, std::log(0.01f / 0.99f), Device::CPU)), 1.0f);

    std::vector<float> reference;
    // Fresh managers also check cold initialization, independent of prior frames.
    for (int session = 0; session < 2; ++session) {
        lfs::vis::screen::ScreenService views;
        lfs::vis::RenderingManager manager(views);
        auto& renderer = Access::install(manager, graphics);
        const auto live_target = Access::target(manager);
        struct ObserverGuard {
            ~ObserverGuard() { Access::observed = nullptr; }
        } observer_guard;
        auto settings = manager.getSettings();
        settings.sh_degree = 0;
        // Output must stay alpha-blended even if the viewer uses SteadySplats.
        settings.transparency_mode = lfs::rendering::GaussianTransparencyMode::SteadySplats;
        manager.updateSettings(settings);
        for (int frame = 0; frame < 2; ++frame) {
            Access::reset();
            auto image = manager.renderPreviewImage(
                model, {.combined_model = &model}, glm::mat3(1.0f), glm::vec3(0.0f),
                35.0f, 64, 64);
            ASSERT_TRUE(image && image->is_valid());
            EXPECT_EQ(Access::stable_binds, 2u); // prepare model IDs, restore depths
            ASSERT_GT(Access::macro_binds, 0u);  // Cannot pass via the legacy chain.
            const auto pixels = image->cpu().contiguous();
            ASSERT_EQ(pixels.dtype(), lfs::core::DataType::Float32);
            const auto* data = pixels.ptr<float>();
            if (reference.empty()) {
                reference.assign(data, data + pixels.numel());
                EXPECT_GT(reference[(32 * 64 + 32) * 3], 0.0f);
            } else {
                ASSERT_EQ(pixels.numel(), reference.size());
                EXPECT_EQ(std::memcmp(reference.data(), data, reference.size() * sizeof(float)), 0);
            }
        }
        // Warmed main viewport uses HiGS but pays for no deterministic pre-sort.
        lfs::rendering::ViewportRenderRequest live;
        live.frame_view.size = {64, 64};
        live.sh_degree = 0;
        EXPECT_FALSE(live.deterministic_export);
        Access::reset();
        for (int frame = 0; frame < 2; ++frame) {
            Access::reset();
            auto rendered = renderer.render(context, model, live, false, live_target);
            ASSERT_TRUE(rendered) << rendered.error();
            if (frame != 0)
                EXPECT_GT(Access::macro_binds, 0u);
            EXPECT_EQ(Access::stable_binds, 0u);
        }
        // Both stochastic settings must produce deterministic alpha output when
        // explicitly captured, independent of the interactive frame sequence.
        live.deterministic_export = true;
        std::vector<float> captured_reference;
        for (const auto mode : {lfs::rendering::GaussianTransparencyMode::Stochastic,
                                lfs::rendering::GaussianTransparencyMode::SteadySplats}) {
            live.transparency_mode = mode;
            Access::reset();
            auto captured = renderer.render(context, model, live, false, live_target);
            ASSERT_TRUE(captured) << captured.error();
            EXPECT_FALSE(captured->stochastic_accumulation_active);
            EXPECT_EQ(Access::stable_binds, 2u);
            EXPECT_GT(Access::macro_binds, 0u);
            auto image = renderer.readOutputImage(context, live_target);
            ASSERT_TRUE(image) << image.error();
            const auto* data = (*image)->ptr<float>();
            if (captured_reference.empty())
                captured_reference.assign(data, data + (*image)->numel());
            else {
                ASSERT_EQ((*image)->numel(), captured_reference.size());
                EXPECT_EQ(std::memcmp(captured_reference.data(), data,
                                      captured_reference.size() * sizeof(float)),
                          0);
            }
        }
        // The interactive preview opt-out follows the same dispatch contract.
        Access::reset();
        auto preview = manager.renderPreviewImage(
            model, {.combined_model = &model}, glm::mat3(1.0f), glm::vec3(0.0f),
            35.0f, 64, 64, std::nullopt, std::nullopt, std::nullopt, false, false);
        ASSERT_TRUE(preview && preview->is_valid());
        EXPECT_GT(Access::macro_binds, 0u);
        EXPECT_EQ(Access::stable_binds, 0u);
    }
}
