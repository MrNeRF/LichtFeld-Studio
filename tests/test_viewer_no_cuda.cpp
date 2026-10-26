/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/gpu_preflight.hpp"
#include "combined_model_worker_fixture.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/gpu_device_info.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_readback.hpp"
#include "rendering/selection_ops.hpp"
#if LFS_TENSOR_METAL && !defined(LFS_GRAPHICS_VULKAN)
#include "window/metal_graphics_context.hpp"
#include <SDL3/SDL.h>
#include <cstdlib>
#endif

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {

    using namespace lfs::core;
    using lfs::app::decide_gpu_preflight;
    using lfs::app::GpuPreflightDecision;

} // namespace

#if LFS_TENSOR_METAL && !defined(LFS_GRAPHICS_VULKAN)
TEST(ViewerNoCuda, MetalShutdownReleasesReservedFrameSlot) {
    if (!gpu_backend_available(GpuBackend::Metal))
        GTEST_SKIP() << "Metal backend unavailable";
    lfs::vis::MetalGraphicsContext graphics;
    ASSERT_TRUE(graphics.initializeHeadless());
    ASSERT_TRUE(graphics.waitForNextFrameSlot());
    graphics.shutdown();
    // Destroying a semaphore with an unreleased reservation traps in libdispatch.
}

TEST(ViewerNoCuda, MetalRenderPassPresentsAndResizesDrawable) {
    if (!std::getenv("LFS_TEST_METAL_PRESENTATION"))
        GTEST_SKIP() << "Set LFS_TEST_METAL_PRESENTATION=1 to exercise a real macOS drawable";
    if (!gpu_backend_available(GpuBackend::Metal))
        GTEST_SKIP() << "Metal backend unavailable";
    const bool owns_video = !SDL_WasInit(SDL_INIT_VIDEO);
    if (owns_video)
        ASSERT_TRUE(SDL_InitSubSystem(SDL_INIT_VIDEO));
    struct VideoCleanup {
        bool owns;
        ~VideoCleanup() {
            if (owns)
                SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }
    } cleanup{owns_video};
    std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window(
        SDL_CreateWindow("Metal presentation regression", 96, 64, SDL_WINDOW_METAL), SDL_DestroyWindow);
    ASSERT_NE(window, nullptr) << SDL_GetError();
    lfs::vis::MetalGraphicsContext graphics;
    ASSERT_TRUE(graphics.initialize(window.get(), 96, 64));
    for (const auto size : {std::array{96, 64}, std::array{128, 80}, std::array{64, 96}}) {
        ASSERT_TRUE(SDL_SetWindowSize(window.get(), size[0], size[1]));
        SDL_PumpEvents();
        graphics.notifyFramebufferResized(size[0], size[1], lfs::vis::GraphicsResizeIntent::Interactive);
        auto frame = graphics.beginFrame({0.2f, 0.4f, 0.8f, 1.0f});
        ASSERT_TRUE(frame.has_value());
        ASSERT_TRUE(frame->has_value());
        auto* image = graphics.finalImageTensor(**frame);
        ASSERT_NE(image, nullptr);
        std::vector<std::uint8_t> pixels(size[0] * size[1] * 4);
        for (int y = 0; y < size[1]; ++y) {
            for (int x = 0; x < size[0]; ++x) {
                const auto offset = (y * size[0] + x) * 4;
                pixels[offset] = x * 255 / (size[0] - 1);
                pixels[offset + 1] = y * 255 / (size[1] - 1);
                pixels[offset + 2] = 64;
                pixels[offset + 3] = 255;
            }
        }
        GpuBackendScope scope(GpuBackend::Metal);
        *image = Tensor::from_blob(pixels.data(), {static_cast<size_t>(size[1]), static_cast<size_t>(size[0]), 4},
                                   Device::CPU, DataType::UInt8)
                     .to(Device::GPU);
        ASSERT_TRUE(graphics.endFrame().has_value()) << graphics.lastError();
        ASSERT_TRUE(graphics.waitForSubmittedFrames()) << graphics.lastError();
    }
    graphics.shutdown();
}

#endif

TEST(ViewerNoCuda, PreflightViewerOnlyCudaUsableUsesCuda) {
    EXPECT_EQ(decide_gpu_preflight(true, true, true), GpuPreflightDecision::UseCuda);
    EXPECT_EQ(decide_gpu_preflight(true, true, false), GpuPreflightDecision::UseCuda);
}

TEST(ViewerNoCuda, HardwareDiscoveryDoesNotInitializeTensorContext) {
    const auto backend = configured_gpu_backend();
    const bool live = gpu_backend_live(GpuBackend::Vulkan);
    const auto hardware = enumerate_vulkan_hardware();
    EXPECT_EQ(configured_gpu_backend(), backend);
    EXPECT_EQ(gpu_backend_live(GpuBackend::Vulkan), live);
    for (const auto& device : hardware) {
        EXPECT_FALSE(device.name.empty());
        EXPECT_EQ(device.uuid.size(), 32);
    }
}

TEST(ViewerNoCuda, IterationTelemetryKeepsUnusedCudaCold) {
    if (configured_gpu_backend() == GpuBackend::CUDA)
        GTEST_SKIP() << "Requires a non-CUDA default backend";
    const bool live = gpu_backend_live(GpuBackend::CUDA);
    Tensor::set_memory_pool_iteration(17);
    EXPECT_EQ(gpu_backend_live(GpuBackend::CUDA), live);
}

TEST(ViewerNoCuda, PreflightViewerOnlyCudaUnusableUsesVulkanWhenAvailable) {
    EXPECT_EQ(decide_gpu_preflight(true, false, true), GpuPreflightDecision::UseVulkan);
}

TEST(ViewerNoCuda, PreflightTrainingCudaUsableUsesCuda) {
    EXPECT_EQ(decide_gpu_preflight(false, true, true), GpuPreflightDecision::UseCuda);
}

TEST(ViewerNoCuda, PreflightTrainingCudaUnusableUsesVulkan) {
    EXPECT_EQ(decide_gpu_preflight(false, false, true), GpuPreflightDecision::UseVulkan);
    EXPECT_EQ(decide_gpu_preflight(true, false, false), GpuPreflightDecision::Fatal);
}

TEST(ViewerNoCuda, PreflightNeverOverridesAnExplicitCudaRequest) {
    for (const bool viewer : {false, true}) {
        EXPECT_EQ(decide_gpu_preflight(viewer, false, true, true), GpuPreflightDecision::Fatal);
        EXPECT_EQ(decide_gpu_preflight(viewer, false, false, true), GpuPreflightDecision::Fatal);
        EXPECT_EQ(decide_gpu_preflight(viewer, true, true, true), GpuPreflightDecision::UseCuda);
    }
}

TEST(ViewerNoCuda, PreflightAutoDoesNotHideACudaHardwareOrDriverProblem) {
    for (const bool viewer : {false, true}) {
        EXPECT_EQ(decide_gpu_preflight(viewer, false, true, false, true), GpuPreflightDecision::Fatal);
        EXPECT_EQ(decide_gpu_preflight(viewer, false, false, false, true), GpuPreflightDecision::Fatal);
        EXPECT_EQ(decide_gpu_preflight(viewer, true, true, false, true), GpuPreflightDecision::UseCuda);
    }
}

TEST(ViewerNoCuda, HoverGroupCountMatchesCpu) {
    if (!gpu_backend_available(GpuBackend::Vulkan)) {
        GTEST_SKIP() << "Vulkan backend unavailable";
    }

    GpuBackendScope scope(GpuBackend::Vulkan);
    constexpr size_t n = 4096;
    std::mt19937 rng(20260906);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<std::uint8_t> host(n);
    std::array<size_t, 256> cpu_counts{};
    for (size_t i = 0; i < n; ++i) {
        host[i] = static_cast<std::uint8_t>(dist(rng));
        if (host[i] != 0) {
            ++cpu_counts[host[i]];
        }
    }

    Tensor cpu_mask = Tensor::empty({n}, Device::CPU, DataType::UInt8);
    std::memcpy(cpu_mask.ptr<std::uint8_t>(), host.data(), n);
    Tensor mask = cpu_mask.to(Device::GPU);
    ASSERT_TRUE(mask.is_valid());
    EXPECT_EQ(gpu_backend_of(mask), GpuBackend::Vulkan);

    Tensor scratch;
    lfs::rendering::count_selection_groups_async(mask, scratch);
    std::array<int, 256> host_counts{};
    TensorReadback readback;
    readback.enqueue(scratch);
    while (!readback.poll(std::as_writable_bytes(std::span(host_counts)))) {}
    for (size_t group = 0; group < 256; ++group) {
        EXPECT_EQ(static_cast<size_t>(host_counts[group]), cpu_counts[group]) << "group " << group;
    }
}

TEST(ViewerNoCuda, CameraPathRenderDoesNotRequireTrainerOrCuda) {
    lfs::core::param::TrainingParameters params;
    params.optimization.headless = true;
    EXPECT_FALSE(lfs::app::training_params_are_viewer_only(params));
    params.render_path.emplace();
    EXPECT_TRUE(lfs::app::training_params_are_viewer_only(params));
}

TEST(ViewerNoCuda, RevealingEarlierSplatKeepsCoherentAliasAndRequestsRedraw) {
    namespace combined = lfs::test::combined_model;
    Scene scene;
    combined::HeldWorker worker;
    constexpr size_t count = 500001;
    const auto make_model = [](float value) {
        return std::make_unique<SplatData>(
            0, Tensor::full({count, 3}, value, Device::GPU),
            Tensor::zeros({count, 1, 3}, Device::GPU), Tensor::zeros({count, 0, 3}, Device::GPU),
            Tensor::full({count, 3}, -2.0f, Device::GPU),
            Tensor::full({count, 4}, 0.5f, Device::GPU),
            Tensor::full({count, 1}, 1.0f, Device::GPU), 1.0f);
    };
    const auto first = scene.addSplat("first", make_model(0.0f));
    const auto second = scene.addSplat("second", make_model(1.0f));
    glm::mat4 first_transform(1.0f), second_transform(1.0f);
    first_transform[3] = glm::vec4(10.0f, 0.0f, 0.0f, 1.0f);
    second_transform[3] = glm::vec4(0.0f, 20.0f, 0.0f, 1.0f);
    scene.setNodeTransform(first, first_transform);
    scene.setNodeTransform(second, second_transform);
    scene.setNodeVisibility(first, false);
    const auto* alias = scene.getNodeById(second)->model.get();
    ASSERT_EQ(scene.getCombinedModel(), alias);
    ASSERT_NE(scene.getTransformIndices(), nullptr);
    EXPECT_EQ(scene.getVisibleNodeTransforms(), (std::vector<glm::mat4>{second_transform}));
    scene.setCombinedModelAllocator(worker.allocator());
    std::atomic<int> ready{0};
    lfs::event::ScopedHandler handler;
    handler.subscribe<events::state::CombinedModelBuildReady>([&](const auto& event) {
        if (event.scene == &scene)
            ++ready;
    });
    scene.setNodeVisibility(first, true);
    const auto retained = combined::combinedModelWithin(scene, worker);
    ASSERT_TRUE(retained.prompt);
    ASSERT_TRUE(worker.waitEntered());
    ASSERT_TRUE(scene.combinedModelBuildPending());
    ASSERT_EQ(retained.model, alias);
    // Slot zero must still mean second while its old geometry is displayed.
    const auto retained_indices = scene.getTransformIndices();
    ASSERT_NE(retained_indices, nullptr);
    EXPECT_EQ(retained_indices->to_vector_int(), std::vector<int>(count, 0));
    EXPECT_EQ(scene.getVisibleNodeTransforms(), (std::vector<glm::mat4>{second_transform}));
    const auto slots = scene.getCombinedSplatNodeSlots();
    ASSERT_EQ(slots.size(), 1);
    ASSERT_NE(slots[0].node, nullptr);
    EXPECT_EQ(slots[0].node->id, second);
    EXPECT_EQ(slots[0].slot_index, 0);

    worker.release();
    const auto* aggregate = combined::settledCombinedModel(scene);
    ASSERT_FALSE(scene.combinedModelBuildPending());
    ASSERT_NE(aggregate, nullptr);
    ASSERT_EQ(aggregate->size(), 2 * count);
    EXPECT_GE(ready.load(), 1);
    EXPECT_EQ(scene.getVisibleNodeTransforms(), (std::vector<glm::mat4>{first_transform, second_transform}));
    ASSERT_NE(scene.getTransformIndices(), nullptr);
    const auto indices = scene.getTransformIndices()->cpu();
    EXPECT_EQ(indices.ptr<int>()[0], 0);
    EXPECT_EQ(indices.ptr<int>()[count], 1);
}
