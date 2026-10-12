/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/cuda_vulkan_interop.hpp"
#include "rendering/passes/vulkan_viewport_pass.hpp"
#include "rendering/viewport_interop_service.hpp"
#include "window/vulkan_context.hpp"

#include <SDL3/SDL.h>
#include <gtest/gtest.h>

using lfs::vis::decideViewportInteropEarly;
using lfs::vis::ViewportInteropAction;
using lfs::vis::ViewportInteropDecision;
using lfs::vis::ViewportInteropSlotInputs;

namespace {

    ViewportInteropSlotInputs baseValidInputs() {
        ViewportInteropSlotInputs in{};
        in.source_ok = true;
        in.frame_slot_in_range = true;
        in.target_present = true;
        in.target_size_matches = true;       // bucket match
        in.target_valid_size_matches = true; // logical size match
        in.target_interop_valid = true;
        in.target_layout_read_only = true;
        in.source_generation = 7;
        in.uploaded_source_generation = 7;
        return in;
    }

} // namespace

TEST(ViewportInteropDecision, DisabledReturnsDisabled) {
    auto in = baseValidInputs();
    in.disabled = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::Disabled);
    EXPECT_FALSE(d.clear_published);
    EXPECT_FALSE(d.publish_from_target);
}

TEST(ViewportInteropDecision, SceneExternalHandleEarlyOut) {
    auto in = baseValidInputs();
    in.external_handle_early_out = true;
    in.has_external_scene_image = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::ExternalSkip);
}

TEST(ViewportInteropDecision, NonSceneIgnoresExternalHandle) {
    auto in = baseValidInputs();
    in.external_handle_early_out = false;
    in.has_external_scene_image = true;
    // Valid cache hit should still win for split/depth policy.
    in.publishes_published = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::CacheHit);
    EXPECT_TRUE(d.publish_from_target);
}

TEST(ViewportInteropDecision, InvalidSourceSceneDoesNotClearPublished) {
    auto in = baseValidInputs();
    in.source_ok = false;
    in.publishes_published = false;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::InvalidReset);
    EXPECT_FALSE(d.clear_published);
}

TEST(ViewportInteropDecision, InvalidSourceSplitClearsPublished) {
    auto in = baseValidInputs();
    in.source_ok = false;
    in.publishes_published = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::InvalidReset);
    EXPECT_TRUE(d.clear_published);
}

TEST(ViewportInteropDecision, CacheHitSceneDoesNotPublish) {
    auto in = baseValidInputs();
    in.publishes_published = false;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::CacheHit);
    EXPECT_FALSE(d.publish_from_target);
}

TEST(ViewportInteropDecision, CacheHitSplitPublishes) {
    auto in = baseValidInputs();
    in.publishes_published = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::CacheHit);
    EXPECT_TRUE(d.publish_from_target);
}

TEST(ViewportInteropDecision, CacheHitRequiresGenerationAndLayout) {
    {
        auto in = baseValidInputs();
        in.source_generation = 0;
        EXPECT_EQ(decideViewportInteropEarly(in).action, ViewportInteropAction::SlowPath);
    }
    {
        auto in = baseValidInputs();
        in.uploaded_source_generation = 3;
        EXPECT_EQ(decideViewportInteropEarly(in).action, ViewportInteropAction::SlowPath);
    }
    {
        auto in = baseValidInputs();
        in.target_layout_read_only = false;
        EXPECT_EQ(decideViewportInteropEarly(in).action, ViewportInteropAction::SlowPath);
    }
}

TEST(ViewportInteropDecision, DeferBailClearsPublishedForSplit) {
    auto in = baseValidInputs();
    in.publishes_published = true;
    in.target_size_matches = false; // bucket mismatch → recreate needed
    in.resize_deferring = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::DeferBail);
    EXPECT_TRUE(d.clear_published);
}

TEST(ViewportInteropDecision, DeferBailDoesNotClearForScene) {
    auto in = baseValidInputs();
    in.publishes_published = false;
    in.target_size_matches = false;
    in.resize_deferring = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::DeferBail);
    EXPECT_FALSE(d.clear_published);
}

TEST(ViewportInteropDecision, DeferBailWhenSlotArrayResizeNeeded) {
    auto in = baseValidInputs();
    in.slot_array_resize_needed = true;
    in.resize_deferring = true;
    in.publishes_published = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::DeferBail);
    EXPECT_TRUE(d.clear_published);
}

TEST(ViewportInteropDecision, SlowPathWhenContentChangedAndNotDeferring) {
    auto in = baseValidInputs();
    in.uploaded_source_generation = 1;
    in.source_generation = 2;
    in.resize_deferring = false;
    EXPECT_EQ(decideViewportInteropEarly(in).action, ViewportInteropAction::SlowPath);
}

TEST(ViewportInteropDecision, SlowPathWhenRecreateNeededAndNotDeferring) {
    auto in = baseValidInputs();
    in.target_present = false;
    in.resize_deferring = false;
    EXPECT_EQ(decideViewportInteropEarly(in).action, ViewportInteropAction::SlowPath);
}

// #1566: within-bucket valid-size change is re-upload (SlowPath), not recreate/DeferBail.
TEST(ViewportInteropDecision, WithinBucketValidSizeChangeIsSlowPathNotDeferBail) {
    auto in = baseValidInputs();
    in.target_size_matches = true;        // same 64-px bucket
    in.target_valid_size_matches = false; // logical size changed
    in.resize_deferring = true;
    in.publishes_published = true;
    // recreate_needed is false (bucket matches), so DeferBail does not apply.
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::SlowPath);
    EXPECT_FALSE(d.clear_published);
}

TEST(ViewportInteropDecision, WithinBucketValidSizeChangeWithoutDeferIsSlowPath) {
    auto in = baseValidInputs();
    in.target_size_matches = true;
    in.target_valid_size_matches = false;
    in.resize_deferring = false;
    EXPECT_EQ(decideViewportInteropEarly(in).action, ViewportInteropAction::SlowPath);
}

TEST(ViewportInteropDecision, CrossBucketMismatchWithDeferIsDeferBail) {
    auto in = baseValidInputs();
    in.target_size_matches = false; // crossed 64-px bucket
    in.target_valid_size_matches = false;
    in.resize_deferring = true;
    in.publishes_published = true;
    const auto d = decideViewportInteropEarly(in);
    EXPECT_EQ(d.action, ViewportInteropAction::DeferBail);
    EXPECT_TRUE(d.clear_published);
}

TEST(ViewportInteropDecision, CacheHitRequiresValidSizeMatch) {
    auto in = baseValidInputs();
    in.target_valid_size_matches = false;
    EXPECT_EQ(decideViewportInteropEarly(in).action, ViewportInteropAction::SlowPath);
}

namespace {
    class ViewportInteropQueuedUpload : public ::testing::Test {
    protected:
        void SetUp() override {
            owns_video = !SDL_WasInit(SDL_INIT_VIDEO);
            if (owns_video && !SDL_InitSubSystem(SDL_INIT_VIDEO))
                GTEST_SKIP() << "SDL video unavailable for Vulkan presentation context: " << SDL_GetError();
            window = SDL_CreateWindow("Viewport interop regression", 96, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
            if (!window)
                GTEST_SKIP() << "Vulkan presentation window unavailable: " << SDL_GetError();
            context = std::make_unique<lfs::vis::VulkanContext>();
            if (!context->init(window, 96, 64))
                GTEST_SKIP() << "Vulkan presentation context unavailable: " << context->lastError();
            lfs::rendering::setExpectedVulkanDeviceUuid(context->deviceUUID());
            interop = std::make_unique<lfs::vis::ViewportInteropService>();
            image = std::make_shared<lfs::core::Tensor>(lfs::core::Tensor::ones(
                {size_t{8}, size_t{8}, size_t{4}}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8));
        }

        void TearDown() override {
            interop.reset();
            image.reset();
            context.reset();
            if (window)
                SDL_DestroyWindow(window);
            if (owns_video)
                SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }

        void queue(std::uint64_t generation) {
            interop->setSceneImage(image, {8, 8}, false, generation);
            interop->prepareFrame(*context, false);
        }

        lfs::vis::VulkanViewportPassParams bind(bool export_locked = false) {
            lfs::vis::VulkanViewportPassParams params{};
            interop->bindViewportParams(params, context->currentFrameSlot(), export_locked, false);
            return params;
        }

        bool owns_video = false;
        SDL_Window* window = nullptr;
        std::unique_ptr<lfs::vis::VulkanContext> context;
        std::unique_ptr<lfs::vis::ViewportInteropService> interop;
        std::shared_ptr<lfs::core::Tensor> image;
    };
} // namespace

TEST_F(ViewportInteropQueuedUpload, NewContentBindsBeforeFrameRecording) {
    queue(7);
    const auto queued = bind();
    EXPECT_NE(queued.external_scene_image, VK_NULL_HANDLE);
    EXPECT_NE(queued.external_scene_image_view, VK_NULL_HANDLE);
    EXPECT_EQ(queued.external_scene_image_layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    EXPECT_EQ(queued.external_scene_image_generation, 1u);
}

TEST_F(ViewportInteropQueuedUpload, NullCommandBufferDiscardsUnrecordedBinding) {
    queue(7);
    interop->recordFrameBarriers(VK_NULL_HANDLE, *context);
    EXPECT_EQ(bind().external_scene_image, VK_NULL_HANDLE);
    EXPECT_EQ(bind(true).external_scene_image, VK_NULL_HANDLE);
}

TEST_F(ViewportInteropQueuedUpload, QueuedOlderContentDoesNotBindNewSourceGeneration) {
    queue(7);
    interop->setSceneImage(image, {8, 8}, false, 8);
    EXPECT_EQ(bind().external_scene_image, VK_NULL_HANDLE);
}

TEST_F(ViewportInteropQueuedUpload, ZeroGenerationDoesNotBindQueuedContent) {
    queue(0);
    EXPECT_EQ(bind().external_scene_image, VK_NULL_HANDLE);
}

TEST_F(ViewportInteropQueuedUpload, ExportSyncDiscardsUnrecordedBinding) {
    queue(7);
    interop->syncUnsubmittedLayoutCommits(*context);
    EXPECT_EQ(bind(true).external_scene_image, VK_NULL_HANDLE);
    queue(8);
    EXPECT_NE(bind().external_scene_image, VK_NULL_HANDLE);
    EXPECT_EQ(bind().external_scene_image_generation, 1u);
}

TEST_F(ViewportInteropQueuedUpload, RecordedGenerationMatchesPredictionAndExportCache) {
    queue(7);
    const auto queued = bind();
    lfs::vis::VulkanContext::Frame frame{};
    const VkClearValue clear{};
    ASSERT_TRUE(context->beginFrame(clear, frame)) << context->lastError();
    ASSERT_TRUE(context->finishActiveRendering(frame.command_buffer));
    interop->recordFrameBarriers(frame.command_buffer, *context);
    const auto recorded = bind();
    EXPECT_EQ(recorded.external_scene_image, queued.external_scene_image);
    EXPECT_EQ(recorded.external_scene_image_generation, queued.external_scene_image_generation);
    ASSERT_TRUE(context->restartActiveRendering(frame.command_buffer, frame));
    ASSERT_TRUE(context->endFrame()) << context->lastError();
    ASSERT_TRUE(context->waitForSubmittedFrames());
    interop->syncUnsubmittedLayoutCommits(*context);
    const auto cached = bind(true);
    EXPECT_EQ(cached.external_scene_image, recorded.external_scene_image);
    EXPECT_EQ(cached.external_scene_image_generation, recorded.external_scene_image_generation);
}

TEST_F(ViewportInteropQueuedUpload, RejectedFrameRollsBackBindingBeforeExport) {
    queue(7);
    const auto successful = context->lastSuccessfulFrameSubmitSerial();
    lfs::vis::VulkanContext::Frame frame{};
    const VkClearValue clear{};
    ASSERT_TRUE(context->beginFrame(clear, frame)) << context->lastError();
    ASSERT_TRUE(context->finishActiveRendering(frame.command_buffer));
    interop->recordFrameBarriers(frame.command_buffer, *context);
    EXPECT_NE(bind().external_scene_image, VK_NULL_HANDLE);
    // Reject the frame before queue submission without faulting the GPU.
    EXPECT_FALSE(context->addFrameTimelineWait(VK_NULL_HANDLE, 1, VK_PIPELINE_STAGE_TRANSFER_BIT));
    EXPECT_FALSE(context->endFrame());
    EXPECT_EQ(context->lastSuccessfulFrameSubmitSerial(), successful);
    interop->syncUnsubmittedLayoutCommits(*context);
    EXPECT_EQ(bind(true).external_scene_image, VK_NULL_HANDLE);
    queue(7);
    EXPECT_NE(bind().external_scene_image, VK_NULL_HANDLE);
    EXPECT_EQ(bind().external_scene_image_generation, 2u);
}
