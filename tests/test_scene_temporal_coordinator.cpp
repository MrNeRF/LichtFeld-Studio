/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/passes/vulkan_scene_motion_pass.hpp"
#include "visualizer/rendering/passes/vulkan_viewport_pass.hpp"
#include "visualizer/rendering/scene_temporal_coordinator.hpp"

#include <gtest/gtest.h>

namespace lfs::vis {
    namespace {
        SceneTemporalRequest requestFor(const TemporalViewId view,
                                        const SceneUpscalerRequirements requirements) {
            SceneTemporalRequest request{
                .view = view,
                .requirements = requirements,
                .render_extent = {1280, 720},
                .output_extent = {1920, 1080},
            };
            request.frame.view.size = request.render_extent;
            request.frame.render_scale = 2.0f / 3.0f;
            request.frame.scene_generation = 7;
            request.frame.backend_key = 42;
            return request;
        }

        SceneUpscalerRequirements temporalRequirements(const bool history = true) {
            return {
                .depth = true,
                .motion_vectors = true,
                .jitter = true,
                .history = history,
            };
        }
    } // namespace

    TEST(SceneTemporalCoordinator, NativeAndSpatialDoNotAdvanceTemporalState) {
        SceneTemporalCoordinator coordinator;
        auto native = requestFor(TemporalViewId::Main,
                                 nativeSceneUpscalerDescriptor().requirements);
        auto spatial = requestFor(TemporalViewId::Main,
                                  spatialSceneUpscalerDescriptor().requirements);

        EXPECT_TRUE(coordinator.prepare(native).plan.zeroCost());
        EXPECT_TRUE(coordinator.prepare(spatial).plan.zeroCost());
        EXPECT_FALSE(coordinator.commit(native, coordinator.prepare(native)));

        auto temporal = requestFor(TemporalViewId::Main, temporalRequirements());
        const auto first = coordinator.prepare(temporal);
        EXPECT_FALSE(first.frame.history_valid);
        EXPECT_EQ(first.frame.sequence, 0u);
        EXPECT_FALSE(first.history.available());
    }

    TEST(SceneTemporalCoordinator, CommitsFrameAndHistoryAsOneTransaction) {
        SceneTemporalCoordinator coordinator;
        auto request = requestFor(TemporalViewId::Main, temporalRequirements());
        const auto first = coordinator.prepare(request);
        ASSERT_TRUE(coordinator.commit(request,
                                       first,
                                       SceneHistoryStorage::VulkanImage,
                                       SceneHistoryStorage::VulkanImage));

        const auto second = coordinator.prepare(request);
        EXPECT_TRUE(second.frame.history_valid);
        EXPECT_EQ(second.frame.sequence, 1u);
        EXPECT_TRUE(second.history.available());
        EXPECT_TRUE(second.history.hasDepth());
        EXPECT_EQ(second.history.sequence, second.frame.sequence);
    }

    TEST(SceneTemporalCoordinator, BackendDisableInvalidatesPriorHistory) {
        SceneTemporalCoordinator coordinator;
        auto temporal = requestFor(TemporalViewId::Main, temporalRequirements());
        const auto first = coordinator.prepare(temporal);
        ASSERT_TRUE(coordinator.commit(temporal,
                                       first,
                                       SceneHistoryStorage::VulkanImage,
                                       SceneHistoryStorage::VulkanImage));

        auto native = requestFor(TemporalViewId::Main,
                                 nativeSceneUpscalerDescriptor().requirements);
        EXPECT_TRUE(coordinator.prepare(native).plan.zeroCost());

        const auto reenabled = coordinator.prepare(temporal);
        EXPECT_FALSE(reenabled.frame.history_valid);
        EXPECT_FALSE(reenabled.history.available());
        EXPECT_TRUE(hasTemporalResetReason(reenabled.frame.reset_reasons,
                                           TemporalResetReason::HistoryDisabled));
    }

    TEST(SceneTemporalCoordinator, KeepsSplitViewsIndependent) {
        SceneTemporalCoordinator coordinator;
        auto left = requestFor(TemporalViewId::SplitLeft, temporalRequirements(false));
        auto right = requestFor(TemporalViewId::SplitRight, temporalRequirements(false));

        const auto left_first = coordinator.prepare(left);
        ASSERT_TRUE(coordinator.commit(left, left_first));
        EXPECT_TRUE(coordinator.prepare(left).frame.history_valid);
        EXPECT_FALSE(coordinator.prepare(right).frame.history_valid);
    }

    TEST(SceneTemporalCoordinator, RejectsStaleOrInvalidPreparedFrames) {
        SceneTemporalCoordinator coordinator;
        auto request = requestFor(TemporalViewId::Main, temporalRequirements());
        const auto prepared = coordinator.prepare(request);

        request.output_extent = {2560, 1440};
        EXPECT_FALSE(coordinator.commit(request,
                                        prepared,
                                        SceneHistoryStorage::VulkanImage,
                                        SceneHistoryStorage::VulkanImage));

        auto malformed = requestFor(TemporalViewId::Main, temporalRequirements());
        malformed.render_extent = {0, 720};
        EXPECT_FALSE(coordinator.prepare(malformed).active());
        EXPECT_FALSE(coordinator.commit(malformed, coordinator.prepare(malformed)));
    }

    TEST(SceneTemporalCoordinator, FailedHistoryCommitCannotPartiallyAdvanceFrameState) {
        SceneTemporalCoordinator coordinator;
        auto request = requestFor(TemporalViewId::Main, temporalRequirements());
        const auto first = coordinator.prepare(request);

        EXPECT_FALSE(coordinator.commit(request,
                                        first,
                                        SceneHistoryStorage::VulkanImage,
                                        SceneHistoryStorage::None));
        const auto retry = coordinator.prepare(request);
        EXPECT_FALSE(retry.frame.history_valid);
        EXPECT_FALSE(retry.history.available());
        EXPECT_EQ(retry.frame.sequence, 0u);
        EXPECT_TRUE(hasTemporalResetReason(retry.frame.reset_reasons,
                                           TemporalResetReason::InvalidInput));
    }

    TEST(SceneTemporalCoordinator, InvalidatesHistoryAcrossEveryRuntimeBoundary) {
        const auto verify_reset = [](const auto mutate,
                                     const TemporalResetReason expected) {
            SceneTemporalCoordinator coordinator;
            auto request = requestFor(TemporalViewId::Main, temporalRequirements());
            const auto first = coordinator.prepare(request);
            EXPECT_TRUE(coordinator.commit(request,
                                           first,
                                           SceneHistoryStorage::VulkanImage,
                                           SceneHistoryStorage::VulkanImage));
            mutate(request);
            const auto changed = coordinator.prepare(request);
            EXPECT_FALSE(changed.frame.history_valid);
            EXPECT_FALSE(changed.history.available());
            EXPECT_TRUE(hasTemporalResetReason(changed.frame.reset_reasons, expected));
        };

        verify_reset([](auto& request) {
            request.render_extent = {960, 540};
            request.frame.view.size = request.render_extent;
        },
                     TemporalResetReason::RenderSize);
        verify_reset([](auto& request) { request.frame.render_scale = 0.5f; },
                     TemporalResetReason::RenderScale);
        verify_reset([](auto& request) { ++request.frame.scene_generation; },
                     TemporalResetReason::Scene);
        verify_reset([](auto& request) { ++request.frame.backend_key; },
                     TemporalResetReason::Backend);
        verify_reset([](auto& request) { request.frame.camera_cut = true; },
                     TemporalResetReason::CameraCut);
    }

    TEST(SceneTemporalCoordinator, ResetAllInvalidatesEveryViewWithoutCrossContamination) {
        SceneTemporalCoordinator coordinator;
        for (const auto view : {TemporalViewId::Main,
                                TemporalViewId::SplitLeft,
                                TemporalViewId::SplitRight}) {
            auto request = requestFor(view, temporalRequirements());
            const auto first = coordinator.prepare(request);
            ASSERT_TRUE(coordinator.commit(request,
                                           first,
                                           SceneHistoryStorage::VulkanImage,
                                           SceneHistoryStorage::VulkanImage));
        }

        coordinator.resetAll(TemporalResetReason::Scene);
        for (const auto view : {TemporalViewId::Main,
                                TemporalViewId::SplitLeft,
                                TemporalViewId::SplitRight}) {
            const auto reset = coordinator.prepare(
                requestFor(view, temporalRequirements()));
            EXPECT_FALSE(reset.frame.history_valid);
            EXPECT_FALSE(reset.history.available());
            EXPECT_TRUE(hasTemporalResetReason(reset.frame.reset_reasons,
                                               TemporalResetReason::Scene));
        }
    }

    TEST(SceneUpscalerRegistry, DescriptorLookupFallsBackSafelyForUnknownEnum) {
        EXPECT_EQ(sceneUpscalerDescriptor(SceneUpscalerBackend::Native).id, "native");
        EXPECT_EQ(sceneUpscalerDescriptor(SceneUpscalerBackend::Spatial).id, "spatial");
        EXPECT_EQ(sceneUpscalerDescriptor(SceneUpscalerBackend::Temporal).id, "temporal");
        EXPECT_EQ(sceneUpscalerDescriptor(static_cast<SceneUpscalerBackend>(255)).id, "native");
    }

    TEST(SceneTemporalCoordinator, VulkanScopeBreakRequiresEnabledValidMotionAndDepth) {
        VulkanSceneMotionParams params;
        params.render_extent = {1280, 720};

        EXPECT_FALSE(needsVulkanSceneMotionPreRender(params, true));
        params.enabled = true;
        EXPECT_FALSE(needsVulkanSceneMotionPreRender(params, false));
        EXPECT_TRUE(needsVulkanSceneMotionPreRender(params, true));
        params.render_extent = {0, 720};
        EXPECT_FALSE(needsVulkanSceneMotionPreRender(params, true));
    }

    TEST(SceneTemporalCoordinator, RuntimeEligibilityRejectsUnsupportedViewportPaths) {
        constexpr glm::ivec2 render{1100, 738};
        constexpr glm::ivec2 output{2200, 1476};
        EXPECT_TRUE(temporalViewportRuntimeEligible(
            false, true, true, true, render, output));
        EXPECT_FALSE(temporalViewportRuntimeEligible(
            true, true, true, true, render, output));
        EXPECT_FALSE(temporalViewportRuntimeEligible(
            false, false, true, true, render, output));
        EXPECT_FALSE(temporalViewportRuntimeEligible(
            false, true, false, true, render, output));
        EXPECT_FALSE(temporalViewportRuntimeEligible(
            false, true, true, false, render, output));
        EXPECT_FALSE(temporalViewportRuntimeEligible(
            false, true, true, true, {0, 738}, output));
        EXPECT_FALSE(temporalViewportRuntimeEligible(
            false, true, true, true, render, {2200, 0}));
    }

    TEST(SceneTemporalCoordinator, SourceReuseRequiresMatchingImageAndContentGeneration) {
        const auto image_a = reinterpret_cast<VkImage>(static_cast<std::uintptr_t>(0x1000));
        const auto image_b = reinterpret_cast<VkImage>(static_cast<std::uintptr_t>(0x2000));

        EXPECT_TRUE(temporalSourceUnchanged(image_a, 7, image_a, 7));
        EXPECT_FALSE(temporalSourceUnchanged(image_a, 7, image_b, 7));
        EXPECT_FALSE(temporalSourceUnchanged(image_a, 7, image_a, 8));
        EXPECT_FALSE(temporalSourceUnchanged(VK_NULL_HANDLE, 7, VK_NULL_HANDLE, 7));
    }

    TEST(SceneTemporalCoordinator, ExplicitCameraGenerationInvalidatesStableSceneHistory) {
        EXPECT_FALSE(temporalHistoryRequiresReset(false, 10, 10, 3, 4));
        EXPECT_FALSE(temporalHistoryRequiresReset(true, 10, 10, 3, 3));
        EXPECT_TRUE(temporalHistoryRequiresReset(true, 10, 11, 3, 3));
        EXPECT_TRUE(temporalHistoryRequiresReset(true, 10, 10, 3, 4));
        EXPECT_TRUE(temporalHistoryRequiresReset(true,
                                                 10,
                                                 10,
                                                 3,
                                                 3,
                                                 SceneTemporalQuality::Balanced,
                                                 SceneTemporalQuality::Quality));
    }

} // namespace lfs::vis
