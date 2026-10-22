/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/ipc/render_settings_convert.hpp"
#include "visualizer/rendering/vksplat_viewport_renderer.hpp"
#include <gtest/gtest.h>

namespace lfs::vis {
    struct StochasticHistoryTestAccess {
        static auto prepare(VksplatViewportRenderer& renderer, VulkanGSRendererUniforms& camera,
                            const core::SplatData& model, const VksplatViewportRenderer::OutputSlot slot,
                            const bool content_changed = false) {
            return renderer.prepareStochasticFrame(camera, model, slot, content_changed);
        }
        static void commit(VksplatViewportRenderer& renderer,
                           const VksplatViewportRenderer::StochasticFrame& frame,
                           const VksplatViewportRenderer::OutputSlot slot) {
            renderer.commitStochasticFrame(frame, slot);
        }
        static auto projection(const VksplatViewportRenderer::StochasticFrame& frame,
                               const VksplatViewportRenderer::OutputSlot slot) {
            return VksplatViewportRenderer::StochasticProjectionCache{
                .frame = frame,
                .slot = slot,
                .first_submission = 10,
                .last_submission = 12};
        }
        static bool reuseProjection(const VksplatViewportRenderer::StochasticFrame& frame,
                                    const VksplatViewportRenderer::OutputSlot slot,
                                    const VksplatViewportRenderer::StochasticProjectionCache& cache) {
            return VksplatViewportRenderer::canReuseStochasticProjection(frame, slot, cache);
        }
        static void projectionStats(VksplatViewportRenderer::StochasticProjectionCache& cache,
                                    const VulkanGSRenderer::TileInstanceStats& stats) {
            VksplatViewportRenderer::updateStochasticProjectionStats(cache, stats);
        }
        static bool allocated(const VksplatViewportRenderer& renderer) {
            for (const auto& history : renderer.stochastic_histories_)
                if (history.buffers.color[0].deviceBuffer.buffer != VK_NULL_HANDLE ||
                    history.buffers.ids_depths.deviceBuffer.buffer != VK_NULL_HANDLE)
                    return true;
            return false;
        }
    };
} // namespace lfs::vis

namespace {
    using Renderer = lfs::vis::VksplatViewportRenderer;
    using Slot = Renderer::OutputSlot;
    using Access = lfs::vis::StochasticHistoryTestAccess;

    VulkanGSRendererUniforms camera(uint32_t mode) {
        VulkanGSRendererUniforms u{};
        u.image_width = 64;
        u.image_height = 32;
        u.num_splats = 10;
        u.stochastic_mode = mode;
        u.fx = u.fy = 40;
        for (int i = 0; i < 4; ++i)
            u.world_view_transform[5 * i] = 1;
        return u;
    }

    TEST(StochasticViewport, NoiseSequenceIsConsecutivePerViewAndSurvivesHistoryResets) {
        Renderer renderer;
        lfs::core::SplatData model;
        auto left = camera(2), right = camera(2);
        auto frame = Access::prepare(renderer, left, model, Slot::SplitLeft);
        EXPECT_EQ(left.stochastic_frame, 1u);
        Access::commit(renderer, frame, Slot::SplitLeft);
        frame = Access::prepare(renderer, right, model, Slot::SplitRight);
        EXPECT_EQ(right.stochastic_frame, 1u);
        Access::commit(renderer, frame, Slot::SplitRight);
        // Merely preparing a frame does not consume a slice.
        Access::prepare(renderer, right, model, Slot::SplitRight);
        renderer.invalidateStochasticHistory();
        frame = Access::prepare(renderer, left, model, Slot::SplitLeft);
        EXPECT_EQ(left.stochastic_frame, 2u);
        EXPECT_FALSE(frame.history_valid);
        Access::commit(renderer, frame, Slot::SplitLeft);
        Access::prepare(renderer, right, model, Slot::SplitRight);
        EXPECT_EQ(right.stochastic_frame, 2u);
    }

    TEST(StochasticViewport, BasicAccumulationResetsForCameraAndContentChanges) {
        Renderer renderer;
        lfs::core::SplatData model;
        auto u = camera(1);
        auto frame = Access::prepare(renderer, u, model, Slot::Main);
        EXPECT_FALSE(frame.history_valid);
        EXPECT_EQ(frame.samples, 0u);
        Access::commit(renderer, frame, Slot::Main);
        frame = Access::prepare(renderer, u, model, Slot::Main);
        EXPECT_TRUE(frame.history_valid);
        EXPECT_EQ(frame.samples, 1u);
        // Dispatch-only fields do not restart a stationary accumulation.
        u.sort_capacity = 8192;
        u.depth_wave = 7;
        u.stochastic_frame = 123;
        EXPECT_EQ(Access::prepare(renderer, u, model, Slot::Main).samples, 1u);
        u.world_view_transform[12] = 1;
        frame = Access::prepare(renderer, u, model, Slot::Main);
        EXPECT_FALSE(frame.history_valid);
        EXPECT_EQ(frame.samples, 0u);
        Access::commit(renderer, frame, Slot::Main);
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot::Main, true).history_valid);
        renderer.invalidateStochasticHistory();
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot::Main).history_valid);
        EXPECT_FALSE(Access::allocated(renderer));
    }

    TEST(StochasticViewport, SteadySplatsReprojectsAcrossMotionAndResetsForResizeModelAndMode) {
        Renderer renderer;
        lfs::core::SplatData model, other;
        auto u = camera(2);
        Access::commit(renderer, Access::prepare(renderer, u, model, Slot::Main), Slot::Main);
        u.world_view_transform[12] = 1;
        auto frame = Access::prepare(renderer, u, model, Slot::Main);
        EXPECT_TRUE(frame.history_valid);
        EXPECT_EQ(frame.samples, 0u);
        Access::commit(renderer, frame, Slot::Main);
        EXPECT_EQ(Access::prepare(renderer, u, model, Slot::Main).samples, 1u);
        EXPECT_FALSE(Access::prepare(renderer, u, other, Slot::Main).history_valid);
        ++u.image_width;
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot::Main).history_valid);
        --u.image_width;
        u.stochastic_mode = 1;
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot::Main).history_valid);
        u.stochastic_mode = 2;
        renderer.invalidateStochasticHistory();
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot::Main).history_valid);
    }

    TEST(StochasticViewport, LogicalViewsAndUncommittedFramesDoNotShareHistory) {
        Renderer renderer;
        lfs::core::SplatData model;
        auto u = camera(2);
        Access::commit(renderer, Access::prepare(renderer, u, model, Slot::Main), Slot::Main);
        auto candidate = Access::prepare(renderer, u, model, Slot::Main);
        EXPECT_EQ(candidate.samples, 1u);
        // Recording a candidate is pure; only a submitted, composed frame is committed.
        EXPECT_EQ(Access::prepare(renderer, u, model, Slot::Main).samples, 1u);
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot::SplitLeft).history_valid);
        Access::commit(renderer, Access::prepare(renderer, u, model, Slot::SplitLeft), Slot::SplitLeft);
        Access::commit(renderer, candidate, Slot::Main);
        EXPECT_EQ(Access::prepare(renderer, u, model, Slot::Main).samples, 2u);
        EXPECT_EQ(Access::prepare(renderer, u, model, Slot::SplitLeft).samples, 1u);
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot::SplitRight).history_valid);
        u.stochastic_mode = 0;
        candidate = Access::prepare(renderer, u, model, Slot::Main);
        EXPECT_FALSE(candidate.enabled);
        Access::commit(renderer, candidate, Slot::Main);
        u.stochastic_mode = 2;
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot::Main).history_valid);
        EXPECT_FALSE(Access::allocated(renderer));
    }

    TEST(StochasticViewport, SettingsDefaultToAlphaBlendAndRoundTripThroughTheProxy) {
        lfs::vis::RenderSettings settings;
        EXPECT_EQ(settings.transparency_mode, lfs::rendering::GaussianTransparencyMode::AlphaBlend);
        settings.transparency_mode = lfs::rendering::GaussianTransparencyMode::SteadySplats;
        settings.stochastic_max_frames = 128;
        auto proxy = lfs::vis::to_proxy(settings);
        lfs::vis::RenderSettings restored;
        lfs::vis::apply_proxy(restored, proxy);
        EXPECT_EQ(restored.transparency_mode, settings.transparency_mode);
        EXPECT_EQ(restored.stochastic_max_frames, 128);
        proxy.transparency_mode = -1;
        proxy.stochastic_max_frames = 2048;
        lfs::vis::apply_proxy(restored, proxy);
        EXPECT_EQ(restored.transparency_mode, lfs::rendering::GaussianTransparencyMode::AlphaBlend);
        EXPECT_EQ(restored.stochastic_max_frames, 1024);
    }

    TEST(StochasticViewport, ProjectionReuseRequiresIdenticalInputsAndTheResidentView) {
        Renderer renderer;
        lfs::core::SplatData model, other;
        auto u = camera(2);
        auto frame = Access::prepare(renderer, u, model, Slot::Main);
        auto cache = Access::projection(frame, Slot::Main);
        EXPECT_FALSE(Access::reuseProjection(frame, Slot::Main, cache));
        Access::commit(renderer, frame, Slot::Main);
        frame = Access::prepare(renderer, u, model, Slot::Main);
        EXPECT_TRUE(Access::reuseProjection(frame, Slot::Main, cache));
        EXPECT_FALSE(Access::reuseProjection(frame, Slot::SplitLeft, cache));
        EXPECT_FALSE(Access::reuseProjection(frame, Slot::Preview, cache));
        auto changed = frame;
        changed.model = &other;
        EXPECT_FALSE(Access::reuseProjection(changed, Slot::Main, cache));
        changed = frame;
        ++changed.generation;
        EXPECT_FALSE(Access::reuseProjection(changed, Slot::Main, cache));
        changed = frame;
        changed.camera.fx += 1;
        EXPECT_FALSE(Access::reuseProjection(changed, Slot::Main, cache));
        changed = frame;
        changed.history_valid = false;
        EXPECT_FALSE(Access::reuseProjection(changed, Slot::Main, cache));
        u.world_view_transform[12] = 1;
        changed = Access::prepare(renderer, u, model, Slot::Main);
        EXPECT_TRUE(changed.history_valid); // Temporal reuse still handles motion.
        EXPECT_FALSE(Access::reuseProjection(changed, Slot::Main, cache));
        cache.first_submission = 0;
        EXPECT_FALSE(Access::reuseProjection(frame, Slot::Main, cache));
    }

    TEST(StochasticViewport, WaveBoundsOnlyUseResultsFromTheUnchangedProjection) {
        Renderer renderer;
        lfs::core::SplatData model;
        auto u = camera(2);
        auto cache = Access::projection(Access::prepare(renderer, u, model, Slot::Main), Slot::Main);
        VulkanGSRenderer::TileInstanceStats stats{.waves_needed = 3, .submission_value = 9};
        Access::projectionStats(cache, stats);
        EXPECT_EQ(cache.wave_count, 0u);
        stats.submission_value = 13;
        Access::projectionStats(cache, stats);
        EXPECT_EQ(cache.wave_count, 0u);
        stats.submission_value = 11;
        Access::projectionStats(cache, stats);
        EXPECT_EQ(cache.wave_count, 3u);
        stats.count_overflow = true;
        stats.waves_needed = 1;
        Access::projectionStats(cache, stats);
        EXPECT_EQ(cache.wave_count, 3u);
        stats.count_overflow = false;
        stats.waves_needed = 0;
        Access::projectionStats(cache, stats);
        EXPECT_EQ(cache.wave_count, 1u); // Empty scenes still initialize background.
    }
} // namespace
