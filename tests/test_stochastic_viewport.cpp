/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/ipc/render_settings_convert.hpp"
#include "visualizer/rendering/vksplat_viewport_renderer.hpp"
#include <gtest/gtest.h>

namespace lfs::vis {
    struct StochasticHistoryTestAccess {
        static auto prepare(VksplatViewportRenderer& renderer, VulkanGSRendererUniforms& camera,
                            const core::SplatData& model, const RenderTargetId slot,
                            const bool content_changed = false,
                            VksplatViewportRenderer::StochasticEffect effect = {}) {
            return renderer.prepareStochasticFrame(camera, model, slot, content_changed, std::move(effect));
        }
        static void commit(VksplatViewportRenderer& renderer,
                           const VksplatViewportRenderer::StochasticFrame& frame,
                           const RenderTargetId slot) {
            renderer.commitStochasticFrame(frame, slot);
        }
        static auto projection(const VksplatViewportRenderer::StochasticFrame& frame,
                               const RenderTargetId slot) {
            return VksplatViewportRenderer::StochasticProjectionCache{
                .frame = frame,
                .slot = slot,
                .first_submission = 10,
                .last_submission = 12};
        }
        static bool reuseProjection(const VksplatViewportRenderer::StochasticFrame& frame,
                                    const RenderTargetId slot,
                                    const VksplatViewportRenderer::StochasticProjectionCache& cache) {
            return VksplatViewportRenderer::canReuseStochasticProjection(frame, slot, cache);
        }
        static void projectionStats(VksplatViewportRenderer::StochasticProjectionCache& cache,
                                    const VulkanGSRenderer::TileInstanceStats& stats) {
            VksplatViewportRenderer::updateStochasticProjectionStats(cache, stats);
        }
        static void checkUnusedBuffersAreRetired() {
            using Slot = RenderTargetId;
            VksplatViewportRenderer renderer;
            std::uintptr_t next = 0x100;
            const auto fake = [&](_VulkanBuffer& buffer) {
                buffer.buffer = reinterpret_cast<VkBuffer>(next++);
                buffer.allocation = reinterpret_cast<VmaAllocation>(next++);
                buffer.size = buffer.capacity = buffer.allocSize = 64;
            };
            auto& main = renderer.stochastic_histories_[Slot{1}].buffers;
            for (auto* buffer : {&main.occluders.deviceBuffer,
                                 &main.outgoing_weights.deviceBuffer, &main.rest_average.deviceBuffer,
                                 &main.color[0].deviceBuffer, &main.color[1].deviceBuffer})
                fake(*buffer);
            fake(renderer.stochastic_histories_[Slot{2}].buffers.color[0].deviceBuffer);
            fake(renderer.buffers_.primitive_sort_indices.deviceBuffer);
            renderer.stochastic_sort_indices_owned_ = true;
            renderer.render_complete_timeline_ = reinterpret_cast<VkSemaphore>(0x301);
            renderer.last_submitted_render_value_ = 7;
            const auto& retired = renderer.retired_private_scratch_buffers_;

            renderer.releaseUnusedStochasticBuffers(Slot{1}, 2);
            EXPECT_TRUE(retired.empty());
            // Basic accumulation keeps only its color history.
            renderer.releaseUnusedStochasticBuffers(Slot{1}, 1);
            EXPECT_EQ(retired.size(), 3u);
            EXPECT_NE(main.color[0].deviceBuffer.buffer, VK_NULL_HANDLE);
            // Another view still holds history, so the sort-index copy stays.
            renderer.releaseUnusedStochasticBuffers(Slot{1}, 0);
            EXPECT_EQ(retired.size(), 5u);
            EXPECT_NE(renderer.buffers_.primitive_sort_indices.deviceBuffer.buffer, VK_NULL_HANDLE);
            renderer.releaseUnusedStochasticBuffers(Slot{2}, 0);
            EXPECT_EQ(retired.size(), 7u);
            EXPECT_EQ(renderer.buffers_.primitive_sort_indices.deviceBuffer.buffer, VK_NULL_HANDLE);
            for (const auto& [value, buffer] : retired)
                EXPECT_EQ(value, 7u); // Destroyed once the last submitted frame retires.
            // The handles above are scripted; they must never reach Vulkan teardown.
            renderer.retired_private_scratch_buffers_.clear();
            renderer.render_complete_timeline_ = VK_NULL_HANDLE;
        }
    };
} // namespace lfs::vis

namespace {
    using Renderer = lfs::vis::VksplatViewportRenderer;
    using Slot = lfs::vis::RenderTargetId;
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
        auto frame = Access::prepare(renderer, left, model, Slot{2});
        EXPECT_EQ(left.stochastic_frame, 1u);
        Access::commit(renderer, frame, Slot{2});
        frame = Access::prepare(renderer, right, model, Slot{3});
        EXPECT_EQ(right.stochastic_frame, 1u);
        Access::commit(renderer, frame, Slot{3});
        // Merely preparing a frame does not consume a slice.
        Access::prepare(renderer, right, model, Slot{3});
        renderer.invalidateStochasticHistory();
        frame = Access::prepare(renderer, left, model, Slot{2});
        EXPECT_EQ(left.stochastic_frame, 2u);
        EXPECT_FALSE(frame.history_valid);
        Access::commit(renderer, frame, Slot{2});
        Access::prepare(renderer, right, model, Slot{3});
        EXPECT_EQ(right.stochastic_frame, 2u);
    }

    TEST(StochasticViewport, CameraAndOverlayRedrawsPreserveHistoryWhileSceneChangesResetIt) {
        Renderer renderer;
        lfs::core::SplatData model;
        auto u = camera(2);
        Access::commit(renderer, Access::prepare(renderer, u, model, Slot{1}), Slot{1});
        for (const auto dirty : {lfs::vis::DirtyMask{0}, lfs::vis::DirtyFlag::CAMERA,
                                 lfs::vis::DirtyFlag::OVERLAY,
                                 lfs::vis::DirtyFlag::CAMERA | lfs::vis::DirtyFlag::OVERLAY}) {
            renderer.invalidateStochasticHistory(dirty);
            u.world_view_transform[12] += 0.01f;
            const auto frame = Access::prepare(renderer, u, model, Slot{1});
            EXPECT_TRUE(frame.history_valid) << "dirty flags " << dirty;
            EXPECT_EQ(frame.samples, 0u); // The camera moved; radiance reprojects.
            Access::commit(renderer, frame, Slot{1});
        }
        for (const auto dirty : {lfs::vis::DirtyFlag::SPLATS, lfs::vis::DirtyFlag::MESH,
                                 lfs::vis::DirtyFlag::VIEWPORT, lfs::vis::DirtyFlag::PPISP,
                                 lfs::vis::DirtyFlag::SELECTION, lfs::vis::DirtyFlag::BACKGROUND,
                                 lfs::vis::DirtyFlag::SPLIT_VIEW, lfs::vis::DirtyFlag::SPLIT_POSITION}) {
            renderer.invalidateStochasticHistory(dirty);
            const auto frame = Access::prepare(renderer, u, model, Slot{1});
            EXPECT_FALSE(frame.history_valid) << "dirty flags " << dirty;
            Access::commit(renderer, frame, Slot{1});
        }
    }

    TEST(StochasticViewport, BasicAccumulationResetsForCameraAndContentChanges) {
        Renderer renderer;
        lfs::core::SplatData model;
        auto u = camera(1);
        auto frame = Access::prepare(renderer, u, model, Slot{1});
        EXPECT_FALSE(frame.history_valid);
        EXPECT_EQ(frame.samples, 0u);
        Access::commit(renderer, frame, Slot{1});
        frame = Access::prepare(renderer, u, model, Slot{1});
        EXPECT_TRUE(frame.history_valid);
        EXPECT_EQ(frame.samples, 1u);
        // Dispatch-only fields do not restart a stationary accumulation.
        u.sort_capacity = 8192;
        u.depth_wave = 7;
        u.stochastic_frame = 123;
        EXPECT_EQ(Access::prepare(renderer, u, model, Slot{1}).samples, 1u);
        u.world_view_transform[12] = 1;
        frame = Access::prepare(renderer, u, model, Slot{1});
        EXPECT_FALSE(frame.history_valid);
        EXPECT_EQ(frame.samples, 0u);
        Access::commit(renderer, frame, Slot{1});
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot{1}, true).history_valid);
        renderer.invalidateStochasticHistory();
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot{1}).history_valid);
    }

    TEST(StochasticViewport, SteadySplatsReprojectsAcrossMotionAndResetsForResizeModelAndMode) {
        Renderer renderer;
        lfs::core::SplatData model, other;
        auto u = camera(2);
        Access::commit(renderer, Access::prepare(renderer, u, model, Slot{1}), Slot{1});
        u.world_view_transform[12] = 1;
        auto frame = Access::prepare(renderer, u, model, Slot{1});
        EXPECT_TRUE(frame.history_valid);
        EXPECT_EQ(frame.samples, 0u);
        Access::commit(renderer, frame, Slot{1});
        EXPECT_EQ(Access::prepare(renderer, u, model, Slot{1}).samples, 1u);
        frame = Access::prepare(renderer, u, model, Slot{1}, true);
        EXPECT_FALSE(frame.history_valid);
        EXPECT_EQ(frame.samples, 0u);
        EXPECT_FALSE(Access::prepare(renderer, u, other, Slot{1}).history_valid);
        ++u.image_width;
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot{1}).history_valid);
        --u.image_width;
        u.stochastic_mode = 1;
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot{1}).history_valid);
        u.stochastic_mode = 2;
        renderer.invalidateStochasticHistory();
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot{1}).history_valid);
    }

    TEST(StochasticViewport, LogicalViewsAndUncommittedFramesDoNotShareHistory) {
        Renderer renderer;
        lfs::core::SplatData model;
        auto u = camera(2);
        Access::commit(renderer, Access::prepare(renderer, u, model, Slot{1}), Slot{1});
        auto candidate = Access::prepare(renderer, u, model, Slot{1});
        EXPECT_EQ(candidate.samples, 1u);
        // Recording a candidate is pure; only a submitted, composed frame is committed.
        EXPECT_EQ(Access::prepare(renderer, u, model, Slot{1}).samples, 1u);
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot{2}).history_valid);
        Access::commit(renderer, Access::prepare(renderer, u, model, Slot{2}), Slot{2});
        Access::commit(renderer, candidate, Slot{1});
        EXPECT_EQ(Access::prepare(renderer, u, model, Slot{1}).samples, 2u);
        EXPECT_EQ(Access::prepare(renderer, u, model, Slot{2}).samples, 1u);
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot{3}).history_valid);
        u.stochastic_mode = 0;
        candidate = Access::prepare(renderer, u, model, Slot{1});
        EXPECT_FALSE(candidate.enabled);
        Access::commit(renderer, candidate, Slot{1});
        u.stochastic_mode = 2;
        EXPECT_FALSE(Access::prepare(renderer, u, model, Slot{1}).history_valid);
    }

    TEST(StochasticViewport, DisabledModesRetireUnusedBuffers) {
        Access::checkUnusedBuffersAreRetired();
    }

    TEST(StochasticViewport, SettingsDefaultToAlphaBlendAndRoundTripThroughTheProxy) {
        lfs::vis::RenderSettings settings;
        EXPECT_EQ(settings.transparency_mode, lfs::rendering::GaussianTransparencyMode::AlphaBlend);
        EXPECT_EQ(settings.stochastic_max_frames, 16);
        EXPECT_EQ(lfs::vis::RenderSettingsProxy{}.stochastic_max_frames, settings.stochastic_max_frames);
        EXPECT_EQ(lfs::rendering::ViewportRenderRequest{}.stochastic_max_frames, settings.stochastic_max_frames);
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
        auto frame = Access::prepare(renderer, u, model, Slot{1});
        auto cache = Access::projection(frame, Slot{1});
        EXPECT_FALSE(Access::reuseProjection(frame, Slot{1}, cache));
        Access::commit(renderer, frame, Slot{1});
        frame = Access::prepare(renderer, u, model, Slot{1});
        EXPECT_TRUE(Access::reuseProjection(frame, Slot{1}, cache));
        EXPECT_FALSE(Access::reuseProjection(frame, Slot{2}, cache));
        EXPECT_FALSE(Access::reuseProjection(frame, Slot{4}, cache));
        auto changed = frame;
        changed.model = &other;
        EXPECT_FALSE(Access::reuseProjection(changed, Slot{1}, cache));
        changed = frame;
        ++changed.generation;
        EXPECT_FALSE(Access::reuseProjection(changed, Slot{1}, cache));
        changed = frame;
        changed.camera.fx += 1;
        EXPECT_FALSE(Access::reuseProjection(changed, Slot{1}, cache));
        changed = frame;
        changed.history_valid = false;
        EXPECT_FALSE(Access::reuseProjection(changed, Slot{1}, cache));
        u.world_view_transform[12] = 1;
        changed = Access::prepare(renderer, u, model, Slot{1});
        EXPECT_TRUE(changed.history_valid); // Temporal reuse still handles motion.
        EXPECT_FALSE(Access::reuseProjection(changed, Slot{1}, cache));
        cache.first_submission = 0;
        EXPECT_FALSE(Access::reuseProjection(frame, Slot{1}, cache));
    }

    TEST(StochasticViewport, WaveBoundsOnlyUseResultsFromTheUnchangedProjection) {
        Renderer renderer;
        lfs::core::SplatData model;
        auto u = camera(2);
        auto cache = Access::projection(Access::prepare(renderer, u, model, Slot{1}), Slot{1});
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
