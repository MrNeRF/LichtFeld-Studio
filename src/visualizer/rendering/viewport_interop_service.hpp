/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "rendering/cuda_vulkan_interop.hpp"

#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    class VulkanContext;
    struct VulkanViewportPassParams;

    // Pure early-path decision for per-slot interop prepare. No Vulkan/CUDA calls.
    // Used by ViewportInteropService::prepareChannel and unit tests.
    //
    // target_size_matches: bucket/alloc match (ceil64 source vs target alloc).
    // target_valid_size_matches: logical source size equals stored valid size.
    // Within-bucket valid-size change is SlowPath re-upload, not recreate (DeferBail).
    struct ViewportInteropSlotInputs {
        bool disabled = false;
        bool external_handle_early_out = false;
        bool has_external_scene_image = false;
        bool source_ok = false;
        bool publishes_published = false;
        bool resize_deferring = false;
        bool slot_array_resize_needed = false;
        bool frame_slot_in_range = false;
        bool target_present = false;
        bool target_size_matches = false;       // bucket match
        bool target_valid_size_matches = false; // logical size match
        bool target_interop_valid = false;
        bool target_layout_read_only = false;
        std::uint64_t source_generation = 0;
        std::uint64_t uploaded_source_generation = 0;
    };

    enum class ViewportInteropAction : std::uint8_t {
        Disabled,
        ExternalSkip,
        InvalidReset,
        CacheHit,
        DeferBail,
        SlowPath,
    };

    struct ViewportInteropDecision {
        ViewportInteropAction action = ViewportInteropAction::SlowPath;
        bool clear_published = false;
        bool publish_from_target = false;
    };

    [[nodiscard]] inline ViewportInteropDecision
    decideViewportInteropEarly(const ViewportInteropSlotInputs& in) {
        if (in.disabled) {
            return {.action = ViewportInteropAction::Disabled};
        }
        if (in.external_handle_early_out && in.has_external_scene_image) {
            return {.action = ViewportInteropAction::ExternalSkip};
        }
        if (!in.source_ok) {
            return {
                .action = ViewportInteropAction::InvalidReset,
                .clear_published = in.publishes_published,
            };
        }

        // Recreate only when the pooled unit is missing, bucket mismatches, or interop is dead.
        // Within-bucket valid-size changes are re-uploads (SlowPath), not recreates.
        const bool recreate_needed =
            !in.target_present || !in.target_size_matches || !in.target_interop_valid;
        if (!in.slot_array_resize_needed && in.frame_slot_in_range) {
            if (!recreate_needed &&
                in.target_valid_size_matches &&
                in.source_generation != 0 &&
                in.uploaded_source_generation == in.source_generation &&
                in.target_layout_read_only) {
                return {
                    .action = ViewportInteropAction::CacheHit,
                    .publish_from_target = in.publishes_published,
                };
            }
            if (in.resize_deferring && recreate_needed) {
                return {
                    .action = ViewportInteropAction::DeferBail,
                    .clear_published = in.publishes_published,
                };
            }
        } else if (in.resize_deferring) {
            return {
                .action = ViewportInteropAction::DeferBail,
                .clear_published = in.publishes_published,
            };
        }
        return {.action = ViewportInteropAction::SlowPath};
    }

    // Owns CUDA/Vulkan viewport interop lifecycle for scene, split-right, and depth-blit
    // channels. Visualizer presentation concern (namespace lfs::vis).
    class ViewportInteropService {
    public:
        ViewportInteropService();
        ~ViewportInteropService();

        ViewportInteropService(const ViewportInteropService&) = delete;
        ViewportInteropService& operator=(const ViewportInteropService&) = delete;

        void setSceneImage(std::shared_ptr<const lfs::core::Tensor> image,
                           glm::ivec2 size,
                           bool flip_y,
                           std::uint64_t generation,
                           VkSemaphore completion_semaphore = VK_NULL_HANDLE,
                           std::uint64_t completion_value = 0);
        void setExternalSceneImage(VkImage image,
                                   VkImageView image_view,
                                   VkImageLayout layout,
                                   glm::ivec2 size,
                                   bool flip_y,
                                   std::uint64_t generation,
                                   VkSemaphore completion_semaphore = VK_NULL_HANDLE,
                                   std::uint64_t completion_value = 0,
                                   glm::ivec2 alloc_size = {0, 0});
        void setSplitRightImage(std::shared_ptr<const lfs::core::Tensor> image,
                                glm::ivec2 size,
                                bool flip_y,
                                std::uint64_t generation);
        void clearSplitRightImage();
        void setDepthBlitImage(std::shared_ptr<const lfs::core::Tensor> depth,
                               glm::ivec2 size,
                               std::uint64_t generation);
        void clearDepthBlitImage();

        // Throws std::runtime_error on hard interop failure (callers catch).
        // Phase 1–2 of #1575: coalesce →GENERAL immediates + CUDA upload; defers
        // GENERAL→READ_ONLY barriers to recordFrameBarriers.
        void prepareFrame(VulkanContext& context, bool resize_deferring);

        // F2-1: run layout-commit rollback + discard unrecorded frame barriers on
        // every GUI frame that may endFrame, including export-locked frames that
        // skip prepareFrame Phases 1–2. prepareFrame calls this at its head.
        void syncUnsubmittedLayoutCommits(VulkanContext& context);

        // Phase 3 of #1575: record GENERAL→SHADER_READ_ONLY barriers into the open
        // frame CB and attach CUDA S2 timeline waits to the frame submit. Call
        // immediately after beginFrame succeeds, before any sampling of interop images.
        void recordFrameBarriers(VkCommandBuffer frame_cb, VulkanContext& context);

        void bindViewportParams(VulkanViewportPassParams& params,
                                std::size_t frame_slot,
                                bool export_locked,
                                bool resize_deferring) const;

        struct FrameCompletion {
            VkSemaphore semaphore = VK_NULL_HANDLE;
            std::uint64_t value = 0;
        };
        [[nodiscard]] FrameCompletion frameCompletion() const;

        // Drain upload stream, destroy all targets, release the stream. Optional
        // context overrides the prepare-cached pointer for teardown.
        void shutdown(VulkanContext* context = nullptr);

    private:
        enum class ChannelId : std::uint8_t {
            Scene,
            SplitRight,
            DepthBlit,
        };

        struct ChannelPolicy {
            ChannelId id = ChannelId::Scene;
            VkFormat vk_format = VK_FORMAT_R8G8B8A8_UNORM;
            lfs::rendering::CudaVulkanImageFormat cuda_format =
                lfs::rendering::CudaVulkanImageFormat::Rgba8Unorm;
            const char* debug_name_prefix = "scene";
            const char* failure_log_prefix = "Required Vulkan/CUDA viewport interop failed";
            bool external_handle_early_out = false;
            bool publishes_published = false;
            bool log_timer_perf = false;
        };

        struct Channel;
        struct PooledInteropUnit;
        struct VulkanSceneInteropTarget;

        // Decision-pass plan: channel is ready for Phase 1–2 upload.
        struct ChannelUploadPlan {
            Channel* channel = nullptr;
            VulkanSceneInteropTarget* target = nullptr;
        };
        // After CUDA signal S2: defer GENERAL→READ_ONLY + publish to the frame CB.
        struct PendingFrameBarrier {
            PooledInteropUnit* unit = nullptr;
            VulkanSceneInteropTarget* target = nullptr;
            Channel* channel = nullptr;
            std::uint64_t cuda_signal_value = 0;
            std::uint64_t source_generation = 0;
        };
        // Layout committed to READ_ONLY at record time; marker is lastSuccessful
        // frame serial at record — rolled back if no newer successful submit exists.
        struct PendingLayoutCommit {
            PooledInteropUnit* unit = nullptr;
            Channel* channel = nullptr;
            std::uint64_t frame_submit_marker = 0;
        };

        // Decision + setup only; may append to pending_uploads_ for Phase 1–2.
        void prepareChannel(VulkanContext& context, Channel& channel, bool resize_deferring);
        void resetChannel(Channel& channel);
        void clearPublished(Channel& channel);
        void publishFromTarget(Channel& channel, const VulkanSceneInteropTarget& target);
        void ensureUploadStream();
        void drainInteropPool(VulkanContext& context, bool force);
        void releaseSlotTarget(VulkanContext& context, VulkanSceneInteropTarget& target);
        void rollbackUnsubmittedLayoutCommits(VulkanContext& context);
        [[nodiscard]] bool sourceOk(const Channel& channel) const;
        [[nodiscard]] static ChannelPolicy policyFor(ChannelId id);

        lfs::rendering::CudaVulkanUploadStream upload_stream_;
        bool upload_stream_init_attempted_ = false;
        VulkanContext* teardown_context_ = nullptr;
        bool shut_down_ = false;

        // Built during prepareFrame decision pass; consumed by Phase 1–2 in prepareFrame.
        std::vector<ChannelUploadPlan> pending_uploads_;
        // After CUDA signal: GENERAL→READ_ONLY + publish deferred to recordFrameBarriers.
        std::vector<PendingFrameBarrier> pending_frame_barriers_;
        // Layout set to READ_ONLY at record time; rolled back if endFrame never submitted.
        std::vector<PendingLayoutCommit> pending_layout_commits_;

        // Scene-only external image path (VkSplat / compositor output).
        VkImage external_scene_image_ = VK_NULL_HANDLE;
        VkImageView external_scene_image_view_ = VK_NULL_HANDLE;
        VkImageLayout external_scene_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        glm::ivec2 external_scene_image_size_{0, 0};
        glm::ivec2 external_scene_image_alloc_size_{0, 0};
        bool external_scene_image_flip_y_ = false;
        std::uint64_t external_scene_image_generation_ = 0;

        VkSemaphore frame_completion_semaphore_ = VK_NULL_HANDLE;
        std::uint64_t frame_completion_value_ = 0;

        // Channels are heap-allocated so Channel can hold incomplete types.
        struct ChannelStorage;
        std::unique_ptr<ChannelStorage> channels_;

        // One pool across Scene / SplitRight / DepthBlit (keys partition by format).
        struct InteropPoolStorage;
        std::unique_ptr<InteropPoolStorage> interop_pool_;
    };

} // namespace lfs::vis
