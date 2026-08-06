/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "viewport_interop_service.hpp"

#include "core/logger.hpp"
#include "passes/vulkan_viewport_pass.hpp"
#include "window/vulkan_context.hpp"

#include <cassert>
#include <format>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lfs::vis {
    struct VulkanSceneInteropTarget {
        VulkanContext::ExternalImage image;
        VulkanContext::ExternalSemaphore semaphore;
        lfs::rendering::CudaVulkanInterop interop;
        glm::ivec2 size{0, 0};
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        std::uint64_t timeline_value = 0;
        std::uint64_t generation = 0;
        // Generation of the source content (renderer-supplied) most recently
        // copied into this slot's external image. Used to skip re-uploads when
        // the renderer returns the same logical image (cache HIT) even though
        // it allocated a fresh Tensor pointer.
        std::uint64_t uploaded_source_generation = 0;

        void destroy(VulkanContext& context) {
            if (!context.waitForImmediateSubmits()) {
                LOG_ERROR("Could not drain Vulkan interop transitions before target destruction: {}",
                          context.lastError());
            }
            interop.reset();
            context.destroyExternalSemaphore(semaphore);
            context.destroyExternalImage(image);
            size = {0, 0};
            layout = VK_IMAGE_LAYOUT_UNDEFINED;
            timeline_value = 0;
            uploaded_source_generation = 0;
            ++generation;
        }
    };

    struct ViewportInteropService::Channel {
        ChannelPolicy policy;
        std::vector<std::unique_ptr<VulkanSceneInteropTarget>> targets;
        std::shared_ptr<const lfs::core::Tensor> source_image;
        std::uint64_t source_generation = 0;
        glm::ivec2 source_size{0, 0};
        bool flip_y = false;
        bool disabled = false;
        VkImage published_image = VK_NULL_HANDLE;
        VkImageView published_image_view = VK_NULL_HANDLE;
        VkImageLayout published_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        std::uint64_t published_image_generation = 0;
    };

    struct ViewportInteropService::ChannelStorage {
        Channel scene;
        Channel split_right;
        Channel depth_blit;
    };

    ViewportInteropService::ChannelPolicy ViewportInteropService::policyFor(const ChannelId id) {
        switch (id) {
        case ChannelId::Scene:
            return ChannelPolicy{
                .id = ChannelId::Scene,
                .vk_format = VK_FORMAT_R8G8B8A8_UNORM,
                .cuda_format = lfs::rendering::CudaVulkanImageFormat::Rgba8Unorm,
                .debug_name_prefix = "scene",
                .failure_log_prefix = "Required Vulkan/CUDA viewport interop failed",
                .external_handle_early_out = true,
                .publishes_published = false,
                .log_timer_perf = true,
            };
        case ChannelId::SplitRight:
            return ChannelPolicy{
                .id = ChannelId::SplitRight,
                .vk_format = VK_FORMAT_R8G8B8A8_UNORM,
                .cuda_format = lfs::rendering::CudaVulkanImageFormat::Rgba8Unorm,
                .debug_name_prefix = "split_right",
                .failure_log_prefix = "Required Vulkan/CUDA split-view interop failed",
                .external_handle_early_out = false,
                .publishes_published = true,
                .log_timer_perf = false,
            };
        case ChannelId::DepthBlit:
            return ChannelPolicy{
                .id = ChannelId::DepthBlit,
                .vk_format = VK_FORMAT_R32_SFLOAT,
                .cuda_format = lfs::rendering::CudaVulkanImageFormat::R32Sfloat,
                .debug_name_prefix = "depth_blit",
                .failure_log_prefix = "Required Vulkan/CUDA depth-blit interop failed",
                .external_handle_early_out = false,
                .publishes_published = true,
                .log_timer_perf = false,
            };
        }
        return policyFor(ChannelId::Scene);
    }

    ViewportInteropService::ViewportInteropService()
        : channels_(std::make_unique<ChannelStorage>()) {
        channels_->scene.policy = policyFor(ChannelId::Scene);
        channels_->split_right.policy = policyFor(ChannelId::SplitRight);
        channels_->depth_blit.policy = policyFor(ChannelId::DepthBlit);
    }

    ViewportInteropService::~ViewportInteropService() {
        shutdown();
    }

    void ViewportInteropService::ensureUploadStream() {
        if (upload_stream_init_attempted_) {
            return;
        }
        upload_stream_init_attempted_ = true;
        if (!upload_stream_.init()) {
            LOG_ERROR("Could not create the non-blocking CUDA/Vulkan GUI upload stream: {}",
                      upload_stream_.lastError());
        }
    }

    void ViewportInteropService::clearPublished(Channel& channel) {
        channel.published_image = VK_NULL_HANDLE;
        channel.published_image_view = VK_NULL_HANDLE;
        channel.published_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        channel.published_image_generation = 0;
    }

    void ViewportInteropService::publishFromTarget(Channel& channel,
                                                   const VulkanSceneInteropTarget& target) {
        channel.published_image = target.image.image;
        channel.published_image_view = target.image.view;
        channel.published_image_layout = target.layout;
        channel.published_image_generation = target.generation;
    }

    bool ViewportInteropService::sourceOk(const Channel& channel) const {
        return channel.source_image &&
               channel.source_image->is_valid() &&
               channel.source_image->device() == lfs::core::Device::CUDA &&
               channel.source_size.x > 0 &&
               channel.source_size.y > 0;
    }

    void ViewportInteropService::setSceneImage(std::shared_ptr<const lfs::core::Tensor> image,
                                               const glm::ivec2 size,
                                               const bool flip_y,
                                               const std::uint64_t generation,
                                               const VkSemaphore completion_semaphore,
                                               const std::uint64_t completion_value) {
        auto& channel = channels_->scene;
        const bool target_changed =
            channel.source_image.get() != image.get() ||
            channel.source_size != size;
        if (target_changed) {
            channel.disabled = false;
        }
        external_scene_image_ = VK_NULL_HANDLE;
        external_scene_image_view_ = VK_NULL_HANDLE;
        external_scene_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        external_scene_image_size_ = {0, 0};
        external_scene_image_alloc_size_ = {0, 0};
        frame_completion_semaphore_ = completion_semaphore;
        frame_completion_value_ = completion_value;
        channel.source_image = std::move(image);
        channel.source_generation = generation;
        channel.source_size = size;
        channel.flip_y = flip_y;
    }

    void ViewportInteropService::setExternalSceneImage(const VkImage image,
                                                       const VkImageView image_view,
                                                       const VkImageLayout layout,
                                                       const glm::ivec2 size,
                                                       const bool flip_y,
                                                       const std::uint64_t generation,
                                                       const VkSemaphore completion_semaphore,
                                                       const std::uint64_t completion_value,
                                                       const glm::ivec2 alloc_size) {
        auto& channel = channels_->scene;
        channel.source_image.reset();
        channel.source_size = size;
        channel.flip_y = flip_y;
        external_scene_image_ = image;
        external_scene_image_view_ = image_view;
        external_scene_image_layout_ = layout;
        external_scene_image_size_ = size;
        external_scene_image_alloc_size_ =
            alloc_size.x > 0 && alloc_size.y > 0 ? alloc_size : size;
        external_scene_image_flip_y_ = flip_y;
        external_scene_image_generation_ = generation;
        frame_completion_semaphore_ = completion_semaphore;
        frame_completion_value_ = completion_value;
    }

    void ViewportInteropService::setSplitRightImage(std::shared_ptr<const lfs::core::Tensor> image,
                                                    const glm::ivec2 size,
                                                    const bool flip_y,
                                                    const std::uint64_t generation) {
        auto& channel = channels_->split_right;
        const bool target_changed =
            channel.source_image.get() != image.get() ||
            channel.source_size != size;
        if (target_changed) {
            channel.disabled = false;
        }
        channel.source_image = std::move(image);
        channel.source_generation = generation;
        channel.source_size = size;
        channel.flip_y = flip_y;
    }

    void ViewportInteropService::clearSplitRightImage() {
        auto& channel = channels_->split_right;
        channel.source_image.reset();
        channel.source_size = {0, 0};
        channel.flip_y = false;
        channel.source_generation = 0;
        clearPublished(channel);
    }

    void ViewportInteropService::setDepthBlitImage(std::shared_ptr<const lfs::core::Tensor> depth,
                                                   const glm::ivec2 size,
                                                   const std::uint64_t generation) {
        auto& channel = channels_->depth_blit;
        const bool target_changed =
            channel.source_image.get() != depth.get() ||
            channel.source_size != size;
        if (target_changed) {
            channel.disabled = false;
        }
        channel.source_image = std::move(depth);
        channel.source_generation = generation;
        channel.source_size = size;
    }

    void ViewportInteropService::clearDepthBlitImage() {
        auto& channel = channels_->depth_blit;
        channel.source_image.reset();
        channel.source_size = {0, 0};
        channel.source_generation = 0;
        clearPublished(channel);
    }

    void ViewportInteropService::resetChannel(Channel& channel) {
        // split_right / depth_blit clear published_* before the empty-vector early return;
        // scene does not.
        if (channel.policy.publishes_published) {
            clearPublished(channel);
        }
        if (channel.targets.empty()) {
            return;
        }
        // A previous frame's submit may still sample one of these slots; drain
        // before vkDestroyImage to avoid VK_ERROR_DEVICE_LOST.
        if (teardown_context_) {
            (void)teardown_context_->waitForSubmittedFrames();
        }
        for (auto& target : channel.targets) {
            if (!target) {
                continue;
            }
            if (teardown_context_) {
                target->destroy(*teardown_context_);
            } else {
                target->interop.reset();
            }
        }
        channel.targets.clear();
    }

    void ViewportInteropService::prepareChannel(VulkanContext& context,
                                                Channel& channel,
                                                const bool resize_deferring) {
        teardown_context_ = &context;

        const std::size_t frame_slot = context.currentFrameSlot();
        const bool slot_array_resize_needed =
            channel.targets.size() != context.framesInFlight();
        const bool frame_slot_in_range = frame_slot < channel.targets.size();

        ViewportInteropSlotInputs inputs{};
        inputs.disabled = channel.disabled;
        inputs.external_handle_early_out = channel.policy.external_handle_early_out;
        inputs.has_external_scene_image = external_scene_image_ != VK_NULL_HANDLE;
        inputs.source_ok = sourceOk(channel);
        inputs.publishes_published = channel.policy.publishes_published;
        inputs.resize_deferring = resize_deferring;
        inputs.slot_array_resize_needed = slot_array_resize_needed;
        inputs.frame_slot_in_range = frame_slot_in_range;
        if (frame_slot_in_range) {
            const auto& target_ptr = channel.targets[frame_slot];
            inputs.target_present = static_cast<bool>(target_ptr);
            inputs.target_size_matches = target_ptr && target_ptr->size == channel.source_size;
            inputs.target_interop_valid = target_ptr && target_ptr->interop.valid();
            inputs.target_layout_read_only =
                target_ptr &&
                target_ptr->layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            inputs.uploaded_source_generation =
                target_ptr ? target_ptr->uploaded_source_generation : 0;
        }
        inputs.source_generation = channel.source_generation;

        const auto fail_required_interop = [this, &channel](std::string message) -> void {
            channel.disabled = true;
            if (!upload_stream_.synchronize()) {
                message += std::format("; CUDA upload drain failed: {}",
                                       upload_stream_.lastError());
            }
            resetChannel(channel);
            LOG_ERROR("{}: {}", channel.policy.failure_log_prefix, message);
            throw std::runtime_error(std::move(message));
        };

        // Cache-HIT fast path: when nothing about the source image changed since the last
        // upload into THIS slot's interop target, there's no work to do — and crucially no
        // need to vkWaitForFences this slot. The previous unconditional wait was costing
        // ~kFrameDuration ms per frame (10–12 ms with kFramesInFlight=1) for no reason on
        // every renderer cache-HIT frame, which dominated gui_render time.
        const ViewportInteropDecision decision = decideViewportInteropEarly(inputs);
        if (decision.action == ViewportInteropAction::Disabled ||
            decision.action == ViewportInteropAction::ExternalSkip) {
            return;
        }
        if (decision.action == ViewportInteropAction::InvalidReset) {
            if (!channel.targets.empty()) {
                resetChannel(channel);
            }
            if (decision.clear_published) {
                clearPublished(channel);
            }
            return;
        }

        // Original order: stream validity is checked after the disabled / external / invalid-source
        // early outs and before cache-HIT, defer and slow-path work.
        if (!upload_stream_.valid()) {
            fail_required_interop("non-blocking CUDA upload stream is unavailable");
        }

        if (decision.action == ViewportInteropAction::CacheHit) {
            const auto& target_ptr = channel.targets[frame_slot];
            if (decision.publish_from_target && target_ptr) {
                publishFromTarget(channel, *target_ptr);
            }
            if (channel.policy.log_timer_perf) {
                LOG_PERF("interop slot={} cache-HIT-skip cur_gen={} layout={}",
                         frame_slot, channel.source_generation,
                         static_cast<int>(target_ptr->layout));
            }
            return;
        }
        if (decision.action == ViewportInteropAction::DeferBail) {
            if (decision.clear_published) {
                clearPublished(channel);
            }
            return;
        }

        // Slow path: we will write to the interop image (recreate, transition, or copy).
        // Wait for any in-flight GPU use of this slot to finish before we touch it.
        {
            std::optional<lfs::core::ScopedTimer> timer;
            if (channel.policy.log_timer_perf) {
                timer.emplace("interop.waitForCurrentFrameSlot",
                              lfs::core::LogLevel::Performance,
                              LFS_SOURCE_SITE_CURRENT());
            }
            if (!context.waitForCurrentFrameSlot()) {
                fail_required_interop(std::format("frame slot wait failed: {}", context.lastError()));
            }
        }

        if (slot_array_resize_needed) {
            resetChannel(channel);
            channel.targets.resize(context.framesInFlight());
        }
        if (frame_slot >= channel.targets.size()) {
            fail_required_interop(std::format("invalid frame slot {}", frame_slot));
        }
        auto& target_ptr = channel.targets[frame_slot];
        const auto reset_frame_target = [&]() {
            if (target_ptr) {
                target_ptr->destroy(context);
                target_ptr.reset();
            }
        };

        const glm::ivec2 target_size = channel.source_size;
        const bool recreate =
            !target_ptr ||
            target_ptr->size != target_size ||
            !target_ptr->interop.valid();
        if (channel.policy.log_timer_perf) {
            LOG_PERF("interop slot={} recreate={} cur_gen={} uploaded_gen={} layout={}",
                     frame_slot, recreate,
                     channel.source_generation,
                     target_ptr ? target_ptr->uploaded_source_generation : 0,
                     target_ptr ? static_cast<int>(target_ptr->layout) : -1);
        }
        if (recreate) {
            reset_frame_target();
            auto target = std::make_unique<VulkanSceneInteropTarget>();
            const VkExtent2D extent{
                static_cast<std::uint32_t>(target_size.x),
                static_cast<std::uint32_t>(target_size.y),
            };
            if (!context.createExternalImage(extent,
                                             channel.policy.vk_format,
                                             target->image,
                                             "vulkan.gui.interop_image",
                                             std::format("{}.frame{}",
                                                         channel.policy.debug_name_prefix,
                                                         frame_slot)) ||
                !context.createExternalTimelineSemaphore(0, target->semaphore, "vulkan.gui.interop_semaphore")) {
                const std::string error = std::format("target creation failed: {}", context.lastError());
                if (target->image.image != VK_NULL_HANDLE || target->semaphore.semaphore != VK_NULL_HANDLE) {
                    target->destroy(context);
                }
                fail_required_interop(error);
            }
            const std::uint64_t vulkan_ready_value = ++target->timeline_value;
            if (!context.transitionImageLayoutImmediate(target->image.image,
                                                        VK_IMAGE_LAYOUT_UNDEFINED,
                                                        VK_IMAGE_LAYOUT_GENERAL,
                                                        VulkanContext::ImmediateTransitionOptions::signalAt(
                                                            {target->semaphore.semaphore, vulkan_ready_value}))) {
                const std::string error = std::format("image initialization failed: {}", context.lastError());
                target->destroy(context);
                fail_required_interop(error);
            }
            // Complete the one-time Vulkan initialization before exporting the
            // timeline to CUDA. Later handoffs remain asynchronous, but no
            // external producer may advance this semaphore past the pending
            // Vulkan signal that establishes its initial image ownership.
            if (!context.waitForImmediateSubmits()) {
                const std::string error = std::format(
                    "image initialization handoff failed: {}", context.lastError());
                target->destroy(context);
                fail_required_interop(error);
            }

            const auto memory_handle = context.releaseExternalImageNativeHandle(target->image);
            const auto semaphore_handle = context.releaseExternalSemaphoreNativeHandle(target->semaphore);
            lfs::rendering::CudaVulkanExternalImageImport image_import{
                .memory_handle = memory_handle,
                .allocation_size = static_cast<std::size_t>(target->image.allocation_size),
                .extent = {.width = extent.width, .height = extent.height},
                .format = channel.policy.cuda_format,
                .dedicated_allocation = context.externalMemoryDedicatedAllocationEnabled(),
            };
            lfs::rendering::CudaVulkanExternalSemaphoreImport semaphore_import{
                .semaphore_handle = semaphore_handle,
                .initial_value = 0,
            };
            if (!target->interop.init(image_import, semaphore_import)) {
                const std::string error = std::format("CUDA import failed: {}", target->interop.lastError());
                target->destroy(context);
                fail_required_interop(error);
            }
            target->size = target_size;
            target->layout = VK_IMAGE_LAYOUT_GENERAL;
            target_ptr = std::move(target);
            switch (channel.policy.id) {
            case ChannelId::Scene:
                LOG_INFO("Vulkan/CUDA viewport interop target initialized for frame slot {}: {}x{}",
                         frame_slot, target_size.x, target_size.y);
                break;
            case ChannelId::SplitRight:
                LOG_INFO("Vulkan/CUDA split-view right-panel interop initialized for slot {}: {}x{}",
                         frame_slot, target_size.x, target_size.y);
                break;
            case ChannelId::DepthBlit:
                LOG_INFO("Vulkan/CUDA depth-blit interop initialized for slot {}: {}x{}",
                         frame_slot, target_size.x, target_size.y);
                break;
            }
        }

        auto& target = *target_ptr;
        // Skip the upload (and the queue-blocking layout transitions inside it)
        // when this slot already holds the same content. Renderer cache-HIT
        // frames keep image_generation stable while alternating tensor pointers,
        // so identity-by-pointer is unsafe — use the source generation.
        if (channel.source_generation != 0 &&
            target.uploaded_source_generation == channel.source_generation &&
            target.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            if (channel.policy.publishes_published) {
                publishFromTarget(channel, target);
            }
            return;
        }

        if (target.layout != VK_IMAGE_LAYOUT_GENERAL) {
            std::optional<lfs::core::ScopedTimer> timer;
            if (channel.policy.log_timer_perf) {
                timer.emplace("interop.transition_to_GENERAL",
                              lfs::core::LogLevel::Performance,
                              LFS_SOURCE_SITE_CURRENT());
            }
            const std::uint64_t vulkan_ready_value = ++target.timeline_value;
            if (!context.transitionImageLayoutImmediate(target.image.image,
                                                        target.layout,
                                                        VK_IMAGE_LAYOUT_GENERAL,
                                                        VulkanContext::ImmediateTransitionOptions::signalAt(
                                                            {target.semaphore.semaphore, vulkan_ready_value}))) {
                fail_required_interop(std::format("image transition to GENERAL failed: {}", context.lastError()));
            }
            target.layout = VK_IMAGE_LAYOUT_GENERAL;
        }

        assert(target.layout == VK_IMAGE_LAYOUT_GENERAL &&
               "CUDA surf2Dwrite requires VK_IMAGE_LAYOUT_GENERAL");
        {
            std::optional<lfs::core::ScopedTimer> timer;
            if (channel.policy.log_timer_perf) {
                timer.emplace("interop.copyTensorToSurface",
                              lfs::core::LogLevel::Performance,
                              LFS_SOURCE_SITE_CURRENT());
            }
            if (!target.interop.wait(target.timeline_value, upload_stream_.stream())) {
                fail_required_interop(std::format("CUDA wait for Vulkan image release failed: {}",
                                                  target.interop.lastError()));
            }
            if (!target.interop.copyTensorToSurface(*channel.source_image, upload_stream_.stream())) {
                fail_required_interop(std::format("CUDA copy failed: {}", target.interop.lastError()));
            }
        }
        const std::uint64_t signal_value = ++target.timeline_value;
        {
            std::optional<lfs::core::ScopedTimer> timer;
            if (channel.policy.log_timer_perf) {
                timer.emplace("interop.cuda_signal",
                              lfs::core::LogLevel::Performance,
                              LFS_SOURCE_SITE_CURRENT());
            }
            if (!target.interop.signal(signal_value, upload_stream_.stream())) {
                fail_required_interop(std::format("CUDA signal failed: {}", target.interop.lastError()));
            }
        }
        {
            std::optional<lfs::core::ScopedTimer> timer;
            if (channel.policy.log_timer_perf) {
                timer.emplace("interop.transition_to_READ_ONLY",
                              lfs::core::LogLevel::Performance,
                              LFS_SOURCE_SITE_CURRENT());
            }
            if (!context.transitionImageLayoutImmediate(target.image.image,
                                                        VK_IMAGE_LAYOUT_GENERAL,
                                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                        VulkanContext::ImmediateTransitionOptions::waitOn(
                                                            {target.semaphore.semaphore, signal_value},
                                                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT))) {
                fail_required_interop(std::format("Vulkan wait for CUDA signal failed: {}", context.lastError()));
            }
        }
        target.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        target.uploaded_source_generation = channel.source_generation;
        ++target.generation;
        if (channel.policy.publishes_published) {
            publishFromTarget(channel, target);
        }
    }

    void ViewportInteropService::prepareFrame(VulkanContext& context, const bool resize_deferring) {
        ensureUploadStream();
        prepareChannel(context, channels_->scene, resize_deferring);
        prepareChannel(context, channels_->split_right, resize_deferring);
        prepareChannel(context, channels_->depth_blit, resize_deferring);
    }

    void ViewportInteropService::bindViewportParams(VulkanViewportPassParams& params,
                                                    const std::size_t frame_slot,
                                                    const bool export_locked,
                                                    const bool resize_deferring) const {
        const auto& scene = channels_->scene;
        params.scene_image = scene.source_image;
        params.scene_image_size = scene.source_size;
        params.scene_image_flip_y = scene.flip_y;
        if (external_scene_image_ != VK_NULL_HANDLE &&
            external_scene_image_view_ != VK_NULL_HANDLE &&
            external_scene_image_size_.x > 0 &&
            external_scene_image_size_.y > 0) {
            params.scene_image_size = external_scene_image_size_;
            params.scene_image_alloc_size =
                external_scene_image_alloc_size_.x > 0 && external_scene_image_alloc_size_.y > 0
                    ? external_scene_image_alloc_size_
                    : external_scene_image_size_;
            params.scene_image_flip_y = external_scene_image_flip_y_;
            params.external_scene_image = external_scene_image_;
            params.external_scene_image_view = external_scene_image_view_;
            params.external_scene_image_layout = external_scene_image_layout_;
            params.external_scene_image_generation = external_scene_image_generation_;
        } else {
            params.scene_image_alloc_size =
                params.scene_image_alloc_size.x > 0 && params.scene_image_alloc_size.y > 0
                    ? params.scene_image_alloc_size
                    : params.scene_image_size;
        }
        const auto bind_cached_interop_slot = [&](const std::size_t slot) -> bool {
            if (slot >= scene.targets.size()) {
                return false;
            }
            const auto& target = scene.targets[slot];
            if (!target ||
                !target->interop.valid() ||
                target->layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ||
                target->size != params.scene_image_size ||
                scene.source_generation == 0 ||
                target->uploaded_source_generation != scene.source_generation) {
                return false;
            }

            params.external_scene_image = target->image.image;
            params.external_scene_image_view = target->image.view;
            params.external_scene_image_layout = target->layout;
            params.external_scene_image_generation = target->generation;
            return true;
        };
        if (params.external_scene_image == VK_NULL_HANDLE) {
            const bool bound_current_slot = bind_cached_interop_slot(frame_slot);
            if (!bound_current_slot && export_locked) {
                // Export mode freezes the viewport and skips new CUDA/Vulkan interop uploads.
                // Reuse any already-prepared slot so multi-buffered frames keep the same image.
                for (std::size_t slot = 0; slot < scene.targets.size(); ++slot) {
                    if (slot != frame_slot && bind_cached_interop_slot(slot)) {
                        break;
                    }
                }
            }
            params.preserve_scene_image_binding =
                params.external_scene_image == VK_NULL_HANDLE &&
                params.scene_image &&
                resize_deferring;
        }

        const auto& depth = channels_->depth_blit;
        if (depth.published_image_view != VK_NULL_HANDLE) {
            params.depth_blit.external_image_view = depth.published_image_view;
            params.depth_blit.external_image_generation = depth.published_image_generation;
        }

        // Stitch in CUDA/Vulkan interop views: left reuses the existing scene
        // interop slot; right has its own parallel slot. When set, the split-view
        // pass binds these directly and skips the CPU staging upload.
        if (params.split_view.enabled) {
            if (params.external_scene_image_view != VK_NULL_HANDLE) {
                params.split_view.left.external_image_view = params.external_scene_image_view;
                params.split_view.left.external_image_generation = params.external_scene_image_generation;
            }
            const auto& split = channels_->split_right;
            if (split.published_image_view != VK_NULL_HANDLE) {
                params.split_view.right.external_image_view = split.published_image_view;
                params.split_view.right.external_image_generation = split.published_image_generation;
            }
        }
    }

    ViewportInteropService::FrameCompletion ViewportInteropService::frameCompletion() const {
        return {frame_completion_semaphore_, frame_completion_value_};
    }

    void ViewportInteropService::shutdown(VulkanContext* context) {
        if (shut_down_) {
            return;
        }
        if (context) {
            teardown_context_ = context;
        }
        if (!upload_stream_.synchronize()) {
            LOG_WARN("CUDA/Vulkan GUI upload stream synchronization failed during shutdown: {}",
                     upload_stream_.lastError());
        }
        resetChannel(channels_->scene);
        resetChannel(channels_->split_right);
        resetChannel(channels_->depth_blit);
        upload_stream_.reset();
        channels_->scene.source_image.reset();
        channels_->split_right.source_image.reset();
        channels_->depth_blit.source_image.reset();
        external_scene_image_ = VK_NULL_HANDLE;
        external_scene_image_view_ = VK_NULL_HANDLE;
        external_scene_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        external_scene_image_size_ = {0, 0};
        external_scene_image_alloc_size_ = {0, 0};
        frame_completion_semaphore_ = VK_NULL_HANDLE;
        frame_completion_value_ = 0;
        shut_down_ = true;
        teardown_context_ = nullptr;
    }

} // namespace lfs::vis
