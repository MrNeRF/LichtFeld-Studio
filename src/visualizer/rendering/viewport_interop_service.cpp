/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "viewport_interop_service.hpp"

#include "core/logger.hpp"
#include "output_image_pool.hpp"
#include "passes/vulkan_viewport_pass.hpp"
#include "window/vulkan_context.hpp"

#include <algorithm>
#include <cassert>
#include <format>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lfs::vis {
    namespace {
        constexpr VkImageUsageFlags kInteropExternalImageUsage =
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        [[nodiscard]] glm::ivec2 bucketExtent(const glm::ivec2 valid) noexcept {
            return {
                static_cast<int>(ceil64(static_cast<std::uint32_t>(std::max(valid.x, 0)))),
                static_cast<int>(ceil64(static_cast<std::uint32_t>(std::max(valid.y, 0)))),
            };
        }
    } // namespace

    // Pooled unit: image + CUDA import + timeline ride together. Reuse skips re-import.
    struct ViewportInteropService::PooledInteropUnit {
        VulkanContext::ExternalImage image;
        VulkanContext::ExternalSemaphore semaphore;
        lfs::rendering::CudaVulkanInterop interop;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        std::uint64_t timeline_value = 0;
    };

    // Per-slot binding into the shared interop pool (payload stays pool-owned).
    struct ViewportInteropService::VulkanSceneInteropTarget {
        std::uint64_t pool_serial = 0;
        PooledInteropUnit* unit = nullptr;
        glm::ivec2 valid_size{0, 0};
        glm::ivec2 alloc_size{0, 0};
        std::uint64_t generation = 0;
        // Generation of the source content (renderer-supplied) most recently
        // copied into this slot's external image. Used to skip re-uploads when
        // the renderer returns the same logical image (cache HIT) even though
        // it allocated a fresh Tensor pointer.
        std::uint64_t uploaded_source_generation = 0;
    };

    struct ViewportInteropService::Channel {
        ChannelPolicy policy;
        std::vector<std::unique_ptr<VulkanSceneInteropTarget>> targets;
        std::shared_ptr<const lfs::core::Tensor> source_image;
        std::uint64_t source_generation = 0;
        glm::ivec2 source_size{0, 0};
        bool flip_y = false;
        bool disabled = false;
        VkImageView published_image_view = VK_NULL_HANDLE;
        std::uint64_t published_image_generation = 0;
        glm::ivec2 published_valid_size{0, 0};
        glm::ivec2 published_alloc_size{0, 0};
    };

    struct ViewportInteropService::ChannelStorage {
        Channel scene;
        Channel split_right;
        Channel depth_blit;
    };

    struct ViewportInteropService::InteropPoolStorage {
        GpuResourcePool<PooledInteropUnit> pool{
            [](const PooledInteropUnit& unit) {
                return static_cast<std::size_t>(unit.image.allocation_size);
            }};
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
        : channels_(std::make_unique<ChannelStorage>()),
          interop_pool_(std::make_unique<InteropPoolStorage>()) {
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
        channel.published_image_view = VK_NULL_HANDLE;
        channel.published_image_generation = 0;
        channel.published_valid_size = {0, 0};
        channel.published_alloc_size = {0, 0};
    }

    void ViewportInteropService::publishFromTarget(Channel& channel,
                                                   const VulkanSceneInteropTarget& target) {
        if (!target.unit) {
            clearPublished(channel);
            return;
        }
        channel.published_image_view = target.unit->image.view;
        channel.published_image_generation = target.generation;
        channel.published_valid_size = target.valid_size;
        channel.published_alloc_size = target.alloc_size;
    }

    bool ViewportInteropService::sourceOk(const Channel& channel) const {
        return channel.source_image &&
               channel.source_image->is_valid() &&
               channel.source_image->device() == lfs::core::Device::CUDA &&
               channel.source_size.x > 0 &&
               channel.source_size.y > 0;
    }

    void ViewportInteropService::drainInteropPool(VulkanContext& context, const bool force) {
        if (!interop_pool_) {
            return;
        }
        const auto destroy_fn = [&context](PooledInteropUnit& unit) {
            // Same sequence as the former VulkanSceneInteropTarget::destroy: rare path
            // (trim/shutdown/force). Includes waitForImmediateSubmits.
            if (!context.waitForImmediateSubmits()) {
                LOG_ERROR("Could not drain Vulkan interop transitions before pooled unit destruction: {}",
                          context.lastError());
            }
            unit.interop.reset();
            context.destroyExternalSemaphore(unit.semaphore);
            context.destroyExternalImage(unit.image);
            unit.layout = VK_IMAGE_LAYOUT_UNDEFINED;
            unit.timeline_value = 0;
        };
        auto producer_pred = [&context](const PooledInteropUnit& unit, const std::uint64_t value) {
            if (value == 0) {
                return true;
            }
            if (unit.semaphore.semaphore == VK_NULL_HANDLE) {
                return true;
            }
            std::uint64_t counter = 0;
            if (!context.getTimelineSemaphoreCounterValue(unit.semaphore.semaphore, counter)) {
                // Non-blocking drain: treat query failure as not-yet-done.
                return false;
            }
            return counter >= value;
        };
        const std::uint64_t retired_serial = context.retiredFrameSubmitSerial();
        auto consumer_pred = [retired_serial](const std::uint64_t serial) {
            return serial <= retired_serial;
        };
        interop_pool_->pool.drain(force, producer_pred, consumer_pred, destroy_fn);
    }

    void ViewportInteropService::releaseSlotTarget(VulkanContext& context,
                                                   VulkanSceneInteropTarget& target) {
        // Pending vectors must never outlive the units they reference.
        if (target.unit != nullptr) {
            std::erase_if(pending_layout_commits_, [&](const PendingLayoutCommit& commit) {
                return commit.unit == target.unit;
            });
        }
        if (target.pool_serial == 0 || !interop_pool_) {
            target = {};
            return;
        }
        // Layout + timeline live on the pooled unit (already up to date).
        const std::uint64_t producer =
            target.unit != nullptr ? target.unit->timeline_value : 0;
        const std::uint64_t consumer = context.lastFrameSubmitSerial();
        interop_pool_->pool.release(target.pool_serial, producer, consumer);
        target = {};
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
        // Pending vectors must never outlive the units they reference.
        std::erase_if(pending_layout_commits_, [&](const PendingLayoutCommit& commit) {
            return commit.channel == &channel;
        });
        // split_right / depth_blit clear published_* before the empty-vector early return;
        // scene does not.
        if (channel.policy.publishes_published) {
            clearPublished(channel);
        }
        if (channel.targets.empty()) {
            return;
        }
        // A previous frame's submit may still sample one of these slots; drain
        // before retiring units so consumer serial is meaningful.
        if (teardown_context_) {
            (void)teardown_context_->waitForSubmittedFrames();
            for (auto& target : channel.targets) {
                if (target) {
                    releaseSlotTarget(*teardown_context_, *target);
                }
            }
            // Force-drain retired/free units (destroy path includes waitForImmediateSubmits).
            drainInteropPool(*teardown_context_, /*force=*/true);
        } else {
            // No context: drop CUDA side only; Vulkan objects orphaned (same as prior interop.reset).
            for (auto& target : channel.targets) {
                if (target && target->unit) {
                    target->unit->interop.reset();
                }
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
        const glm::ivec2 source_bucket = bucketExtent(channel.source_size);

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
            inputs.target_present = static_cast<bool>(target_ptr) && target_ptr->unit != nullptr;
            inputs.target_size_matches =
                target_ptr && target_ptr->unit && target_ptr->alloc_size == source_bucket;
            inputs.target_valid_size_matches =
                target_ptr && target_ptr->valid_size == channel.source_size;
            inputs.target_interop_valid =
                target_ptr && target_ptr->unit && target_ptr->unit->interop.valid();
            inputs.target_layout_read_only =
                target_ptr && target_ptr->unit &&
                target_ptr->unit->layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
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
                         target_ptr && target_ptr->unit
                             ? static_cast<int>(target_ptr->unit->layout)
                             : -1);
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
        // waitForCurrentFrameSlot protects the CURRENT unit about to be mutated (layout
        // transition + CUDA write) while a prior GUI frame may still sample this FIF slot.
        // Pool retirement covers OLD units released on bucket change — not the live unit.
        // Keep the wait: it is not redundant with retirement.
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
        if (!target_ptr) {
            target_ptr = std::make_unique<VulkanSceneInteropTarget>();
        }

        const glm::ivec2 valid_size = channel.source_size;
        const glm::ivec2 alloc_size = source_bucket;
        const bool recreate =
            target_ptr->unit == nullptr ||
            target_ptr->alloc_size != alloc_size ||
            !target_ptr->unit->interop.valid();
        if (channel.policy.log_timer_perf) {
            LOG_PERF("interop slot={} recreate={} cur_gen={} uploaded_gen={} layout={} valid={}x{} alloc={}x{}",
                     frame_slot, recreate,
                     channel.source_generation,
                     target_ptr->uploaded_source_generation,
                     target_ptr->unit ? static_cast<int>(target_ptr->unit->layout) : -1,
                     valid_size.x, valid_size.y, alloc_size.x, alloc_size.y);
        }

        if (recreate) {
            // Retire previous unit into the pool (Live→Retired); no destroy on this path.
            if (target_ptr->pool_serial != 0) {
                releaseSlotTarget(context, *target_ptr);
            }

            const VkExtent2D extent{
                static_cast<std::uint32_t>(alloc_size.x),
                static_cast<std::uint32_t>(alloc_size.y),
            };
            const GpuResourcePoolKey key{
                .format = channel.policy.vk_format,
                .extent = extent,
                .usage = kInteropExternalImageUsage,
                .external = true,
            };

            if (auto hit = interop_pool_->pool.acquire(key)) {
                // Pool hit: reuse unit — no create, no UNDEFINED→GENERAL, no wait, no re-import.
                target_ptr->pool_serial = hit->acquisition_serial;
                target_ptr->unit = hit->payload;
                // Timeline continues from the stored unit (strictly monotonic — never reset).
                // Layout continues from the stored layout; per-frame path transitions tracked→GENERAL.
                // NEVER emit UNDEFINED-source transition for a reused unit.
            } else {
                // Cold path: create + one-shot init + CUDA import, then register as Live.
                PooledInteropUnit created{};
                if (!context.createExternalImage(extent,
                                                 channel.policy.vk_format,
                                                 created.image,
                                                 "vulkan.gui.interop_image",
                                                 std::format("{}.frame{}",
                                                             channel.policy.debug_name_prefix,
                                                             frame_slot)) ||
                    !context.createExternalTimelineSemaphore(0, created.semaphore,
                                                             "vulkan.gui.interop_semaphore")) {
                    const std::string error = std::format("target creation failed: {}", context.lastError());
                    if (created.image.image != VK_NULL_HANDLE ||
                        created.semaphore.semaphore != VK_NULL_HANDLE) {
                        if (!context.waitForImmediateSubmits()) {
                            LOG_ERROR("Could not drain before interop create-fail cleanup: {}",
                                      context.lastError());
                        }
                        created.interop.reset();
                        context.destroyExternalSemaphore(created.semaphore);
                        context.destroyExternalImage(created.image);
                    }
                    fail_required_interop(error);
                }
                const std::uint64_t vulkan_ready_value = ++created.timeline_value;
                if (!context.transitionImageLayoutImmediate(
                        created.image.image,
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VulkanContext::ImmediateTransitionOptions::signalAt(
                            {created.semaphore.semaphore, vulkan_ready_value}))) {
                    const std::string error =
                        std::format("image initialization failed: {}", context.lastError());
                    if (!context.waitForImmediateSubmits()) {
                        LOG_ERROR("Could not drain before interop init-fail cleanup: {}",
                                  context.lastError());
                    }
                    created.interop.reset();
                    context.destroyExternalSemaphore(created.semaphore);
                    context.destroyExternalImage(created.image);
                    fail_required_interop(error);
                }
                // Complete the one-time Vulkan initialization before exporting the
                // timeline to CUDA. Later handoffs remain asynchronous, but no
                // external producer may advance this semaphore past the pending
                // Vulkan signal that establishes its initial image ownership.
                if (!context.waitForImmediateSubmits()) {
                    const std::string error = std::format(
                        "image initialization handoff failed: {}", context.lastError());
                    created.interop.reset();
                    context.destroyExternalSemaphore(created.semaphore);
                    context.destroyExternalImage(created.image);
                    fail_required_interop(error);
                }

                // Handle rule: release native handles exactly once per physical unit (first import).
                const auto memory_handle = context.releaseExternalImageNativeHandle(created.image);
                const auto semaphore_handle =
                    context.releaseExternalSemaphoreNativeHandle(created.semaphore);
                // Windows allocation-info rule: allocation_size + dedicated flag are captured
                // at import time and ride with the unit; reuse skips re-import so they stay
                // coherent by construction.
                lfs::rendering::CudaVulkanExternalImageImport image_import{
                    .memory_handle = memory_handle,
                    .allocation_size = static_cast<std::size_t>(created.image.allocation_size),
                    .extent = {.width = extent.width, .height = extent.height},
                    .format = channel.policy.cuda_format,
                    .dedicated_allocation = context.externalMemoryDedicatedAllocationEnabled(),
                };
                lfs::rendering::CudaVulkanExternalSemaphoreImport semaphore_import{
                    .semaphore_handle = semaphore_handle,
                    .initial_value = 0,
                };
                if (!created.interop.init(image_import, semaphore_import)) {
                    const std::string error =
                        std::format("CUDA import failed: {}", created.interop.lastError());
                    if (!context.waitForImmediateSubmits()) {
                        LOG_ERROR("Could not drain before interop import-fail cleanup: {}",
                                  context.lastError());
                    }
                    created.interop.reset();
                    context.destroyExternalSemaphore(created.semaphore);
                    context.destroyExternalImage(created.image);
                    fail_required_interop(error);
                }
                created.layout = VK_IMAGE_LAYOUT_GENERAL;

                auto reg = interop_pool_->pool.registerCreated(key, std::move(created));
                target_ptr->pool_serial = reg.acquisition_serial;
                target_ptr->unit = reg.payload;

                switch (channel.policy.id) {
                case ChannelId::Scene:
                    LOG_INFO("Vulkan/CUDA viewport interop target initialized for frame slot {}: valid {}x{} alloc {}x{}",
                             frame_slot, valid_size.x, valid_size.y, alloc_size.x, alloc_size.y);
                    break;
                case ChannelId::SplitRight:
                    LOG_INFO("Vulkan/CUDA split-view right-panel interop initialized for slot {}: valid {}x{} alloc {}x{}",
                             frame_slot, valid_size.x, valid_size.y, alloc_size.x, alloc_size.y);
                    break;
                case ChannelId::DepthBlit:
                    LOG_INFO("Vulkan/CUDA depth-blit interop initialized for slot {}: valid {}x{} alloc {}x{}",
                             frame_slot, valid_size.x, valid_size.y, alloc_size.x, alloc_size.y);
                    break;
                }
            }

            target_ptr->valid_size = valid_size;
            target_ptr->alloc_size = alloc_size;
            // Force re-upload after acquire/create (content/size identity reset).
            target_ptr->uploaded_source_generation = 0;
        } else if (target_ptr->valid_size != valid_size) {
            // Within-bucket size change: update valid only, force re-upload, no recreate.
            target_ptr->valid_size = valid_size;
            target_ptr->uploaded_source_generation = 0;
        }

        auto& target = *target_ptr;
        assert(target.unit != nullptr);
        auto& unit = *target.unit;

        // Skip the upload when this slot already holds the same content.
        // Cache-hit publish is unchanged (immediate, not deferred to frame barriers).
        if (channel.source_generation != 0 &&
            target.uploaded_source_generation == channel.source_generation &&
            unit.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            if (channel.policy.publishes_published) {
                publishFromTarget(channel, target);
            }
            return;
        }

        // Defer Phase 1–2 work to prepareFrame so →GENERAL transitions coalesce.
        pending_uploads_.push_back(ChannelUploadPlan{
            .channel = &channel,
            .target = &target,
        });
    }

    void ViewportInteropService::rollbackUnsubmittedLayoutCommits(VulkanContext& context) {
        // If endFrame never successfully submitted after recordFrameBarriers, the
        // GENERAL→READ_ONLY barrier never ran on-device; restore tracked layout.
        const std::uint64_t successful = context.lastSuccessfulFrameSubmitSerial();
        for (const auto& commit : pending_layout_commits_) {
            if (commit.unit == nullptr) {
                continue;
            }
            if (successful <= commit.frame_submit_marker) {
                commit.unit->layout = VK_IMAGE_LAYOUT_GENERAL;
                if (commit.channel != nullptr && commit.channel->policy.publishes_published) {
                    clearPublished(*commit.channel);
                }
            }
        }
        pending_layout_commits_.clear();
    }

    void ViewportInteropService::syncUnsubmittedLayoutCommits(VulkanContext& context) {
        teardown_context_ = &context;
        // Prior frame: unrecorded barriers (beginFrame failed) leave layout GENERAL.
        pending_frame_barriers_.clear();
        rollbackUnsubmittedLayoutCommits(context);
    }

    void ViewportInteropService::prepareFrame(VulkanContext& context, const bool resize_deferring) {
        teardown_context_ = &context;
        // Frame accounting starts here (prepare runs before beginFrame).
        context.resetImmediateSubmitsThisFrame();
        ensureUploadStream();
        // Non-blocking drain once per prepareFrame (retired → free when safe).
        drainInteropPool(context, /*force=*/false);

        syncUnsubmittedLayoutCommits(context);

        pending_uploads_.clear();
        prepareChannel(context, channels_->scene, resize_deferring);
        prepareChannel(context, channels_->split_right, resize_deferring);
        prepareChannel(context, channels_->depth_blit, resize_deferring);

        // Phase 1 — one coalesced pre-frame →GENERAL submit for every unit that needs it.
        std::vector<VulkanContext::ImmediateLayoutTransition> general_transitions;
        general_transitions.reserve(pending_uploads_.size());
        std::vector<PooledInteropUnit*> units_transitioned_to_general;
        units_transitioned_to_general.reserve(pending_uploads_.size());
        {
            std::optional<lfs::core::ScopedTimer> timer;
            // Scene-channel timer name retained when any scene upload is present.
            for (const auto& plan : pending_uploads_) {
                if (plan.channel != nullptr && plan.channel->policy.log_timer_perf) {
                    timer.emplace("interop.transition_to_GENERAL",
                                  lfs::core::LogLevel::Performance,
                                  LFS_SOURCE_SITE_CURRENT());
                    break;
                }
            }
            for (const auto& plan : pending_uploads_) {
                assert(plan.channel != nullptr && plan.target != nullptr && plan.target->unit != nullptr);
                auto& unit = *plan.target->unit;
                if (unit.layout == VK_IMAGE_LAYOUT_GENERAL) {
                    continue;
                }
                // Reused units start from a defined tracked layout (typically READ_ONLY).
                // Never transition from UNDEFINED here — only cold create uses UNDEFINED→GENERAL.
                const std::uint64_t vulkan_ready_value = ++unit.timeline_value;
                general_transitions.push_back(VulkanContext::ImmediateLayoutTransition{
                    .image = unit.image.image,
                    .old_layout = unit.layout,
                    .new_layout = VK_IMAGE_LAYOUT_GENERAL,
                    .options = VulkanContext::ImmediateTransitionOptions::signalAt(
                        {unit.semaphore.semaphore, vulkan_ready_value}),
                });
                units_transitioned_to_general.push_back(&unit);
            }
            if (!general_transitions.empty()) {
                if (!context.transitionImageLayoutsImmediate(general_transitions)) {
                    // Timeline values were advanced for the batch before submit; on failure
                    // those values never signal. Reset every planned channel so units (and
                    // their counters) are destroyed rather than left waiting forever.
                    const std::string message = std::format(
                        "batched image transition to GENERAL failed: {}", context.lastError());
                    if (!upload_stream_.synchronize()) {
                        // best-effort
                    }
                    for (const auto& plan : pending_uploads_) {
                        if (plan.channel == nullptr) {
                            continue;
                        }
                        plan.channel->disabled = true;
                        resetChannel(*plan.channel);
                        LOG_ERROR("{}: {}", plan.channel->policy.failure_log_prefix, message);
                    }
                    pending_uploads_.clear();
                    throw std::runtime_error(message);
                }
                for (auto* unit : units_transitioned_to_general) {
                    unit->layout = VK_IMAGE_LAYOUT_GENERAL;
                }
            }
        }

        // Phase 2 — CUDA wait/copy/signal per channel (shared upload stream, unchanged order).
        for (const auto& plan : pending_uploads_) {
            auto& channel = *plan.channel;
            auto& target = *plan.target;
            auto& unit = *target.unit;

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

            assert(unit.layout == VK_IMAGE_LAYOUT_GENERAL &&
                   "CUDA surf2Dwrite requires VK_IMAGE_LAYOUT_GENERAL");
            {
                std::optional<lfs::core::ScopedTimer> timer;
                if (channel.policy.log_timer_perf) {
                    timer.emplace("interop.copyTensorToSurface",
                                  lfs::core::LogLevel::Performance,
                                  LFS_SOURCE_SITE_CURRENT());
                }
                if (!unit.interop.wait(unit.timeline_value, upload_stream_.stream())) {
                    fail_required_interop(std::format("CUDA wait for Vulkan image release failed: {}",
                                                      unit.interop.lastError()));
                }
                if (!unit.interop.copyTensorToSurface(*channel.source_image, upload_stream_.stream())) {
                    fail_required_interop(std::format("CUDA copy failed: {}", unit.interop.lastError()));
                }
            }
            const std::uint64_t signal_value = ++unit.timeline_value;
            {
                std::optional<lfs::core::ScopedTimer> timer;
                if (channel.policy.log_timer_perf) {
                    timer.emplace("interop.cuda_signal",
                                  lfs::core::LogLevel::Performance,
                                  LFS_SOURCE_SITE_CURRENT());
                }
                if (!unit.interop.signal(signal_value, upload_stream_.stream())) {
                    fail_required_interop(std::format("CUDA signal failed: {}", unit.interop.lastError()));
                }
            }
            // GENERAL→READ_ONLY + publish move to recordFrameBarriers (frame CB).
            pending_frame_barriers_.push_back(PendingFrameBarrier{
                .unit = &unit,
                .target = &target,
                .channel = &channel,
                .cuda_signal_value = signal_value,
                .source_generation = channel.source_generation,
            });
        }
        pending_uploads_.clear();
    }

    void ViewportInteropService::recordFrameBarriers(VkCommandBuffer frame_cb,
                                                     VulkanContext& context) {
        if (pending_frame_barriers_.empty()) {
            return;
        }
        if (frame_cb == VK_NULL_HANDLE) {
            LOG_ERROR("recordFrameBarriers requires a non-null frame command buffer");
            pending_frame_barriers_.clear();
            return;
        }

        std::vector<VkImageMemoryBarrier2> barriers;
        barriers.reserve(pending_frame_barriers_.size());
        for (const auto& pending : pending_frame_barriers_) {
            if (pending.unit == nullptr) {
                continue;
            }
            // Explicit scopes from layoutAccess; unit.layout is source of truth (no tracker map).
            const auto source = VulkanImageBarrierTracker::layoutAccess(
                VK_IMAGE_LAYOUT_GENERAL, VulkanImageBarrierTracker::AccessDirection::Source);
            const auto destination = VulkanImageBarrierTracker::layoutAccess(
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VulkanImageBarrierTracker::AccessDirection::Destination);
            VkImageMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.srcStageMask = source.stage;
            barrier.srcAccessMask = source.access;
            barrier.dstStageMask = destination.stage;
            barrier.dstAccessMask = destination.access;
            barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = pending.unit->image.image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
            barriers.push_back(barrier);
        }
        if (!barriers.empty()) {
            VkDependencyInfo dependency{};
            dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size());
            dependency.pImageMemoryBarriers = barriers.data();
            vkCmdPipelineBarrier2(frame_cb, &dependency);
        }

        // Marker: last successful submit serial at record time. Next prepareFrame
        // (or export-locked syncUnsubmittedLayoutCommits) rolls back if no newer
        // successful endFrame submit occurred.
        const std::uint64_t commit_marker = context.lastSuccessfulFrameSubmitSerial();
        for (const auto& pending : pending_frame_barriers_) {
            if (pending.unit == nullptr || pending.target == nullptr || pending.channel == nullptr) {
                continue;
            }
            // F2-2: only commit layout/publish when the frame wait is accepted.
            // A rejected wait marks the frame invalid for submit; leaving layout
            // GENERAL keeps CacheHit/bind from sampling a still-GENERAL GPU image.
            if (!context.addFrameTimelineWait(pending.unit->semaphore.semaphore,
                                              pending.cuda_signal_value,
                                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)) {
                LOG_ERROR(
                    "recordFrameBarriers: addFrameTimelineWait failed; leaving layout GENERAL: {}",
                    context.lastError());
                continue;
            }
            pending.unit->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            pending.target->uploaded_source_generation = pending.source_generation;
            ++pending.target->generation;
            // Publishing channels publish at sample-frame time, not before CUDA handoff.
            if (pending.channel->policy.publishes_published) {
                publishFromTarget(*pending.channel, *pending.target);
            }
            pending_layout_commits_.push_back(PendingLayoutCommit{
                .unit = pending.unit,
                .channel = pending.channel,
                .frame_submit_marker = commit_marker,
            });
        }
        pending_frame_barriers_.clear();
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
            // Tensor path: a successful cached bind below sets valid/alloc from the
            // pooled unit. Until then the sampled fallback texture is tight, so the
            // default must stay preserve-or-size, never the source bucket.
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
                !target->unit ||
                !target->unit->interop.valid() ||
                target->unit->layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ||
                target->valid_size != params.scene_image_size ||
                scene.source_generation == 0 ||
                target->uploaded_source_generation != scene.source_generation) {
                return false;
            }

            params.external_scene_image = target->unit->image.image;
            params.external_scene_image_view = target->unit->image.view;
            params.external_scene_image_layout = target->unit->layout;
            params.external_scene_image_generation = target->generation;
            params.scene_image_size = target->valid_size;
            params.scene_image_alloc_size = target->alloc_size;
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
            const glm::ivec2 d_valid = depth.published_valid_size;
            const glm::ivec2 d_alloc =
                depth.published_alloc_size.x > 0 && depth.published_alloc_size.y > 0
                    ? depth.published_alloc_size
                    : d_valid;
            params.depth_blit.uv_scale = outputUvScale(d_valid, d_alloc);
            params.depth_blit.uv_clamp_max = outputUvClampMax(d_valid, d_alloc);
        }

        // Stitch in CUDA/Vulkan interop views: left reuses the existing scene
        // interop slot; right has its own parallel slot. When set, the split-view
        // pass binds these directly and skips the CPU staging upload.
        if (params.split_view.enabled) {
            if (params.external_scene_image_view != VK_NULL_HANDLE) {
                params.split_view.left.external_image_view = params.external_scene_image_view;
                params.split_view.left.external_image_generation = params.external_scene_image_generation;
                // Left panel UV: scene valid/alloc (tensor or external).
                params.split_view.left.uv_scale =
                    outputUvScale(params.scene_image_size, params.scene_image_alloc_size);
                params.split_view.left.uv_clamp_max =
                    outputUvClampMax(params.scene_image_size, params.scene_image_alloc_size);
            }
            const auto& split = channels_->split_right;
            if (split.published_image_view != VK_NULL_HANDLE) {
                params.split_view.right.external_image_view = split.published_image_view;
                params.split_view.right.external_image_generation = split.published_image_generation;
                const glm::ivec2 r_valid = split.published_valid_size;
                const glm::ivec2 r_alloc =
                    split.published_alloc_size.x > 0 && split.published_alloc_size.y > 0
                        ? split.published_alloc_size
                        : r_valid;
                params.split_view.right.uv_scale = outputUvScale(r_valid, r_alloc);
                params.split_view.right.uv_clamp_max = outputUvClampMax(r_valid, r_alloc);
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
        pending_uploads_.clear();
        pending_frame_barriers_.clear();
        pending_layout_commits_.clear();
        if (!upload_stream_.synchronize()) {
            LOG_WARN("CUDA/Vulkan GUI upload stream synchronization failed during shutdown: {}",
                     upload_stream_.lastError());
        }
        resetChannel(channels_->scene);
        resetChannel(channels_->split_right);
        resetChannel(channels_->depth_blit);
        if (teardown_context_ && interop_pool_) {
            drainInteropPool(*teardown_context_, /*force=*/true);
            interop_pool_->pool.trimIdle([&](PooledInteropUnit& unit) {
                if (!teardown_context_->waitForImmediateSubmits()) {
                    LOG_ERROR("Could not drain before interop trimIdle destroy: {}",
                              teardown_context_->lastError());
                }
                unit.interop.reset();
                teardown_context_->destroyExternalSemaphore(unit.semaphore);
                teardown_context_->destroyExternalImage(unit.image);
                unit.layout = VK_IMAGE_LAYOUT_UNDEFINED;
                unit.timeline_value = 0;
            });
        }
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
