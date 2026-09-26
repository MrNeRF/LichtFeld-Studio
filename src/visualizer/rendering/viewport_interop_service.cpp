/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "viewport_interop_service.hpp"

#include "core/logger.hpp"
#include "core/tensor_backend.hpp"
#include "output_image_pool.hpp"
#include "passes/vulkan_viewport_pass.hpp"
#include "rendering/image_tensor.hpp"
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

    struct ViewportInteropService::PooledInteropUnit {
        struct Image {
            VkImage image = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkFormat format = VK_FORMAT_UNDEFINED;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkDeviceSize allocation_size = 0;
        } image;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        lfs::core::Tensor snapshot;
        lfs::core::TensorVulkanBuffer storage;
        lfs::core::TensorCompletion ready;

        void destroy(VulkanContext& context) {
            if (image.view)
                vkDestroyImageView(context.device(), image.view, nullptr);
            if (image.image)
                vmaDestroyImage(context.allocator(), image.image, image.allocation);
            image = {};
            snapshot = {};
            storage = {};
            ready = {};
        }
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
        VkImage published_image = VK_NULL_HANDLE;
        VkImageView published_image_view = VK_NULL_HANDLE;
        VkImageLayout published_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkFormat published_image_format = VK_FORMAT_UNDEFINED;
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
                .debug_name_prefix = "scene",
                .failure_log_prefix = "Required Vulkan/tensor viewport interop failed",
                .external_handle_early_out = true,
                .publishes_published = false,
                .log_timer_perf = true,
            };
        case ChannelId::SplitRight:
            return ChannelPolicy{
                .id = ChannelId::SplitRight,
                .vk_format = VK_FORMAT_R8G8B8A8_UNORM,
                .debug_name_prefix = "split_right",
                .failure_log_prefix = "Required Vulkan/tensor split-view interop failed",
                .external_handle_early_out = false,
                .publishes_published = true,
                .log_timer_perf = false,
            };
        case ChannelId::DepthBlit:
            return ChannelPolicy{
                .id = ChannelId::DepthBlit,
                .vk_format = VK_FORMAT_R32_SFLOAT,
                .debug_name_prefix = "depth_blit",
                .failure_log_prefix = "Required Vulkan/tensor depth-blit interop failed",
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

    void ViewportInteropService::clearPublished(Channel& channel) {
        channel.published_image = VK_NULL_HANDLE;
        channel.published_image_view = VK_NULL_HANDLE;
        channel.published_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        channel.published_image_format = VK_FORMAT_UNDEFINED;
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
        channel.published_image = target.unit->image.image;
        channel.published_image_view = target.unit->image.view;
        channel.published_image_layout = target.unit->layout;
        channel.published_image_format = target.unit->image.format;
        channel.published_image_generation = target.generation;
        channel.published_valid_size = target.valid_size;
        channel.published_alloc_size = target.alloc_size;
    }

    bool ViewportInteropService::sourceOk(const Channel& channel) const {
        return channel.source_image &&
               channel.source_image->is_valid() &&
               channel.source_size.x > 0 &&
               channel.source_size.y > 0;
    }

    void ViewportInteropService::drainInteropPool(VulkanContext& context, const bool force) {
        if (!interop_pool_) {
            return;
        }
        const auto destroy_fn = [&context](PooledInteropUnit& unit) { unit.destroy(context); };
        const auto producer_pred = [](const PooledInteropUnit&, std::uint64_t) { return true; };
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
        const std::uint64_t producer = 0;
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
                target_ptr && target_ptr->unit && (target_ptr->unit->image.image != VK_NULL_HANDLE);
            inputs.target_layout_read_only =
                target_ptr && target_ptr->unit &&
                target_ptr->unit->layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            inputs.uploaded_source_generation =
                target_ptr ? target_ptr->uploaded_source_generation : 0;
        }
        inputs.source_generation = channel.source_generation;

        const auto fail_required_interop = [this, &channel](std::string message) -> void {
            channel.disabled = true;
            resetChannel(channel);
            LOG_ERROR("{}: {}", channel.policy.failure_log_prefix, message);
            throw std::runtime_error(std::move(message));
        };

        // An unchanged image can be sampled again without rewriting its slot.
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
        // transition + tensor write) while a prior GUI frame may still sample this FIF slot.
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
            !(target_ptr->unit->image.image != VK_NULL_HANDLE);
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
                .external = false,
            };

            if (auto hit = interop_pool_->pool.acquire(key)) {
                target_ptr->pool_serial = hit->acquisition_serial;
                target_ptr->unit = hit->payload;
            } else {
                PooledInteropUnit created{};
                VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                image.imageType = VK_IMAGE_TYPE_2D;
                image.format = channel.policy.vk_format;
                image.extent = {extent.width, extent.height, 1};
                image.mipLevels = image.arrayLayers = 1;
                image.samples = VK_SAMPLE_COUNT_1_BIT;
                image.tiling = VK_IMAGE_TILING_OPTIMAL;
                image.usage = kInteropExternalImageUsage;
                image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                VmaAllocationCreateInfo allocation{};
                allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                VmaAllocationInfo allocated{};
                if (vmaCreateImage(context.allocator(), &image, &allocation, &created.image.image,
                                   &created.image.allocation, &allocated) != VK_SUCCESS)
                    fail_required_interop("Viewport image allocation failed");
                created.image.allocation_size = allocated.size;
                created.image.format = image.format;
                VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                view.image = created.image.image;
                view.viewType = VK_IMAGE_VIEW_TYPE_2D;
                view.format = image.format;
                view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                if (vkCreateImageView(context.device(), &view, nullptr, &created.image.view) != VK_SUCCESS) {
                    created.destroy(context);
                    fail_required_interop("Viewport image view creation failed");
                }
                auto reg = interop_pool_->pool.registerCreated(key, std::move(created));
                target_ptr->pool_serial = reg.acquisition_serial;
                target_ptr->unit = reg.payload;

                switch (channel.policy.id) {
                case ChannelId::Scene:
                    LOG_INFO("Vulkan/tensor viewport interop target initialized for frame slot {}: valid {}x{} alloc {}x{}",
                             frame_slot, valid_size.x, valid_size.y, alloc_size.x, alloc_size.y);
                    break;
                case ChannelId::SplitRight:
                    LOG_INFO("Vulkan/tensor split-view right-panel interop initialized for slot {}: valid {}x{} alloc {}x{}",
                             frame_slot, valid_size.x, valid_size.y, alloc_size.x, alloc_size.y);
                    break;
                case ChannelId::DepthBlit:
                    LOG_INFO("Vulkan/tensor depth-blit interop initialized for slot {}: valid {}x{} alloc {}x{}",
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

        // Conversion and upload recording share the frame preparation boundary.
        pending_uploads_.push_back(ChannelUploadPlan{
            .channel = &channel,
            .target = &target,
        });
    }

    void ViewportInteropService::rollbackUnsubmittedLayoutCommits(VulkanContext& context) {
        // If endFrame never successfully submitted after recordFrameBarriers, the
        // upload barriers never ran on-device; restore the tracked layout.
        const std::uint64_t successful = context.lastSuccessfulFrameSubmitSerial();
        for (const auto& commit : pending_layout_commits_) {
            if (commit.unit == nullptr) {
                continue;
            }
            if (successful <= commit.frame_submit_marker) {
                commit.unit->layout = commit.old_layout;
                if (commit.target)
                    commit.target->uploaded_source_generation = 0;
                if (commit.channel != nullptr && commit.channel->policy.publishes_published) {
                    clearPublished(*commit.channel);
                }
            }
        }
        pending_layout_commits_.clear();
    }

    void ViewportInteropService::syncUnsubmittedLayoutCommits(VulkanContext& context) {
        teardown_context_ = &context;
        // Unrecorded barriers leave the image layout unchanged.
        pending_frame_barriers_.clear();
        rollbackUnsubmittedLayoutCommits(context);
    }

    void ViewportInteropService::prepareFrame(VulkanContext& context, const bool resize_deferring) {
        teardown_context_ = &context;
        context.resetImmediateSubmitsThisFrame();
        drainInteropPool(context, false);
        syncUnsubmittedLayoutCommits(context);
        pending_uploads_.clear();
        prepareChannel(context, channels_->scene, resize_deferring);
        prepareChannel(context, channels_->split_right, resize_deferring);
        prepareChannel(context, channels_->depth_blit, resize_deferring);
        for (const auto& plan : pending_uploads_) {
            auto& channel = *plan.channel;
            auto& unit = *plan.target->unit;
            const auto backend = lfs::core::gpu_backend_of(*channel.source_image).value_or(lfs::core::default_gpu_backend());
            const auto scope = context.tensorInterop().execution_scope(backend);
            auto prepared = channel.policy.vk_format == VK_FORMAT_R32_SFLOAT
                                ? channel.source_image->to(lfs::core::DataType::Float32).contiguous()
                                : lfs::rendering::prepareImageRgba8(*channel.source_image);
            const auto bytes = std::size_t(channel.source_size.x) * channel.source_size.y * 4;
            if (!prepared.is_valid() || prepared.bytes() != bytes)
                throw std::runtime_error("Viewport image dimensions do not match the source tensor");
            if (!unit.snapshot.is_valid() || unit.snapshot.shape() != prepared.shape() ||
                unit.snapshot.dtype() != prepared.dtype() || lfs::core::gpu_backend_of(unit.snapshot) != backend)
                unit.snapshot = context.tensorInterop().empty(prepared.shape(), prepared.dtype(), backend);
            unit.snapshot.copy_from(prepared);
            const auto storage = context.tensorInterop().buffer(unit.snapshot);
            if (!storage)
                throw std::runtime_error("Viewport image upload requires Vulkan-visible tensor storage");
            unit.storage = *storage;
            const lfs::core::Tensor* input = &unit.snapshot;
            unit.ready = context.tensorInterop().ready({&input, 1});
            pending_frame_barriers_.push_back({&unit, plan.target, &channel, channel.source_generation});
        }
        pending_uploads_.clear();
    }

    void ViewportInteropService::recordFrameBarriers(VkCommandBuffer frame_cb, VulkanContext& context) {
        if (frame_cb == VK_NULL_HANDLE) {
            pending_frame_barriers_.clear();
            return;
        }
        const auto commit_marker = context.lastSuccessfulFrameSubmitSerial();
        for (const auto& pending : pending_frame_barriers_) {
            auto& unit = *pending.unit;
            if (unit.ready.timeline().value && !context.addFrameTimelineWait(static_cast<VkSemaphore>(unit.ready.timeline().semaphore),
                                                                             unit.ready.timeline().value, VK_PIPELINE_STAGE_TRANSFER_BIT))
                throw std::runtime_error(context.lastError());
            const auto source = VulkanImageBarrierTracker::layoutAccess(
                unit.layout, VulkanImageBarrierTracker::AccessDirection::Source);
            VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            barrier.srcStageMask = source.stage;
            barrier.srcAccessMask = source.access;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.oldLayout = unit.layout;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = unit.image.image;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.imageMemoryBarrierCount = 1;
            dependency.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(frame_cb, &dependency);
            VkBufferImageCopy copy{};
            copy.bufferOffset = unit.storage.offset;
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {std::uint32_t(pending.target->valid_size.x), std::uint32_t(pending.target->valid_size.y), 1};
            vkCmdCopyBufferToImage(frame_cb, static_cast<VkBuffer>(unit.storage.buffer), unit.image.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            vkCmdPipelineBarrier2(frame_cb, &dependency);
            pending_layout_commits_.push_back({&unit, pending.channel, commit_marker, pending.target, unit.layout});
            unit.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            pending.target->uploaded_source_generation = pending.source_generation;
            ++pending.target->generation;
            if (pending.channel->policy.publishes_published)
                publishFromTarget(*pending.channel, *pending.target);
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
                !(target->unit->image.image != VK_NULL_HANDLE) ||
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
                // Export mode freezes the viewport and skips new Vulkan interop uploads.
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
            params.depth_blit.external_image = depth.published_image;
            params.depth_blit.external_image_view = depth.published_image_view;
            params.depth_blit.external_image_layout = depth.published_image_layout;
            params.depth_blit.external_image_format = depth.published_image_format;
            params.depth_blit.external_image_generation = depth.published_image_generation;
            const glm::ivec2 d_valid = depth.published_valid_size;
            const glm::ivec2 d_alloc =
                depth.published_alloc_size.x > 0 && depth.published_alloc_size.y > 0
                    ? depth.published_alloc_size
                    : d_valid;
            params.depth_blit.external_image_size = d_valid;
            params.depth_blit.external_image_allocation_size = d_alloc;
            params.depth_blit.uv_scale = outputUvScale(d_valid, d_alloc);
            params.depth_blit.uv_clamp_max = outputUvClampMax(d_valid, d_alloc);
        }

        // Split-view left samples the scene
        // interop slot; right has its own parallel slot. When set, the split-view
        // pass binds these directly and skips the CPU staging upload.
        if (params.split_view.enabled) {
            if (params.external_scene_image_view != VK_NULL_HANDLE) {
                params.split_view.left.external_image = params.external_scene_image;
                params.split_view.left.external_image_view = params.external_scene_image_view;
                params.split_view.left.external_image_layout = params.external_scene_image_layout;
                params.split_view.left.external_image_generation = params.external_scene_image_generation;
                // Left panel UV: scene valid/alloc (tensor or external).
                params.split_view.left.uv_scale =
                    outputUvScale(params.scene_image_size, params.scene_image_alloc_size);
                params.split_view.left.uv_clamp_max =
                    outputUvClampMax(params.scene_image_size, params.scene_image_alloc_size);
            }
            const auto& split = channels_->split_right;
            if (split.published_image_view != VK_NULL_HANDLE) {
                params.split_view.right.external_image = split.published_image;
                params.split_view.right.external_image_view = split.published_image_view;
                params.split_view.right.external_image_layout = split.published_image_layout;
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
        resetChannel(channels_->scene);
        resetChannel(channels_->split_right);
        resetChannel(channels_->depth_blit);
        if (teardown_context_ && interop_pool_) {
            drainInteropPool(*teardown_context_, /*force=*/true);
            interop_pool_->pool.trimIdle([&](PooledInteropUnit& unit) {
                unit.destroy(*teardown_context_);
            });
        }
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
