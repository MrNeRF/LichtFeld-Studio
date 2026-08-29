/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_scene_dlss_pipeline.hpp"

#include "diagnostics/vram_profiler.hpp"
#include "rendering/nvidia_dlss_plugin.hpp"
#include "rendering/scene_temporal_resolve.hpp"
#include "window/vulkan_barrier2.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_result.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <optional>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>

namespace lfs::vis {
    namespace {
        constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R8G8B8A8_UNORM;

        [[nodiscard]] constexpr std::size_t viewIndex(const TemporalViewId view) noexcept {
            return static_cast<std::size_t>(view);
        }

        [[nodiscard]] constexpr std::uint32_t pluginView(
            const TemporalViewId view) noexcept {
            return static_cast<std::uint32_t>(viewIndex(view));
        }

        [[nodiscard]] constexpr std::uint32_t pluginQuality(
            const SceneTemporalQuality quality) noexcept {
            switch (quality) {
            case SceneTemporalQuality::Quality:
                return LFS_SCENE_UPSCALER_PLUGIN_QUALITY;
            case SceneTemporalQuality::Performance:
                return LFS_SCENE_UPSCALER_PLUGIN_PERFORMANCE;
            case SceneTemporalQuality::Balanced:
                return LFS_SCENE_UPSCALER_PLUGIN_BALANCED;
            }
            return LFS_SCENE_UPSCALER_PLUGIN_BALANCED;
        }

        [[nodiscard]] constexpr std::uint32_t pluginResetFlags(
            const TemporalResetReason reasons) noexcept {
            std::uint32_t flags = LFS_SCENE_UPSCALER_PLUGIN_RESET_NONE;
            if (hasTemporalResetReason(reasons, TemporalResetReason::CameraCut) ||
                hasTemporalResetReason(reasons, TemporalResetReason::Projection))
                flags |= LFS_SCENE_UPSCALER_PLUGIN_RESET_CAMERA_CUT;
            if (hasTemporalResetReason(reasons, TemporalResetReason::RenderSize) ||
                hasTemporalResetReason(reasons, TemporalResetReason::RenderScale))
                flags |= LFS_SCENE_UPSCALER_PLUGIN_RESET_RENDER_SIZE;
            if (hasTemporalResetReason(reasons, TemporalResetReason::OutputExtent))
                flags |= LFS_SCENE_UPSCALER_PLUGIN_RESET_OUTPUT_SIZE;
            if (hasTemporalResetReason(reasons, TemporalResetReason::Scene) ||
                hasTemporalResetReason(reasons, TemporalResetReason::Backend))
                flags |= LFS_SCENE_UPSCALER_PLUGIN_RESET_SCENE;
            if (hasTemporalResetReason(reasons, TemporalResetReason::Quality))
                flags |= LFS_SCENE_UPSCALER_PLUGIN_RESET_QUALITY;
            if (hasTemporalResetReason(reasons, TemporalResetReason::FirstFrame) ||
                hasTemporalResetReason(reasons, TemporalResetReason::Requested) ||
                hasTemporalResetReason(reasons, TemporalResetReason::HistoryDisabled))
                flags |= LFS_SCENE_UPSCALER_PLUGIN_RESET_REQUESTED;
            if (hasTemporalResetReason(reasons, TemporalResetReason::RuntimeUnavailable) ||
                hasTemporalResetReason(reasons, TemporalResetReason::ResolveFailure) ||
                hasTemporalResetReason(reasons, TemporalResetReason::InvalidInput))
                flags |= LFS_SCENE_UPSCALER_PLUGIN_RESET_RUNTIME;
            return flags;
        }
    } // namespace

    struct VulkanSceneDlssPipeline::Impl {
        struct OutputResource {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            glm::ivec2 extent{0, 0};
            std::size_t allocation_bytes = 0;
            bool initialized = false;
            std::string vram_label;
        };

        struct ViewState {
            LfsSceneUpscalerFeatureConfigV1 feature{};
            bool feature_configured = false;
            std::uint32_t pending_reset_flags = LFS_SCENE_UPSCALER_PLUGIN_RESET_REQUESTED;
            std::chrono::steady_clock::time_point previous_evaluation{};
            bool has_previous_evaluation = false;
        };

        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        SceneTemporalCoordinator coordinator;
        VulkanSceneMotionPass motion;
        VulkanSceneDlssDepthPass depth;
        std::vector<OutputResource> outputs;
        std::array<ViewState, static_cast<std::size_t>(TemporalViewId::Count)> views{};
        bool motion_initialized = false;
        bool depth_initialized = false;
        bool runtime_initialized = false;

        ~Impl() { destroy(); }

        [[nodiscard]] bool init(VulkanContext& ctx) {
            context = &ctx;
            device = ctx.device();
            allocator = ctx.allocator();
            if (ctx.instance() == VK_NULL_HANDLE || ctx.physicalDevice() == VK_NULL_HANDLE ||
                device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE)
                return false;
            VkFormatProperties format{};
            vkGetPhysicalDeviceFormatProperties(ctx.physicalDevice(), OUTPUT_FORMAT, &format);
            return (format.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0 &&
                   (format.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
        }

        [[nodiscard]] bool ensureRuntime() {
            if (runtime_initialized)
                return true;
            if (context == nullptr)
                return false;
            const LfsSceneUpscalerRuntimeConfigV1 config{
                .struct_size = sizeof(LfsSceneUpscalerRuntimeConfigV1),
                .instance = context->instance(),
                .physical_device = context->physicalDevice(),
                .device = context->device(),
                .get_instance_proc_addr = vkGetInstanceProcAddr,
                .get_device_proc_addr = vkGetDeviceProcAddr,
            };
            runtime_initialized = NvidiaDlssPlugin::instance().initializeRuntime(config);
            return runtime_initialized;
        }

        void destroyOutput(OutputResource& output) {
            if (!output.vram_label.empty()) {
                lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                    "vulkan.scene_dlss.output", output.vram_label, 0);
            }
            if (output.view != VK_NULL_HANDLE)
                vkDestroyImageView(device, output.view, nullptr);
            if (output.image != VK_NULL_HANDLE)
                vmaDestroyImage(allocator, output.image, output.allocation);
            output = {};
        }

        void destroyOutputs() {
            for (auto& output : outputs)
                destroyOutput(output);
            outputs.clear();
        }

        void destroy() {
            for (std::size_t index = 0; index < views.size(); ++index) {
                NvidiaDlssPlugin::instance().releaseFeature(
                    pluginView(static_cast<TemporalViewId>(index)));
            }
            destroyOutputs();
            motion.shutdown();
            motion_initialized = false;
            depth.shutdown();
            depth_initialized = false;
            runtime_initialized = false;
        }

        [[nodiscard]] bool createOutput(OutputResource& output,
                                        const std::size_t slot,
                                        const glm::ivec2 extent) {
            VkImageCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = OUTPUT_FORMAT;
            info.extent = {static_cast<std::uint32_t>(extent.x),
                           static_cast<std::uint32_t>(extent.y), 1};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo allocation_info{};
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            VmaAllocationInfo allocation_result{};
            if (!vk_try_bool(vmaCreateImage(allocator,
                                            &info,
                                            &allocation_info,
                                            &output.image,
                                            &output.allocation,
                                            &allocation_result),
                             "vmaCreateImage(scene_dlss.output)",
                             "DLSS output allocation failed"))
                return false;
            VkImageViewCreateInfo view_info{};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = output.image;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = OUTPUT_FORMAT;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.layerCount = 1;
            if (!vk_try_bool(vkCreateImageView(device, &view_info, nullptr, &output.view),
                             "vkCreateImageView(scene_dlss.output)",
                             "DLSS output view creation failed")) {
                destroyOutput(output);
                return false;
            }
            output.extent = extent;
            output.allocation_bytes = static_cast<std::size_t>(allocation_result.size);
            output.vram_label = std::format("slot{}:{}x{}", slot, extent.x, extent.y);
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.scene_dlss.output", output.vram_label, output.allocation_bytes);
            context->setDebugObjectName(
                VK_OBJECT_TYPE_IMAGE, output.image, "scene_dlss.output");
            context->setDebugObjectName(
                VK_OBJECT_TYPE_IMAGE_VIEW, output.view, "scene_dlss.output.view");
            return true;
        }

        [[nodiscard]] OutputResource* ensureOutput(const std::size_t slot,
                                                   const glm::ivec2 extent) {
            if (slot >= outputs.size())
                outputs.resize(slot + 1);
            auto& output = outputs[slot];
            if (output.image != VK_NULL_HANDLE && output.extent == extent)
                return &output;
            if (output.image != VK_NULL_HANDLE && !context->waitForSubmittedFrames())
                return nullptr;
            destroyOutput(output);
            return createOutput(output, slot, extent) ? &output : nullptr;
        }

        [[nodiscard]] static float frameTimeMilliseconds(ViewState& view) {
            const auto now = std::chrono::steady_clock::now();
            float result = 1000.0f / 60.0f;
            if (view.has_previous_evaluation) {
                result = std::chrono::duration<float, std::milli>(
                             now - view.previous_evaluation)
                             .count();
                result = std::clamp(result, 1.0f, 100.0f);
            }
            view.previous_evaluation = now;
            view.has_previous_evaluation = true;
            return result;
        }

        [[nodiscard]] bool ensureFeature(const VkCommandBuffer command_buffer,
                                         const PreparedSceneTemporalFrame& prepared,
                                         const SceneTemporalQuality quality) {
            auto& view = views.at(viewIndex(prepared.view));
            const LfsSceneUpscalerFeatureConfigV1 feature{
                .struct_size = sizeof(LfsSceneUpscalerFeatureConfigV1),
                .view = pluginView(prepared.view),
                .quality = pluginQuality(quality),
                .render_width = static_cast<std::uint32_t>(prepared.plan.render_extent.x),
                .render_height = static_cast<std::uint32_t>(prepared.plan.render_extent.y),
                .output_width = static_cast<std::uint32_t>(prepared.plan.output_extent.x),
                .output_height = static_cast<std::uint32_t>(prepared.plan.output_extent.y),
                .motion_vectors_include_jitter = 0,
            };
            const bool changed = !view.feature_configured ||
                                 view.feature.quality != feature.quality ||
                                 view.feature.render_width != feature.render_width ||
                                 view.feature.render_height != feature.render_height ||
                                 view.feature.output_width != feature.output_width ||
                                 view.feature.output_height != feature.output_height;
            if (changed && view.feature_configured && !context->waitForSubmittedFrames())
                return false;
            if (!NvidiaDlssPlugin::instance().createFeature(command_buffer, feature))
                return false;
            if (changed)
                view.pending_reset_flags |= LFS_SCENE_UPSCALER_PLUGIN_RESET_QUALITY |
                                            LFS_SCENE_UPSCALER_PLUGIN_RESET_RENDER_SIZE |
                                            LFS_SCENE_UPSCALER_PLUGIN_RESET_OUTPUT_SIZE;
            view.feature = feature;
            view.feature_configured = true;
            return true;
        }

        [[nodiscard]] VulkanSceneDlssPipelineResult fail(
            const PreparedSceneTemporalFrame& prepared,
            const VulkanSceneDlssPipelineStatus status,
            const TemporalResetReason reason) {
            coordinator.discard(prepared, reason);
            views.at(viewIndex(prepared.view)).pending_reset_flags |=
                LFS_SCENE_UPSCALER_PLUGIN_RESET_RUNTIME;
            return {.status = status, .view = prepared.view};
        }

        [[nodiscard]] VulkanSceneDlssPipelineResult record(
            const VkCommandBuffer command_buffer,
            const VulkanSceneDlssPipelineRequest& request) {
            if (!validVulkanSceneDlssPipelineRequest(request) ||
                command_buffer == VK_NULL_HANDLE) {
                return {.status = VulkanSceneDlssPipelineStatus::InvalidRequest,
                        .view = request.temporal.temporal.view};
            }
            if (!ensureRuntime())
                return {.status = VulkanSceneDlssPipelineStatus::RuntimeUnavailable,
                        .view = request.temporal.temporal.view};

            const auto prepared = coordinator.prepare(request.temporal.temporal);
            if (!prepared.active())
                return {.status = VulkanSceneDlssPipelineStatus::Inactive,
                        .view = request.temporal.temporal.view};
            const auto view_projections = makeTemporalMotionViewProjectionPair(prepared.frame);
            if (!view_projections)
                return fail(prepared,
                            VulkanSceneDlssPipelineStatus::InvalidRequest,
                            TemporalResetReason::Projection);

            if (!motion_initialized)
                motion_initialized = motion.init(*context);
            if (!motion_initialized)
                return fail(prepared,
                            VulkanSceneDlssPipelineStatus::MotionUnavailable,
                            TemporalResetReason::RuntimeUnavailable);
            const auto resource_slot = temporalMotionResourceSlot(
                request.temporal.frame_slot, request.temporal.temporal.view);
            if (!resource_slot)
                return fail(prepared,
                            VulkanSceneDlssPipelineStatus::InvalidRequest,
                            TemporalResetReason::InvalidInput);
            auto motion_params = request.temporal.motion;
            motion_params.inverse_current_view_projection = glm::inverse(view_projections->current);
            motion_params.previous_view_projection = view_projections->previous;
            if (!motion.record(command_buffer, motion_params, *resource_slot))
                return fail(prepared,
                            VulkanSceneDlssPipelineStatus::MotionFailure,
                            TemporalResetReason::ResolveFailure);

            if (!depth_initialized)
                depth_initialized = depth.init(*context);
            if (!depth_initialized)
                return fail(prepared,
                            VulkanSceneDlssPipelineStatus::DepthUnavailable,
                            TemporalResetReason::RuntimeUnavailable);
            const VulkanSceneDlssDepthParams depth_params{
                .enabled = true,
                .current_depth_view = request.temporal.motion.depth_view,
                .current_depth_layout =
                    request.temporal.resolve.current_depth.current_depth_layout,
                .depth = request.temporal.motion.depth,
                .allocation_extent =
                    request.temporal.resolve.current_depth.allocation_extent,
            };
            if (!depth.record(command_buffer, depth_params, *resource_slot))
                return fail(prepared,
                            VulkanSceneDlssPipelineStatus::DepthFailure,
                            TemporalResetReason::ResolveFailure);

            auto* const output = ensureOutput(*resource_slot, prepared.plan.output_extent);
            if (output == nullptr)
                return fail(prepared,
                            VulkanSceneDlssPipelineStatus::OutputFailure,
                            TemporalResetReason::ResolveFailure);
            if (!ensureFeature(command_buffer, prepared, request.quality))
                return fail(prepared,
                            VulkanSceneDlssPipelineStatus::FeatureFailure,
                            TemporalResetReason::RuntimeUnavailable);

            cmdImageBarrier2(command_buffer,
                             output->image,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             output->initialized ? VK_IMAGE_LAYOUT_GENERAL
                                                 : VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL,
                             output->initialized
                                 ? (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)
                                 : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             output->initialized
                                 ? (VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)
                                 : VK_ACCESS_2_NONE,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            auto& view = views.at(viewIndex(prepared.view));
            const glm::vec2 jitter_pixels = sceneTemporalJitterPixels(
                prepared.frame.current_jitter,
                prepared.plan.render_extent,
                request.temporal.motion.flip_y);
            const LfsSceneUpscalerEvaluateV1 evaluation{
                .struct_size = sizeof(LfsSceneUpscalerEvaluateV1),
                .view = pluginView(prepared.view),
                .command_buffer = command_buffer,
                .color = {
                    .struct_size = sizeof(LfsSceneUpscalerImageV1),
                    .image = request.color_image,
                    .view = request.temporal.resolve.current_color_view,
                    .layout = request.temporal.resolve.current_color_layout,
                    .format = request.color_format,
                    .allocation_width = static_cast<std::uint32_t>(
                        request.temporal.resolve.current_allocation_extent.x),
                    .allocation_height = static_cast<std::uint32_t>(
                        request.temporal.resolve.current_allocation_extent.y),
                    .valid_width = static_cast<std::uint32_t>(prepared.plan.render_extent.x),
                    .valid_height = static_cast<std::uint32_t>(prepared.plan.render_extent.y),
                    .writable = 0,
                },
                .depth = {
                    .struct_size = sizeof(LfsSceneUpscalerImageV1),
                    .image = depth.depthImage(*resource_slot),
                    .view = depth.depthView(*resource_slot),
                    .layout = VK_IMAGE_LAYOUT_GENERAL,
                    .format = VK_FORMAT_R32_SFLOAT,
                    .allocation_width = static_cast<std::uint32_t>(prepared.plan.render_extent.x),
                    .allocation_height = static_cast<std::uint32_t>(prepared.plan.render_extent.y),
                    .valid_width = static_cast<std::uint32_t>(prepared.plan.render_extent.x),
                    .valid_height = static_cast<std::uint32_t>(prepared.plan.render_extent.y),
                    .writable = 0,
                },
                .motion = {
                    .struct_size = sizeof(LfsSceneUpscalerImageV1),
                    .image = motion.motionImage(*resource_slot),
                    .view = motion.motionView(*resource_slot),
                    .layout = VK_IMAGE_LAYOUT_GENERAL,
                    .format = VK_FORMAT_R16G16_SFLOAT,
                    .allocation_width = static_cast<std::uint32_t>(prepared.plan.render_extent.x),
                    .allocation_height = static_cast<std::uint32_t>(prepared.plan.render_extent.y),
                    .valid_width = static_cast<std::uint32_t>(prepared.plan.render_extent.x),
                    .valid_height = static_cast<std::uint32_t>(prepared.plan.render_extent.y),
                    .writable = 0,
                },
                .output = {
                    .struct_size = sizeof(LfsSceneUpscalerImageV1),
                    .image = output->image,
                    .view = output->view,
                    .layout = VK_IMAGE_LAYOUT_GENERAL,
                    .format = OUTPUT_FORMAT,
                    .allocation_width = static_cast<std::uint32_t>(prepared.plan.output_extent.x),
                    .allocation_height = static_cast<std::uint32_t>(prepared.plan.output_extent.y),
                    .valid_width = static_cast<std::uint32_t>(prepared.plan.output_extent.x),
                    .valid_height = static_cast<std::uint32_t>(prepared.plan.output_extent.y),
                    .writable = 1,
                },
                .jitter_x_pixels = jitter_pixels.x,
                .jitter_y_pixels = jitter_pixels.y,
                .motion_scale_x = 1.0f,
                .motion_scale_y = 1.0f,
                .pre_exposure = 1.0f,
                .frame_time_milliseconds = frameTimeMilliseconds(view),
                .reset_flags = view.pending_reset_flags | pluginResetFlags(prepared.frame.reset_reasons),
            };
            if (!NvidiaDlssPlugin::instance().evaluate(evaluation))
                return fail(prepared,
                            VulkanSceneDlssPipelineStatus::EvaluateFailure,
                            TemporalResetReason::ResolveFailure);
            view.pending_reset_flags = LFS_SCENE_UPSCALER_PLUGIN_RESET_NONE;
            output->initialized = true;
            cmdImageBarrier2(command_buffer,
                             output->image,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

            if (!coordinator.commit(prepared,
                                    SceneHistoryStorage::VulkanImage,
                                    SceneHistoryStorage::VulkanImage))
                return fail(prepared,
                            VulkanSceneDlssPipelineStatus::CommitFailure,
                            TemporalResetReason::ResolveFailure);
            const SceneHistoryContract history{
                .color_storage = SceneHistoryStorage::VulkanImage,
                .depth_storage = SceneHistoryStorage::VulkanImage,
                .color_extent = prepared.plan.output_extent,
                .depth_extent = prepared.plan.render_extent,
                .sequence = prepared.frame.sequence + 1,
            };
            return {
                .status = VulkanSceneDlssPipelineStatus::Resolved,
                .view = prepared.view,
                .sequence = history.sequence,
                .output_view = output->view,
                .history = history,
            };
        }

        void reset(const TemporalViewId view, const TemporalResetReason reason) {
            coordinator.reset(view, reason);
            if (validTemporalViewId(view)) {
                auto& state = views.at(viewIndex(view));
                state.pending_reset_flags |= pluginResetFlags(reason);
                state.has_previous_evaluation = false;
            }
        }

        void resetAll(const TemporalResetReason reason) {
            coordinator.resetAll(reason);
            for (auto& state : views) {
                state.pending_reset_flags |= pluginResetFlags(reason);
                state.has_previous_evaluation = false;
            }
        }

        void releaseResources(const TemporalResetReason reason) {
            resetAll(reason);
            for (std::size_t index = 0; index < views.size(); ++index) {
                NvidiaDlssPlugin::instance().releaseFeature(
                    pluginView(static_cast<TemporalViewId>(index)));
                views[index].feature = {};
                views[index].feature_configured = false;
            }
            destroyOutputs();
            motion.shutdown();
            motion_initialized = false;
            depth.shutdown();
            depth_initialized = false;
        }
    };

    VulkanSceneDlssPipeline::VulkanSceneDlssPipeline() = default;
    VulkanSceneDlssPipeline::~VulkanSceneDlssPipeline() = default;
    VulkanSceneDlssPipeline::VulkanSceneDlssPipeline(
        VulkanSceneDlssPipeline&&) noexcept = default;
    VulkanSceneDlssPipeline& VulkanSceneDlssPipeline::operator=(
        VulkanSceneDlssPipeline&&) noexcept = default;

    bool VulkanSceneDlssPipeline::init(VulkanContext& context) {
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        return impl_->init(context);
    }

    VulkanSceneDlssPipelineResult VulkanSceneDlssPipeline::record(
        const VkCommandBuffer command_buffer,
        const VulkanSceneDlssPipelineRequest& request) {
        return impl_ ? impl_->record(command_buffer, request)
                     : VulkanSceneDlssPipelineResult{
                           .status = VulkanSceneDlssPipelineStatus::InvalidRequest,
                           .view = request.temporal.temporal.view};
    }

    void VulkanSceneDlssPipeline::reset(const TemporalViewId view,
                                        const TemporalResetReason reason) {
        if (impl_)
            impl_->reset(view, reason);
    }

    void VulkanSceneDlssPipeline::resetAll(const TemporalResetReason reason) {
        if (impl_)
            impl_->resetAll(reason);
    }

    void VulkanSceneDlssPipeline::releaseResources(const TemporalResetReason reason) {
        if (impl_)
            impl_->releaseResources(reason);
    }

    void VulkanSceneDlssPipeline::shutdown() { impl_.reset(); }

    std::size_t VulkanSceneDlssPipeline::residentOutputCount() const {
        if (!impl_)
            return 0;
        return static_cast<std::size_t>(std::count_if(
            impl_->outputs.begin(),
            impl_->outputs.end(),
            [](const Impl::OutputResource& output) {
                return output.image != VK_NULL_HANDLE;
            }));
    }
} // namespace lfs::vis
