/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_scene_temporal_resolve_pass.hpp"

#include "rendering/scene_temporal_resolve.hpp"
#include "vulkan_scene_depth_history_pass.hpp"

#include "core/logger.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "window/vulkan_barrier2.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_result.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <string>
#include <vk_mem_alloc.h>

#include "viewport/scene_temporal_resolve.comp.spv.h"

namespace lfs::vis {
    namespace {
        struct alignas(16) ResolvePush {
            glm::ivec4 extents{0};
            glm::vec4 control{0.0f};
            glm::vec4 current_uv{1.0f};
            glm::vec4 depth_control{0.0f};
            glm::vec4 jitter_pixels{0.0f};
            glm::vec4 reconstruction{0.0f};
        };
        static_assert(sizeof(ResolvePush) == 96);

        constexpr std::size_t viewIndex(const TemporalViewId view) {
            return static_cast<std::size_t>(view);
        }
    } // namespace

    struct VulkanSceneTemporalResolvePass::Impl {
        struct ImageResource {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            std::size_t allocation_bytes = 0;
            bool initialized = false;
            std::string vram_label;
        };

        struct ViewResource {
            std::array<ImageResource, 2> images;
            glm::ivec2 extent{0, 0};
            std::size_t read_index = 0;
            bool has_history = false;
            SceneHistoryContract contract{};
        };

        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
        VkSampler linear_sampler = VK_NULL_HANDLE;
        VkSampler nearest_sampler = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        std::array<ViewResource, static_cast<std::size_t>(TemporalViewId::Count)> views;
        VulkanSceneDepthHistoryPass depth_history;
        bool depth_history_initialized = false;
        std::array<std::uint32_t, 3> max_group_count{};

        ~Impl() { destroy(); }

        [[nodiscard]] bool init(VulkanContext& ctx) {
            context = &ctx;
            device = ctx.device();
            allocator = ctx.allocator();
            pipeline_cache = ctx.pipelineCache();
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE ||
                ctx.vkCmdPushDescriptorSet() == nullptr) {
                return false;
            }
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(ctx.physicalDevice(), &properties);
            max_group_count = {properties.limits.maxComputeWorkGroupCount[0],
                               properties.limits.maxComputeWorkGroupCount[1],
                               properties.limits.maxComputeWorkGroupCount[2]};
            VkFormatProperties format_properties{};
            vkGetPhysicalDeviceFormatProperties(
                ctx.physicalDevice(), VK_FORMAT_R16G16B16A16_SFLOAT, &format_properties);
            return (format_properties.optimalTilingFeatures &
                    VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
        }

        void destroyImage(ImageResource& image) {
            if (!image.vram_label.empty()) {
                lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                    "vulkan.scene_temporal.history", image.vram_label, 0);
            }
            if (image.view != VK_NULL_HANDLE)
                vkDestroyImageView(device, image.view, nullptr);
            if (image.image != VK_NULL_HANDLE)
                vmaDestroyImage(allocator, image.image, image.allocation);
            image = {};
        }

        void resetView(const TemporalViewId view) {
            auto& resource = views.at(viewIndex(view));
            for (auto& image : resource.images)
                destroyImage(image);
            resource = {};
        }

        void invalidateView(const TemporalViewId view) {
            if (!validTemporalViewId(view))
                return;
            auto& resource = views[viewIndex(view)];
            resource.has_history = false;
            resource.contract = {};
            for (std::size_t ping = 0; ping < 2; ++ping) {
                if (const auto slot = temporalDepthHistoryResourceSlot(view, ping))
                    depth_history.invalidate(*slot);
            }
        }

        void destroy() {
            for (std::size_t index = 0; index < views.size(); ++index)
                resetView(static_cast<TemporalViewId>(index));
            destroyStaticResources();
        }

        void releaseHistory() {
            for (std::size_t index = 0; index < views.size(); ++index)
                resetView(static_cast<TemporalViewId>(index));
            depth_history.shutdown();
            depth_history_initialized = false;
        }

        void destroyStaticResources() {
            if (pipeline != VK_NULL_HANDLE)
                vkDestroyPipeline(device, pipeline, nullptr);
            if (pipeline_layout != VK_NULL_HANDLE)
                vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            if (descriptor_layout != VK_NULL_HANDLE)
                vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
            if (linear_sampler != VK_NULL_HANDLE)
                vkDestroySampler(device, linear_sampler, nullptr);
            if (nearest_sampler != VK_NULL_HANDLE)
                vkDestroySampler(device, nearest_sampler, nullptr);
            pipeline = VK_NULL_HANDLE;
            pipeline_layout = VK_NULL_HANDLE;
            descriptor_layout = VK_NULL_HANDLE;
            linear_sampler = VK_NULL_HANDLE;
            nearest_sampler = VK_NULL_HANDLE;
        }

        [[nodiscard]] bool createStaticResources() {
            if (pipeline != VK_NULL_HANDLE)
                return true;
            if (linear_sampler != VK_NULL_HANDLE || nearest_sampler != VK_NULL_HANDLE ||
                descriptor_layout != VK_NULL_HANDLE ||
                pipeline_layout != VK_NULL_HANDLE)
                destroyStaticResources();

            VkSamplerCreateInfo sampler_info{};
            sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sampler_info.magFilter = VK_FILTER_LINEAR;
            sampler_info.minFilter = VK_FILTER_LINEAR;
            sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (!vk_try_bool(vkCreateSampler(device, &sampler_info, nullptr, &linear_sampler),
                             "vkCreateSampler(scene_temporal.linear)",
                             "Scene temporal linear sampler creation failed"))
                return false;

            sampler_info.magFilter = VK_FILTER_NEAREST;
            sampler_info.minFilter = VK_FILTER_NEAREST;
            if (!vk_try_bool(vkCreateSampler(device,
                                             &sampler_info,
                                             nullptr,
                                             &nearest_sampler),
                             "vkCreateSampler(scene_temporal.nearest)",
                             "Scene temporal nearest sampler creation failed")) {
                destroyStaticResources();
                return false;
            }

            std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
            for (std::uint32_t index = 0; index < 5; ++index) {
                bindings[index] = {index, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            }
            bindings[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                           VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo descriptor_info{};
            descriptor_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptor_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            descriptor_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
            descriptor_info.pBindings = bindings.data();
            if (!vk_try_bool(vkCreateDescriptorSetLayout(
                                 device, &descriptor_info, nullptr, &descriptor_layout),
                             "vkCreateDescriptorSetLayout(scene_temporal)",
                             "Scene temporal descriptor layout creation failed"))
                return false;

            VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ResolvePush)};
            VkPipelineLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layout_info.setLayoutCount = 1;
            layout_info.pSetLayouts = &descriptor_layout;
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &push_range;
            if (!vk_try_bool(vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout),
                             "vkCreatePipelineLayout(scene_temporal)",
                             "Scene temporal pipeline layout creation failed"))
                return false;

            VkShaderModuleCreateInfo shader_info{};
            shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shader_info.codeSize = sizeof(viewport_shaders::kSceneTemporalResolveCompSpv);
            shader_info.pCode = viewport_shaders::kSceneTemporalResolveCompSpv;
            VkShaderModule shader = VK_NULL_HANDLE;
            if (!vk_try_bool(vkCreateShaderModule(device, &shader_info, nullptr, &shader),
                             "vkCreateShaderModule(scene_temporal)",
                             "Scene temporal shader module creation failed"))
                return false;
            VkPipelineShaderStageCreateInfo stage{};
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            VkComputePipelineCreateInfo pipeline_info{};
            pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipeline_info.stage = stage;
            pipeline_info.layout = pipeline_layout;
            const VkResult result = vkCreateComputePipelines(
                device, pipeline_cache, 1, &pipeline_info, nullptr, &pipeline);
            vkDestroyShaderModule(device, shader, nullptr);
            return vk_try_bool(result,
                               "vkCreateComputePipelines(scene_temporal)",
                               "Scene temporal pipeline creation failed");
        }

        [[nodiscard]] bool createImage(ImageResource& image,
                                       const TemporalViewId view,
                                       const std::size_t ping,
                                       const glm::ivec2 extent) {
            VkImageCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            info.extent = {static_cast<std::uint32_t>(extent.x),
                           static_cast<std::uint32_t>(extent.y), 1};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo allocation_info{};
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            VmaAllocationInfo allocation_result{};
            if (!vk_try_bool(vmaCreateImage(allocator,
                                            &info,
                                            &allocation_info,
                                            &image.image,
                                            &image.allocation,
                                            &allocation_result),
                             "vmaCreateImage(scene_temporal.history)",
                             "Scene temporal history allocation failed"))
                return false;
            VkImageViewCreateInfo view_info{};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = image.image;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = info.format;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.layerCount = 1;
            if (!vk_try_bool(vkCreateImageView(device, &view_info, nullptr, &image.view),
                             "vkCreateImageView(scene_temporal.history)",
                             "Scene temporal history view creation failed"))
                return false;
            image.vram_label = std::format("view{}:ping{}:{}x{}",
                                           viewIndex(view), ping, extent.x, extent.y);
            image.allocation_bytes = static_cast<std::size_t>(allocation_result.size);
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.scene_temporal.history",
                image.vram_label,
                image.allocation_bytes);
            return true;
        }

        [[nodiscard]] VulkanSceneTemporalResourceStats resourceStats() const {
            VulkanSceneTemporalResourceStats stats{
                .static_resources_initialized = pipeline != VK_NULL_HANDLE,
            };
            for (const auto& resource : views) {
                bool view_resident = false;
                for (const auto& image : resource.images) {
                    if (image.image == VK_NULL_HANDLE)
                        continue;
                    stats.history_bytes += image.allocation_bytes;
                    ++stats.history_images;
                    view_resident = true;
                }
                if (view_resident)
                    ++stats.resident_views;
            }
            return stats;
        }

        [[nodiscard]] bool ensureView(const TemporalViewId view, const glm::ivec2 extent) {
            auto& resource = views.at(viewIndex(view));
            if (resource.extent == extent && resource.images[0].image != VK_NULL_HANDLE &&
                resource.images[1].image != VK_NULL_HANDLE)
                return true;
            if (resource.images[0].image != VK_NULL_HANDLE && !context->waitForSubmittedFrames())
                return false;
            resetView(view);
            resource.extent = extent;
            if (!createImage(resource.images[0], view, 0, extent) ||
                !createImage(resource.images[1], view, 1, extent)) {
                resetView(view);
                return false;
            }
            return true;
        }

        [[nodiscard]] bool record(const VkCommandBuffer command_buffer,
                                  const VulkanSceneTemporalResolveParams& params) {
            if (!validTemporalViewId(params.view))
                return false;
            if (!params.enabled) {
                invalidateView(params.view);
                return true;
            }
            const glm::ivec2 allocation =
                params.current_allocation_extent.x > 0 && params.current_allocation_extent.y > 0
                    ? params.current_allocation_extent
                    : params.render_extent;
            const glm::vec4 current_uv = temporalCurrentUvTransform(params.render_extent,
                                                                    allocation);
            const glm::vec2 current_jitter_pixels = sceneTemporalJitterPixels(
                params.current_jitter_ndc, params.render_extent, params.jitter_flip_y);
            const glm::vec2 previous_jitter_pixels = sceneTemporalJitterPixels(
                params.previous_jitter_ndc, params.render_extent, params.jitter_flip_y);
            if (command_buffer == VK_NULL_HANDLE || params.current_color_view == VK_NULL_HANDLE ||
                params.motion_view == VK_NULL_HANDLE || params.render_extent.x <= 0 ||
                params.render_extent.y <= 0 || params.output_extent.x <= 0 ||
                params.output_extent.y <= 0 || current_uv == glm::vec4(0.0f) ||
                !validTemporalDepthInputs(params) ||
                !createStaticResources() ||
                !ensureView(params.view, params.output_extent))
                return false;

            auto& resource = views.at(viewIndex(params.view));
            const std::size_t write_index =
                nextTemporalHistoryWriteIndex(resource.has_history, resource.read_index);
            auto& output = resource.images[write_index];
            auto& history = resource.images[resource.read_index];

            VkImageView current_depth_view = params.current_color_view;
            VkImageView history_depth_view = params.current_color_view;
            VkImageLayout current_depth_layout = params.current_color_layout;
            VkImageLayout history_depth_layout = params.current_color_layout;
            bool depth_rejection = false;
            if (params.current_depth.enabled) {
                if (!depth_history_initialized) {
                    depth_history_initialized = depth_history.init(*context);
                }
                const auto write_slot = temporalDepthHistoryResourceSlot(params.view, write_index);
                const auto read_slot = temporalDepthHistoryResourceSlot(params.view,
                                                                        resource.read_index);
                if (!depth_history_initialized || !write_slot || !read_slot) {
                    return false;
                }
                depth_history.invalidate(*write_slot);
                if (!depth_history.record(command_buffer, params.current_depth, *write_slot))
                    return false;
                current_depth_view = depth_history.depthView(*write_slot);
                current_depth_layout = VK_IMAGE_LAYOUT_GENERAL;
                const auto previous_depth_contract = depth_history.contract(*read_slot);
                depth_rejection = resource.has_history && params.history_valid &&
                                  previous_depth_contract.valid();
                if (depth_rejection) {
                    history_depth_view = depth_history.depthView(*read_slot);
                    history_depth_layout = VK_IMAGE_LAYOUT_GENERAL;
                } else {
                    history_depth_view = current_depth_view;
                    history_depth_layout = current_depth_layout;
                }
            } else {
                for (std::size_t ping = 0; ping < 2; ++ping) {
                    if (const auto slot = temporalDepthHistoryResourceSlot(params.view, ping))
                        depth_history.invalidate(*slot);
                }
            }
            cmdImageBarrier2(command_buffer,
                             output.image,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             output.initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL,
                             output.initialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                                : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             output.initialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                                                : VK_ACCESS_2_NONE,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            VkDescriptorImageInfo current_info{linear_sampler,
                                               params.current_color_view,
                                               params.current_color_layout};
            VkDescriptorImageInfo history_info{linear_sampler,
                                               resource.has_history ? history.view
                                                                    : params.current_color_view,
                                               resource.has_history ? VK_IMAGE_LAYOUT_GENERAL
                                                                    : params.current_color_layout};
            // Motion vectors describe one rasterized surface sample. Linear filtering
            // across primitive/disocclusion edges invents velocities and causes ghosting.
            VkDescriptorImageInfo motion_info{
                nearest_sampler, params.motion_view, params.motion_layout};
            VkDescriptorImageInfo current_depth_info{
                nearest_sampler, current_depth_view, current_depth_layout};
            VkDescriptorImageInfo history_depth_info{
                nearest_sampler, history_depth_view, history_depth_layout};
            VkDescriptorImageInfo output_info{VK_NULL_HANDLE, output.view, VK_IMAGE_LAYOUT_GENERAL};
            std::array<VkDescriptorImageInfo*, 6> infos{&current_info,
                                                        &history_info,
                                                        &motion_info,
                                                        &current_depth_info,
                                                        &history_depth_info,
                                                        &output_info};
            std::array<VkWriteDescriptorSet, 6> writes{};
            for (std::uint32_t index = 0; index < writes.size(); ++index) {
                writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[index].dstBinding = index;
                writes[index].descriptorCount = 1;
                writes[index].descriptorType = index == 5
                                                   ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                   : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[index].pImageInfo = infos[index];
            }
            ResolvePush push{
                .extents = {params.render_extent.x,
                            params.render_extent.y,
                            params.output_extent.x,
                            params.output_extent.y},
                .control = {resource.has_history && params.history_valid ? 1.0f : 0.0f,
                            sceneTemporalHistoryWeight(params.history_weight, params.sequence),
                            std::max(0.0f, params.motion_rejection_pixels),
                            0.0f},
                .current_uv = current_uv,
                .depth_control = {
                    depth_rejection ? 1.0f : 0.0f,
                    std::max(0.0f, params.depth_relative_threshold),
                    std::max(0.0f, params.depth_absolute_threshold),
                    depth_rejection ? params.current_depth.depth.far_plane : 0.0f},
                .jitter_pixels = glm::vec4(current_jitter_pixels, previous_jitter_pixels),
                .reconstruction = {std::clamp(params.current_sharpness, 0.0f, 0.25f), std::max(0.0f, params.motion_confidence_pixels), 0.0f, 0.0f},
            };
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            context->vkCmdPushDescriptorSet()(command_buffer,
                                              VK_PIPELINE_BIND_POINT_COMPUTE,
                                              pipeline_layout,
                                              0,
                                              static_cast<std::uint32_t>(writes.size()),
                                              writes.data());
            vkCmdPushConstants(command_buffer,
                               pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0,
                               sizeof(push),
                               &push);
            const std::uint32_t groups_x =
                (static_cast<std::uint32_t>(params.output_extent.x) + 7u) / 8u;
            const std::uint32_t groups_y =
                (static_cast<std::uint32_t>(params.output_extent.y) + 7u) / 8u;
            if (groups_x > max_group_count[0] || groups_y > max_group_count[1])
                return false;
            vkCmdDispatch(command_buffer, groups_x, groups_y, 1);
            cmdImageBarrier2(command_buffer,
                             output.image,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            output.initialized = true;
            resource.read_index = write_index;
            resource.has_history = true;
            resource.contract = {
                .color_storage = SceneHistoryStorage::VulkanImage,
                .depth_storage = params.current_depth.enabled
                                     ? SceneHistoryStorage::VulkanImage
                                     : SceneHistoryStorage::None,
                .color_extent = params.output_extent,
                .depth_extent = params.current_depth.enabled ? params.render_extent
                                                             : glm::ivec2(0),
                .sequence = params.sequence + 1,
            };
            return resource.contract.valid();
        }
    };

    VulkanSceneTemporalResolvePass::VulkanSceneTemporalResolvePass() = default;
    VulkanSceneTemporalResolvePass::~VulkanSceneTemporalResolvePass() = default;
    VulkanSceneTemporalResolvePass::VulkanSceneTemporalResolvePass(
        VulkanSceneTemporalResolvePass&&) noexcept = default;
    VulkanSceneTemporalResolvePass& VulkanSceneTemporalResolvePass::operator=(
        VulkanSceneTemporalResolvePass&&) noexcept = default;

    bool VulkanSceneTemporalResolvePass::init(VulkanContext& context) {
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        return impl_->init(context);
    }

    bool VulkanSceneTemporalResolvePass::record(
        const VkCommandBuffer command_buffer,
        const VulkanSceneTemporalResolveParams& params) {
        return impl_ && impl_->record(command_buffer, params);
    }

    void VulkanSceneTemporalResolvePass::reset(const TemporalViewId view) {
        if (impl_)
            impl_->invalidateView(view);
    }

    void VulkanSceneTemporalResolvePass::resetAll() {
        if (!impl_)
            return;
        for (std::size_t index = 0; index < static_cast<std::size_t>(TemporalViewId::Count); ++index)
            impl_->invalidateView(static_cast<TemporalViewId>(index));
    }

    void VulkanSceneTemporalResolvePass::releaseHistory() {
        if (impl_)
            impl_->releaseHistory();
    }

    void VulkanSceneTemporalResolvePass::shutdown() {
        impl_.reset();
    }

    VkImageView VulkanSceneTemporalResolvePass::outputView(const TemporalViewId view) const {
        if (!impl_ || !validTemporalViewId(view))
            return VK_NULL_HANDLE;
        const auto& resource = impl_->views.at(viewIndex(view));
        return resource.has_history ? resource.images[resource.read_index].view : VK_NULL_HANDLE;
    }

    SceneHistoryContract VulkanSceneTemporalResolvePass::contract(const TemporalViewId view) const {
        return impl_ && validTemporalViewId(view) ? impl_->views[viewIndex(view)].contract
                                                  : SceneHistoryContract{};
    }

    bool VulkanSceneTemporalResolvePass::initialized() const {
        return impl_ && impl_->pipeline != VK_NULL_HANDLE;
    }

    VulkanSceneTemporalResourceStats VulkanSceneTemporalResolvePass::resourceStats() const {
        return impl_ ? impl_->resourceStats() : VulkanSceneTemporalResourceStats{};
    }
} // namespace lfs::vis
