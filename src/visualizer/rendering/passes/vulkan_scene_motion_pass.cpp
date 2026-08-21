/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_scene_motion_pass.hpp"

#include "core/logger.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "rendering/scene_motion_reprojection.hpp"
#include "window/vulkan_barrier2.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_result.hpp"

#include <array>
#include <cstring>
#include <format>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>

#include "viewport/scene_motion_reproject.comp.spv.h"

namespace lfs::vis {

    namespace {
        struct alignas(16) MotionUniform {
            glm::mat4 inverse_current_view_projection{1.0f};
            glm::mat4 previous_view_projection{1.0f};
            glm::ivec4 render_info{0};
            glm::vec4 depth_info{0.0f};
        };
        static_assert(sizeof(MotionUniform) == 160);
    } // namespace

    struct VulkanSceneMotionPass::Impl {
        struct FrameResource {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation image_allocation = VK_NULL_HANDLE;
            VkImageView image_view = VK_NULL_HANDLE;
            VkBuffer uniform_buffer = VK_NULL_HANDLE;
            VmaAllocation uniform_allocation = VK_NULL_HANDLE;
            glm::ivec2 extent{0, 0};
            bool image_initialized = false;
            std::string vram_label;
        };

        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
        VkSampler depth_sampler = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        std::vector<FrameResource> frames;
        std::array<std::uint32_t, 3> max_group_count{};

        ~Impl() { destroy(); }

        [[nodiscard]] bool init(VulkanContext& ctx) {
            context = &ctx;
            device = ctx.device();
            allocator = ctx.allocator();
            pipeline_cache = ctx.pipelineCache();
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE) {
                return false;
            }
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(ctx.physicalDevice(), &properties);
            max_group_count = {
                properties.limits.maxComputeWorkGroupCount[0],
                properties.limits.maxComputeWorkGroupCount[1],
                properties.limits.maxComputeWorkGroupCount[2],
            };
            VkFormatProperties format_properties{};
            vkGetPhysicalDeviceFormatProperties(
                ctx.physicalDevice(), VK_FORMAT_R16G16_SFLOAT, &format_properties);
            if ((format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) == 0) {
                LOG_ERROR("Scene-motion RG16F storage images are unsupported on this device");
                return false;
            }
            return true;
        }

        void destroyFrame(FrameResource& frame) {
            if (!frame.vram_label.empty()) {
                lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                    "vulkan.scene_motion.image", frame.vram_label, 0);
                frame.vram_label.clear();
            }
            if (frame.image_view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, frame.image_view, nullptr);
            }
            if (frame.image != VK_NULL_HANDLE) {
                vmaDestroyImage(allocator, frame.image, frame.image_allocation);
            }
            if (frame.uniform_buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, frame.uniform_buffer, frame.uniform_allocation);
            }
            frame = {};
        }

        void destroy() {
            for (auto& frame : frames) {
                destroyFrame(frame);
            }
            frames.clear();
            destroyStaticResources();
        }

        void destroyStaticResources() {
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, pipeline, nullptr);
                pipeline = VK_NULL_HANDLE;
            }
            if (pipeline_layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
                pipeline_layout = VK_NULL_HANDLE;
            }
            if (descriptor_layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
                descriptor_layout = VK_NULL_HANDLE;
            }
            if (depth_sampler != VK_NULL_HANDLE) {
                vkDestroySampler(device, depth_sampler, nullptr);
                depth_sampler = VK_NULL_HANDLE;
            }
        }

        [[nodiscard]] bool createStaticResources() {
            if (pipeline != VK_NULL_HANDLE) {
                return true;
            }

            VkSamplerCreateInfo sampler_info{};
            sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sampler_info.magFilter = VK_FILTER_NEAREST;
            sampler_info.minFilter = VK_FILTER_NEAREST;
            sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (!vk_try_bool(
                    vkCreateSampler(device, &sampler_info, nullptr, &depth_sampler),
                    "vkCreateSampler(device, &sampler_info, nullptr, &depth_sampler)",
                    "Scene-motion depth sampler creation failed")) {
                destroyStaticResources();
                return false;
            }
            context->setDebugObjectName(
                VK_OBJECT_TYPE_SAMPLER, depth_sampler, "scene_motion.depth.sampler");

            std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
            bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                           VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                           VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            bindings[2] = {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                           VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo descriptor_info{};
            descriptor_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptor_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            descriptor_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
            descriptor_info.pBindings = bindings.data();
            if (!vk_try_bool(
                    vkCreateDescriptorSetLayout(
                        device, &descriptor_info, nullptr, &descriptor_layout),
                    "vkCreateDescriptorSetLayout(device, &descriptor_info, nullptr, &descriptor_layout)",
                    "Scene-motion descriptor layout creation failed")) {
                destroyStaticResources();
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                        descriptor_layout,
                                        "scene_motion.descriptor.layout");

            VkPipelineLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layout_info.setLayoutCount = 1;
            layout_info.pSetLayouts = &descriptor_layout;
            if (!vk_try_bool(
                    vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout),
                    "vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout)",
                    "Scene-motion pipeline layout creation failed")) {
                destroyStaticResources();
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                        pipeline_layout,
                                        "scene_motion.pipeline.layout");

            VkShaderModuleCreateInfo shader_info{};
            shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shader_info.codeSize = sizeof(viewport_shaders::kSceneMotionReprojectCompSpv);
            shader_info.pCode = viewport_shaders::kSceneMotionReprojectCompSpv;
            VkShaderModule shader = VK_NULL_HANDLE;
            if (!vk_try_bool(
                    vkCreateShaderModule(device, &shader_info, nullptr, &shader),
                    "vkCreateShaderModule(device, &shader_info, nullptr, &shader)",
                    "Scene-motion shader module creation failed")) {
                destroyStaticResources();
                return false;
            }

            VkPipelineShaderStageCreateInfo stage{};
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            VkComputePipelineCreateInfo pipeline_info{};
            pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipeline_info.stage = stage;
            pipeline_info.layout = pipeline_layout;
            const VkResult pipeline_result = vkCreateComputePipelines(
                device, pipeline_cache, 1, &pipeline_info, nullptr, &pipeline);
            vkDestroyShaderModule(device, shader, nullptr);
            if (!vk_try_bool(pipeline_result,
                             "vkCreateComputePipelines(device, pipeline_cache, 1, &pipeline_info, nullptr, &pipeline)",
                             "Scene-motion compute pipeline creation failed")) {
                destroyStaticResources();
                return false;
            }
            context->setDebugObjectName(
                VK_OBJECT_TYPE_PIPELINE, pipeline, "scene_motion.pipeline");
            return true;
        }

        [[nodiscard]] bool createUniform(FrameResource& frame, const std::size_t frame_slot) {
            VkBufferCreateInfo buffer_info{};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.size = sizeof(MotionUniform);
            buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo allocation_info{};
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            if (!vk_try_bool(
                    vmaCreateBuffer(allocator,
                                    &buffer_info,
                                    &allocation_info,
                                    &frame.uniform_buffer,
                                    &frame.uniform_allocation,
                                    nullptr),
                    "vmaCreateBuffer(scene_motion.uniform)",
                    "Scene-motion uniform allocation failed")) {
                return false;
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                         frame.uniform_buffer,
                                         "scene_motion.uniform[{}]",
                                         frame_slot);
            return true;
        }

        [[nodiscard]] bool createImage(FrameResource& frame,
                                       const glm::ivec2 extent,
                                       const std::size_t frame_slot) {
            VkImageCreateInfo image_info{};
            image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.format = VK_FORMAT_R16G16_SFLOAT;
            image_info.extent = {static_cast<std::uint32_t>(extent.x),
                                 static_cast<std::uint32_t>(extent.y), 1};
            image_info.mipLevels = 1;
            image_info.arrayLayers = 1;
            image_info.samples = VK_SAMPLE_COUNT_1_BIT;
            image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo allocation_info{};
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            VmaAllocationInfo result_info{};
            if (!vk_try_bool(
                    vmaCreateImage(allocator,
                                   &image_info,
                                   &allocation_info,
                                   &frame.image,
                                   &frame.image_allocation,
                                   &result_info),
                    "vmaCreateImage(scene_motion.image)",
                    "Scene-motion image allocation failed")) {
                return false;
            }

            VkImageViewCreateInfo view_info{};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = frame.image;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = image_info.format;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.layerCount = 1;
            if (!vk_try_bool(
                    vkCreateImageView(device, &view_info, nullptr, &frame.image_view),
                    "vkCreateImageView(device, &view_info, nullptr, &frame.image_view)",
                    "Scene-motion image view creation failed")) {
                return false;
            }
            frame.extent = extent;
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE,
                                         frame.image,
                                         "scene_motion.image[{}][{}x{}]",
                                         frame_slot,
                                         extent.x,
                                         extent.y);
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW,
                                         frame.image_view,
                                         "scene_motion.image[{}].view",
                                         frame_slot);
            frame.vram_label = std::format("rg16f:{}:{}x{}", frame_slot, extent.x, extent.y);
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.scene_motion.image",
                frame.vram_label,
                static_cast<std::size_t>(result_info.size));
            return true;
        }

        [[nodiscard]] bool ensureFrame(const std::size_t frame_slot,
                                       const glm::ivec2 extent) {
            if (frame_slot >= frames.size()) {
                frames.resize(frame_slot + 1);
            }
            auto& frame = frames[frame_slot];
            if (frame.image != VK_NULL_HANDLE && frame.extent == extent) {
                return true;
            }
            if (frame.image != VK_NULL_HANDLE && !context->waitForSubmittedFrames()) {
                LOG_ERROR("VulkanSceneMotionPass: could not retire frames before resize: {}",
                          context->lastError());
                return false;
            }
            destroyFrame(frame);
            return createUniform(frame, frame_slot) && createImage(frame, extent, frame_slot);
        }

        [[nodiscard]] bool updateUniform(FrameResource& frame,
                                         const VulkanSceneMotionParams& params) {
            MotionUniform uniform{
                .inverse_current_view_projection = params.inverse_current_view_projection,
                .previous_view_projection = params.previous_view_projection,
                .render_info = {params.render_extent.x,
                                params.render_extent.y,
                                params.flip_y ? 1 : 0,
                                params.depth.encoding == SceneDepthEncoding::LinearView
                                    ? (params.depth.orthographic ? 2 : 1)
                                    : 0},
                .depth_info = {params.depth.near_plane,
                               params.depth.far_plane,
                               params.depth.flip_y ? 1.0f : 0.0f,
                               0.0f},
            };
            void* mapped = nullptr;
            if (!vk_try_bool(vmaMapMemory(allocator, frame.uniform_allocation, &mapped),
                             "vmaMapMemory(allocator, frame.uniform_allocation, &mapped)",
                             "Scene-motion uniform mapping failed")) {
                return false;
            }
            std::memcpy(mapped, &uniform, sizeof(uniform));
            const VkResult flush_result =
                vmaFlushAllocation(allocator, frame.uniform_allocation, 0, sizeof(uniform));
            vmaUnmapMemory(allocator, frame.uniform_allocation);
            return vk_try_bool(flush_result,
                               "vmaFlushAllocation(allocator, frame.uniform_allocation, 0, sizeof(uniform))",
                               "Scene-motion uniform flush failed");
        }

        [[nodiscard]] bool record(VkCommandBuffer command_buffer,
                                  const VulkanSceneMotionParams& params,
                                  const std::size_t frame_slot) {
            if (!params.enabled) {
                return true;
            }
            const SceneMotionReprojectionParams reprojection{
                .inverse_current_view_projection = params.inverse_current_view_projection,
                .previous_view_projection = params.previous_view_projection,
                .render_extent = params.render_extent,
                .flip_y = params.flip_y,
            };
            if (command_buffer == VK_NULL_HANDLE || !canRecordVulkanSceneMotion(params) ||
                !reprojection.valid()) {
                return false;
            }
            if (!createStaticResources() || !ensureFrame(frame_slot, params.render_extent)) {
                return false;
            }
            auto& frame = frames[frame_slot];
            if (!updateUniform(frame, params)) {
                return false;
            }
            const auto group_x = (static_cast<std::uint32_t>(params.render_extent.x) + 7u) / 8u;
            const auto group_y = (static_cast<std::uint32_t>(params.render_extent.y) + 7u) / 8u;
            if (group_x == 0 || group_y == 0 || group_x > max_group_count[0] ||
                group_y > max_group_count[1] || max_group_count[2] == 0) {
                LOG_ERROR("Scene-motion dispatch {}x{} exceeds device limits {}x{}x{}",
                          group_x,
                          group_y,
                          max_group_count[0],
                          max_group_count[1],
                          max_group_count[2]);
                return false;
            }

            cmdImageBarrier2(command_buffer,
                             frame.image,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             frame.image_initialized ? VK_IMAGE_LAYOUT_GENERAL
                                                     : VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL,
                             frame.image_initialized
                                 ? (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)
                                 : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             frame.image_initialized
                                 ? (VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)
                                 : VK_ACCESS_2_NONE,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            VkDescriptorImageInfo depth_info{};
            depth_info.sampler = depth_sampler;
            depth_info.imageView = params.depth_view;
            depth_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkDescriptorImageInfo motion_info{};
            motion_info.imageView = frame.image_view;
            motion_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorBufferInfo uniform_info{};
            uniform_info.buffer = frame.uniform_buffer;
            uniform_info.range = sizeof(MotionUniform);
            std::array<VkWriteDescriptorSet, 3> writes{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[0].pImageInfo = &depth_info;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[1].pImageInfo = &motion_info;
            writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2].dstBinding = 2;
            writes[2].descriptorCount = 1;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[2].pBufferInfo = &uniform_info;

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            context->vkCmdPushDescriptorSet()(command_buffer,
                                              VK_PIPELINE_BIND_POINT_COMPUTE,
                                              pipeline_layout,
                                              0,
                                              static_cast<std::uint32_t>(writes.size()),
                                              writes.data());
            vkCmdDispatch(command_buffer, group_x, group_y, 1);
            cmdImageBarrier2(command_buffer,
                             frame.image,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            frame.image_initialized = true;
            return true;
        }
    };

    VulkanSceneMotionPass::VulkanSceneMotionPass() = default;
    VulkanSceneMotionPass::~VulkanSceneMotionPass() = default;
    VulkanSceneMotionPass::VulkanSceneMotionPass(VulkanSceneMotionPass&&) noexcept = default;
    VulkanSceneMotionPass& VulkanSceneMotionPass::operator=(VulkanSceneMotionPass&&) noexcept = default;

    bool VulkanSceneMotionPass::init(VulkanContext& context) {
        if (!impl_) {
            impl_ = std::make_unique<Impl>();
        }
        return impl_->init(context);
    }

    bool VulkanSceneMotionPass::record(VkCommandBuffer command_buffer,
                                       const VulkanSceneMotionParams& params,
                                       const std::size_t frame_slot) {
        return impl_ && impl_->record(command_buffer, params, frame_slot);
    }

    void VulkanSceneMotionPass::shutdown() {
        if (impl_) {
            impl_->destroy();
            impl_.reset();
        }
    }

    VkImageView VulkanSceneMotionPass::motionView(const std::size_t frame_slot) const {
        return impl_ && frame_slot < impl_->frames.size()
                   ? impl_->frames[frame_slot].image_view
                   : VK_NULL_HANDLE;
    }

    VkImage VulkanSceneMotionPass::motionImage(const std::size_t frame_slot) const {
        return impl_ && frame_slot < impl_->frames.size() ? impl_->frames[frame_slot].image
                                                          : VK_NULL_HANDLE;
    }

    bool VulkanSceneMotionPass::initialized() const {
        return impl_ && impl_->pipeline != VK_NULL_HANDLE;
    }

} // namespace lfs::vis
