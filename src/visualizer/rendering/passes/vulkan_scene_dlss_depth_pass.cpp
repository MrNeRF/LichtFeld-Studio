/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_scene_dlss_depth_pass.hpp"

#include "core/logger.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "window/vulkan_barrier2.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_result.hpp"

#include <array>
#include <format>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>

#include "viewport/scene_dlss_depth.comp.spv.h"

namespace lfs::vis {
    namespace {
        struct alignas(16) DlssDepthPush {
            glm::ivec4 extent_encoding{0};
            glm::vec4 planes{0.0f};
        };
        static_assert(sizeof(DlssDepthPush) == 32);
    } // namespace

    struct VulkanSceneDlssDepthPass::Impl {
        struct Resource {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            glm::ivec2 extent{0, 0};
            bool initialized = false;
            std::string vram_label;
        };

        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        std::vector<Resource> resources;
        std::array<std::uint32_t, 3> max_group_count{};

        ~Impl() { destroy(); }

        [[nodiscard]] bool init(VulkanContext& ctx) {
            context = &ctx;
            device = ctx.device();
            allocator = ctx.allocator();
            pipeline_cache = ctx.pipelineCache();
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE ||
                ctx.vkCmdPushDescriptorSet() == nullptr)
                return false;

            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(ctx.physicalDevice(), &properties);
            max_group_count = {properties.limits.maxComputeWorkGroupCount[0],
                               properties.limits.maxComputeWorkGroupCount[1],
                               properties.limits.maxComputeWorkGroupCount[2]};

            VkFormatProperties format{};
            vkGetPhysicalDeviceFormatProperties(
                ctx.physicalDevice(), VK_FORMAT_R32_SFLOAT, &format);
            const auto required = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
                                  VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
            if ((format.optimalTilingFeatures & required) != required) {
                LOG_ERROR("DLSS R32F normalized-depth images are unsupported");
                return false;
            }
            return true;
        }

        void destroyResource(Resource& resource) {
            if (!resource.vram_label.empty()) {
                lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                    "vulkan.scene_dlss.depth", resource.vram_label, 0);
            }
            if (resource.view != VK_NULL_HANDLE)
                vkDestroyImageView(device, resource.view, nullptr);
            if (resource.image != VK_NULL_HANDLE)
                vmaDestroyImage(allocator, resource.image, resource.allocation);
            resource = {};
        }

        void destroyStaticResources() {
            if (pipeline != VK_NULL_HANDLE)
                vkDestroyPipeline(device, pipeline, nullptr);
            if (pipeline_layout != VK_NULL_HANDLE)
                vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            if (descriptor_layout != VK_NULL_HANDLE)
                vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
            if (sampler != VK_NULL_HANDLE)
                vkDestroySampler(device, sampler, nullptr);
            pipeline = VK_NULL_HANDLE;
            pipeline_layout = VK_NULL_HANDLE;
            descriptor_layout = VK_NULL_HANDLE;
            sampler = VK_NULL_HANDLE;
        }

        void destroy() {
            for (auto& resource : resources)
                destroyResource(resource);
            resources.clear();
            destroyStaticResources();
        }

        [[nodiscard]] bool createStaticResources() {
            if (pipeline != VK_NULL_HANDLE)
                return true;

            VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            sampler_info.magFilter = VK_FILTER_NEAREST;
            sampler_info.minFilter = VK_FILTER_NEAREST;
            sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (!vk_try_bool(vkCreateSampler(device, &sampler_info, nullptr, &sampler),
                             "vkCreateSampler(scene_dlss_depth)",
                             "DLSS depth sampler creation failed")) {
                destroyStaticResources();
                return false;
            }

            std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
            bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                           VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                           VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo descriptor_info{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            descriptor_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            descriptor_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
            descriptor_info.pBindings = bindings.data();
            if (!vk_try_bool(vkCreateDescriptorSetLayout(
                                 device, &descriptor_info, nullptr, &descriptor_layout),
                             "vkCreateDescriptorSetLayout(scene_dlss_depth)",
                             "DLSS depth descriptor layout creation failed")) {
                destroyStaticResources();
                return false;
            }

            const VkPushConstantRange push_range{
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DlssDepthPush)};
            VkPipelineLayoutCreateInfo layout_info{
                VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.setLayoutCount = 1;
            layout_info.pSetLayouts = &descriptor_layout;
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &push_range;
            if (!vk_try_bool(vkCreatePipelineLayout(
                                 device, &layout_info, nullptr, &pipeline_layout),
                             "vkCreatePipelineLayout(scene_dlss_depth)",
                             "DLSS depth pipeline layout creation failed")) {
                destroyStaticResources();
                return false;
            }

            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = sizeof(viewport_shaders::kSceneDlssDepthCompSpv);
            shader_info.pCode = viewport_shaders::kSceneDlssDepthCompSpv;
            VkShaderModule shader = VK_NULL_HANDLE;
            if (!vk_try_bool(vkCreateShaderModule(device, &shader_info, nullptr, &shader),
                             "vkCreateShaderModule(scene_dlss_depth)",
                             "DLSS depth shader creation failed")) {
                destroyStaticResources();
                return false;
            }

            VkPipelineShaderStageCreateInfo stage{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            VkComputePipelineCreateInfo pipeline_info{
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline_info.stage = stage;
            pipeline_info.layout = pipeline_layout;
            const VkResult result = vkCreateComputePipelines(
                device, pipeline_cache, 1, &pipeline_info, nullptr, &pipeline);
            vkDestroyShaderModule(device, shader, nullptr);
            if (!vk_try_bool(result,
                             "vkCreateComputePipelines(scene_dlss_depth)",
                             "DLSS depth compute pipeline creation failed")) {
                destroyStaticResources();
                return false;
            }
            return true;
        }

        [[nodiscard]] bool createResource(Resource& resource,
                                          const glm::ivec2 extent,
                                          const std::size_t slot) {
            VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.format = VK_FORMAT_R32_SFLOAT;
            image_info.extent = {static_cast<std::uint32_t>(extent.x),
                                 static_cast<std::uint32_t>(extent.y), 1};
            image_info.mipLevels = 1;
            image_info.arrayLayers = 1;
            image_info.samples = VK_SAMPLE_COUNT_1_BIT;
            image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo allocation_info{};
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            VmaAllocationInfo allocation_result{};
            if (!vk_try_bool(vmaCreateImage(allocator,
                                            &image_info,
                                            &allocation_info,
                                            &resource.image,
                                            &resource.allocation,
                                            &allocation_result),
                             "vmaCreateImage(scene_dlss_depth)",
                             "DLSS normalized-depth allocation failed"))
                return false;

            VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view_info.image = resource.image;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = image_info.format;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.layerCount = 1;
            if (!vk_try_bool(vkCreateImageView(
                                 device, &view_info, nullptr, &resource.view),
                             "vkCreateImageView(scene_dlss_depth)",
                             "DLSS normalized-depth view creation failed"))
                return false;

            resource.extent = extent;
            resource.vram_label = std::format("slot{}:{}x{}", slot, extent.x, extent.y);
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.scene_dlss.depth",
                resource.vram_label,
                static_cast<std::size_t>(allocation_result.size));
            context->setDebugObjectName(
                VK_OBJECT_TYPE_IMAGE, resource.image, "scene_dlss.depth");
            context->setDebugObjectName(
                VK_OBJECT_TYPE_IMAGE_VIEW, resource.view, "scene_dlss.depth.view");
            return true;
        }

        [[nodiscard]] bool ensureResource(const std::size_t slot,
                                          const glm::ivec2 extent) {
            if (slot >= resources.size())
                resources.resize(slot + 1);
            auto& resource = resources[slot];
            if (resource.image != VK_NULL_HANDLE && resource.extent == extent)
                return true;
            if (resource.image != VK_NULL_HANDLE && !context->waitForSubmittedFrames()) {
                LOG_ERROR("Could not retire Vulkan frames before DLSS depth resize: {}",
                          context->lastError());
                return false;
            }
            destroyResource(resource);
            if (!createResource(resource, extent, slot)) {
                destroyResource(resource);
                return false;
            }
            return true;
        }

        [[nodiscard]] bool record(const VkCommandBuffer command_buffer,
                                  const VulkanSceneDlssDepthParams& params,
                                  const std::size_t slot) {
            const glm::ivec2 extent{params.depth.width, params.depth.height};
            const auto encoding = sceneDlssDepthEncodingCode(params.depth);
            if (command_buffer == VK_NULL_HANDLE ||
                !canRecordVulkanSceneDlssDepth(params) || encoding == 0 ||
                !createStaticResources() || !ensureResource(slot, extent))
                return false;

            const std::uint32_t group_x =
                (static_cast<std::uint32_t>(extent.x) + 7u) / 8u;
            const std::uint32_t group_y =
                (static_cast<std::uint32_t>(extent.y) + 7u) / 8u;
            if (group_x == 0 || group_y == 0 || group_x > max_group_count[0] ||
                group_y > max_group_count[1])
                return false;

            auto& resource = resources[slot];
            cmdImageBarrier2(command_buffer,
                             resource.image,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             resource.initialized
                                 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                 : VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL,
                             resource.initialized
                                 ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                 : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             resource.initialized
                                 ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                                 : VK_ACCESS_2_NONE,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            VkDescriptorImageInfo source{
                sampler, params.current_depth_view, params.current_depth_layout};
            VkDescriptorImageInfo output{
                VK_NULL_HANDLE, resource.view, VK_IMAGE_LAYOUT_GENERAL};
            std::array<VkWriteDescriptorSet, 2> writes{};
            writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[0].pImageInfo = &source;
            writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[1].pImageInfo = &output;

            const DlssDepthPush push{
                .extent_encoding = {extent.x,
                                    extent.y,
                                    static_cast<int>(encoding),
                                    params.depth.flip_y ? 1 : 0},
                .planes = {params.depth.near_plane,
                           params.depth.far_plane,
                           0.0f,
                           0.0f},
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
            vkCmdDispatch(command_buffer, group_x, group_y, 1);
            cmdImageBarrier2(command_buffer,
                             resource.image,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            resource.initialized = true;
            return true;
        }
    };

    VulkanSceneDlssDepthPass::VulkanSceneDlssDepthPass() = default;
    VulkanSceneDlssDepthPass::~VulkanSceneDlssDepthPass() = default;
    VulkanSceneDlssDepthPass::VulkanSceneDlssDepthPass(
        VulkanSceneDlssDepthPass&&) noexcept = default;
    VulkanSceneDlssDepthPass& VulkanSceneDlssDepthPass::operator=(
        VulkanSceneDlssDepthPass&&) noexcept = default;

    bool VulkanSceneDlssDepthPass::init(VulkanContext& context) {
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        return impl_->init(context);
    }

    bool VulkanSceneDlssDepthPass::record(
        const VkCommandBuffer command_buffer,
        const VulkanSceneDlssDepthParams& params,
        const std::size_t resource_slot) {
        return impl_ && impl_->record(command_buffer, params, resource_slot);
    }

    void VulkanSceneDlssDepthPass::shutdown() { impl_.reset(); }

    VkImageView VulkanSceneDlssDepthPass::depthView(
        const std::size_t resource_slot) const {
        return impl_ && resource_slot < impl_->resources.size()
                   ? impl_->resources[resource_slot].view
                   : VK_NULL_HANDLE;
    }

    VkImage VulkanSceneDlssDepthPass::depthImage(
        const std::size_t resource_slot) const {
        return impl_ && resource_slot < impl_->resources.size()
                   ? impl_->resources[resource_slot].image
                   : VK_NULL_HANDLE;
    }

    bool VulkanSceneDlssDepthPass::initialized() const {
        return impl_ && impl_->pipeline != VK_NULL_HANDLE;
    }

} // namespace lfs::vis
