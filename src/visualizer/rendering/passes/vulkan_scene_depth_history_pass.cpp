/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_scene_depth_history_pass.hpp"

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

#include "viewport/scene_depth_history.comp.spv.h"

namespace lfs::vis {
    namespace {
        struct alignas(16) DepthHistoryPush {
            glm::ivec4 extent_encoding{0};
            glm::vec4 planes{0.0f};
            glm::vec4 source_uv{0.0f};
        };
        static_assert(sizeof(DepthHistoryPush) == 48);
    } // namespace

    struct VulkanSceneDepthHistoryPass::Impl {
        struct Resource {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            glm::ivec2 extent{0, 0};
            bool initialized = false;
            SceneDepthContract contract{};
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

        bool init(VulkanContext& ctx) {
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
            vkGetPhysicalDeviceFormatProperties(ctx.physicalDevice(), VK_FORMAT_R32_SFLOAT, &format);
            if ((format.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) == 0) {
                LOG_ERROR("Scene depth-history R32F storage images are unsupported");
                return false;
            }
            return true;
        }

        void destroyResource(Resource& resource) {
            if (!resource.vram_label.empty())
                lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                    "vulkan.scene_temporal.depth_history", resource.vram_label, 0);
            if (resource.view != VK_NULL_HANDLE)
                vkDestroyImageView(device, resource.view, nullptr);
            if (resource.image != VK_NULL_HANDLE)
                vmaDestroyImage(allocator, resource.image, resource.allocation);
            resource = {};
        }

        void destroyStatic() {
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
            destroyStatic();
        }

        bool createStatic() {
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
                             "vkCreateSampler(scene_depth_history)",
                             "Scene depth-history sampler creation failed")) {
                destroyStatic();
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
                             "vkCreateDescriptorSetLayout(scene_depth_history)",
                             "Scene depth-history descriptor creation failed")) {
                destroyStatic();
                return false;
            }
            const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                            sizeof(DepthHistoryPush)};
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.setLayoutCount = 1;
            layout_info.pSetLayouts = &descriptor_layout;
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            if (!vk_try_bool(vkCreatePipelineLayout(
                                 device, &layout_info, nullptr, &pipeline_layout),
                             "vkCreatePipelineLayout(scene_depth_history)",
                             "Scene depth-history pipeline layout creation failed")) {
                destroyStatic();
                return false;
            }
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = sizeof(viewport_shaders::kSceneDepthHistoryCompSpv);
            shader_info.pCode = viewport_shaders::kSceneDepthHistoryCompSpv;
            VkShaderModule shader = VK_NULL_HANDLE;
            if (!vk_try_bool(vkCreateShaderModule(device, &shader_info, nullptr, &shader),
                             "vkCreateShaderModule(scene_depth_history)",
                             "Scene depth-history shader creation failed")) {
                destroyStatic();
                return false;
            }
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            info.stage = stage;
            info.layout = pipeline_layout;
            const VkResult result = vkCreateComputePipelines(
                device, pipeline_cache, 1, &info, nullptr, &pipeline);
            vkDestroyShaderModule(device, shader, nullptr);
            if (!vk_try_bool(result, "vkCreateComputePipelines(scene_depth_history)",
                             "Scene depth-history pipeline creation failed")) {
                destroyStatic();
                return false;
            }
            return true;
        }

        bool createImage(Resource& resource, glm::ivec2 extent, std::size_t slot) {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = VK_FORMAT_R32_SFLOAT;
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
            if (!vk_try_bool(vmaCreateImage(allocator, &info, &allocation_info,
                                            &resource.image, &resource.allocation,
                                            &allocation_result),
                             "vmaCreateImage(scene_depth_history)",
                             "Scene depth-history allocation failed"))
                return false;
            VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view_info.image = resource.image;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = info.format;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.layerCount = 1;
            if (!vk_try_bool(vkCreateImageView(device, &view_info, nullptr, &resource.view),
                             "vkCreateImageView(scene_depth_history)",
                             "Scene depth-history view creation failed"))
                return false;
            resource.extent = extent;
            resource.vram_label = std::format("slot{}:{}x{}", slot, extent.x, extent.y);
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.scene_temporal.depth_history", resource.vram_label,
                static_cast<std::size_t>(allocation_result.size));
            return true;
        }

        bool ensure(std::size_t slot, glm::ivec2 extent) {
            if (slot >= resources.size())
                resources.resize(slot + 1);
            auto& resource = resources[slot];
            if (resource.image != VK_NULL_HANDLE && resource.extent == extent)
                return true;
            if (resource.image != VK_NULL_HANDLE && !context->waitForSubmittedFrames()) {
                LOG_ERROR("Could not retire Vulkan frames before depth-history resize: {}",
                          context->lastError());
                return false;
            }
            destroyResource(resource);
            if (!createImage(resource, extent, slot)) {
                destroyResource(resource);
                return false;
            }
            return true;
        }

        bool record(VkCommandBuffer command_buffer,
                    const VulkanSceneDepthHistoryParams& params,
                    std::size_t slot) {
            if (!params.enabled) {
                if (slot < resources.size())
                    resources[slot].contract = {};
                return true;
            }
            const glm::ivec2 extent{params.depth.width, params.depth.height};
            const glm::ivec2 allocation = params.allocation_extent.x > 0 &&
                                                  params.allocation_extent.y > 0
                                              ? params.allocation_extent
                                              : extent;
            const auto uv = sceneDepthHistoryUvTransform(extent, allocation);
            const auto encoding = sceneDepthHistoryEncodingCode(params.depth);
            if (command_buffer == VK_NULL_HANDLE || !canRecordVulkanSceneDepthHistory(params) ||
                !uv || encoding == 0 || !createStatic() || !ensure(slot, extent))
                return false;
            const std::uint32_t gx = (static_cast<std::uint32_t>(extent.x) + 7u) / 8u;
            const std::uint32_t gy = (static_cast<std::uint32_t>(extent.y) + 7u) / 8u;
            if (gx == 0 || gy == 0 || gx > max_group_count[0] || gy > max_group_count[1])
                return false;
            auto& resource = resources[slot];
            cmdImageBarrier2(command_buffer, resource.image, VK_IMAGE_ASPECT_COLOR_BIT,
                             resource.initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL,
                             resource.initialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                                  : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             resource.initialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                                                  : VK_ACCESS_2_NONE,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            VkDescriptorImageInfo source{sampler, params.current_depth_view,
                                         params.current_depth_layout};
            VkDescriptorImageInfo output{VK_NULL_HANDLE, resource.view, VK_IMAGE_LAYOUT_GENERAL};
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
            const DepthHistoryPush push{
                .extent_encoding = {extent.x, extent.y, static_cast<int>(encoding),
                                    params.depth.flip_y ? 1 : 0},
                .planes = {params.depth.near_plane, params.depth.far_plane, 0.0f, 0.0f},
                .source_uv = *uv};
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            context->vkCmdPushDescriptorSet()(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                              pipeline_layout, 0,
                                              static_cast<std::uint32_t>(writes.size()),
                                              writes.data());
            vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(push), &push);
            vkCmdDispatch(command_buffer, gx, gy, 1);
            cmdImageBarrier2(command_buffer, resource.image, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            resource.initialized = true;
            resource.contract = makeVulkanSceneDepthHistoryContract(params);
            return resource.contract.valid();
        }
    };

    VulkanSceneDepthHistoryPass::VulkanSceneDepthHistoryPass() = default;
    VulkanSceneDepthHistoryPass::~VulkanSceneDepthHistoryPass() = default;
    VulkanSceneDepthHistoryPass::VulkanSceneDepthHistoryPass(VulkanSceneDepthHistoryPass&&) noexcept = default;
    VulkanSceneDepthHistoryPass& VulkanSceneDepthHistoryPass::operator=(VulkanSceneDepthHistoryPass&&) noexcept = default;

    bool VulkanSceneDepthHistoryPass::init(VulkanContext& context) {
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        return impl_->init(context);
    }
    bool VulkanSceneDepthHistoryPass::record(VkCommandBuffer command_buffer,
                                             const VulkanSceneDepthHistoryParams& params,
                                             std::size_t resource_slot) {
        return impl_ && impl_->record(command_buffer, params, resource_slot);
    }
    void VulkanSceneDepthHistoryPass::invalidate(std::size_t resource_slot) {
        if (impl_ && resource_slot < impl_->resources.size())
            impl_->resources[resource_slot].contract = {};
    }
    void VulkanSceneDepthHistoryPass::invalidateAll() {
        if (impl_)
            for (auto& resource : impl_->resources)
                resource.contract = {};
    }
    void VulkanSceneDepthHistoryPass::shutdown() { impl_.reset(); }
    VkImageView VulkanSceneDepthHistoryPass::depthView(std::size_t slot) const {
        return impl_ && slot < impl_->resources.size() ? impl_->resources[slot].view
                                                       : VK_NULL_HANDLE;
    }
    VkImage VulkanSceneDepthHistoryPass::depthImage(std::size_t slot) const {
        return impl_ && slot < impl_->resources.size() ? impl_->resources[slot].image
                                                       : VK_NULL_HANDLE;
    }
    SceneDepthContract VulkanSceneDepthHistoryPass::contract(std::size_t slot) const {
        return impl_ && slot < impl_->resources.size() ? impl_->resources[slot].contract
                                                       : SceneDepthContract{};
    }
    bool VulkanSceneDepthHistoryPass::initialized() const {
        return impl_ && impl_->pipeline != VK_NULL_HANDLE;
    }
} // namespace lfs::vis
