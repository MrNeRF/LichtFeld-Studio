/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vr_hand_renderer.hpp"
#include "rendering/vulkan_wait.hpp"
#include "window/vulkan_result.hpp"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string_view>

#include <vk_mem_alloc.h>

#include "viewport/vr_hand.frag.spv.h"
#include "viewport/vr_hand.vert.spv.h"

namespace lfs::vis::vr {

    namespace {
        constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
        // Two hands; the bundled hand models have about 7,000 triangle corners each.
        constexpr std::size_t kMaxVertices = 32768;

        struct HandPush {
            glm::mat4 view_projection;
            glm::vec4 eye;
            glm::vec4 color;
        };

        [[nodiscard]] lfs::Status checkVk(const VkResult result, const std::string_view operation) {
            if (result == VK_SUCCESS)
                return {};
            return lfs::Status::failure(
                lfs::rendering::vulkan_result_to_error(result, operation, LFS_SOURCE_SITE_CURRENT()));
        }
    } // namespace

    struct HandRenderer::Impl {
        VulkanContext* context = nullptr;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        // The hands are see-through, so only their nearest surface may be
        // shaded: a first pass lays down depth, a second one colors what
        // matches it. Without it, every layer of a hand would add its light.
        VkPipeline depth_pipeline = VK_NULL_HANDLE;
        VkPipeline color_pipeline = VK_NULL_HANDLE;
        VkBuffer vertices = VK_NULL_HANDLE;
        VmaAllocation vertices_allocation = VK_NULL_HANDLE;
        HandVertex* mapped = nullptr;
        VkImage depth = VK_NULL_HANDLE;
        VmaAllocation depth_allocation = VK_NULL_HANDLE;
        VkImageView depth_view = VK_NULL_HANDLE;
        glm::ivec2 size{0};
        // Left, right: where each hand lies in the vertex buffer.
        std::array<std::uint32_t, 2> first{};
        std::array<std::uint32_t, 2> count{};
        std::array<glm::vec3, 2> colors{};

        lfs::Status createPipelines(const VkFormat color_format) {
            const VkDevice device = context->device();
            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            push.size = sizeof(HandPush);
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &push;
            if (auto st = checkVk(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout), "vkCreatePipelineLayout(VR hands)"); !st)
                return st;

            const VkShaderModule vertex = createShaderModule(device, viewport_shaders::kVrHandVertSpv, "VR hands");
            const VkShaderModule fragment = createShaderModule(device, viewport_shaders::kVrHandFragSpv, "VR hands");
            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertex;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fragment;
            stages[0].pName = stages[1].pName = "main";

            VkVertexInputBindingDescription binding{};
            binding.stride = sizeof(HandVertex);
            binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            std::array<VkVertexInputAttributeDescription, 2> attributes{};
            attributes[0].location = 0;
            attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributes[0].offset = offsetof(HandVertex, position);
            attributes[1].location = 1;
            attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributes[1].offset = offsetof(HandVertex, normal);
            VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            input.vertexBindingDescriptionCount = 1;
            input.pVertexBindingDescriptions = &binding;
            input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
            input.pVertexAttributeDescriptions = attributes.data();

            VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            viewport.viewportCount = viewport.scissorCount = 1;
            // The depth pass alone sorts the hands out, whichever way a triangle winds.
            VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_NONE;
            raster.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo depth_state{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            depth_state.depthTestEnable = VK_TRUE;
            VkPipelineColorBlendAttachmentState blend{};
            VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            blending.attachmentCount = 1;
            blending.pAttachments = &blend;
            const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
            dynamic.pDynamicStates = dynamic_states.data();
            VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachmentFormats = &color_format;
            rendering.depthAttachmentFormat = kDepthFormat;

            VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            info.pNext = &rendering;
            info.stageCount = static_cast<std::uint32_t>(stages.size());
            info.pStages = stages.data();
            info.pVertexInputState = &input;
            info.pInputAssemblyState = &assembly;
            info.pViewportState = &viewport;
            info.pRasterizationState = &raster;
            info.pMultisampleState = &multisample;
            info.pDepthStencilState = &depth_state;
            info.pColorBlendState = &blending;
            info.pDynamicState = &dynamic;
            info.layout = layout;

            VkResult created = vertex && fragment ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED;
            if (created == VK_SUCCESS) {
                // Depth only: nothing is written to the image.
                depth_state.depthWriteEnable = VK_TRUE;
                depth_state.depthCompareOp = VK_COMPARE_OP_LESS;
                blend.blendEnable = VK_FALSE;
                blend.colorWriteMask = 0;
                created = vkCreateGraphicsPipelines(device, context->pipelineCache(), 1, &info, nullptr, &depth_pipeline);
            }
            if (created == VK_SUCCESS) {
                // Premultiplied light over the scene, on the nearest surface only.
                depth_state.depthWriteEnable = VK_FALSE;
                depth_state.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
                blend.blendEnable = VK_TRUE;
                blend.srcColorBlendFactor = blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                blend.dstColorBlendFactor = blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                blend.colorBlendOp = blend.alphaBlendOp = VK_BLEND_OP_ADD;
                blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                       VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
                created = vkCreateGraphicsPipelines(device, context->pipelineCache(), 1, &info, nullptr, &color_pipeline);
            }
            if (vertex)
                vkDestroyShaderModule(device, vertex, nullptr);
            if (fragment)
                vkDestroyShaderModule(device, fragment, nullptr);
            return checkVk(created, "vkCreateGraphicsPipelines(VR hands)");
        }

        lfs::Status createBuffers() {
            const VmaAllocator allocator = context->allocator();
            VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            buffer.size = kMaxVertices * sizeof(HandVertex);
            buffer.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            buffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo buffer_allocation{};
            buffer_allocation.usage = VMA_MEMORY_USAGE_AUTO;
            buffer_allocation.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                      VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo buffer_info{};
            if (auto st = checkVk(vmaCreateBuffer(allocator, &buffer, &buffer_allocation, &vertices,
                                                  &vertices_allocation, &buffer_info),
                                  "vmaCreateBuffer(VR hands)");
                !st)
                return st;
            mapped = static_cast<HandVertex*>(buffer_info.pMappedData);
            if (!mapped)
                return checkVk(VK_ERROR_MEMORY_MAP_FAILED, "vmaCreateBuffer(VR hands, mapped)");

            VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            image.imageType = VK_IMAGE_TYPE_2D;
            image.format = kDepthFormat;
            image.extent = {static_cast<std::uint32_t>(size.x), static_cast<std::uint32_t>(size.y), 1};
            image.mipLevels = image.arrayLayers = 1;
            image.samples = VK_SAMPLE_COUNT_1_BIT;
            image.tiling = VK_IMAGE_TILING_OPTIMAL;
            image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo image_allocation{};
            image_allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            if (auto st = checkVk(vmaCreateImage(allocator, &image, &image_allocation, &depth, &depth_allocation, nullptr),
                                  "vmaCreateImage(VR hands depth)");
                !st)
                return st;
            VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view.image = depth;
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = kDepthFormat;
            view.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            return checkVk(vkCreateImageView(context->device(), &view, nullptr, &depth_view), "vkCreateImageView(VR hands depth)");
        }

        void destroy() {
            if (!context)
                return;
            const VkDevice device = context->device();
            const VmaAllocator allocator = context->allocator();
            if (depth_view)
                vkDestroyImageView(device, depth_view, nullptr);
            if (depth)
                vmaDestroyImage(allocator, depth, depth_allocation);
            if (vertices)
                vmaDestroyBuffer(allocator, vertices, vertices_allocation);
            if (color_pipeline)
                vkDestroyPipeline(device, color_pipeline, nullptr);
            if (depth_pipeline)
                vkDestroyPipeline(device, depth_pipeline, nullptr);
            if (layout)
                vkDestroyPipelineLayout(device, layout, nullptr);
            *this = {};
        }
    };

    HandRenderer::HandRenderer() : impl_(std::make_unique<Impl>()) {}
    HandRenderer::~HandRenderer() { shutdown(); }

    lfs::Status HandRenderer::init(VulkanContext& context, const VkFormat color_format, const glm::ivec2 size) {
        shutdown();
        auto& p = *impl_;
        p.context = &context;
        p.size = glm::max(size, glm::ivec2(1));
        auto created = p.createPipelines(color_format);
        if (created)
            created = p.createBuffers();
        if (!created)
            shutdown();
        return created;
    }

    void HandRenderer::shutdown() { impl_->destroy(); }

    bool HandRenderer::ready() const {
        return impl_->depth_pipeline && impl_->color_pipeline && impl_->mapped && impl_->depth_view;
    }

    void HandRenderer::setHands(const std::array<std::span<const HandVertex>, 2>& meshes,
                                const std::array<glm::vec3, 2>& colors) {
        auto& p = *impl_;
        p.first = p.count = {};
        if (!ready())
            return;
        std::size_t used = 0;
        for (std::size_t i = 0; i < meshes.size(); ++i) {
            // Whole triangles only, as many as still fit.
            const std::size_t fitting = std::min(meshes[i].size(), kMaxVertices - used) / 3 * 3;
            if (fitting == 0)
                continue;
            std::memcpy(p.mapped + used, meshes[i].data(), fitting * sizeof(HandVertex));
            p.first[i] = static_cast<std::uint32_t>(used);
            p.count[i] = static_cast<std::uint32_t>(fitting);
            p.colors[i] = colors[i];
            used += fitting;
        }
        if (used > 0)
            static_cast<void>(vmaFlushAllocation(p.context->allocator(), p.vertices_allocation, 0, used * sizeof(HandVertex)));
    }

    bool HandRenderer::hasHands() const { return impl_->count[0] + impl_->count[1] > 0; }

    void HandRenderer::record(const VkCommandBuffer command, const Eye& eye) {
        auto& p = *impl_;
        if (!ready() || !hasHands() || !eye.target || eye.size.x <= 0 || eye.size.y <= 0 ||
            eye.size.x > p.size.x || eye.size.y > p.size.y)
            return;

        // Cleared for every eye, so what the depth image held does not matter.
        constexpr VkPipelineStageFlags2 depth_stages =
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = barrier.dstStageMask = depth_stages;
        barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = p.depth;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(command, &dependency);

        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.imageView = eye.target;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView = p.depth_view;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.clearValue.depthStencil = {1.0f, 0};
        const VkRect2D area{{0, 0}, {static_cast<std::uint32_t>(eye.size.x), static_cast<std::uint32_t>(eye.size.y)}};
        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea = area;
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &color;
        rendering.pDepthAttachment = &depth;
        vkCmdBeginRendering(command, &rendering);

        const VkViewport viewport{0.0f, 0.0f, static_cast<float>(eye.size.x), static_cast<float>(eye.size.y), 0.0f, 1.0f};
        vkCmdSetViewport(command, 0, 1, &viewport);
        vkCmdSetScissor(command, 0, 1, &area);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command, 0, 1, &p.vertices, &offset);
        // Both hands lay down depth before either is colored: a hand hides
        // what is behind it of the other one too.
        for (const VkPipeline pipeline : {p.depth_pipeline, p.color_pipeline}) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            for (std::size_t i = 0; i < p.count.size(); ++i) {
                if (p.count[i] == 0)
                    continue;
                const HandPush push{eye.view_projection, glm::vec4(eye.position, 1.0f), glm::vec4(p.colors[i], 1.0f)};
                vkCmdPushConstants(command, p.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                   sizeof(push), &push);
                vkCmdDraw(command, p.count[i], 1, p.first[i], 0);
            }
        }
        vkCmdEndRendering(command);
    }

} // namespace lfs::vis::vr
