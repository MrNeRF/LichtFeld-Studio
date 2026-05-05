/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_depth_blit_pass.hpp"

#include "core/logger.hpp"
#include "window/vulkan_context.hpp"

#include <array>
#include <cstring>
#include <limits>
#include <vk_mem_alloc.h>

#include "viewport/depth_blit.frag.spv.h"
#include "viewport/screen_quad.vert.spv.h"

namespace lfs::vis {

    namespace {

        struct DepthBlitPush {
            float params[4]; // near, far, is_view_depth, flip_y
        };
        static_assert(sizeof(DepthBlitPush) == 16);

        VkShaderModule createShaderModule(VkDevice device, const std::uint32_t* code, std::size_t bytes) {
            VkShaderModuleCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            info.codeSize = bytes;
            info.pCode = code;
            VkShaderModule m = VK_NULL_HANDLE;
            if (vkCreateShaderModule(device, &info, nullptr, &m) != VK_SUCCESS) {
                return VK_NULL_HANDLE;
            }
            return m;
        }

    } // namespace

    struct VulkanDepthBlitPass::Impl {
        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkCommandPool transfer_pool = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
        VkDescriptorPool desc_pool = VK_NULL_HANDLE;
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        VkBuffer screen_quad_buffer = VK_NULL_HANDLE;

        VkImage image = VK_NULL_HANDLE;
        VmaAllocation image_alloc = VK_NULL_HANDLE;
        VkImageView image_view = VK_NULL_HANDLE;
        std::uint32_t image_width = 0;
        std::uint32_t image_height = 0;

        const lfs::core::Tensor* uploaded_tensor = nullptr;
        std::uint64_t uploaded_generation = 0;
        // Last view bound to the descriptor (either our staging-uploaded view or an
        // external interop view). Tracked so we know when to rewrite the descriptor.
        VkImageView bound_view = VK_NULL_HANDLE;
        std::uint64_t bound_generation = 0;

        // Persistent staging path: keep the upload buffer + transfer cmd between
        // frames so per-frame depth uploads don't allocate / map / submit-and-wait.
        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VmaAllocation staging_alloc = VK_NULL_HANDLE;
        void* staging_mapped = nullptr;
        VkDeviceSize staging_capacity = 0;
        VkCommandBuffer transfer_cmd = VK_NULL_HANDLE;
        VkFence transfer_fence = VK_NULL_HANDLE;

        ~Impl() { destroy(); }

        bool init(VulkanContext& ctx, VkFormat color_format, VkFormat depth_format,
                  VkBuffer screen_quad) {
            context = &ctx;
            device = ctx.device();
            allocator = ctx.allocator();
            graphics_queue = ctx.graphicsQueue();
            pipeline_cache = ctx.pipelineCache();
            screen_quad_buffer = screen_quad;
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE) {
                return false;
            }
            VkCommandPoolCreateInfo pool{};
            pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool.queueFamilyIndex = ctx.graphicsQueueFamily();
            if (vkCreateCommandPool(device, &pool, nullptr, &transfer_pool) != VK_SUCCESS) {
                return false;
            }
            return createSampler() && createDescriptors() &&
                   createPipeline(color_format, depth_format);
        }

        void destroy() {
            destroyImage();
            destroyStaging();
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, pipeline, nullptr);
                pipeline = VK_NULL_HANDLE;
            }
            if (pipeline_layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
                pipeline_layout = VK_NULL_HANDLE;
            }
            if (desc_pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, desc_pool, nullptr);
                desc_pool = VK_NULL_HANDLE;
                desc_set = VK_NULL_HANDLE;
            }
            if (desc_layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, desc_layout, nullptr);
                desc_layout = VK_NULL_HANDLE;
            }
            if (sampler != VK_NULL_HANDLE) {
                vkDestroySampler(device, sampler, nullptr);
                sampler = VK_NULL_HANDLE;
            }
            if (transfer_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, transfer_pool, nullptr);
                transfer_pool = VK_NULL_HANDLE;
            }
        }

        void destroyStaging() {
            if (transfer_cmd != VK_NULL_HANDLE && transfer_pool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, transfer_pool, 1, &transfer_cmd);
                transfer_cmd = VK_NULL_HANDLE;
            }
            if (transfer_fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, transfer_fence, nullptr);
                transfer_fence = VK_NULL_HANDLE;
            }
            if (staging_buffer != VK_NULL_HANDLE) {
                if (staging_mapped) {
                    vmaUnmapMemory(allocator, staging_alloc);
                    staging_mapped = nullptr;
                }
                vmaDestroyBuffer(allocator, staging_buffer, staging_alloc);
                staging_buffer = VK_NULL_HANDLE;
                staging_alloc = VK_NULL_HANDLE;
                staging_capacity = 0;
            }
        }

        bool ensureStaging(VkDeviceSize bytes) {
            if (staging_buffer != VK_NULL_HANDLE && staging_capacity >= bytes) {
                return true;
            }
            if (staging_buffer != VK_NULL_HANDLE) {
                if (staging_mapped) {
                    vmaUnmapMemory(allocator, staging_alloc);
                    staging_mapped = nullptr;
                }
                vmaDestroyBuffer(allocator, staging_buffer, staging_alloc);
                staging_buffer = VK_NULL_HANDLE;
                staging_alloc = VK_NULL_HANDLE;
            }
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = bytes;
            bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo sa{};
            sa.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            sa.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo ai{};
            if (vmaCreateBuffer(allocator, &bi, &sa, &staging_buffer, &staging_alloc, &ai) != VK_SUCCESS) {
                return false;
            }
            staging_mapped = ai.pMappedData;
            staging_capacity = bytes;
            return true;
        }

        bool ensureTransferCmd() {
            if (transfer_cmd != VK_NULL_HANDLE && transfer_fence != VK_NULL_HANDLE) {
                return true;
            }
            VkCommandBufferAllocateInfo a{};
            a.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            a.commandPool = transfer_pool;
            a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            a.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(device, &a, &transfer_cmd) != VK_SUCCESS) {
                return false;
            }
            VkFenceCreateInfo fi{};
            fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            return vkCreateFence(device, &fi, nullptr, &transfer_fence) == VK_SUCCESS;
        }

        bool createSampler() {
            VkSamplerCreateInfo s{};
            s.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            s.magFilter = VK_FILTER_NEAREST;
            s.minFilter = VK_FILTER_NEAREST;
            s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            s.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            s.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            s.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            return vkCreateSampler(device, &s, nullptr, &sampler) == VK_SUCCESS;
        }

        bool createDescriptors() {
            VkDescriptorSetLayoutBinding b{};
            b.binding = 0;
            b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b.descriptorCount = 1;
            b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutCreateInfo li{};
            li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            li.bindingCount = 1;
            li.pBindings = &b;
            if (vkCreateDescriptorSetLayout(device, &li, nullptr, &desc_layout) != VK_SUCCESS) {
                return false;
            }
            VkDescriptorPoolSize ps{};
            ps.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ps.descriptorCount = 1;
            VkDescriptorPoolCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pi.maxSets = 1;
            pi.poolSizeCount = 1;
            pi.pPoolSizes = &ps;
            if (vkCreateDescriptorPool(device, &pi, nullptr, &desc_pool) != VK_SUCCESS) {
                return false;
            }
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = desc_pool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &desc_layout;
            return vkAllocateDescriptorSets(device, &ai, &desc_set) == VK_SUCCESS;
        }

        bool createPipeline(VkFormat color_format, VkFormat depth_format) {
            using namespace viewport_shaders;
            VkShaderModule vert = createShaderModule(device, kScreenQuadVertSpv, sizeof(kScreenQuadVertSpv));
            VkShaderModule frag = createShaderModule(device, kDepthBlitFragSpv, sizeof(kDepthBlitFragSpv));
            if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
                if (vert)
                    vkDestroyShaderModule(device, vert, nullptr);
                if (frag)
                    vkDestroyShaderModule(device, frag, nullptr);
                return false;
            }
            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag;
            stages[1].pName = "main";

            VkVertexInputBindingDescription binding{};
            binding.binding = 0;
            binding.stride = sizeof(float) * 4; // x, y, u, v
            binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            std::array<VkVertexInputAttributeDescription, 2> attrs{};
            attrs[0].location = 0;
            attrs[0].binding = 0;
            attrs[0].format = VK_FORMAT_R32G32_SFLOAT;
            attrs[0].offset = 0;
            attrs[1].location = 1;
            attrs[1].binding = 0;
            attrs[1].format = VK_FORMAT_R32G32_SFLOAT;
            attrs[1].offset = sizeof(float) * 2;

            VkPipelineVertexInputStateCreateInfo vertex_input{};
            vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertex_input.vertexBindingDescriptionCount = 1;
            vertex_input.pVertexBindingDescriptions = &binding;
            vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attrs.size());
            vertex_input.pVertexAttributeDescriptions = attrs.data();

            VkPipelineInputAssemblyStateCreateInfo input_assembly{};
            input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo viewport_state{};
            viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewport_state.viewportCount = 1;
            viewport_state.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo raster{};
            raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_NONE;
            raster.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo multisample{};
            multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            // Depth-only writer: write gl_FragDepth, no color writes, depth test always.
            VkPipelineDepthStencilStateCreateInfo depth{};
            depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depth.depthTestEnable = VK_TRUE;
            depth.depthWriteEnable = VK_TRUE;
            depth.depthCompareOp = VK_COMPARE_OP_ALWAYS;

            VkPipelineColorBlendAttachmentState blend_attachment{};
            blend_attachment.colorWriteMask = 0; // disable color writes
            VkPipelineColorBlendStateCreateInfo blend{};
            blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            blend.attachmentCount = 1;
            blend.pAttachments = &blend_attachment;

            std::array<VkDynamicState, 2> dyn{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamic{};
            dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamic.dynamicStateCount = static_cast<std::uint32_t>(dyn.size());
            dynamic.pDynamicStates = dyn.data();

            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            push.offset = 0;
            push.size = sizeof(DepthBlitPush);

            VkPipelineLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layout_info.setLayoutCount = 1;
            layout_info.pSetLayouts = &desc_layout;
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &push;
            if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
                vkDestroyShaderModule(device, vert, nullptr);
                vkDestroyShaderModule(device, frag, nullptr);
                return false;
            }

            VkPipelineRenderingCreateInfo rendering_info{};
            rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            rendering_info.colorAttachmentCount = 1;
            rendering_info.pColorAttachmentFormats = &color_format;
            rendering_info.depthAttachmentFormat = depth_format;
            rendering_info.stencilAttachmentFormat = depth_format;

            VkGraphicsPipelineCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pi.pNext = &rendering_info;
            pi.stageCount = 2;
            pi.pStages = stages.data();
            pi.pVertexInputState = &vertex_input;
            pi.pInputAssemblyState = &input_assembly;
            pi.pViewportState = &viewport_state;
            pi.pRasterizationState = &raster;
            pi.pMultisampleState = &multisample;
            pi.pDepthStencilState = &depth;
            pi.pColorBlendState = &blend;
            pi.pDynamicState = &dynamic;
            pi.layout = pipeline_layout;

            const VkResult r = vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pi, nullptr, &pipeline);
            vkDestroyShaderModule(device, vert, nullptr);
            vkDestroyShaderModule(device, frag, nullptr);
            return r == VK_SUCCESS;
        }

        VkCommandBuffer beginSingleTimeCommands() const {
            VkCommandBufferAllocateInfo a{};
            a.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            a.commandPool = transfer_pool;
            a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            a.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(device, &a, &cb) != VK_SUCCESS) {
                return VK_NULL_HANDLE;
            }
            VkCommandBufferBeginInfo b{};
            b.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            b.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(cb, &b) != VK_SUCCESS) {
                vkFreeCommandBuffers(device, transfer_pool, 1, &cb);
                return VK_NULL_HANDLE;
            }
            return cb;
        }

        bool endSingleTimeCommands(VkCommandBuffer cb) const {
            if (vkEndCommandBuffer(cb) != VK_SUCCESS) {
                vkFreeCommandBuffers(device, transfer_pool, 1, &cb);
                return false;
            }
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            VkFenceCreateInfo fi{};
            fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VkFence fence = VK_NULL_HANDLE;
            VkResult r = vkCreateFence(device, &fi, nullptr, &fence);
            if (r == VK_SUCCESS)
                r = vkQueueSubmit(graphics_queue, 1, &si, fence);
            if (r == VK_SUCCESS)
                r = vkWaitForFences(device, 1, &fence, VK_TRUE,
                                    std::numeric_limits<std::uint64_t>::max());
            if (fence != VK_NULL_HANDLE)
                vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, transfer_pool, 1, &cb);
            return r == VK_SUCCESS;
        }

        void destroyImage() {
            // A pending transfer submit may still reference this image. Drain it
            // before destruction so we don't free in-use device memory.
            if (transfer_fence != VK_NULL_HANDLE) {
                vkWaitForFences(device, 1, &transfer_fence, VK_TRUE,
                                std::numeric_limits<std::uint64_t>::max());
            }
            if (image_view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, image_view, nullptr);
                image_view = VK_NULL_HANDLE;
            }
            if (image != VK_NULL_HANDLE) {
                vmaDestroyImage(allocator, image, image_alloc);
                image = VK_NULL_HANDLE;
                image_alloc = VK_NULL_HANDLE;
            }
            image_width = 0;
            image_height = 0;
            uploaded_tensor = nullptr;
            uploaded_generation = 0;
        }

        bool ensureImage(std::uint32_t w, std::uint32_t h) {
            if (image != VK_NULL_HANDLE && image_width == w && image_height == h) {
                return true;
            }
            destroyImage();
            VkImageCreateInfo img{};
            img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            img.imageType = VK_IMAGE_TYPE_2D;
            img.format = VK_FORMAT_R32_SFLOAT;
            img.extent = {w, h, 1};
            img.mipLevels = 1;
            img.arrayLayers = 1;
            img.samples = VK_SAMPLE_COUNT_1_BIT;
            img.tiling = VK_IMAGE_TILING_OPTIMAL;
            img.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            img.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo ai{};
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            if (vmaCreateImage(allocator, &img, &ai, &image, &image_alloc, nullptr) != VK_SUCCESS) {
                return false;
            }
            VkImageViewCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vi.image = image;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = VK_FORMAT_R32_SFLOAT;
            vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vi.subresourceRange.levelCount = 1;
            vi.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &vi, nullptr, &image_view) != VK_SUCCESS) {
                destroyImage();
                return false;
            }
            image_width = w;
            image_height = h;
            // Force a rebind: the previous bound view now points at destroyed memory.
            bound_view = VK_NULL_HANDLE;
            return true;
        }

        bool uploadDepth(const lfs::core::Tensor& depth) {
            if (depth.ndim() != 3 || depth.size(0) != 1) {
                return false;
            }
            const std::uint32_t h = static_cast<std::uint32_t>(depth.size(1));
            const std::uint32_t w = static_cast<std::uint32_t>(depth.size(2));
            if (w == 0 || h == 0) {
                return false;
            }
            if (!ensureImage(w, h)) {
                return false;
            }
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * sizeof(float);
            if (!ensureStaging(bytes) || !ensureTransferCmd()) {
                return false;
            }

            const auto host = depth.to(lfs::core::Device::CPU).contiguous();
            std::memcpy(staging_mapped, host.ptr<float>(), static_cast<std::size_t>(bytes));
            vmaFlushAllocation(allocator, staging_alloc, 0, bytes);

            // Wait for any prior submit on this transfer CB before re-recording.
            // Fence is created signaled, so first frame is a no-op.
            vkWaitForFences(device, 1, &transfer_fence, VK_TRUE,
                            std::numeric_limits<std::uint64_t>::max());
            vkResetFences(device, 1, &transfer_fence);
            if (vkResetCommandBuffer(transfer_cmd, 0) != VK_SUCCESS) {
                return false;
            }
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(transfer_cmd, &bi) != VK_SUCCESS) {
                return false;
            }

            const VkImageLayout old_layout = (uploaded_tensor != nullptr)
                                                 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                 : VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageMemoryBarrier to_dst{};
            to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_dst.oldLayout = old_layout;
            to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_dst.image = image;
            to_dst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            to_dst.subresourceRange.levelCount = 1;
            to_dst.subresourceRange.layerCount = 1;
            to_dst.srcAccessMask =
                old_layout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_SHADER_READ_BIT;
            to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            const VkPipelineStageFlags src_stage =
                old_layout == VK_IMAGE_LAYOUT_UNDEFINED
                    ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                    : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            vkCmdPipelineBarrier(transfer_cmd, src_stage, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_dst);

            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {w, h, 1};
            vkCmdCopyBufferToImage(transfer_cmd, staging_buffer, image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            VkImageMemoryBarrier to_read{};
            to_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_read.image = image;
            to_read.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            to_read.subresourceRange.levelCount = 1;
            to_read.subresourceRange.layerCount = 1;
            to_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(transfer_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_read);

            if (vkEndCommandBuffer(transfer_cmd) != VK_SUCCESS) {
                return false;
            }
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &transfer_cmd;
            // Async submit: in-order queue execution makes the upload visible to the
            // viewport pass that samples this image right after on the same queue.
            if (vkQueueSubmit(graphics_queue, 1, &si, transfer_fence) != VK_SUCCESS) {
                return false;
            }
            uploaded_tensor = &depth;
            return true;
        }

        void rebindDescriptor(VkImageView view, const std::uint64_t generation = 0) {
            if (view == VK_NULL_HANDLE || (view == bound_view && generation == bound_generation)) {
                return;
            }
            VkDescriptorImageInfo di{};
            di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            di.imageView = view;
            di.sampler = sampler;
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = desc_set;
            w.dstBinding = 0;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &di;
            vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
            bound_view = view;
            bound_generation = generation;
        }

        void prepare(const VulkanDepthBlitParams& params) {
            // Interop fast-path: gui_manager already CUDA-copied the depth tensor into
            // an external Vulkan image and transitioned it to SHADER_READ_ONLY. Just
            // bind that view directly.
            if (params.external_image_view != VK_NULL_HANDLE) {
                rebindDescriptor(params.external_image_view, params.external_image_generation);
                return;
            }
            if (!params.depth || !params.depth->is_valid()) {
                if (image != VK_NULL_HANDLE)
                    destroyImage();
                bound_view = VK_NULL_HANDLE;
                bound_generation = 0;
                return;
            }
            if (params.depth.get() == uploaded_tensor && image != VK_NULL_HANDLE) {
                rebindDescriptor(image_view);
                return;
            }
            if (uploadDepth(*params.depth)) {
                rebindDescriptor(image_view);
            }
        }

        void record(VkCommandBuffer cb, VkExtent2D extent, const VulkanDepthBlitParams& params) {
            if (pipeline == VK_NULL_HANDLE || bound_view == VK_NULL_HANDLE ||
                screen_quad_buffer == VK_NULL_HANDLE) {
                return;
            }
            VkViewport vp{};
            vp.x = 0.0f;
            vp.y = 0.0f;
            vp.width = static_cast<float>(extent.width);
            vp.height = static_cast<float>(extent.height);
            vp.minDepth = 0.0f;
            vp.maxDepth = 1.0f;
            VkRect2D sc{};
            sc.extent = extent;
            vkCmdSetViewport(cb, 0, 1, &vp);
            vkCmdSetScissor(cb, 0, 1, &sc);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &desc_set, 0, nullptr);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cb, 0, 1, &screen_quad_buffer, &off);

            DepthBlitPush push{};
            push.params[0] = params.near_plane;
            push.params[1] = params.far_plane;
            push.params[2] = params.depth_is_ndc ? 0.0f : 1.0f;
            push.params[3] = params.flip_y ? 1.0f : 0.0f;
            vkCmdPushConstants(cb, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(push), &push);
            vkCmdDraw(cb, 6, 1, 0, 0);
        }
    };

    VulkanDepthBlitPass::VulkanDepthBlitPass() = default;
    VulkanDepthBlitPass::~VulkanDepthBlitPass() = default;
    VulkanDepthBlitPass::VulkanDepthBlitPass(VulkanDepthBlitPass&&) noexcept = default;
    VulkanDepthBlitPass& VulkanDepthBlitPass::operator=(VulkanDepthBlitPass&&) noexcept = default;

    bool VulkanDepthBlitPass::init(VulkanContext& context, VkFormat color_format,
                                   VkFormat depth_format, VkBuffer screen_quad_buffer) {
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        return impl_->init(context, color_format, depth_format, screen_quad_buffer);
    }

    void VulkanDepthBlitPass::prepare(const VulkanDepthBlitParams& params) {
        if (impl_)
            impl_->prepare(params);
    }

    void VulkanDepthBlitPass::record(VkCommandBuffer cb, VkExtent2D extent,
                                     const VulkanDepthBlitParams& params) {
        if (impl_)
            impl_->record(cb, extent, params);
    }

    void VulkanDepthBlitPass::shutdown() {
        if (impl_) {
            impl_->destroy();
            impl_.reset();
        }
    }

    bool VulkanDepthBlitPass::hasDepth() const {
        return impl_ && impl_->bound_view != VK_NULL_HANDLE;
    }

} // namespace lfs::vis
