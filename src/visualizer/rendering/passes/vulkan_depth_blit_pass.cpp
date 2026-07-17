/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_depth_blit_pass.hpp"

#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "rendering/vulkan_wait.hpp"
#include "window/vulkan_barrier2.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_result.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <limits>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>
#include <vk_mem_alloc.h>

#include "viewport/depth_blit.frag.spv.h"
#include "viewport/screen_quad.vert.spv.h"

namespace lfs::vis {

    namespace {

        struct DepthBlitPush {
            float params[4]; // near, far, is_view_depth, flip_y
        };
        static_assert(sizeof(DepthBlitPush) == 16);

        [[nodiscard]] const char* waitOutcomeLabel(const lfs::rendering::WaitOutcome outcome) noexcept {
            using lfs::rendering::WaitOutcome;
            switch (outcome) {
            case WaitOutcome::Ready: return "Ready";
            case WaitOutcome::Cancelled: return "Cancelled";
            case WaitOutcome::Shutdown: return "Shutdown";
            case WaitOutcome::Quarantined: return "Quarantined";
            }
            return "Unknown";
        }

        [[nodiscard]] std::string formatWaitFailure(
            const lfs::Result<lfs::rendering::WaitOutcome>& outcome) {
            if (outcome.has_value()) {
                return waitOutcomeLabel(*outcome);
            }
            return std::string(outcome.error().detail());
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
        struct FrameDescriptor {
            VkDescriptorSet set = VK_NULL_HANDLE;
            VkImageView bound_view = VK_NULL_HANDLE;
            std::uint64_t bound_generation = 0;
        };
        std::vector<FrameDescriptor> frame_descriptors;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        VkBuffer screen_quad_buffer = VK_NULL_HANDLE;

        VkImage image = VK_NULL_HANDLE;
        VmaAllocation image_alloc = VK_NULL_HANDLE;
        VkImageView image_view = VK_NULL_HANDLE;
        std::uint32_t image_width = 0;
        std::uint32_t image_height = 0;
        std::string image_vram_label;

        const lfs::core::Tensor* uploaded_tensor = nullptr;

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
                return logVkFailure(std::format(
                    "Depth-blit initialization requires a live device and allocator (device={:#x}, allocator={:#x}, graphics_queue={:#x}, screen_quad_buffer={:#x}) ({}:{})",
                    vkHandleValue(device),
                    reinterpret_cast<std::uintptr_t>(allocator),
                    vkHandleValue(graphics_queue),
                    vkHandleValue(screen_quad_buffer),
                    __FILE__,
                    __LINE__));
            }
            VkCommandPoolCreateInfo pool{};
            pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool.queueFamilyIndex = ctx.graphicsQueueFamily();
            if (!vk_try_bool(
                    vkCreateCommandPool(device, &pool, nullptr, &transfer_pool),
                    "vkCreateCommandPool(device, &pool, nullptr, &transfer_pool)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Depth-blit transfer command-pool creation failed (device={:#x}, queue_family={}, flags={:#x})",
                                vkHandleValue(device),
                                pool.queueFamilyIndex,
                                static_cast<std::uint32_t>(pool.flags)
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_COMMAND_POOL,
                                        transfer_pool,
                                        "depth_blit.transfer.pool");
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
            }
            frame_descriptors.clear();
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
            if (!vk_try_bool(
                    vmaCreateBuffer(allocator, &bi, &sa, &staging_buffer, &staging_alloc, &ai),
                    "vmaCreateBuffer(allocator, &bi, &sa, &staging_buffer, &staging_alloc, &ai)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Depth-blit staging-buffer allocation failed (allocator={:#x}, requested_size={}, usage={:#x})",
                                reinterpret_cast<std::uintptr_t>(allocator),
                                bytes,
                                static_cast<std::uint32_t>(bi.usage)
                            ),
                    std::source_location::current())) {
                return false;
            }
            staging_mapped = ai.pMappedData;
            staging_capacity = bytes;
            context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                         staging_buffer,
                                         "depth_blit.upload.staging[{}]",
                                         staging_capacity);
            return true;
        }

        bool ensureTransferCmd() {
            if (transfer_cmd != VK_NULL_HANDLE && transfer_fence != VK_NULL_HANDLE) {
                return true;
            }
            if (transfer_cmd != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, transfer_pool, 1, &transfer_cmd);
                transfer_cmd = VK_NULL_HANDLE;
            }
            if (transfer_fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, transfer_fence, nullptr);
                transfer_fence = VK_NULL_HANDLE;
            }
            VkCommandBufferAllocateInfo a{};
            a.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            a.commandPool = transfer_pool;
            a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            a.commandBufferCount = 1;
            if (!vk_try_bool(
                    vkAllocateCommandBuffers(device, &a, &transfer_cmd),
                    "vkAllocateCommandBuffers(device, &a, &transfer_cmd)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Depth-blit transfer command-buffer allocation failed (device={:#x}, command_pool={:#x}, requested_count={})",
                                vkHandleValue(device),
                                vkHandleValue(transfer_pool),
                                a.commandBufferCount
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER,
                                        transfer_cmd,
                                        "depth_blit.transfer.command");
            VkFenceCreateInfo fi{};
            fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            const VkResult fence_result = vkCreateFence(device, &fi, nullptr, &transfer_fence);
            if (fence_result != VK_SUCCESS) {
                vkFreeCommandBuffers(device, transfer_pool, 1, &transfer_cmd);
                transfer_cmd = VK_NULL_HANDLE;
                return reportVkFailure(
                    "vkCreateFence(device, &fi, nullptr, &transfer_fence)",
                    fence_result,
                    std::format("Depth-blit transfer fence creation failed (device={:#x}, command_pool={:#x}, flags={:#x})",
                                vkHandleValue(device),
                                vkHandleValue(transfer_pool),
                                static_cast<std::uint32_t>(fi.flags)));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_FENCE,
                                        transfer_fence,
                                        "depth_blit.transfer.fence");
            return true;
        }

        bool replaceTransferFenceSignaled(const char* const failed_operation,
                                          const VkResult failed_result) {
            LOG_ERROR("Vulkan: {}",
                      formatVkCheckFailure(
                          failed_operation,
                          failed_result,
                          std::format("Depth-blit transfer command lifecycle failed (device={:#x}, queue={:#x}, command_pool={:#x}, command_buffer={:#x}, fence={:#x})",
                                      vkHandleValue(device),
                                      vkHandleValue(graphics_queue),
                                      vkHandleValue(transfer_pool),
                                      vkHandleValue(transfer_cmd),
                                      vkHandleValue(transfer_fence)),
                          __FILE__,
                          __LINE__));
            VkFenceCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            VkFence replacement = VK_NULL_HANDLE;
            const VkResult replacement_result = vkCreateFence(device, &info, nullptr, &replacement);
            if (replacement_result != VK_SUCCESS) {
                LOG_ERROR("Vulkan: {}",
                          formatVkCheckFailure(
                              "vkCreateFence(device, &info, nullptr, &replacement)",
                              replacement_result,
                              std::format("Depth-blit failed to replace a poisoned transfer fence (device={:#x}, old_fence={:#x}, command_buffer={:#x}, flags={:#x})",
                                          vkHandleValue(device),
                                          vkHandleValue(transfer_fence),
                                          vkHandleValue(transfer_cmd),
                                          static_cast<std::uint32_t>(info.flags)),
                              __FILE__,
                              __LINE__));
                return false;
            }
            if (transfer_fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, transfer_fence, nullptr);
            }
            transfer_fence = replacement;
            context->setDebugObjectName(VK_OBJECT_TYPE_FENCE,
                                        transfer_fence,
                                        "depth_blit.transfer.fence");
            return false;
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
            if (!vk_try_bool(
                    vkCreateSampler(device, &s, nullptr, &sampler),
                    "vkCreateSampler(device, &s, nullptr, &sampler)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Depth-blit sampler creation failed (device={:#x}, mag_filter={}, min_filter={}, address_mode={})",
                                vkHandleValue(device),
                                static_cast<int>(s.magFilter),
                                static_cast<int>(s.minFilter),
                                static_cast<int>(s.addressModeU)
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_SAMPLER,
                                        sampler,
                                        "depth_blit.depth.sampler");
            return true;
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
            if (!vk_try_bool(
                    vkCreateDescriptorSetLayout(device, &li, nullptr, &desc_layout),
                    "vkCreateDescriptorSetLayout(device, &li, nullptr, &desc_layout)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Depth-blit descriptor-set layout creation failed (device={:#x}, binding_count={}, descriptor_type={})",
                                vkHandleValue(device),
                                li.bindingCount,
                                static_cast<int>(b.descriptorType)
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                        desc_layout,
                                        "depth_blit.descriptor.layout");
            VkDescriptorPoolSize ps{};
            ps.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            const std::uint32_t frame_count = static_cast<std::uint32_t>(
                std::max<std::size_t>(1, context->framesInFlight()));
            ps.descriptorCount = frame_count;
            VkDescriptorPoolCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pi.maxSets = frame_count;
            pi.poolSizeCount = 1;
            pi.pPoolSizes = &ps;
            if (!vk_try_bool(
                    vkCreateDescriptorPool(device, &pi, nullptr, &desc_pool),
                    "vkCreateDescriptorPool(device, &pi, nullptr, &desc_pool)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Depth-blit descriptor-pool creation failed (device={:#x}, frame_count={}, max_sets={}, descriptor_count={})",
                                vkHandleValue(device),
                                frame_count,
                                pi.maxSets,
                                ps.descriptorCount
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                        desc_pool,
                                        "depth_blit.descriptor.pool");
            std::vector<VkDescriptorSetLayout> layouts(frame_count, desc_layout);
            std::vector<VkDescriptorSet> sets(frame_count, VK_NULL_HANDLE);
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = desc_pool;
            ai.descriptorSetCount = frame_count;
            ai.pSetLayouts = layouts.data();
            if (!vk_try_bool(
                    vkAllocateDescriptorSets(device, &ai, sets.data()),
                    "vkAllocateDescriptorSets(device, &ai, sets.data())",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Depth-blit descriptor-set allocation failed (device={:#x}, descriptor_pool={:#x}, descriptor_layout={:#x}, requested_count={})",
                                vkHandleValue(device),
                                vkHandleValue(desc_pool),
                                vkHandleValue(desc_layout),
                                ai.descriptorSetCount
                            ),
                    std::source_location::current())) {
                return false;
            }
            frame_descriptors.resize(frame_count);
            for (std::size_t i = 0; i < sets.size(); ++i) {
                frame_descriptors[i].set = sets[i];
                context->setDebugObjectNamef(VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                             sets[i],
                                             "depth_blit.descriptor[{}]",
                                             i);
            }
            return true;
        }

        [[nodiscard]] FrameDescriptor& descriptorForFrame(const std::size_t frame_slot) {
            if (frame_slot >= frame_descriptors.size()) [[unlikely]] {
                throw std::logic_error(std::format(
                    "Depth-blit frame slot is outside the descriptor ring (frame_slot={}, ring_size={}) ({}:{})",
                    frame_slot,
                    frame_descriptors.size(),
                    __FILE__,
                    __LINE__));
            }
            return frame_descriptors[frame_slot];
        }

        [[nodiscard]] const FrameDescriptor& descriptorForFrame(const std::size_t frame_slot) const {
            if (frame_slot >= frame_descriptors.size()) [[unlikely]] {
                throw std::logic_error(std::format(
                    "Depth-blit frame slot is outside the descriptor ring (frame_slot={}, ring_size={}) ({}:{})",
                    frame_slot,
                    frame_descriptors.size(),
                    __FILE__,
                    __LINE__));
            }
            return frame_descriptors[frame_slot];
        }

        bool createPipeline(VkFormat color_format, VkFormat depth_format) {
            using namespace viewport_shaders;
            VkShaderModule vert = createShaderModule(device, kScreenQuadVertSpv, "Depth-blit");
            VkShaderModule frag = createShaderModule(device, kDepthBlitFragSpv, "Depth-blit");
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
            const VkResult layout_result =
                vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout);
            if (layout_result != VK_SUCCESS) {
                vkDestroyShaderModule(device, vert, nullptr);
                vkDestroyShaderModule(device, frag, nullptr);
                return reportVkFailure(
                    "vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout)",
                    layout_result,
                    std::format("Depth-blit pipeline-layout creation failed (device={:#x}, descriptor_layout={:#x}, set_layout_count={}, push_constant_bytes={})",
                                vkHandleValue(device),
                                vkHandleValue(desc_layout),
                                layout_info.setLayoutCount,
                                push.size));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                        pipeline_layout,
                                        "depth_blit.pipeline.layout");

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
            if (r != VK_SUCCESS) {
                return reportVkFailure(
                    "vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pi, nullptr, &pipeline)",
                    r,
                    std::format("Depth-blit graphics-pipeline creation failed (device={:#x}, pipeline_cache={:#x}, pipeline_layout={:#x}, color_format={}, depth_format={})",
                                vkHandleValue(device),
                                vkHandleValue(pipeline_cache),
                                vkHandleValue(pipeline_layout),
                                static_cast<int>(color_format),
                                static_cast<int>(depth_format)));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE,
                                        pipeline,
                                        "depth_blit.pipeline");
            return true;
        }

        void destroyImage() {
            // A pending transfer submit may still reference this image. Drain it
            // before destruction so we don't free in-use device memory.
            if (transfer_fence != VK_NULL_HANDLE) {
                lfs::rendering::WaitContext wait_ctx;
                wait_ctx.fingerprint = "pass.depth_blit.destroy_drain";
                auto wait_outcome = lfs::rendering::wait_fence_bounded(
                    device,
                    transfer_fence,
                    std::stop_token{},
                    lfs::rendering::VulkanWaitPolicy{},
                    wait_ctx);
                const bool ready = wait_outcome.has_value() &&
                                   *wait_outcome == lfs::rendering::WaitOutcome::Ready;
                if (!ready) {
                    // AMB-4: quarantine/timeout never authorizes free of in-flight
                    // resources — retain image + fence until process exit.
                    LOG_ERROR(
                        "Vulkan: depth-blit destroy_drain did not reach Ready "
                        "(device={:#x}, fence={:#x}, image={:#x}, image_view={:#x}): {} — "
                        "retaining image and fence",
                        vkHandleValue(device),
                        vkHandleValue(transfer_fence),
                        vkHandleValue(image),
                        vkHandleValue(image_view),
                        formatWaitFailure(wait_outcome));
                    return;
                }
            }
            if (image_view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, image_view, nullptr);
                image_view = VK_NULL_HANDLE;
            }
            if (image != VK_NULL_HANDLE) {
                if (!image_vram_label.empty()) {
                    lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                        "vulkan.depth_blit.image",
                        image_vram_label,
                        0);
                }
                vmaDestroyImage(allocator, image, image_alloc);
                image = VK_NULL_HANDLE;
                image_alloc = VK_NULL_HANDLE;
            }
            image_width = 0;
            image_height = 0;
            uploaded_tensor = nullptr;
            for (auto& descriptor : frame_descriptors) {
                descriptor.bound_view = VK_NULL_HANDLE;
                descriptor.bound_generation = 0;
            }
        }

        bool retireAndDestroyImage(const char* const reason) {
            if (image == VK_NULL_HANDLE) {
                return true;
            }
            if (context != nullptr && !context->waitForSubmittedFrames()) {
                LOG_ERROR("VulkanDepthBlitPass: could not retire frames before {}: {}",
                          reason,
                          context->lastError());
                return false;
            }
            destroyImage();
            return true;
        }

        bool ensureImage(std::uint32_t w, std::uint32_t h) {
            if (image != VK_NULL_HANDLE && image_width == w && image_height == h) {
                return true;
            }
            if (!retireAndDestroyImage("depth image resize")) {
                return false;
            }
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
            VmaAllocationInfo allocation_info{};
            if (!vk_try_bool(
                    vmaCreateImage(allocator, &img, &ai, &image, &image_alloc, &allocation_info),
                    "vmaCreateImage(allocator, &img, &ai, &image, &image_alloc, &allocation_info)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Depth-blit image allocation failed (allocator={:#x}, requested_extent={}x{}, format={}, usage={:#x})",
                                reinterpret_cast<std::uintptr_t>(allocator),
                                w,
                                h,
                                static_cast<int>(img.format),
                                static_cast<std::uint32_t>(img.usage)
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE,
                                         image,
                                         "depth_blit.image[{}x{}]",
                                         w,
                                         h);
            image_vram_label = std::format("r32_float:{}x{}", w, h);
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.depth_blit.image",
                image_vram_label,
                static_cast<std::size_t>(allocation_info.size));
            VkImageViewCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vi.image = image;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = VK_FORMAT_R32_SFLOAT;
            vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vi.subresourceRange.levelCount = 1;
            vi.subresourceRange.layerCount = 1;
            const VkResult view_result = vkCreateImageView(device, &vi, nullptr, &image_view);
            if (view_result != VK_SUCCESS) {
                destroyImage();
                return reportVkFailure(
                    "vkCreateImageView(device, &vi, nullptr, &image_view)",
                    view_result,
                    std::format("Depth-blit image-view creation failed (device={:#x}, image={:#x}, requested_extent={}x{}, format={}, aspect_mask={:#x})",
                                vkHandleValue(device),
                                vkHandleValue(vi.image),
                                w,
                                h,
                                static_cast<int>(vi.format),
                                static_cast<std::uint32_t>(vi.subresourceRange.aspectMask)));
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW,
                                         image_view,
                                         "depth_blit.image[{}x{}].view",
                                         w,
                                         h);
            image_width = w;
            image_height = h;
            return true;
        }

        bool uploadDepth(const lfs::core::Tensor& depth) {
            if (depth.ndim() != 3 || depth.size(0) != 1) {
                return logVkFailure(std::format(
                    "Depth upload requires a [1,H,W] tensor (observed_rank={}, observed_channel_count={}) ({}:{})",
                    depth.ndim(),
                    depth.ndim() > 0 ? depth.size(0) : -1,
                    __FILE__,
                    __LINE__));
            }
            const std::uint32_t h = static_cast<std::uint32_t>(depth.size(1));
            const std::uint32_t w = static_cast<std::uint32_t>(depth.size(2));
            if (w == 0 || h == 0) {
                return logVkFailure(std::format(
                    "Depth upload requires non-zero dimensions (observed_width={}, observed_height={}, tensor_rank={}) ({}:{})",
                    w,
                    h,
                    depth.ndim(),
                    __FILE__,
                    __LINE__));
            }
            if (!ensureImage(w, h)) {
                return false;
            }
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * sizeof(float);
            if (!ensureStaging(bytes) || !ensureTransferCmd()) {
                return false;
            }
            if (staging_buffer == VK_NULL_HANDLE || staging_alloc == VK_NULL_HANDLE ||
                staging_mapped == nullptr || staging_capacity < bytes || image == VK_NULL_HANDLE ||
                transfer_cmd == VK_NULL_HANDLE || transfer_fence == VK_NULL_HANDLE ||
                graphics_queue == VK_NULL_HANDLE) {
                return logVkFailure(std::format(
                    "Depth upload resources must cover the copy before recording (staging_buffer={:#x}, staging_allocation={:#x}, staging_mapped={:#x}, staging_capacity={}, copy_size={}, image={:#x}, command_buffer={:#x}, fence={:#x}, queue={:#x}) ({}:{})",
                    vkHandleValue(staging_buffer),
                    reinterpret_cast<std::uintptr_t>(staging_alloc),
                    reinterpret_cast<std::uintptr_t>(staging_mapped),
                    staging_capacity,
                    bytes,
                    vkHandleValue(image),
                    vkHandleValue(transfer_cmd),
                    vkHandleValue(transfer_fence),
                    vkHandleValue(graphics_queue),
                    __FILE__,
                    __LINE__));
            }

            const auto host = depth.to(lfs::core::Device::CPU).contiguous();
            std::memcpy(staging_mapped, host.ptr<float>(), static_cast<std::size_t>(bytes));
            VkResult result = vmaFlushAllocation(allocator, staging_alloc, 0, bytes);
            if (result != VK_SUCCESS) {
                return reportVkFailure(
                    "vmaFlushAllocation(allocator, staging_alloc, 0, bytes)",
                    result,
                    std::format("Depth-blit staging flush failed (allocator={:#x}, allocation={:#x}, offset=0, flush_size={}, capacity={})",
                                reinterpret_cast<std::uintptr_t>(allocator),
                                reinterpret_cast<std::uintptr_t>(staging_alloc),
                                bytes,
                                staging_capacity));
            }

            // Wait for any prior submit on this transfer CB before re-recording.
            // Fence is created signaled, so first frame is a no-op.
            {
                lfs::rendering::WaitContext wait_ctx;
                wait_ctx.fingerprint = "pass.depth_blit.transfer_prewait";
                auto wait_outcome = lfs::rendering::wait_fence_bounded(
                    device,
                    transfer_fence,
                    std::stop_token{},
                    lfs::rendering::VulkanWaitPolicy{},
                    wait_ctx);
                if (!wait_outcome.has_value() ||
                    *wait_outcome != lfs::rendering::WaitOutcome::Ready) {
                    // Soft-fail; retain fence (do not reset/re-record while in-flight).
                    return logVkFailure(std::format(
                        "wait_fence_bounded(pass.depth_blit.transfer_prewait) did not reach Ready "
                        "(device={:#x}, fence={:#x}, command_buffer={:#x}): {}",
                        vkHandleValue(device),
                        vkHandleValue(transfer_fence),
                        vkHandleValue(transfer_cmd),
                        formatWaitFailure(wait_outcome)));
                }
            }
            result = vkResetFences(device, 1, &transfer_fence);
            if (result != VK_SUCCESS) {
                return replaceTransferFenceSignaled("vkResetFences", result);
            }
            result = vkResetCommandBuffer(transfer_cmd, 0);
            if (result != VK_SUCCESS) {
                return replaceTransferFenceSignaled("vkResetCommandBuffer", result);
            }
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            result = vkBeginCommandBuffer(transfer_cmd, &bi);
            if (result != VK_SUCCESS) {
                return replaceTransferFenceSignaled("vkBeginCommandBuffer", result);
            }

            const VkImageLayout old_layout = (uploaded_tensor != nullptr)
                                                 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                 : VK_IMAGE_LAYOUT_UNDEFINED;
            const bool was_undefined = old_layout == VK_IMAGE_LAYOUT_UNDEFINED;
            cmdImageBarrier2(transfer_cmd, image, VK_IMAGE_ASPECT_COLOR_BIT,
                             old_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             was_undefined ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                                           : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             was_undefined ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_READ_BIT,
                             VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {w, h, 1};
            vkCmdCopyBufferToImage(transfer_cmd, staging_buffer, image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            cmdImageBarrier2(transfer_cmd, image, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

            result = vkEndCommandBuffer(transfer_cmd);
            if (result != VK_SUCCESS) {
                return replaceTransferFenceSignaled("vkEndCommandBuffer", result);
            }
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &transfer_cmd;
            if (si.commandBufferCount != 1 || si.pCommandBuffers == nullptr ||
                si.pCommandBuffers[0] == VK_NULL_HANDLE || transfer_fence == VK_NULL_HANDLE ||
                graphics_queue == VK_NULL_HANDLE) {
                return replaceTransferFenceSignaled("Depth-blit submit integrity check",
                                                    VK_ERROR_INITIALIZATION_FAILED);
            }
            // Async submit: in-order queue execution makes the upload visible to the
            // viewport pass that samples this image right after on the same queue.
            result = vkQueueSubmit(graphics_queue, 1, &si, transfer_fence);
            if (result != VK_SUCCESS) {
                return replaceTransferFenceSignaled("vkQueueSubmit", result);
            }
            uploaded_tensor = &depth;
            return true;
        }

        void rebindDescriptor(FrameDescriptor& descriptor,
                              VkImageView view,
                              const std::uint64_t generation = 0) {
            if (view == VK_NULL_HANDLE ||
                (view == descriptor.bound_view && generation == descriptor.bound_generation)) {
                return;
            }
            VkDescriptorImageInfo di{};
            di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            di.imageView = view;
            di.sampler = sampler;
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = descriptor.set;
            w.dstBinding = 0;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &di;
            vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
            descriptor.bound_view = view;
            descriptor.bound_generation = generation;
        }

        void prepare(const VulkanDepthBlitParams& params, const std::size_t frame_slot) {
            auto& descriptor = descriptorForFrame(frame_slot);
            // Interop fast-path: gui_manager already CUDA-copied the depth tensor into
            // an external Vulkan image and transitioned it to SHADER_READ_ONLY. Just
            // bind that view directly.
            if (params.external_image_view != VK_NULL_HANDLE) {
                rebindDescriptor(descriptor,
                                 params.external_image_view,
                                 params.external_image_generation);
                return;
            }
            if (!params.depth || !params.depth->is_valid()) {
                descriptor.bound_view = VK_NULL_HANDLE;
                descriptor.bound_generation = 0;
                retireAndDestroyImage("depth image release");
                return;
            }
            if (params.depth.get() == uploaded_tensor && image != VK_NULL_HANDLE) {
                rebindDescriptor(descriptor, image_view);
                return;
            }
            if (uploadDepth(*params.depth)) {
                rebindDescriptor(descriptor, image_view);
            }
        }

        void record(VkCommandBuffer cb, VkRect2D rect, const VulkanDepthBlitParams& params,
                    const std::size_t frame_slot) {
            const auto& descriptor = descriptorForFrame(frame_slot);
            if (cb == VK_NULL_HANDLE || descriptor.set == VK_NULL_HANDLE) [[unlikely]] {
                throw std::logic_error(std::format(
                    "Depth-blit recording requires a command buffer and per-frame descriptor set (command_buffer={:#x}, frame_slot={}, ring_size={}, descriptor_set={:#x}, bound_view={:#x}) ({}:{})",
                    vkHandleValue(cb),
                    frame_slot,
                    frame_descriptors.size(),
                    vkHandleValue(descriptor.set),
                    vkHandleValue(descriptor.bound_view),
                    __FILE__,
                    __LINE__));
            }
            if (pipeline == VK_NULL_HANDLE || descriptor.bound_view == VK_NULL_HANDLE ||
                screen_quad_buffer == VK_NULL_HANDLE ||
                rect.extent.width == 0 || rect.extent.height == 0) {
                return;
            }
            VkViewport vp{};
            vp.x = static_cast<float>(rect.offset.x);
            vp.y = static_cast<float>(rect.offset.y);
            vp.width = static_cast<float>(rect.extent.width);
            vp.height = static_cast<float>(rect.extent.height);
            vp.minDepth = 0.0f;
            vp.maxDepth = 1.0f;
            VkRect2D sc = rect;
            vkCmdSetViewport(cb, 0, 1, &vp);
            vkCmdSetScissor(cb, 0, 1, &sc);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &descriptor.set, 0, nullptr);
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

    void VulkanDepthBlitPass::prepare(const VulkanDepthBlitParams& params,
                                      const std::size_t frame_slot) {
        if (impl_)
            impl_->prepare(params, frame_slot);
    }

    void VulkanDepthBlitPass::record(VkCommandBuffer cb, VkRect2D rect,
                                     const VulkanDepthBlitParams& params,
                                     const std::size_t frame_slot) {
        if (impl_)
            impl_->record(cb, rect, params, frame_slot);
    }

    void VulkanDepthBlitPass::shutdown() {
        if (impl_) {
            impl_->destroy();
            impl_.reset();
        }
    }

    bool VulkanDepthBlitPass::hasDepth(const std::size_t frame_slot) const {
        return impl_ && impl_->descriptorForFrame(frame_slot).bound_view != VK_NULL_HANDLE;
    }

    VkImageView VulkanDepthBlitPass::depthView(const std::size_t frame_slot) const {
        return impl_ ? impl_->descriptorForFrame(frame_slot).bound_view : VK_NULL_HANDLE;
    }

} // namespace lfs::vis
