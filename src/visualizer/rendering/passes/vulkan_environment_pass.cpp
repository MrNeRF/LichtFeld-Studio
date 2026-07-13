/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_environment_pass.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "internal/resource_paths.hpp"
#include "window/vulkan_barrier2.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_result.hpp"

#include <OpenImageIO/imageio.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <limits>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>

#include "viewport/environment.frag.spv.h"
#include "viewport/screen_quad.vert.spv.h"

namespace lfs::vis {

    namespace {

        struct EnvPush {
            float cam_to_world[16];
            float intrinsics[4];
            float viewport_exposure[4]; // size.xy, exposure, rotation_radians
            float flags[4];             // x = is_equirectangular_view
        };
        static_assert(sizeof(EnvPush) == 112);

        // Pack a float into a 16-bit half-float (IEEE 754 binary16). Avoids dragging in
        // an extra header dep.
        std::uint16_t floatToHalf(float f) {
            std::uint32_t bits;
            std::memcpy(&bits, &f, 4);
            const std::uint32_t sign = (bits >> 31) & 0x1;
            std::int32_t exp = static_cast<std::int32_t>((bits >> 23) & 0xff) - 127 + 15;
            std::uint32_t mant = bits & 0x7fffff;
            if (exp <= 0) {
                if (exp < -10)
                    return static_cast<std::uint16_t>(sign << 15);
                mant |= 0x800000;
                const std::uint32_t shift = 14 - exp;
                return static_cast<std::uint16_t>((sign << 15) | (mant >> shift));
            }
            if (exp >= 31) {
                return static_cast<std::uint16_t>((sign << 15) | (0x1f << 10) | (mant ? 0x200 : 0));
            }
            return static_cast<std::uint16_t>((sign << 15) | (exp << 10) | (mant >> 13));
        }

    } // namespace

    struct VulkanEnvironmentPass::Impl {
        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;

        VkBuffer screen_quad_buffer = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
        VkDescriptorPool desc_pool = VK_NULL_HANDLE;
        struct FrameDescriptor {
            VkDescriptorSet set = VK_NULL_HANDLE;
            VkImageView bound_view = VK_NULL_HANDLE;
        };
        std::vector<FrameDescriptor> frame_descriptors;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkCommandPool transfer_pool = VK_NULL_HANDLE;

        VkImage image = VK_NULL_HANDLE;
        VmaAllocation image_alloc = VK_NULL_HANDLE;
        VkImageView image_view = VK_NULL_HANDLE;
        std::string image_vram_label;
        std::filesystem::path loaded_path;
        bool load_failed_for_path = false;

        ~Impl() { destroy(); }

        bool init(VulkanContext& ctx, VkFormat color_format, VkFormat depth_format, VkBuffer quad) {
            context = &ctx;
            device = ctx.device();
            allocator = ctx.allocator();
            pipeline_cache = ctx.pipelineCache();
            graphics_queue = ctx.graphicsQueue();
            screen_quad_buffer = quad;
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE ||
                graphics_queue == VK_NULL_HANDLE || screen_quad_buffer == VK_NULL_HANDLE) {
                return logVkFailure(std::format(
                    "Environment-pass initialization requires a live device, allocator, graphics queue, and screen quad (device={:#x}, allocator={:#x}, graphics_queue={:#x}, screen_quad_buffer={:#x}) ({}:{})",
                    vkHandleValue(device),
                    reinterpret_cast<std::uintptr_t>(allocator),
                    vkHandleValue(graphics_queue),
                    vkHandleValue(screen_quad_buffer),
                    __FILE__,
                    __LINE__));
            }

            VkCommandPoolCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pi.queueFamilyIndex = ctx.graphicsQueueFamily();
            LFS_VK_CHECK_MSG(vkCreateCommandPool(device, &pi, nullptr, &transfer_pool),
                             "Environment transfer command-pool creation failed (device={:#x}, queue_family={}, flags={:#x})",
                             vkHandleValue(device),
                             pi.queueFamilyIndex,
                             static_cast<std::uint32_t>(pi.flags));
            context->setDebugObjectName(VK_OBJECT_TYPE_COMMAND_POOL,
                                        transfer_pool,
                                        "environment.transfer.pool");

            return createSampler() && createDescriptors() && createPipeline(color_format, depth_format);
        }

        void destroy() {
            destroyImage();
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
            device = VK_NULL_HANDLE;
            allocator = VK_NULL_HANDLE;
        }

        void destroyImage() {
            if (image_view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, image_view, nullptr);
                image_view = VK_NULL_HANDLE;
            }
            if (image != VK_NULL_HANDLE) {
                if (!image_vram_label.empty()) {
                    lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                        "vulkan.environment.image",
                        image_vram_label,
                        0);
                }
                vmaDestroyImage(allocator, image, image_alloc);
                image = VK_NULL_HANDLE;
                image_alloc = VK_NULL_HANDLE;
            }
            image_vram_label.clear();
            loaded_path.clear();
            for (auto& descriptor : frame_descriptors) {
                descriptor.bound_view = VK_NULL_HANDLE;
            }
        }

        bool retireAndDestroyImage(const char* const reason) {
            if (image == VK_NULL_HANDLE) {
                return true;
            }
            if (context != nullptr && !context->waitForSubmittedFrames()) {
                LOG_ERROR("VulkanEnvironmentPass: could not retire frames before {}: {}",
                          reason,
                          context->lastError());
                return false;
            }
            destroyImage();
            return true;
        }

        bool createSampler() {
            VkSamplerCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            info.magFilter = VK_FILTER_LINEAR;
            info.minFilter = VK_FILTER_LINEAR;
            info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.maxLod = 0.0f;
            LFS_VK_CHECK_MSG(vkCreateSampler(device, &info, nullptr, &sampler),
                             "Environment sampler creation failed (device={:#x}, mag_filter={}, min_filter={}, address_mode_u={}, address_mode_v={})",
                             vkHandleValue(device),
                             static_cast<int>(info.magFilter),
                             static_cast<int>(info.minFilter),
                             static_cast<int>(info.addressModeU),
                             static_cast<int>(info.addressModeV));
            context->setDebugObjectName(VK_OBJECT_TYPE_SAMPLER,
                                        sampler,
                                        "environment.texture.sampler");
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
            LFS_VK_CHECK_MSG(vkCreateDescriptorSetLayout(device, &li, nullptr, &desc_layout),
                             "Environment descriptor-set layout creation failed (device={:#x}, binding_count={}, descriptor_type={})",
                             vkHandleValue(device),
                             li.bindingCount,
                             static_cast<int>(b.descriptorType));
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                        desc_layout,
                                        "environment.descriptor.layout");
            VkDescriptorPoolSize ps{};
            ps.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            const std::uint32_t frame_count = static_cast<std::uint32_t>(
                std::max<std::size_t>(1, context->framesInFlight()));
            ps.descriptorCount = frame_count;
            VkDescriptorPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pci.maxSets = frame_count;
            pci.poolSizeCount = 1;
            pci.pPoolSizes = &ps;
            LFS_VK_CHECK_MSG(vkCreateDescriptorPool(device, &pci, nullptr, &desc_pool),
                             "Environment descriptor-pool creation failed (device={:#x}, frame_count={}, max_sets={}, descriptor_count={})",
                             vkHandleValue(device),
                             frame_count,
                             pci.maxSets,
                             ps.descriptorCount);
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                        desc_pool,
                                        "environment.descriptor.pool");
            std::vector<VkDescriptorSetLayout> layouts(frame_count, desc_layout);
            std::vector<VkDescriptorSet> sets(frame_count, VK_NULL_HANDLE);
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = desc_pool;
            ai.descriptorSetCount = frame_count;
            ai.pSetLayouts = layouts.data();
            LFS_VK_CHECK_MSG(vkAllocateDescriptorSets(device, &ai, sets.data()),
                             "Environment descriptor-set allocation failed (device={:#x}, descriptor_pool={:#x}, descriptor_layout={:#x}, requested_count={})",
                             vkHandleValue(device),
                             vkHandleValue(desc_pool),
                             vkHandleValue(desc_layout),
                             ai.descriptorSetCount);
            frame_descriptors.resize(frame_count);
            for (std::size_t i = 0; i < sets.size(); ++i) {
                frame_descriptors[i].set = sets[i];
                context->setDebugObjectNamef(VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                             sets[i],
                                             "environment.descriptor[{}]",
                                             i);
            }
            return true;
        }

        [[nodiscard]] FrameDescriptor& descriptorForFrame(const std::size_t frame_slot) {
            if (frame_slot >= frame_descriptors.size()) [[unlikely]] {
                throw std::logic_error(std::format(
                    "Environment frame slot is outside the descriptor ring (frame_slot={}, ring_size={}) ({}:{})",
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
                    "Environment frame slot is outside the descriptor ring (frame_slot={}, ring_size={}) ({}:{})",
                    frame_slot,
                    frame_descriptors.size(),
                    __FILE__,
                    __LINE__));
            }
            return frame_descriptors[frame_slot];
        }

        void rebindDescriptor(FrameDescriptor& descriptor) const {
            if (image_view == VK_NULL_HANDLE || descriptor.bound_view == image_view) {
                return;
            }
            VkDescriptorImageInfo image_info{};
            image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            image_info.imageView = image_view;
            image_info.sampler = sampler;
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptor.set;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &image_info;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
            descriptor.bound_view = image_view;
        }

        bool createPipeline(VkFormat color_format, VkFormat depth_format) {
            using namespace viewport_shaders;
            VkShaderModule vert = createShaderModule(device, kScreenQuadVertSpv, "Environment");
            VkShaderModule frag = createShaderModule(device, kEnvironmentFragSpv, "Environment");
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
            binding.stride = 4 * sizeof(float);
            binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            std::array<VkVertexInputAttributeDescription, 2> attrs{};
            attrs[0] = {0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
            attrs[1] = {1, 0, VK_FORMAT_R32G32_SFLOAT, 2 * sizeof(float)};
            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vi.vertexBindingDescriptionCount = 1;
            vi.pVertexBindingDescriptions = &binding;
            vi.vertexAttributeDescriptionCount = 2;
            vi.pVertexAttributeDescriptions = attrs.data();

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vp{};
            vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo raster{};
            raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_NONE;
            raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            // Background — no depth test, write opaque color over whatever was cleared.
            VkPipelineDepthStencilStateCreateInfo depth{};
            depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depth.depthTestEnable = VK_FALSE;
            depth.depthWriteEnable = VK_FALSE;

            VkPipelineColorBlendAttachmentState ba{};
            ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            ba.blendEnable = VK_FALSE;
            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = 1;
            cb.pAttachments = &ba;

            std::array<VkDynamicState, 2> dyn{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            ds.dynamicStateCount = static_cast<std::uint32_t>(dyn.size());
            ds.pDynamicStates = dyn.data();

            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            push.offset = 0;
            push.size = sizeof(EnvPush);

            VkPipelineLayoutCreateInfo li{};
            li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            li.setLayoutCount = 1;
            li.pSetLayouts = &desc_layout;
            li.pushConstantRangeCount = 1;
            li.pPushConstantRanges = &push;
            const VkResult layout_result = vkCreatePipelineLayout(device, &li, nullptr, &pipeline_layout);
            if (layout_result != VK_SUCCESS) {
                vkDestroyShaderModule(device, vert, nullptr);
                vkDestroyShaderModule(device, frag, nullptr);
                return reportVkFailure(
                    "vkCreatePipelineLayout(device, &li, nullptr, &pipeline_layout)",
                    layout_result,
                    std::format("Environment pipeline-layout creation failed (device={:#x}, descriptor_layout={:#x}, set_layout_count={}, push_constant_bytes={})",
                                vkHandleValue(device),
                                vkHandleValue(desc_layout),
                                li.setLayoutCount,
                                push.size));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                        pipeline_layout,
                                        "environment.pipeline.layout");

            VkPipelineRenderingCreateInfo ri{};
            ri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachmentFormats = &color_format;
            ri.depthAttachmentFormat = depth_format;
            ri.stencilAttachmentFormat = depth_format;

            VkGraphicsPipelineCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pci.pNext = &ri;
            pci.stageCount = 2;
            pci.pStages = stages.data();
            pci.pVertexInputState = &vi;
            pci.pInputAssemblyState = &ia;
            pci.pViewportState = &vp;
            pci.pRasterizationState = &raster;
            pci.pMultisampleState = &ms;
            pci.pDepthStencilState = &depth;
            pci.pColorBlendState = &cb;
            pci.pDynamicState = &ds;
            pci.layout = pipeline_layout;

            const VkResult r = vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pci, nullptr, &pipeline);
            vkDestroyShaderModule(device, vert, nullptr);
            vkDestroyShaderModule(device, frag, nullptr);
            if (r != VK_SUCCESS) {
                return reportVkFailure(
                    "vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pci, nullptr, &pipeline)",
                    r,
                    std::format("Environment graphics-pipeline creation failed (device={:#x}, pipeline_cache={:#x}, pipeline_layout={:#x}, color_format={}, depth_format={})",
                                vkHandleValue(device),
                                vkHandleValue(pipeline_cache),
                                vkHandleValue(pipeline_layout),
                                static_cast<int>(color_format),
                                static_cast<int>(depth_format)));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE,
                                        pipeline,
                                        "environment.pipeline");
            return true;
        }

        VkCommandBuffer beginCmds() const {
            VkCommandBufferAllocateInfo a{};
            a.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            a.commandPool = transfer_pool;
            a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            a.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            VkResult result = vkAllocateCommandBuffers(device, &a, &cb);
            if (result != VK_SUCCESS) {
                LOG_ERROR("Vulkan: {}",
                          formatVkCheckFailure(
                              "vkAllocateCommandBuffers(device, &a, &cb)",
                              result,
                              std::format("Environment upload command-buffer allocation failed (device={:#x}, command_pool={:#x}, requested_count={})",
                                          vkHandleValue(device),
                                          vkHandleValue(transfer_pool),
                                          a.commandBufferCount),
                              __FILE__,
                              __LINE__));
                return VK_NULL_HANDLE;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER,
                                        cb,
                                        "environment.upload.command");
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            result = vkBeginCommandBuffer(cb, &bi);
            if (result != VK_SUCCESS) {
                LOG_ERROR("Vulkan: {}",
                          formatVkCheckFailure(
                              "vkBeginCommandBuffer(cb, &bi)",
                              result,
                              std::format("Environment upload command buffer did not enter recording state (command_buffer={:#x}, command_pool={:#x})",
                                          vkHandleValue(cb),
                                          vkHandleValue(transfer_pool)),
                              __FILE__,
                              __LINE__));
                vkFreeCommandBuffers(device, transfer_pool, 1, &cb);
                return VK_NULL_HANDLE;
            }
            return cb;
        }

        bool endCmds(VkCommandBuffer cb) const {
            VkResult r = vkEndCommandBuffer(cb);
            if (r != VK_SUCCESS) {
                vkFreeCommandBuffers(device, transfer_pool, 1, &cb);
                return reportVkFailure(
                    "vkEndCommandBuffer(cb)",
                    r,
                    std::format("Environment upload command buffer did not leave recording state (command_buffer={:#x}, command_pool={:#x})",
                                vkHandleValue(cb),
                                vkHandleValue(transfer_pool)));
            }
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            VkFenceCreateInfo fi{};
            fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VkFence fence = VK_NULL_HANDLE;
            r = vkCreateFence(device, &fi, nullptr, &fence);
            if (r == VK_SUCCESS) {
                context->setDebugObjectName(VK_OBJECT_TYPE_FENCE,
                                            fence,
                                            "environment.upload.fence");
            }
            std::string failed_expression;
            std::string failed_context;
            if (r != VK_SUCCESS) {
                failed_expression = "vkCreateFence(device, &fi, nullptr, &fence)";
                failed_context = std::format(
                    "Environment upload fence creation failed (device={:#x}, command_buffer={:#x})",
                    vkHandleValue(device),
                    vkHandleValue(cb));
            } else if (graphics_queue == VK_NULL_HANDLE || cb == VK_NULL_HANDLE ||
                       fence == VK_NULL_HANDLE || si.commandBufferCount != 1 ||
                       si.pCommandBuffers == nullptr || si.pCommandBuffers[0] != cb) {
                r = VK_ERROR_INITIALIZATION_FAILED;
                failed_expression = "environment upload submit integrity check";
                failed_context = std::format(
                    "Environment upload submit requires a non-null queue, one expected command buffer, and a non-null fence (queue={:#x}, command_buffer={:#x}, fence={:#x}, command_buffer_count={}, command_buffer_array={:#x}, submitted_command_buffer={:#x})",
                    vkHandleValue(graphics_queue),
                    vkHandleValue(cb),
                    vkHandleValue(fence),
                    si.commandBufferCount,
                    reinterpret_cast<std::uintptr_t>(si.pCommandBuffers),
                    si.pCommandBuffers != nullptr ? vkHandleValue(si.pCommandBuffers[0]) : 0);
            }
            if (r == VK_SUCCESS) {
                r = vkQueueSubmit(graphics_queue, 1, &si, fence);
                if (r != VK_SUCCESS) {
                    failed_expression = "vkQueueSubmit(graphics_queue, 1, &si, fence)";
                    failed_context = std::format(
                        "Environment upload submission failed (queue={:#x}, command_buffer={:#x}, command_buffer_count=1, wait_semaphore_count=0, signal_semaphore_count=0, fence={:#x})",
                        vkHandleValue(graphics_queue),
                        vkHandleValue(cb),
                        vkHandleValue(fence));
                }
            }
            if (r == VK_SUCCESS) {
                r = vkWaitForFences(device, 1, &fence, VK_TRUE,
                                    std::numeric_limits<std::uint64_t>::max());
                if (r != VK_SUCCESS) {
                    failed_expression =
                        "vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX)";
                    failed_context = std::format(
                        "Environment upload submission did not retire (device={:#x}, fence={:#x}, command_buffer={:#x}, fence_count=1)",
                        vkHandleValue(device),
                        vkHandleValue(fence),
                        vkHandleValue(cb));
                }
            }
            if (fence != VK_NULL_HANDLE)
                vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, transfer_pool, 1, &cb);
            if (r != VK_SUCCESS) {
                return reportVkFailure(
                    failed_expression,
                    r,
                    failed_context);
            }
            return true;
        }

        bool loadFromPath(const std::filesystem::path& path) {
            if (!retireAndDestroyImage("environment texture reload")) {
                return false;
            }
            if (path.empty()) {
                return false;
            }
            std::filesystem::path resolved = path;
            if (!resolved.is_absolute() && !std::filesystem::exists(resolved)) {
                try {
                    resolved = lfs::vis::getAssetPath(lfs::core::path_to_utf8(path));
                } catch (const std::exception&) {
                    resolved = lfs::core::getAssetsDir() / path;
                }
            }
            const std::string utf8 = lfs::core::path_to_utf8(resolved);
            std::unique_ptr<OIIO::ImageInput> in(OIIO::ImageInput::open(utf8));
            if (!in) {
                LOG_WARN("VulkanEnvironmentPass: failed to open environment map {}: {}", utf8, OIIO::geterror());
                return false;
            }
            const OIIO::ImageSpec& spec = in->spec();
            if (spec.width <= 0 || spec.height <= 0 || spec.nchannels <= 0) {
                in->close();
                return false;
            }
            const int w = spec.width;
            const int h = spec.height;
            const int nch = spec.nchannels;
            std::vector<float> source(static_cast<std::size_t>(w) * h * nch);
            if (!in->read_image(0, 0, 0, nch, OIIO::TypeDesc::FLOAT, source.data())) {
                LOG_WARN("VulkanEnvironmentPass: failed to read environment map {}: {}", utf8, in->geterror());
                in->close();
                return false;
            }
            in->close();

            // Repack to RGBA half-float (R16G16B16A16_SFLOAT). 3-channel float formats
            // are spotty in Vulkan, RGBA half is universally supported.
            const std::size_t pixel_count = static_cast<std::size_t>(w) * h;
            std::vector<std::uint16_t> rgba(pixel_count * 4);
            for (std::size_t i = 0; i < pixel_count; ++i) {
                const float r = nch >= 1 ? source[i * nch + 0] : 0.0f;
                const float g = nch >= 2 ? source[i * nch + 1] : r;
                const float b = nch >= 3 ? source[i * nch + 2] : r;
                rgba[i * 4 + 0] = floatToHalf(r);
                rgba[i * 4 + 1] = floatToHalf(g);
                rgba[i * 4 + 2] = floatToHalf(b);
                rgba[i * 4 + 3] = floatToHalf(1.0f);
            }

            VkImageCreateInfo img{};
            img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            img.imageType = VK_IMAGE_TYPE_2D;
            img.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            img.extent = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), 1};
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
            LFS_VK_CHECK_MSG(
                vmaCreateImage(allocator, &img, &ai, &image, &image_alloc, &allocation_info),
                "Environment image allocation failed (allocator={:#x}, path='{}', requested_extent={}x{}, format={}, usage={:#x})",
                reinterpret_cast<std::uintptr_t>(allocator),
                utf8,
                w,
                h,
                static_cast<int>(img.format),
                static_cast<std::uint32_t>(img.usage));
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE,
                                         image,
                                         "environment.image[{}x{}]",
                                         w,
                                         h);
            image_vram_label = std::format("env:{}:{}x{}",
                                           lfs::core::path_to_utf8(path),
                                           w,
                                           h);
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.environment.image",
                image_vram_label,
                static_cast<std::size_t>(allocation_info.size));

            const VkDeviceSize bytes = static_cast<VkDeviceSize>(rgba.size()) * sizeof(std::uint16_t);
            VkBuffer staging = VK_NULL_HANDLE;
            VmaAllocation staging_alloc = VK_NULL_HANDLE;
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = bytes;
            bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo sa{};
            sa.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            sa.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            VkResult result =
                vmaCreateBuffer(allocator, &bi, &sa, &staging, &staging_alloc, nullptr);
            if (result != VK_SUCCESS) {
                destroyImage();
                return reportVkFailure(
                    "vmaCreateBuffer(allocator, &bi, &sa, &staging, &staging_alloc, nullptr)",
                    result,
                    std::format("Environment staging-buffer allocation failed (allocator={:#x}, path='{}', requested_size={}, usage={:#x})",
                                reinterpret_cast<std::uintptr_t>(allocator),
                                utf8,
                                bytes,
                                static_cast<std::uint32_t>(bi.usage)));
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                         staging,
                                         "environment.upload.staging[{}]",
                                         bytes);
            void* mapped = nullptr;
            result = vmaMapMemory(allocator, staging_alloc, &mapped);
            if (result != VK_SUCCESS) {
                vmaDestroyBuffer(allocator, staging, staging_alloc);
                destroyImage();
                return reportVkFailure(
                    "vmaMapMemory(allocator, staging_alloc, &mapped)",
                    result,
                    std::format("Environment staging allocation could not be mapped (allocator={:#x}, allocation={:#x}, buffer={:#x}, requested_size={})",
                                reinterpret_cast<std::uintptr_t>(allocator),
                                reinterpret_cast<std::uintptr_t>(staging_alloc),
                                vkHandleValue(staging),
                                bytes));
            }
            std::memcpy(mapped, rgba.data(), static_cast<std::size_t>(bytes));
            const VkResult flush_result = vmaFlushAllocation(allocator, staging_alloc, 0, bytes);
            vmaUnmapMemory(allocator, staging_alloc);
            if (flush_result != VK_SUCCESS) {
                vmaDestroyBuffer(allocator, staging, staging_alloc);
                destroyImage();
                return reportVkFailure(
                    "vmaFlushAllocation(allocator, staging_alloc, 0, bytes)",
                    flush_result,
                    std::format("Environment staging flush failed (allocator={:#x}, allocation={:#x}, buffer={:#x}, offset=0, flush_size={})",
                                reinterpret_cast<std::uintptr_t>(allocator),
                                reinterpret_cast<std::uintptr_t>(staging_alloc),
                                vkHandleValue(staging),
                                bytes));
            }

            VkCommandBuffer cb = beginCmds();
            if (cb == VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, staging, staging_alloc);
                destroyImage();
                return false;
            }

            cmdImageBarrier2(cb, image, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                             VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), 1};
            vkCmdCopyBufferToImage(cb, staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            cmdImageBarrier2(cb, image, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

            const bool ok = endCmds(cb);
            vmaDestroyBuffer(allocator, staging, staging_alloc);
            if (!ok) {
                destroyImage();
                return false;
            }

            VkImageViewCreateInfo iv{};
            iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            iv.image = image;
            iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            iv.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            iv.subresourceRange.levelCount = 1;
            iv.subresourceRange.layerCount = 1;
            const VkResult view_result = vkCreateImageView(device, &iv, nullptr, &image_view);
            if (view_result != VK_SUCCESS) {
                destroyImage();
                return reportVkFailure(
                    "vkCreateImageView(device, &iv, nullptr, &image_view)",
                    view_result,
                    std::format("Environment image-view creation failed (device={:#x}, image={:#x}, path='{}', extent={}x{}, format={}, aspect_mask={:#x})",
                                vkHandleValue(device),
                                vkHandleValue(iv.image),
                                utf8,
                                w,
                                h,
                                static_cast<int>(iv.format),
                                static_cast<std::uint32_t>(iv.subresourceRange.aspectMask)));
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW,
                                         image_view,
                                         "environment.image[{}x{}].view",
                                         w,
                                         h);

            loaded_path = path;
            return true;
        }

        void prepare(const VulkanEnvironmentParams& params, const std::size_t frame_slot) {
            auto& descriptor = descriptorForFrame(frame_slot);
            if (!params.enabled) {
                retireAndDestroyImage("environment texture release");
                load_failed_for_path = false;
                return;
            }
            if (params.map_path == loaded_path && image != VK_NULL_HANDLE) {
                rebindDescriptor(descriptor);
                return;
            }
            if (params.map_path.empty()) {
                retireAndDestroyImage("empty environment path");
                return;
            }
            // Skip retry of a path we've already failed once for, until it changes.
            if (load_failed_for_path && params.map_path == loaded_path) {
                return;
            }
            const bool ok = loadFromPath(params.map_path);
            load_failed_for_path = !ok;
            loaded_path = params.map_path;
            if (ok) {
                rebindDescriptor(descriptor);
            }
        }

        void record(VkCommandBuffer cb, VkRect2D rect, const VulkanEnvironmentParams& params,
                    const std::size_t frame_slot) {
            const auto& descriptor = descriptorForFrame(frame_slot);
            if (cb == VK_NULL_HANDLE || descriptor.set == VK_NULL_HANDLE) [[unlikely]] {
                throw std::logic_error(std::format(
                    "Environment recording requires a command buffer and per-frame descriptor set (command_buffer={:#x}, frame_slot={}, ring_size={}, descriptor_set={:#x}, bound_view={:#x}) ({}:{})",
                    vkHandleValue(cb),
                    frame_slot,
                    frame_descriptors.size(),
                    vkHandleValue(descriptor.set),
                    vkHandleValue(descriptor.bound_view),
                    __FILE__,
                    __LINE__));
            }
            if (!params.enabled || pipeline == VK_NULL_HANDLE || descriptor.bound_view == VK_NULL_HANDLE ||
                screen_quad_buffer == VK_NULL_HANDLE) {
                return;
            }
            VkViewport vp{};
            vp.x = static_cast<float>(rect.offset.x);
            vp.y = static_cast<float>(rect.offset.y);
            vp.width = static_cast<float>(rect.extent.width);
            vp.height = static_cast<float>(rect.extent.height);
            vp.minDepth = 0.0f;
            vp.maxDepth = 1.0f;
            vkCmdSetViewport(cb, 0, 1, &vp);
            vkCmdSetScissor(cb, 0, 1, &rect);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &descriptor.set, 0, nullptr);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cb, 0, 1, &screen_quad_buffer, &offset);

            EnvPush push{};
            // mat3 → mat4 with column-major glm layout (last column unused).
            const glm::mat3& r = params.camera_to_world;
            const float m[16] = {
                r[0][0], r[0][1], r[0][2], 0.0f,
                r[1][0], r[1][1], r[1][2], 0.0f,
                r[2][0], r[2][1], r[2][2], 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f};
            std::memcpy(push.cam_to_world, m, sizeof(push.cam_to_world));
            push.intrinsics[0] = params.intrinsics.x;
            push.intrinsics[1] = params.intrinsics.y;
            push.intrinsics[2] = params.intrinsics.z;
            push.intrinsics[3] = params.intrinsics.w;
            push.viewport_exposure[0] = params.viewport_size.x;
            push.viewport_exposure[1] = params.viewport_size.y;
            push.viewport_exposure[2] = params.exposure;
            push.viewport_exposure[3] = params.rotation_radians;
            push.flags[0] = params.equirectangular_view ? 1.0f : 0.0f;
            vkCmdPushConstants(cb, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(push), &push);
            vkCmdDraw(cb, 6, 1, 0, 0);
        }
    };

    VulkanEnvironmentPass::VulkanEnvironmentPass() = default;
    VulkanEnvironmentPass::~VulkanEnvironmentPass() = default;
    VulkanEnvironmentPass::VulkanEnvironmentPass(VulkanEnvironmentPass&&) noexcept = default;
    VulkanEnvironmentPass& VulkanEnvironmentPass::operator=(VulkanEnvironmentPass&&) noexcept = default;

    bool VulkanEnvironmentPass::init(VulkanContext& context, VkFormat color_format,
                                     VkFormat depth_format, VkBuffer screen_quad) {
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        return impl_->init(context, color_format, depth_format, screen_quad);
    }

    void VulkanEnvironmentPass::prepare(const VulkanEnvironmentParams& params,
                                        const std::size_t frame_slot) {
        if (impl_)
            impl_->prepare(params, frame_slot);
    }

    void VulkanEnvironmentPass::record(VkCommandBuffer cb, VkRect2D rect,
                                       const VulkanEnvironmentParams& params,
                                       const std::size_t frame_slot) {
        if (impl_)
            impl_->record(cb, rect, params, frame_slot);
    }

    void VulkanEnvironmentPass::shutdown() {
        if (impl_) {
            impl_->destroy();
            impl_.reset();
        }
    }

    bool VulkanEnvironmentPass::hasTexture(const std::size_t frame_slot) const {
        return impl_ && impl_->descriptorForFrame(frame_slot).bound_view != VK_NULL_HANDLE;
    }

} // namespace lfs::vis
