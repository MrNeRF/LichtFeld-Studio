/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_environment_pass.hpp"
#include "shared_viewport_gpu_assets.hpp"

#include "core/logger.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_result.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <glm/glm.hpp>
#include <memory>
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

    } // namespace

    struct VulkanEnvironmentPass::Impl {
        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;

        VkBuffer screen_quad_buffer = VK_NULL_HANDLE;
        VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
        VkDescriptorPool desc_pool = VK_NULL_HANDLE;
        struct FrameDescriptor {
            VkDescriptorSet set = VK_NULL_HANDLE;
            VkImageView bound_view = VK_NULL_HANDLE;
        };
        std::vector<FrameDescriptor> frame_descriptors;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        std::shared_ptr<SharedViewportGpuAssets> assets;

        ~Impl() { destroy(); }

        bool init(VulkanContext& ctx,
                  VkFormat color_format,
                  VkFormat depth_format,
                  VkBuffer quad,
                  std::shared_ptr<SharedViewportGpuAssets> shared_assets) {
            if (!shared_assets) {
                shared_assets = std::make_shared<SharedViewportGpuAssets>();
            }
            assets = std::move(shared_assets);
            if (!assets->ensureContext(ctx)) {
                return logVkFailure(std::format(
                    "Environment-pass initialization could not bind shared scene GPU assets ({}:{})",
                    __FILE__,
                    __LINE__));
            }
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

            return createDescriptors() && createPipeline(color_format, depth_format);
        }

        void destroy() {
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
            device = VK_NULL_HANDLE;
            allocator = VK_NULL_HANDLE;
            assets.reset();
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
                        "Environment descriptor-set layout creation failed (device={:#x}, binding_count={}, descriptor_type={})",
                        vkHandleValue(device),
                        li.bindingCount,
                        static_cast<int>(b.descriptorType)),
                    std::source_location::current())) {
                return false;
            }
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
            if (!vk_try_bool(
                    vkCreateDescriptorPool(device, &pci, nullptr, &desc_pool),
                    "vkCreateDescriptorPool(device, &pci, nullptr, &desc_pool)",
                    lfs::rendering::formatVulkanDiagnostic(
                        "Environment descriptor-pool creation failed (device={:#x}, frame_count={}, max_sets={}, descriptor_count={})",
                        vkHandleValue(device),
                        frame_count,
                        pci.maxSets,
                        ps.descriptorCount),
                    std::source_location::current())) {
                return false;
            }
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
            if (!vk_try_bool(
                    vkAllocateDescriptorSets(device, &ai, sets.data()),
                    "vkAllocateDescriptorSets(device, &ai, sets.data())",
                    lfs::rendering::formatVulkanDiagnostic(
                        "Environment descriptor-set allocation failed (device={:#x}, descriptor_pool={:#x}, descriptor_layout={:#x}, requested_count={})",
                        vkHandleValue(device),
                        vkHandleValue(desc_pool),
                        vkHandleValue(desc_layout),
                        ai.descriptorSetCount),
                    std::source_location::current())) {
                return false;
            }
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

        void rebindDescriptor(FrameDescriptor& descriptor, const SharedEnvironmentTexture& texture) const {
            if (texture.image_view == VK_NULL_HANDLE || descriptor.bound_view == texture.image_view) {
                return;
            }
            VkDescriptorImageInfo image_info{};
            image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            image_info.imageView = texture.image_view;
            image_info.sampler = texture.sampler;
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptor.set;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &image_info;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
            descriptor.bound_view = texture.image_view;
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

        void prepare(const VulkanEnvironmentParams& params, const std::size_t frame_slot) {
            auto& descriptor = descriptorForFrame(frame_slot);
            if (!assets || (context != nullptr && !assets->ensureContext(*context))) {
                descriptor.bound_view = VK_NULL_HANDLE;
                return;
            }
            assets->prepareEnvironment(params, frame_slot);
            if (!params.enabled) {
                descriptor.bound_view = VK_NULL_HANDLE;
                return;
            }
            const SharedEnvironmentTexture texture = assets->environmentTexture();
            if (texture.image_view == VK_NULL_HANDLE) {
                descriptor.bound_view = VK_NULL_HANDLE;
                return;
            }
            rebindDescriptor(descriptor, texture);
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
                                     VkFormat depth_format, VkBuffer screen_quad,
                                     std::shared_ptr<SharedViewportGpuAssets> shared_assets) {
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        return impl_->init(context, color_format, depth_format, screen_quad, std::move(shared_assets));
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
