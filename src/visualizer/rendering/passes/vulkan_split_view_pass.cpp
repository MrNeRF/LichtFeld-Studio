/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_split_view_pass.hpp"

#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "rendering/image_layout.hpp"
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
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <vk_mem_alloc.h>

#include "viewport/screen_quad.vert.spv.h"
#include "viewport/split_view.frag.spv.h"

namespace lfs::vis {

    namespace {

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

        struct SplitPush {
            float split[4];       // x = position, y = left_flip_y, z = right_flip_y, w = pad
            float rect[4];        // x, y, w, h
            float panel_norm[4];  // left_start, left_end, right_start, right_end
            float panel_flags[4]; // left_normalize, right_normalize, pad, pad
            float background[4];  // rgb + pad
            float divider[4];     // bar_half_w, handle_half_w, handle_half_h, corner_radius
            float grip[4];        // spacing, half_w, half_l, line_count
        };
        static_assert(sizeof(SplitPush) == 7 * 16);

        // Convert a CHW float [0,1] tensor (CUDA or CPU) into a tightly packed RGBA8
        // buffer at `dst`. TBB-parallel over rows. Caller owns the destination memory
        // (we point straight at a persistently-mapped VMA staging buffer when possible)
        // so this function performs zero allocations on the hot path.
        bool packPanelToRgba8(const lfs::core::Tensor& tensor,
                              std::uint8_t* dst,
                              std::uint32_t& out_w,
                              std::uint32_t& out_h) {
            if (!tensor.is_valid() || tensor.ndim() != 3) {
                return false;
            }
            const auto layout = lfs::rendering::detectImageLayout(tensor);
            if (layout == lfs::rendering::ImageLayout::Unknown) {
                return false;
            }
            lfs::core::Tensor cpu = tensor;
            if (cpu.dtype() == lfs::core::DataType::UInt8) {
                cpu = cpu.to(lfs::core::DataType::Float32) / 255.0f;
            } else if (cpu.dtype() != lfs::core::DataType::Float32) {
                cpu = cpu.to(lfs::core::DataType::Float32);
            }
            if (layout == lfs::rendering::ImageLayout::HWC) {
                cpu = cpu.permute({2, 0, 1}).contiguous();
            }
            cpu = cpu.cpu().contiguous();

            const int channels = static_cast<int>(cpu.size(0));
            const int h = static_cast<int>(cpu.size(1));
            const int w = static_cast<int>(cpu.size(2));
            if (channels < 3 || h <= 0 || w <= 0) {
                return false;
            }
            out_w = static_cast<std::uint32_t>(w);
            out_h = static_cast<std::uint32_t>(h);
            const float* data = cpu.ptr<float>();
            const std::size_t plane = static_cast<std::size_t>(w) * h;

            tbb::parallel_for(
                tbb::blocked_range<int>(0, h),
                [&](const tbb::blocked_range<int>& r) {
                    for (int y = r.begin(); y != r.end(); ++y) {
                        std::uint8_t* row_dst = dst + static_cast<std::size_t>(y) * w * 4u;
                        const float* row_r = data + static_cast<std::size_t>(y) * w;
                        const float* row_g = data + plane + static_cast<std::size_t>(y) * w;
                        const float* row_b = data + 2u * plane + static_cast<std::size_t>(y) * w;
                        const float* row_a = (channels >= 4)
                                                 ? data + 3u * plane + static_cast<std::size_t>(y) * w
                                                 : nullptr;
                        for (int x = 0; x < w; ++x) {
                            const float fr = std::clamp(row_r[x], 0.0f, 1.0f);
                            const float fg = std::clamp(row_g[x], 0.0f, 1.0f);
                            const float fb = std::clamp(row_b[x], 0.0f, 1.0f);
                            const float fa = row_a ? std::clamp(row_a[x], 0.0f, 1.0f) : 1.0f;
                            row_dst[x * 4 + 0] = static_cast<std::uint8_t>(fr * 255.0f + 0.5f);
                            row_dst[x * 4 + 1] = static_cast<std::uint8_t>(fg * 255.0f + 0.5f);
                            row_dst[x * 4 + 2] = static_cast<std::uint8_t>(fb * 255.0f + 0.5f);
                            row_dst[x * 4 + 3] = static_cast<std::uint8_t>(fa * 255.0f + 0.5f);
                        }
                    }
                });
            return true;
        }

    } // namespace

    struct VulkanSplitViewPass::Impl {
        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;
        VkCommandPool transfer_pool = VK_NULL_HANDLE;

        VkSampler sampler = VK_NULL_HANDLE;
        VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
        VkDescriptorPool desc_pool = VK_NULL_HANDLE;
        struct FrameDescriptor {
            VkDescriptorSet set = VK_NULL_HANDLE;
            VkImageView left_view = VK_NULL_HANDLE;
            VkImageView right_view = VK_NULL_HANDLE;
            std::uint64_t left_generation = 0;
            std::uint64_t right_generation = 0;
            bool ready = false;
        };
        std::vector<FrameDescriptor> frame_descriptors;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkBuffer screen_quad_buffer = VK_NULL_HANDLE;

        struct PanelImage {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation alloc = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            const lfs::core::Tensor* uploaded_tensor = nullptr;
            // Persistent staging: kept alive between frames so identical-size uploads
            // don't repeatedly allocate / map / unmap an 8 MB buffer at 1080p.
            VkBuffer staging_buffer = VK_NULL_HANDLE;
            VmaAllocation staging_alloc = VK_NULL_HANDLE;
            void* staging_mapped = nullptr;
            VkDeviceSize staging_capacity = 0;
            // Persistent CHW->RGBA pack buffer. tbb::parallel_for fills this in place
            // every frame; growing only when the panel resolution increases.
            std::vector<std::uint8_t> pack_bytes;
            // Persistent command buffer for the upload submit. Reused across frames
            // via vkResetCommandBuffer instead of vkAllocate/Free per upload.
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            VkFence fence = VK_NULL_HANDLE;
            std::string image_vram_label;
        };
        PanelImage left{};
        PanelImage right{};

        ~Impl() { destroy(); }

        bool init(VulkanContext& ctx, VkFormat color_format, VkFormat depth_format,
                  VkBuffer screen_quad) {
            context = &ctx;
            device = ctx.device();
            allocator = ctx.allocator();
            pipeline_cache = ctx.pipelineCache();
            graphics_queue = ctx.graphicsQueue();
            screen_quad_buffer = screen_quad;
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE) {
                return logVkFailure(std::format(
                    "Split-view initialization requires a live device and allocator (device={:#x}, allocator={:#x}, graphics_queue={:#x}, screen_quad_buffer={:#x}) ({}:{})",
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
                        "Split-view transfer command-pool creation failed (device={:#x}, queue_family={}, flags={:#x})",
                        vkHandleValue(device),
                        pool.queueFamilyIndex,
                        static_cast<std::uint32_t>(pool.flags)),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_COMMAND_POOL,
                                        transfer_pool,
                                        "split_view.transfer.pool");
            return createSampler() && createDescriptors() &&
                   createPipeline(color_format, depth_format);
        }

        void destroy() {
            destroyPanel(left);
            destroyPanel(right);
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

        bool createSampler() {
            VkSamplerCreateInfo s{};
            s.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            s.magFilter = VK_FILTER_LINEAR;
            s.minFilter = VK_FILTER_LINEAR;
            s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            s.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            s.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            s.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (!vk_try_bool(
                    vkCreateSampler(device, &s, nullptr, &sampler),
                    "vkCreateSampler(device, &s, nullptr, &sampler)",
                    lfs::rendering::formatVulkanDiagnostic(
                        "Split-view sampler creation failed (device={:#x}, mag_filter={}, min_filter={}, address_mode={})",
                        vkHandleValue(device),
                        static_cast<int>(s.magFilter),
                        static_cast<int>(s.minFilter),
                        static_cast<int>(s.addressModeU)),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_SAMPLER,
                                        sampler,
                                        "split_view.panel.sampler");
            return true;
        }

        bool createDescriptors() {
            std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
            for (std::uint32_t i = 0; i < 2; ++i) {
                bindings[i].binding = i;
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            }
            VkDescriptorSetLayoutCreateInfo li{};
            li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            li.bindingCount = static_cast<std::uint32_t>(bindings.size());
            li.pBindings = bindings.data();
            if (!vk_try_bool(
                    vkCreateDescriptorSetLayout(device, &li, nullptr, &desc_layout),
                    "vkCreateDescriptorSetLayout(device, &li, nullptr, &desc_layout)",
                    lfs::rendering::formatVulkanDiagnostic(
                        "Split-view descriptor-set layout creation failed (device={:#x}, binding_count={}, descriptor_type={})",
                        vkHandleValue(device),
                        li.bindingCount,
                        static_cast<int>(bindings[0].descriptorType)),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                        desc_layout,
                                        "split_view.descriptor.layout");
            VkDescriptorPoolSize ps{};
            ps.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            const std::uint32_t frame_count = static_cast<std::uint32_t>(
                std::max<std::size_t>(1, context->framesInFlight()));
            ps.descriptorCount = frame_count * 2;
            VkDescriptorPoolCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pi.maxSets = frame_count;
            pi.poolSizeCount = 1;
            pi.pPoolSizes = &ps;
            if (!vk_try_bool(
                    vkCreateDescriptorPool(device, &pi, nullptr, &desc_pool),
                    "vkCreateDescriptorPool(device, &pi, nullptr, &desc_pool)",
                    lfs::rendering::formatVulkanDiagnostic(
                        "Split-view descriptor-pool creation failed (device={:#x}, frame_count={}, max_sets={}, descriptor_count={})",
                        vkHandleValue(device),
                        frame_count,
                        pi.maxSets,
                        ps.descriptorCount),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                        desc_pool,
                                        "split_view.descriptor.pool");
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
                        "Split-view descriptor-set allocation failed (device={:#x}, descriptor_pool={:#x}, descriptor_layout={:#x}, requested_count={})",
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
                                             "split_view.descriptor[{}]",
                                             i);
            }
            return true;
        }

        [[nodiscard]] FrameDescriptor& descriptorForFrame(const std::size_t frame_slot) {
            if (frame_slot >= frame_descriptors.size()) [[unlikely]] {
                throw std::logic_error(std::format(
                    "Split-view frame slot is outside the descriptor ring (frame_slot={}, ring_size={}) ({}:{})",
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
                    "Split-view frame slot is outside the descriptor ring (frame_slot={}, ring_size={}) ({}:{})",
                    frame_slot,
                    frame_descriptors.size(),
                    __FILE__,
                    __LINE__));
            }
            return frame_descriptors[frame_slot];
        }

        bool createPipeline(VkFormat color_format, VkFormat depth_format) {
            using namespace viewport_shaders;
            VkShaderModule vert = createShaderModule(device, kScreenQuadVertSpv, "Split-view");
            VkShaderModule frag = createShaderModule(device, kSplitViewFragSpv, "Split-view");
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
            binding.stride = sizeof(float) * 4;
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

            VkPipelineDepthStencilStateCreateInfo depth{};
            depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depth.depthTestEnable = VK_FALSE;
            depth.depthWriteEnable = VK_FALSE;

            VkPipelineColorBlendAttachmentState blend_attachment{};
            blend_attachment.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            blend_attachment.blendEnable = VK_FALSE;
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
            push.size = sizeof(SplitPush);

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
                    std::format("Split-view pipeline-layout creation failed (device={:#x}, descriptor_layout={:#x}, set_layout_count={}, push_constant_bytes={})",
                                vkHandleValue(device),
                                vkHandleValue(desc_layout),
                                layout_info.setLayoutCount,
                                push.size));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                        pipeline_layout,
                                        "split_view.pipeline.layout");

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
                    std::format("Split-view graphics-pipeline creation failed (device={:#x}, pipeline_cache={:#x}, pipeline_layout={:#x}, color_format={}, depth_format={})",
                                vkHandleValue(device),
                                vkHandleValue(pipeline_cache),
                                vkHandleValue(pipeline_layout),
                                static_cast<int>(color_format),
                                static_cast<int>(depth_format)));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE,
                                        pipeline,
                                        "split_view.pipeline");
            return true;
        }

        void destroyPanel(PanelImage& p) {
            // Drain any pending transfer submit so we never destroy device memory
            // that the GPU is still reading from.
            if (p.fence != VK_NULL_HANDLE) {
                lfs::rendering::WaitContext wait_ctx;
                wait_ctx.fingerprint = "pass.split_view.destroy_drain";
                auto wait_outcome = lfs::rendering::wait_fence_bounded(
                    device,
                    p.fence,
                    std::stop_token{},
                    lfs::rendering::VulkanWaitPolicy{},
                    wait_ctx);
                const bool ready = wait_outcome.has_value() &&
                                   *wait_outcome == lfs::rendering::WaitOutcome::Ready;
                if (!ready) {
                    // AMB-4: retain panel image + fence while GPU may still read them.
                    LOG_ERROR(
                        "Vulkan: split-view destroy_drain did not reach Ready "
                        "(device={:#x}, fence={:#x}, image={:#x}, image_view={:#x}, "
                        "panel_size={}x{}): {} — retaining panel resources",
                        vkHandleValue(device),
                        vkHandleValue(p.fence),
                        vkHandleValue(p.image),
                        vkHandleValue(p.view),
                        p.width,
                        p.height,
                        formatWaitFailure(wait_outcome));
                    return;
                }
            }
            if (p.view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, p.view, nullptr);
                p.view = VK_NULL_HANDLE;
            }
            if (p.image != VK_NULL_HANDLE) {
                if (!p.image_vram_label.empty()) {
                    lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                        "vulkan.split_view.panel_image",
                        p.image_vram_label,
                        0);
                }
                vmaDestroyImage(allocator, p.image, p.alloc);
                p.image = VK_NULL_HANDLE;
                p.alloc = VK_NULL_HANDLE;
            }
            if (p.staging_buffer != VK_NULL_HANDLE) {
                if (p.staging_mapped) {
                    vmaUnmapMemory(allocator, p.staging_alloc);
                    p.staging_mapped = nullptr;
                }
                vmaDestroyBuffer(allocator, p.staging_buffer, p.staging_alloc);
                p.staging_buffer = VK_NULL_HANDLE;
                p.staging_alloc = VK_NULL_HANDLE;
                p.staging_capacity = 0;
            }
            if (p.cmd != VK_NULL_HANDLE && transfer_pool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, transfer_pool, 1, &p.cmd);
                p.cmd = VK_NULL_HANDLE;
            }
            if (p.fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, p.fence, nullptr);
                p.fence = VK_NULL_HANDLE;
            }
            // ensurePanelImage() calls destroyPanel() on resize, which runs
            // *between* packPanelToRgba8() filling pack_bytes and the staging
            // memcpy reading from it — so we must NOT clear pack_bytes here.
            p.width = 0;
            p.height = 0;
            p.image_vram_label.clear();
            p.uploaded_tensor = nullptr;
            for (auto& descriptor : frame_descriptors) {
                if (&p == &left) {
                    descriptor.left_view = VK_NULL_HANDLE;
                    descriptor.left_generation = 0;
                } else {
                    descriptor.right_view = VK_NULL_HANDLE;
                    descriptor.right_generation = 0;
                }
                descriptor.ready = false;
            }
        }

        bool ensureStaging(PanelImage& p, VkDeviceSize bytes) {
            const char* const side = &p == &left ? "left" : "right";
            if (p.staging_buffer != VK_NULL_HANDLE && p.staging_capacity >= bytes) {
                return true;
            }
            if (p.staging_buffer != VK_NULL_HANDLE) {
                if (p.staging_mapped) {
                    vmaUnmapMemory(allocator, p.staging_alloc);
                    p.staging_mapped = nullptr;
                }
                vmaDestroyBuffer(allocator, p.staging_buffer, p.staging_alloc);
                p.staging_buffer = VK_NULL_HANDLE;
                p.staging_alloc = VK_NULL_HANDLE;
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
                    vmaCreateBuffer(allocator, &bi, &sa, &p.staging_buffer, &p.staging_alloc, &ai),
                    "vmaCreateBuffer(allocator, &bi, &sa, &p.staging_buffer, &p.staging_alloc, &ai)",
                    lfs::rendering::formatVulkanDiagnostic(
                        "Split-view panel staging-buffer allocation failed (side={}, allocator={:#x}, requested_size={}, usage={:#x})",
                        side,
                        reinterpret_cast<std::uintptr_t>(allocator),
                        bytes,
                        static_cast<std::uint32_t>(bi.usage)),
                    std::source_location::current())) {
                return false;
            }
            p.staging_mapped = ai.pMappedData;
            p.staging_capacity = bytes;
            context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                         p.staging_buffer,
                                         "split_view.{}.upload.staging[{}]",
                                         side,
                                         p.staging_capacity);
            return true;
        }

        bool ensurePanelCmd(PanelImage& p) {
            const char* const side = &p == &left ? "left" : "right";
            if (p.cmd != VK_NULL_HANDLE && p.fence != VK_NULL_HANDLE) {
                return true;
            }
            if (p.cmd != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, transfer_pool, 1, &p.cmd);
                p.cmd = VK_NULL_HANDLE;
            }
            if (p.fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, p.fence, nullptr);
                p.fence = VK_NULL_HANDLE;
            }
            VkCommandBufferAllocateInfo a{};
            a.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            a.commandPool = transfer_pool;
            a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            a.commandBufferCount = 1;
            if (!vk_try_bool(
                    vkAllocateCommandBuffers(device, &a, &p.cmd),
                    "vkAllocateCommandBuffers(device, &a, &p.cmd)",
                    lfs::rendering::formatVulkanDiagnostic(
                        "Split-view panel command-buffer allocation failed (side={}, device={:#x}, command_pool={:#x}, requested_count={})",
                        side,
                        vkHandleValue(device),
                        vkHandleValue(transfer_pool),
                        a.commandBufferCount),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_COMMAND_BUFFER,
                                         p.cmd,
                                         "split_view.{}.transfer.command",
                                         side);
            VkFenceCreateInfo fi{};
            fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            // Created signaled so the first vkWaitForFences before recording is a no-op.
            fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            const VkResult fence_result = vkCreateFence(device, &fi, nullptr, &p.fence);
            if (fence_result != VK_SUCCESS) {
                vkFreeCommandBuffers(device, transfer_pool, 1, &p.cmd);
                p.cmd = VK_NULL_HANDLE;
                return reportVkFailure(
                    "vkCreateFence(device, &fi, nullptr, &p.fence)",
                    fence_result,
                    std::format("Split-view panel transfer-fence creation failed (side={}, device={:#x}, command_pool={:#x}, flags={:#x})",
                                side,
                                vkHandleValue(device),
                                vkHandleValue(transfer_pool),
                                static_cast<std::uint32_t>(fi.flags)));
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_FENCE,
                                         p.fence,
                                         "split_view.{}.transfer.fence",
                                         side);
            return true;
        }

        bool replacePanelFenceSignaled(PanelImage& panel,
                                       const char* const failed_operation,
                                       const VkResult failed_result) {
            const char* const side = &panel == &left ? "left" : "right";
            LOG_ERROR("Vulkan: {}",
                      formatVkCheckFailure(
                          failed_operation,
                          failed_result,
                          std::format("Split-view panel transfer command lifecycle failed (side={}, device={:#x}, queue={:#x}, command_pool={:#x}, command_buffer={:#x}, fence={:#x})",
                                      side,
                                      vkHandleValue(device),
                                      vkHandleValue(graphics_queue),
                                      vkHandleValue(transfer_pool),
                                      vkHandleValue(panel.cmd),
                                      vkHandleValue(panel.fence)),
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
                              std::format("Split-view failed to replace a poisoned panel transfer fence (side={}, device={:#x}, old_fence={:#x}, command_buffer={:#x}, flags={:#x})",
                                          side,
                                          vkHandleValue(device),
                                          vkHandleValue(panel.fence),
                                          vkHandleValue(panel.cmd),
                                          static_cast<std::uint32_t>(info.flags)),
                              __FILE__,
                              __LINE__));
                return false;
            }
            if (panel.fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, panel.fence, nullptr);
            }
            panel.fence = replacement;
            context->setDebugObjectNamef(VK_OBJECT_TYPE_FENCE,
                                         panel.fence,
                                         "split_view.{}.transfer.fence",
                                         side);
            return false;
        }

        bool ensurePanelImage(PanelImage& p, std::uint32_t w, std::uint32_t h) {
            if (p.image != VK_NULL_HANDLE && p.width == w && p.height == h) {
                return true;
            }
            if (p.image != VK_NULL_HANDLE && context != nullptr && !context->waitForSubmittedFrames()) {
                LOG_ERROR("VulkanSplitViewPass: could not retire frames before panel resize: {}",
                          context->lastError());
                return false;
            }
            destroyPanel(p);
            VkImageCreateInfo img{};
            img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            img.imageType = VK_IMAGE_TYPE_2D;
            img.format = VK_FORMAT_R8G8B8A8_UNORM;
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
            const char* const side = &p == &left ? "left" : "right";
            if (!vk_try_bool(
                    vmaCreateImage(allocator, &img, &ai, &p.image, &p.alloc, &allocation_info),
                    "vmaCreateImage(allocator, &img, &ai, &p.image, &p.alloc, &allocation_info)",
                    lfs::rendering::formatVulkanDiagnostic(
                        "Split-view panel image allocation failed (side={}, allocator={:#x}, requested_extent={}x{}, format={}, usage={:#x})",
                        side,
                        reinterpret_cast<std::uintptr_t>(allocator),
                        w,
                        h,
                        static_cast<int>(img.format),
                        static_cast<std::uint32_t>(img.usage)),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE,
                                         p.image,
                                         "split_view.{}.image[{}x{}]",
                                         side,
                                         w,
                                         h);
            p.image_vram_label = std::format("{}:{}x{}", side, w, h);
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.split_view.panel_image",
                p.image_vram_label,
                static_cast<std::size_t>(allocation_info.size));
            VkImageViewCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vi.image = p.image;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = VK_FORMAT_R8G8B8A8_UNORM;
            vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vi.subresourceRange.levelCount = 1;
            vi.subresourceRange.layerCount = 1;
            const VkResult view_result = vkCreateImageView(device, &vi, nullptr, &p.view);
            if (view_result != VK_SUCCESS) {
                destroyPanel(p);
                return reportVkFailure(
                    "vkCreateImageView(device, &vi, nullptr, &p.view)",
                    view_result,
                    std::format("Split-view panel image-view creation failed (side={}, device={:#x}, image={:#x}, requested_extent={}x{}, format={}, aspect_mask={:#x})",
                                side,
                                vkHandleValue(device),
                                vkHandleValue(vi.image),
                                w,
                                h,
                                static_cast<int>(vi.format),
                                static_cast<std::uint32_t>(vi.subresourceRange.aspectMask)));
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW,
                                         p.view,
                                         "split_view.{}.image[{}x{}].view",
                                         side,
                                         w,
                                         h);
            p.width = w;
            p.height = h;
            return true;
        }

        bool uploadPanel(PanelImage& panel, const lfs::core::Tensor& tensor) {
            const char* const side = &panel == &left ? "left" : "right";
            // Probe size from the tensor; resize the persistent pack buffer only when
            // the tensor's resolution exceeds the current capacity.
            if (!tensor.is_valid() || tensor.ndim() != 3) {
                return logVkFailure(std::format(
                    "Split-view panel upload requires a valid rank-3 image tensor (side={}, tensor_valid={}, observed_rank={}) ({}:{})",
                    side,
                    tensor.is_valid(),
                    tensor.is_valid() ? tensor.ndim() : 0,
                    __FILE__,
                    __LINE__));
            }
            const auto layout = lfs::rendering::detectImageLayout(tensor);
            const int probe_w = static_cast<int>(layout == lfs::rendering::ImageLayout::HWC
                                                     ? tensor.size(1)
                                                     : tensor.size(2));
            const int probe_h = static_cast<int>(layout == lfs::rendering::ImageLayout::HWC
                                                     ? tensor.size(0)
                                                     : tensor.size(1));
            if (probe_w <= 0 || probe_h <= 0) {
                return logVkFailure(std::format(
                    "Split-view panel upload requires positive image dimensions (side={}, layout={}, observed_width={}, observed_height={}, tensor_rank={}) ({}:{})",
                    side,
                    static_cast<int>(layout),
                    probe_w,
                    probe_h,
                    tensor.ndim(),
                    __FILE__,
                    __LINE__));
            }
            const std::size_t need_pack = static_cast<std::size_t>(probe_w) * probe_h * 4u;
            if (panel.pack_bytes.size() < need_pack) {
                panel.pack_bytes.resize(need_pack);
            }

            std::uint32_t pkt_w = 0;
            std::uint32_t pkt_h = 0;
            if (!packPanelToRgba8(tensor, panel.pack_bytes.data(), pkt_w, pkt_h)) {
                return logVkFailure(std::format(
                    "Split-view panel packing failed for a valid upload request (side={}, tensor_rank={}, layout={}, probe_size={}x{}, destination_capacity={}) ({}:{})",
                    side,
                    tensor.ndim(),
                    static_cast<int>(layout),
                    probe_w,
                    probe_h,
                    panel.pack_bytes.size(),
                    __FILE__,
                    __LINE__));
            }
            if (!ensurePanelImage(panel, pkt_w, pkt_h)) {
                return false;
            }

            const VkDeviceSize bytes =
                static_cast<VkDeviceSize>(pkt_w) * pkt_h * 4u;
            if (!ensureStaging(panel, bytes) || !ensurePanelCmd(panel)) {
                return false;
            }
            if (panel.staging_buffer == VK_NULL_HANDLE || panel.staging_alloc == VK_NULL_HANDLE ||
                panel.staging_mapped == nullptr || panel.staging_capacity < bytes ||
                panel.image == VK_NULL_HANDLE || panel.cmd == VK_NULL_HANDLE ||
                panel.fence == VK_NULL_HANDLE || graphics_queue == VK_NULL_HANDLE) {
                return logVkFailure(std::format(
                    "Split-view panel upload resources must cover the copy before recording (side={}, staging_buffer={:#x}, staging_allocation={:#x}, staging_mapped={:#x}, staging_capacity={}, copy_size={}, image={:#x}, command_buffer={:#x}, fence={:#x}, queue={:#x}) ({}:{})",
                    side,
                    vkHandleValue(panel.staging_buffer),
                    reinterpret_cast<std::uintptr_t>(panel.staging_alloc),
                    reinterpret_cast<std::uintptr_t>(panel.staging_mapped),
                    panel.staging_capacity,
                    bytes,
                    vkHandleValue(panel.image),
                    vkHandleValue(panel.cmd),
                    vkHandleValue(panel.fence),
                    vkHandleValue(graphics_queue),
                    __FILE__,
                    __LINE__));
            }
            std::memcpy(panel.staging_mapped, panel.pack_bytes.data(), static_cast<std::size_t>(bytes));
            VkResult result = vmaFlushAllocation(allocator, panel.staging_alloc, 0, bytes);
            if (result != VK_SUCCESS) {
                return reportVkFailure(
                    "vmaFlushAllocation(allocator, panel.staging_alloc, 0, bytes)",
                    result,
                    std::format("Split-view panel staging flush failed (side={}, allocator={:#x}, allocation={:#x}, offset=0, flush_size={}, capacity={})",
                                side,
                                reinterpret_cast<std::uintptr_t>(allocator),
                                reinterpret_cast<std::uintptr_t>(panel.staging_alloc),
                                bytes,
                                panel.staging_capacity));
            }

            // Wait for any prior submit on this command buffer before re-recording. The fence is
            // created signaled, so the first upload does not block.
            {
                lfs::rendering::WaitContext wait_ctx;
                wait_ctx.fingerprint = "pass.split_view.transfer_prewait";
                auto wait_outcome = lfs::rendering::wait_fence_bounded(
                    device,
                    panel.fence,
                    std::stop_token{},
                    lfs::rendering::VulkanWaitPolicy{},
                    wait_ctx);
                if (!wait_outcome.has_value() ||
                    *wait_outcome != lfs::rendering::WaitOutcome::Ready) {
                    return logVkFailure(std::format(
                        "wait_fence_bounded(pass.split_view.transfer_prewait) did not reach Ready "
                        "(side={}, device={:#x}, fence={:#x}, command_buffer={:#x}): {}",
                        side,
                        vkHandleValue(device),
                        vkHandleValue(panel.fence),
                        vkHandleValue(panel.cmd),
                        formatWaitFailure(wait_outcome)));
                }
            }
            result = vkResetFences(device, 1, &panel.fence);
            if (result != VK_SUCCESS) {
                return replacePanelFenceSignaled(panel, "vkResetFences", result);
            }
            result = vkResetCommandBuffer(panel.cmd, 0);
            if (result != VK_SUCCESS) {
                return replacePanelFenceSignaled(panel, "vkResetCommandBuffer", result);
            }
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            result = vkBeginCommandBuffer(panel.cmd, &bi);
            if (result != VK_SUCCESS) {
                return replacePanelFenceSignaled(panel, "vkBeginCommandBuffer", result);
            }

            // After the first upload the panel image already sits in
            // SHADER_READ_ONLY_OPTIMAL; transition back to TRANSFER_DST.
            const VkImageLayout old_layout = (panel.uploaded_tensor != nullptr)
                                                 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                 : VK_IMAGE_LAYOUT_UNDEFINED;
            const bool was_undefined = old_layout == VK_IMAGE_LAYOUT_UNDEFINED;
            cmdImageBarrier2(panel.cmd, panel.image, VK_IMAGE_ASPECT_COLOR_BIT,
                             old_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             was_undefined ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                                           : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             was_undefined ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_READ_BIT,
                             VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {panel.width, panel.height, 1};
            vkCmdCopyBufferToImage(panel.cmd, panel.staging_buffer, panel.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            cmdImageBarrier2(panel.cmd, panel.image, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

            result = vkEndCommandBuffer(panel.cmd);
            if (result != VK_SUCCESS) {
                return replacePanelFenceSignaled(panel, "vkEndCommandBuffer", result);
            }
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &panel.cmd;
            if (si.commandBufferCount != 1 || si.pCommandBuffers == nullptr ||
                si.pCommandBuffers[0] == VK_NULL_HANDLE || panel.fence == VK_NULL_HANDLE ||
                graphics_queue == VK_NULL_HANDLE) {
                return logVkFailure(std::format(
                    "Split-view panel submit requires a non-null queue, one command buffer, and a non-null fence (side={}, queue={:#x}, command_buffer_count={}, command_buffer_array={:#x}, command_buffer={:#x}, fence={:#x}) ({}:{})",
                    side,
                    vkHandleValue(graphics_queue),
                    si.commandBufferCount,
                    reinterpret_cast<std::uintptr_t>(si.pCommandBuffers),
                    si.pCommandBuffers != nullptr ? vkHandleValue(si.pCommandBuffers[0]) : 0,
                    vkHandleValue(panel.fence),
                    __FILE__,
                    __LINE__));
            }
            result = vkQueueSubmit(graphics_queue, 1, &si, panel.fence);
            if (result != VK_SUCCESS) {
                return replacePanelFenceSignaled(panel, "vkQueueSubmit", result);
            }
            // The viewport pass that consumes this image runs on the same queue right
            // after, so submission order alone makes the upload visible — no fence wait
            // here. We only block on the fence on the NEXT upload to this panel.
            panel.uploaded_tensor = &tensor;
            return true;
        }

        bool rebindDescriptorsIfDirty(FrameDescriptor& descriptor,
                                      const VulkanSplitViewPanel& left_spec,
                                      VkImageView left_view,
                                      const VulkanSplitViewPanel& right_spec,
                                      VkImageView right_view) {
            if (left_view == VK_NULL_HANDLE || right_view == VK_NULL_HANDLE) {
                descriptor.ready = false;
                return false;
            }
            const std::uint64_t left_generation =
                left_spec.external_image_view != VK_NULL_HANDLE ? left_spec.external_image_generation : 0;
            const std::uint64_t right_generation =
                right_spec.external_image_view != VK_NULL_HANDLE ? right_spec.external_image_generation : 0;
            const bool changed =
                left_view != descriptor.left_view ||
                right_view != descriptor.right_view ||
                left_generation != descriptor.left_generation ||
                right_generation != descriptor.right_generation;
            if (!changed) {
                descriptor.ready = true;
                return true;
            }
            std::array<VkDescriptorImageInfo, 2> infos{};
            infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            infos[0].imageView = left_view;
            infos[0].sampler = sampler;
            infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            infos[1].imageView = right_view;
            infos[1].sampler = sampler;
            std::array<VkWriteDescriptorSet, 2> writes{};
            for (std::uint32_t i = 0; i < 2; ++i) {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = descriptor.set;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[i].pImageInfo = &infos[i];
            }
            vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
            descriptor.left_view = left_view;
            descriptor.left_generation = left_generation;
            descriptor.right_view = right_view;
            descriptor.right_generation = right_generation;
            descriptor.ready = true;
            return true;
        }

        // Returns the view to bind for one panel: the externally-supplied interop view
        // when present, otherwise the staging-uploaded view (filling it in if needed).
        VkImageView resolvePanelView(PanelImage& panel, const VulkanSplitViewPanel& spec) {
            if (spec.external_image_view != VK_NULL_HANDLE) {
                return spec.external_image_view;
            }
            if (spec.image && spec.image.get() != panel.uploaded_tensor) {
                if (!uploadPanel(panel, *spec.image)) {
                    return VK_NULL_HANDLE;
                }
            }
            return panel.view;
        }

        void prepare(const VulkanSplitViewParams& params, const std::size_t frame_slot) {
            auto& descriptor = descriptorForFrame(frame_slot);
            descriptor.ready = false;
            if (!params.enabled) {
                return;
            }
            const VkImageView left_view = resolvePanelView(left, params.left);
            const VkImageView right_view = resolvePanelView(right, params.right);
            rebindDescriptorsIfDirty(descriptor,
                                     params.left,
                                     left_view,
                                     params.right,
                                     right_view);
        }

        void record(VkCommandBuffer cb, const VkRect2D& panel_rect,
                    const VulkanSplitViewParams& params, const std::size_t frame_slot) {
            const auto& descriptor = descriptorForFrame(frame_slot);
            if (cb == VK_NULL_HANDLE || descriptor.set == VK_NULL_HANDLE) [[unlikely]] {
                throw std::logic_error(std::format(
                    "Split-view recording requires a command buffer and per-frame descriptor set (command_buffer={:#x}, frame_slot={}, ring_size={}, descriptor_set={:#x}, left_view={:#x}, right_view={:#x}, descriptor_ready={}) ({}:{})",
                    vkHandleValue(cb),
                    frame_slot,
                    frame_descriptors.size(),
                    vkHandleValue(descriptor.set),
                    vkHandleValue(descriptor.left_view),
                    vkHandleValue(descriptor.right_view),
                    descriptor.ready,
                    __FILE__,
                    __LINE__));
            }
            if (!ready(frame_slot) || !params.enabled || screen_quad_buffer == VK_NULL_HANDLE) {
                return;
            }

            VkViewport vp{};
            vp.x = static_cast<float>(panel_rect.offset.x);
            vp.y = static_cast<float>(panel_rect.offset.y);
            vp.width = static_cast<float>(panel_rect.extent.width);
            vp.height = static_cast<float>(panel_rect.extent.height);
            vp.minDepth = 0.0f;
            vp.maxDepth = 1.0f;
            vkCmdSetViewport(cb, 0, 1, &vp);
            vkCmdSetScissor(cb, 0, 1, &panel_rect);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &descriptor.set, 0, nullptr);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cb, 0, 1, &screen_quad_buffer, &off);

            SplitPush push{};
            push.split[0] = std::clamp(params.split_position, 0.0f, 1.0f);
            push.split[1] = params.left.flip_y ? 1.0f : 0.0f;
            push.split[2] = params.right.flip_y ? 1.0f : 0.0f;
            push.split[3] = 0.0f;

            const float rect_x = static_cast<float>(params.content_rect.x);
            const float rect_y = static_cast<float>(params.content_rect.y);
            const float rect_w = static_cast<float>(std::max(params.content_rect.z, 1));
            const float rect_h = static_cast<float>(std::max(params.content_rect.w, 1));
            push.rect[0] = rect_x;
            push.rect[1] = rect_y;
            push.rect[2] = rect_w;
            push.rect[3] = rect_h;

            push.panel_norm[0] = params.left.start_position;
            push.panel_norm[1] = params.left.end_position;
            push.panel_norm[2] = params.right.start_position;
            push.panel_norm[3] = params.right.end_position;

            push.panel_flags[0] = params.left.normalize_x_to_panel ? 1.0f : 0.0f;
            push.panel_flags[1] = params.right.normalize_x_to_panel ? 1.0f : 0.0f;

            push.background[0] = params.background.r;
            push.background[1] = params.background.g;
            push.background[2] = params.background.b;
            push.background[3] = 1.0f;

            // Mirrors compositeSplitImages constants (kMinBarWidthPx etc).
            push.divider[0] = 4.0f * 0.5f;  // bar half-width
            push.divider[1] = 24.0f * 0.5f; // handle half-width
            push.divider[2] = 80.0f * 0.5f; // handle half-height
            push.divider[3] = 6.0f;         // corner radius

            push.grip[0] = 10.0f;        // grip spacing
            push.grip[1] = 2.0f;         // grip half-width
            push.grip[2] = 12.0f * 0.5f; // grip half-length
            push.grip[3] = 2.0f;         // line count (kGripLineCount)

            vkCmdPushConstants(cb, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(push), &push);
            vkCmdDraw(cb, 6, 1, 0, 0);
        }

        bool ready(const std::size_t frame_slot) const {
            const auto& descriptor = descriptorForFrame(frame_slot);
            return descriptor.ready && pipeline != VK_NULL_HANDLE &&
                   descriptor.left_view != VK_NULL_HANDLE &&
                   descriptor.right_view != VK_NULL_HANDLE;
        }
    };

    VulkanSplitViewPass::VulkanSplitViewPass() = default;
    VulkanSplitViewPass::~VulkanSplitViewPass() = default;
    VulkanSplitViewPass::VulkanSplitViewPass(VulkanSplitViewPass&&) noexcept = default;
    VulkanSplitViewPass& VulkanSplitViewPass::operator=(VulkanSplitViewPass&&) noexcept = default;

    bool VulkanSplitViewPass::init(VulkanContext& context, VkFormat color_format,
                                   VkFormat depth_format, VkBuffer screen_quad_buffer) {
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        return impl_->init(context, color_format, depth_format, screen_quad_buffer);
    }

    void VulkanSplitViewPass::prepare(const VulkanSplitViewParams& params,
                                      const std::size_t frame_slot) {
        if (impl_)
            impl_->prepare(params, frame_slot);
    }

    void VulkanSplitViewPass::record(VkCommandBuffer cb, const VkRect2D& panel_rect,
                                     const VulkanSplitViewParams& params,
                                     const std::size_t frame_slot) {
        if (impl_)
            impl_->record(cb, panel_rect, params, frame_slot);
    }

    void VulkanSplitViewPass::shutdown() {
        if (impl_) {
            impl_->destroy();
            impl_.reset();
        }
    }

    bool VulkanSplitViewPass::ready(const std::size_t frame_slot) const {
        return impl_ && impl_->ready(frame_slot);
    }

} // namespace lfs::vis
