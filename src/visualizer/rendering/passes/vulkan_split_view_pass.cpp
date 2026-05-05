/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_split_view_pass.hpp"

#include "core/logger.hpp"
#include "rendering/image_layout.hpp"
#include "window/vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <vk_mem_alloc.h>

#include "viewport/screen_quad.vert.spv.h"
#include "viewport/split_view.frag.spv.h"

namespace lfs::vis {

    namespace {

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
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
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
            // Last bound view (either our staging-uploaded view or an external interop
            // view supplied via params). Used to detect descriptor-rebind needs.
            VkImageView bound_view = VK_NULL_HANDLE;
            std::uint64_t bound_generation = 0;

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
        };
        PanelImage left{};
        PanelImage right{};
        bool descriptors_dirty = true;

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

        bool createSampler() {
            VkSamplerCreateInfo s{};
            s.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            s.magFilter = VK_FILTER_LINEAR;
            s.minFilter = VK_FILTER_LINEAR;
            s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            s.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            s.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            s.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            return vkCreateSampler(device, &s, nullptr, &sampler) == VK_SUCCESS;
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
            if (vkCreateDescriptorSetLayout(device, &li, nullptr, &desc_layout) != VK_SUCCESS) {
                return false;
            }
            VkDescriptorPoolSize ps{};
            ps.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ps.descriptorCount = 2;
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
            VkShaderModule frag = createShaderModule(device, kSplitViewFragSpv, sizeof(kSplitViewFragSpv));
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

        void destroyPanel(PanelImage& p) {
            // Drain any pending transfer submit so we never destroy device memory
            // that the GPU is still reading from.
            if (p.fence != VK_NULL_HANDLE) {
                vkWaitForFences(device, 1, &p.fence, VK_TRUE,
                                std::numeric_limits<std::uint64_t>::max());
            }
            if (p.view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, p.view, nullptr);
                p.view = VK_NULL_HANDLE;
            }
            if (p.image != VK_NULL_HANDLE) {
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
            p.uploaded_tensor = nullptr;
        }

        bool ensureStaging(PanelImage& p, VkDeviceSize bytes) {
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
            if (vmaCreateBuffer(allocator, &bi, &sa, &p.staging_buffer, &p.staging_alloc, &ai) != VK_SUCCESS) {
                return false;
            }
            p.staging_mapped = ai.pMappedData;
            p.staging_capacity = bytes;
            return true;
        }

        bool ensurePanelCmd(PanelImage& p) {
            if (p.cmd != VK_NULL_HANDLE && p.fence != VK_NULL_HANDLE) {
                return true;
            }
            VkCommandBufferAllocateInfo a{};
            a.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            a.commandPool = transfer_pool;
            a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            a.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(device, &a, &p.cmd) != VK_SUCCESS) {
                return false;
            }
            VkFenceCreateInfo fi{};
            fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            // Created signaled so the first vkWaitForFences before recording is a no-op.
            fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            return vkCreateFence(device, &fi, nullptr, &p.fence) == VK_SUCCESS;
        }

        bool ensurePanelImage(PanelImage& p, std::uint32_t w, std::uint32_t h) {
            if (p.image != VK_NULL_HANDLE && p.width == w && p.height == h) {
                return true;
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
            if (vmaCreateImage(allocator, &img, &ai, &p.image, &p.alloc, nullptr) != VK_SUCCESS) {
                return false;
            }
            VkImageViewCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vi.image = p.image;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = VK_FORMAT_R8G8B8A8_UNORM;
            vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vi.subresourceRange.levelCount = 1;
            vi.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &vi, nullptr, &p.view) != VK_SUCCESS) {
                destroyPanel(p);
                return false;
            }
            p.width = w;
            p.height = h;
            descriptors_dirty = true;
            return true;
        }

        bool uploadPanel(PanelImage& panel, const lfs::core::Tensor& tensor) {
            // Probe size from the tensor; resize the persistent pack buffer only when
            // the tensor's resolution exceeds the current capacity.
            if (!tensor.is_valid() || tensor.ndim() != 3) {
                return false;
            }
            const auto layout = lfs::rendering::detectImageLayout(tensor);
            const int probe_w = static_cast<int>(layout == lfs::rendering::ImageLayout::HWC
                                                     ? tensor.size(1)
                                                     : tensor.size(2));
            const int probe_h = static_cast<int>(layout == lfs::rendering::ImageLayout::HWC
                                                     ? tensor.size(0)
                                                     : tensor.size(1));
            if (probe_w <= 0 || probe_h <= 0) {
                return false;
            }
            const std::size_t need_pack = static_cast<std::size_t>(probe_w) * probe_h * 4u;
            if (panel.pack_bytes.size() < need_pack) {
                panel.pack_bytes.resize(need_pack);
            }

            std::uint32_t pkt_w = 0;
            std::uint32_t pkt_h = 0;
            if (!packPanelToRgba8(tensor, panel.pack_bytes.data(), pkt_w, pkt_h)) {
                return false;
            }
            if (!ensurePanelImage(panel, pkt_w, pkt_h)) {
                return false;
            }

            const VkDeviceSize bytes =
                static_cast<VkDeviceSize>(pkt_w) * pkt_h * 4u;
            if (!ensureStaging(panel, bytes) || !ensurePanelCmd(panel)) {
                return false;
            }
            std::memcpy(panel.staging_mapped, panel.pack_bytes.data(), static_cast<std::size_t>(bytes));
            vmaFlushAllocation(allocator, panel.staging_alloc, 0, bytes);

            // Wait for any prior submit on this command buffer before re-recording.
            // First-frame the fence is in unsignaled state and vkWaitForFences with
            // timeout=0 returns VK_TIMEOUT; the reset below makes the next submit valid.
            vkWaitForFences(device, 1, &panel.fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
            vkResetFences(device, 1, &panel.fence);
            if (vkResetCommandBuffer(panel.cmd, 0) != VK_SUCCESS) {
                return false;
            }
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(panel.cmd, &bi) != VK_SUCCESS) {
                return false;
            }

            // After the first upload the panel image already sits in
            // SHADER_READ_ONLY_OPTIMAL; transition back to TRANSFER_DST.
            const VkImageLayout old_layout = (panel.uploaded_tensor != nullptr)
                                                 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                 : VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageMemoryBarrier to_dst{};
            to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_dst.oldLayout = old_layout;
            to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_dst.image = panel.image;
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
            vkCmdPipelineBarrier(panel.cmd, src_stage, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_dst);

            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {panel.width, panel.height, 1};
            vkCmdCopyBufferToImage(panel.cmd, panel.staging_buffer, panel.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            VkImageMemoryBarrier to_read{};
            to_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_read.image = panel.image;
            to_read.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            to_read.subresourceRange.levelCount = 1;
            to_read.subresourceRange.layerCount = 1;
            to_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(panel.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_read);

            if (vkEndCommandBuffer(panel.cmd) != VK_SUCCESS) {
                return false;
            }
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &panel.cmd;
            if (vkQueueSubmit(graphics_queue, 1, &si, panel.fence) != VK_SUCCESS) {
                return false;
            }
            // The viewport pass that consumes this image runs on the same queue right
            // after, so submission order alone makes the upload visible — no fence wait
            // here. We only block on the fence on the NEXT upload to this panel.
            panel.uploaded_tensor = &tensor;
            return true;
        }

        void rebindDescriptorsIfDirty(const VulkanSplitViewPanel& left_spec,
                                      VkImageView left_view,
                                      const VulkanSplitViewPanel& right_spec,
                                      VkImageView right_view) {
            if (left_view == VK_NULL_HANDLE || right_view == VK_NULL_HANDLE) {
                return;
            }
            const std::uint64_t left_generation =
                left_spec.external_image_view != VK_NULL_HANDLE ? left_spec.external_image_generation : 0;
            const std::uint64_t right_generation =
                right_spec.external_image_view != VK_NULL_HANDLE ? right_spec.external_image_generation : 0;
            const bool changed =
                left_view != left.bound_view ||
                right_view != right.bound_view ||
                left_generation != left.bound_generation ||
                right_generation != right.bound_generation;
            if (!descriptors_dirty && !changed) {
                return;
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
                writes[i].dstSet = desc_set;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[i].pImageInfo = &infos[i];
            }
            vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
            descriptors_dirty = false;
            left.bound_view = left_view;
            left.bound_generation = left_generation;
            right.bound_view = right_view;
            right.bound_generation = right_generation;
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
                descriptors_dirty = true;
            }
            return panel.view;
        }

        void prepare(const VulkanSplitViewParams& params) {
            if (!params.enabled) {
                return;
            }
            const VkImageView left_view = resolvePanelView(left, params.left);
            const VkImageView right_view = resolvePanelView(right, params.right);
            rebindDescriptorsIfDirty(params.left, left_view, params.right, right_view);
        }

        void record(VkCommandBuffer cb, const VkRect2D& panel_rect, const VulkanSplitViewParams& params) {
            if (!ready() || !params.enabled || screen_quad_buffer == VK_NULL_HANDLE) {
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
                                    0, 1, &desc_set, 0, nullptr);
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

        bool ready() const {
            return pipeline != VK_NULL_HANDLE && left.bound_view != VK_NULL_HANDLE &&
                   right.bound_view != VK_NULL_HANDLE;
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

    void VulkanSplitViewPass::prepare(const VulkanSplitViewParams& params) {
        if (impl_)
            impl_->prepare(params);
    }

    void VulkanSplitViewPass::record(VkCommandBuffer cb, const VkRect2D& panel_rect,
                                     const VulkanSplitViewParams& params) {
        if (impl_)
            impl_->record(cb, panel_rect, params);
    }

    void VulkanSplitViewPass::shutdown() {
        if (impl_) {
            impl_->destroy();
            impl_.reset();
        }
    }

    bool VulkanSplitViewPass::ready() const {
        return impl_ && impl_->ready();
    }

} // namespace lfs::vis
