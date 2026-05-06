/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "point_cloud_vulkan_renderer.hpp"

#ifdef LFS_VULKAN_VIEWER_ENABLED

#include "core/logger.hpp"

#include <array>
#include <cstring>
#include <format>
#include <glm/gtc/type_ptr.hpp>
#include <utility>
#include <vk_mem_alloc.h>

#include "viewport/point_cloud.frag.spv.h"
#include "viewport/point_cloud.vert.spv.h"

namespace lfs::vis {

    namespace {

        constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
        constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
        constexpr std::size_t kSlotCount = 3;
        constexpr std::size_t kPlaceholderSize = 16;

        // Push constants exactly mirror the layout in point_cloud.vert.
        // Packed to 256 bytes (the Vulkan portable upper bound). Counts/flags
        // are bitpacked into a single ivec4 to save 16 bytes vs. an int-per-flag
        // layout that would push us to 272 bytes.
        enum FlagBits : int {
            kFlagHasCrop = 1 << 0,
            kFlagCropInverse = 1 << 1,
            kFlagCropDesaturate = 1 << 2,
            kFlagOrthographic = 1 << 3,
            kFlagHasIndices = 1 << 4,
        };
        struct PushConstants {
            float view_proj[16];        // 64
            float view[16];             // 64
            float crop_to_local[16];    // 64
            float crop_min[4];          // 16
            float crop_max[4];          // 16
            float voxel_focal_ortho[4]; // 16
            int counts[4]{};            // 16: x=n_transforms, y=n_visibility, z=flags, w=max_point_size
        };
        static_assert(sizeof(PushConstants) == 256,
                      "Push constants must fit Vulkan's portable upper bound");

        struct ManagedBuffer {
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkDeviceSize size = 0;
            VkBufferUsageFlags usage = 0;
        };

        void destroyBuffer(VmaAllocator allocator, ManagedBuffer& buf) {
            if (buf.buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, buf.buffer, buf.allocation);
            }
            buf = {};
        }

        bool createDeviceBuffer(VmaAllocator allocator, VkDeviceSize size,
                                VkBufferUsageFlags usage, ManagedBuffer& out) {
            destroyBuffer(allocator, out);
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = std::max<VkDeviceSize>(size, kPlaceholderSize);
            bi.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo ai{};
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

            const VkResult r = vmaCreateBuffer(allocator, &bi, &ai,
                                               &out.buffer, &out.allocation, nullptr);
            if (r != VK_SUCCESS) {
                out = {};
                return false;
            }
            out.size = bi.size;
            out.usage = bi.usage;
            return true;
        }

        bool stageBufferUpload(VmaAllocator allocator,
                               VkCommandBuffer cb,
                               const void* src,
                               VkDeviceSize bytes,
                               ManagedBuffer& dst,
                               ManagedBuffer& staging_scratch) {
            destroyBuffer(allocator, staging_scratch);
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = bytes;
            bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo ai{};
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo info{};
            if (vmaCreateBuffer(allocator, &bi, &ai, &staging_scratch.buffer,
                                &staging_scratch.allocation, &info) != VK_SUCCESS) {
                staging_scratch = {};
                return false;
            }
            staging_scratch.size = bytes;
            staging_scratch.usage = bi.usage;
            std::memcpy(info.pMappedData, src, bytes);
            vmaFlushAllocation(allocator, staging_scratch.allocation, 0, bytes);

            VkBufferCopy region{};
            region.size = bytes;
            vkCmdCopyBuffer(cb, staging_scratch.buffer, dst.buffer, 1, &region);
            return true;
        }

        VkShaderModule createShaderModule(VkDevice device, const std::uint32_t* code,
                                          std::size_t bytes) {
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

        void writePushConstants(PushConstants& pc, const PointCloudVulkanRenderer::RenderRequest& req,
                                int max_point_size, int n_transforms, int n_visibility,
                                bool has_transform_indices) {
            std::memcpy(pc.view_proj, glm::value_ptr(req.view_projection), sizeof(pc.view_proj));
            std::memcpy(pc.view, glm::value_ptr(req.view), sizeof(pc.view));

            const glm::mat4 crop_to_local = req.crop ? req.crop->to_local : glm::mat4(1.0f);
            std::memcpy(pc.crop_to_local, glm::value_ptr(crop_to_local), sizeof(pc.crop_to_local));

            const glm::vec3 crop_min = req.crop ? req.crop->min : glm::vec3(0.0f);
            const glm::vec3 crop_max = req.crop ? req.crop->max : glm::vec3(0.0f);
            pc.crop_min[0] = crop_min.x;
            pc.crop_min[1] = crop_min.y;
            pc.crop_min[2] = crop_min.z;
            pc.crop_min[3] = 0.0f;
            pc.crop_max[0] = crop_max.x;
            pc.crop_max[1] = crop_max.y;
            pc.crop_max[2] = crop_max.z;
            pc.crop_max[3] = 0.0f;

            const float voxel = req.voxel_size * req.scaling_modifier;
            const float ortho_pixels_per_world = req.ortho_scale > 1e-5f
                                                     ? static_cast<float>(req.size.y) / req.ortho_scale
                                                     : static_cast<float>(req.size.y);
            pc.voxel_focal_ortho[0] = voxel;
            pc.voxel_focal_ortho[1] = req.focal_y;
            pc.voxel_focal_ortho[2] = ortho_pixels_per_world;
            pc.voxel_focal_ortho[3] = 0.0f;

            int flags = 0;
            if (req.crop) {
                flags |= kFlagHasCrop;
                if (req.crop->inverse)
                    flags |= kFlagCropInverse;
                if (req.crop->desaturate)
                    flags |= kFlagCropDesaturate;
            }
            if (req.orthographic) {
                flags |= kFlagOrthographic;
            }
            if (has_transform_indices) {
                flags |= kFlagHasIndices;
            }
            pc.counts[0] = n_transforms;
            pc.counts[1] = n_visibility;
            pc.counts[2] = flags;
            pc.counts[3] = max_point_size;
        }

        // Copy a [N, 3] float Tensor (CPU or CUDA) to a contiguous host buffer.
        bool tensorToHost(const lfs::core::Tensor& t, std::vector<float>& out) {
            if (!t.is_valid() || t.ndim() != 2 || t.size(1) != 3) {
                return false;
            }
            const auto host = t.to(lfs::core::Device::CPU).contiguous();
            const std::size_t n = static_cast<std::size_t>(host.size(0)) * 3;
            out.resize(n);
            std::memcpy(out.data(), host.ptr<float>(), n * sizeof(float));
            return true;
        }

    } // namespace

    struct PointCloudVulkanRenderer::Impl {
        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;

        // Pipeline
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        // Descriptor pool / set (transforms + indices + visibility)
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

        // Cached input buffers — point clouds rarely change, so we re-upload only
        // when the cache key (tensor pointer / size) changes.
        struct InputCache {
            ManagedBuffer positions;
            ManagedBuffer colors;
            ManagedBuffer transforms;
            ManagedBuffer transform_indices;
            ManagedBuffer visibility;

            const void* cached_positions_ptr = nullptr;
            std::size_t cached_positions_count = 0;
            const void* cached_colors_ptr = nullptr;
            std::size_t cached_colors_count = 0;
            std::size_t cached_n_transforms = 0;
            std::size_t cached_n_visibility = 0;
            const void* cached_transform_indices_ptr = nullptr;
            std::size_t cached_transform_indices_count = 0;
            std::vector<glm::mat4> cached_transforms;
            std::vector<std::uint32_t> cached_visibility;
        };
        InputCache cache;

        // Placeholder bound to descriptors when an SSBO is unused — avoids
        // descriptor-indexing churn while keeping the bindings stable.
        ManagedBuffer placeholder;

        // One-time staging buffers, kept around so we don't reallocate per upload.
        std::vector<ManagedBuffer> pending_stagings;

        // Output slots: each owns its own color + depth attachment.
        struct OutputSlotResources {
            VkImage color_image = VK_NULL_HANDLE;
            VmaAllocation color_alloc = VK_NULL_HANDLE;
            VkImageView color_view = VK_NULL_HANDLE;

            VkImage depth_image = VK_NULL_HANDLE;
            VmaAllocation depth_alloc = VK_NULL_HANDLE;
            VkImageView depth_view = VK_NULL_HANDLE;

            glm::ivec2 size{0, 0};
            VkImageLayout color_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageLayout depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::uint64_t generation = 0;
        };
        std::array<OutputSlotResources, kSlotCount> slots{};

        // Transient command pool / fence reused across frames.
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;

        bool initialized = false;

        ~Impl() { destroy(); }

        std::string vkError(const char* what, VkResult r) {
            return std::format("{} failed (VkResult={})", what, static_cast<int>(r));
        }

        // Allocate a fresh device-local buffer and stage `bytes` from `src` into
        // it. The transient host staging buffer is parked in pending_stagings
        // and released after the next fence-wait.
        std::expected<void, std::string> uploadInto(VkCommandBuffer cb,
                                                    VkBufferUsageFlags usage,
                                                    const void* src,
                                                    VkDeviceSize bytes,
                                                    ManagedBuffer& dst,
                                                    const char* what) {
            if (!createDeviceBuffer(allocator, bytes, usage, dst)) {
                return std::unexpected<std::string>(std::format("Failed to allocate {} buffer", what));
            }
            ManagedBuffer staging{};
            if (!stageBufferUpload(allocator, cb, src, bytes, dst, staging)) {
                destroyBuffer(allocator, staging);
                return std::unexpected<std::string>(std::format("Failed to upload {}", what));
            }
            pending_stagings.push_back(std::move(staging));
            return {};
        }

        std::expected<void, std::string> ensureInitialized(VulkanContext& ctx) {
            if (initialized) {
                return {};
            }
            context = &ctx;
            device = ctx.device();
            allocator = ctx.allocator();
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE) {
                return std::unexpected<std::string>("Vulkan context not initialized");
            }

            if (auto r = createPipeline(); !r) {
                return r;
            }
            if (auto r = createDescriptorPool(); !r) {
                return r;
            }
            if (auto r = createCommandResources(); !r) {
                return r;
            }
            if (!createDeviceBuffer(allocator, kPlaceholderSize,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, placeholder)) {
                return std::unexpected<std::string>("Failed to create placeholder buffer");
            }
            initialized = true;
            return {};
        }

        std::expected<void, std::string> createPipeline() {
            using namespace viewport_shaders;
            VkShaderModule vert = createShaderModule(device, kPointCloudVertSpv,
                                                     sizeof(kPointCloudVertSpv));
            VkShaderModule frag = createShaderModule(device, kPointCloudFragSpv,
                                                     sizeof(kPointCloudFragSpv));
            if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
                if (vert)
                    vkDestroyShaderModule(device, vert, nullptr);
                if (frag)
                    vkDestroyShaderModule(device, frag, nullptr);
                return std::unexpected<std::string>("vkCreateShaderModule(point_cloud) failed");
            }

            std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
            for (std::uint32_t i = 0; i < bindings.size(); ++i) {
                bindings[i].binding = i;
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            }
            VkDescriptorSetLayoutCreateInfo dl{};
            dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dl.bindingCount = static_cast<std::uint32_t>(bindings.size());
            dl.pBindings = bindings.data();
            VkResult r = vkCreateDescriptorSetLayout(device, &dl, nullptr, &descriptor_set_layout);
            if (r != VK_SUCCESS) {
                vkDestroyShaderModule(device, vert, nullptr);
                vkDestroyShaderModule(device, frag, nullptr);
                return std::unexpected<std::string>(vkError("vkCreateDescriptorSetLayout", r));
            }

            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            push.size = sizeof(PushConstants);

            VkPipelineLayoutCreateInfo pli{};
            pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pli.setLayoutCount = 1;
            pli.pSetLayouts = &descriptor_set_layout;
            pli.pushConstantRangeCount = 1;
            pli.pPushConstantRanges = &push;
            r = vkCreatePipelineLayout(device, &pli, nullptr, &pipeline_layout);
            if (r != VK_SUCCESS) {
                vkDestroyShaderModule(device, vert, nullptr);
                vkDestroyShaderModule(device, frag, nullptr);
                return std::unexpected<std::string>(vkError("vkCreatePipelineLayout", r));
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

            std::array<VkVertexInputBindingDescription, 2> input_bindings{};
            input_bindings[0].binding = 0;
            input_bindings[0].stride = sizeof(float) * 3;
            input_bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            input_bindings[1].binding = 1;
            input_bindings[1].stride = sizeof(float) * 3;
            input_bindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            std::array<VkVertexInputAttributeDescription, 2> attrs{};
            attrs[0].location = 0;
            attrs[0].binding = 0;
            attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
            attrs[0].offset = 0;
            attrs[1].location = 1;
            attrs[1].binding = 1;
            attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
            attrs[1].offset = 0;

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vi.vertexBindingDescriptionCount = static_cast<std::uint32_t>(input_bindings.size());
            vi.pVertexBindingDescriptions = input_bindings.data();
            vi.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attrs.size());
            vi.pVertexAttributeDescriptions = attrs.data();

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

            VkPipelineViewportStateCreateInfo vps{};
            vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vps.viewportCount = 1;
            vps.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = VK_CULL_MODE_NONE;
            rs.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable = VK_TRUE;
            ds.depthWriteEnable = VK_TRUE;
            ds.depthCompareOp = VK_COMPARE_OP_LESS;

            VkPipelineColorBlendAttachmentState ba{};
            ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo bs{};
            bs.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            bs.attachmentCount = 1;
            bs.pAttachments = &ba;

            std::array<VkDynamicState, 2> dyn{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamic{};
            dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamic.dynamicStateCount = static_cast<std::uint32_t>(dyn.size());
            dynamic.pDynamicStates = dyn.data();

            VkFormat color_format = kColorFormat;
            VkPipelineRenderingCreateInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachmentFormats = &color_format;
            rendering.depthAttachmentFormat = kDepthFormat;

            VkGraphicsPipelineCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pi.pNext = &rendering;
            pi.stageCount = static_cast<std::uint32_t>(stages.size());
            pi.pStages = stages.data();
            pi.pVertexInputState = &vi;
            pi.pInputAssemblyState = &ia;
            pi.pViewportState = &vps;
            pi.pRasterizationState = &rs;
            pi.pMultisampleState = &ms;
            pi.pDepthStencilState = &ds;
            pi.pColorBlendState = &bs;
            pi.pDynamicState = &dynamic;
            pi.layout = pipeline_layout;

            r = vkCreateGraphicsPipelines(device, context->pipelineCache(), 1, &pi,
                                          nullptr, &pipeline);
            vkDestroyShaderModule(device, vert, nullptr);
            vkDestroyShaderModule(device, frag, nullptr);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vkCreateGraphicsPipelines", r));
            }
            return {};
        }

        std::expected<void, std::string> createDescriptorPool() {
            VkDescriptorPoolSize ps{};
            ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ps.descriptorCount = 3;
            VkDescriptorPoolCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pi.maxSets = 1;
            pi.poolSizeCount = 1;
            pi.pPoolSizes = &ps;
            VkResult r = vkCreateDescriptorPool(device, &pi, nullptr, &descriptor_pool);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vkCreateDescriptorPool", r));
            }
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = descriptor_pool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &descriptor_set_layout;
            r = vkAllocateDescriptorSets(device, &ai, &descriptor_set);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vkAllocateDescriptorSets", r));
            }
            return {};
        }

        std::expected<void, std::string> createCommandResources() {
            VkCommandPoolCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pi.queueFamilyIndex = context->graphicsQueueFamily();
            VkResult r = vkCreateCommandPool(device, &pi, nullptr, &command_pool);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vkCreateCommandPool", r));
            }
            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = command_pool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            r = vkAllocateCommandBuffers(device, &ai, &command_buffer);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vkAllocateCommandBuffers", r));
            }
            VkFenceCreateInfo fi{};
            fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            r = vkCreateFence(device, &fi, nullptr, &fence);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vkCreateFence", r));
            }
            return {};
        }

        std::expected<void, std::string> ensureOutputImages(OutputSlotResources& slot,
                                                            glm::ivec2 size) {
            if (slot.color_image != VK_NULL_HANDLE && slot.depth_image != VK_NULL_HANDLE &&
                slot.size == size) {
                return {};
            }
            destroySlot(slot);

            const VkExtent3D extent{static_cast<std::uint32_t>(size.x),
                                    static_cast<std::uint32_t>(size.y), 1u};

            // Color: R8G8B8A8_UNORM, used as render target + sampled by scene blit.
            VkImageCreateInfo color_info{};
            color_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            color_info.imageType = VK_IMAGE_TYPE_2D;
            color_info.format = kColorFormat;
            color_info.extent = extent;
            color_info.mipLevels = 1;
            color_info.arrayLayers = 1;
            color_info.samples = VK_SAMPLE_COUNT_1_BIT;
            color_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            color_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT |
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            color_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo ai{};
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            VkResult r = vmaCreateImage(allocator, &color_info, &ai, &slot.color_image,
                                        &slot.color_alloc, nullptr);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vmaCreateImage(color)", r));
            }
            VkImageViewCreateInfo cv{};
            cv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            cv.image = slot.color_image;
            cv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            cv.format = kColorFormat;
            cv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            r = vkCreateImageView(device, &cv, nullptr, &slot.color_view);
            if (r != VK_SUCCESS) {
                destroySlot(slot);
                return std::unexpected<std::string>(vkError("vkCreateImageView(color)", r));
            }

            // Depth: D32_SFLOAT, used as depth attachment + sampled by depth-blit.
            VkImageCreateInfo depth_info = color_info;
            depth_info.format = kDepthFormat;
            depth_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT;
            r = vmaCreateImage(allocator, &depth_info, &ai, &slot.depth_image,
                               &slot.depth_alloc, nullptr);
            if (r != VK_SUCCESS) {
                destroySlot(slot);
                return std::unexpected<std::string>(vkError("vmaCreateImage(depth)", r));
            }
            VkImageViewCreateInfo dv{};
            dv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            dv.image = slot.depth_image;
            dv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            dv.format = kDepthFormat;
            dv.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            r = vkCreateImageView(device, &dv, nullptr, &slot.depth_view);
            if (r != VK_SUCCESS) {
                destroySlot(slot);
                return std::unexpected<std::string>(vkError("vkCreateImageView(depth)", r));
            }

            context->imageBarriers().registerImage(slot.color_image,
                                                   VK_IMAGE_ASPECT_COLOR_BIT,
                                                   VK_IMAGE_LAYOUT_UNDEFINED);
            context->imageBarriers().registerImage(slot.depth_image,
                                                   VK_IMAGE_ASPECT_DEPTH_BIT,
                                                   VK_IMAGE_LAYOUT_UNDEFINED);

            slot.size = size;
            slot.color_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            slot.depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            ++slot.generation;
            return {};
        }

        void destroySlot(OutputSlotResources& slot) {
            if (slot.color_image != VK_NULL_HANDLE) {
                context->imageBarriers().forgetImage(slot.color_image);
                if (slot.color_view != VK_NULL_HANDLE) {
                    vkDestroyImageView(device, slot.color_view, nullptr);
                }
                vmaDestroyImage(allocator, slot.color_image, slot.color_alloc);
            }
            if (slot.depth_image != VK_NULL_HANDLE) {
                context->imageBarriers().forgetImage(slot.depth_image);
                if (slot.depth_view != VK_NULL_HANDLE) {
                    vkDestroyImageView(device, slot.depth_view, nullptr);
                }
                vmaDestroyImage(allocator, slot.depth_image, slot.depth_alloc);
            }
            slot = {};
        }

        void destroy() {
            if (device == VK_NULL_HANDLE) {
                return;
            }
            if (fence != VK_NULL_HANDLE) {
                vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
                vkDestroyFence(device, fence, nullptr);
                fence = VK_NULL_HANDLE;
            }
            if (command_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, command_pool, nullptr);
                command_pool = VK_NULL_HANDLE;
                command_buffer = VK_NULL_HANDLE;
            }
            for (auto& s : slots) {
                destroySlot(s);
            }
            destroyBuffer(allocator, cache.positions);
            destroyBuffer(allocator, cache.colors);
            destroyBuffer(allocator, cache.transforms);
            destroyBuffer(allocator, cache.transform_indices);
            destroyBuffer(allocator, cache.visibility);
            destroyBuffer(allocator, placeholder);
            for (auto& s : pending_stagings) {
                destroyBuffer(allocator, s);
            }
            pending_stagings.clear();
            if (descriptor_pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
                descriptor_pool = VK_NULL_HANDLE;
                descriptor_set = VK_NULL_HANDLE;
            }
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, pipeline, nullptr);
                pipeline = VK_NULL_HANDLE;
            }
            if (pipeline_layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
                pipeline_layout = VK_NULL_HANDLE;
            }
            if (descriptor_set_layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
                descriptor_set_layout = VK_NULL_HANDLE;
            }
            initialized = false;
            context = nullptr;
            device = VK_NULL_HANDLE;
            allocator = VK_NULL_HANDLE;
        }

        std::expected<void, std::string> uploadIfChanged(VkCommandBuffer cb,
                                                         const RenderRequest& req) {
            if (!req.positions || !req.colors) {
                return std::unexpected<std::string>("RenderRequest is missing positions/colors");
            }
            if (!req.positions->is_valid() || req.positions->ndim() != 2 ||
                req.positions->size(1) != 3) {
                return std::unexpected<std::string>("positions must be [N, 3] float");
            }
            if (!req.colors->is_valid() || req.colors->ndim() != 2 ||
                req.colors->size(1) != 3 ||
                req.colors->size(0) != req.positions->size(0)) {
                return std::unexpected<std::string>("colors must match positions [N, 3]");
            }

            const std::size_t n_points = static_cast<std::size_t>(req.positions->size(0));

            // positions
            const void* pos_key = req.positions->ptr<float>();
            if (pos_key != cache.cached_positions_ptr || cache.cached_positions_count != n_points ||
                cache.positions.buffer == VK_NULL_HANDLE) {
                std::vector<float> host;
                if (!tensorToHost(*req.positions, host)) {
                    return std::unexpected<std::string>("Failed to read positions to CPU");
                }
                if (auto r = uploadInto(cb, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                        host.data(), host.size() * sizeof(float),
                                        cache.positions, "positions");
                    !r) {
                    return r;
                }
                cache.cached_positions_ptr = pos_key;
                cache.cached_positions_count = n_points;
            }

            // colors (handle uint8 / float alike via Tensor::to)
            const lfs::core::Tensor colors_f32 = (req.colors->dtype() == lfs::core::DataType::Float32)
                                                     ? *req.colors
                                                     : (req.colors->dtype() == lfs::core::DataType::UInt8
                                                            ? req.colors->to(lfs::core::DataType::Float32) / 255.0f
                                                            : req.colors->to(lfs::core::DataType::Float32));
            const void* col_key = colors_f32.ptr<float>();
            if (col_key != cache.cached_colors_ptr || cache.cached_colors_count != n_points ||
                cache.colors.buffer == VK_NULL_HANDLE) {
                std::vector<float> host;
                if (!tensorToHost(colors_f32, host)) {
                    return std::unexpected<std::string>("Failed to read colors to CPU");
                }
                if (auto r = uploadInto(cb, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                        host.data(), host.size() * sizeof(float),
                                        cache.colors, "colors");
                    !r) {
                    return r;
                }
                cache.cached_colors_ptr = col_key;
                cache.cached_colors_count = n_points;
            }

            // model_transforms (CPU vector of mat4)
            const std::vector<glm::mat4> empty_transforms;
            const auto& transforms = req.model_transforms ? *req.model_transforms : empty_transforms;
            const bool transforms_changed =
                cache.cached_n_transforms != transforms.size() ||
                cache.cached_transforms != transforms;
            if (transforms_changed || cache.transforms.buffer == VK_NULL_HANDLE) {
                if (!transforms.empty()) {
                    if (auto r = uploadInto(cb, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            transforms.data(), transforms.size() * sizeof(glm::mat4),
                                            cache.transforms, "transforms");
                        !r) {
                        return r;
                    }
                } else if (cache.transforms.buffer == VK_NULL_HANDLE) {
                    if (!createDeviceBuffer(allocator, kPlaceholderSize,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, cache.transforms)) {
                        return std::unexpected<std::string>("Failed to allocate transforms placeholder");
                    }
                }
                cache.cached_transforms = transforms;
                cache.cached_n_transforms = transforms.size();
            }

            // transform_indices (Tensor)
            const std::size_t expected_indices_count =
                (cache.cached_n_transforms > 0 && req.transform_indices &&
                 req.transform_indices->is_valid() &&
                 req.transform_indices->numel() == n_points)
                    ? n_points
                    : 0;
            if (expected_indices_count > 0) {
                const lfs::core::Tensor indices_i32 =
                    (req.transform_indices->dtype() == lfs::core::DataType::Int32)
                        ? *req.transform_indices
                        : req.transform_indices->to(lfs::core::DataType::Int32);
                const void* idx_key = indices_i32.ptr<int>();
                if (idx_key != cache.cached_transform_indices_ptr ||
                    cache.cached_transform_indices_count != n_points ||
                    cache.transform_indices.buffer == VK_NULL_HANDLE) {
                    const auto host = indices_i32.to(lfs::core::Device::CPU).contiguous();
                    if (auto r = uploadInto(cb, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            host.ptr<int>(), n_points * sizeof(std::int32_t),
                                            cache.transform_indices, "transform indices");
                        !r) {
                        return r;
                    }
                    cache.cached_transform_indices_ptr = idx_key;
                    cache.cached_transform_indices_count = n_points;
                }
            } else {
                if (cache.transform_indices.buffer == VK_NULL_HANDLE) {
                    if (!createDeviceBuffer(allocator, kPlaceholderSize,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            cache.transform_indices)) {
                        return std::unexpected<std::string>("Failed to allocate indices placeholder");
                    }
                }
                cache.cached_transform_indices_ptr = nullptr;
                cache.cached_transform_indices_count = 0;
            }

            // visibility_mask (vector<bool> → uint32 SSBO)
            std::vector<std::uint32_t> vis_host;
            if (req.node_visibility_mask && !req.node_visibility_mask->empty()) {
                vis_host.reserve(req.node_visibility_mask->size());
                for (bool b : *req.node_visibility_mask) {
                    vis_host.push_back(b ? 1u : 0u);
                }
            }
            const bool visibility_changed = vis_host != cache.cached_visibility;
            if (visibility_changed || cache.visibility.buffer == VK_NULL_HANDLE) {
                if (!vis_host.empty()) {
                    if (auto r = uploadInto(cb, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            vis_host.data(), vis_host.size() * sizeof(std::uint32_t),
                                            cache.visibility, "visibility");
                        !r) {
                        return r;
                    }
                } else if (cache.visibility.buffer == VK_NULL_HANDLE) {
                    if (!createDeviceBuffer(allocator, kPlaceholderSize,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, cache.visibility)) {
                        return std::unexpected<std::string>("Failed to allocate visibility placeholder");
                    }
                }
                cache.cached_visibility = std::move(vis_host);
                cache.cached_n_visibility = cache.cached_visibility.size();
            }

            return {};
        }

        void updateDescriptorSet() {
            VkBuffer transforms_buf = (cache.cached_n_transforms > 0)
                                          ? cache.transforms.buffer
                                          : placeholder.buffer;
            VkBuffer indices_buf = (cache.cached_transform_indices_count > 0)
                                       ? cache.transform_indices.buffer
                                       : placeholder.buffer;
            VkBuffer visibility_buf = (cache.cached_n_visibility > 0)
                                          ? cache.visibility.buffer
                                          : placeholder.buffer;

            std::array<VkDescriptorBufferInfo, 3> infos{};
            infos[0].buffer = transforms_buf;
            infos[0].range = VK_WHOLE_SIZE;
            infos[1].buffer = indices_buf;
            infos[1].range = VK_WHOLE_SIZE;
            infos[2].buffer = visibility_buf;
            infos[2].range = VK_WHOLE_SIZE;

            std::array<VkWriteDescriptorSet, 3> writes{};
            for (std::uint32_t i = 0; i < writes.size(); ++i) {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = descriptor_set;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &infos[i];
            }
            vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
        }

        std::expected<RenderResult, std::string> doRender(const RenderRequest& req,
                                                          OutputSlot output_slot) {
            const std::size_t slot_idx = static_cast<std::size_t>(output_slot);
            if (slot_idx >= kSlotCount) {
                return std::unexpected<std::string>("invalid output_slot");
            }
            if (req.size.x <= 0 || req.size.y <= 0) {
                return std::unexpected<std::string>("invalid render size");
            }

            // Wait for the previous frame on this renderer to finish before
            // touching slot resources — the cb is shared across slots and a
            // size change would otherwise destroy images the in-flight submit
            // is still sampling.
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

            auto& slot = slots[slot_idx];
            const bool will_recreate = slot.color_image == VK_NULL_HANDLE ||
                                       slot.depth_image == VK_NULL_HANDLE ||
                                       slot.size != req.size;
            if (will_recreate) {
                // pcFence covers this renderer's CB only; the context's frame CB also
                // samples slot.color_image and must finish before destroySlot frees it.
                if (!context->waitForSubmittedFrames()) {
                    return std::unexpected<std::string>(
                        std::format("waitForSubmittedFrames failed before slot recreate: {}",
                                    context->lastError()));
                }
            }
            if (auto r = ensureOutputImages(slot, req.size); !r) {
                return std::unexpected<std::string>(r.error());
            }
            for (auto& s : pending_stagings) {
                destroyBuffer(allocator, s);
            }
            pending_stagings.clear();
            vkResetCommandBuffer(command_buffer, 0);

            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VkResult r = vkBeginCommandBuffer(command_buffer, &bi);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vkBeginCommandBuffer", r));
            }

            if (auto u = uploadIfChanged(command_buffer, req); !u) {
                vkEndCommandBuffer(command_buffer);
                return std::unexpected<std::string>(u.error());
            }

            // Make the just-uploaded data visible to the vertex stage.
            VkMemoryBarrier xfer_to_vert{};
            xfer_to_vert.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            xfer_to_vert.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            xfer_to_vert.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                                         VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(command_buffer,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                                     VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                                 0, 1, &xfer_to_vert, 0, nullptr, 0, nullptr);

            updateDescriptorSet();

            // Transition output images for rendering.
            context->imageBarriers().transitionImage(command_buffer, slot.color_image,
                                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            context->imageBarriers().transitionImage(command_buffer, slot.depth_image,
                                                     VK_IMAGE_ASPECT_DEPTH_BIT,
                                                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

            VkClearValue color_clear{};
            color_clear.color = {{req.background_color.r, req.background_color.g,
                                  req.background_color.b,
                                  req.transparent_background ? 0.0f : 1.0f}};
            VkClearValue depth_clear{};
            depth_clear.depthStencil = {1.0f, 0};

            VkRenderingAttachmentInfo color_attach{};
            color_attach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            color_attach.imageView = slot.color_view;
            color_attach.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color_attach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            color_attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color_attach.clearValue = color_clear;

            VkRenderingAttachmentInfo depth_attach{};
            depth_attach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attach.imageView = slot.depth_view;
            depth_attach.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depth_attach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_attach.clearValue = depth_clear;

            VkRect2D area{};
            area.extent = {static_cast<std::uint32_t>(req.size.x),
                           static_cast<std::uint32_t>(req.size.y)};

            VkRenderingInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering.renderArea = area;
            rendering.layerCount = 1;
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachments = &color_attach;
            rendering.pDepthAttachment = &depth_attach;

            vkCmdBeginRendering(command_buffer, &rendering);

            VkViewport vp{};
            vp.x = 0.0f;
            vp.y = 0.0f;
            vp.width = static_cast<float>(req.size.x);
            vp.height = static_cast<float>(req.size.y);
            vp.minDepth = 0.0f;
            vp.maxDepth = 1.0f;
            vkCmdSetViewport(command_buffer, 0, 1, &vp);
            vkCmdSetScissor(command_buffer, 0, 1, &area);

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

            VkDeviceSize zero_offsets[2] = {0, 0};
            VkBuffer vbufs[2] = {cache.positions.buffer, cache.colors.buffer};
            vkCmdBindVertexBuffers(command_buffer, 0, 2, vbufs, zero_offsets);

            // 64 is the lower bound of Vulkan's guaranteed pointSizeRange; the
            // shader clamps gl_PointSize to this so very-near points don't
            // exceed device limits.
            constexpr int kMaxPointSize = 64;
            PushConstants push{};
            writePushConstants(push, req,
                               kMaxPointSize,
                               static_cast<int>(cache.cached_n_transforms),
                               static_cast<int>(cache.cached_n_visibility),
                               cache.cached_transform_indices_count > 0);
            vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(push), &push);

            const std::uint32_t vertex_count =
                static_cast<std::uint32_t>(cache.cached_positions_count);
            vkCmdDraw(command_buffer, vertex_count, 1, 0, 0);

            vkCmdEndRendering(command_buffer);

            // Transition both outputs to SHADER_READ_ONLY for downstream sampling.
            context->imageBarriers().transitionImage(command_buffer, slot.color_image,
                                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            context->imageBarriers().transitionImage(command_buffer, slot.depth_image,
                                                     VK_IMAGE_ASPECT_DEPTH_BIT,
                                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            r = vkEndCommandBuffer(command_buffer);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vkEndCommandBuffer", r));
            }

            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &command_buffer;
            r = vkResetFences(device, 1, &fence);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vkResetFences", r));
            }
            r = vkQueueSubmit(context->graphicsQueue(), 1, &si, fence);
            if (r != VK_SUCCESS) {
                return std::unexpected<std::string>(vkError("vkQueueSubmit", r));
            }

            slot.color_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            slot.depth_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ++slot.generation;

            RenderResult result{};
            result.image = slot.color_image;
            result.image_view = slot.color_view;
            result.image_layout = slot.color_layout;
            result.generation = slot.generation;
            result.depth_image = slot.depth_image;
            result.depth_image_view = slot.depth_view;
            result.depth_image_layout = slot.depth_layout;
            result.depth_generation = slot.generation;
            result.size = slot.size;
            result.flip_y = false;
            return result;
        }
    };

    PointCloudVulkanRenderer::PointCloudVulkanRenderer()
        : impl_(std::make_unique<Impl>()) {}

    PointCloudVulkanRenderer::~PointCloudVulkanRenderer() = default;

    std::expected<PointCloudVulkanRenderer::RenderResult, std::string>
    PointCloudVulkanRenderer::render(VulkanContext& context, const RenderRequest& request,
                                     OutputSlot output_slot) {
        if (auto r = impl_->ensureInitialized(context); !r) {
            return std::unexpected<std::string>(r.error());
        }
        return impl_->doRender(request, output_slot);
    }

    void PointCloudVulkanRenderer::reset() {
        impl_->destroy();
    }

} // namespace lfs::vis

#endif // LFS_VULKAN_VIEWER_ENABLED
