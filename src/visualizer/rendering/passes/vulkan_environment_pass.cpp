/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_environment_pass.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "internal/resource_paths.hpp"
#include "window/vulkan_context.hpp"

#include <OpenImageIO/imageio.h>
#include <array>
#include <cstring>
#include <limits>
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
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkCommandPool transfer_pool = VK_NULL_HANDLE;

        VkImage image = VK_NULL_HANDLE;
        VmaAllocation image_alloc = VK_NULL_HANDLE;
        VkImageView image_view = VK_NULL_HANDLE;
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
                return false;
            }

            VkCommandPoolCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pi.queueFamilyIndex = ctx.graphicsQueueFamily();
            if (vkCreateCommandPool(device, &pi, nullptr, &transfer_pool) != VK_SUCCESS) {
                return false;
            }

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
            device = VK_NULL_HANDLE;
            allocator = VK_NULL_HANDLE;
        }

        void destroyImage() {
            if (image_view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, image_view, nullptr);
                image_view = VK_NULL_HANDLE;
            }
            if (image != VK_NULL_HANDLE) {
                vmaDestroyImage(allocator, image, image_alloc);
                image = VK_NULL_HANDLE;
                image_alloc = VK_NULL_HANDLE;
            }
            loaded_path.clear();
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
            return vkCreateSampler(device, &info, nullptr, &sampler) == VK_SUCCESS;
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
            VkDescriptorPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pci.maxSets = 1;
            pci.poolSizeCount = 1;
            pci.pPoolSizes = &ps;
            if (vkCreateDescriptorPool(device, &pci, nullptr, &desc_pool) != VK_SUCCESS) {
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
            VkShaderModule frag = createShaderModule(device, kEnvironmentFragSpv, sizeof(kEnvironmentFragSpv));
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
            if (vkCreatePipelineLayout(device, &li, nullptr, &pipeline_layout) != VK_SUCCESS) {
                vkDestroyShaderModule(device, vert, nullptr);
                vkDestroyShaderModule(device, frag, nullptr);
                return false;
            }

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
            return r == VK_SUCCESS;
        }

        VkCommandBuffer beginCmds() const {
            VkCommandBufferAllocateInfo a{};
            a.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            a.commandPool = transfer_pool;
            a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            a.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(device, &a, &cb) != VK_SUCCESS)
                return VK_NULL_HANDLE;
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) {
                vkFreeCommandBuffers(device, transfer_pool, 1, &cb);
                return VK_NULL_HANDLE;
            }
            return cb;
        }

        bool endCmds(VkCommandBuffer cb) const {
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

        bool loadFromPath(const std::filesystem::path& path) {
            destroyImage();
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
            if (vmaCreateImage(allocator, &img, &ai, &image, &image_alloc, nullptr) != VK_SUCCESS) {
                return false;
            }

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
            if (vmaCreateBuffer(allocator, &bi, &sa, &staging, &staging_alloc, nullptr) != VK_SUCCESS) {
                destroyImage();
                return false;
            }
            void* mapped = nullptr;
            if (vmaMapMemory(allocator, staging_alloc, &mapped) != VK_SUCCESS) {
                vmaDestroyBuffer(allocator, staging, staging_alloc);
                destroyImage();
                return false;
            }
            std::memcpy(mapped, rgba.data(), static_cast<std::size_t>(bytes));
            vmaFlushAllocation(allocator, staging_alloc, 0, bytes);
            vmaUnmapMemory(allocator, staging_alloc);

            VkCommandBuffer cb = beginCmds();
            if (cb == VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, staging, staging_alloc);
                destroyImage();
                return false;
            }

            VkImageMemoryBarrier to_dst{};
            to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_dst.image = image;
            to_dst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            to_dst.subresourceRange.levelCount = 1;
            to_dst.subresourceRange.layerCount = 1;
            to_dst.srcAccessMask = 0;
            to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_dst);

            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), 1};
            vkCmdCopyBufferToImage(cb, staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            VkImageMemoryBarrier to_shader{};
            to_shader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_shader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            to_shader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_shader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_shader.image = image;
            to_shader.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            to_shader.subresourceRange.levelCount = 1;
            to_shader.subresourceRange.layerCount = 1;
            to_shader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_shader);

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
            if (vkCreateImageView(device, &iv, nullptr, &image_view) != VK_SUCCESS) {
                destroyImage();
                return false;
            }

            VkDescriptorImageInfo dii{};
            dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            dii.imageView = image_view;
            dii.sampler = sampler;
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = desc_set;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &dii;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

            loaded_path = path;
            return true;
        }

        void prepare(const VulkanEnvironmentParams& params) {
            if (!params.enabled) {
                if (image != VK_NULL_HANDLE)
                    destroyImage();
                load_failed_for_path = false;
                return;
            }
            if (params.map_path == loaded_path && image != VK_NULL_HANDLE) {
                return;
            }
            if (params.map_path.empty()) {
                if (image != VK_NULL_HANDLE)
                    destroyImage();
                return;
            }
            // Skip retry of a path we've already failed once for, until it changes.
            if (load_failed_for_path && params.map_path == loaded_path) {
                return;
            }
            const bool ok = loadFromPath(params.map_path);
            load_failed_for_path = !ok;
            loaded_path = params.map_path;
        }

        void record(VkCommandBuffer cb, VkExtent2D extent, const VulkanEnvironmentParams& params) {
            if (!params.enabled || pipeline == VK_NULL_HANDLE || image_view == VK_NULL_HANDLE ||
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

    void VulkanEnvironmentPass::prepare(const VulkanEnvironmentParams& params) {
        if (impl_)
            impl_->prepare(params);
    }

    void VulkanEnvironmentPass::record(VkCommandBuffer cb, VkExtent2D extent,
                                       const VulkanEnvironmentParams& params) {
        if (impl_)
            impl_->record(cb, extent, params);
    }

    void VulkanEnvironmentPass::shutdown() {
        if (impl_) {
            impl_->destroy();
            impl_.reset();
        }
    }

    bool VulkanEnvironmentPass::hasTexture() const {
        return impl_ && impl_->image != VK_NULL_HANDLE;
    }

} // namespace lfs::vis
