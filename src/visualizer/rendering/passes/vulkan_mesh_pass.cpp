/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_mesh_pass.hpp"

#include "core/logger.hpp"
#include "core/material.hpp"
#include "core/mesh_data.hpp"
#include "window/vulkan_context.hpp"

#include <array>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <unordered_map>
#include <vector>
#include <vk_mem_alloc.h>

#include "viewport/mesh.frag.spv.h"
#include "viewport/mesh.vert.spv.h"
#include "viewport/mesh_shadow.frag.spv.h"
#include "viewport/mesh_shadow.vert.spv.h"
#include "viewport/mesh_wireframe.frag.spv.h"
#include "viewport/mesh_wireframe.vert.spv.h"

namespace lfs::vis {

    namespace {

        struct MeshVertex {
            float position[3];
            float normal[3];
            float tangent[4]; // xyz + handedness w
            float texcoord[2];
            float color[4];
        };
        static_assert(sizeof(MeshVertex) == 64, "MeshVertex layout — 16-byte aligned");

        struct MeshPushConstants {
            float mvp[16];
            float model[16];
        };
        static_assert(sizeof(MeshPushConstants) == 128, "Push constants must fit in 128B");

        struct LightUbo {
            float camera_pos[4];
            float light_dir[4]; // xyz, w unused
            float params[4];    // x = intensity, y = ambient, z = shadow_enabled
            float light_vp[16]; // light view-projection (column-major) for shadow sampling
        };
        static_assert(sizeof(LightUbo) == 112, "LightUbo layout");

        struct MaterialUbo {
            float base_color[4];
            float emissive_metallic[4];     // xyz emissive, w metallic
            float roughness_flags[4];       // x roughness, y has_albedo, z has_normal, w has_metallic_roughness
            float vertex_color_emphasis[4]; // x has_vertex_colors, y is_emphasized, z dim_non_emphasized, w flash_intensity
        };
        static_assert(sizeof(MaterialUbo) == 64, "MaterialUbo layout");

        struct ShadowPush {
            float light_mvp[16];
        };
        static_assert(sizeof(ShadowPush) == 64);

        struct WireframePush {
            float mvp[16];
            float color[4];
        };
        static_assert(sizeof(WireframePush) == 80);

        VkShaderModule createShaderModule(VkDevice device, const std::uint32_t* code, std::size_t bytes) {
            VkShaderModuleCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            info.codeSize = bytes;
            info.pCode = code;
            VkShaderModule module = VK_NULL_HANDLE;
            if (vkCreateShaderModule(device, &info, nullptr, &module) != VK_SUCCESS) {
                return VK_NULL_HANDLE;
            }
            return module;
        }

    } // namespace

    struct VulkanMeshPass::Impl {
        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache = VK_NULL_HANDLE;

        VkDescriptorSetLayout light_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout material_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline_cull = VK_NULL_HANDLE;    // backface culling (default)
        VkPipeline pipeline_no_cull = VK_NULL_HANDLE; // double-sided / culling off

        VkPipelineLayout shadow_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline shadow_pipeline = VK_NULL_HANDLE;
        VkFormat shadow_format = VK_FORMAT_UNDEFINED;

        VkPipelineLayout wireframe_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline wireframe_pipeline = VK_NULL_HANDLE;
        VkFormat color_format_cached = VK_FORMAT_UNDEFINED;
        VkFormat depth_format_cached = VK_FORMAT_UNDEFINED;

        VkSampler sampler = VK_NULL_HANDLE;
        VkSampler shadow_sampler = VK_NULL_HANDLE; // sampler2DShadow with comparison
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkCommandPool transfer_pool = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;

        // Per-frame light UBO (pushed once per record()).
        VkBuffer light_buffer = VK_NULL_HANDLE;
        VmaAllocation light_alloc = VK_NULL_HANDLE;
        VkDescriptorSet light_descriptor = VK_NULL_HANDLE;

        // 1x1 white fallback texture for materials missing a given texture.
        struct GpuTexture {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation alloc = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
        };
        GpuTexture white_pixel{};

        struct GpuMaterial {
            VkBuffer ubo = VK_NULL_HANDLE;
            VmaAllocation ubo_alloc = VK_NULL_HANDLE;
            VkDescriptorSet descriptor = VK_NULL_HANDLE;
            GpuTexture albedo{};
            GpuTexture normal{};
            GpuTexture metallic_roughness{};
        };

        struct GpuSubmesh {
            std::uint32_t start_index = 0;
            std::uint32_t index_count = 0;
            std::size_t material_index = 0;
        };

        struct ShadowTarget {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation alloc = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            int resolution = 0;
        };

        struct GpuMesh {
            VkBuffer vertex_buffer = VK_NULL_HANDLE;
            VmaAllocation vertex_alloc = VK_NULL_HANDLE;
            VkBuffer index_buffer = VK_NULL_HANDLE;
            VmaAllocation index_alloc = VK_NULL_HANDLE;
            std::uint32_t total_index_count = 0;
            std::uint32_t generation = 0;
            std::uint64_t last_used_frame = 0;
            std::vector<GpuMaterial> materials;
            std::vector<GpuSubmesh> submeshes;

            glm::vec3 aabb_min{0.0f};
            glm::vec3 aabb_max{0.0f};

            ShadowTarget shadow{};
            glm::mat4 cached_light_vp{1.0f};
            bool cached_light_vp_valid = false;
        };

        // Placeholder 1x1 shadow image bound when shadow_enabled=false; satisfies the
        // sampler2DShadow descriptor without the validation layer complaining.
        ShadowTarget shadow_dummy{};

        std::unordered_map<const lfs::core::MeshData*, GpuMesh> mesh_cache;
        std::uint64_t frame_counter = 0;

        ~Impl() { destroy(); }

        bool init(VulkanContext& ctx, VkFormat color_format, VkFormat depth_format) {
            context = &ctx;
            device = ctx.device();
            allocator = ctx.allocator();
            pipeline_cache = ctx.pipelineCache();
            graphics_queue = ctx.graphicsQueue();
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE ||
                graphics_queue == VK_NULL_HANDLE) {
                LOG_ERROR("VulkanMeshPass: invalid context");
                return false;
            }

            VkCommandPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool_info.queueFamilyIndex = ctx.graphicsQueueFamily();
            if (vkCreateCommandPool(device, &pool_info, nullptr, &transfer_pool) != VK_SUCCESS) {
                LOG_ERROR("VulkanMeshPass: command pool creation failed");
                return false;
            }

            color_format_cached = color_format;
            depth_format_cached = depth_format;
            shadow_format = depth_format; // re-use the swapchain's depth format

            return createSamplers() &&
                   createDescriptorLayouts() &&
                   createDescriptorPool() &&
                   createLightUbo() &&
                   createWhitePixel() &&
                   createDummyShadow() &&
                   createMainPipelines(color_format, depth_format) &&
                   createShadowPipeline(depth_format) &&
                   createWireframePipeline(color_format, depth_format);
        }

        VkCommandBuffer beginSingleTimeCommands() const {
            VkCommandBufferAllocateInfo alloc{};
            alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            alloc.commandPool = transfer_pool;
            alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(device, &alloc, &cb) != VK_SUCCESS) {
                return VK_NULL_HANDLE;
            }
            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(cb, &begin) != VK_SUCCESS) {
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
            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cb;
            VkFenceCreateInfo finfo{};
            finfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VkFence fence = VK_NULL_HANDLE;
            VkResult r = vkCreateFence(device, &finfo, nullptr, &fence);
            if (r == VK_SUCCESS) {
                r = vkQueueSubmit(graphics_queue, 1, &submit, fence);
            }
            if (r == VK_SUCCESS) {
                r = vkWaitForFences(device, 1, &fence, VK_TRUE,
                                    std::numeric_limits<std::uint64_t>::max());
            }
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, fence, nullptr);
            }
            vkFreeCommandBuffers(device, transfer_pool, 1, &cb);
            return r == VK_SUCCESS;
        }

        void destroy() {
            for (auto& [_, gpu] : mesh_cache) {
                destroyMesh(gpu);
            }
            mesh_cache.clear();
            destroyTexture(white_pixel);
            destroyShadow(shadow_dummy);
            if (light_buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, light_buffer, light_alloc);
                light_buffer = VK_NULL_HANDLE;
                light_alloc = VK_NULL_HANDLE;
            }
            if (descriptor_pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
                descriptor_pool = VK_NULL_HANDLE;
                light_descriptor = VK_NULL_HANDLE;
            }
            for (VkPipeline* p : {&pipeline_cull, &pipeline_no_cull, &shadow_pipeline, &wireframe_pipeline}) {
                if (*p != VK_NULL_HANDLE) {
                    vkDestroyPipeline(device, *p, nullptr);
                    *p = VK_NULL_HANDLE;
                }
            }
            for (VkPipelineLayout* l : {&pipeline_layout, &shadow_pipeline_layout, &wireframe_pipeline_layout}) {
                if (*l != VK_NULL_HANDLE) {
                    vkDestroyPipelineLayout(device, *l, nullptr);
                    *l = VK_NULL_HANDLE;
                }
            }
            if (material_layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, material_layout, nullptr);
                material_layout = VK_NULL_HANDLE;
            }
            if (light_layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, light_layout, nullptr);
                light_layout = VK_NULL_HANDLE;
            }
            if (sampler != VK_NULL_HANDLE) {
                vkDestroySampler(device, sampler, nullptr);
                sampler = VK_NULL_HANDLE;
            }
            if (shadow_sampler != VK_NULL_HANDLE) {
                vkDestroySampler(device, shadow_sampler, nullptr);
                shadow_sampler = VK_NULL_HANDLE;
            }
            if (transfer_pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, transfer_pool, nullptr);
                transfer_pool = VK_NULL_HANDLE;
            }
            device = VK_NULL_HANDLE;
            allocator = VK_NULL_HANDLE;
        }

        bool createSamplers() {
            VkSamplerCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            info.magFilter = VK_FILTER_LINEAR;
            info.minFilter = VK_FILTER_LINEAR;
            info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            info.maxLod = VK_LOD_CLAMP_NONE;
            info.anisotropyEnable = VK_FALSE;
            if (vkCreateSampler(device, &info, nullptr, &sampler) != VK_SUCCESS) {
                return false;
            }

            // Shadow comparison sampler — pairs with sampler2DShadow in mesh.frag.
            VkSamplerCreateInfo shadow_info{};
            shadow_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            shadow_info.magFilter = VK_FILTER_LINEAR;
            shadow_info.minFilter = VK_FILTER_LINEAR;
            shadow_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            shadow_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            shadow_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            shadow_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            shadow_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            shadow_info.compareEnable = VK_TRUE;
            shadow_info.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
            return vkCreateSampler(device, &shadow_info, nullptr, &shadow_sampler) == VK_SUCCESS;
        }

        bool createDescriptorLayouts() {
            // Set 0: light UBO + shadow map sampler.
            std::array<VkDescriptorSetLayoutBinding, 2> light_b{};
            light_b[0].binding = 0;
            light_b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            light_b[0].descriptorCount = 1;
            light_b[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            light_b[1].binding = 1;
            light_b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            light_b[1].descriptorCount = 1;
            light_b[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutCreateInfo light_info{};
            light_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            light_info.bindingCount = static_cast<std::uint32_t>(light_b.size());
            light_info.pBindings = light_b.data();
            if (vkCreateDescriptorSetLayout(device, &light_info, nullptr, &light_layout) != VK_SUCCESS) {
                return false;
            }

            // Set 1: material UBO + 3 sampled textures
            std::array<VkDescriptorSetLayoutBinding, 4> mat_b{};
            mat_b[0].binding = 0;
            mat_b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            mat_b[0].descriptorCount = 1;
            mat_b[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            for (int i = 1; i < 4; ++i) {
                mat_b[i].binding = static_cast<std::uint32_t>(i);
                mat_b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                mat_b[i].descriptorCount = 1;
                mat_b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            }
            VkDescriptorSetLayoutCreateInfo mat_info{};
            mat_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            mat_info.bindingCount = static_cast<std::uint32_t>(mat_b.size());
            mat_info.pBindings = mat_b.data();
            return vkCreateDescriptorSetLayout(device, &mat_info, nullptr, &material_layout) == VK_SUCCESS;
        }

        bool createDescriptorPool() {
            // 256 materials should cover practical scenes; recreate the pool if we ever
            // blow past that (rare).
            constexpr std::uint32_t kMaxMaterials = 256;
            std::array<VkDescriptorPoolSize, 2> sizes{};
            sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            sizes[0].descriptorCount = kMaxMaterials + 1;
            sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            // Material textures (3 per material) + 1 shadow map sampler on light set.
            sizes[1].descriptorCount = kMaxMaterials * 3 + 1;
            VkDescriptorPoolCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            info.maxSets = kMaxMaterials + 1;
            info.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
            info.pPoolSizes = sizes.data();
            return vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) == VK_SUCCESS;
        }

        bool createLightUbo() {
            VkBufferCreateInfo b{};
            b.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            b.size = sizeof(LightUbo);
            b.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            b.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo a{};
            a.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            a.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            if (vmaCreateBuffer(allocator, &b, &a, &light_buffer, &light_alloc, nullptr) != VK_SUCCESS) {
                return false;
            }

            VkDescriptorSetAllocateInfo alloc{};
            alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc.descriptorPool = descriptor_pool;
            alloc.descriptorSetCount = 1;
            alloc.pSetLayouts = &light_layout;
            if (vkAllocateDescriptorSets(device, &alloc, &light_descriptor) != VK_SUCCESS) {
                return false;
            }

            // UBO write happens here; shadow-map descriptor (binding 1) is updated each
            // frame in record() so it can swap between the per-mesh shadow image and the
            // dummy fallback.
            VkDescriptorBufferInfo bi{};
            bi.buffer = light_buffer;
            bi.range = sizeof(LightUbo);
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = light_descriptor;
            w.dstBinding = 0;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w.pBufferInfo = &bi;
            vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
            return true;
        }

        bool createDummyShadow() {
            return createShadowTarget(1, shadow_dummy);
        }

        bool createShadowTarget(int resolution, ShadowTarget& out) {
            destroyShadow(out);
            VkImageCreateInfo img{};
            img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            img.imageType = VK_IMAGE_TYPE_2D;
            img.format = shadow_format;
            img.extent = {static_cast<std::uint32_t>(resolution),
                          static_cast<std::uint32_t>(resolution), 1};
            img.mipLevels = 1;
            img.arrayLayers = 1;
            img.samples = VK_SAMPLE_COUNT_1_BIT;
            img.tiling = VK_IMAGE_TILING_OPTIMAL;
            img.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            img.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo a{};
            a.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            if (vmaCreateImage(allocator, &img, &a, &out.image, &out.alloc, nullptr) != VK_SUCCESS) {
                return false;
            }
            VkImageViewCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vi.image = out.image;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = shadow_format;
            vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            vi.subresourceRange.levelCount = 1;
            vi.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &vi, nullptr, &out.view) != VK_SUCCESS) {
                vmaDestroyImage(allocator, out.image, out.alloc);
                out = {};
                return false;
            }
            out.resolution = resolution;

            // Initial transition UNDEFINED → SHADER_READ_ONLY so binding it without a
            // shadow render is valid (e.g. when shadows are disabled).
            VkCommandBuffer cb = beginSingleTimeCommands();
            if (cb != VK_NULL_HANDLE) {
                VkImageMemoryBarrier b{};
                b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                b.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = out.image;
                b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                b.subresourceRange.levelCount = 1;
                b.subresourceRange.layerCount = 1;
                b.srcAccessMask = 0;
                b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                     0, 0, nullptr, 0, nullptr, 1, &b);
                endSingleTimeCommands(cb);
            }
            return true;
        }

        void destroyShadow(ShadowTarget& t) const {
            if (t.view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, t.view, nullptr);
            }
            if (t.image != VK_NULL_HANDLE) {
                vmaDestroyImage(allocator, t.image, t.alloc);
            }
            t = {};
        }

        bool writeBuffer(VmaAllocation alloc, const void* src, std::size_t bytes) const {
            void* mapped = nullptr;
            if (vmaMapMemory(allocator, alloc, &mapped) != VK_SUCCESS || !mapped) {
                return false;
            }
            std::memcpy(mapped, src, bytes);
            vmaFlushAllocation(allocator, alloc, 0, bytes);
            vmaUnmapMemory(allocator, alloc);
            return true;
        }

        bool createTexture(const std::uint8_t* rgba, int w, int h, GpuTexture& out) {
            VkImageCreateInfo img{};
            img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            img.imageType = VK_IMAGE_TYPE_2D;
            img.format = VK_FORMAT_R8G8B8A8_UNORM;
            img.extent = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), 1};
            img.mipLevels = 1;
            img.arrayLayers = 1;
            img.samples = VK_SAMPLE_COUNT_1_BIT;
            img.tiling = VK_IMAGE_TILING_OPTIMAL;
            img.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            img.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo a{};
            a.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            if (vmaCreateImage(allocator, &img, &a, &out.image, &out.alloc, nullptr) != VK_SUCCESS) {
                return false;
            }

            // Stage to upload buffer → copy via transient command.
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * 4u;
            VkBuffer staging = VK_NULL_HANDLE;
            VmaAllocation staging_alloc = VK_NULL_HANDLE;
            VkBufferCreateInfo sb{};
            sb.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            sb.size = bytes;
            sb.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            sb.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo sa{};
            sa.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            sa.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            if (vmaCreateBuffer(allocator, &sb, &sa, &staging, &staging_alloc, nullptr) != VK_SUCCESS) {
                vmaDestroyImage(allocator, out.image, out.alloc);
                out = {};
                return false;
            }
            void* mapped = nullptr;
            if (vmaMapMemory(allocator, staging_alloc, &mapped) != VK_SUCCESS) {
                vmaDestroyBuffer(allocator, staging, staging_alloc);
                vmaDestroyImage(allocator, out.image, out.alloc);
                out = {};
                return false;
            }
            std::memcpy(mapped, rgba, static_cast<std::size_t>(bytes));
            vmaFlushAllocation(allocator, staging_alloc, 0, bytes);
            vmaUnmapMemory(allocator, staging_alloc);

            VkCommandBuffer cb = beginSingleTimeCommands();
            if (cb == VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, staging, staging_alloc);
                vmaDestroyImage(allocator, out.image, out.alloc);
                out = {};
                return false;
            }

            // UNDEFINED → TRANSFER_DST_OPTIMAL
            VkImageMemoryBarrier to_dst{};
            to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_dst.image = out.image;
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
            vkCmdCopyBufferToImage(cb, staging, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   1, &region);

            VkImageMemoryBarrier to_shader{};
            to_shader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_shader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            to_shader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_shader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_shader.image = out.image;
            to_shader.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            to_shader.subresourceRange.levelCount = 1;
            to_shader.subresourceRange.layerCount = 1;
            to_shader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_shader);

            if (!endSingleTimeCommands(cb)) {
                vmaDestroyBuffer(allocator, staging, staging_alloc);
                vmaDestroyImage(allocator, out.image, out.alloc);
                out = {};
                return false;
            }
            vmaDestroyBuffer(allocator, staging, staging_alloc);

            VkImageViewCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vi.image = out.image;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = VK_FORMAT_R8G8B8A8_UNORM;
            vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vi.subresourceRange.levelCount = 1;
            vi.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &vi, nullptr, &out.view) != VK_SUCCESS) {
                vmaDestroyImage(allocator, out.image, out.alloc);
                out = {};
                return false;
            }
            return true;
        }

        bool createWhitePixel() {
            const std::uint8_t white[4] = {255, 255, 255, 255};
            return createTexture(white, 1, 1, white_pixel);
        }

        void destroyTexture(GpuTexture& t) const {
            if (t.view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, t.view, nullptr);
            }
            if (t.image != VK_NULL_HANDLE) {
                vmaDestroyImage(allocator, t.image, t.alloc);
            }
            t = {};
        }

        void destroyMaterial(GpuMaterial& m) const {
            if (m.ubo != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, m.ubo, m.ubo_alloc);
            }
            destroyTexture(m.albedo);
            destroyTexture(m.normal);
            destroyTexture(m.metallic_roughness);
            // descriptor set is freed when its pool is destroyed; we use a single shared
            // pool (reset on shutdown only) so per-mesh teardown leaves dangling sets,
            // which is fine because we never reuse them.
            m = {};
        }

        void destroyMesh(GpuMesh& m) const {
            if (m.vertex_buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, m.vertex_buffer, m.vertex_alloc);
            }
            if (m.index_buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, m.index_buffer, m.index_alloc);
            }
            for (auto& mat : m.materials) {
                destroyMaterial(mat);
            }
            destroyShadow(m.shadow);
            m = {};
        }

        bool createMainPipelines(VkFormat color_format, VkFormat depth_format) {
            using namespace viewport_shaders;

            VkShaderModule vert = createShaderModule(device, kMeshVertSpv, sizeof(kMeshVertSpv));
            VkShaderModule frag = createShaderModule(device, kMeshFragSpv, sizeof(kMeshFragSpv));
            if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
                if (vert)
                    vkDestroyShaderModule(device, vert, nullptr);
                if (frag)
                    vkDestroyShaderModule(device, frag, nullptr);
                LOG_ERROR("VulkanMeshPass: main shader module creation failed");
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
            binding.stride = sizeof(MeshVertex);
            binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            std::array<VkVertexInputAttributeDescription, 5> attrs{};
            attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(MeshVertex, position)};
            attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(MeshVertex, normal)};
            attrs[2] = {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(MeshVertex, tangent)};
            attrs[3] = {3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(MeshVertex, texcoord)};
            attrs[4] = {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(MeshVertex, color)};

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

            VkPipelineMultisampleStateCreateInfo multisample{};
            multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo depth{};
            depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depth.depthTestEnable = VK_TRUE;
            depth.depthWriteEnable = VK_TRUE;
            depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            VkPipelineColorBlendAttachmentState blend_attachment{};
            blend_attachment.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            blend_attachment.blendEnable = VK_FALSE;

            VkPipelineColorBlendStateCreateInfo blend{};
            blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            blend.attachmentCount = 1;
            blend.pAttachments = &blend_attachment;

            std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamic{};
            dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
            dynamic.pDynamicStates = dynamic_states.data();

            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            push.offset = 0;
            push.size = sizeof(MeshPushConstants);

            std::array<VkDescriptorSetLayout, 2> set_layouts{light_layout, material_layout};
            VkPipelineLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layout_info.setLayoutCount = static_cast<std::uint32_t>(set_layouts.size());
            layout_info.pSetLayouts = set_layouts.data();
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &push;
            if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
                vkDestroyShaderModule(device, vert, nullptr);
                vkDestroyShaderModule(device, frag, nullptr);
                LOG_ERROR("VulkanMeshPass: pipeline layout creation failed");
                return false;
            }

            VkPipelineRenderingCreateInfo rendering_info{};
            rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            rendering_info.colorAttachmentCount = 1;
            rendering_info.pColorAttachmentFormats = &color_format;
            rendering_info.depthAttachmentFormat = depth_format;
            rendering_info.stencilAttachmentFormat = depth_format;

            // Build cull and no-cull variants by tweaking only the rasterization state.
            // Mesh vertices use the same GL-convention projection matrices as FastGS,
            // then receive a clip-space Y correction before Vulkan's positive-height
            // viewport. That preserves normal CCW winding for back-face culling.
            const auto build = [&](VkCullModeFlags cull, VkPipeline& out) -> bool {
                VkPipelineRasterizationStateCreateInfo raster{};
                raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
                raster.polygonMode = VK_POLYGON_MODE_FILL;
                raster.cullMode = cull;
                raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
                raster.lineWidth = 1.0f;

                VkGraphicsPipelineCreateInfo pipeline_info{};
                pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
                pipeline_info.pNext = &rendering_info;
                pipeline_info.stageCount = 2;
                pipeline_info.pStages = stages.data();
                pipeline_info.pVertexInputState = &vertex_input;
                pipeline_info.pInputAssemblyState = &input_assembly;
                pipeline_info.pViewportState = &viewport_state;
                pipeline_info.pRasterizationState = &raster;
                pipeline_info.pMultisampleState = &multisample;
                pipeline_info.pDepthStencilState = &depth;
                pipeline_info.pColorBlendState = &blend;
                pipeline_info.pDynamicState = &dynamic;
                pipeline_info.layout = pipeline_layout;
                return vkCreateGraphicsPipelines(device, pipeline_cache, 1,
                                                 &pipeline_info, nullptr, &out) == VK_SUCCESS;
            };

            const bool ok = build(VK_CULL_MODE_BACK_BIT, pipeline_cull) &&
                            build(VK_CULL_MODE_NONE, pipeline_no_cull);
            vkDestroyShaderModule(device, vert, nullptr);
            vkDestroyShaderModule(device, frag, nullptr);
            if (!ok) {
                LOG_ERROR("VulkanMeshPass: main pipeline creation failed");
                return false;
            }
            return true;
        }

        bool createShadowPipeline(VkFormat depth_format) {
            using namespace viewport_shaders;
            VkShaderModule vert = createShaderModule(device, kMeshShadowVertSpv, sizeof(kMeshShadowVertSpv));
            VkShaderModule frag = createShaderModule(device, kMeshShadowFragSpv, sizeof(kMeshShadowFragSpv));
            if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
                if (vert)
                    vkDestroyShaderModule(device, vert, nullptr);
                if (frag)
                    vkDestroyShaderModule(device, frag, nullptr);
                LOG_ERROR("VulkanMeshPass: shadow shader module creation failed");
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
            binding.stride = sizeof(MeshVertex);
            binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            VkVertexInputAttributeDescription attr{};
            attr.location = 0;
            attr.binding = 0;
            attr.format = VK_FORMAT_R32G32B32_SFLOAT;
            attr.offset = offsetof(MeshVertex, position);
            VkPipelineVertexInputStateCreateInfo vertex_input{};
            vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertex_input.vertexBindingDescriptionCount = 1;
            vertex_input.pVertexBindingDescriptions = &binding;
            vertex_input.vertexAttributeDescriptionCount = 1;
            vertex_input.pVertexAttributeDescriptions = &attr;

            VkPipelineInputAssemblyStateCreateInfo input_assembly{};
            input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo viewport_state{};
            viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewport_state.viewportCount = 1;
            viewport_state.scissorCount = 1;

            // Master enables polygon offset (1.1, 4.0) and front-face culling for shadow.
            // Same Vulkan-vs-GL winding flip applies here — declare front face as
            // CLOCKWISE so VK_CULL_MODE_FRONT_BIT culls what is logically the front.
            VkPipelineRasterizationStateCreateInfo raster{};
            raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_FRONT_BIT;
            raster.frontFace = VK_FRONT_FACE_CLOCKWISE;
            raster.lineWidth = 1.0f;
            raster.depthBiasEnable = VK_TRUE;
            raster.depthBiasConstantFactor = 4.0f;
            raster.depthBiasSlopeFactor = 1.1f;

            VkPipelineMultisampleStateCreateInfo multisample{};
            multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo depth{};
            depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depth.depthTestEnable = VK_TRUE;
            depth.depthWriteEnable = VK_TRUE;
            depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            VkPipelineColorBlendStateCreateInfo blend{};
            blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            blend.attachmentCount = 0;

            std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamic{};
            dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
            dynamic.pDynamicStates = dynamic_states.data();

            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            push.offset = 0;
            push.size = sizeof(ShadowPush);

            VkPipelineLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &push;
            if (vkCreatePipelineLayout(device, &layout_info, nullptr, &shadow_pipeline_layout) != VK_SUCCESS) {
                vkDestroyShaderModule(device, vert, nullptr);
                vkDestroyShaderModule(device, frag, nullptr);
                return false;
            }

            VkPipelineRenderingCreateInfo rendering_info{};
            rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            rendering_info.colorAttachmentCount = 0;
            rendering_info.depthAttachmentFormat = depth_format;
            rendering_info.stencilAttachmentFormat = depth_format;

            VkGraphicsPipelineCreateInfo pipeline_info{};
            pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipeline_info.pNext = &rendering_info;
            pipeline_info.stageCount = 2;
            pipeline_info.pStages = stages.data();
            pipeline_info.pVertexInputState = &vertex_input;
            pipeline_info.pInputAssemblyState = &input_assembly;
            pipeline_info.pViewportState = &viewport_state;
            pipeline_info.pRasterizationState = &raster;
            pipeline_info.pMultisampleState = &multisample;
            pipeline_info.pDepthStencilState = &depth;
            pipeline_info.pColorBlendState = &blend;
            pipeline_info.pDynamicState = &dynamic;
            pipeline_info.layout = shadow_pipeline_layout;

            const VkResult result = vkCreateGraphicsPipelines(device, pipeline_cache, 1,
                                                              &pipeline_info, nullptr, &shadow_pipeline);
            vkDestroyShaderModule(device, vert, nullptr);
            vkDestroyShaderModule(device, frag, nullptr);
            return result == VK_SUCCESS;
        }

        bool createWireframePipeline(VkFormat color_format, VkFormat depth_format) {
            using namespace viewport_shaders;
            VkShaderModule vert = createShaderModule(device, kMeshWireframeVertSpv, sizeof(kMeshWireframeVertSpv));
            VkShaderModule frag = createShaderModule(device, kMeshWireframeFragSpv, sizeof(kMeshWireframeFragSpv));
            if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
                if (vert)
                    vkDestroyShaderModule(device, vert, nullptr);
                if (frag)
                    vkDestroyShaderModule(device, frag, nullptr);
                LOG_ERROR("VulkanMeshPass: wireframe shader module creation failed");
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
            binding.stride = sizeof(MeshVertex);
            binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            VkVertexInputAttributeDescription attr{};
            attr.location = 0;
            attr.binding = 0;
            attr.format = VK_FORMAT_R32G32B32_SFLOAT;
            attr.offset = offsetof(MeshVertex, position);
            VkPipelineVertexInputStateCreateInfo vertex_input{};
            vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertex_input.vertexBindingDescriptionCount = 1;
            vertex_input.pVertexBindingDescriptions = &binding;
            vertex_input.vertexAttributeDescriptionCount = 1;
            vertex_input.pVertexAttributeDescriptions = &attr;

            VkPipelineInputAssemblyStateCreateInfo input_assembly{};
            input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo viewport_state{};
            viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewport_state.viewportCount = 1;
            viewport_state.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo raster{};
            raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            raster.polygonMode = VK_POLYGON_MODE_LINE;
            raster.cullMode = VK_CULL_MODE_NONE;
            raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo multisample{};
            multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            // Wireframe sits on top of the shaded mesh — depth test enabled with
            // OP_LESS_OR_EQUAL but write disabled so the underlying mesh's depth wins.
            VkPipelineDepthStencilStateCreateInfo depth{};
            depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depth.depthTestEnable = VK_TRUE;
            depth.depthWriteEnable = VK_FALSE;
            depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            VkPipelineColorBlendAttachmentState blend_attachment{};
            blend_attachment.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            blend_attachment.blendEnable = VK_FALSE;

            VkPipelineColorBlendStateCreateInfo blend{};
            blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            blend.attachmentCount = 1;
            blend.pAttachments = &blend_attachment;

            std::array<VkDynamicState, 3> dynamic_states{
                VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_LINE_WIDTH};
            VkPipelineDynamicStateCreateInfo dynamic{};
            dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
            dynamic.pDynamicStates = dynamic_states.data();

            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            push.offset = 0;
            push.size = sizeof(WireframePush);

            VkPipelineLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &push;
            if (vkCreatePipelineLayout(device, &layout_info, nullptr, &wireframe_pipeline_layout) != VK_SUCCESS) {
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

            VkGraphicsPipelineCreateInfo pipeline_info{};
            pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipeline_info.pNext = &rendering_info;
            pipeline_info.stageCount = 2;
            pipeline_info.pStages = stages.data();
            pipeline_info.pVertexInputState = &vertex_input;
            pipeline_info.pInputAssemblyState = &input_assembly;
            pipeline_info.pViewportState = &viewport_state;
            pipeline_info.pRasterizationState = &raster;
            pipeline_info.pMultisampleState = &multisample;
            pipeline_info.pDepthStencilState = &depth;
            pipeline_info.pColorBlendState = &blend;
            pipeline_info.pDynamicState = &dynamic;
            pipeline_info.layout = wireframe_pipeline_layout;

            const VkResult result = vkCreateGraphicsPipelines(device, pipeline_cache, 1,
                                                              &pipeline_info, nullptr, &wireframe_pipeline);
            vkDestroyShaderModule(device, vert, nullptr);
            vkDestroyShaderModule(device, frag, nullptr);
            return result == VK_SUCCESS;
        }

        bool uploadTextureFromMesh(const lfs::core::MeshData& mesh, std::uint32_t tex_index, GpuTexture& out) {
            if (tex_index == 0 || tex_index > mesh.texture_images.size()) {
                return false;
            }
            const auto& img = mesh.texture_images[tex_index - 1];
            if (img.pixels.empty() || img.width <= 0 || img.height <= 0) {
                return false;
            }
            std::vector<std::uint8_t> rgba(static_cast<std::size_t>(img.width) * img.height * 4u);
            const int ch = img.channels;
            for (int y = 0; y < img.height; ++y) {
                for (int x = 0; x < img.width; ++x) {
                    const std::size_t src = (static_cast<std::size_t>(y) * img.width + x) * static_cast<std::size_t>(ch);
                    const std::size_t dst = (static_cast<std::size_t>(y) * img.width + x) * 4u;
                    rgba[dst + 0] = ch >= 1 ? img.pixels[src + 0] : 255;
                    rgba[dst + 1] = ch >= 2 ? img.pixels[src + 1] : rgba[dst + 0];
                    rgba[dst + 2] = ch >= 3 ? img.pixels[src + 2] : rgba[dst + 0];
                    rgba[dst + 3] = ch >= 4 ? img.pixels[src + 3] : 255;
                }
            }
            return createTexture(rgba.data(), img.width, img.height, out);
        }

        bool uploadMaterial(const lfs::core::MeshData& mesh, std::size_t material_index, GpuMaterial& out) {
            const auto& mat = material_index < mesh.materials.size() ? mesh.materials[material_index]
                                                                     : lfs::core::Material{};

            // UBO
            VkBufferCreateInfo b{};
            b.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            b.size = sizeof(MaterialUbo);
            b.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            b.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo a{};
            a.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            a.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            if (vmaCreateBuffer(allocator, &b, &a, &out.ubo, &out.ubo_alloc, nullptr) != VK_SUCCESS) {
                return false;
            }

            const bool has_albedo = uploadTextureFromMesh(mesh, mat.albedo_tex, out.albedo);
            const bool has_normal = uploadTextureFromMesh(mesh, mat.normal_tex, out.normal);
            const bool has_mr = uploadTextureFromMesh(mesh, mat.metallic_roughness_tex, out.metallic_roughness);
            const bool has_vc = mesh.has_colors();

            MaterialUbo ubo{};
            ubo.base_color[0] = mat.base_color.r;
            ubo.base_color[1] = mat.base_color.g;
            ubo.base_color[2] = mat.base_color.b;
            ubo.base_color[3] = mat.base_color.a;
            ubo.emissive_metallic[0] = mat.emissive.r;
            ubo.emissive_metallic[1] = mat.emissive.g;
            ubo.emissive_metallic[2] = mat.emissive.b;
            ubo.emissive_metallic[3] = mat.metallic;
            ubo.roughness_flags[0] = mat.roughness;
            ubo.roughness_flags[1] = has_albedo ? 1.0f : 0.0f;
            ubo.roughness_flags[2] = has_normal ? 1.0f : 0.0f;
            ubo.roughness_flags[3] = has_mr ? 1.0f : 0.0f;
            // Emphasis (y/z/w) is rewritten per frame in updateEmphasis() to track the
            // current selection state without re-uploading the full UBO.
            ubo.vertex_color_emphasis[0] = has_vc ? 1.0f : 0.0f;
            ubo.vertex_color_emphasis[1] = 0.0f;
            ubo.vertex_color_emphasis[2] = 0.0f;
            ubo.vertex_color_emphasis[3] = 0.0f;
            writeBuffer(out.ubo_alloc, &ubo, sizeof(ubo));

            // Allocate descriptor set
            VkDescriptorSetAllocateInfo alloc{};
            alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc.descriptorPool = descriptor_pool;
            alloc.descriptorSetCount = 1;
            alloc.pSetLayouts = &material_layout;
            if (vkAllocateDescriptorSets(device, &alloc, &out.descriptor) != VK_SUCCESS) {
                LOG_WARN("VulkanMeshPass: descriptor pool exhausted");
                return false;
            }

            VkDescriptorBufferInfo bi{};
            bi.buffer = out.ubo;
            bi.range = sizeof(MaterialUbo);

            const auto pick_view = [&](const GpuTexture& t) {
                return t.view != VK_NULL_HANDLE ? t.view : white_pixel.view;
            };
            std::array<VkDescriptorImageInfo, 3> ii{};
            ii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ii[0].imageView = pick_view(out.albedo);
            ii[0].sampler = sampler;
            ii[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ii[1].imageView = pick_view(out.normal);
            ii[1].sampler = sampler;
            ii[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ii[2].imageView = pick_view(out.metallic_roughness);
            ii[2].sampler = sampler;

            std::array<VkWriteDescriptorSet, 4> writes{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = out.descriptor;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo = &bi;
            for (int i = 0; i < 3; ++i) {
                writes[i + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i + 1].dstSet = out.descriptor;
                writes[i + 1].dstBinding = static_cast<std::uint32_t>(i + 1);
                writes[i + 1].descriptorCount = 1;
                writes[i + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[i + 1].pImageInfo = &ii[i];
            }
            vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
            return true;
        }

        bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                          VkBuffer& buffer, VmaAllocation& alloc) const {
            VkBufferCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            info.size = size;
            info.usage = usage;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo ai{};
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            return vmaCreateBuffer(allocator, &info, &ai, &buffer, &alloc, nullptr) == VK_SUCCESS;
        }

        bool uploadMesh(const lfs::core::MeshData& mesh, GpuMesh& gpu) {
            const std::int64_t vcount = mesh.vertex_count();
            const std::int64_t fcount = mesh.face_count();
            if (vcount <= 0 || fcount <= 0) {
                return false;
            }

            auto verts_cpu = mesh.vertices.cpu().contiguous();
            auto idx_cpu = mesh.indices.cpu().contiguous();
            const float* pos = verts_cpu.ptr<float>();
            const std::int32_t* idx = idx_cpu.ptr<std::int32_t>();

            const float* nrm = nullptr;
            lfs::core::Tensor nrm_cpu;
            if (mesh.has_normals()) {
                nrm_cpu = mesh.normals.cpu().contiguous();
                nrm = nrm_cpu.ptr<float>();
            }
            const float* tan = nullptr;
            lfs::core::Tensor tan_cpu;
            if (mesh.has_tangents()) {
                tan_cpu = mesh.tangents.cpu().contiguous();
                tan = tan_cpu.ptr<float>();
            }
            const float* uv = nullptr;
            lfs::core::Tensor uv_cpu;
            if (mesh.has_texcoords()) {
                uv_cpu = mesh.texcoords.cpu().contiguous();
                uv = uv_cpu.ptr<float>();
            }
            const float* col = nullptr;
            lfs::core::Tensor col_cpu;
            if (mesh.has_colors()) {
                col_cpu = mesh.colors.cpu().contiguous();
                col = col_cpu.ptr<float>();
            }

            glm::vec3 aabb_min(std::numeric_limits<float>::max());
            glm::vec3 aabb_max(std::numeric_limits<float>::lowest());

            std::vector<MeshVertex> vertices(static_cast<std::size_t>(vcount));
            for (std::int64_t i = 0; i < vcount; ++i) {
                MeshVertex& v = vertices[static_cast<std::size_t>(i)];
                v.position[0] = pos[i * 3 + 0];
                v.position[1] = pos[i * 3 + 1];
                v.position[2] = pos[i * 3 + 2];
                aabb_min = glm::min(aabb_min, glm::vec3(v.position[0], v.position[1], v.position[2]));
                aabb_max = glm::max(aabb_max, glm::vec3(v.position[0], v.position[1], v.position[2]));
                if (nrm) {
                    v.normal[0] = nrm[i * 3 + 0];
                    v.normal[1] = nrm[i * 3 + 1];
                    v.normal[2] = nrm[i * 3 + 2];
                } else {
                    v.normal[0] = 0.0f;
                    v.normal[1] = 1.0f;
                    v.normal[2] = 0.0f;
                }
                if (tan) {
                    v.tangent[0] = tan[i * 4 + 0];
                    v.tangent[1] = tan[i * 4 + 1];
                    v.tangent[2] = tan[i * 4 + 2];
                    v.tangent[3] = tan[i * 4 + 3];
                } else {
                    v.tangent[0] = 0.0f;
                    v.tangent[1] = 0.0f;
                    v.tangent[2] = 0.0f;
                    v.tangent[3] = 1.0f;
                }
                if (uv) {
                    v.texcoord[0] = uv[i * 2 + 0];
                    v.texcoord[1] = uv[i * 2 + 1];
                } else {
                    v.texcoord[0] = 0.0f;
                    v.texcoord[1] = 0.0f;
                }
                if (col) {
                    v.color[0] = col[i * 4 + 0];
                    v.color[1] = col[i * 4 + 1];
                    v.color[2] = col[i * 4 + 2];
                    v.color[3] = col[i * 4 + 3];
                } else {
                    v.color[0] = 1.0f;
                    v.color[1] = 1.0f;
                    v.color[2] = 1.0f;
                    v.color[3] = 1.0f;
                }
            }

            const std::size_t total_indices = static_cast<std::size_t>(fcount) * 3u;
            std::vector<std::uint32_t> indices(total_indices);
            for (std::size_t i = 0; i < total_indices; ++i) {
                indices[i] = static_cast<std::uint32_t>(idx[i]);
            }

            destroyMesh(gpu);

            const std::size_t vbytes = vertices.size() * sizeof(MeshVertex);
            const std::size_t ibytes = indices.size() * sizeof(std::uint32_t);
            if (!createBuffer(vbytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, gpu.vertex_buffer, gpu.vertex_alloc) ||
                !createBuffer(ibytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, gpu.index_buffer, gpu.index_alloc)) {
                destroyMesh(gpu);
                return false;
            }
            if (!writeBuffer(gpu.vertex_alloc, vertices.data(), vbytes) ||
                !writeBuffer(gpu.index_alloc, indices.data(), ibytes)) {
                destroyMesh(gpu);
                return false;
            }
            gpu.total_index_count = static_cast<std::uint32_t>(total_indices);

            // Materials
            const std::size_t mat_count = std::max<std::size_t>(mesh.materials.size(), 1);
            gpu.materials.resize(mat_count);
            for (std::size_t i = 0; i < mat_count; ++i) {
                if (!uploadMaterial(mesh, i, gpu.materials[i])) {
                    LOG_WARN("VulkanMeshPass: failed to upload material {} for mesh", i);
                }
            }

            // Submeshes — fall back to single submesh covering all indices
            if (!mesh.submeshes.empty()) {
                gpu.submeshes.reserve(mesh.submeshes.size());
                for (const auto& sm : mesh.submeshes) {
                    gpu.submeshes.push_back({static_cast<std::uint32_t>(sm.start_index),
                                             static_cast<std::uint32_t>(sm.index_count),
                                             sm.material_index});
                }
            } else {
                gpu.submeshes.push_back({0, gpu.total_index_count, 0});
            }

            gpu.aabb_min = aabb_min;
            gpu.aabb_max = aabb_max;
            gpu.generation = mesh.generation();
            return true;
        }

        void writeLightUbo(const VulkanMeshPassParams& params, const VulkanMeshDrawItem& item,
                           const glm::mat4& light_vp, bool shadow_enabled) {
            LightUbo ubo{};
            ubo.camera_pos[0] = params.camera_position.x;
            ubo.camera_pos[1] = params.camera_position.y;
            ubo.camera_pos[2] = params.camera_position.z;
            ubo.camera_pos[3] = 1.0f;
            ubo.light_dir[0] = item.light_dir.x;
            ubo.light_dir[1] = item.light_dir.y;
            ubo.light_dir[2] = item.light_dir.z;
            ubo.light_dir[3] = 0.0f;
            ubo.params[0] = item.light_intensity;
            ubo.params[1] = item.ambient;
            ubo.params[2] = shadow_enabled ? 1.0f : 0.0f;
            ubo.params[3] = 0.0f;
            std::memcpy(ubo.light_vp, &light_vp[0][0], sizeof(ubo.light_vp));
            writeBuffer(light_alloc, &ubo, sizeof(ubo));
        }

        void bindShadowMap(const ShadowTarget& target) {
            VkDescriptorImageInfo info{};
            info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            info.imageView = target.view != VK_NULL_HANDLE ? target.view : shadow_dummy.view;
            info.sampler = shadow_sampler;
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = light_descriptor;
            w.dstBinding = 1;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &info;
            vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
        }

        void updateMaterialEmphasis(GpuMaterial& mat, const VulkanMeshDrawItem& item) const {
            // Patch only the emphasis word (offset 48) of MaterialUbo to track per-frame
            // selection state without re-uploading the rest of the material.
            float emphasis[4] = {
                0.0f, // x = has_vertex_colors — preserved by ORing? we just rewrite full vec4
                item.is_emphasized ? 1.0f : 0.0f,
                item.dim_non_emphasized ? 1.0f : 0.0f,
                item.flash_intensity,
            };
            // To preserve the has_vertex_colors flag set at upload time we can't rewrite
            // the whole vec4. Map and rewrite only y/z/w (offsets +52, +56, +60).
            void* mapped = nullptr;
            if (vmaMapMemory(allocator, mat.ubo_alloc, &mapped) != VK_SUCCESS || !mapped) {
                return;
            }
            auto* dst = static_cast<float*>(mapped);
            // vertex_color_emphasis sits at the start of the 4th vec4 (offset 48 floats / 4 = 12).
            constexpr std::size_t kVcEmphasisFloatOffset = 12;
            dst[kVcEmphasisFloatOffset + 1] = emphasis[1];
            dst[kVcEmphasisFloatOffset + 2] = emphasis[2];
            dst[kVcEmphasisFloatOffset + 3] = emphasis[3];
            vmaFlushAllocation(allocator, mat.ubo_alloc, 0, sizeof(MaterialUbo));
            vmaUnmapMemory(allocator, mat.ubo_alloc);
        }

        glm::mat4 computeLightVp(const GpuMesh& gpu, const glm::mat4& model,
                                 const glm::vec3& light_dir) const {
            const std::array<glm::vec3, 8> corners{
                glm::vec3{gpu.aabb_min.x, gpu.aabb_min.y, gpu.aabb_min.z},
                glm::vec3{gpu.aabb_max.x, gpu.aabb_min.y, gpu.aabb_min.z},
                glm::vec3{gpu.aabb_min.x, gpu.aabb_max.y, gpu.aabb_min.z},
                glm::vec3{gpu.aabb_max.x, gpu.aabb_max.y, gpu.aabb_min.z},
                glm::vec3{gpu.aabb_min.x, gpu.aabb_min.y, gpu.aabb_max.z},
                glm::vec3{gpu.aabb_max.x, gpu.aabb_min.y, gpu.aabb_max.z},
                glm::vec3{gpu.aabb_min.x, gpu.aabb_max.y, gpu.aabb_max.z},
                glm::vec3{gpu.aabb_max.x, gpu.aabb_max.y, gpu.aabb_max.z},
            };
            glm::vec3 ws_min(std::numeric_limits<float>::max());
            glm::vec3 ws_max(std::numeric_limits<float>::lowest());
            for (const auto& c : corners) {
                const glm::vec3 wp = glm::vec3(model * glm::vec4(c, 1.0f));
                ws_min = glm::min(ws_min, wp);
                ws_max = glm::max(ws_max, wp);
            }

            const glm::vec3 center = (ws_min + ws_max) * 0.5f;
            const float radius = glm::length(ws_max - ws_min) * 0.5f;
            const glm::vec3 dir = glm::length(light_dir) > 1e-6f ? glm::normalize(light_dir)
                                                                 : glm::vec3(0.0f, 1.0f, 0.0f);
            const glm::vec3 eye = center + dir * radius * 2.0f;
            glm::vec3 up(0.0f, 1.0f, 0.0f);
            if (std::abs(glm::dot(dir, up)) > 0.99f) {
                up = glm::vec3(0.0f, 0.0f, 1.0f);
            }
            const glm::mat4 light_view = glm::lookAt(eye, center, up);
            const glm::mat4 light_proj = glm::ortho(-radius, radius, -radius, radius,
                                                    0.01f, radius * 4.0f);
            return light_proj * light_view;
        }

        bool ensureShadowTarget(GpuMesh& gpu, int resolution) {
            if (gpu.shadow.image != VK_NULL_HANDLE && gpu.shadow.resolution == resolution) {
                return true;
            }
            return createShadowTarget(resolution, gpu.shadow);
        }

        bool recordShadowPass(GpuMesh& gpu, const glm::mat4& light_mvp) {
            if (gpu.shadow.image == VK_NULL_HANDLE || shadow_pipeline == VK_NULL_HANDLE) {
                return false;
            }
            VkCommandBuffer cb = beginSingleTimeCommands();
            if (cb == VK_NULL_HANDLE) {
                return false;
            }

            VkImageMemoryBarrier to_attach{};
            to_attach.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_attach.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            to_attach.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            to_attach.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_attach.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_attach.image = gpu.shadow.image;
            to_attach.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            to_attach.subresourceRange.levelCount = 1;
            to_attach.subresourceRange.layerCount = 1;
            to_attach.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            to_attach.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_attach);

            VkClearValue clear{};
            clear.depthStencil = {1.0f, 0};
            VkRenderingAttachmentInfo depth_attachment{};
            depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView = gpu.shadow.view;
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_attachment.clearValue = clear;

            VkRenderingInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering.renderArea.offset = {0, 0};
            rendering.renderArea.extent = {static_cast<std::uint32_t>(gpu.shadow.resolution),
                                           static_cast<std::uint32_t>(gpu.shadow.resolution)};
            rendering.layerCount = 1;
            rendering.pDepthAttachment = &depth_attachment;
            vkCmdBeginRendering(cb, &rendering);

            VkViewport vp{};
            vp.x = 0.0f;
            vp.y = 0.0f;
            vp.width = static_cast<float>(gpu.shadow.resolution);
            vp.height = static_cast<float>(gpu.shadow.resolution);
            vp.minDepth = 0.0f;
            vp.maxDepth = 1.0f;
            VkRect2D sc{};
            sc.extent = rendering.renderArea.extent;
            vkCmdSetViewport(cb, 0, 1, &vp);
            vkCmdSetScissor(cb, 0, 1, &sc);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);
            ShadowPush push{};
            std::memcpy(push.light_mvp, &light_mvp[0][0], sizeof(push.light_mvp));
            vkCmdPushConstants(cb, shadow_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(push), &push);

            VkBuffer vbuf = gpu.vertex_buffer;
            VkDeviceSize voff = 0;
            vkCmdBindVertexBuffers(cb, 0, 1, &vbuf, &voff);
            vkCmdBindIndexBuffer(cb, gpu.index_buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cb, gpu.total_index_count, 1, 0, 0, 0);

            vkCmdEndRendering(cb);

            VkImageMemoryBarrier to_read{};
            to_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_read.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            to_read.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_read.image = gpu.shadow.image;
            to_read.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            to_read.subresourceRange.levelCount = 1;
            to_read.subresourceRange.layerCount = 1;
            to_read.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_read);

            return endSingleTimeCommands(cb);
        }

        void prepare(const VulkanMeshPassParams& params) {
            ++frame_counter;
            for (const auto& item : params.items) {
                if (!item.mesh)
                    continue;
                auto it = mesh_cache.find(item.mesh);
                if (it == mesh_cache.end()) {
                    GpuMesh gpu{};
                    if (uploadMesh(*item.mesh, gpu)) {
                        gpu.last_used_frame = frame_counter;
                        mesh_cache.emplace(item.mesh, std::move(gpu));
                    }
                } else if (it->second.generation != item.mesh->generation()) {
                    if (uploadMesh(*item.mesh, it->second)) {
                        it->second.last_used_frame = frame_counter;
                    }
                } else {
                    it->second.last_used_frame = frame_counter;
                }
            }

            for (const auto& item : params.items) {
                if (!item.mesh || !item.shadow_enabled || item.shadow_map_resolution <= 0) {
                    continue;
                }
                auto it = mesh_cache.find(item.mesh);
                if (it == mesh_cache.end() || it->second.vertex_buffer == VK_NULL_HANDLE) {
                    continue;
                }
                auto& gpu = it->second;
                if (!ensureShadowTarget(gpu, item.shadow_map_resolution)) {
                    continue;
                }
                const glm::mat4 light_vp = computeLightVp(gpu, item.model, item.light_dir);
                gpu.cached_light_vp = light_vp;
                gpu.cached_light_vp_valid = true;
                recordShadowPass(gpu, light_vp * item.model);
            }

            constexpr std::uint64_t kEvictAfter = 120;
            for (auto it = mesh_cache.begin(); it != mesh_cache.end();) {
                if (frame_counter - it->second.last_used_frame > kEvictAfter) {
                    destroyMesh(it->second);
                    it = mesh_cache.erase(it);
                } else {
                    ++it;
                }
            }
        }

        void record(VkCommandBuffer cb, VkRect2D viewport_rect, const VulkanMeshPassParams& params) {
            if (pipeline_cull == VK_NULL_HANDLE || params.items.empty()) {
                return;
            }

            VkViewport viewport{};
            viewport.x = static_cast<float>(viewport_rect.offset.x);
            viewport.y = static_cast<float>(viewport_rect.offset.y);
            viewport.width = static_cast<float>(viewport_rect.extent.width);
            viewport.height = static_cast<float>(viewport_rect.extent.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            VkRect2D scissor{};
            scissor.offset = viewport_rect.offset;
            scissor.extent = viewport_rect.extent;
            vkCmdSetViewport(cb, 0, 1, &viewport);
            vkCmdSetScissor(cb, 0, 1, &scissor);

            glm::mat4 clip_y_flip(1.0f);
            clip_y_flip[1][1] = -1.0f;
            const glm::mat4 view_projection = clip_y_flip * params.view_projection;

            for (const auto& item : params.items) {
                if (!item.mesh)
                    continue;
                auto it = mesh_cache.find(item.mesh);
                if (it == mesh_cache.end() || it->second.vertex_buffer == VK_NULL_HANDLE) {
                    continue;
                }
                auto& gpu = it->second;

                const bool shadow_active = item.shadow_enabled &&
                                           gpu.shadow.image != VK_NULL_HANDLE &&
                                           gpu.cached_light_vp_valid;
                writeLightUbo(params, item,
                              shadow_active ? gpu.cached_light_vp : glm::mat4(1.0f),
                              shadow_active);
                bindShadowMap(shadow_active ? gpu.shadow : shadow_dummy);

                // Patch per-frame emphasis state into the material UBOs for this mesh.
                for (auto& mat : gpu.materials) {
                    if (mat.ubo_alloc != VK_NULL_HANDLE) {
                        updateMaterialEmphasis(mat, item);
                    }
                }

                VkPipeline main_pipeline = item.backface_culling ? pipeline_cull : pipeline_no_cull;
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, main_pipeline);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                        0, 1, &light_descriptor, 0, nullptr);

                MeshPushConstants pc{};
                const glm::mat4 mvp = view_projection * item.model;
                std::memcpy(pc.mvp, &mvp[0][0], sizeof(pc.mvp));
                std::memcpy(pc.model, &item.model[0][0], sizeof(pc.model));
                vkCmdPushConstants(cb, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                                   0, sizeof(pc), &pc);

                VkBuffer vbuf = gpu.vertex_buffer;
                VkDeviceSize voff = 0;
                vkCmdBindVertexBuffers(cb, 0, 1, &vbuf, &voff);
                vkCmdBindIndexBuffer(cb, gpu.index_buffer, 0, VK_INDEX_TYPE_UINT32);

                for (const auto& sm : gpu.submeshes) {
                    if (sm.index_count == 0)
                        continue;
                    const std::size_t mat_idx = std::min(sm.material_index, gpu.materials.size() - 1);
                    const auto& mat = gpu.materials[mat_idx];
                    if (mat.descriptor != VK_NULL_HANDLE) {
                        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                                1, 1, &mat.descriptor, 0, nullptr);
                    }
                    vkCmdDrawIndexed(cb, sm.index_count, 1, sm.start_index, 0, 0);
                }

                // Wireframe overlay drawn on top of the shaded mesh.
                if (item.wireframe_overlay && wireframe_pipeline != VK_NULL_HANDLE) {
                    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, wireframe_pipeline);
                    vkCmdSetLineWidth(cb, std::max(item.wireframe_width, 1.0f));
                    WireframePush wpush{};
                    std::memcpy(wpush.mvp, &mvp[0][0], sizeof(wpush.mvp));
                    wpush.color[0] = item.wireframe_color.r;
                    wpush.color[1] = item.wireframe_color.g;
                    wpush.color[2] = item.wireframe_color.b;
                    wpush.color[3] = 1.0f;
                    vkCmdPushConstants(cb, wireframe_pipeline_layout,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                       0, sizeof(wpush), &wpush);
                    vkCmdDrawIndexed(cb, gpu.total_index_count, 1, 0, 0, 0);
                }
            }
        }
    };

    VulkanMeshPass::VulkanMeshPass() = default;
    VulkanMeshPass::~VulkanMeshPass() = default;
    VulkanMeshPass::VulkanMeshPass(VulkanMeshPass&&) noexcept = default;
    VulkanMeshPass& VulkanMeshPass::operator=(VulkanMeshPass&&) noexcept = default;

    bool VulkanMeshPass::init(VulkanContext& context, VkFormat color_format, VkFormat depth_stencil_format) {
        if (!impl_) {
            impl_ = std::make_unique<Impl>();
        }
        return impl_->init(context, color_format, depth_stencil_format);
    }

    void VulkanMeshPass::prepare(VulkanContext&, const VulkanMeshPassParams& params) {
        if (!impl_)
            return;
        impl_->prepare(params);
    }

    void VulkanMeshPass::record(VkCommandBuffer command_buffer, VkRect2D viewport_rect,
                                const VulkanMeshPassParams& params) {
        if (!impl_)
            return;
        impl_->record(command_buffer, viewport_rect, params);
    }

    void VulkanMeshPass::shutdown() {
        if (impl_) {
            impl_->destroy();
            impl_.reset();
        }
    }

} // namespace lfs::vis
