/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_mesh_pass.hpp"

#include "core/logger.hpp"
#include "core/material.hpp"
#include "core/mesh_data.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "window/vulkan_barrier2.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_result.hpp"

#include <array>
#include <cstring>
#include <format>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <string_view>
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
            float selection[4]; // x = emphasized, y = dim others, z = flash intensity
            float light_vp[16]; // light view-projection (column-major) for shadow sampling
        };
        static_assert(sizeof(LightUbo) == 128, "LightUbo layout");

        struct MaterialUbo {
            float base_color[4];
            float emissive_metallic[4];  // xyz emissive, w metallic
            float roughness_flags[4];    // x roughness, y has_albedo, z has_normal, w has_metallic_roughness
            float vertex_color_flags[4]; // x has_vertex_colors, yzw reserved
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
        std::vector<VkDescriptorPool> material_descriptor_pools;
        VkCommandPool transfer_pool = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;

        struct LightDrawResources {
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkDescriptorSet descriptor = VK_NULL_HANDLE;
        };

        struct FrameLightResources {
            VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
            std::vector<LightDrawResources> draws;
            std::size_t descriptor_capacity = 0;
        };

        std::vector<FrameLightResources> frame_light_resources;

        // 1x1 white fallback texture for materials missing a given texture.
        struct GpuTexture {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation alloc = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            std::string vram_label;
        };
        GpuTexture white_pixel{};

        struct GpuMaterial {
            VkBuffer ubo = VK_NULL_HANDLE;
            VmaAllocation ubo_alloc = VK_NULL_HANDLE;
            VkDescriptorSet descriptor = VK_NULL_HANDLE;
            VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
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
            std::string vram_label;
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

            // Inputs the rendered shadow map depends on. The shadow is re-rendered (a
            // blocking GPU submit) only when one of these changes.
            glm::mat4 shadow_key_model{0.0f};
            glm::vec3 shadow_key_light_dir{0.0f};
            int shadow_key_resolution = -1;
            std::uint32_t shadow_key_generation = std::numeric_limits<std::uint32_t>::max();
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
                return logVkFailure(std::format(
                    "Mesh-pass initialization requires a live device, allocator, and graphics queue (device={:#x}, allocator={:#x}, graphics_queue={:#x}, pipeline_cache={:#x}) ({}:{})",
                    vkHandleValue(device),
                    reinterpret_cast<std::uintptr_t>(allocator),
                    vkHandleValue(graphics_queue),
                    vkHandleValue(pipeline_cache),
                    __FILE__,
                    __LINE__));
            }

            VkCommandPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool_info.queueFamilyIndex = ctx.graphicsQueueFamily();
            LFS_VK_CHECK_MSG(vkCreateCommandPool(device, &pool_info, nullptr, &transfer_pool),
                             "Mesh transfer command-pool creation failed (device={:#x}, queue_family={}, flags={:#x})",
                             vkHandleValue(device),
                             pool_info.queueFamilyIndex,
                             static_cast<std::uint32_t>(pool_info.flags));
            context->setDebugObjectName(VK_OBJECT_TYPE_COMMAND_POOL,
                                        transfer_pool,
                                        "mesh.transfer.pool");

            color_format_cached = color_format;
            depth_format_cached = depth_format;
            shadow_format = depth_format; // re-use the swapchain's depth format

            return createSamplers() &&
                   createDescriptorLayouts() &&
                   createInitialMaterialDescriptorPool() &&
                   createFrameLightResources() &&
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
            VkResult result = vkAllocateCommandBuffers(device, &alloc, &cb);
            if (result != VK_SUCCESS) {
                LOG_ERROR("Vulkan: {}",
                          formatVkCheckFailure(
                              "vkAllocateCommandBuffers(device, &alloc, &cb)",
                              result,
                              std::format("Mesh one-shot command-buffer allocation failed (device={:#x}, command_pool={:#x}, requested_count={})",
                                          vkHandleValue(device),
                                          vkHandleValue(transfer_pool),
                                          alloc.commandBufferCount),
                              __FILE__,
                              __LINE__));
                return VK_NULL_HANDLE;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER,
                                        cb,
                                        "mesh.transfer.command");
            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            result = vkBeginCommandBuffer(cb, &begin);
            if (result != VK_SUCCESS) {
                LOG_ERROR("Vulkan: {}",
                          formatVkCheckFailure(
                              "vkBeginCommandBuffer(cb, &begin)",
                              result,
                              std::format("Mesh one-shot command buffer did not enter recording state (command_buffer={:#x}, command_pool={:#x})",
                                          vkHandleValue(cb),
                                          vkHandleValue(transfer_pool)),
                              __FILE__,
                              __LINE__));
                vkFreeCommandBuffers(device, transfer_pool, 1, &cb);
                return VK_NULL_HANDLE;
            }
            return cb;
        }

        bool endSingleTimeCommands(VkCommandBuffer cb) const {
            VkResult r = vkEndCommandBuffer(cb);
            if (r != VK_SUCCESS) {
                vkFreeCommandBuffers(device, transfer_pool, 1, &cb);
                return reportVkFailure(
                    "vkEndCommandBuffer(cb)",
                    r,
                    std::format("Mesh one-shot command buffer did not leave recording state (command_buffer={:#x}, command_pool={:#x})",
                                vkHandleValue(cb),
                                vkHandleValue(transfer_pool)));
            }
            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cb;
            VkFenceCreateInfo finfo{};
            finfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VkFence fence = VK_NULL_HANDLE;
            r = vkCreateFence(device, &finfo, nullptr, &fence);
            std::string failed_expression;
            std::string failed_context;
            if (r != VK_SUCCESS) {
                failed_expression = "vkCreateFence(device, &finfo, nullptr, &fence)";
                failed_context = std::format(
                    "Mesh one-shot fence creation failed (device={:#x}, command_buffer={:#x})",
                    vkHandleValue(device),
                    vkHandleValue(cb));
            } else {
                context->setDebugObjectName(VK_OBJECT_TYPE_FENCE,
                                            fence,
                                            "mesh.transfer.fence");
            }
            if (r == VK_SUCCESS &&
                (graphics_queue == VK_NULL_HANDLE || cb == VK_NULL_HANDLE ||
                 fence == VK_NULL_HANDLE || submit.commandBufferCount != 1 ||
                 submit.pCommandBuffers == nullptr || submit.pCommandBuffers[0] != cb)) {
                r = VK_ERROR_INITIALIZATION_FAILED;
                failed_expression = "mesh one-shot submit integrity check";
                failed_context = std::format(
                    "Mesh one-shot submit requires a non-null queue, one expected command buffer, and a non-null fence (queue={:#x}, command_buffer={:#x}, fence={:#x}, command_buffer_count={}, command_buffer_array={:#x}, submitted_command_buffer={:#x})",
                    vkHandleValue(graphics_queue),
                    vkHandleValue(cb),
                    vkHandleValue(fence),
                    submit.commandBufferCount,
                    reinterpret_cast<std::uintptr_t>(submit.pCommandBuffers),
                    submit.pCommandBuffers != nullptr ? vkHandleValue(submit.pCommandBuffers[0]) : 0);
            }
            if (r == VK_SUCCESS) {
                r = vkQueueSubmit(graphics_queue, 1, &submit, fence);
                if (r != VK_SUCCESS) {
                    failed_expression = "vkQueueSubmit(graphics_queue, 1, &submit, fence)";
                    failed_context = std::format(
                        "Mesh one-shot submission failed (queue={:#x}, command_buffer={:#x}, command_buffer_count=1, wait_semaphore_count=0, signal_semaphore_count=0, fence={:#x})",
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
                        "Mesh one-shot submission did not retire (device={:#x}, fence={:#x}, command_buffer={:#x}, fence_count=1)",
                        vkHandleValue(device),
                        vkHandleValue(fence),
                        vkHandleValue(cb));
                }
            }
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, fence, nullptr);
            }
            vkFreeCommandBuffers(device, transfer_pool, 1, &cb);
            if (r != VK_SUCCESS) {
                return reportVkFailure(
                    failed_expression,
                    r,
                    failed_context);
            }
            return true;
        }

        void destroy() {
            for (auto& [_, gpu] : mesh_cache) {
                destroyMesh(gpu);
            }
            mesh_cache.clear();
            destroyTexture(white_pixel);
            destroyShadow(shadow_dummy);
            for (auto& frame : frame_light_resources) {
                for (auto& draw : frame.draws) {
                    if (draw.buffer != VK_NULL_HANDLE) {
                        vmaDestroyBuffer(allocator, draw.buffer, draw.allocation);
                    }
                }
                if (frame.descriptor_pool != VK_NULL_HANDLE) {
                    vkDestroyDescriptorPool(device, frame.descriptor_pool, nullptr);
                }
            }
            frame_light_resources.clear();
            for (const VkDescriptorPool pool : material_descriptor_pools) {
                vkDestroyDescriptorPool(device, pool, nullptr);
            }
            material_descriptor_pools.clear();
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
            LFS_VK_CHECK_MSG(vkCreateSampler(device, &info, nullptr, &sampler),
                             "Mesh material sampler creation failed (device={:#x}, mag_filter={}, min_filter={}, address_mode={})",
                             vkHandleValue(device),
                             static_cast<int>(info.magFilter),
                             static_cast<int>(info.minFilter),
                             static_cast<int>(info.addressModeU));
            context->setDebugObjectName(VK_OBJECT_TYPE_SAMPLER,
                                        sampler,
                                        "mesh.material.sampler");

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
            LFS_VK_CHECK_MSG(vkCreateSampler(device, &shadow_info, nullptr, &shadow_sampler),
                             "Mesh shadow sampler creation failed (device={:#x}, compare_enable={}, compare_op={}, address_mode={})",
                             vkHandleValue(device),
                             shadow_info.compareEnable == VK_TRUE,
                             static_cast<int>(shadow_info.compareOp),
                             static_cast<int>(shadow_info.addressModeU));
            context->setDebugObjectName(VK_OBJECT_TYPE_SAMPLER,
                                        shadow_sampler,
                                        "mesh.shadow.sampler");
            return true;
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
            LFS_VK_CHECK_MSG(vkCreateDescriptorSetLayout(device, &light_info, nullptr, &light_layout),
                             "Mesh light descriptor-set layout creation failed (device={:#x}, binding_count={})",
                             vkHandleValue(device),
                             light_info.bindingCount);
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                        light_layout,
                                        "mesh.light.descriptor.layout");

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
            LFS_VK_CHECK_MSG(vkCreateDescriptorSetLayout(device, &mat_info, nullptr, &material_layout),
                             "Mesh material descriptor-set layout creation failed (device={:#x}, binding_count={})",
                             vkHandleValue(device),
                             mat_info.bindingCount);
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                        material_layout,
                                        "mesh.material.descriptor.layout");
            return true;
        }

        [[nodiscard]] VkDescriptorPool createMaterialDescriptorPool() const {
            constexpr std::uint32_t kMaxMaterials = 256;
            std::array<VkDescriptorPoolSize, 2> sizes{};
            sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            sizes[0].descriptorCount = kMaxMaterials;
            sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            sizes[1].descriptorCount = kMaxMaterials * 3;
            VkDescriptorPoolCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
            info.maxSets = kMaxMaterials;
            info.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
            info.pPoolSizes = sizes.data();
            VkDescriptorPool pool = VK_NULL_HANDLE;
            const VkResult result = vkCreateDescriptorPool(device, &info, nullptr, &pool);
            if (result != VK_SUCCESS) {
                LOG_ERROR("Vulkan: {}",
                          formatVkCheckFailure(
                              "vkCreateDescriptorPool(device, &info, nullptr, &pool)",
                              result,
                              std::format("Mesh material descriptor-pool creation failed (device={:#x}, max_sets={}, pool_size_count={}, uniform_descriptor_count={}, image_descriptor_count={})",
                                          vkHandleValue(device),
                                          info.maxSets,
                                          info.poolSizeCount,
                                          sizes[0].descriptorCount,
                                          sizes[1].descriptorCount),
                              __FILE__,
                              __LINE__));
                return VK_NULL_HANDLE;
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                         pool,
                                         "mesh.material.descriptor.pool[{}]",
                                         material_descriptor_pools.size());
            return pool;
        }

        bool createInitialMaterialDescriptorPool() {
            const VkDescriptorPool pool = createMaterialDescriptorPool();
            if (pool == VK_NULL_HANDLE) {
                return false;
            }
            material_descriptor_pools.push_back(pool);
            return true;
        }

        bool createFrameLightResources() {
            if (context == nullptr) {
                return false;
            }
            frame_light_resources.resize(std::max<std::size_t>(1, context->framesInFlight()));
            return true;
        }

        [[nodiscard]] FrameLightResources& lightResourcesForFrame(const std::size_t frame_slot) {
            if (frame_slot >= frame_light_resources.size()) [[unlikely]] {
                throw std::logic_error(std::format(
                    "Mesh frame slot is outside the draw-resource ring (frame_slot={}, ring_size={}) ({}:{})",
                    frame_slot,
                    frame_light_resources.size(),
                    __FILE__,
                    __LINE__));
            }
            return frame_light_resources[frame_slot];
        }

        bool createDummyShadow() {
            return createShadowTarget(1, shadow_dummy);
        }

        bool createShadowTarget(int resolution, ShadowTarget& out) {
            if (resolution <= 0) {
                return logVkFailure(std::format(
                    "Mesh shadow target resolution must be positive (observed_resolution={}, shadow_format={}, target_address={:#x}) ({}:{})",
                    resolution,
                    static_cast<int>(shadow_format),
                    reinterpret_cast<std::uintptr_t>(&out),
                    __FILE__,
                    __LINE__));
            }
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
            VmaAllocationInfo allocation_info{};
            LFS_VK_CHECK_MSG(
                vmaCreateImage(allocator, &img, &a, &out.image, &out.alloc, &allocation_info),
                "Mesh shadow image allocation failed (allocator={:#x}, target_address={:#x}, requested_resolution={}, format={}, usage={:#x})",
                reinterpret_cast<std::uintptr_t>(allocator),
                reinterpret_cast<std::uintptr_t>(&out),
                resolution,
                static_cast<int>(img.format),
                static_cast<std::uint32_t>(img.usage));
            const bool is_dummy = &out == &shadow_dummy;
            if (is_dummy) {
                context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE,
                                             out.image,
                                             "mesh.shadow.dummy.depth[{}]",
                                             resolution);
            } else {
                context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE,
                                             out.image,
                                             "mesh.shadow.depth[{}]",
                                             resolution);
            }
            out.vram_label = std::format("shadow:{}x{}@{}", resolution, resolution, static_cast<const void*>(&out));
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.mesh.shadow_image",
                out.vram_label,
                static_cast<std::size_t>(allocation_info.size));
            VkImageViewCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vi.image = out.image;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = shadow_format;
            vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            vi.subresourceRange.levelCount = 1;
            vi.subresourceRange.layerCount = 1;
            const VkResult view_result = vkCreateImageView(device, &vi, nullptr, &out.view);
            if (view_result != VK_SUCCESS) {
                destroyShadow(out);
                return reportVkFailure(
                    "vkCreateImageView(device, &vi, nullptr, &out.view)",
                    view_result,
                    std::format("Mesh shadow image-view creation failed (device={:#x}, image={:#x}, target_address={:#x}, resolution={}, format={}, aspect_mask={:#x})",
                                vkHandleValue(device),
                                vkHandleValue(vi.image),
                                reinterpret_cast<std::uintptr_t>(&out),
                                resolution,
                                static_cast<int>(vi.format),
                                static_cast<std::uint32_t>(vi.subresourceRange.aspectMask)));
            }
            if (is_dummy) {
                context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW,
                                             out.view,
                                             "mesh.shadow.dummy.depth[{}].view",
                                             resolution);
            } else {
                context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW,
                                             out.view,
                                             "mesh.shadow.depth[{}].view",
                                             resolution);
            }
            out.resolution = resolution;

            // Initial transition UNDEFINED → SHADER_READ_ONLY so binding it without a
            // shadow render is valid (e.g. when shadows are disabled).
            VkCommandBuffer cb = beginSingleTimeCommands();
            if (cb == VK_NULL_HANDLE) {
                destroyShadow(out);
                return false;
            }
            cmdImageBarrier2(cb, out.image, context->depthStencilAspectMask(),
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
            if (!endSingleTimeCommands(cb)) {
                destroyShadow(out);
                return false;
            }
            return true;
        }

        void destroyShadow(ShadowTarget& t) const {
            if (t.view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, t.view, nullptr);
            }
            if (t.image != VK_NULL_HANDLE) {
                if (!t.vram_label.empty()) {
                    lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                        "vulkan.mesh.shadow_image",
                        t.vram_label,
                        0);
                }
                vmaDestroyImage(allocator, t.image, t.alloc);
            }
            t = {};
        }

        bool writeBuffer(VmaAllocation alloc, const void* src, std::size_t bytes) const {
            void* mapped = nullptr;
            const VkResult map_result = vmaMapMemory(allocator, alloc, &mapped);
            if (map_result != VK_SUCCESS) {
                return reportVkFailure(
                    "vmaMapMemory(allocator, alloc, &mapped)",
                    map_result,
                    std::format("Mesh buffer allocation could not be mapped (allocator={:#x}, allocation={:#x}, source={:#x}, write_size={})",
                                reinterpret_cast<std::uintptr_t>(allocator),
                                reinterpret_cast<std::uintptr_t>(alloc),
                                reinterpret_cast<std::uintptr_t>(src),
                                bytes));
            }
            if (mapped == nullptr || src == nullptr || bytes == 0) {
                if (mapped != nullptr) {
                    vmaUnmapMemory(allocator, alloc);
                }
                return logVkFailure(std::format(
                    "Mesh buffer write requires mapped memory, a source pointer, and non-zero size (allocator={:#x}, allocation={:#x}, mapped={:#x}, source={:#x}, write_size={}) ({}:{})",
                    reinterpret_cast<std::uintptr_t>(allocator),
                    reinterpret_cast<std::uintptr_t>(alloc),
                    reinterpret_cast<std::uintptr_t>(mapped),
                    reinterpret_cast<std::uintptr_t>(src),
                    bytes,
                    __FILE__,
                    __LINE__));
            }
            std::memcpy(mapped, src, bytes);
            const VkResult flush_result = vmaFlushAllocation(allocator, alloc, 0, bytes);
            vmaUnmapMemory(allocator, alloc);
            if (flush_result != VK_SUCCESS) {
                return reportVkFailure(
                    "vmaFlushAllocation(allocator, alloc, 0, bytes)",
                    flush_result,
                    std::format("Mesh buffer flush failed (allocator={:#x}, allocation={:#x}, offset=0, flush_size={})",
                                reinterpret_cast<std::uintptr_t>(allocator),
                                reinterpret_cast<std::uintptr_t>(alloc),
                                bytes));
            }
            return true;
        }

        bool allocateMaterialDescriptor(GpuMaterial& material) {
            VkDescriptorSetAllocateInfo alloc{};
            alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc.descriptorSetCount = 1;
            alloc.pSetLayouts = &material_layout;
            for (const VkDescriptorPool pool : material_descriptor_pools) {
                alloc.descriptorPool = pool;
                const VkResult allocation_result =
                    vkAllocateDescriptorSets(device, &alloc, &material.descriptor);
                if (allocation_result == VK_SUCCESS) {
                    material.descriptor_pool = pool;
                    context->setDebugObjectNamef(VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                                 material.descriptor,
                                                 "mesh.material.descriptor[{}]",
                                                 vkHandleValue(material.descriptor));
                    return true;
                }
                if (allocation_result != VK_ERROR_OUT_OF_POOL_MEMORY &&
                    allocation_result != VK_ERROR_FRAGMENTED_POOL) {
                    LOG_ERROR("Vulkan: {}",
                              formatVkCheckFailure(
                                  "vkAllocateDescriptorSets(device, &alloc, &material.descriptor)",
                                  allocation_result,
                                  std::format("Mesh material descriptor allocation failed for a reusable pool (device={:#x}, descriptor_pool={:#x}, descriptor_layout={:#x}, requested_count=1)",
                                              vkHandleValue(device),
                                              vkHandleValue(pool),
                                              vkHandleValue(material_layout)),
                                  __FILE__,
                                  __LINE__));
                    return false;
                }
            }

            const VkDescriptorPool pool = createMaterialDescriptorPool();
            if (pool == VK_NULL_HANDLE) {
                return false;
            }
            material_descriptor_pools.push_back(pool);
            alloc.descriptorPool = pool;
            const VkResult allocation_result =
                vkAllocateDescriptorSets(device, &alloc, &material.descriptor);
            if (allocation_result != VK_SUCCESS) {
                vkDestroyDescriptorPool(device, pool, nullptr);
                material_descriptor_pools.pop_back();
                return reportVkFailure(
                    "vkAllocateDescriptorSets(device, &alloc, &material.descriptor)",
                    allocation_result,
                    std::format("Mesh material descriptor allocation failed for a new pool (device={:#x}, descriptor_pool={:#x}, descriptor_layout={:#x}, requested_count=1)",
                                vkHandleValue(device),
                                vkHandleValue(pool),
                                vkHandleValue(material_layout)));
            }
            material.descriptor_pool = pool;
            context->setDebugObjectNamef(VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                         material.descriptor,
                                         "mesh.material.descriptor[{}]",
                                         vkHandleValue(material.descriptor));
            return true;
        }

        bool ensureLightDrawCapacity(const std::size_t frame_slot, const std::size_t required) {
            if (required == 0) {
                return true;
            }
            if (required > std::numeric_limits<std::uint32_t>::max()) {
                return logVkFailure(std::format(
                    "Mesh light draw count must fit a Vulkan descriptor count (frame_slot={}, required_count={}, maximum_count={}) ({}:{})",
                    frame_slot,
                    required,
                    std::numeric_limits<std::uint32_t>::max(),
                    __FILE__,
                    __LINE__));
            }
            auto& frame = lightResourcesForFrame(frame_slot);
            if (frame.descriptor_pool != VK_NULL_HANDLE &&
                frame.descriptor_capacity >= required) {
                return true;
            }

            std::size_t capacity = std::max<std::size_t>(16, frame.descriptor_capacity);
            while (capacity < required) {
                capacity = std::min<std::size_t>(capacity * 2,
                                                 std::numeric_limits<std::uint32_t>::max());
            }

            VkDescriptorPoolSize pool_sizes[] = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<std::uint32_t>(capacity)},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<std::uint32_t>(capacity)},
            };
            VkDescriptorPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = static_cast<std::uint32_t>(capacity);
            pool_info.poolSizeCount = static_cast<std::uint32_t>(std::size(pool_sizes));
            pool_info.pPoolSizes = pool_sizes;
            VkDescriptorPool new_pool = VK_NULL_HANDLE;
            LFS_VK_CHECK_MSG(vkCreateDescriptorPool(device, &pool_info, nullptr, &new_pool),
                             "Mesh light descriptor-pool creation failed (device={:#x}, frame_slot={}, required_count={}, capacity={}, max_sets={}, pool_size_count={})",
                             vkHandleValue(device),
                             frame_slot,
                             required,
                             capacity,
                             pool_info.maxSets,
                             pool_info.poolSizeCount);
            context->setDebugObjectNamef(VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                         new_pool,
                                         "mesh.light.descriptor.pool[{}]",
                                         frame_slot);

            frame.draws.resize(capacity);
            for (auto& draw : frame.draws) {
                if (draw.buffer != VK_NULL_HANDLE) {
                    continue;
                }
                VkBufferCreateInfo buffer_info{};
                buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                buffer_info.size = sizeof(LightUbo);
                buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                VmaAllocationCreateInfo allocation_info{};
                allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
                allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
                const VkResult buffer_result = vmaCreateBuffer(allocator,
                                                               &buffer_info,
                                                               &allocation_info,
                                                               &draw.buffer,
                                                               &draw.allocation,
                                                               nullptr);
                if (buffer_result != VK_SUCCESS) {
                    vkDestroyDescriptorPool(device, new_pool, nullptr);
                    return reportVkFailure(
                        "vmaCreateBuffer(allocator, &buffer_info, &allocation_info, &draw.buffer, &draw.allocation, nullptr)",
                        buffer_result,
                        std::format("Mesh light UBO allocation failed (allocator={:#x}, frame_slot={}, required_count={}, capacity={}, draw_index={}, requested_size={})",
                                    reinterpret_cast<std::uintptr_t>(allocator),
                                    frame_slot,
                                    required,
                                    capacity,
                                    static_cast<std::size_t>(&draw - frame.draws.data()),
                                    buffer_info.size));
                }
                context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                             draw.buffer,
                                             "mesh.light[{}].ubo[{}]",
                                             frame_slot,
                                             static_cast<std::size_t>(&draw - frame.draws.data()));
            }

            std::vector<VkDescriptorSetLayout> layouts(capacity, light_layout);
            std::vector<VkDescriptorSet> descriptors(capacity, VK_NULL_HANDLE);
            VkDescriptorSetAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = new_pool;
            alloc_info.descriptorSetCount = static_cast<std::uint32_t>(capacity);
            alloc_info.pSetLayouts = layouts.data();
            const VkResult descriptors_result =
                vkAllocateDescriptorSets(device, &alloc_info, descriptors.data());
            if (descriptors_result != VK_SUCCESS) {
                vkDestroyDescriptorPool(device, new_pool, nullptr);
                return reportVkFailure(
                    "vkAllocateDescriptorSets(device, &alloc_info, descriptors.data())",
                    descriptors_result,
                    std::format("Mesh light descriptor-set allocation failed (device={:#x}, frame_slot={}, descriptor_pool={:#x}, descriptor_layout={:#x}, requested_count={})",
                                vkHandleValue(device),
                                frame_slot,
                                vkHandleValue(new_pool),
                                vkHandleValue(light_layout),
                                alloc_info.descriptorSetCount));
            }

            std::vector<VkDescriptorBufferInfo> buffer_infos(capacity);
            std::vector<VkWriteDescriptorSet> writes(capacity);
            for (std::size_t i = 0; i < capacity; ++i) {
                auto& draw = frame.draws[i];
                draw.descriptor = descriptors[i];
                context->setDebugObjectNamef(VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                             draw.descriptor,
                                             "mesh.light[{}].descriptor[{}]",
                                             frame_slot,
                                             i);
                buffer_infos[i].buffer = draw.buffer;
                buffer_infos[i].range = sizeof(LightUbo);
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = draw.descriptor;
                writes[i].dstBinding = 0;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writes[i].pBufferInfo = &buffer_infos[i];
            }
            vkUpdateDescriptorSets(device,
                                   static_cast<std::uint32_t>(writes.size()),
                                   writes.data(),
                                   0,
                                   nullptr);

            if (frame.descriptor_pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, frame.descriptor_pool, nullptr);
            }
            frame.descriptor_pool = new_pool;
            frame.descriptor_capacity = capacity;
            return true;
        }

        bool createTexture(const std::uint8_t* rgba,
                           int w,
                           int h,
                           GpuTexture& out,
                           std::string_view label = "texture") {
            if (rgba == nullptr || w <= 0 || h <= 0) {
                return logVkFailure(std::format(
                    "Mesh texture upload requires source pixels and positive dimensions (label='{}', source={:#x}, observed_width={}, observed_height={}) ({}:{})",
                    label,
                    reinterpret_cast<std::uintptr_t>(rgba),
                    w,
                    h,
                    __FILE__,
                    __LINE__));
            }
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
            VmaAllocationInfo allocation_info{};
            LFS_VK_CHECK_MSG(
                vmaCreateImage(allocator, &img, &a, &out.image, &out.alloc, &allocation_info),
                "Mesh texture image allocation failed (label='{}', allocator={:#x}, requested_extent={}x{}, format={}, usage={:#x})",
                label,
                reinterpret_cast<std::uintptr_t>(allocator),
                w,
                h,
                static_cast<int>(img.format),
                static_cast<std::uint32_t>(img.usage));
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE,
                                         out.image,
                                         "mesh.texture.{}[{}x{}]",
                                         label,
                                         w,
                                         h);
            out.vram_label = std::format("{}:{}x{}@{}", label, w, h, static_cast<const void*>(&out));
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.mesh.texture",
                out.vram_label,
                static_cast<std::size_t>(allocation_info.size));

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
            VkResult result =
                vmaCreateBuffer(allocator, &sb, &sa, &staging, &staging_alloc, nullptr);
            if (result != VK_SUCCESS) {
                destroyTexture(out);
                return reportVkFailure(
                    "vmaCreateBuffer(allocator, &sb, &sa, &staging, &staging_alloc, nullptr)",
                    result,
                    std::format("Mesh texture staging-buffer allocation failed (label='{}', allocator={:#x}, requested_size={}, usage={:#x})",
                                label,
                                reinterpret_cast<std::uintptr_t>(allocator),
                                bytes,
                                static_cast<std::uint32_t>(sb.usage)));
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                         staging,
                                         "mesh.texture.{}.upload.staging[{}]",
                                         label,
                                         bytes);
            void* mapped = nullptr;
            result = vmaMapMemory(allocator, staging_alloc, &mapped);
            if (result != VK_SUCCESS) {
                vmaDestroyBuffer(allocator, staging, staging_alloc);
                destroyTexture(out);
                return reportVkFailure(
                    "vmaMapMemory(allocator, staging_alloc, &mapped)",
                    result,
                    std::format("Mesh texture staging allocation could not be mapped (label='{}', allocator={:#x}, allocation={:#x}, buffer={:#x}, requested_size={})",
                                label,
                                reinterpret_cast<std::uintptr_t>(allocator),
                                reinterpret_cast<std::uintptr_t>(staging_alloc),
                                vkHandleValue(staging),
                                bytes));
            }
            std::memcpy(mapped, rgba, static_cast<std::size_t>(bytes));
            const VkResult flush_result = vmaFlushAllocation(allocator, staging_alloc, 0, bytes);
            vmaUnmapMemory(allocator, staging_alloc);
            if (flush_result != VK_SUCCESS) {
                vmaDestroyBuffer(allocator, staging, staging_alloc);
                destroyTexture(out);
                return reportVkFailure(
                    "vmaFlushAllocation(allocator, staging_alloc, 0, bytes)",
                    flush_result,
                    std::format("Mesh texture staging flush failed (label='{}', allocator={:#x}, allocation={:#x}, buffer={:#x}, offset=0, flush_size={})",
                                label,
                                reinterpret_cast<std::uintptr_t>(allocator),
                                reinterpret_cast<std::uintptr_t>(staging_alloc),
                                vkHandleValue(staging),
                                bytes));
            }

            VkCommandBuffer cb = beginSingleTimeCommands();
            if (cb == VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, staging, staging_alloc);
                destroyTexture(out);
                return false;
            }

            // UNDEFINED → TRANSFER_DST_OPTIMAL
            cmdImageBarrier2(cb, out.image, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                             VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), 1};
            vkCmdCopyBufferToImage(cb, staging, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   1, &region);

            cmdImageBarrier2(cb, out.image, VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

            if (!endSingleTimeCommands(cb)) {
                vmaDestroyBuffer(allocator, staging, staging_alloc);
                destroyTexture(out);
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
            const VkResult view_result = vkCreateImageView(device, &vi, nullptr, &out.view);
            if (view_result != VK_SUCCESS) {
                destroyTexture(out);
                return reportVkFailure(
                    "vkCreateImageView(device, &vi, nullptr, &out.view)",
                    view_result,
                    std::format("Mesh texture image-view creation failed (label='{}', device={:#x}, image={:#x}, extent={}x{}, format={}, aspect_mask={:#x})",
                                label,
                                vkHandleValue(device),
                                vkHandleValue(vi.image),
                                w,
                                h,
                                static_cast<int>(vi.format),
                                static_cast<std::uint32_t>(vi.subresourceRange.aspectMask)));
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW,
                                         out.view,
                                         "mesh.texture.{}[{}x{}].view",
                                         label,
                                         w,
                                         h);
            return true;
        }

        bool createWhitePixel() {
            const std::uint8_t white[4] = {255, 255, 255, 255};
            return createTexture(white, 1, 1, white_pixel, "white_pixel");
        }

        void destroyTexture(GpuTexture& t) const {
            if (t.view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, t.view, nullptr);
            }
            if (t.image != VK_NULL_HANDLE) {
                if (!t.vram_label.empty()) {
                    lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                        "vulkan.mesh.texture",
                        t.vram_label,
                        0);
                }
                vmaDestroyImage(allocator, t.image, t.alloc);
            }
            t = {};
        }

        void destroyMaterial(GpuMaterial& m) const {
            if (m.descriptor != VK_NULL_HANDLE && m.descriptor_pool != VK_NULL_HANDLE) {
                const VkResult result =
                    vkFreeDescriptorSets(device, m.descriptor_pool, 1, &m.descriptor);
                if (result != VK_SUCCESS) {
                    LOG_ERROR("Vulkan: {}",
                              formatVkCheckFailure(
                                  "vkFreeDescriptorSets(device, m.descriptor_pool, 1, &m.descriptor)",
                                  result,
                                  std::format("Mesh material descriptor-set release failed (device={:#x}, descriptor_pool={:#x}, descriptor_set={:#x}, descriptor_count=1)",
                                              vkHandleValue(device),
                                              vkHandleValue(m.descriptor_pool),
                                              vkHandleValue(m.descriptor)),
                                  __FILE__,
                                  __LINE__));
                }
            }
            if (m.ubo != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, m.ubo, m.ubo_alloc);
            }
            destroyTexture(m.albedo);
            destroyTexture(m.normal);
            destroyTexture(m.metallic_roughness);
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

            VkShaderModule vert = createShaderModule(device, kMeshVertSpv, "Mesh");
            VkShaderModule frag = createShaderModule(device, kMeshFragSpv, "Mesh");
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
            const VkResult layout_result =
                vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout);
            if (layout_result != VK_SUCCESS) {
                vkDestroyShaderModule(device, vert, nullptr);
                vkDestroyShaderModule(device, frag, nullptr);
                return reportVkFailure(
                    "vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout)",
                    layout_result,
                    std::format("Mesh main pipeline-layout creation failed (device={:#x}, light_layout={:#x}, material_layout={:#x}, set_layout_count={}, push_constant_bytes={})",
                                vkHandleValue(device),
                                vkHandleValue(light_layout),
                                vkHandleValue(material_layout),
                                layout_info.setLayoutCount,
                                push.size));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                        pipeline_layout,
                                        "mesh.main.pipeline.layout");

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
                const VkResult result = vkCreateGraphicsPipelines(device,
                                                                  pipeline_cache,
                                                                  1,
                                                                  &pipeline_info,
                                                                  nullptr,
                                                                  &out);
                if (result != VK_SUCCESS) {
                    return reportVkFailure(
                        "vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline_info, nullptr, &out)",
                        result,
                        std::format("Mesh main graphics-pipeline creation failed (device={:#x}, pipeline_cache={:#x}, pipeline_layout={:#x}, cull_mode={:#x}, color_format={}, depth_format={})",
                                    vkHandleValue(device),
                                    vkHandleValue(pipeline_cache),
                                    vkHandleValue(pipeline_layout),
                                    static_cast<std::uint32_t>(cull),
                                    static_cast<int>(color_format),
                                    static_cast<int>(depth_format)));
                }
                return true;
            };

            const bool ok = build(VK_CULL_MODE_BACK_BIT, pipeline_cull) &&
                            build(VK_CULL_MODE_NONE, pipeline_no_cull);
            vkDestroyShaderModule(device, vert, nullptr);
            vkDestroyShaderModule(device, frag, nullptr);
            if (!ok) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE,
                                        pipeline_cull,
                                        "mesh.main.pipeline.cull");
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE,
                                        pipeline_no_cull,
                                        "mesh.main.pipeline.no_cull");
            return true;
        }

        bool createShadowPipeline(VkFormat depth_format) {
            using namespace viewport_shaders;
            VkShaderModule vert = createShaderModule(device, kMeshShadowVertSpv, "Mesh");
            VkShaderModule frag = createShaderModule(device, kMeshShadowFragSpv, "Mesh");
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
            const VkResult layout_result =
                vkCreatePipelineLayout(device, &layout_info, nullptr, &shadow_pipeline_layout);
            if (layout_result != VK_SUCCESS) {
                vkDestroyShaderModule(device, vert, nullptr);
                vkDestroyShaderModule(device, frag, nullptr);
                return reportVkFailure(
                    "vkCreatePipelineLayout(device, &layout_info, nullptr, &shadow_pipeline_layout)",
                    layout_result,
                    std::format("Mesh shadow pipeline-layout creation failed (device={:#x}, set_layout_count={}, push_constant_bytes={})",
                                vkHandleValue(device),
                                layout_info.setLayoutCount,
                                push.size));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                        shadow_pipeline_layout,
                                        "mesh.shadow.pipeline.layout");

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
            if (result != VK_SUCCESS) {
                return reportVkFailure(
                    "vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline_info, nullptr, &shadow_pipeline)",
                    result,
                    std::format("Mesh shadow graphics-pipeline creation failed (device={:#x}, pipeline_cache={:#x}, pipeline_layout={:#x}, depth_format={})",
                                vkHandleValue(device),
                                vkHandleValue(pipeline_cache),
                                vkHandleValue(shadow_pipeline_layout),
                                static_cast<int>(depth_format)));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE,
                                        shadow_pipeline,
                                        "mesh.shadow.pipeline");
            return true;
        }

        bool createWireframePipeline(VkFormat color_format, VkFormat depth_format) {
            if (context == nullptr || !context->hasFillModeNonSolid()) {
                // VK_POLYGON_MODE_LINE is optional. Keep the pass usable on
                // conformant devices that do not expose fillModeNonSolid; the UI
                // reports the same capability and does not offer the overlay.
                return true;
            }
            using namespace viewport_shaders;
            VkShaderModule vert = createShaderModule(device, kMeshWireframeVertSpv, "Mesh");
            VkShaderModule frag = createShaderModule(device, kMeshWireframeFragSpv, "Mesh");
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
            const VkResult layout_result =
                vkCreatePipelineLayout(device, &layout_info, nullptr, &wireframe_pipeline_layout);
            if (layout_result != VK_SUCCESS) {
                vkDestroyShaderModule(device, vert, nullptr);
                vkDestroyShaderModule(device, frag, nullptr);
                return reportVkFailure(
                    "vkCreatePipelineLayout(device, &layout_info, nullptr, &wireframe_pipeline_layout)",
                    layout_result,
                    std::format("Mesh wireframe pipeline-layout creation failed (device={:#x}, set_layout_count={}, push_constant_bytes={})",
                                vkHandleValue(device),
                                layout_info.setLayoutCount,
                                push.size));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                        wireframe_pipeline_layout,
                                        "mesh.wireframe.pipeline.layout");

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
            if (result != VK_SUCCESS) {
                return reportVkFailure(
                    "vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline_info, nullptr, &wireframe_pipeline)",
                    result,
                    std::format("Mesh wireframe graphics-pipeline creation failed (device={:#x}, pipeline_cache={:#x}, pipeline_layout={:#x}, color_format={}, depth_format={})",
                                vkHandleValue(device),
                                vkHandleValue(pipeline_cache),
                                vkHandleValue(wireframe_pipeline_layout),
                                static_cast<int>(color_format),
                                static_cast<int>(depth_format)));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE,
                                        wireframe_pipeline,
                                        "mesh.wireframe.pipeline");
            return true;
        }

        bool uploadTextureFromMesh(const lfs::core::MeshData& mesh,
                                   std::uint32_t tex_index,
                                   GpuTexture& out,
                                   std::string_view label) {
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
            return createTexture(rgba.data(),
                                 img.width,
                                 img.height,
                                 out,
                                 std::format("{}.tex{}", label, tex_index));
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
            LFS_VK_CHECK_MSG(
                vmaCreateBuffer(allocator, &b, &a, &out.ubo, &out.ubo_alloc, nullptr),
                "Mesh material UBO allocation failed (allocator={:#x}, material_index={}, requested_size={}, usage={:#x})",
                reinterpret_cast<std::uintptr_t>(allocator),
                material_index,
                b.size,
                static_cast<std::uint32_t>(b.usage));
            context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                         out.ubo,
                                         "mesh.material[{}].ubo",
                                         material_index);

            const bool has_albedo = uploadTextureFromMesh(mesh, mat.albedo_tex, out.albedo, "albedo");
            const bool has_normal = uploadTextureFromMesh(mesh, mat.normal_tex, out.normal, "normal");
            const bool has_mr = uploadTextureFromMesh(mesh, mat.metallic_roughness_tex, out.metallic_roughness, "metallic_roughness");
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
            ubo.vertex_color_flags[0] = has_vc ? 1.0f : 0.0f;
            if (!writeBuffer(out.ubo_alloc, &ubo, sizeof(ubo))) {
                destroyMaterial(out);
                return false;
            }

            // Allocate descriptor set
            if (!allocateMaterialDescriptor(out)) {
                LOG_ERROR("Mesh material descriptor allocation failed after pool growth (material_index={}, descriptor_pool_count={}, descriptor_layout={:#x}, device={:#x}) ({}:{})",
                          material_index,
                          material_descriptor_pools.size(),
                          vkHandleValue(material_layout),
                          vkHandleValue(device),
                          __FILE__,
                          __LINE__);
                destroyMaterial(out);
                return false;
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                         out.descriptor,
                                         "mesh.material[{}].descriptor",
                                         material_index);

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

        bool createDeviceLocalBuffer(VkDeviceSize size,
                                     VkBufferUsageFlags usage,
                                     VkBuffer& buffer,
                                     VmaAllocation& alloc) const {
            VkBufferCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            info.size = size;
            info.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo ai{};
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            LFS_VK_CHECK_MSG(vmaCreateBuffer(allocator, &info, &ai, &buffer, &alloc, nullptr),
                             "Mesh device-local buffer allocation failed (allocator={:#x}, requested_size={}, usage={:#x})",
                             reinterpret_cast<std::uintptr_t>(allocator),
                             size,
                             static_cast<std::uint32_t>(info.usage));
            return true;
        }

        bool createStagingBuffer(VkDeviceSize size,
                                 const void* data,
                                 VkBuffer& buffer,
                                 VmaAllocation& alloc) const {
            VkBufferCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            info.size = size;
            info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo ai{};
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            LFS_VK_CHECK_MSG(vmaCreateBuffer(allocator, &info, &ai, &buffer, &alloc, nullptr),
                             "Mesh staging-buffer allocation failed (allocator={:#x}, source={:#x}, requested_size={}, usage={:#x})",
                             reinterpret_cast<std::uintptr_t>(allocator),
                             reinterpret_cast<std::uintptr_t>(data),
                             size,
                             static_cast<std::uint32_t>(info.usage));
            if (!writeBuffer(alloc, data, static_cast<std::size_t>(size))) {
                vmaDestroyBuffer(allocator, buffer, alloc);
                buffer = VK_NULL_HANDLE;
                alloc = VK_NULL_HANDLE;
                return false;
            }
            return true;
        }

        bool uploadMesh(const lfs::core::MeshData& mesh, GpuMesh& destination) {
            const std::int64_t vcount = mesh.vertex_count();
            const std::int64_t fcount = mesh.face_count();
            if (vcount <= 0 || fcount <= 0) {
                return logVkFailure(std::format(
                    "Mesh upload requires positive vertex and face counts (observed_vertices={}, observed_faces={}, generation={}) ({}:{})",
                    vcount,
                    fcount,
                    mesh.generation(),
                    __FILE__,
                    __LINE__));
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

            GpuMesh gpu{};

            const std::size_t vbytes = vertices.size() * sizeof(MeshVertex);
            const std::size_t ibytes = indices.size() * sizeof(std::uint32_t);
            VkBuffer vertex_staging = VK_NULL_HANDLE;
            VmaAllocation vertex_staging_alloc = VK_NULL_HANDLE;
            VkBuffer index_staging = VK_NULL_HANDLE;
            VmaAllocation index_staging_alloc = VK_NULL_HANDLE;
            if (!createDeviceLocalBuffer(vbytes,
                                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                         gpu.vertex_buffer,
                                         gpu.vertex_alloc) ||
                !createDeviceLocalBuffer(ibytes,
                                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                         gpu.index_buffer,
                                         gpu.index_alloc) ||
                !createStagingBuffer(vbytes,
                                     vertices.data(),
                                     vertex_staging,
                                     vertex_staging_alloc) ||
                !createStagingBuffer(ibytes,
                                     indices.data(),
                                     index_staging,
                                     index_staging_alloc)) {
                if (vertex_staging != VK_NULL_HANDLE) {
                    vmaDestroyBuffer(allocator, vertex_staging, vertex_staging_alloc);
                }
                if (index_staging != VK_NULL_HANDLE) {
                    vmaDestroyBuffer(allocator, index_staging, index_staging_alloc);
                }
                destroyMesh(gpu);
                return false;
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                         gpu.vertex_buffer,
                                         "mesh.geometry.vertex[{}]",
                                         vbytes);
            context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                         gpu.index_buffer,
                                         "mesh.geometry.index[{}]",
                                         ibytes);
            context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                         vertex_staging,
                                         "mesh.geometry.vertex.upload.staging[{}]",
                                         vbytes);
            context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                         index_staging,
                                         "mesh.geometry.index.upload.staging[{}]",
                                         ibytes);

            VkCommandBuffer upload_commands = beginSingleTimeCommands();
            if (upload_commands == VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, vertex_staging, vertex_staging_alloc);
                vmaDestroyBuffer(allocator, index_staging, index_staging_alloc);
                destroyMesh(gpu);
                return false;
            }
            VkBufferCopy vertex_copy{};
            vertex_copy.size = static_cast<VkDeviceSize>(vbytes);
            VkBufferCopy index_copy{};
            index_copy.size = static_cast<VkDeviceSize>(ibytes);
            if (vertex_staging == VK_NULL_HANDLE || gpu.vertex_buffer == VK_NULL_HANDLE ||
                index_staging == VK_NULL_HANDLE || gpu.index_buffer == VK_NULL_HANDLE ||
                vertex_copy.size == 0 || vertex_copy.size > vbytes || index_copy.size == 0 ||
                index_copy.size > ibytes) {
                const std::string error = std::format(
                    "Mesh upload copies require non-null buffers and ranges within their allocations (vertex_src={:#x}, vertex_dst={:#x}, vertex_copy_size={}, vertex_allocation_size={}, index_src={:#x}, index_dst={:#x}, index_copy_size={}, index_allocation_size={}) ({}:{})",
                    vkHandleValue(vertex_staging),
                    vkHandleValue(gpu.vertex_buffer),
                    vertex_copy.size,
                    vbytes,
                    vkHandleValue(index_staging),
                    vkHandleValue(gpu.index_buffer),
                    index_copy.size,
                    ibytes,
                    __FILE__,
                    __LINE__);
                vmaDestroyBuffer(allocator, vertex_staging, vertex_staging_alloc);
                vmaDestroyBuffer(allocator, index_staging, index_staging_alloc);
                destroyMesh(gpu);
                return logVkFailure(error);
            }
            vkCmdCopyBuffer(upload_commands, vertex_staging, gpu.vertex_buffer, 1, &vertex_copy);
            vkCmdCopyBuffer(upload_commands, index_staging, gpu.index_buffer, 1, &index_copy);

            std::array<VkBufferMemoryBarrier2, 2> barriers{};
            for (auto& barrier : barriers) {
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.offset = 0;
                barrier.size = VK_WHOLE_SIZE;
            }
            barriers[0].dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
            barriers[0].buffer = gpu.vertex_buffer;
            barriers[1].dstAccessMask = VK_ACCESS_2_INDEX_READ_BIT;
            barriers[1].buffer = gpu.index_buffer;
            VkDependencyInfo dependency{};
            dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dependency.bufferMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size());
            dependency.pBufferMemoryBarriers = barriers.data();
            vkCmdPipelineBarrier2(upload_commands, &dependency);

            const bool upload_ok = endSingleTimeCommands(upload_commands);
            vmaDestroyBuffer(allocator, vertex_staging, vertex_staging_alloc);
            vmaDestroyBuffer(allocator, index_staging, index_staging_alloc);
            if (!upload_ok) {
                destroyMesh(gpu);
                return false;
            }
            gpu.total_index_count = static_cast<std::uint32_t>(total_indices);

            // Materials
            const std::size_t mat_count = std::max<std::size_t>(mesh.materials.size(), 1);
            gpu.materials.resize(mat_count);
            for (std::size_t i = 0; i < mat_count; ++i) {
                if (!uploadMaterial(mesh, i, gpu.materials[i])) {
                    LOG_ERROR("VulkanMeshPass: failed to upload material {} for mesh", i);
                    destroyMesh(gpu);
                    return false;
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
            destroyMesh(destination);
            destination = std::move(gpu);
            return true;
        }

        bool writeLightUbo(LightDrawResources& resources,
                           const VulkanMeshPassParams& params,
                           const VulkanMeshDrawItem& item,
                           const glm::mat4& light_vp,
                           bool shadow_enabled) const {
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
            ubo.selection[0] = item.is_emphasized ? 1.0f : 0.0f;
            ubo.selection[1] = item.dim_non_emphasized ? 1.0f : 0.0f;
            ubo.selection[2] = item.flash_intensity;
            ubo.selection[3] = 0.0f;
            std::memcpy(ubo.light_vp, &light_vp[0][0], sizeof(ubo.light_vp));
            return writeBuffer(resources.allocation, &ubo, sizeof(ubo));
        }

        void bindShadowMap(const LightDrawResources& resources, const ShadowTarget& target) const {
            VkDescriptorImageInfo info{};
            info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            info.imageView = target.view != VK_NULL_HANDLE ? target.view : shadow_dummy.view;
            info.sampler = shadow_sampler;
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = resources.descriptor;
            w.dstBinding = 1;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &info;
            vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
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
            ShadowTarget replacement{};
            if (!createShadowTarget(resolution, replacement)) {
                return false;
            }
            // The initialization submit above is ordered after all prior graphics work
            // and waited to completion, so the old target is no longer referenced.
            destroyShadow(gpu.shadow);
            gpu.shadow = std::move(replacement);
            gpu.cached_light_vp_valid = false;
            return true;
        }

        bool recordShadowPass(GpuMesh& gpu, const glm::mat4& light_mvp) {
            if (gpu.shadow.image == VK_NULL_HANDLE || shadow_pipeline == VK_NULL_HANDLE) {
                return false;
            }
            VkCommandBuffer cb = beginSingleTimeCommands();
            if (cb == VK_NULL_HANDLE) {
                return false;
            }

            cmdImageBarrier2(cb, gpu.shadow.image, context->depthStencilAspectMask(),
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                             VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

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

            cmdImageBarrier2(cb, gpu.shadow.image, context->depthStencilAspectMask(),
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

            return endSingleTimeCommands(cb);
        }

        void prepare(const VulkanMeshPassParams& params) {
            ++frame_counter;
            if (!params.items.empty() &&
                params.draw_group_count > std::numeric_limits<std::size_t>::max() / params.items.size()) {
                LOG_ERROR("VulkanMeshPass: draw resource count overflow");
                return;
            }
            const std::size_t required_draw_resources = params.items.size() * params.draw_group_count;
            if (!ensureLightDrawCapacity(params.frame_slot, required_draw_resources)) {
                LOG_ERROR("VulkanMeshPass: failed to allocate {} frame-local draw resources",
                          required_draw_resources);
                return;
            }
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
                // recordShadowPass does a blocking GPU submit; only pay it when an input
                // the shadow map depends on actually changed. A static shadowed mesh then
                // renders its shadow once instead of every frame.
                const bool shadow_dirty =
                    !gpu.cached_light_vp_valid ||
                    gpu.shadow_key_generation != gpu.generation ||
                    gpu.shadow_key_resolution != item.shadow_map_resolution ||
                    gpu.shadow_key_model != item.model ||
                    gpu.shadow_key_light_dir != item.light_dir;
                if (!shadow_dirty) {
                    continue;
                }
                const glm::mat4 light_vp = computeLightVp(gpu, item.model, item.light_dir);
                if (!recordShadowPass(gpu, light_vp * item.model)) {
                    gpu.cached_light_vp_valid = false;
                    continue;
                }
                gpu.cached_light_vp = light_vp;
                gpu.cached_light_vp_valid = true;
                gpu.shadow_key_generation = gpu.generation;
                gpu.shadow_key_resolution = item.shadow_map_resolution;
                gpu.shadow_key_model = item.model;
                gpu.shadow_key_light_dir = item.light_dir;
            }

            constexpr std::uint64_t kEvictAfter = 120;
            bool has_stale_mesh = false;
            for (const auto& [_, gpu] : mesh_cache) {
                if (frame_counter - gpu.last_used_frame > kEvictAfter) {
                    has_stale_mesh = true;
                    break;
                }
            }
            bool submitted_frames_retired = !has_stale_mesh;
            if (has_stale_mesh) {
                if (context == nullptr) {
                    LOG_ERROR("VulkanMeshPass deferred stale mesh eviction because retirement cannot be proven (context={:#x}, frame_counter={}, cache_size={}, eviction_age={})",
                              vkHandleValue(context),
                              frame_counter,
                              mesh_cache.size(),
                              kEvictAfter);
                    return;
                }
                submitted_frames_retired = context->waitForSubmittedFrames();
                if (!submitted_frames_retired) {
                    LOG_WARN("VulkanMeshPass deferred stale mesh eviction because submitted frames did not retire (frame_counter={}, cache_size={}, eviction_age={}, error='{}')",
                             frame_counter,
                             mesh_cache.size(),
                             kEvictAfter,
                             context->lastError());
                    return;
                }
            }
            for (auto it = mesh_cache.begin(); it != mesh_cache.end();) {
                if (frame_counter - it->second.last_used_frame > kEvictAfter) {
                    LFS_VK_DEBUG_ASSERT(
                        submitted_frames_retired,
                        "Deferred mesh destruction requires all submitted frames to retire (frames_retired={}, frame_counter={}, last_used_frame={}, age={}, eviction_age={}, vertex_buffer={:#x}, index_buffer={:#x})",
                        submitted_frames_retired,
                        frame_counter,
                        it->second.last_used_frame,
                        frame_counter - it->second.last_used_frame,
                        kEvictAfter,
                        vkHandleValue(it->second.vertex_buffer),
                        vkHandleValue(it->second.index_buffer));
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
            if (cb == VK_NULL_HANDLE || params.frame_slot >= frame_light_resources.size()) [[unlikely]] {
                throw std::logic_error(std::format(
                    "Mesh recording requires a command buffer and in-range frame slot (command_buffer={:#x}, frame_slot={}, ring_size={}, item_count={}) ({}:{})",
                    vkHandleValue(cb),
                    params.frame_slot,
                    frame_light_resources.size(),
                    params.items.size(),
                    __FILE__,
                    __LINE__));
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

            if (params.draw_group >= params.draw_group_count ||
                params.draw_group > std::numeric_limits<std::size_t>::max() / params.items.size()) {
                LOG_ERROR("VulkanMeshPass: invalid draw group {} of {}",
                          params.draw_group,
                          params.draw_group_count);
                return;
            }
            const std::size_t draw_base = params.draw_group * params.items.size();
            auto& frame = lightResourcesForFrame(params.frame_slot);
            for (std::size_t item_index = 0; item_index < params.items.size(); ++item_index) {
                const auto& item = params.items[item_index];
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
                const std::size_t draw_index = draw_base + item_index;
                if (draw_index >= frame.draws.size()) {
                    throw std::logic_error(std::format(
                        "Mesh draw index is outside the retired frame's resource array (frame_slot={}, draw_index={}, draw_count={}, draw_base={}, item_index={}) ({}:{})",
                        params.frame_slot,
                        draw_index,
                        frame.draws.size(),
                        draw_base,
                        item_index,
                        __FILE__,
                        __LINE__));
                }
                auto& draw = frame.draws[draw_index];
                if (draw.descriptor == VK_NULL_HANDLE || draw.buffer == VK_NULL_HANDLE) [[unlikely]] {
                    throw std::logic_error(std::format(
                        "Mesh draw recording requires a non-null per-frame descriptor and UBO (frame_slot={}, draw_index={}, draw_count={}, descriptor_set={:#x}, ubo={:#x}, descriptor_capacity={}) ({}:{})",
                        params.frame_slot,
                        draw_index,
                        frame.draws.size(),
                        vkHandleValue(draw.descriptor),
                        vkHandleValue(draw.buffer),
                        frame.descriptor_capacity,
                        __FILE__,
                        __LINE__));
                }
                if (!writeLightUbo(draw,
                                   params,
                                   item,
                                   shadow_active ? gpu.cached_light_vp : glm::mat4(1.0f),
                                   shadow_active)) {
                    LOG_ERROR("VulkanMeshPass: failed to prepare light state for draw {}", draw_index);
                    continue;
                }
                bindShadowMap(draw, shadow_active ? gpu.shadow : shadow_dummy);

                VkPipeline main_pipeline = item.backface_culling ? pipeline_cull : pipeline_no_cull;
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, main_pipeline);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                        0, 1, &draw.descriptor, 0, nullptr);

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
                    if (mat.descriptor == VK_NULL_HANDLE) {
                        continue;
                    }
                    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                            1, 1, &mat.descriptor, 0, nullptr);
                    vkCmdDrawIndexed(cb, sm.index_count, 1, sm.start_index, 0, 0);
                }

                // Wireframe overlay drawn on top of the shaded mesh.
                if (item.wireframe_overlay && wireframe_pipeline != VK_NULL_HANDLE) {
                    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, wireframe_pipeline);
                    const float line_width = context->hasWideLines()
                                                 ? std::clamp(item.wireframe_width,
                                                              context->minLineWidth(),
                                                              context->maxLineWidth())
                                                 : 1.0f;
                    vkCmdSetLineWidth(cb, line_width);
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
