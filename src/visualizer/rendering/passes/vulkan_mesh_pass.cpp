/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_mesh_pass.hpp"
#include "rendering/viewport_geometry.hpp"
#include "shared_viewport_gpu_assets.hpp"

#include "core/collision.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "rendering/vulkan_wait.hpp"
#include "window/vulkan_barrier2.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_result.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <format>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <vk_mem_alloc.h>

#include "viewport/collision_overlay.frag.spv.h"
#include "viewport/collision_overlay.vert.spv.h"
#include "viewport/mesh.frag.spv.h"
#include "viewport/mesh.geom.spv.h"
#include "viewport/mesh.vert.spv.h"
#include "viewport/mesh_shadow.frag.spv.h"
#include "viewport/mesh_shadow.vert.spv.h"
#include "viewport/mesh_wireframe.frag.spv.h"
#include "viewport/mesh_wireframe.vert.spv.h"

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

        [[nodiscard]] constexpr bool hasStencilAspect(const VkFormat format) noexcept {
            return format == VK_FORMAT_D16_UNORM_S8_UINT ||
                   format == VK_FORMAT_D24_UNORM_S8_UINT ||
                   format == VK_FORMAT_D32_SFLOAT_S8_UINT;
        }

        [[nodiscard]] constexpr VkImageAspectFlags depthAspectMask(const VkFormat format) noexcept {
            VkImageAspectFlags aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (hasStencilAspect(format)) {
                aspects |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
            return aspects;
        }

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
            float light_dir[4];      // xyz, w unused
            float params[4];         // x = intensity, y = ambient, z = shadow_enabled
            float selection[4];      // x = emphasized, y = dim others, zw = reserved
            float light_vp[16];      // light view-projection (column-major) for shadow sampling
            float camera_forward[4]; // xyz = world-space view direction
            float depth_view[4];     // x = enabled, y = near, z = far, w = grayscale
            float view[16];
            float projection[16];
        };
        static_assert(sizeof(LightUbo) == 288, "LightUbo layout");

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
        struct CollisionPush {
            glm::mat4 clip_to_grid;
            glm::vec4 pixel{0}, panorama{0};
        };
        static_assert(sizeof(CollisionPush) == 96);

    } // namespace

    struct VulkanMeshPass::Impl {
        struct OneShotResult {
            bool completed = false;
            // A submitted command can only be cleaned up after Ready or a
            // failure-only device-idle containment pass.
            bool safe_to_release = true;
        };

        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache = VK_NULL_HANDLE;

        VkDescriptorSetLayout light_layout = VK_NULL_HANDLE;
        struct MaterialBinding {
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkDescriptorSet descriptor = VK_NULL_HANDLE;
        };
        struct MaterialBindings {
            VkDescriptorPool pool = VK_NULL_HANDLE;
            std::vector<MaterialBinding> materials;
            std::uint32_t generation = 0;
            std::uint64_t last_used = 0;
        };
        std::unordered_map<std::uint64_t, MaterialBindings> material_bindings;
        struct RetiredMaterials {
            MaterialBindings bindings;
            std::uint64_t consumer = 0;
        };
        std::vector<RetiredMaterials> retired_materials;

        void destroyMaterials(MaterialBindings& bindings) {
            if (bindings.pool != VK_NULL_HANDLE)
                vkDestroyDescriptorPool(device, bindings.pool, nullptr);
            for (auto& material : bindings.materials)
                if (material.buffer != VK_NULL_HANDLE)
                    vmaDestroyBuffer(allocator, material.buffer, material.allocation);
            bindings = {};
        }

        bool prepareMaterials(std::uint64_t id, const SharedMeshDrawAsset& mesh) {
            auto& current = material_bindings[id];
            current.last_used = context->lastFrameSubmitSerial();
            if (current.pool != VK_NULL_HANDLE && current.generation == mesh.generation)
                return true;
            MaterialBindings bindings;
            bindings.generation = mesh.generation;
            bindings.last_used = current.last_used;
            const auto count = static_cast<std::uint32_t>(mesh.materials.size());
            if (count == 0)
                return false;
            const std::array<VkDescriptorPoolSize, 2> sizes{{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, count},
                                                             {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, count * 3}}};
            VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pool.maxSets = count;
            pool.poolSizeCount = sizes.size();
            pool.pPoolSizes = sizes.data();
            if (!vk_try_bool(vkCreateDescriptorPool(device, &pool, nullptr, &bindings.pool),
                             "vkCreateDescriptorPool", "Mesh material descriptors"))
                return false;
            bindings.materials.resize(count);
            for (std::size_t i = 0; i < count; ++i) {
                auto& material = bindings.materials[i];
                VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                buffer.size = sizeof(mesh.materials[i].uniform);
                buffer.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                buffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                VmaAllocationCreateInfo allocation{};
                allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
                allocation.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
                if (!vk_try_bool(vmaCreateBuffer(allocator, &buffer, &allocation, &material.buffer,
                                                 &material.allocation, nullptr),
                                 "vmaCreateBuffer", "Mesh material uniform") ||
                    !writeBuffer(material.allocation, mesh.materials[i].uniform.data(), buffer.size)) {
                    destroyMaterials(bindings);
                    return false;
                }
                VkDescriptorSetAllocateInfo descriptor{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                descriptor.descriptorPool = bindings.pool;
                descriptor.descriptorSetCount = 1;
                descriptor.pSetLayouts = &material_layout;
                if (!vk_try_bool(vkAllocateDescriptorSets(device, &descriptor, &material.descriptor),
                                 "vkAllocateDescriptorSets", "Mesh material descriptor")) {
                    destroyMaterials(bindings);
                    return false;
                }
                VkDescriptorBufferInfo info{material.buffer, 0, buffer.size};
                std::array<VkWriteDescriptorSet, 4> writes{};
                for (std::size_t binding = 0; binding < writes.size(); ++binding) {
                    auto& write = writes[binding];
                    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    write.dstSet = material.descriptor;
                    write.dstBinding = binding;
                    write.descriptorCount = 1;
                    write.descriptorType = binding == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    if (binding == 0)
                        write.pBufferInfo = &info;
                    else
                        write.pImageInfo = &mesh.materials[i].textures[binding - 1];
                }
                vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
            }
            if (current.pool != VK_NULL_HANDLE)
                retired_materials.push_back({std::move(current), context->lastFrameSubmitSerial() + 1});
            current = std::move(bindings);
            return true;
        }

        VkDescriptorSetLayout material_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline_cull = VK_NULL_HANDLE;    // backface culling (default)
        VkPipeline pipeline_no_cull = VK_NULL_HANDLE; // double-sided / culling off
        VkPipeline pipeline_cull_blend = VK_NULL_HANDLE;
        VkPipeline pipeline_no_cull_blend = VK_NULL_HANDLE;

        struct TransparentSubmeshDraw {
            const SharedMeshDrawAsset* mesh = nullptr;
            const SharedMeshSubmesh* submesh = nullptr;
            VkDescriptorSet material = VK_NULL_HANDLE;
            std::size_t item_index = 0;
            std::size_t draw_index = 0;
            std::size_t submesh_order = 0;
            float view_depth = 0.0f;
        };
        std::vector<TransparentSubmeshDraw> transparent_draws;

        VkPipelineLayout shadow_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline shadow_pipeline = VK_NULL_HANDLE;
        VkFormat shadow_format = VK_FORMAT_UNDEFINED;

        VkPipelineLayout wireframe_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline wireframe_pipeline = VK_NULL_HANDLE;
        VkFormat color_format_cached = VK_FORMAT_UNDEFINED;
        VkFormat depth_format_cached = VK_FORMAT_UNDEFINED;
        VkDescriptorSetLayout collision_layout = VK_NULL_HANDLE;
        VkPipelineLayout collision_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline collision_pipeline = VK_NULL_HANDLE;
        bool collision_pipeline_attempted = false;
        struct CollisionFrame {
            VkDevice device = VK_NULL_HANDLE;
            VmaAllocator allocator = VK_NULL_HANDLE;
            std::shared_ptr<const core::CollisionVolume> owner;
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            VkDescriptorSet descriptor = VK_NULL_HANDLE;
            VkBuffer staging = VK_NULL_HANDLE;
            VmaAllocation staging_allocation = VK_NULL_HANDLE;
            VkFence pending_upload = VK_NULL_HANDLE;
            ~CollisionFrame() {
                if (pending_upload && vkGetFenceStatus(device, pending_upload) != VK_SUCCESS) {
                    LOG_ERROR("Collision upload has not retired; retaining its buffers until device teardown");
                    return;
                }
                if (pending_upload)
                    vkDestroyFence(device, pending_upload, nullptr);
                if (staging)
                    vmaDestroyBuffer(allocator, staging, staging_allocation);
                if (buffer)
                    vmaDestroyBuffer(allocator, buffer, allocation);
                if (pool)
                    vkDestroyDescriptorPool(device, pool, nullptr);
            }
        };
        std::vector<std::vector<std::shared_ptr<CollisionFrame>>> collision_frames;
        std::unordered_map<const core::CollisionVolume*, std::weak_ptr<CollisionFrame>> collision_cache;

        VkSampler shadow_sampler = VK_NULL_HANDLE; // sampler2DShadow with comparison
        VkCommandPool transfer_pool = VK_NULL_HANDLE;
        mutable bool retain_transfer_pool = false;
        VkQueue graphics_queue = VK_NULL_HANDLE;
        std::shared_ptr<SharedViewportGpuAssets> assets;

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

        struct ShadowTarget {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation alloc = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            int resolution = 0;
            std::string vram_label;
        };

        struct ShadowGroupState {
            ShadowTarget shadow{};
            glm::mat4 cached_light_vp{1.0f};
            bool cached_light_vp_valid = false;
            glm::mat4 shadow_key_model{0.0f};
            glm::vec3 shadow_key_light_dir{0.0f};
            int shadow_key_resolution = -1;
            std::uint32_t shadow_key_generation = std::numeric_limits<std::uint32_t>::max();
        };

        struct MeshShadowState {
            std::vector<ShadowGroupState> groups;
            std::uint64_t last_used_epoch = 0;
        };

        // Placeholder 1x1 shadow image bound when shadow_enabled=false; satisfies the
        // sampler2DShadow descriptor without the validation layer complaining.
        ShadowTarget shadow_dummy{};
        std::unordered_map<std::uint64_t, MeshShadowState> mesh_shadows;

        ~Impl() { destroy(); }

        bool init(VulkanContext& ctx,
                  VkFormat color_format,
                  VkFormat depth_format,
                  std::shared_ptr<SharedViewportGpuAssets> shared_assets) {
            if (!shared_assets) {
                shared_assets = std::make_shared<SharedViewportGpuAssets>();
            }
            assets = std::move(shared_assets);
            if (!assets->ensureContext(ctx)) {
                return logVkFailure(std::format(
                    "Mesh-pass initialization could not bind shared scene GPU assets ({}:{})",
                    __FILE__,
                    __LINE__));
            }
            context = &ctx;
            device = ctx.device();
            allocator = ctx.allocator();
            pipeline_cache = ctx.pipelineCache();
            graphics_queue = ctx.graphicsQueue();
            material_layout = assets->meshMaterialLayout();
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE ||
                graphics_queue == VK_NULL_HANDLE || material_layout == VK_NULL_HANDLE) {
                return logVkFailure(std::format(
                    "Mesh-pass initialization requires a live device, allocator, graphics queue, and shared material layout (device={:#x}, allocator={:#x}, graphics_queue={:#x}, pipeline_cache={:#x}, material_layout={:#x}) ({}:{})",
                    vkHandleValue(device),
                    reinterpret_cast<std::uintptr_t>(allocator),
                    vkHandleValue(graphics_queue),
                    vkHandleValue(pipeline_cache),
                    vkHandleValue(material_layout),
                    __FILE__,
                    __LINE__));
            }

            VkCommandPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool_info.queueFamilyIndex = ctx.graphicsQueueFamily();
            if (!vk_try_bool(
                    vkCreateCommandPool(device, &pool_info, nullptr, &transfer_pool),
                    "vkCreateCommandPool(device, &pool_info, nullptr, &transfer_pool)",
                    lfs::rendering::formatVulkanDiagnostic(
                        "Mesh transfer command-pool creation failed (device={:#x}, queue_family={}, flags={:#x})",
                        vkHandleValue(device),
                        pool_info.queueFamilyIndex,
                        static_cast<std::uint32_t>(pool_info.flags)),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_COMMAND_POOL,
                                        transfer_pool,
                                        "mesh.transfer.pool");

            color_format_cached = color_format;
            depth_format_cached = depth_format;
            shadow_format = depth_format; // re-use the swapchain's depth format

            return createShadowSampler() &&
                   createLightDescriptorLayout() &&
                   createFrameLightResources() &&
                   createDummyShadow() &&
                   createMainPipelines(color_format, depth_format) &&
                   createShadowPipeline(depth_format) &&
                   createOverlayPipeline(color_format, depth_format);
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

        OneShotResult endSingleTimeCommands(VkCommandBuffer cb, VkFence* pending = nullptr) const {
            VkResult r = vkEndCommandBuffer(cb);
            if (r != VK_SUCCESS) {
                vkFreeCommandBuffers(device, transfer_pool, 1, &cb);
                static_cast<void>(reportVkFailure(
                    "vkEndCommandBuffer(cb)",
                    r,
                    std::format("Mesh one-shot command buffer did not leave recording state (command_buffer={:#x}, command_pool={:#x})",
                                vkHandleValue(cb),
                                vkHandleValue(transfer_pool))));

                return {false, true};
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
            bool submitted = false;
            bool wait_ready = false;
            bool safe_to_release = true;
            if (r == VK_SUCCESS) {
                r = lfs::rendering::vk_queue_submit_synced(graphics_queue, 1, &submit, fence);
                if (r != VK_SUCCESS) {
                    failed_expression = "lfs::rendering::vk_queue_submit_synced(graphics_queue, 1, &submit, fence)";
                    failed_context = std::format(
                        "Mesh one-shot submission failed (queue={:#x}, command_buffer={:#x}, command_buffer_count=1, wait_semaphore_count=0, signal_semaphore_count=0, fence={:#x})",
                        vkHandleValue(graphics_queue),
                        vkHandleValue(cb),
                        vkHandleValue(fence));
                } else {
                    submitted = true;
                }
            }
            if (r == VK_SUCCESS) {
                lfs::rendering::WaitContext wait_ctx;
                wait_ctx.fingerprint = "pass.mesh.oneshot_wait";
                auto wait_outcome = lfs::rendering::wait_fence_bounded(
                    device,
                    fence,
                    std::stop_token{},
                    lfs::rendering::VulkanWaitPolicy{},
                    wait_ctx);
                if (wait_outcome.has_value() &&
                    *wait_outcome == lfs::rendering::WaitOutcome::Ready) {
                    wait_ready = true;
                } else {
                    r = VK_TIMEOUT;
                    failed_expression = "wait_fence_bounded(pass.mesh.oneshot_wait)";
                    failed_context = std::format(
                        "Mesh one-shot submission did not retire "
                        "(device={:#x}, fence={:#x}, command_buffer={:#x}): {}",
                        vkHandleValue(device),
                        vkHandleValue(fence),
                        vkHandleValue(cb),
                        formatWaitFailure(wait_outcome));
                    // A non-Ready wait leaves the transfer potentially using
                    // its destination. Contain only this failure path; normal
                    // successful uploads retain their existing synchronization.
                    if (context != nullptr && context->deviceWaitIdle()) {
                        wait_ready = true;
                        LOG_WARN("Vulkan: mesh one-shot wait required failure-only device idle before cleanup");
                    } else {
                        safe_to_release = false;
                        LOG_ERROR("Vulkan: mesh one-shot resources are abandoned after non-Ready wait and failed device idle containment");
                    }
                }
            }
            if (submitted && !wait_ready) {
                retain_transfer_pool = true;
                if (pending)
                    *pending = fence;
            }
            // AMB-4: destroy fence/CB only when never submitted or wait Ready.
            if (fence != VK_NULL_HANDLE) {
                if (!submitted || wait_ready) {
                    vkDestroyFence(device, fence, nullptr);
                } else {
                    LOG_ERROR(
                        "Vulkan: retaining mesh one-shot fence after non-Ready wait "
                        "(fence={:#x}, command_buffer={:#x})",
                        vkHandleValue(fence),
                        vkHandleValue(cb));
                }
            }
            if (!submitted || wait_ready) {
                vkFreeCommandBuffers(device, transfer_pool, 1, &cb);
            } else {
                LOG_ERROR(
                    "Vulkan: retaining mesh one-shot command buffer after non-Ready wait "
                    "(command_buffer={:#x})",
                    vkHandleValue(cb));
            }
            if (r != VK_SUCCESS) {
                static_cast<void>(reportVkFailure(
                    failed_expression,
                    r,
                    failed_context));
                return {false, safe_to_release};
            }
            return {true, safe_to_release};
        }

        void destroy() {
            collision_frames.clear();
            collision_cache.clear();
            for (auto& [_, shadow] : mesh_shadows) {
                for (auto& group : shadow.groups)
                    destroyShadow(group.shadow);
            }
            mesh_shadows.clear();
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
            for (VkPipeline* p : {&pipeline_cull, &pipeline_no_cull, &pipeline_cull_blend,
                                  &pipeline_no_cull_blend, &shadow_pipeline, &wireframe_pipeline, &collision_pipeline}) {
                if (*p != VK_NULL_HANDLE) {
                    vkDestroyPipeline(device, *p, nullptr);
                    *p = VK_NULL_HANDLE;
                }
            }
            for (VkPipelineLayout* l : {&pipeline_layout, &shadow_pipeline_layout, &wireframe_pipeline_layout, &collision_pipeline_layout}) {
                if (*l != VK_NULL_HANDLE) {
                    vkDestroyPipelineLayout(device, *l, nullptr);
                    *l = VK_NULL_HANDLE;
                }
            }
            for (auto& [id, bindings] : material_bindings)
                destroyMaterials(bindings);
            material_bindings.clear();
            for (auto& retired : retired_materials)
                destroyMaterials(retired.bindings);
            retired_materials.clear();
            material_layout = VK_NULL_HANDLE;
            if (light_layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, light_layout, nullptr);
                light_layout = VK_NULL_HANDLE;
            }
            if (collision_layout) {
                vkDestroyDescriptorSetLayout(device, collision_layout, nullptr);
                collision_layout = VK_NULL_HANDLE;
            }
            if (shadow_sampler != VK_NULL_HANDLE) {
                vkDestroySampler(device, shadow_sampler, nullptr);
                shadow_sampler = VK_NULL_HANDLE;
            }
            if (transfer_pool != VK_NULL_HANDLE && !retain_transfer_pool) {
                vkDestroyCommandPool(device, transfer_pool, nullptr);
                transfer_pool = VK_NULL_HANDLE;
            }
            device = VK_NULL_HANDLE;
            allocator = VK_NULL_HANDLE;
            assets.reset();
        }

        bool createShadowSampler() {
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
            if (!vk_try_bool(
                    vkCreateSampler(device, &shadow_info, nullptr, &shadow_sampler),
                    "vkCreateSampler(device, &shadow_info, nullptr, &shadow_sampler)",
                    lfs::rendering::formatVulkanDiagnostic(
                        "Mesh shadow sampler creation failed (device={:#x}, compare_enable={}, compare_op={}, address_mode={})",
                        vkHandleValue(device),
                        shadow_info.compareEnable == VK_TRUE,
                        static_cast<int>(shadow_info.compareOp),
                        static_cast<int>(shadow_info.addressModeU)),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_SAMPLER,
                                        shadow_sampler,
                                        "mesh.shadow.sampler");
            return true;
        }

        bool createLightDescriptorLayout() {
            // Set 0: light UBO + shadow map sampler.
            std::array<VkDescriptorSetLayoutBinding, 2> light_b{};
            light_b[0].binding = 0;
            light_b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            light_b[0].descriptorCount = 1;
            light_b[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            if (context->geometryShaderEnabled()) {
                light_b[0].stageFlags |= VK_SHADER_STAGE_GEOMETRY_BIT;
            }
            light_b[1].binding = 1;
            light_b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            light_b[1].descriptorCount = 1;
            light_b[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutCreateInfo light_info{};
            light_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            light_info.bindingCount = static_cast<std::uint32_t>(light_b.size());
            light_info.pBindings = light_b.data();
            if (!vk_try_bool(
                    vkCreateDescriptorSetLayout(device, &light_info, nullptr, &light_layout),
                    "vkCreateDescriptorSetLayout(device, &light_info, nullptr, &light_layout)",
                    lfs::rendering::formatVulkanDiagnostic(
                        "Mesh light descriptor-set layout creation failed (device={:#x}, binding_count={})",
                        vkHandleValue(device),
                        light_info.bindingCount),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                        light_layout,
                                        "mesh.light.descriptor.layout");
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
            if (!vk_try_bool(
                    vmaCreateImage(allocator, &img, &a, &out.image, &out.alloc, &allocation_info),
                    "vmaCreateImage(allocator, &img, &a, &out.image, &out.alloc, &allocation_info)",
                    lfs::rendering::formatVulkanDiagnostic(
                        "Mesh shadow image allocation failed (allocator={:#x}, target_address={:#x}, requested_resolution={}, format={}, usage={:#x})",
                        reinterpret_cast<std::uintptr_t>(allocator),
                        reinterpret_cast<std::uintptr_t>(&out),
                        resolution,
                        static_cast<int>(img.format),
                        static_cast<std::uint32_t>(img.usage)),
                    std::source_location::current())) {
                return false;
            }
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
            vmaSetAllocationName(allocator, out.alloc,
                                 is_dummy ? "Mesh shadow dummy depth" : "Mesh shadow depth");
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
            cmdImageBarrier2(cb, out.image, depthAspectMask(shadow_format),
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
            const auto transition_result = endSingleTimeCommands(cb);
            if (!transition_result.completed) {
                if (transition_result.safe_to_release) {
                    destroyShadow(out);
                } else {
                    // The abandoned command may still reference this image;
                    // detach it so later pass teardown cannot free it again.
                    abandonShadow(out);
                }
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

        void abandonShadow(ShadowTarget& t) const {
            if (t.image != VK_NULL_HANDLE && !t.vram_label.empty()) {
                lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                    "vulkan.mesh.shadow_image", t.vram_label, 0);
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
            if (!vk_try_bool(
                    vkCreateDescriptorPool(device, &pool_info, nullptr, &new_pool),
                    "vkCreateDescriptorPool(device, &pool_info, nullptr, &new_pool)",
                    lfs::rendering::formatVulkanDiagnostic(
                        "Mesh light descriptor-pool creation failed (device={:#x}, frame_slot={}, required_count={}, capacity={}, max_sets={}, pool_size_count={})",
                        vkHandleValue(device),
                        frame_slot,
                        required,
                        capacity,
                        pool_info.maxSets,
                        pool_info.poolSizeCount),
                    std::source_location::current())) {
                return false;
            }
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

        bool createMainPipelines(VkFormat color_format, VkFormat depth_format) {
            using namespace viewport_shaders;

            VkShaderModule vert = createShaderModule(device, kMeshVertSpv, "Mesh");
            VkShaderModule geom = context->geometryShaderEnabled()
                                      ? createShaderModule(device, kMeshGeomSpv, "Mesh")
                                      : VK_NULL_HANDLE;
            VkShaderModule frag = createShaderModule(device, kMeshFragSpv, "Mesh");
            if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE ||
                (context->geometryShaderEnabled() && geom == VK_NULL_HANDLE)) {
                if (vert)
                    vkDestroyShaderModule(device, vert, nullptr);
                if (geom)
                    vkDestroyShaderModule(device, geom, nullptr);
                if (frag)
                    vkDestroyShaderModule(device, frag, nullptr);
                LOG_ERROR("VulkanMeshPass: main shader module creation failed");
                return false;
            }

            std::array<VkPipelineShaderStageCreateInfo, 3> stages{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert;
            stages[0].pName = "main";
            std::uint32_t stage_count = 1;
            if (geom != VK_NULL_HANDLE) {
                stages[stage_count].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                stages[stage_count].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
                stages[stage_count].module = geom;
                stages[stage_count++].pName = "main";
            }
            stages[stage_count].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[stage_count].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[stage_count].module = frag;
            stages[stage_count++].pName = "main";

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
                if (geom)
                    vkDestroyShaderModule(device, geom, nullptr);
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
            rendering_info.stencilAttachmentFormat =
                hasStencilAspect(depth_format) ? depth_format : VK_FORMAT_UNDEFINED;

            // Build cull and no-cull variants by tweaking only the rasterization state.
            // Mesh vertices use the same GL-convention projection matrices as FastGS,
            // then receive a clip-space Y correction before Vulkan's positive-height
            // viewport. That preserves normal CCW winding for back-face culling.
            const auto build = [&](VkCullModeFlags cull, const bool transparent, VkPipeline& out) -> bool {
                VkPipelineRasterizationStateCreateInfo raster{};
                raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
                raster.polygonMode = VK_POLYGON_MODE_FILL;
                raster.cullMode = cull;
                raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
                raster.lineWidth = 1.0f;

                VkGraphicsPipelineCreateInfo pipeline_info{};
                pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
                pipeline_info.pNext = &rendering_info;
                pipeline_info.stageCount = stage_count;
                pipeline_info.pStages = stages.data();
                pipeline_info.pVertexInputState = &vertex_input;
                pipeline_info.pInputAssemblyState = &input_assembly;
                pipeline_info.pViewportState = &viewport_state;
                pipeline_info.pRasterizationState = &raster;
                pipeline_info.pMultisampleState = &multisample;
                auto depth_state = depth;
                depth_state.depthWriteEnable = transparent ? VK_FALSE : VK_TRUE;
                auto blend_attachment_state = blend_attachment;
                if (transparent) {
                    blend_attachment_state.blendEnable = VK_TRUE;
                    blend_attachment_state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
                    blend_attachment_state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                    blend_attachment_state.colorBlendOp = VK_BLEND_OP_ADD;
                    blend_attachment_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                    blend_attachment_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                    blend_attachment_state.alphaBlendOp = VK_BLEND_OP_ADD;
                }
                auto blend_state = blend;
                blend_state.pAttachments = &blend_attachment_state;
                pipeline_info.pDepthStencilState = &depth_state;
                pipeline_info.pColorBlendState = &blend_state;
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

            const bool ok = build(VK_CULL_MODE_BACK_BIT, false, pipeline_cull) &&
                            build(VK_CULL_MODE_NONE, false, pipeline_no_cull) &&
                            build(VK_CULL_MODE_BACK_BIT, true, pipeline_cull_blend) &&
                            build(VK_CULL_MODE_NONE, true, pipeline_no_cull_blend);
            vkDestroyShaderModule(device, vert, nullptr);
            if (geom)
                vkDestroyShaderModule(device, geom, nullptr);
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
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE, pipeline_cull_blend,
                                        "mesh.main.pipeline.cull.blend");
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE, pipeline_no_cull_blend,
                                        "mesh.main.pipeline.no_cull.blend");
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
            rendering_info.stencilAttachmentFormat =
                hasStencilAspect(depth_format) ? depth_format : VK_FORMAT_UNDEFINED;

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

        bool createOverlayPipeline(VkFormat color_format, VkFormat depth_format, bool collision = false) {
            if (collision) {
                collision_pipeline_attempted = true;
                VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
                const VkDescriptorSetLayoutCreateInfo descriptors{
                    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &binding};
                if (!vk_try_bool(vkCreateDescriptorSetLayout(device, &descriptors, nullptr, &collision_layout),
                                 "vkCreateDescriptorSetLayout", "Collision preview descriptor layout"))
                    return false;
            }
            if (!collision && (context == nullptr || !context->hasFillModeNonSolid())) {
                // VK_POLYGON_MODE_LINE is optional. Keep the pass usable on
                // conformant devices that do not expose fillModeNonSolid; the UI
                // reports the same capability and does not offer the overlay.
                return true;
            }
            using namespace viewport_shaders;
            const auto label = collision ? "Collision preview" : "Mesh wireframe";
            auto& target_layout = collision ? collision_pipeline_layout : wireframe_pipeline_layout;
            auto& target_pipeline = collision ? collision_pipeline : wireframe_pipeline;
            const auto vert = createShaderModule(device, collision ? std::span<const uint32_t>(kCollisionOverlayVertSpv) : std::span<const uint32_t>(kMeshWireframeVertSpv), label);
            const auto frag = createShaderModule(device, collision ? std::span<const uint32_t>(kCollisionOverlayFragSpv) : std::span<const uint32_t>(kMeshWireframeFragSpv), label);
            if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
                if (vert)
                    vkDestroyShaderModule(device, vert, nullptr);
                if (frag)
                    vkDestroyShaderModule(device, frag, nullptr);
                LOG_ERROR("{} shader module creation failed", label);
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
            vertex_input.vertexBindingDescriptionCount = collision ? 0 : 1;
            vertex_input.pVertexBindingDescriptions = &binding;
            vertex_input.vertexAttributeDescriptionCount = collision ? 0 : 1;
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
            raster.polygonMode = collision ? VK_POLYGON_MODE_FILL : VK_POLYGON_MODE_LINE;
            raster.cullMode = VK_CULL_MODE_NONE;
            raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo multisample{};
            multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            // Wireframe tests the mesh depth; collision is a scene overlay. Neither writes depth.
            VkPipelineDepthStencilStateCreateInfo depth{};
            depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depth.depthTestEnable = !collision;
            depth.depthWriteEnable = VK_FALSE;
            depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            VkPipelineColorBlendAttachmentState blend_attachment{};
            blend_attachment.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            blend_attachment.blendEnable = collision;
            blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

            VkPipelineColorBlendStateCreateInfo blend{};
            blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            blend.attachmentCount = 1;
            blend.pAttachments = &blend_attachment;

            std::array<VkDynamicState, 3> dynamic_states{
                VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_LINE_WIDTH};
            VkPipelineDynamicStateCreateInfo dynamic{};
            dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamic.dynamicStateCount = collision ? 2 : static_cast<std::uint32_t>(dynamic_states.size());
            dynamic.pDynamicStates = dynamic_states.data();

            VkPushConstantRange push{};
            push.stageFlags = collision ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            push.offset = 0;
            push.size = collision ? sizeof(CollisionPush) : sizeof(WireframePush);

            VkPipelineLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layout_info.setLayoutCount = collision ? 1 : 0;
            layout_info.pSetLayouts = collision ? &collision_layout : nullptr;
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &push;
            const VkResult layout_result =
                vkCreatePipelineLayout(device, &layout_info, nullptr, &target_layout);
            if (layout_result != VK_SUCCESS) {
                vkDestroyShaderModule(device, vert, nullptr);
                vkDestroyShaderModule(device, frag, nullptr);
                return reportVkFailure(
                    "vkCreatePipelineLayout(device, &layout_info, nullptr, &target_layout)",
                    layout_result,
                    std::format("{} pipeline-layout creation failed (device={:#x}, set_layout_count={}, push_constant_bytes={})",
                                label, vkHandleValue(device),
                                layout_info.setLayoutCount,
                                push.size));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                        target_layout,
                                        collision ? "collision.preview.pipeline.layout" : "mesh.wireframe.pipeline.layout");

            VkPipelineRenderingCreateInfo rendering_info{};
            rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            rendering_info.colorAttachmentCount = 1;
            rendering_info.pColorAttachmentFormats = &color_format;
            rendering_info.depthAttachmentFormat = depth_format;
            rendering_info.stencilAttachmentFormat =
                hasStencilAspect(depth_format) ? depth_format : VK_FORMAT_UNDEFINED;

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
            pipeline_info.layout = target_layout;

            const VkResult result = vkCreateGraphicsPipelines(device, pipeline_cache, 1,
                                                              &pipeline_info, nullptr, &target_pipeline);
            vkDestroyShaderModule(device, vert, nullptr);
            vkDestroyShaderModule(device, frag, nullptr);
            if (result != VK_SUCCESS) {
                return reportVkFailure(
                    "vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline_info, nullptr, &target_pipeline)",
                    result,
                    std::format("{} graphics-pipeline creation failed (device={:#x}, pipeline_cache={:#x}, pipeline_layout={:#x}, color_format={}, depth_format={})",
                                label, vkHandleValue(device),
                                vkHandleValue(pipeline_cache),
                                vkHandleValue(target_layout),
                                static_cast<int>(color_format),
                                static_cast<int>(depth_format)));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_PIPELINE,
                                        target_pipeline,
                                        collision ? "collision.preview.pipeline" : "mesh.wireframe.pipeline");
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
            if (!vk_try_bool(
                    vmaCreateBuffer(allocator, &info, &ai, &buffer, &alloc, nullptr),
                    "vmaCreateBuffer(allocator, &info, &ai, &buffer, &alloc, nullptr)",
                    lfs::rendering::formatVulkanDiagnostic(
                        "Mesh staging-buffer allocation failed (allocator={:#x}, source={:#x}, requested_size={}, usage={:#x})",
                        reinterpret_cast<std::uintptr_t>(allocator),
                        reinterpret_cast<std::uintptr_t>(data),
                        size,
                        static_cast<std::uint32_t>(info.usage)),
                    std::source_location::current())) {
                return false;
            }
            if (!writeBuffer(alloc, data, static_cast<std::size_t>(size))) {
                vmaDestroyBuffer(allocator, buffer, alloc);
                buffer = VK_NULL_HANDLE;
                alloc = VK_NULL_HANDLE;
                return false;
            }
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
            const glm::vec3 light_dir = item.camera_relative_light ? params.camera_backward : item.light_dir;
            ubo.light_dir[0] = light_dir.x;
            ubo.light_dir[1] = light_dir.y;
            ubo.light_dir[2] = light_dir.z;
            ubo.light_dir[3] = 0.0f;
            ubo.params[0] = item.light_intensity;
            ubo.params[1] = item.ambient;
            ubo.params[2] = shadow_enabled ? 1.0f : 0.0f;
            ubo.params[3] = 0.0f;
            ubo.selection[0] = item.is_emphasized ? 1.0f : 0.0f;
            ubo.selection[1] = item.dim_non_emphasized ? 1.0f : 0.0f;
            ubo.selection[2] = 0.0f;
            ubo.selection[3] = 0.0f;
            std::memcpy(ubo.light_vp, &light_vp[0][0], sizeof(ubo.light_vp));
            ubo.camera_forward[0] = -params.camera_backward.x;
            ubo.camera_forward[1] = -params.camera_backward.y;
            ubo.camera_forward[2] = -params.camera_backward.z;
            ubo.depth_view[0] = params.depth_view ? 1.0f : 0.0f;
            ubo.depth_view[1] = params.depth_view_min;
            ubo.depth_view[2] = params.depth_view_max;
            ubo.depth_view[3] = params.depth_grayscale ? 1.0f : 0.0f;
            const VulkanMeshViewportPanel* projection_state =
                params.draw_group < params.panels.size() ? &params.panels[params.draw_group] : nullptr;
            const bool equirectangular = projection_state && projection_state->equirectangular;
            ubo.selection[2] = equirectangular ? 1.0f : 0.0f;
            if (equirectangular) {
                std::memcpy(ubo.view, &projection_state->view[0][0], sizeof(ubo.view));
                glm::mat4 clip_y_flip(1.0f);
                clip_y_flip[1][1] = -1.0f;
                const glm::mat4 projection = clip_y_flip * projection_state->projection;
                std::memcpy(ubo.projection, &projection[0][0], sizeof(ubo.projection));
            }
            const std::size_t upload_size = equirectangular ? sizeof(ubo) : offsetof(LightUbo, view);
            return writeBuffer(resources.allocation, &ubo, upload_size);
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

        bool ensureShadowTarget(ShadowGroupState& shadow, int resolution) {
            if (shadow.shadow.image != VK_NULL_HANDLE && shadow.shadow.resolution == resolution) {
                return true;
            }
            ShadowTarget replacement{};
            if (!createShadowTarget(resolution, replacement)) {
                return false;
            }
            // Target creation waits for its initialization submit; the previous image
            // is then no longer referenced by queued viewport work.
            destroyShadow(shadow.shadow);
            shadow.shadow = std::move(replacement);
            shadow.cached_light_vp_valid = false;
            return true;
        }

        bool recordShadowPass(VkCommandBuffer cb,
                              ShadowGroupState& shadow,
                              const SharedMeshDrawAsset& mesh,
                              const glm::mat4& light_mvp) {
            if (cb == VK_NULL_HANDLE || shadow.shadow.image == VK_NULL_HANDLE ||
                shadow_pipeline == VK_NULL_HANDLE || mesh.vertex_buffer == VK_NULL_HANDLE ||
                mesh.index_buffer == VK_NULL_HANDLE) {
                return false;
            }

            cmdImageBarrier2(cb, shadow.shadow.image, depthAspectMask(shadow_format),
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                             VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            VkClearValue clear{};
            clear.depthStencil = {1.0f, 0};
            VkRenderingAttachmentInfo depth_attachment{};
            depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView = shadow.shadow.view;
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_attachment.clearValue = clear;

            VkRenderingInfo rendering{};
            rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering.renderArea.offset = {0, 0};
            rendering.renderArea.extent = {static_cast<std::uint32_t>(shadow.shadow.resolution),
                                           static_cast<std::uint32_t>(shadow.shadow.resolution)};
            rendering.layerCount = 1;
            rendering.pDepthAttachment = &depth_attachment;
            vkCmdBeginRendering(cb, &rendering);

            VkViewport vp{};
            vp.x = 0.0f;
            vp.y = 0.0f;
            vp.width = static_cast<float>(shadow.shadow.resolution);
            vp.height = static_cast<float>(shadow.shadow.resolution);
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

            VkBuffer vbuf = mesh.vertex_buffer;
            VkDeviceSize voff = 0;
            vkCmdBindVertexBuffers(cb, 0, 1, &vbuf, &voff);
            vkCmdBindIndexBuffer(cb, mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cb, mesh.total_index_count, 1, 0, 0, 0);
            vkCmdEndRendering(cb);

            cmdImageBarrier2(cb, shadow.shadow.image, depthAspectMask(shadow_format),
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
            return true;
        }

        void prepareCollision(const VulkanMeshPassParams& params) {
            if (collision_frames.size() <= params.frame_slot)
                collision_frames.resize(params.frame_slot + 1);
            std::vector<std::shared_ptr<CollisionFrame>> resources;
            for (const auto& item : params.items) {
                if (!item.collision)
                    continue;
                if (!collision_pipeline_attempted && !createOverlayPipeline(color_format_cached, depth_format_cached, true))
                    LOG_ERROR("Collision preview pipeline failed; other rendering is unaffected");
                if (!collision_pipeline)
                    break;
                auto& cached = collision_cache[item.collision.get()];
                auto resource = cached.lock();
                if (!resource) {
                    resource = std::make_shared<CollisionFrame>();
                    resource->owner = item.collision;
                    cached = resource;
                    uploadCollision(*resource);
                }
                resources.push_back(std::move(resource));
            }
            // Replace only the retired slot; other in-flight frames retain their buffers.
            collision_frames[params.frame_slot] = std::move(resources);
            std::erase_if(collision_cache, [](const auto& entry) { return entry.second.expired(); });
        }

        void uploadCollision(CollisionFrame& frame) {
            frame.device = device;
            frame.allocator = allocator;
            const auto& volume = *frame.owner;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(context->physicalDevice(), &properties);
            const VkDeviceSize bytes = (volume.words.size() + 4) * sizeof(uint32_t);
            if (bytes > properties.limits.maxStorageBufferRange) {
                LOG_ERROR("Collision preview requires {} storage bytes; device limit is {}", bytes, properties.limits.maxStorageBufferRange);
                return;
            }
            VkBufferCreateInfo buffer_info{};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.size = bytes;
            buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo allocation_info{};
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                    VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT;
            if (!vk_try_bool(vmaCreateBuffer(allocator, &buffer_info, &allocation_info,
                                             &frame.buffer, &frame.allocation, nullptr),
                             "vmaCreateBuffer", "Collision preview storage buffer"))
                return;

            VkMemoryPropertyFlags memory_properties{};
            vmaGetAllocationMemoryProperties(allocator, frame.allocation, &memory_properties);
            const bool direct_upload = (memory_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
            VmaAllocation upload_allocation = frame.allocation;
            if (!direct_upload) {
                buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
                allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
                if (!vk_try_bool(vmaCreateBuffer(allocator, &buffer_info, &allocation_info,
                                                 &frame.staging, &frame.staging_allocation, nullptr),
                                 "vmaCreateBuffer", "Collision preview staging buffer"))
                    return;
                upload_allocation = frame.staging_allocation;
            }

            void* mapped = nullptr;
            if (!vk_try_bool(vmaMapMemory(allocator, upload_allocation, &mapped),
                             "vmaMapMemory", "Collision preview upload mapping"))
                return;
            const std::array<uint32_t, 4> header{uint32_t(volume.dimensions.x), uint32_t(volume.dimensions.y),
                                                 uint32_t(volume.dimensions.z), uint32_t(volume.edge)};
            std::memcpy(mapped, header.data(), sizeof(header));
            if (!volume.words.empty())
                std::memcpy(static_cast<uint32_t*>(mapped) + header.size(), volume.words.data(),
                            volume.words.size() * sizeof(uint32_t));
            // VMA flushes non-coherent memory and does nothing for coherent allocations.
            const VkResult flush_result = vmaFlushAllocation(allocator, upload_allocation, 0, bytes);
            vmaUnmapMemory(allocator, upload_allocation);
            if (!vk_try_bool(flush_result, "vmaFlushAllocation", "Collision preview upload flush"))
                return;

            // The immutable buffer is written and flushed before the draw's queue submission.
            // Submission makes these host writes visible to fragment shader reads (or to the
            // staging transfer below), so direct uploads need no copy, barrier, or upload fence.
            if (!direct_upload) {
                const auto commands = beginSingleTimeCommands();
                if (commands == VK_NULL_HANDLE)
                    return;
                const VkBufferCopy copy{0, 0, bytes};
                vkCmdCopyBuffer(commands, frame.staging, frame.buffer, 1, &copy);
                const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                              VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
                vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                     0, 1, &barrier, 0, nullptr, 0, nullptr);
                // One upload fence, never a device-wide wait or a per-frame transfer.
                if (!endSingleTimeCommands(commands, &frame.pending_upload).completed)
                    return;
                vmaDestroyBuffer(allocator, frame.staging, frame.staging_allocation);
                frame.staging = VK_NULL_HANDLE;
                frame.staging_allocation = VK_NULL_HANDLE;
            }
            const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
            const VkDescriptorPoolCreateInfo pool{
                VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &size};
            if (!vk_try_bool(vkCreateDescriptorPool(device, &pool, nullptr, &frame.pool),
                             "vkCreateDescriptorPool", "Collision preview descriptors"))
                return;
            const VkDescriptorSetAllocateInfo allocation{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, frame.pool, 1, &collision_layout};
            if (!vk_try_bool(vkAllocateDescriptorSets(device, &allocation, &frame.descriptor),
                             "vkAllocateDescriptorSets", "Collision preview descriptor"))
                return;
            const VkDescriptorBufferInfo buffer{frame.buffer, 0, bytes};
            const VkWriteDescriptorSet write{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, frame.descriptor, 0, 0, 1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffer, nullptr};
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
            LOG_DEBUG("Collision preview uploaded {} bytes for immutable volume", bytes);
        }

        void prepare(VulkanContext& ctx, const VulkanMeshPassParams& params) {
            if (!assets) {
                assets = std::make_shared<SharedViewportGpuAssets>();
            }
            if (!assets->ensureContext(ctx)) {
                LOG_ERROR("VulkanMeshPass: shared scene GPU assets are not bound to the live context");
                return;
            }
            if (device != VK_NULL_HANDLE &&
                (context != &ctx || device != assets->device() || allocator != ctx.allocator())) {
                const VkFormat color = color_format_cached;
                const VkFormat depth = depth_format_cached;
                auto shared = assets;
                destroy();
                if (!init(ctx, color, depth, std::move(shared))) {
                    LOG_ERROR("VulkanMeshPass: failed to rebuild per-view resources after a context switch");
                    return;
                }
            }
            assets->prepareMeshes(params.items, params.frame_slot);
            std::erase_if(retired_materials, [this](auto& retired) {
                if (retired.consumer > context->retiredFrameSubmitSerial())
                    return false;
                destroyMaterials(retired.bindings);
                return true;
            });
            for (const auto& item : params.items) {
                if (const auto* mesh = item.mesh ? assets->findMesh(item.mesh->id()) : nullptr)
                    (void)prepareMaterials(item.mesh->id(), *mesh);
            }
            for (auto it = material_bindings.begin(); it != material_bindings.end();) {
                if (context->retiredFrameSubmitSerial() > it->second.last_used + 120) {
                    destroyMaterials(it->second);
                    it = material_bindings.erase(it);
                } else {
                    ++it;
                }
            }
            prepareCollision(params);
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

            VkCommandBuffer shadow_cb = VK_NULL_HANDLE;
            for (const auto& item : params.items) {
                if (!item.mesh || !item.shadow_enabled || item.shadow_map_resolution <= 0)
                    continue;
                const auto* asset = assets->findMesh(item.mesh->id());
                if (!asset)
                    continue;
                const auto& mesh = *asset;
                auto& gpu = mesh_shadows[item.mesh->id()];
                if (gpu.groups.size() < params.draw_group_count)
                    gpu.groups.resize(params.draw_group_count);
                for (std::size_t group_index = 0; group_index < params.draw_group_count; ++group_index) {
                    auto& group = gpu.groups[group_index];
                    if (!ensureShadowTarget(group, item.shadow_map_resolution))
                        continue;
                    const glm::vec3 light_dir = item.camera_relative_light
                                                    ? (group_index < params.panels.size()
                                                           ? params.panels[group_index].camera_backward
                                                           : params.camera_backward)
                                                    : item.light_dir;
                    const bool shadow_dirty =
                        !group.cached_light_vp_valid ||
                        group.shadow_key_generation != mesh.generation ||
                        group.shadow_key_resolution != item.shadow_map_resolution ||
                        group.shadow_key_model != item.model ||
                        group.shadow_key_light_dir != light_dir;
                    if (!shadow_dirty)
                        continue;
                    if (shadow_cb == VK_NULL_HANDLE)
                        shadow_cb = beginSingleTimeCommands();
                    if (shadow_cb == VK_NULL_HANDLE)
                        continue;
                    const glm::mat4 light_vp =
                        meshShadowViewProjection(mesh.aabb_min, mesh.aabb_max, item.model, light_dir);
                    if (!recordShadowPass(shadow_cb, group, mesh, light_vp * item.model)) {
                        group.cached_light_vp_valid = false;
                        continue;
                    }
                    group.cached_light_vp = light_vp;
                    group.cached_light_vp_valid = true;
                    group.shadow_key_generation = mesh.generation;
                    group.shadow_key_resolution = item.shadow_map_resolution;
                    group.shadow_key_model = item.model;
                    group.shadow_key_light_dir = light_dir;
                }
            }
            if (shadow_cb != VK_NULL_HANDLE) {
                const auto result = endSingleTimeCommands(shadow_cb);
                if (!result.completed) {
                    for (auto& [_, gpu] : mesh_shadows) {
                        for (auto& group : gpu.groups) {
                            group.cached_light_vp_valid = false;
                            if (!result.safe_to_release)
                                abandonShadow(group.shadow);
                        }
                    }
                }
            }

            constexpr std::uint64_t kEvictAfter = 120;
            std::unordered_map<std::uint64_t, bool> used_this_prepare;
            used_this_prepare.reserve(params.items.size());
            for (const auto& item : params.items) {
                if (item.mesh) {
                    used_this_prepare[item.mesh->id()] = true;
                }
            }
            for (auto& [id, gpu] : mesh_shadows) {
                if (used_this_prepare.contains(id)) {
                    gpu.last_used_epoch = 0;
                } else if (gpu.last_used_epoch < kEvictAfter) {
                    ++gpu.last_used_epoch;
                }
            }
            bool has_stale_shadow = false;
            for (const auto& [_, gpu] : mesh_shadows) {
                if (gpu.last_used_epoch > kEvictAfter) {
                    has_stale_shadow = true;
                    break;
                }
            }
            bool submitted_frames_retired = !has_stale_shadow;
            if (has_stale_shadow) {
                if (context == nullptr) {
                    LOG_ERROR("VulkanMeshPass deferred stale shadow eviction because retirement cannot be proven (cache_size={}, eviction_age={})",
                              mesh_shadows.size(),
                              kEvictAfter);
                    return;
                }
                submitted_frames_retired = context->waitForSubmittedFrames();
                if (!submitted_frames_retired) {
                    LOG_WARN("VulkanMeshPass deferred stale shadow eviction because submitted frames did not retire (cache_size={}, eviction_age={}, error='{}')",
                             mesh_shadows.size(),
                             kEvictAfter,
                             context->lastError());
                    return;
                }
            }
            for (auto it = mesh_shadows.begin(); it != mesh_shadows.end();) {
                if (it->second.last_used_epoch > kEvictAfter) {
                    LFS_VK_DEBUG_ASSERT(
                        submitted_frames_retired,
                        "Deferred mesh shadow destruction requires all submitted frames to retire (frames_retired={}, age={}, eviction_age={})",
                        submitted_frames_retired,
                        it->second.last_used_epoch,
                        kEvictAfter);
                    for (auto& group : it->second.groups)
                        destroyShadow(group.shadow);
                    it = mesh_shadows.erase(it);
                } else {
                    ++it;
                }
            }
        }

        void record(VkCommandBuffer cb, VkRect2D viewport_rect, const VulkanMeshPassParams& params,
                    const std::optional<VkRect2D> scissor_rect) {
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
            const VkRect2D scissor = scissor_rect.value_or(viewport_rect);
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
            transparent_draws.clear();
            for (std::size_t item_index = 0; item_index < params.items.size(); ++item_index) {
                const auto& item = params.items[item_index];
                if (!item.mesh || !assets || params.draw_group >= 32 ||
                    !((item.panel_mask >> params.draw_group) & 1u))
                    continue;
                const auto* asset = assets->findMesh(item.mesh->id());
                if (!asset)
                    continue;
                const auto& mesh = *asset;
                const auto shadow_it = mesh_shadows.find(item.mesh->id());
                const MeshShadowState* shadow_state =
                    shadow_it != mesh_shadows.end() ? &shadow_it->second : nullptr;
                const ShadowGroupState* shadow_group =
                    shadow_state != nullptr && params.draw_group < shadow_state->groups.size()
                        ? &shadow_state->groups[params.draw_group]
                        : nullptr;

                const bool shadow_active = item.shadow_enabled &&
                                           shadow_group != nullptr &&
                                           shadow_group->shadow.image != VK_NULL_HANDLE &&
                                           shadow_group->cached_light_vp_valid;
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
                                   shadow_active ? shadow_group->cached_light_vp : glm::mat4(1.0f),
                                   shadow_active)) {
                    LOG_ERROR("VulkanMeshPass: failed to prepare light state for draw {}", draw_index);
                    continue;
                }
                bindShadowMap(draw, shadow_active ? shadow_group->shadow : shadow_dummy);

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

                VkBuffer vbuf = mesh.vertex_buffer;
                VkDeviceSize voff = 0;
                vkCmdBindVertexBuffers(cb, 0, 1, &vbuf, &voff);
                vkCmdBindIndexBuffer(cb, mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);

                const auto materials = material_bindings.find(item.mesh->id());
                if (materials == material_bindings.end() || materials->second.materials.empty() ||
                    materials->second.generation != mesh.generation)
                    continue;
                for (const auto& sm : mesh.submeshes) {
                    if (sm.index_count == 0)
                        continue;
                    const std::size_t mat_idx =
                        std::min(sm.material_index, materials->second.materials.size() - 1);
                    const VkDescriptorSet material_descriptor = materials->second.materials[mat_idx].descriptor;
                    if (material_descriptor == VK_NULL_HANDLE) {
                        continue;
                    }
                    if (mat_idx < mesh.materials.size() &&
                        mesh.materials[mat_idx].alpha_mode == lfs::core::AlphaMode::Blend) {
                        const glm::vec3 local_center = sm.local_center;
                        const glm::vec3 world_center = glm::vec3(item.model * glm::vec4(local_center, 1.0f));
                        transparent_draws.push_back({&mesh, &sm, material_descriptor, item_index,
                                                     draw_index, static_cast<std::size_t>(&sm - mesh.submeshes.data()),
                                                     glm::dot(world_center - params.camera_position,
                                                              -params.camera_backward)});
                        continue;
                    }
                    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                            1, 1, &material_descriptor, 0, nullptr);
                    vkCmdDrawIndexed(cb, sm.index_count, 1, sm.start_index, 0, 0);
                }

                // Wireframe overlay drawn on top of the shaded mesh.
                if (item.wireframe_overlay && !params.depth_view && wireframe_pipeline != VK_NULL_HANDLE) {
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
                    vkCmdDrawIndexed(cb, mesh.total_index_count, 1, 0, 0, 0);
                }
            }
            std::sort(transparent_draws.begin(), transparent_draws.end(),
                      [](const TransparentSubmeshDraw& lhs, const TransparentSubmeshDraw& rhs) {
                          if (lhs.view_depth != rhs.view_depth)
                              return lhs.view_depth > rhs.view_depth;
                          if (lhs.item_index != rhs.item_index)
                              return lhs.item_index < rhs.item_index;
                          return lhs.submesh_order < rhs.submesh_order;
                      });
            for (const auto& transparent : transparent_draws) {
                const auto& item = params.items[transparent.item_index];
                const auto& mesh = *transparent.mesh;
                const auto& sm = *transparent.submesh;
                const auto& draw = frame.draws[transparent.draw_index];
                const VkPipeline pipeline = item.backface_culling ? pipeline_cull_blend : pipeline_no_cull_blend;
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                        0, 1, &draw.descriptor, 0, nullptr);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                        1, 1, &transparent.material, 0, nullptr);
                MeshPushConstants pc{};
                const glm::mat4 mvp = view_projection * item.model;
                std::memcpy(pc.mvp, &mvp[0][0], sizeof(pc.mvp));
                std::memcpy(pc.model, &item.model[0][0], sizeof(pc.model));
                vkCmdPushConstants(cb, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
                VkBuffer vbuf = mesh.vertex_buffer;
                const VkDeviceSize voff = 0;
                vkCmdBindVertexBuffers(cb, 0, 1, &vbuf, &voff);
                vkCmdBindIndexBuffer(cb, mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cb, sm.index_count, 1, sm.start_index, 0, 0);
            }
            if (collision_pipeline && !params.depth_view) {
                for (const auto& item : params.items) {
                    const auto found = collision_cache.find(item.collision.get());
                    const auto resource = found == collision_cache.end() ? nullptr : found->second.lock();
                    if (!resource || item.collision_opacity <= 0 || !resource->descriptor ||
                        params.draw_group >= 32 || !((item.panel_mask >> params.draw_group) & 1u))
                        continue;
                    const auto& frame = *resource;
                    const auto& volume = *item.collision;
                    const auto model = glm::scale(glm::translate(item.model, volume.origin), glm::vec3(volume.settings.voxel_size));
                    const auto grid_to_clip = view_projection * model;
                    if (!std::isfinite(glm::determinant(grid_to_clip)) || glm::determinant(grid_to_clip) == 0) {
                        LOG_ERROR("Collision preview requires an invertible node and projection transform");
                        continue;
                    }
                    CollisionPush push{glm::inverse(grid_to_clip)};
                    const auto* panel = params.draw_group < params.panels.size() ? &params.panels[params.draw_group] : nullptr;
                    const auto rows = glm::transpose(view_projection);
                    push.pixel.x = 2.0f / (viewport.height * glm::length(glm::vec3(rows[1])));
                    push.pixel.y = glm::length(glm::vec3(rows[3])) == 0.0f;
                    if (panel && panel->equirectangular) {
                        push.clip_to_grid = glm::inverse(panel->view * model);
                        push.panorama = {1, panel->projection[3][2] / panel->projection[2][2], 0, 0};
                    }
                    const glm::vec3 camera = glm::vec3(glm::inverse(model) * glm::vec4(panel ? panel->camera_position : params.camera_position, 1));
                    push.panorama.w = glm::all(glm::greaterThanEqual(camera, glm::vec3(0))) &&
                                      glm::all(glm::lessThan(camera, glm::vec3(volume.dimensions))) && volume.occupied(glm::ivec3(camera));
                    push.panorama.z = item.collision_opacity;
                    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, collision_pipeline);
                    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, collision_pipeline_layout, 0, 1, &frame.descriptor, 0, nullptr);
                    vkCmdPushConstants(cb, collision_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
                    vkCmdDraw(cb, 3, 1, 0, 0);
                }
            }
        }
    };

    VulkanMeshPass::VulkanMeshPass() = default;
    VulkanMeshPass::~VulkanMeshPass() = default;
    VulkanMeshPass::VulkanMeshPass(VulkanMeshPass&&) noexcept = default;
    VulkanMeshPass& VulkanMeshPass::operator=(VulkanMeshPass&&) noexcept = default;

    bool VulkanMeshPass::init(VulkanContext& context,
                              VkFormat color_format,
                              VkFormat depth_stencil_format,
                              std::shared_ptr<SharedViewportGpuAssets> shared_assets) {
        if (!impl_) {
            impl_ = std::make_unique<Impl>();
        }
        return impl_->init(context, color_format, depth_stencil_format, std::move(shared_assets));
    }

    void VulkanMeshPass::prepare(VulkanContext& context, const VulkanMeshPassParams& params) {
        if (!impl_)
            return;
        impl_->prepare(context, params);
    }

    void VulkanMeshPass::discardImport(uint64_t mesh_id) {
        if (impl_ && impl_->assets)
            impl_->assets->discardImportMesh(mesh_id);
    }

    void VulkanMeshPass::record(VkCommandBuffer command_buffer, VkRect2D viewport_rect,
                                const VulkanMeshPassParams& params,
                                const std::optional<VkRect2D> scissor) {
        if (!impl_)
            return;
        impl_->record(command_buffer, viewport_rect, params, scissor);
    }

    void VulkanMeshPass::shutdown() {
        if (impl_) {
            impl_->destroy();
            impl_.reset();
        }
    }

} // namespace lfs::vis
