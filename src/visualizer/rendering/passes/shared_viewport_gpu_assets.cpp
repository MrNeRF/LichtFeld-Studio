/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "shared_viewport_gpu_assets_impl.hpp"
#include "vulkan_environment_pass.hpp"
#include "vulkan_mesh_pass.hpp"

#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/material.hpp"
#include "core/mesh_data.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "internal/resource_paths.hpp"
#include "rendering/vulkan_wait.hpp"
#include "window/vulkan_barrier2.hpp"
#include "window/vulkan_result.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <glm/glm.hpp>
#include <limits>
#include <source_location>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::vis {

    namespace shared_viewport_gpu_detail {

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

        [[nodiscard]] OneShotResult submitOneShot(VulkanContext* context,
                                                  VkDevice device,
                                                  VkQueue queue,
                                                  VkCommandPool pool,
                                                  VkCommandBuffer cb,
                                                  const char* wait_fingerprint,
                                                  const char* label) {
            VkResult r = vkEndCommandBuffer(cb);
            if (r != VK_SUCCESS) {
                vkFreeCommandBuffers(device, pool, 1, &cb);
                static_cast<void>(reportVkFailure(
                    "vkEndCommandBuffer(cb)",
                    r,
                    std::format("{} command buffer did not leave recording state (command_buffer={:#x}, command_pool={:#x})",
                                label,
                                vkHandleValue(cb),
                                vkHandleValue(pool))));
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
                    "{} one-shot fence creation failed (device={:#x}, command_buffer={:#x})",
                    label,
                    vkHandleValue(device),
                    vkHandleValue(cb));
            } else if (context != nullptr) {
                context->setDebugObjectName(VK_OBJECT_TYPE_FENCE, fence,
                                            std::string(label) + ".transfer.fence");
            }
            if (r == VK_SUCCESS &&
                (queue == VK_NULL_HANDLE || cb == VK_NULL_HANDLE || fence == VK_NULL_HANDLE ||
                 submit.commandBufferCount != 1 || submit.pCommandBuffers == nullptr ||
                 submit.pCommandBuffers[0] != cb)) {
                r = VK_ERROR_INITIALIZATION_FAILED;
                failed_expression = "shared viewport gpu assets one-shot submit integrity check";
                failed_context = std::format(
                    "{} one-shot submit requires a non-null queue, one expected command buffer, and a non-null fence (queue={:#x}, command_buffer={:#x}, fence={:#x})",
                    label,
                    vkHandleValue(queue),
                    vkHandleValue(cb),
                    vkHandleValue(fence));
            }
            bool submitted = false;
            bool wait_ready = false;
            bool safe_to_release = true;
            if (r == VK_SUCCESS) {
                r = lfs::rendering::vk_queue_submit_synced(queue, 1, &submit, fence);
                if (r != VK_SUCCESS) {
                    failed_expression =
                        "lfs::rendering::vk_queue_submit_synced(queue, 1, &submit, fence)";
                    failed_context = std::format(
                        "{} one-shot submission failed (queue={:#x}, command_buffer={:#x}, fence={:#x})",
                        label,
                        vkHandleValue(queue),
                        vkHandleValue(cb),
                        vkHandleValue(fence));
                } else {
                    submitted = true;
                }
            }
            if (r == VK_SUCCESS) {
                lfs::rendering::WaitContext wait_ctx;
                wait_ctx.fingerprint = wait_fingerprint;
                auto wait_outcome = lfs::rendering::wait_fence_bounded(
                    device, fence, std::stop_token{}, lfs::rendering::VulkanWaitPolicy{}, wait_ctx);
                if (wait_outcome.has_value() &&
                    *wait_outcome == lfs::rendering::WaitOutcome::Ready) {
                    wait_ready = true;
                } else {
                    r = VK_TIMEOUT;
                    failed_expression = wait_fingerprint;
                    failed_context = std::format(
                        "{} one-shot submission did not retire (device={:#x}, fence={:#x}, command_buffer={:#x}): {}",
                        label,
                        vkHandleValue(device),
                        vkHandleValue(fence),
                        vkHandleValue(cb),
                        formatWaitFailure(wait_outcome));
                    // Wait on this failure path before callers release buffers referenced by the command.
                    if (context != nullptr && context->deviceWaitIdle()) {
                        wait_ready = true;
                        LOG_WARN("Vulkan: {} one-shot wait required failure-only device idle before cleanup",
                                 label);
                    } else {
                        safe_to_release = false;
                        LOG_ERROR("Vulkan: {} one-shot resources are abandoned after non-Ready wait and failed device idle containment",
                                  label);
                    }
                }
            }
            if (fence != VK_NULL_HANDLE) {
                if (!submitted || wait_ready) {
                    vkDestroyFence(device, fence, nullptr);
                } else {
                    LOG_ERROR("Vulkan: retaining {} one-shot fence after non-Ready wait (fence={:#x})",
                              label,
                              vkHandleValue(fence));
                }
            }
            if (!submitted || wait_ready) {
                vkFreeCommandBuffers(device, pool, 1, &cb);
            } else {
                LOG_ERROR("Vulkan: retaining {} one-shot command buffer after non-Ready wait (command_buffer={:#x})",
                          label,
                          vkHandleValue(cb));
            }
            if (r != VK_SUCCESS) {
                static_cast<void>(reportVkFailure(failed_expression, r, failed_context));
                return {false, safe_to_release};
            }
            return {true, safe_to_release};
        }

        [[nodiscard]] VkCommandBuffer allocateOneShot(VulkanContext* context,
                                                      VkDevice device,
                                                      VkCommandPool pool,
                                                      const char* debug_name) {
            VkCommandBufferAllocateInfo alloc{};
            alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            alloc.commandPool = pool;
            alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            VkResult result = vkAllocateCommandBuffers(device, &alloc, &cb);
            if (result != VK_SUCCESS) {
                LOG_ERROR("Vulkan: {}",
                          formatVkCheckFailure(
                              "vkAllocateCommandBuffers(device, &alloc, &cb)",
                              result,
                              std::format("{} command-buffer allocation failed (device={:#x}, command_pool={:#x})",
                                          debug_name,
                                          vkHandleValue(device),
                                          vkHandleValue(pool)),
                              __FILE__,
                              __LINE__));
                return VK_NULL_HANDLE;
            }
            if (context != nullptr) {
                context->setDebugObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER, cb, debug_name);
            }
            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            result = vkBeginCommandBuffer(cb, &begin);
            if (result != VK_SUCCESS) {
                LOG_ERROR("Vulkan: {}",
                          formatVkCheckFailure(
                              "vkBeginCommandBuffer(cb, &begin)",
                              result,
                              std::format("{} command buffer did not enter recording state (command_buffer={:#x})",
                                          debug_name,
                                          vkHandleValue(cb)),
                              __FILE__,
                              __LINE__));
                vkFreeCommandBuffers(device, pool, 1, &cb);
                return VK_NULL_HANDLE;
            }
            return cb;
        }

    } // namespace shared_viewport_gpu_detail

    SharedViewportGpuAssets::SharedViewportGpuAssets() : impl_(std::make_unique<Impl>()) {}

    SharedViewportGpuAssets::~SharedViewportGpuAssets() = default;

    SharedViewportGpuAssets::SharedViewportGpuAssets(SharedViewportGpuAssets&&) noexcept = default;

    SharedViewportGpuAssets& SharedViewportGpuAssets::operator=(SharedViewportGpuAssets&&) noexcept = default;

    bool SharedViewportGpuAssets::ensureContext(VulkanContext& context) {
        return impl_ && impl_->ensureContext(context);
    }

    void SharedViewportGpuAssets::prepareMeshes(const std::vector<VulkanMeshDrawItem>& items,
                                                const std::size_t frame_slot) {
        if (!impl_) {
            return;
        }
        impl_->beginFrame(frame_slot);
        impl_->prepareMeshes(items);
    }

    bool SharedViewportGpuAssets::findMesh(const std::uint64_t mesh_id, SharedMeshDrawAsset& out) const {
        return impl_ && impl_->findMesh(mesh_id, out);
    }

    void SharedViewportGpuAssets::prepareEnvironment(const VulkanEnvironmentParams& params,
                                                     const std::size_t frame_slot) {
        if (!impl_) {
            return;
        }
        impl_->beginFrame(frame_slot);
        impl_->prepareEnvironment(params);
    }

    SharedEnvironmentTexture SharedViewportGpuAssets::environmentTexture() const {
        return impl_ ? impl_->environmentTexture() : SharedEnvironmentTexture{};
    }

    VkDescriptorSetLayout SharedViewportGpuAssets::meshMaterialLayout() const {
        return impl_ ? impl_->material_layout : VK_NULL_HANDLE;
    }

    VkDevice SharedViewportGpuAssets::device() const {
        return impl_ ? impl_->device : VK_NULL_HANDLE;
    }

    bool SharedViewportGpuAssets::Impl::ensureContext(VulkanContext& ctx) {
        if (matches(ctx)) {
            return true;
        }
        shutdown();
        context = &ctx;
        device = ctx.device();
        allocator = ctx.allocator();
        graphics_queue = ctx.graphicsQueue();
        if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE ||
            graphics_queue == VK_NULL_HANDLE) {
            shutdown();
            return logVkFailure(std::format(
                "Shared viewport GPU assets require a live device, allocator, and graphics queue (device={:#x}, allocator={:#x}, graphics_queue={:#x}) ({}:{})",
                vkHandleValue(device),
                reinterpret_cast<std::uintptr_t>(allocator),
                vkHandleValue(graphics_queue),
                __FILE__,
                __LINE__));
        }
        if (!initMeshInfrastructure() || !initEnvironmentInfrastructure()) {
            shutdown();
            return false;
        }
        return true;
    }

    void SharedViewportGpuAssets::Impl::beginFrame(const std::size_t frame_slot) {
        if (!epoch_frame_slot.has_value() || *epoch_frame_slot != frame_slot) {
            ++epoch;
            epoch_frame_slot = frame_slot;
        }
    }

    void SharedViewportGpuAssets::Impl::shutdown() {
        if (device != VK_NULL_HANDLE && context != nullptr &&
            (!mesh_cache.empty() || !environment_images.empty())) {
            // The shared owner is also usable outside GuiManager. Retirement
            // cannot depend on a presentation pass having been destroyed first.
            if (!context->waitForSubmittedFrames() && !context->deviceWaitIdle()) {
                LOG_WARN("Shared viewport assets could not retire submitted frames: {}",
                         context->lastError());
            }
            if (!context->waitForImmediateSubmits()) {
                LOG_WARN("Shared viewport assets could not retire immediate submits: {}",
                         context->lastError());
            }
        }
        shutdownMeshes();
        shutdownEnvironment();
        context = nullptr;
        device = VK_NULL_HANDLE;
        allocator = VK_NULL_HANDLE;
        graphics_queue = VK_NULL_HANDLE;
        epoch = 0;
        epoch_frame_slot.reset();
    }

    bool SharedViewportGpuAssets::Impl::initMeshInfrastructure() {
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = context->graphicsQueueFamily();
        if (!vk_try_bool(
                vkCreateCommandPool(device, &pool_info, nullptr, &mesh_transfer_pool),
                "vkCreateCommandPool(device, &pool_info, nullptr, &mesh_transfer_pool)",
                lfs::rendering::formatVulkanDiagnostic(
                    "Shared mesh transfer command-pool creation failed (device={:#x}, queue_family={})",
                    vkHandleValue(device),
                    pool_info.queueFamilyIndex),
                std::source_location::current())) {
            return false;
        }
        context->setDebugObjectName(VK_OBJECT_TYPE_COMMAND_POOL,
                                    mesh_transfer_pool,
                                    "shared.mesh.transfer.pool");
        return createMeshSampler() && createMaterialLayout() &&
               createInitialMaterialDescriptorPool() && createWhitePixel();
    }

    void SharedViewportGpuAssets::Impl::shutdownMeshes() {
        for (auto& [_, gpu] : mesh_cache) {
            destroyMesh(gpu);
        }
        mesh_cache.clear();
        destroyTexture(white_pixel);
        for (const VkDescriptorPool pool : material_descriptor_pools) {
            if (pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, pool, nullptr);
            }
        }
        material_descriptor_pools.clear();
        if (material_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, material_layout, nullptr);
            material_layout = VK_NULL_HANDLE;
        }
        if (mesh_sampler != VK_NULL_HANDLE) {
            vkDestroySampler(device, mesh_sampler, nullptr);
            mesh_sampler = VK_NULL_HANDLE;
        }
        if (mesh_transfer_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, mesh_transfer_pool, nullptr);
            mesh_transfer_pool = VK_NULL_HANDLE;
        }
    }

    bool SharedViewportGpuAssets::Impl::initEnvironmentInfrastructure() {
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = context->graphicsQueueFamily();
        if (!vk_try_bool(
                vkCreateCommandPool(device, &pool_info, nullptr, &environment_transfer_pool),
                "vkCreateCommandPool(device, &pool_info, nullptr, &environment_transfer_pool)",
                lfs::rendering::formatVulkanDiagnostic(
                    "Shared environment transfer command-pool creation failed (device={:#x}, queue_family={})",
                    vkHandleValue(device),
                    pool_info.queueFamilyIndex),
                std::source_location::current())) {
            return false;
        }
        context->setDebugObjectName(VK_OBJECT_TYPE_COMMAND_POOL,
                                    environment_transfer_pool,
                                    "shared.environment.transfer.pool");
        return createEnvironmentSampler();
    }

    void SharedViewportGpuAssets::Impl::shutdownEnvironment() {
        for (auto& image : environment_images) {
            destroyEnvironmentImage(image);
        }
        environment_images.clear();
        live_environment_index = std::numeric_limits<std::size_t>::max();
        environment_failed_path.clear();
        environment_load_failed = false;
        environment_last_enabled_epoch = 0;
        if (environment_sampler != VK_NULL_HANDLE) {
            vkDestroySampler(device, environment_sampler, nullptr);
            environment_sampler = VK_NULL_HANDLE;
        }
        if (environment_transfer_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, environment_transfer_pool, nullptr);
            environment_transfer_pool = VK_NULL_HANDLE;
        }
    }

    VkCommandBuffer SharedViewportGpuAssets::Impl::beginMeshCommands() const {
        return shared_viewport_gpu_detail::allocateOneShot(
            context, device, mesh_transfer_pool, "shared.mesh.transfer.command");
    }

    shared_viewport_gpu_detail::OneShotResult
    SharedViewportGpuAssets::Impl::endMeshCommands(VkCommandBuffer cb) const {
        return shared_viewport_gpu_detail::submitOneShot(
            context, device, graphics_queue, mesh_transfer_pool, cb,
            "shared.mesh.oneshot_wait", "Shared mesh");
    }

    VkCommandBuffer SharedViewportGpuAssets::Impl::beginEnvironmentCommands() const {
        return shared_viewport_gpu_detail::allocateOneShot(
            context, device, environment_transfer_pool, "shared.environment.upload.command");
    }

    shared_viewport_gpu_detail::OneShotResult
    SharedViewportGpuAssets::Impl::endEnvironmentCommands(VkCommandBuffer cb) const {
        return shared_viewport_gpu_detail::submitOneShot(
            context, device, graphics_queue, environment_transfer_pool, cb,
            "shared.environment.upload_wait", "Shared environment");
    }

    bool SharedViewportGpuAssets::Impl::createMeshSampler() {
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
        if (!vk_try_bool(
                vkCreateSampler(device, &info, nullptr, &mesh_sampler),
                "vkCreateSampler(device, &info, nullptr, &mesh_sampler)",
                lfs::rendering::formatVulkanDiagnostic(
                    "Shared mesh material sampler creation failed (device={:#x})",
                    vkHandleValue(device)),
                std::source_location::current())) {
            return false;
        }
        context->setDebugObjectName(VK_OBJECT_TYPE_SAMPLER, mesh_sampler, "shared.mesh.material.sampler");
        return true;
    }

    bool SharedViewportGpuAssets::Impl::createMaterialLayout() {
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
        if (!vk_try_bool(
                vkCreateDescriptorSetLayout(device, &mat_info, nullptr, &material_layout),
                "vkCreateDescriptorSetLayout(device, &mat_info, nullptr, &material_layout)",
                lfs::rendering::formatVulkanDiagnostic(
                    "Shared mesh material descriptor-set layout creation failed (device={:#x})",
                    vkHandleValue(device)),
                std::source_location::current())) {
            return false;
        }
        context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                    material_layout,
                                    "shared.mesh.material.descriptor.layout");
        return true;
    }

    VkDescriptorPool SharedViewportGpuAssets::Impl::createMaterialDescriptorPool() {
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
                          std::format("Shared mesh material descriptor-pool creation failed (device={:#x})",
                                      vkHandleValue(device)),
                          __FILE__,
                          __LINE__));
            return VK_NULL_HANDLE;
        }
        context->setDebugObjectNamef(VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                     pool,
                                     "shared.mesh.material.descriptor.pool[{}]",
                                     material_descriptor_pools.size());
        return pool;
    }

    bool SharedViewportGpuAssets::Impl::createInitialMaterialDescriptorPool() {
        const VkDescriptorPool pool = createMaterialDescriptorPool();
        if (pool == VK_NULL_HANDLE) {
            return false;
        }
        material_descriptor_pools.push_back(pool);
        return true;
    }

    bool SharedViewportGpuAssets::Impl::allocateMaterialDescriptor(GpuMaterial& material) {
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
                                             "shared.mesh.material.descriptor[{}]",
                                             vkHandleValue(material.descriptor));
                return true;
            }
            if (allocation_result != VK_ERROR_OUT_OF_POOL_MEMORY &&
                allocation_result != VK_ERROR_FRAGMENTED_POOL) {
                LOG_ERROR("Vulkan: {}",
                          formatVkCheckFailure(
                              "vkAllocateDescriptorSets(device, &alloc, &material.descriptor)",
                              allocation_result,
                              std::format("Shared mesh material descriptor allocation failed (device={:#x}, descriptor_pool={:#x})",
                                          vkHandleValue(device),
                                          vkHandleValue(pool)),
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
                std::format("Shared mesh material descriptor allocation failed for a new pool (device={:#x})",
                            vkHandleValue(device)));
        }
        material.descriptor_pool = pool;
        context->setDebugObjectNamef(VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                     material.descriptor,
                                     "shared.mesh.material.descriptor[{}]",
                                     vkHandleValue(material.descriptor));
        return true;
    }

    bool SharedViewportGpuAssets::Impl::writeBuffer(VmaAllocation alloc,
                                                    const void* src,
                                                    std::size_t bytes) const {
        void* mapped = nullptr;
        const VkResult map_result = vmaMapMemory(allocator, alloc, &mapped);
        if (map_result != VK_SUCCESS) {
            return reportVkFailure(
                "vmaMapMemory(allocator, alloc, &mapped)",
                map_result,
                std::format("Shared mesh buffer allocation could not be mapped (allocator={:#x}, write_size={})",
                            reinterpret_cast<std::uintptr_t>(allocator),
                            bytes));
        }
        if (mapped == nullptr || src == nullptr || bytes == 0) {
            if (mapped != nullptr) {
                vmaUnmapMemory(allocator, alloc);
            }
            return logVkFailure(std::format(
                "Shared mesh buffer write requires mapped memory, a source pointer, and non-zero size (write_size={}) ({}:{})",
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
                std::format("Shared mesh buffer flush failed (flush_size={})", bytes));
        }
        return true;
    }

    bool SharedViewportGpuAssets::Impl::createTexture(const std::uint8_t* rgba,
                                                      int w,
                                                      int h,
                                                      GpuTexture& out,
                                                      std::string_view label) {
        if (rgba == nullptr || w <= 0 || h <= 0) {
            return logVkFailure(std::format(
                "Shared mesh texture upload requires source pixels and positive dimensions (label='{}', observed_width={}, observed_height={}) ({}:{})",
                label,
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
        if (!vk_try_bool(
                vmaCreateImage(allocator, &img, &a, &out.image, &out.alloc, &allocation_info),
                "vmaCreateImage(allocator, &img, &a, &out.image, &out.alloc, &allocation_info)",
                lfs::rendering::formatVulkanDiagnostic(
                    "Shared mesh texture image allocation failed (label='{}', requested_extent={}x{})",
                    label,
                    w,
                    h),
                std::source_location::current())) {
            return false;
        }
        context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE, out.image, "shared.mesh.texture.{}[{}x{}]",
                                     label, w, h);
        out.vram_label = std::format("{}:{}x{}@{}", label, w, h, static_cast<const void*>(&out));
        vmaSetAllocationName(allocator, out.alloc, "Shared mesh texture image");
        lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
            "vulkan.mesh.texture", out.vram_label, static_cast<std::size_t>(allocation_info.size));

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
        VkResult result = vmaCreateBuffer(allocator, &sb, &sa, &staging, &staging_alloc, nullptr);
        if (result != VK_SUCCESS) {
            destroyTexture(out);
            return reportVkFailure(
                "vmaCreateBuffer(allocator, &sb, &sa, &staging, &staging_alloc, nullptr)",
                result,
                std::format("Shared mesh texture staging-buffer allocation failed (label='{}', requested_size={})",
                            label, bytes));
        }
        context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER, staging,
                                     "shared.mesh.texture.{}.upload.staging[{}]", label, bytes);
        void* mapped = nullptr;
        result = vmaMapMemory(allocator, staging_alloc, &mapped);
        if (result != VK_SUCCESS) {
            vmaDestroyBuffer(allocator, staging, staging_alloc);
            destroyTexture(out);
            return reportVkFailure(
                "vmaMapMemory(allocator, staging_alloc, &mapped)",
                result,
                std::format("Shared mesh texture staging allocation could not be mapped (label='{}')", label));
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
                std::format("Shared mesh texture staging flush failed (label='{}')", label));
        }

        VkCommandBuffer cb = beginMeshCommands();
        if (cb == VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator, staging, staging_alloc);
            destroyTexture(out);
            return false;
        }
        cmdImageBarrier2(cb, out.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                         VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), 1};
        vkCmdCopyBufferToImage(cb, staging, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        cmdImageBarrier2(cb, out.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        const auto transfer_result = endMeshCommands(cb);
        if (!transfer_result.completed) {
            if (transfer_result.safe_to_release) {
                vmaDestroyBuffer(allocator, staging, staging_alloc);
                destroyTexture(out);
            } else {
                // The failed submit may still reference both allocations.
                // Detach them so a later material/mesh cleanup cannot free
                // resources that remain owned by the abandoned submission.
                staging = VK_NULL_HANDLE;
                staging_alloc = VK_NULL_HANDLE;
                abandonTexture(out);
            }
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
                std::format("Shared mesh texture image-view creation failed (label='{}', extent={}x{})",
                            label, w, h));
        }
        context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW, out.view,
                                     "shared.mesh.texture.{}[{}x{}].view", label, w, h);
        return true;
    }

    void SharedViewportGpuAssets::Impl::destroyTexture(GpuTexture& texture) const {
        if (texture.view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, texture.view, nullptr);
        }
        if (texture.image != VK_NULL_HANDLE) {
            if (!texture.vram_label.empty()) {
                lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                    "vulkan.mesh.texture", texture.vram_label, 0);
            }
            vmaDestroyImage(allocator, texture.image, texture.alloc);
        }
        texture = {};
    }

    void SharedViewportGpuAssets::Impl::abandonTexture(GpuTexture& texture) const {
        if (texture.image != VK_NULL_HANDLE && !texture.vram_label.empty()) {
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.mesh.texture", texture.vram_label, 0);
        }
        texture = {};
    }

    bool SharedViewportGpuAssets::Impl::createWhitePixel() {
        const std::uint8_t white[4] = {255, 255, 255, 255};
        return createTexture(white, 1, 1, white_pixel, "white_pixel");
    }

    bool SharedViewportGpuAssets::Impl::uploadTextureFromMesh(const lfs::core::MeshData& mesh,
                                                              int tex_index,
                                                              GpuTexture& out,
                                                              std::string_view label) {
        if (tex_index == 0 || tex_index > static_cast<int>(mesh.texture_images.size())) {
            return false;
        }
        const auto& img = mesh.texture_images[static_cast<std::size_t>(tex_index) - 1];
        if (img.pixels.empty() || img.width <= 0 || img.height <= 0) {
            return false;
        }
        std::vector<std::uint8_t> rgba(static_cast<std::size_t>(img.width) * img.height * 4u);
        const int ch = img.channels;
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                const std::size_t src =
                    (static_cast<std::size_t>(y) * img.width + x) * static_cast<std::size_t>(ch);
                const std::size_t dst = (static_cast<std::size_t>(y) * img.width + x) * 4u;
                rgba[dst + 0] = ch >= 1 ? img.pixels[src + 0] : 255;
                rgba[dst + 1] = ch >= 2 ? img.pixels[src + 1] : rgba[dst + 0];
                rgba[dst + 2] = ch >= 3 ? img.pixels[src + 2] : rgba[dst + 0];
                rgba[dst + 3] = ch >= 4 ? img.pixels[src + 3] : 255;
            }
        }
        return createTexture(rgba.data(), img.width, img.height, out,
                             std::format("{}.tex{}", label, tex_index));
    }

    bool SharedViewportGpuAssets::Impl::uploadMaterial(const lfs::core::MeshData& mesh,
                                                       std::size_t material_index,
                                                       GpuMaterial& out) {
        const auto& mat = material_index < mesh.materials.size() ? mesh.materials[material_index]
                                                                 : lfs::core::Material{};
        VkBufferCreateInfo b{};
        b.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        b.size = sizeof(SharedMaterialUbo);
        b.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        b.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo a{};
        a.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        a.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        if (!vk_try_bool(
                vmaCreateBuffer(allocator, &b, &a, &out.ubo, &out.ubo_alloc, nullptr),
                "vmaCreateBuffer(allocator, &b, &a, &out.ubo, &out.ubo_alloc, nullptr)",
                lfs::rendering::formatVulkanDiagnostic(
                    "Shared mesh material UBO allocation failed (material_index={}, requested_size={})",
                    material_index, b.size),
                std::source_location::current())) {
            return false;
        }
        context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER, out.ubo, "shared.mesh.material[{}].ubo",
                                     material_index);

        const bool has_albedo = uploadTextureFromMesh(mesh, mat.albedo_tex, out.albedo, "albedo");
        const bool has_normal = uploadTextureFromMesh(mesh, mat.normal_tex, out.normal, "normal");
        const bool has_mr = uploadTextureFromMesh(mesh, mat.metallic_roughness_tex, out.metallic_roughness,
                                                  "metallic_roughness");
        const bool has_vc = mesh.has_colors();

        SharedMaterialUbo ubo{};
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
        if (!allocateMaterialDescriptor(out)) {
            LOG_ERROR("Shared mesh material descriptor allocation failed (material_index={}, descriptor_pool_count={})",
                      material_index, material_descriptor_pools.size());
            destroyMaterial(out);
            return false;
        }
        context->setDebugObjectNamef(VK_OBJECT_TYPE_DESCRIPTOR_SET, out.descriptor,
                                     "shared.mesh.material[{}].descriptor", material_index);

        VkDescriptorBufferInfo bi{};
        bi.buffer = out.ubo;
        bi.range = sizeof(SharedMaterialUbo);
        const auto pick_view = [&](const GpuTexture& t) {
            return t.view != VK_NULL_HANDLE ? t.view : white_pixel.view;
        };
        std::array<VkDescriptorImageInfo, 3> ii{};
        ii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii[0].imageView = pick_view(out.albedo);
        ii[0].sampler = mesh_sampler;
        ii[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii[1].imageView = pick_view(out.normal);
        ii[1].sampler = mesh_sampler;
        ii[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii[2].imageView = pick_view(out.metallic_roughness);
        ii[2].sampler = mesh_sampler;
        std::array<VkWriteDescriptorSet, 4> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = out.descriptor;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &bi;
        for (int i = 0; i < 3; ++i) {
            writes[static_cast<std::size_t>(i) + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[static_cast<std::size_t>(i) + 1].dstSet = out.descriptor;
            writes[static_cast<std::size_t>(i) + 1].dstBinding = static_cast<std::uint32_t>(i + 1);
            writes[static_cast<std::size_t>(i) + 1].descriptorCount = 1;
            writes[static_cast<std::size_t>(i) + 1].descriptorType =
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[static_cast<std::size_t>(i) + 1].pImageInfo = &ii[static_cast<std::size_t>(i)];
        }
        vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        return true;
    }

    bool SharedViewportGpuAssets::Impl::createDeviceLocalBuffer(VkDeviceSize size,
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
        if (!vk_try_bool(
                vmaCreateBuffer(allocator, &info, &ai, &buffer, &alloc, nullptr),
                "vmaCreateBuffer(allocator, &info, &ai, &buffer, &alloc, nullptr)",
                lfs::rendering::formatVulkanDiagnostic(
                    "Shared mesh device-local buffer allocation failed (requested_size={}, usage={:#x})",
                    size, static_cast<std::uint32_t>(info.usage)),
                std::source_location::current())) {
            return false;
        }
        return true;
    }

    bool SharedViewportGpuAssets::Impl::createStagingBuffer(VkDeviceSize size,
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
                    "Shared mesh staging-buffer allocation failed (requested_size={})", size),
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

    bool SharedViewportGpuAssets::Impl::uploadMesh(const lfs::core::MeshData& mesh, GpuMesh& destination) {
        const std::int64_t vcount = mesh.vertex_count();
        const std::int64_t fcount = mesh.face_count();
        if (vcount <= 0 || fcount <= 0) {
            return logVkFailure(std::format(
                "Shared mesh upload requires positive vertex and face counts (observed_vertices={}, observed_faces={}, generation={}) ({}:{})",
                vcount, fcount, mesh.generation(), __FILE__, __LINE__));
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
        std::vector<SharedMeshVertex> vertices(static_cast<std::size_t>(vcount));
        for (std::int64_t i = 0; i < vcount; ++i) {
            SharedMeshVertex& v = vertices[static_cast<std::size_t>(i)];
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
        const std::size_t vbytes = vertices.size() * sizeof(SharedMeshVertex);
        const std::size_t ibytes = indices.size() * sizeof(std::uint32_t);
        VkBuffer vertex_staging = VK_NULL_HANDLE;
        VmaAllocation vertex_staging_alloc = VK_NULL_HANDLE;
        VkBuffer index_staging = VK_NULL_HANDLE;
        VmaAllocation index_staging_alloc = VK_NULL_HANDLE;
        if (!createDeviceLocalBuffer(vbytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, gpu.vertex_buffer,
                                     gpu.vertex_alloc) ||
            !createDeviceLocalBuffer(ibytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, gpu.index_buffer,
                                     gpu.index_alloc) ||
            !createStagingBuffer(vbytes, vertices.data(), vertex_staging, vertex_staging_alloc) ||
            !createStagingBuffer(ibytes, indices.data(), index_staging, index_staging_alloc)) {
            if (vertex_staging != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, vertex_staging, vertex_staging_alloc);
            }
            if (index_staging != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, index_staging, index_staging_alloc);
            }
            destroyMesh(gpu);
            return false;
        }
        context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER, gpu.vertex_buffer,
                                     "shared.mesh.geometry.vertex[{}]", vbytes);
        context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER, gpu.index_buffer,
                                     "shared.mesh.geometry.index[{}]", ibytes);

        VkCommandBuffer upload_commands = beginMeshCommands();
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
        const auto upload_result = endMeshCommands(upload_commands);
        if (!upload_result.completed) {
            if (upload_result.safe_to_release) {
                vmaDestroyBuffer(allocator, vertex_staging, vertex_staging_alloc);
                vmaDestroyBuffer(allocator, index_staging, index_staging_alloc);
                destroyMesh(gpu);
            } else {
                // See createTexture(): do not release allocations referenced
                // by a submission whose retirement could not be proven.
                vertex_staging = VK_NULL_HANDLE;
                vertex_staging_alloc = VK_NULL_HANDLE;
                index_staging = VK_NULL_HANDLE;
                index_staging_alloc = VK_NULL_HANDLE;
                gpu = {};
            }
            return false;
        }
        vmaDestroyBuffer(allocator, vertex_staging, vertex_staging_alloc);
        vmaDestroyBuffer(allocator, index_staging, index_staging_alloc);
        gpu.total_index_count = static_cast<std::uint32_t>(total_indices);

        const std::size_t mat_count = std::max<std::size_t>(mesh.materials.size(), 1);
        gpu.materials.resize(mat_count);
        for (std::size_t i = 0; i < mat_count; ++i) {
            if (!uploadMaterial(mesh, i, gpu.materials[i])) {
                LOG_ERROR("SharedViewportGpuAssets: failed to upload material {} for mesh", i);
                destroyMesh(gpu);
                return false;
            }
        }
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

    void SharedViewportGpuAssets::Impl::destroyMaterial(GpuMaterial& material) const {
        if (material.descriptor != VK_NULL_HANDLE && material.descriptor_pool != VK_NULL_HANDLE) {
            const VkResult result =
                vkFreeDescriptorSets(device, material.descriptor_pool, 1, &material.descriptor);
            if (result != VK_SUCCESS) {
                LOG_ERROR("Vulkan: {}",
                          formatVkCheckFailure(
                              "vkFreeDescriptorSets(device, material.descriptor_pool, 1, &material.descriptor)",
                              result,
                              std::format("Shared mesh material descriptor-set release failed (descriptor_set={:#x})",
                                          vkHandleValue(material.descriptor)),
                              __FILE__,
                              __LINE__));
            }
        }
        if (material.ubo != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator, material.ubo, material.ubo_alloc);
        }
        destroyTexture(material.albedo);
        destroyTexture(material.normal);
        destroyTexture(material.metallic_roughness);
        material = {};
    }

    void SharedViewportGpuAssets::Impl::destroyMesh(GpuMesh& mesh) const {
        if (mesh.vertex_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator, mesh.vertex_buffer, mesh.vertex_alloc);
        }
        if (mesh.index_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator, mesh.index_buffer, mesh.index_alloc);
        }
        for (auto& mat : mesh.materials) {
            destroyMaterial(mat);
        }
        mesh = {};
    }

    void SharedViewportGpuAssets::Impl::prepareMeshes(const std::vector<VulkanMeshDrawItem>& items) {
        for (const auto& item : items) {
            if (!item.mesh) {
                continue;
            }
            auto it = mesh_cache.find(item.mesh->id());
            if (it == mesh_cache.end()) {
                GpuMesh gpu{};
                if (uploadMesh(*item.mesh, gpu)) {
                    gpu.last_used_epoch = epoch;
                    mesh_cache.emplace(item.mesh->id(), std::move(gpu));
                }
            } else if (it->second.generation != item.mesh->generation()) {
                if (uploadMesh(*item.mesh, it->second)) {
                    it->second.last_used_epoch = epoch;
                }
            } else {
                it->second.last_used_epoch = epoch;
            }
        }
        evictUnusedMeshes();
    }

    bool SharedViewportGpuAssets::Impl::findMesh(const std::uint64_t mesh_id,
                                                 SharedMeshDrawAsset& out) const {
        const auto it = mesh_cache.find(mesh_id);
        if (it == mesh_cache.end() || it->second.vertex_buffer == VK_NULL_HANDLE) {
            return false;
        }
        const GpuMesh& gpu = it->second;
        out.vertex_buffer = gpu.vertex_buffer;
        out.index_buffer = gpu.index_buffer;
        out.total_index_count = gpu.total_index_count;
        out.generation = gpu.generation;
        out.aabb_min = gpu.aabb_min;
        out.aabb_max = gpu.aabb_max;
        out.submeshes.clear();
        out.submeshes.reserve(gpu.submeshes.size());
        for (const auto& sm : gpu.submeshes) {
            out.submeshes.push_back({sm.start_index, sm.index_count, sm.material_index});
        }
        out.material_descriptors.clear();
        out.material_descriptors.reserve(gpu.materials.size());
        for (const auto& mat : gpu.materials) {
            out.material_descriptors.push_back(mat.descriptor);
        }
        return true;
    }

    void SharedViewportGpuAssets::Impl::evictUnusedMeshes() {
        constexpr std::uint64_t kEvictAfter = 120;
        bool has_stale_mesh = false;
        for (const auto& [_, gpu] : mesh_cache) {
            if (epoch - gpu.last_used_epoch > kEvictAfter) {
                has_stale_mesh = true;
                break;
            }
        }
        bool submitted_frames_retired = !has_stale_mesh;
        if (has_stale_mesh) {
            if (context == nullptr) {
                LOG_ERROR("SharedViewportGpuAssets deferred stale mesh eviction because retirement cannot be proven (frame_epoch={}, cache_size={}, eviction_age={})",
                          epoch, mesh_cache.size(), kEvictAfter);
                return;
            }
            submitted_frames_retired = context->waitForSubmittedFrames();
            if (!submitted_frames_retired) {
                LOG_WARN("SharedViewportGpuAssets deferred stale mesh eviction because submitted frames did not retire (frame_epoch={}, cache_size={}, eviction_age={}, error='{}')",
                         epoch, mesh_cache.size(), kEvictAfter, context->lastError());
                return;
            }
        }
        for (auto it = mesh_cache.begin(); it != mesh_cache.end();) {
            if (epoch - it->second.last_used_epoch > kEvictAfter) {
                LFS_VK_DEBUG_ASSERT(
                    submitted_frames_retired,
                    "Deferred shared mesh destruction requires all submitted frames to retire (frames_retired={}, epoch={}, last_used_epoch={}, vertex_buffer={:#x})",
                    submitted_frames_retired,
                    epoch,
                    it->second.last_used_epoch,
                    vkHandleValue(it->second.vertex_buffer));
                destroyMesh(it->second);
                it = mesh_cache.erase(it);
            } else {
                ++it;
            }
        }
    }

    bool SharedViewportGpuAssets::Impl::createEnvironmentSampler() {
        VkSamplerCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.maxLod = 0.0f;
        if (!vk_try_bool(
                vkCreateSampler(device, &info, nullptr, &environment_sampler),
                "vkCreateSampler(device, &info, nullptr, &environment_sampler)",
                lfs::rendering::formatVulkanDiagnostic(
                    "Shared environment sampler creation failed (device={:#x})",
                    vkHandleValue(device)),
                std::source_location::current())) {
            return false;
        }
        context->setDebugObjectName(VK_OBJECT_TYPE_SAMPLER, environment_sampler,
                                    "shared.environment.texture.sampler");
        return true;
    }

    void SharedViewportGpuAssets::Impl::destroyEnvironmentImage(EnvironmentImage& image) const {
        if (image.view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, image.view, nullptr);
        }
        if (image.image != VK_NULL_HANDLE) {
            if (!image.vram_label.empty()) {
                lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                    "vulkan.environment.image", image.vram_label, 0);
            }
            vmaDestroyImage(allocator, image.image, image.alloc);
        }
        image = {};
    }

    void SharedViewportGpuAssets::Impl::abandonEnvironmentImage(EnvironmentImage& image) const {
        if (image.image != VK_NULL_HANDLE && !image.vram_label.empty()) {
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.environment.image", image.vram_label, 0);
        }
        image = {};
    }

    bool SharedViewportGpuAssets::Impl::hasLiveEnvironmentTexture() const {
        return live_environment_index < environment_images.size() &&
               environment_images[live_environment_index].view != VK_NULL_HANDLE;
    }

    SharedEnvironmentTexture SharedViewportGpuAssets::Impl::environmentTexture() const {
        SharedEnvironmentTexture texture{};
        if (!hasLiveEnvironmentTexture()) {
            return texture;
        }
        texture.image_view = environment_images[live_environment_index].view;
        texture.sampler = environment_sampler;
        return texture;
    }

    void SharedViewportGpuAssets::Impl::retireEnvironmentImages(const bool force_wait) {
        const std::size_t in_flight = context != nullptr ? std::max<std::size_t>(1, context->framesInFlight()) : 1;
        if (hasLiveEnvironmentTexture() && environment_last_enabled_epoch > 0 &&
            epoch > environment_last_enabled_epoch &&
            epoch - environment_last_enabled_epoch > in_flight) {
            environment_images[live_environment_index].retire_after_epoch = environment_last_enabled_epoch;
            live_environment_index = std::numeric_limits<std::size_t>::max();
        }

        bool needs_wait = false;
        for (const auto& image : environment_images) {
            if (image.image == VK_NULL_HANDLE) {
                continue;
            }
            if (image.retire_after_epoch == 0) {
                continue;
            }
            if (force_wait || epoch > image.retire_after_epoch) {
                needs_wait = true;
                break;
            }
        }
        if (!needs_wait) {
            return;
        }
        if (context == nullptr || !context->waitForSubmittedFrames()) {
            if (context != nullptr) {
                LOG_WARN("SharedViewportGpuAssets deferred environment image retirement: {}",
                         context->lastError());
            }
            return;
        }
        std::size_t write = 0;
        std::size_t new_live = std::numeric_limits<std::size_t>::max();
        for (std::size_t read = 0; read < environment_images.size(); ++read) {
            auto& image = environment_images[read];
            const bool retiring = image.retire_after_epoch != 0 &&
                                  (force_wait || epoch > image.retire_after_epoch);
            if (retiring) {
                destroyEnvironmentImage(image);
                continue;
            }
            if (read == live_environment_index) {
                new_live = write;
            }
            if (write != read) {
                environment_images[write] = std::move(image);
            }
            ++write;
        }
        environment_images.resize(write);
        live_environment_index = new_live;
    }

    bool SharedViewportGpuAssets::Impl::loadEnvironmentFromPath(const std::filesystem::path& path) {
        if (path.empty()) {
            return false;
        }
        std::filesystem::path resolved = path;
        if (!resolved.is_absolute() && !std::filesystem::exists(resolved)) {
            try {
                resolved = lfs::vis::getAssetPath(lfs::core::path_to_utf8(path));
            } catch (const std::exception& error) {
                LOG_DEBUG("Environment resource lookup failed; trying the assets directory: {}", error.what());
                resolved = lfs::core::getAssetsDir() / path;
            }
        }
        const std::string utf8 = lfs::core::path_to_utf8(resolved);
        auto [source_data, w, h, nch] = lfs::core::load_image_float(resolved);
        if (!source_data) {
            LOG_WARN("SharedViewportGpuAssets: failed to read environment map {}", utf8);
            return false;
        }
        if (w <= 0 || h <= 0 || nch <= 0) {
            lfs::core::free_image_float(source_data);
            return false;
        }

        const std::size_t pixel_count = static_cast<std::size_t>(w) * h;
        std::vector<std::uint16_t> rgba(pixel_count * 4);
        for (std::size_t i = 0; i < pixel_count; ++i) {
            const float r = nch >= 1 ? source_data[i * nch + 0] : 0.0f;
            const float g = nch >= 2 ? source_data[i * nch + 1] : r;
            const float b = nch >= 3 ? source_data[i * nch + 2] : r;
            rgba[i * 4 + 0] = shared_viewport_gpu_detail::floatToHalf(r);
            rgba[i * 4 + 1] = shared_viewport_gpu_detail::floatToHalf(g);
            rgba[i * 4 + 2] = shared_viewport_gpu_detail::floatToHalf(b);
            rgba[i * 4 + 3] = shared_viewport_gpu_detail::floatToHalf(1.0f);
        }
        lfs::core::free_image_float(source_data);

        EnvironmentImage image{};
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
        if (!vk_try_bool(
                vmaCreateImage(allocator, &img, &ai, &image.image, &image.alloc, &allocation_info),
                "vmaCreateImage(allocator, &img, &ai, &image.image, &image.alloc, &allocation_info)",
                lfs::rendering::formatVulkanDiagnostic(
                    "Shared environment image allocation failed (path='{}', requested_extent={}x{})",
                    utf8, w, h),
                std::source_location::current())) {
            return false;
        }
        context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE, image.image, "shared.environment.image[{}x{}]",
                                     w, h);
        vmaSetAllocationName(allocator, image.alloc, "Shared environment image");
        image.vram_label = std::format("env:{}:{}x{}", lfs::core::path_to_utf8(path), w, h);
        lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
            "vulkan.environment.image", image.vram_label, static_cast<std::size_t>(allocation_info.size));

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
        VkResult result = vmaCreateBuffer(allocator, &bi, &sa, &staging, &staging_alloc, nullptr);
        if (result != VK_SUCCESS) {
            destroyEnvironmentImage(image);
            return reportVkFailure(
                "vmaCreateBuffer(allocator, &bi, &sa, &staging, &staging_alloc, nullptr)",
                result,
                std::format("Shared environment staging-buffer allocation failed (path='{}', requested_size={})",
                            utf8, bytes));
        }
        context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER, staging,
                                     "shared.environment.upload.staging[{}]", bytes);
        void* mapped = nullptr;
        result = vmaMapMemory(allocator, staging_alloc, &mapped);
        if (result != VK_SUCCESS) {
            vmaDestroyBuffer(allocator, staging, staging_alloc);
            destroyEnvironmentImage(image);
            return reportVkFailure(
                "vmaMapMemory(allocator, staging_alloc, &mapped)",
                result,
                std::format("Shared environment staging allocation could not be mapped (path='{}')", utf8));
        }
        std::memcpy(mapped, rgba.data(), static_cast<std::size_t>(bytes));
        const VkResult flush_result = vmaFlushAllocation(allocator, staging_alloc, 0, bytes);
        vmaUnmapMemory(allocator, staging_alloc);
        if (flush_result != VK_SUCCESS) {
            vmaDestroyBuffer(allocator, staging, staging_alloc);
            destroyEnvironmentImage(image);
            return reportVkFailure(
                "vmaFlushAllocation(allocator, staging_alloc, 0, bytes)",
                flush_result,
                std::format("Shared environment staging flush failed (path='{}')", utf8));
        }

        VkCommandBuffer cb = beginEnvironmentCommands();
        if (cb == VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator, staging, staging_alloc);
            destroyEnvironmentImage(image);
            return false;
        }
        cmdImageBarrier2(cb, image.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                         VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), 1};
        vkCmdCopyBufferToImage(cb, staging, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        cmdImageBarrier2(cb, image.image, VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        const auto upload_result = endEnvironmentCommands(cb);
        if (!upload_result.completed) {
            if (upload_result.safe_to_release) {
                vmaDestroyBuffer(allocator, staging, staging_alloc);
                destroyEnvironmentImage(image);
            } else {
                staging = VK_NULL_HANDLE;
                staging_alloc = VK_NULL_HANDLE;
                abandonEnvironmentImage(image);
            }
            return false;
        }
        vmaDestroyBuffer(allocator, staging, staging_alloc);

        VkImageViewCreateInfo iv{};
        iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        iv.image = image.image;
        iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
        iv.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        iv.subresourceRange.levelCount = 1;
        iv.subresourceRange.layerCount = 1;
        const VkResult view_result = vkCreateImageView(device, &iv, nullptr, &image.view);
        if (view_result != VK_SUCCESS) {
            destroyEnvironmentImage(image);
            return reportVkFailure(
                "vkCreateImageView(device, &iv, nullptr, &image.view)",
                view_result,
                std::format("Shared environment image-view creation failed (path='{}', extent={}x{})",
                            utf8, w, h));
        }
        context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW, image.view,
                                     "shared.environment.image[{}x{}].view", w, h);
        image.path = path;
        image.last_used_epoch = epoch;

        if (hasLiveEnvironmentTexture()) {
            environment_images[live_environment_index].retire_after_epoch = epoch;
        }
        live_environment_index = environment_images.size();
        environment_images.push_back(std::move(image));
        return true;
    }

    void SharedViewportGpuAssets::Impl::prepareEnvironment(const VulkanEnvironmentParams& params) {
        if (!params.enabled || params.map_path.empty()) {
            retireEnvironmentImages(false);
            return;
        }
        environment_last_enabled_epoch = epoch;
        if (environment_load_failed && params.map_path == environment_failed_path) {
            return;
        }
        if (hasLiveEnvironmentTexture() &&
            environment_images[live_environment_index].path == params.map_path) {
            environment_images[live_environment_index].last_used_epoch = epoch;
            environment_images[live_environment_index].retire_after_epoch = 0;
            retireEnvironmentImages(false);
            return;
        }
        for (std::size_t i = 0; i < environment_images.size(); ++i) {
            if (environment_images[i].path == params.map_path &&
                environment_images[i].view != VK_NULL_HANDLE) {
                if (hasLiveEnvironmentTexture() && live_environment_index != i) {
                    environment_images[live_environment_index].retire_after_epoch = epoch;
                }
                live_environment_index = i;
                environment_images[i].last_used_epoch = epoch;
                environment_images[i].retire_after_epoch = 0;
                environment_load_failed = false;
                environment_failed_path.clear();
                retireEnvironmentImages(false);
                return;
            }
        }
        const bool ok = loadEnvironmentFromPath(params.map_path);
        environment_load_failed = !ok;
        environment_failed_path = ok ? std::filesystem::path{} : params.map_path;
        retireEnvironmentImages(false);
    }

} // namespace lfs::vis
