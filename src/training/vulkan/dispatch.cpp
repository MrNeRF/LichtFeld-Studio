/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "dispatch.hpp"
#include "core/logger.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "fast_state.hpp"
#include "training_shader_table.hpp"
#include <map>
#include <mutex>
#include <ranges>
#include <tuple>
namespace lfs::training::vulkan {
    using namespace core::internal;
    namespace {
        struct Push {
            uint64_t parameters;
        };
        static_assert(sizeof(Push) == 8);
        struct ShaderModule {
            VkDevice device;
            VkShaderModule handle;
            ~ShaderModule() { vkDestroyShaderModule(device, handle, nullptr); }
        };
        struct Pipeline {
            std::shared_ptr<VulkanContext> context;
            VkPipelineLayout layout = VK_NULL_HANDLE;
            VkPipeline handle = VK_NULL_HANDLE;
            ~Pipeline() {
                if (!context || context->device() == VK_NULL_HANDLE)
                    return;
                if (handle != VK_NULL_HANDLE)
                    vkDestroyPipeline(context->device(), handle, nullptr);
                if (layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(context->device(), layout, nullptr);
            }
        };

        std::shared_ptr<Pipeline> pipeline_for(const std::shared_ptr<VulkanContext>& context, std::string_view name, uint32_t specialization) {
            static std::mutex mutex;
            static std::map<std::tuple<uint64_t, std::string, uint32_t>, std::shared_ptr<Pipeline>> cache;
            const auto key = std::tuple{context->context_id(), std::string(name), specialization};
            std::lock_guard lock(mutex);
            if (const auto found = cache.find(key); found != cache.end())
                return found->second;
            const bool fast = name == "fast_forward" || name == "fast_backward" || name == "fast_backward_atomic";
            const bool fast_raster = fast && (specialization & 15u) == (name == "fast_forward" ? 5u : 0u);
            const bool gsplat_raster =
                (name == "gsplat_backward" || name == "gsplat_backward_atomic") && specialization == 0;
            VkPhysicalDeviceSubgroupSizeControlProperties size_properties{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            properties.pNext = &size_properties;
            vkGetPhysicalDeviceProperties2(context->physical_device(), &properties);
            // Raster kernels with a 32-lane layout require wave32 wherever the
            // device lets a compute pipeline require it.
            const bool wave32 = context->caps().subgroup_size_control &&
                                context->caps().min_subgroup_size <= 32 && context->caps().max_subgroup_size >= 32 &&
                                (size_properties.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
                                size_properties.maxComputeWorkgroupSubgroups >= 4;
            // Every Fast pipeline of a device drops a requested quadrant layout
            // together, so forward and backward always agree on it.
            if (fast && !wave32)
                specialization &= ~fast_quadrants;
            const bool quadrant = fast && (specialization & fast_quadrants) != 0;
            const std::string_view module_name = fast_raster && name == "fast_forward" && quadrant ? "fast_forward_quadrant" : name;
            const auto modules = vulkan::embedded_training_shaders();
            const auto module = std::ranges::find(modules, module_name,
                                                  &vulkan::EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan training shader module is missing");
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = module->words.size_bytes();
            shader_info.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &shader_info, nullptr, &shader),
                     "vkCreateShaderModule(training.dispatch)");
            const ShaderModule shader_lifetime{context->device(), shader};
            auto pipeline = std::make_shared<Pipeline>();
            pipeline->context = context;
            VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            const VkResult layout_created = vkCreatePipelineLayout(context->device(), &layout_info, nullptr, &pipeline->layout);
            if (layout_created != VK_SUCCESS)
                pipeline->layout = VK_NULL_HANDLE; // Failed creation leaves the output undefined.
            vk_check(context.get(), layout_created, "vkCreatePipelineLayout(training.dispatch)");
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            VkPipelineShaderStageRequiredSubgroupSizeCreateInfo required_size{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
            // gsplat keeps its shuffle path on default 64-lane devices only.
            if ((fast_raster && quadrant) || (gsplat_raster && wave32 && context->caps().subgroup_size == 64)) {
                required_size.requiredSubgroupSize = 32;
                stage.pNext = &required_size;
            }
            const VkSpecializationMapEntry entry{0, 0, sizeof(specialization)};
            const VkSpecializationInfo constants{1, &entry, sizeof(specialization), &specialization};
            if (specialization != UINT32_MAX)
                stage.pSpecializationInfo = &constants;
            VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            info.stage = stage;
            info.layout = pipeline->layout;
            auto result = vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &info, nullptr, &pipeline->handle);
            if (result != VK_SUCCESS && info.stage.pNext && !fast && result != VK_ERROR_DEVICE_LOST) {
                LOG_WARN("Vulkan training kernel {}: wave32 pipeline rejected (VkResult {}); falling back to the device default subgroup size",
                         name, static_cast<int>(result));
                if (pipeline->handle != VK_NULL_HANDLE) {
                    vkDestroyPipeline(context->device(), pipeline->handle, nullptr);
                    pipeline->handle = VK_NULL_HANDLE;
                }
                info.stage.pNext = nullptr;
                result = vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &info, nullptr, &pipeline->handle);
            }
            vk_check(context.get(), result, "vkCreateComputePipelines(training.dispatch)");
            if (info.stage.pNext)
                LOG_DEBUG("Vulkan training kernel {} stage {} uses required subgroup size 32", name, specialization);
            if (info.stage.pNext && name == "fast_forward")
                LOG_INFO("Vulkan Fast raster forward: using subgroup size 32");
            cache.emplace(key, pipeline);
            release_at_shutdown(*context, mutex, cache);
            return pipeline;
        }

    } // namespace
    StorageRef ref(const core::Tensor& tensor) {
        auto storage = storage_ref(tensor);
        LFS_ASSERT_MSG(storage.backend == core::GpuBackend::Vulkan,
                       "Vulkan training received storage from another backend");
        return storage;
    }
    uint64_t address(const core::Tensor& tensor) {
        return tensor.is_valid() && tensor.numel() ? vk::address(ref(tensor)) : 0;
    }
    void dispatch(std::string_view name, const void* parameters, size_t bytes,
                  std::span<const StorageRef> reads, std::span<const StorageRef> writes,
                  uint32_t group_count, uint32_t specialization, StorageRef indirect, VkDeviceSize indirect_offset) {
        if (!group_count && !indirect.meta)
            return;
        auto context = acquire_vulkan_context();
        auto block = std::make_shared<vk::ScopedAllocation>(*context, bytes);
        context->memory().copy_host_to_device(CopyRequest{
            .src = raw_storage_ref(const_cast<void*>(parameters)),
            .dst = block->storage(),
            .bytes = bytes,
            .synchronous = false});
        auto pipeline = pipeline_for(context, name, specialization);
        const Push push{vk::address(block->storage())};
        std::vector<StorageRef> all_reads(reads.begin(), reads.end());
        all_reads.push_back(block->storage());
        if (indirect.meta)
            all_reads.push_back(indirect);
        context->recorders().record(all_reads, writes, [pipeline, push, group_count, indirect, indirect_offset](VkCommandBuffer command) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
            vkCmdPushConstants(command, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            if (indirect.meta)
                vkCmdDispatchIndirect(command, VulkanMemory::buffer_for(indirect), VulkanMemory::offset_for(indirect) + indirect_offset);
            else vkCmdDispatch(command, group_count, 1, 1); }, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | (indirect.meta ? VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT : 0), VK_WHOLE_SIZE, block);
    }
    uint32_t groups(size_t work) { return vk::dispatch_groups(*acquire_vulkan_context(), work); }
} // namespace lfs::training::vulkan
