/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "dispatch.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
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
            const auto modules = vulkan::embedded_training_shaders();
            const auto module = std::ranges::find(modules, name,
                                                  &vulkan::EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan training shader module is missing");
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = module->words.size_bytes();
            shader_info.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &shader_info, nullptr, &shader),
                     "vkCreateShaderModule(training.dispatch)");
            auto pipeline = std::make_shared<Pipeline>();
            pipeline->context = context;
            VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &layout_info, nullptr, &pipeline->layout),
                     "vkCreatePipelineLayout(training.dispatch)");
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            const VkSpecializationMapEntry entry{0, 0, sizeof(specialization)};
            const VkSpecializationInfo constants{1, &entry, sizeof(specialization), &specialization};
            if (specialization != UINT32_MAX)
                stage.pSpecializationInfo = &constants;
            VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            info.stage = stage;
            info.layout = pipeline->layout;
            vk_check(context.get(), vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &info, nullptr, &pipeline->handle),
                     "vkCreateComputePipelines(training.dispatch)");
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cache.emplace(key, pipeline);
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
                  uint32_t group_count, uint32_t specialization) {
        if (!group_count)
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
        context->recorders().record(all_reads, writes, [pipeline, push, group_count](VkCommandBuffer command) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
            vkCmdPushConstants(command, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(command, group_count, 1, 1); }, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_WHOLE_SIZE, block);
    }
    uint32_t groups(size_t work) { return vk::dispatch_groups(*acquire_vulkan_context(), work); }
} // namespace lfs::training::vulkan
