/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/extra_loss_vulkan.hpp"

#include "core/assert.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_memory.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "training_shader_table.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <ranges>
#include <string_view>
#include <vector>

namespace lfs::training {
    namespace {
        using namespace lfs::core::internal;
        using namespace lfs::gpu_ops;

        struct Push {
            uint64_t source;
            uint64_t gradient;
            uint64_t loss;
            uint64_t scratch;
            uint64_t z;
            uint64_t u;
            uint32_t count;
            uint32_t partial_count;
            float weight;
            float grad_loss;
            uint32_t kind;
            uint32_t accumulate;
        };
        static_assert(sizeof(Push) == 72);
        static_assert(offsetof(Push, count) == 48);
        static_assert(offsetof(Push, accumulate) == 68);

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

        std::shared_ptr<Pipeline> pipeline_for(const std::shared_ptr<VulkanContext>& context,
                                               const uint32_t operation) {
            static std::mutex mutex;
            static std::map<std::pair<uint64_t, uint32_t>, std::shared_ptr<Pipeline>> cache;
            const auto key = std::pair{context->context_id(), operation};
            std::lock_guard lock(mutex);
            if (const auto found = cache.find(key); found != cache.end())
                return found->second;

            const auto modules = vulkan::embedded_training_shaders();
            const auto module = std::ranges::find(modules, std::string_view("extra_loss"),
                                                  &vulkan::EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan extra-loss shader module is missing");
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = module->words.size_bytes();
            shader_info.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &shader_info, nullptr, &shader),
                     "vkCreateShaderModule(training.extra_loss)");

            auto pipeline = std::make_shared<Pipeline>();
            pipeline->context = context;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(context->physical_device(), &properties);
            LFS_ASSERT_MSG(sizeof(Push) <= properties.limits.maxPushConstantsSize,
                           "Vulkan extra-loss parameters exceed device push-constant limit");
            VkPushConstantRange range{};
            range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            range.offset = 0;
            range.size = sizeof(Push);
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &layout_info, nullptr, &pipeline->layout),
                     "vkCreatePipelineLayout(training.extra_loss)");

            const VkSpecializationMapEntry entry{0, 0, sizeof(uint32_t)};
            VkSpecializationInfo specialization{};
            specialization.mapEntryCount = 1;
            specialization.pMapEntries = &entry;
            specialization.dataSize = sizeof(operation);
            specialization.pData = &operation;
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            stage.pSpecializationInfo = &specialization;
            VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline_info.stage = stage;
            pipeline_info.layout = pipeline->layout;
            vk_check(context.get(), vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &pipeline_info, nullptr, &pipeline->handle),
                     "vkCreateComputePipelines(training.extra_loss)");
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cache.emplace(key, pipeline);
            return pipeline;
        }

        StorageRef ref(const Tensor& tensor) {
            const StorageRef storage = storage_ref(tensor);
            LFS_ASSERT_MSG(storage.backend == core::GpuBackend::Vulkan,
                           "Vulkan extra-loss op received storage from another backend");
            return storage;
        }

        void launch(const Push& push, const uint32_t operation,
                    const std::span<const StorageRef> reads,
                    const std::span<const StorageRef> writes, const uint32_t group_count) {
            if (group_count == 0)
                return;
            const auto context = acquire_vulkan_context();
            const auto pipeline = pipeline_for(context, operation);
            context->recorders().record(reads, writes, [&](const VkCommandBuffer command) {
                                            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                              pipeline->handle);
                                            vkCmdPushConstants(command, pipeline->layout,
                                                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                                               sizeof(push), &push);
                                            vkCmdDispatch(command, group_count, 1, 1); }, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_WHOLE_SIZE, pipeline);
        }

        void regularize(In raw, Out gradient, Out loss, Out reduction_temp,
                        const Regularizer kind, const float weight) {
            const size_t count = raw.numel();
            if (count == 0 || weight == 0.f)
                return;
            LFS_ASSERT_MSG(count <= UINT32_MAX, "Vulkan regularizer input exceeds 32-bit indexing");
            const uint32_t blocks = static_cast<uint32_t>(std::min((count + 255) / 256, size_t{1024}));
            const StorageRef source = ref(raw);
            const StorageRef result = ref(loss);
            const StorageRef scratch = ref(reduction_temp);
            const StorageRef grad = gradient.is_valid() ? ref(gradient) : StorageRef{};
            const Push push{vk::address(source), gradient.is_valid() ? vk::address(grad) : vk::address(source),
                            vk::address(result), vk::address(scratch), 0, 0,
                            static_cast<uint32_t>(count), blocks, weight, 0,
                            kind == Regularizer::Scale ? 0u : 1u, gradient.is_valid() ? 1u : 0u};
            std::vector<StorageRef> reads{source};
            std::vector<StorageRef> writes{scratch};
            if (gradient.is_valid()) {
                reads.push_back(grad);
                writes.push_back(grad);
            }
            launch(push, 0, reads, writes, blocks);

            const std::array reduction_reads{scratch};
            const std::array reduction_writes{result};
            launch(push, 1, reduction_reads, reduction_writes, 1);
        }

        void admm(In sigmoid, In z, In u, Out gradient,
                  const float rho, const float grad_loss, const bool accumulate) {
            const size_t count = sigmoid.numel();
            if (count == 0)
                return;
            LFS_ASSERT_MSG(count <= UINT32_MAX, "Vulkan ADMM input exceeds 32-bit indexing");
            const StorageRef source = ref(sigmoid), z_storage = ref(z), u_storage = ref(u), grad = ref(gradient);
            const Push push{vk::address(source), vk::address(grad), 0, 0,
                            vk::address(z_storage), vk::address(u_storage),
                            static_cast<uint32_t>(count), 0, rho, grad_loss, 0, accumulate ? 1u : 0u};
            const std::array reads{source, z_storage, u_storage, grad};
            const std::array writes{grad};
            const auto context = acquire_vulkan_context();
            launch(push, 2, reads, writes, vk::dispatch_groups(*context, count));
        }

        const ExtraLossOps kVulkanExtraLossOps{
            .regularize = regularize,
            .admm = admm,
        };
    } // namespace

    const lfs::gpu_ops::ExtraLossOps& vulkan_extra_loss_ops() { return kVulkanExtraLossOps; }
} // namespace lfs::training
