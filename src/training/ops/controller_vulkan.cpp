/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/ppisp_vulkan.hpp"

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
            uint64_t grad_output, activation, weight, weight_gradient, bias_gradient, grad_input;
            uint32_t count;
            int32_t m, n;
            float exposure_prior;
        };
        static_assert(sizeof(Push) == 64);

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

        std::shared_ptr<Pipeline> pipeline_for(const std::shared_ptr<VulkanContext>& context, uint32_t op) {
            static std::mutex mutex;
            static std::map<std::pair<uint64_t, uint32_t>, std::shared_ptr<Pipeline>> cache;
            const auto key = std::pair{context->context_id(), op};
            std::lock_guard lock(mutex);
            if (const auto it = cache.find(key); it != cache.end())
                return it->second;
            const auto modules = vulkan::embedded_training_shaders();
            const auto module = std::ranges::find(modules, std::string_view("controller"), &vulkan::EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan controller shader module is missing");
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = module->words.size_bytes();
            shader_info.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &shader_info, nullptr, &shader), "vkCreateShaderModule(training.controller)");
            auto pipeline = std::make_shared<Pipeline>();
            pipeline->context = context;
            VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &layout_info, nullptr, &pipeline->layout), "vkCreatePipelineLayout(training.controller)");
            const VkSpecializationMapEntry entry{0, 0, sizeof(op)};
            VkSpecializationInfo specialization{1, &entry, sizeof(op), &op};
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            stage.pSpecializationInfo = &specialization;
            VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            info.stage = stage;
            info.layout = pipeline->layout;
            vk_check(context.get(), vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &info, nullptr, &pipeline->handle), "vkCreateComputePipelines(training.controller)");
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cache.emplace(key, pipeline);
            vulkan::release_at_shutdown(*context, mutex, cache);
            return pipeline;
        }

        StorageRef ref(const Tensor& t) {
            const auto s = storage_ref(t);
            LFS_ASSERT_MSG(s.backend == core::GpuBackend::Vulkan, "Vulkan controller received storage from another backend");
            return s;
        }
        void launch(const std::shared_ptr<VulkanContext>& context, const Push& p, uint32_t op,
                    const std::vector<StorageRef>& reads, const std::vector<StorageRef>& writes) {
            if (!p.count)
                return;
            const auto pipeline = pipeline_for(context, op);
            context->recorders().record(reads, writes, [&](VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
                vkCmdPushConstants(command, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
                vkCmdDispatch(command, (p.count + 255) / 256, 1, 1); }, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_WHOLE_SIZE, pipeline);
        }

        void prepare_input(In features, Out fc_input, float exposure_prior) {
            constexpr size_t feature_count = 1600;
            LFS_ASSERT_MSG(fc_input.numel() >= feature_count + 1, "controller input must hold 1601 values");
            if (features.is_valid()) {
                LFS_ASSERT_MSG(features.numel() >= feature_count, "controller features must hold 1600 values");
                fc_input.flatten().slice(0, 0, feature_count).copy_from(features.flatten().slice(0, 0, feature_count));
                if (exposure_prior == 1.0f)
                    return;
            }
            Push p{};
            p.grad_input = vk::address(ref(fc_input));
            p.count = 1;
            p.exposure_prior = exposure_prior;
            const auto ctx = acquire_vulkan_context();
            launch(ctx, p, 0, features.is_valid() ? std::vector<StorageRef>{ref(features), ref(fc_input)} : std::vector<StorageRef>{ref(fc_input)}, {ref(fc_input)});
        }

        void backward_layer(In grad_output, In activation, In weight, Out weight_gradient,
                            Out bias_gradient, Out grad_input) {
            const size_t m = grad_output.numel(), n = activation.numel();
            LFS_ASSERT_MSG(m && n && m <= INT32_MAX && n <= INT32_MAX && m * n <= UINT32_MAX,
                           "controller layer dimensions exceed Vulkan indexing");
            LFS_ASSERT_MSG(weight.numel() == m * n && weight_gradient.numel() == m * n && bias_gradient.numel() == m,
                           "controller layer binding shape mismatch");
            LFS_ASSERT_MSG(!grad_input.is_valid() || grad_input.numel() == n, "controller input gradient shape mismatch");
            Push p{};
            const auto g = ref(grad_output), a = ref(activation), w = ref(weight), wg = ref(weight_gradient), bg = ref(bias_gradient);
            const StorageRef gi = grad_input.is_valid() ? ref(grad_input) : StorageRef{};
            p.grad_output = vk::address(g);
            p.activation = vk::address(a);
            p.weight = vk::address(w);
            p.weight_gradient = vk::address(wg);
            p.bias_gradient = vk::address(bg);
            p.grad_input = grad_input.is_valid() ? vk::address(gi) : 0;
            p.count = static_cast<uint32_t>(std::max(m * n, std::max(m, grad_input.is_valid() ? n : size_t{0})));
            p.m = static_cast<int32_t>(m);
            p.n = static_cast<int32_t>(n);
            std::vector<StorageRef> reads{g, a, w};
            std::vector<StorageRef> writes{wg, bg};
            if (grad_input.is_valid())
                writes.push_back(gi);
            launch(acquire_vulkan_context(), p, 1, reads, writes);
        }

        const ControllerOps kControllerOps{.prepare_input = prepare_input, .backward_layer = backward_layer};
    } // namespace
    const gpu_ops::ControllerOps& vulkan_controller_ops() { return kControllerOps; }
} // namespace lfs::training
