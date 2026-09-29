/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/mcmc_vulkan.hpp"

#include "core/assert.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "core/tensor_completion.hpp"
#include "core/vulkan_helpers.hpp"
#include "training_shader_table.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <ranges>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace lfs::training {
    namespace {
        using namespace lfs::core;
        using namespace lfs::core::internal;
        using namespace lfs::gpu_ops;

        constexpr uint32_t kRelocationMax = 51;

        struct McmcPush {
            uint64_t a, b, c, d, e, f, g, h;
            uint64_t seed;
            uint32_t count, count2, mode, reserved;
            float first, second;
        };
        static_assert(sizeof(McmcPush) == 96);
        static_assert(offsetof(McmcPush, seed) == 64);
        static_assert(offsetof(McmcPush, count) == 72);
        static_assert(offsetof(McmcPush, first) == 88);

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
            using namespace lfs::training::vulkan;
            static std::mutex mutex;
            static std::map<std::pair<uint64_t, uint32_t>, std::shared_ptr<Pipeline>> cache;
            const auto key = std::pair{context->context_id(), operation};
            std::lock_guard lock(mutex);
            if (const auto found = cache.find(key); found != cache.end())
                return found->second;
            const auto modules = embedded_training_shaders();
            const auto module = std::ranges::find(modules, std::string_view("mcmc"), &EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan MCMC shader module is missing");
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = module->words.size_bytes();
            shader_info.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &shader_info, nullptr, &shader),
                     "vkCreateShaderModule(training.mcmc)");
            auto pipeline = std::make_shared<Pipeline>();
            pipeline->context = context;
            VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(McmcPush)};
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &layout_info, nullptr, &pipeline->layout),
                     "vkCreatePipelineLayout(training.mcmc)");
            const VkSpecializationMapEntry entry{0, 0, sizeof(operation)};
            const VkSpecializationInfo specialization{1, &entry, sizeof(operation), &operation};
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            stage.pSpecializationInfo = &specialization;
            VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline_info.stage = stage;
            pipeline_info.layout = pipeline->layout;
            vk_check(context.get(), vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &pipeline_info, nullptr, &pipeline->handle),
                     "vkCreateComputePipelines(training.mcmc)");
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cache.emplace(key, pipeline);
            return pipeline;
        }

        void dispatch(const std::shared_ptr<VulkanContext>& context, const uint32_t operation,
                      const McmcPush& push, const std::span<const StorageRef> reads,
                      const std::span<const StorageRef> writes, const size_t count,
                      std::shared_ptr<void> lifetime = {}) {
            if (count == 0)
                return;
            const auto pipeline = pipeline_for(context, operation);
            context->recorders().record(reads, writes, [&](const VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
                vkCmdPushConstants(command, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                vkCmdDispatch(command, vk::dispatch_groups(*context, count), 1, 1); }, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_WHOLE_SIZE, lifetime ? std::move(lifetime) : pipeline);
        }

        // Host copy of the relocation coefficients, uploaded per relocation: a
        // cached GPU table would outlive the Vulkan backend across shutdowns.
        std::mutex coefficient_mutex;
        std::vector<float> coefficient_values;

        Tensor coefficient_table() {
            std::lock_guard lock(coefficient_mutex);
            if (coefficient_values.empty())
                throw std::logic_error("Vulkan MCMC relocation coefficients are not initialized");
            return Tensor::from_vector(coefficient_values, {coefficient_values.size()}, Device::GPU);
        }

        void initialize(const int n_max) {
            if (n_max < 0 || n_max > static_cast<int>(kRelocationMax))
                throw std::invalid_argument("MCMC relocation coefficient limit must be in [0, 51]");
            std::vector<float> coefficients(kRelocationMax * kRelocationMax, 0.0f);
            for (int n = 0; n < n_max; ++n) {
                float binomial = 1.0f;
                for (int k = 0; k <= n; ++k) {
                    const float sign = (k % 2 == 0) ? 1.0f : -1.0f;
                    const float inverse_sqrt = static_cast<float>(1.0 / std::sqrt(double(k + 1)));
                    coefficients[n * kRelocationMax + k] = binomial * sign * inverse_sqrt;
                    if (k < n)
                        binomial *= float(n - k) / float(k + 1);
                }
            }
            std::lock_guard lock(coefficient_mutex);
            coefficient_values = std::move(coefficients);
        }

        void relocate(const Tensor& opacity, const Tensor& scales, const Tensor& ratios,
                      Tensor& new_opacity, Tensor& new_scales, const float min_opacity) {
            const size_t count = opacity.numel();
            if (count == 0)
                return;
            const auto context = acquire_vulkan_context();
            const Tensor coefficients = coefficient_table();
            const McmcPush push{.a = vk::address(storage_ref(opacity)), .b = vk::address(storage_ref(scales)), .c = vk::address(storage_ref(ratios)), .d = vk::address(storage_ref(new_opacity)), .e = vk::address(storage_ref(new_scales)), .f = vk::address(storage_ref(coefficients)), .count = vk::checked_u32(count, "MCMC relocation count exceeds uint32"), .first = min_opacity};
            const std::array reads{storage_ref(opacity), storage_ref(scales), storage_ref(ratios), storage_ref(coefficients)};
            const std::array writes{storage_ref(new_opacity), storage_ref(new_scales)};
            dispatch(context, 0, push, reads, writes, count, std::make_shared<Tensor>(coefficients));
        }

        void noise(const Tensor& opacity, const Tensor& scales, const Tensor& quats, const Tensor& frozen,
                   Tensor& means, const uint64_t seed, const float lr) {
            const size_t count = means.size(0);
            if (count == 0)
                return;
            const auto context = acquire_vulkan_context();
            const StorageRef frozen_ref = frozen.is_valid() ? storage_ref(frozen) : StorageRef{};
            const McmcPush push{.a = vk::address(storage_ref(opacity)), .b = vk::address(storage_ref(scales)), .c = vk::address(storage_ref(quats)), .d = frozen.is_valid() ? vk::address(frozen_ref) : 0, .e = vk::address(storage_ref(means)), .seed = seed, .count = vk::checked_u32(count, "MCMC noise count exceeds uint32"), .count2 = frozen.is_valid() ? vk::checked_u32(frozen.numel(), "MCMC frozen-mask count exceeds uint32") : 0u, .first = lr};
            std::vector<StorageRef> reads{storage_ref(opacity), storage_ref(scales), storage_ref(quats)};
            if (frozen.is_valid())
                reads.push_back(frozen_ref);
            const std::array writes{storage_ref(means)};
            dispatch(context, 1, push, reads, writes, count);
        }

        void copy_rows(const Tensor& source, const Tensor& destination, const McmcRows& rows) {
            const size_t count = destination.numel(), row_count = rows.means.size(0);
            if (count == 0)
                return;
            const auto context = acquire_vulkan_context();
            const McmcPush push{.a = vk::address(storage_ref(source)), .b = vk::address(storage_ref(destination)), .c = vk::address(storage_ref(rows.means)), .d = vk::address(storage_ref(rows.sh0)), .e = vk::address(storage_ref(rows.raw_scales)), .f = vk::address(storage_ref(rows.raw_quats)), .g = vk::address(storage_ref(rows.raw_opacity)), .count = vk::checked_u32(count, "MCMC row-copy count exceeds uint32"), .count2 = vk::checked_u32(row_count, "MCMC row count exceeds uint32")};
            const std::array reads{storage_ref(source), storage_ref(destination), storage_ref(rows.means),
                                   storage_ref(rows.sh0), storage_ref(rows.raw_scales), storage_ref(rows.raw_quats),
                                   storage_ref(rows.raw_opacity)};
            const std::array writes{storage_ref(rows.means), storage_ref(rows.sh0), storage_ref(rows.raw_scales),
                                    storage_ref(rows.raw_quats), storage_ref(rows.raw_opacity)};
            dispatch(context, 3, push, reads, writes, count);
        }

        void update_rows(const Tensor& indices, const Tensor& scales, const Tensor& opacity,
                         Tensor& scales_out, Tensor& opacity_out) {
            const size_t count = indices.numel();
            if (count == 0)
                return;
            const auto context = acquire_vulkan_context();
            const McmcPush push{.a = vk::address(storage_ref(indices)), .b = vk::address(storage_ref(scales)), .c = vk::address(storage_ref(opacity)), .d = vk::address(storage_ref(scales_out)), .e = vk::address(storage_ref(opacity_out)), .count = vk::checked_u32(count, "MCMC update count exceeds uint32"), .count2 = vk::checked_u32(scales_out.size(0), "MCMC row count exceeds uint32")};
            const std::array reads{storage_ref(indices), storage_ref(scales), storage_ref(opacity)};
            const std::array writes{storage_ref(scales_out), storage_ref(opacity_out)};
            dispatch(context, 2, push, reads, writes, count);
        }

        void sample(const Tensor& weights, const Tensor& opacity, const Tensor& scales, const Tensor& alive,
                    Tensor& indices, Tensor& sampled_opacity, Tensor& sampled_scales,
                    const SampleDomain domain, const uint64_t seed) {
            const size_t samples = indices.numel();
            const size_t domain_count = domain == SampleDomain::AliveIndices ? alive.numel() : weights.numel();
            if (samples == 0 || domain_count == 0)
                return;
            const auto context = acquire_vulkan_context();
            Tensor cdf;
            Tensor sample_weights;
            if (domain == SampleDomain::AliveIndices) {
                sample_weights = Tensor::empty({domain_count}, Device::GPU, DataType::Float32);
                const McmcPush gather_push{.a = vk::address(storage_ref(weights)), .b = vk::address(storage_ref(alive)), .c = vk::address(storage_ref(sample_weights)), .count = vk::checked_u32(domain_count, "MCMC alive count exceeds uint32")};
                const std::array gather_reads{storage_ref(weights), storage_ref(alive)};
                const std::array gather_writes{storage_ref(sample_weights)};
                dispatch(context, 6, gather_push, gather_reads, gather_writes, domain_count);
            } else {
                sample_weights = weights.contiguous();
            }
            cdf = sample_weights.cumsum(0);
            const McmcPush push{.a = vk::address(storage_ref(cdf)), .b = domain == SampleDomain::AliveIndices ? vk::address(storage_ref(alive)) : 0, .c = vk::address(storage_ref(opacity)), .d = vk::address(storage_ref(scales)), .e = vk::address(storage_ref(indices)), .f = vk::address(storage_ref(sampled_opacity)), .g = vk::address(storage_ref(sampled_scales)), .seed = seed, .count = vk::checked_u32(samples, "MCMC sample count exceeds uint32"), .count2 = vk::checked_u32(domain_count, "MCMC domain count exceeds uint32"), .mode = domain == SampleDomain::AliveIndices ? 1u : 0u};
            std::vector<StorageRef> reads{storage_ref(cdf), storage_ref(opacity), storage_ref(scales)};
            if (domain == SampleDomain::AliveIndices)
                reads.push_back(storage_ref(alive));
            const std::array writes{storage_ref(indices), storage_ref(sampled_opacity), storage_ref(sampled_scales)};
            dispatch(context, 4, push, reads, writes, samples,
                     std::make_shared<Tensor>(std::move(cdf)));
        }

        void fold_error(Tensor& error_max, Tensor& densification) {
            const size_t count = error_max.numel();
            if (count == 0)
                return;
            const auto context = acquire_vulkan_context();
            const McmcPush push{.a = vk::address(storage_ref(error_max)), .b = vk::address(storage_ref(densification)), .count = vk::checked_u32(count, "MCMC error count exceeds uint32")};
            const std::array reads{storage_ref(error_max), storage_ref(densification)};
            const std::array writes{storage_ref(error_max), storage_ref(densification)};
            dispatch(context, 5, push, reads, writes, count);
        }

        const McmcOps kVulkanMcmcOps{.initialize = initialize,
                                     .relocate = relocate,
                                     .noise = noise,
                                     .copy_rows = copy_rows,
                                     .update_rows = update_rows,
                                     .sample = sample,
                                     .fold_error = fold_error};
    } // namespace

    const McmcOps& vulkan_mcmc_ops() { return kVulkanMcmcOps; }
} // namespace lfs::training
