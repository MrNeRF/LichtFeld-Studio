/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/masks_vulkan.hpp"

#include "core/assert.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_memory.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "training_shader_table.hpp"

#include <algorithm>
#include <array>
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
            uint64_t alpha, mask, roi, output, partials;
            uint32_t count, mask_bytes, has_roi, mode;
            float power, grad_scale, keep_min, segment_min;
        };
        static_assert(sizeof(Push) == 72);
        static_assert(offsetof(Push, count) == 40);
        static_assert(offsetof(Push, power) == 56);

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
            const auto module = std::ranges::find(modules, std::string_view("masks"),
                                                  &vulkan::EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan mask shader module is missing");
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = module->words.size_bytes();
            shader_info.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &shader_info, nullptr, &shader),
                     "vkCreateShaderModule(training.masks)");

            auto pipeline = std::make_shared<Pipeline>();
            pipeline->context = context;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(context->physical_device(), &properties);
            LFS_ASSERT_MSG(sizeof(Push) <= properties.limits.maxPushConstantsSize,
                           "Vulkan mask parameters exceed device push-constant limit");
            VkPushConstantRange range{};
            range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            range.size = sizeof(Push);
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &layout_info, nullptr, &pipeline->layout),
                     "vkCreatePipelineLayout(training.masks)");

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
                     "vkCreateComputePipelines(training.masks)");
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cache.emplace(key, pipeline);
            vulkan::release_at_shutdown(*context, mutex, cache);
            return pipeline;
        }

        StorageRef ref(const Tensor& tensor) {
            const StorageRef storage = storage_ref(tensor);
            LFS_ASSERT_MSG(storage.backend == core::GpuBackend::Vulkan,
                           "Vulkan mask op received storage from another backend");
            return storage;
        }

        void launch(const std::shared_ptr<VulkanContext>& context, const Push& push, const uint32_t operation,
                    const std::vector<StorageRef>& reads, const std::vector<StorageRef>& writes,
                    const uint32_t group_count) {
            if (group_count == 0)
                return;
            const auto pipeline = pipeline_for(context, operation);
            context->recorders().record(reads, writes, [&](const VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
                vkCmdPushConstants(command, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(push), &push);
                vkCmdDispatch(command, group_count, 1, 1); }, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_WHOLE_SIZE, pipeline);
        }

        bool has_roi(In roi) { return roi.is_valid() && roi.numel() > 0; }
        bool byte_mask(In mask) {
            return mask.dtype() == core::DataType::UInt8 || mask.dtype() == core::DataType::Bool;
        }
        uint32_t pixel_count(In mask) {
            LFS_ASSERT_MSG(mask.ndim() == 2, "Vulkan mask ops expect a 2D mask");
            const size_t count = mask.shape()[0] * mask.shape()[1];
            LFS_ASSERT_MSG(count <= UINT32_MAX, "Vulkan mask op exceeds 32-bit indexing");
            return static_cast<uint32_t>(count);
        }
        void validate_mask(In mask) {
            LFS_ASSERT_MSG(byte_mask(mask) || mask.dtype() == core::DataType::Float32,
                           "Vulkan mask dtype must be Float32, UInt8 or Bool");
        }
        uint32_t reduce_groups(uint32_t count) { return std::min((count + 255u) / 256u, 1024u); }

        Push make_push(In alpha, In mask, In roi, Out output, Out partials,
                       uint32_t count, uint32_t mode) {
            validate_mask(mask);
            const StorageRef m = ref(mask), out = ref(output);
            const bool use_roi = has_roi(roi);
            const StorageRef roi_ref = use_roi ? ref(roi) : StorageRef{};
            const StorageRef alpha_ref = alpha.is_valid() ? ref(alpha) : StorageRef{};
            const StorageRef partial_ref = partials.is_valid() ? ref(partials) : StorageRef{};
            return {alpha.is_valid() ? vk::address(alpha_ref) : 0, vk::address(m),
                    use_roi ? vk::address(roi_ref) : 0, vk::address(out),
                    partials.is_valid() ? vk::address(partial_ref) : 0,
                    count, byte_mask(mask) ? 1u : 0u, use_roi ? 1u : 0u, mode,
                    0.0f, 0.0f, kernels::kMaskKeepMin, kernels::kMaskSegmentMin};
        }

        std::vector<StorageRef> mask_reads(In mask, In roi) {
            std::vector<StorageRef> reads{ref(mask)};
            if (has_roi(roi))
                reads.push_back(ref(roi));
            return reads;
        }

        void photometric_weight(In mask, In roi, Out weight, MaskPhotoMode mode) {
            const uint32_t count = pixel_count(mask);
            if (count == 0)
                return;
            Tensor no_alpha, no_partials;
            const Push push = make_push(no_alpha, mask, roi, weight, no_partials, count, static_cast<uint32_t>(mode));
            const auto context = acquire_vulkan_context();
            launch(context, push, 0, mask_reads(mask, roi), {ref(weight)}, vk::dispatch_groups(*context, count));
        }

        void penalty(In alpha, In mask, In roi, Out grad_alpha, Out reduction_temp, Out loss,
                     const MaskOpacityMode mode, const float power, const float scale, const bool consistency) {
            const uint32_t count = pixel_count(mask);
            if (count == 0)
                return;
            const uint32_t groups = reduce_groups(count);
            const StorageRef grad = ref(grad_alpha), scratch = ref(reduction_temp), loss_ref = ref(loss);
            const auto context = acquire_vulkan_context();
            if (!consistency && scale == 0.0f) {
                Tensor no_alpha, no_roi, no_partials;
                Push push = make_push(no_alpha, mask, no_roi, grad_alpha, no_partials, count, 0);
                const StorageRef loss_address = loss_ref;
                push.alpha = vk::address(loss_address);
                launch(context, push, 4, {}, {grad, loss_ref}, vk::dispatch_groups(*context, count));
                return;
            }

            Push push = make_push(alpha, mask, roi, grad_alpha, reduction_temp, count,
                                  consistency ? 0u : static_cast<uint32_t>(mode));
            push.power = power;
            push.grad_scale = (consistency ? scale : scale) / static_cast<float>(count);
            std::vector<StorageRef> reads{ref(alpha), ref(mask)};
            if (has_roi(roi))
                reads.push_back(ref(roi));
            std::vector<StorageRef> writes{grad, scratch};
            launch(context, push, consistency ? 2u : 1u, reads, writes, groups);

            Push reduce{};
            reduce.output = vk::address(loss_ref);
            reduce.partials = vk::address(scratch);
            reduce.count = count;
            reduce.mode = groups;
            reduce.power = scale;
            launch(context, reduce, 3, {scratch}, {loss_ref}, 1);
        }

        void opacity_penalty(In alpha, In mask, In roi, Out grad_alpha, Out reduction_temp,
                             Out loss, MaskOpacityMode mode, float power, float scale) {
            penalty(alpha, mask, roi, grad_alpha, reduction_temp, loss, mode, power, scale, false);
        }
        void alpha_consistency(In alpha, In mask, In roi, Out grad_alpha, Out reduction_temp,
                               Out loss, float weight) {
            penalty(alpha, mask, roi, grad_alpha, reduction_temp, loss, MaskOpacityMode::BinaryGt0,
                    0.0f, weight, true);
        }

        const MaskOps kVulkanMaskOps{
            .photometric_weight = photometric_weight,
            .opacity_penalty = opacity_penalty,
            .alpha_consistency = alpha_consistency,
        };
    } // namespace

    const gpu_ops::MaskOps& vulkan_masks_ops() { return kVulkanMaskOps; }
} // namespace lfs::training
