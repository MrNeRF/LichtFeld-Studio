/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/adam_vulkan.hpp"

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
#include <stdexcept>
#include <string_view>
#include <vector>

namespace lfs::training {
    namespace {
        using namespace lfs::core::internal;
        using namespace lfs::gpu_ops;

        struct Params {
            uint64_t parameter, packed, bounds, gradient, value_bounds;
            uint64_t frozen, crop, raw_scales, far_mask, screen_share;
            uint64_t indices, scratch;
            uint32_t primitives, attributes, bits, operation;
            uint32_t slots, active_bases, value_bits, value_cells;
            uint32_t index_count, layout, entry_count, stage;
            uint32_t frozen_count, crop_count, scale_count, far_count;
            uint32_t screen_count;
            float lr, bc1, bc2, beta1, beta2, eps;
            float frozen_scale, crop_scale, median_extent, r_min;
            float r_max, screen_limit, screen_penalty, step_size;
        };
        static_assert(sizeof(Params) == 224);
        static_assert(offsetof(Params, primitives) == 96);
        static_assert(offsetof(Params, lr) == 164);

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

        std::shared_ptr<Pipeline> pipeline_for(const std::shared_ptr<VulkanContext>& context) {
            static std::mutex mutex;
            static std::map<uint64_t, std::shared_ptr<Pipeline>> cache;
            std::lock_guard lock(mutex);
            if (const auto found = cache.find(context->context_id()); found != cache.end())
                return found->second;
            const auto modules = vulkan::embedded_training_shaders();
            const auto module = std::ranges::find(modules, std::string_view("adam"),
                                                  &vulkan::EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan Adam shader module is missing");
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = module->words.size_bytes();
            shader_info.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &shader_info, nullptr, &shader),
                     "vkCreateShaderModule(training.adam)");
            auto pipeline = std::make_shared<Pipeline>();
            pipeline->context = context;
            VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &layout_info, nullptr, &pipeline->layout),
                     "vkCreatePipelineLayout(training.adam)");
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            info.stage = stage;
            info.layout = pipeline->layout;
            vk_check(context.get(), vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &info, nullptr, &pipeline->handle),
                     "vkCreateComputePipelines(training.adam)");
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cache.emplace(context->context_id(), pipeline);
            return pipeline;
        }

        struct ParamsBlock {
            VulkanContext* context;
            StorageRef storage;
            ~ParamsBlock() { context->memory().deallocate(storage); }
        };

        std::shared_ptr<ParamsBlock> upload_params(VulkanContext& context, Params params) {
            auto block = std::make_shared<ParamsBlock>();
            block->context = &context;
            block->storage = context.memory().allocate(sizeof(Params), 16, {});
            context.memory().copy_host_to_device(CopyRequest{
                .src = raw_storage_ref(&params),
                .dst = block->storage,
                .bytes = sizeof(params),
                .synchronous = false});
            return block;
        }

        StorageRef ref(const Tensor& tensor) {
            const auto storage = storage_ref(tensor);
            LFS_ASSERT_MSG(storage.backend == core::GpuBackend::Vulkan,
                           "Vulkan Adam received storage from another backend");
            return storage;
        }

        uint64_t address(const Tensor& tensor) {
            return tensor.is_valid() ? vk::address(ref(tensor)) : 0;
        }

        void launch(Params params, const std::span<const StorageRef> reads,
                    const std::span<const StorageRef> writes, const uint32_t groups) {
            if (groups == 0)
                return;
            auto context = acquire_vulkan_context();
            auto parameter_block = upload_params(*context, params);
            auto pipeline = pipeline_for(context);
            const Push push{vk::address(parameter_block->storage)};
            std::vector<StorageRef> all_reads(reads.begin(), reads.end());
            all_reads.push_back(parameter_block->storage);
            context->recorders().record(
                all_reads, writes,
                [pipeline, push, groups](VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
                    vkCmdPushConstants(command, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                       0, sizeof(push), &push);
                    vkCmdDispatch(command, groups, 1, 1);
                },
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_WHOLE_SIZE, parameter_block);
        }

        uint32_t groups(const size_t work) {
            const auto context = acquire_vulkan_context();
            return vk::dispatch_groups(*context, work);
        }

        template <typename T>
        uint64_t optional(const Tensor& tensor) {
            (void)sizeof(T);
            return tensor.is_valid() && tensor.numel() != 0 ? address(tensor) : 0;
        }

        uint32_t optional_count(const Tensor& tensor) {
            return tensor.is_valid() ? vk::checked_u32(tensor.numel(), "Adam mask count exceeds 32-bit indexing") : 0;
        }

        void validate_far_mask(const bool* pointer) {
            if (!pointer)
                throw std::invalid_argument("mean-step far mask must not be null");
            const auto context = acquire_vulkan_context();
            LFS_ASSERT_MSG(context->memory().owns_address(pointer),
                           "Vulkan Adam far-mask pointer is not a live Vulkan allocation");
        }

        void step_rows(std::span<const JointStep> steps, const AdamMasks& masks,
                       const AdamHyper& hyper, const AdamModifiers& modifiers, const bool fused) {
            std::vector<const JointStep*> present;
            for (const auto& step : steps) {
                if (step.parameter.is_valid())
                    present.push_back(&step);
            }
            if (present.empty())
                return;
            if (present.size() > 6)
                throw std::runtime_error("adam_step_batch: too many steps");
            for (const JointStep* step : present) {
                if (step->bits != 16 && !(fused && step->bits == 8))
                    throw std::runtime_error("adam_step_batch: moments must be 16-bit");
                LFS_ASSERT_MSG(step->primitives > 0 && step->attributes > 0 && step->attributes <= 16,
                               "Vulkan Adam rows require 1..16 attributes and positive rows");
                Params p{};
                p.parameter = address(step->parameter);
                p.packed = address(step->packed);
                p.bounds = address(step->bounds);
                p.gradient = address(step->gradient);
                p.frozen = optional<bool>(masks.frozen);
                p.crop = optional<bool>(masks.crop_damping);
                p.raw_scales = optional<float>(masks.raw_scales);
                p.far_mask = optional<bool>(masks.far_mask);
                p.screen_share = optional<float>(masks.screen_share);
                p.frozen_count = optional_count(masks.frozen);
                p.crop_count = optional_count(masks.crop_damping);
                p.scale_count = optional_count(masks.raw_scales);
                p.far_count = optional_count(masks.far_mask);
                p.screen_count = optional_count(masks.screen_share);
                p.primitives = vk::checked_u32(step->primitives, "Adam row count exceeds 32-bit indexing");
                p.attributes = static_cast<uint32_t>(step->attributes);
                p.bits = static_cast<uint32_t>(step->bits);
                p.operation = 0;
                p.lr = step->lr;
                p.bc1 = step->bc1_rcp;
                p.bc2 = step->bc2_sqrt_rcp;
                p.beta1 = hyper.beta1;
                p.beta2 = hyper.beta2;
                p.eps = hyper.eps;
                p.frozen_scale = modifiers.frozen_lr_scale;
                p.crop_scale = modifiers.cropbox_lr_scale;
                p.median_extent = modifiers.median_extent;
                p.r_min = modifiers.r_min;
                p.r_max = modifiers.r_max;
                p.screen_limit = modifiers.screen_share_limit;
                p.screen_penalty = modifiers.screen_share_penalty;
                p.entry_count = (step->apply_mean_step ? 1u : 0u) | (step->apply_screen_share ? 2u : 0u);
                std::vector<StorageRef> reads{ref(step->parameter), ref(step->packed), ref(step->bounds),
                                              ref(step->gradient)};
                for (const Tensor* tensor : {&masks.frozen, &masks.crop_damping, &masks.raw_scales,
                                             &masks.far_mask, &masks.screen_share}) {
                    if (tensor->is_valid() && tensor->numel() != 0)
                        reads.push_back(ref(*tensor));
                }
                const std::array writes{ref(step->parameter), ref(step->packed), ref(step->bounds)};
                launch(p, reads, writes, (p.primitives + 255u) / 256u);
            }
        }

        void step_batch(std::span<const JointStep> steps, const AdamMasks& masks,
                        const AdamHyper& hyper, const AdamModifiers& modifiers) {
            step_rows(steps, masks, hyper, modifiers, false);
        }

        void step_sh(Out parameter, Out packed, Out bounds, Out value_bounds, In gradient,
                     const AdamMasks& masks, const AdamHyper& hyper,
                     const AdamModifiers& modifiers, const ShStepParams& step) {
            if (step.primitives <= 0 || step.layout_slots <= 0)
                return;
            LFS_ASSERT_MSG(step.active_bases == 4 || step.active_bases == 9 || step.active_bases == 16,
                           "Vulkan SH Adam active_bases must be 4, 9, or 16");
            LFS_ASSERT_MSG(step.value_bits == 0 || step.value_bits == 16,
                           "Vulkan SH Adam value_bits must be 0 or 16");
            Params p{};
            p.parameter = address(parameter);
            p.packed = address(packed);
            p.bounds = address(bounds);
            p.gradient = address(gradient);
            p.value_bounds = optional<float>(value_bounds);
            p.frozen = optional<bool>(masks.frozen);
            p.crop = optional<bool>(masks.crop_damping);
            p.primitives = static_cast<uint32_t>(step.primitives);
            p.slots = static_cast<uint32_t>(step.layout_slots);
            p.active_bases = static_cast<uint32_t>(step.active_bases);
            p.value_bits = static_cast<uint32_t>(step.value_bits);
            p.value_cells = static_cast<uint32_t>(step.value_cells);
            p.frozen_count = optional_count(masks.frozen);
            p.crop_count = optional_count(masks.crop_damping);
            p.bits = 8;
            p.operation = 1;
            p.lr = step.step_size;
            p.step_size = step.step_size;
            p.bc2 = step.bc2_sqrt_rcp;
            p.beta1 = hyper.beta1;
            p.beta2 = hyper.beta2;
            p.eps = hyper.eps;
            p.frozen_scale = modifiers.frozen_lr_scale;
            p.crop_scale = modifiers.cropbox_lr_scale;
            p.entry_count = 0;
            std::vector<StorageRef> reads{ref(parameter), ref(packed), ref(bounds), ref(gradient)};
            std::vector<StorageRef> writes{ref(parameter), ref(packed), ref(bounds)};
            if (value_bounds.is_valid()) {
                reads.push_back(ref(value_bounds));
                writes.push_back(ref(value_bounds));
            }
            for (const Tensor* tensor : {&masks.frozen, &masks.crop_damping}) {
                if (tensor->is_valid() && tensor->numel() != 0)
                    reads.push_back(ref(*tensor));
            }
            launch(p, reads, writes, (p.primitives + 255u) / 256u);
        }

        void encode_zero(Out packed, Out bounds, In indices, const JointCodecParams& codec) {
            if (indices.numel() == 0)
                return;
            LFS_ASSERT_MSG(codec.primitives > 0 && codec.attributes_or_slots > 0 &&
                               (codec.bits == 8 || codec.bits == 16),
                           "Vulkan Adam encode_zero parameters are invalid");
            Params p{};
            p.packed = address(packed);
            p.bounds = address(bounds);
            p.indices = address(indices);
            p.primitives = static_cast<uint32_t>(codec.primitives);
            p.attributes = codec.layout == JointLayout::Rows ? codec.attributes_or_slots : 0;
            p.slots = codec.layout == JointLayout::SwizzledSH ? codec.attributes_or_slots : 0;
            p.bits = static_cast<uint32_t>(codec.bits);
            p.layout = codec.layout == JointLayout::Rows ? 0u : 1u;
            p.operation = 2;
            p.index_count = vk::checked_u32(indices.numel(), "Adam index count exceeds 32-bit indexing");
            const uint32_t blocks = (p.primitives + 255u) / 256u;
            auto flags = Tensor::zeros({size_t(p.primitives) + blocks}, core::Device::GPU, core::DataType::Int32);
            p.scratch = address(flags);
            const std::array mark_reads{ref(indices)};
            const std::array mark_writes{ref(flags)};
            p.operation = 3;
            launch(p, mark_reads, mark_writes, groups(p.index_count));
            p.operation = 2;
            const std::array reads{ref(packed), ref(bounds), ref(flags)};
            const std::array writes{ref(packed), ref(bounds)};
            launch(p, reads, writes, blocks);
        }

        const AdamOps kVulkanAdamOps{
            .validate_far_mask = validate_far_mask,
            .step_batch = step_batch,
            .step_sh = step_sh,
            .encode_zero = encode_zero,
        };
    } // namespace

    void vulkan_fused_adam_rows(const gpu_ops::JointStep& step, const gpu_ops::AdamMasks& masks,
                                const gpu_ops::AdamHyper& hyper, const gpu_ops::AdamModifiers& modifiers) {
        step_rows(std::span(&step, 1), masks, hyper, modifiers, true);
    }

    const AdamOps& vulkan_adam_ops() { return kVulkanAdamOps; }
} // namespace lfs::training
