/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/morton_vulkan.hpp"
#include "lfs/training/idle_arena_scratch.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/ops/pair_sort_vulkan.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/sh_value_storage.hpp"

#include "core/assert.hpp"
#include "core/sh_layout.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_completion.hpp"

#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_memory.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "core/vulkan_helpers.hpp"
#include "training_shader_table.hpp"

#include <array>
#include <map>
#include <mutex>
#include <optional>
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

        struct MortonPush {
            uint64_t means, minimum, maximum, codes, indices;
            uint32_t count;
        };
        static_assert(sizeof(MortonPush) == 48);

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

        std::shared_ptr<Pipeline> training_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                    const char* const name, const uint32_t push_size,
                                                    const std::optional<uint32_t> phase) {
            using namespace lfs::training::vulkan;
            static std::mutex mutex;
            static std::map<std::tuple<uint64_t, std::string, uint32_t>, std::shared_ptr<Pipeline>> cache;
            const uint32_t key_phase = phase.value_or(UINT32_MAX);
            const auto key = std::tuple{context->context_id(), std::string(name), key_phase};
            std::lock_guard lock(mutex);
            if (const auto found = cache.find(key); found != cache.end())
                return found->second;
            const auto modules = embedded_training_shaders();
            const auto module = std::ranges::find(modules, std::string_view(name), &EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan Morton shader module is missing");
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = module->words.size_bytes();
            shader_info.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &shader_info, nullptr, &shader),
                     "vkCreateShaderModule(training.morton)");
            auto pipeline = std::make_shared<Pipeline>();
            pipeline->context = context;
            VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size};
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &layout_info, nullptr, &pipeline->layout),
                     "vkCreatePipelineLayout(training.morton)");
            VkSpecializationMapEntry entry{0, 0, sizeof(uint32_t)};
            VkSpecializationInfo specialization{};
            specialization.mapEntryCount = phase ? 1u : 0u;
            specialization.pMapEntries = phase ? &entry : nullptr;
            specialization.dataSize = phase ? sizeof(uint32_t) : 0u;
            specialization.pData = phase ? &*phase : nullptr;
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            stage.pSpecializationInfo = phase ? &specialization : nullptr;
            VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline_info.stage = stage;
            pipeline_info.layout = pipeline->layout;
            vk_check(context.get(), vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &pipeline_info, nullptr, &pipeline->handle),
                     "vkCreateComputePipelines(training.morton)");
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cache.emplace(key, pipeline);
            return pipeline;
        }

        template <typename Push>
        void dispatch(const std::shared_ptr<VulkanContext>& context, const std::shared_ptr<Pipeline>& pipeline,
                      const Push& push, const std::span<const StorageRef> reads,
                      const std::span<const StorageRef> writes, const uint32_t groups,
                      std::shared_ptr<void> lifetime = {}) {
            if (groups == 0)
                return;
            context->recorders().record(reads, writes, [&](const VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
                vkCmdPushConstants(command, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                vkCmdDispatch(command, groups, 1, 1); }, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_WHOLE_SIZE, lifetime ? lifetime : pipeline);
        }

        struct JointPush {
            uint64_t source, source_bounds, destination, destination_bounds, permutation;
            uint32_t primitives, width, swizzled, bits, slot_begin, slot_end;
        };
        static_assert(sizeof(JointPush) == 64);

        void launch_joint(const Tensor& source, const Tensor& bounds, Tensor& destination,
                          Tensor& destination_bounds, const Tensor& indices,
                          const JointCodecParams& codec, const uint32_t phase,
                          const uint32_t slot_begin = 0, const uint32_t slot_end = 0) {
            if (codec.bits != 8 && codec.bits != 16)
                throw std::invalid_argument("Morton joint permutation needs 8 or 16 bit moments");
            if (codec.primitives <= 0 || codec.attributes_or_slots <= 0)
                return;
            const bool swizzled = codec.layout == JointLayout::SwizzledSH;
            const JointPush push{vk::address(storage_ref(source)), vk::address(storage_ref(bounds)),
                                 vk::address(storage_ref(destination)), vk::address(storage_ref(destination_bounds)),
                                 vk::address(storage_ref(indices)), static_cast<uint32_t>(codec.primitives),
                                 static_cast<uint32_t>(codec.attributes_or_slots), swizzled ? 1u : 0u,
                                 static_cast<uint32_t>(codec.bits), slot_begin,
                                 slot_end == 0 ? static_cast<uint32_t>(codec.attributes_or_slots) : slot_end};
            const auto context = acquire_vulkan_context();
            const auto pipeline = training_pipeline(context, "joint_morton", sizeof(push), phase);
            const std::array reads{storage_ref(source), storage_ref(bounds), storage_ref(indices)};
            const std::array writes{storage_ref(destination), storage_ref(destination_bounds)};
            const size_t work = phase == 0 ? (static_cast<size_t>(codec.primitives) + 255) / 256 * 256
                                           : static_cast<size_t>(codec.primitives);
            const uint32_t groups = phase == 0 ? static_cast<uint32_t>((codec.primitives + 255) / 256)
                                               : vk::dispatch_groups(*context, work);
            dispatch(context, pipeline, push, reads, writes, groups);
        }

        Tensor permutation(const Tensor& means) {
            if (!means.is_valid() || means.ndim() != 2 || means.size(1) != 3 ||
                means.dtype() != DataType::Float32 || means.device() != Device::GPU) {
                throw std::invalid_argument("Morton permutation needs GPU float32 [N,3] means");
            }
            const size_t count = means.size(0);
            if (count == 0)
                return {};
            if (count > UINT32_MAX)
                throw std::invalid_argument("Morton permutation exceeds uint32 indexing");
            const Tensor input = means.contiguous();
            const Tensor minimum = input.min(0).contiguous();
            const Tensor maximum = input.max(0).contiguous();
            Tensor codes = Tensor::empty({count}, Device::GPU, DataType::UInt32);
            Tensor alternate_codes = Tensor::empty({count}, Device::GPU, DataType::UInt32);
            Tensor indices = Tensor::empty({count}, Device::GPU, DataType::UInt32);
            Tensor alternate_indices = Tensor::empty({count}, Device::GPU, DataType::UInt32);
            const MortonPush push{vk::address(storage_ref(input)), vk::address(storage_ref(minimum)),
                                  vk::address(storage_ref(maximum)), vk::address(storage_ref(codes)),
                                  vk::address(storage_ref(indices)), static_cast<uint32_t>(count)};
            const auto context = acquire_vulkan_context();
            const auto pipeline = training_pipeline(context, "morton", sizeof(push), 0);
            const std::array reads{storage_ref(input), storage_ref(minimum), storage_ref(maximum)};
            const std::array writes{storage_ref(codes), storage_ref(indices)};
            dispatch(context, pipeline, push, reads, writes, vk::dispatch_groups(*context, count));
            const bool sorted_in_a = vulkan_pair_sort({&codes, &alternate_codes, &indices, &alternate_indices},
                                                      static_cast<uint32_t>(count), 0, 30, false);
            Tensor result = Tensor::empty({count}, Device::GPU, DataType::Int64);
            MortonPush widen{vk::address(storage_ref(sorted_in_a ? indices : alternate_indices)), 0, 0, 0,
                             vk::address(storage_ref(result)), static_cast<uint32_t>(count)};
            const auto widen_pipeline = training_pipeline(context, "morton", sizeof(widen), 1);
            const std::array widen_reads{storage_ref(sorted_in_a ? indices : alternate_indices)};
            const std::array widen_writes{storage_ref(result)};
            dispatch(context, widen_pipeline, widen, widen_reads, widen_writes, vk::dispatch_groups(*context, count));
            return result;
        }

        void permute_joint(const Tensor& source, const Tensor& bounds, const Tensor& indices,
                           Tensor& destination, Tensor& destination_bounds, const JointCodecParams& codec) {
            launch_joint(source, bounds, destination, destination_bounds, indices, codec, 0);
            launch_joint(source, bounds, destination, destination_bounds, indices, codec, 1);
        }

        void permute_joint_grouped(Tensor& packed, const Tensor& bounds, const Tensor& indices,
                                   Tensor& destination_bounds, Tensor& scratch,
                                   const JointCodecParams& codec) {
            if (!scratch.is_valid() || scratch.bytes() == 0)
                throw std::invalid_argument("Morton grouped joint permutation needs scratch storage");
            if (codec.layout != JointLayout::SwizzledSH)
                throw std::invalid_argument("Morton grouped joint permutation requires swizzled SH layout");
            constexpr size_t reorder = core::kShReorderSize;
            const size_t tiles = (static_cast<size_t>(codec.primitives) + reorder - 1) / reorder;
            const size_t slot_bytes = reorder * 4 * static_cast<size_t>(joint_adam::bytes_per_cell(codec.bits));
            if (scratch.bytes() < tiles * slot_bytes)
                throw std::invalid_argument("Morton grouped scratch is smaller than one encoded slot");
            const auto context = acquire_vulkan_context();
            launch_joint(packed, bounds, scratch, destination_bounds, indices, codec, 0);
            const size_t slots = static_cast<size_t>(codec.attributes_or_slots);
            const size_t slots_per_group = std::max<size_t>(1, std::min(slots, scratch.bytes() / (tiles * slot_bytes)));
            for (size_t first = 0; first < slots; first += slots_per_group) {
                const size_t last = std::min(first + slots_per_group, slots);
                scratch.zero_();
                launch_joint(packed, bounds, scratch, destination_bounds, indices, codec, 1,
                             static_cast<uint32_t>(first), static_cast<uint32_t>(last));
                const StorageRef temp = storage_ref(scratch), live = storage_ref(packed);
                for (size_t tile = 0; tile < tiles; ++tile) {
                    for (size_t local = 0; local < last - first; ++local) {
                        backend_ops(GpuBackend::Vulkan).copy_device_to_device(CopyRequest{.src = offset_storage_ref(temp, tile * (last - first) * slot_bytes + local * slot_bytes), .dst = offset_storage_ref(live, tile * slots * slot_bytes + (first + local) * slot_bytes), .bytes = slot_bytes, .operation = "training.morton.copy_joint_group"});
                    }
                }
            }
        }

        void gather_gradient(const Tensor& source, const Tensor& indices, Tensor& destination,
                             const uint32_t rest, core::TensorExecutionTarget) {
            const ShRowsParams params{.source_rows = indices.numel(), .count = indices.numel(), .source_rest = rest, .destination_rest = rest};
            training_ops(GpuBackend::Vulkan).sh->gather_swizzled(source, indices, destination, params);
        }

        void permute_sh_fp32(core::SplatData& splat, const Tensor& indices,
                             core::TensorExecutionTarget, const IdleArenaScratch&) {
            const bool expanded = sh_value::ensure_shN_fp32_for_mutation(splat);
            auto& live = splat.shN();
            const size_t rows = static_cast<size_t>(splat.size());
            const auto rest = static_cast<uint32_t>(splat.max_sh_coeffs_rest());
            const size_t logical = core::sh_swizzled_float_count(rows, rest);
            if (!live.is_valid() || live.dtype() != DataType::Float32 || live.numel() < logical) {
                if (expanded)
                    (void)sh_value::commit_shN_after_mutation(splat);
                return;
            }
            Tensor gathered = Tensor::zeros({logical}, Device::GPU, DataType::Float32);
            const ShRowsParams params{.source_rows = rows, .count = rows, .source_rest = rest, .destination_rest = rest};
            training_ops(GpuBackend::Vulkan).sh->gather_swizzled(live, indices, gathered, params);
            if (live.numel() == logical)
                live.copy_from(gathered);
            else
                live.slice(0, 0, logical).copy_from(gathered);
            if (expanded)
                (void)sh_value::commit_shN_after_mutation(splat);
        }

        void permute_sh_q16(core::SplatData& splat, const Tensor& indices,
                            core::TensorExecutionTarget target, const IdleArenaScratch& scratch) {
            permute_sh_fp32(splat, indices, target, scratch);
        }

        void copy_back(Tensor& live, const void* source, const size_t bytes,
                       core::TensorExecutionTarget) {
            const auto context = acquire_vulkan_context();
            const StorageRef src = context->memory().borrow_address(source, bytes);
            const StorageRef dst = storage_ref(live);
            backend_ops(GpuBackend::Vulkan).copy_device_to_device(CopyRequest{.src = src, .dst = dst, .bytes = bytes, .synchronous = false});
        }

        const MortonOps kVulkanMortonOps{.permutation = permutation,
                                         .permute_joint = permute_joint,
                                         .permute_joint_grouped = permute_joint_grouped,
                                         .permute_sh_q16 = permute_sh_q16,
                                         .permute_sh_fp32 = permute_sh_fp32,
                                         .copy_back = copy_back,
                                         .gather_gradient = gather_gradient};
    } // namespace

    const MortonOps& vulkan_morton_ops() { return kVulkanMortonOps; }
} // namespace lfs::training
