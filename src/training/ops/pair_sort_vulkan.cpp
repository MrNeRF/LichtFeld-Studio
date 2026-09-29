/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/pair_sort_vulkan.hpp"

#include "core/gpu_elapsed.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_memory.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "training_shader_table.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <mutex>
#include <ranges>
#include <string_view>
#include <vector>

namespace lfs::training {
    namespace {
        using namespace lfs::core::internal;

        constexpr uint32_t kPartitionKeys = 2048;
        constexpr uint32_t kFallbackPartitionKeys = 2048;
        constexpr uint32_t kDigitWidth = 8;
        constexpr uint32_t kHistogramStage = 0;
        constexpr uint32_t kPartitionScanStage = 1;
        constexpr uint32_t kDigitBaseScanStage = 2;
        constexpr uint32_t kScatterStage = 3;
        constexpr uint32_t kStageSpecializationId = 0;

        struct PairSortPush {
            uint64_t source_keys;
            uint64_t destination_keys;
            uint64_t source_values;
            uint64_t destination_values;
            uint64_t partition_counts;
            uint64_t digit_offsets;
            uint32_t element_count;
            uint32_t partition_count;
            uint32_t bit_offset;
            uint32_t key_width;
            uint32_t significant_bits;
            uint32_t pass_count;
            uint32_t pass_index;
            uint64_t control;
        };
        static_assert(sizeof(PairSortPush) == 88);
        static_assert(offsetof(PairSortPush, source_keys) == 0);
        static_assert(offsetof(PairSortPush, destination_keys) == 8);
        static_assert(offsetof(PairSortPush, source_values) == 16);
        static_assert(offsetof(PairSortPush, destination_values) == 24);
        static_assert(offsetof(PairSortPush, partition_counts) == 32);
        static_assert(offsetof(PairSortPush, digit_offsets) == 40);
        static_assert(offsetof(PairSortPush, element_count) == 48);
        static_assert(offsetof(PairSortPush, pass_count) == 68);
        static_assert(offsetof(PairSortPush, pass_index) == 72);
        static_assert(offsetof(PairSortPush, control) == 80);

        struct Pipelines {
            std::shared_ptr<VulkanContext> context;
            VkPipelineLayout layout = VK_NULL_HANDLE;
            std::array<VkPipeline, 4> stages{};

            ~Pipelines() {
                if (!context || context->device() == VK_NULL_HANDLE)
                    return;
                for (const VkPipeline pipeline : stages) {
                    if (pipeline != VK_NULL_HANDLE)
                        vkDestroyPipeline(context->device(), pipeline, nullptr);
                }
                if (layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(context->device(), layout, nullptr);
            }
        };

        bool supports_subgroup_match(const VulkanContext& context) {
            if (context.caps().subgroup_size != 32)
                return false;
            uint32_t count = 0;
            if (vkEnumerateDeviceExtensionProperties(context.physical_device(), nullptr,
                                                     &count, nullptr) != VK_SUCCESS)
                return false;
            std::vector<VkExtensionProperties> extensions(count);
            if (count != 0 && vkEnumerateDeviceExtensionProperties(
                                  context.physical_device(), nullptr, &count,
                                  extensions.data()) != VK_SUCCESS)
                return false;
            constexpr std::string_view required = "VK_NV_shader_subgroup_partitioned";
            return std::ranges::any_of(extensions, [&](const VkExtensionProperties& extension) {
                return required == extension.extensionName;
            });
        }

        std::shared_ptr<Pipelines> pipelines_for(const std::shared_ptr<VulkanContext>& context,
                                                 const bool subgroup_match) {
            static std::mutex mutex;
            static std::map<std::pair<uint64_t, bool>, std::shared_ptr<Pipelines>> cached;
            const auto key = std::pair{context->context_id(), subgroup_match};
            std::lock_guard lock(mutex);
            if (const auto found = cached.find(key); found != cached.end())
                return found->second;
            const auto modules = lfs::training::vulkan::embedded_training_shaders();
            const std::string_view module_name = subgroup_match ? "pair_sort_match" : "pair_sort";
            const auto module = std::ranges::find(modules, module_name, &lfs::training::vulkan::EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan pair sort shader module is missing");

            auto pipelines = std::make_shared<Pipelines>();
            pipelines->context = context;
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = module->words.size_bytes();
            shader_info.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &shader_info, nullptr, &shader),
                     "vkCreateShaderModule(training.pair_sort)");

            VkPushConstantRange push_range{};
            push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            push_range.offset = 0;
            push_range.size = sizeof(PairSortPush);
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &push_range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &layout_info, nullptr, &pipelines->layout),
                     "vkCreatePipelineLayout(training.pair_sort)");

            const VkSpecializationMapEntry stage_entry{
                kStageSpecializationId, 0, sizeof(uint32_t)};
            for (uint32_t stage = 0; stage < pipelines->stages.size(); ++stage) {
                VkSpecializationInfo specialization{};
                specialization.mapEntryCount = 1;
                specialization.pMapEntries = &stage_entry;
                specialization.dataSize = sizeof(stage);
                specialization.pData = &stage;
                VkPipelineShaderStageCreateInfo shader_stage{
                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
                shader_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                shader_stage.module = shader;
                shader_stage.pName = "main";
                shader_stage.pSpecializationInfo = &specialization;
                VkComputePipelineCreateInfo pipeline_info{
                    VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
                pipeline_info.stage = shader_stage;
                pipeline_info.layout = pipelines->layout;
                vk_check(context.get(), vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &pipeline_info, nullptr, &pipelines->stages[stage]),
                         "vkCreateComputePipelines(training.pair_sort)");
            }
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cached.emplace(key, pipelines);
            return pipelines;
        }

        void dispatch(VulkanContext& context, const std::shared_ptr<Pipelines>& pipelines,
                      const uint32_t stage, const uint32_t groups,
                      const PairSortPush& push,
                      const std::span<const StorageRef> reads,
                      const std::span<const StorageRef> writes, StorageRef indirect = {}) {
            context.recorders().record(
                reads, writes,
                [&](const VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines->stages[stage]);
                    vkCmdPushConstants(command, pipelines->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(push), &push);
                    if (indirect.meta) {
                        const VkDeviceSize word = stage == kHistogramStage || stage == kScatterStage ? 8 : stage == kDigitBaseScanStage ? 28
                                                                                                                                        : 16;
                        vkCmdDispatchIndirect(command, VulkanMemory::buffer_for(indirect), VulkanMemory::offset_for(indirect) + word * 4);
                    } else
                        vkCmdDispatch(command, groups, 1, 1);
                },
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | (indirect.meta ? VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT : 0), VK_WHOLE_SIZE,
                pipelines);
        }
    } // namespace

    bool vulkan_pair_sort(const PairSortBuffers buffers, const uint32_t element_count,
                          const uint32_t begin_bit, const uint32_t end_bit,
                          const bool wide_keys,
                          std::vector<PairSortPassTimings>* pass_timings, const core::Tensor* indirect) {
        LFS_ASSERT_MSG(buffers.keys_a && buffers.keys_b && buffers.values_a && buffers.values_b,
                       "Vulkan pair sort needs four ping-pong buffers");
        if (element_count == 0)
            return true;
        const auto context = acquire_vulkan_context();
        const auto storage = [](const core::Tensor* tensor) {
            const StorageRef ref = storage_ref(*tensor);
            LFS_ASSERT_MSG(ref.backend == core::GpuBackend::Vulkan,
                           "Vulkan pair sort received storage from another backend");
            return ref;
        };
        const StorageRef keys_a = storage(buffers.keys_a);
        const StorageRef keys_b = storage(buffers.keys_b);
        const StorageRef values_a = storage(buffers.values_a);
        const StorageRef values_b = storage(buffers.values_b);
        const core::DataType key_type = wide_keys ? core::DataType::Int64
                                                  : core::DataType::UInt32;
        for (const core::Tensor* keys : {buffers.keys_a, buffers.keys_b}) {
            LFS_ASSERT_MSG(keys->dtype() == key_type && keys->numel() >= element_count,
                           "Vulkan pair sort key buffer has the wrong type or capacity");
        }
        for (const core::Tensor* values : {buffers.values_a, buffers.values_b}) {
            LFS_ASSERT_MSG(values->dtype() == core::DataType::UInt32 &&
                               values->numel() >= element_count,
                           "Vulkan pair sort payload buffer has the wrong type or capacity");
        }
        LFS_ASSERT_MSG(begin_bit < end_bit && end_bit <= (wide_keys ? 64u : 32u),
                       "Vulkan pair sort bit range is invalid");
        const bool subgroup_match = supports_subgroup_match(*context);
        const uint32_t partition_keys = subgroup_match ? kPartitionKeys : kFallbackPartitionKeys;
        const uint32_t partitions = element_count / partition_keys +
                                    (element_count % partition_keys != 0);
        LFS_ASSERT_MSG(partitions <= context->caps().max_workgroup_count[0],
                       "Vulkan pair sort partition count exceeds the device dispatch limit");
        const uint32_t pass_count = (end_bit - begin_bit + kDigitWidth - 1) / kDigitWidth;
        const StorageRef counts = context->memory().allocate(
            static_cast<size_t>(partitions) * 256 * pass_count * sizeof(uint32_t), 16, {});
        const StorageRef offsets = context->memory().allocate(
            static_cast<size_t>(256) * pass_count * sizeof(uint32_t), 16, {});
        struct Release {
            VulkanContext& context;
            StorageRef counts;
            StorageRef offsets;
            ~Release() {
                context.memory().deallocate(counts);
                context.memory().deallocate(offsets);
            }
        } release{*context, counts, offsets};

        const auto pipelines = pipelines_for(context, subgroup_match);
        if (indirect)
            LFS_ASSERT_MSG(indirect->is_contiguous() && indirect->dtype() == core::DataType::UInt32 && indirect->numel() >= 32,
                           "Vulkan pair sort indirect control has the wrong type or size");
        const StorageRef control = indirect ? storage(indirect) : StorageRef{};
        bool in_a = true;
        const size_t dispatch_count = static_cast<size_t>(pass_count) * 4;
        core::GpuElapsed gpu_elapsed(core::GpuBackend::Vulkan,
                                     pass_timings == nullptr ? 0 : dispatch_count * 2);
        const auto target = core::TensorExecutionTarget::current();
        size_t timestamp = 0;
        std::vector<std::pair<size_t, float*>> timing_records;
        timing_records.reserve(dispatch_count);
        if (pass_timings != nullptr)
            pass_timings->assign(pass_count, {});
        const auto timed_dispatch = [&](const uint32_t stage, const uint32_t groups,
                                        const PairSortPush& push,
                                        const std::span<const StorageRef> reads,
                                        const std::span<const StorageRef> writes,
                                        float* elapsed_ms) {
            if (pass_timings != nullptr && gpu_elapsed.ready()) {
                (void)gpu_elapsed.mark(timestamp, target);
                timing_records.emplace_back(timestamp, elapsed_ms);
            }
            if (control.meta) {
                std::array<StorageRef, 5> dependencies{};
                LFS_ASSERT_MSG(reads.size() < dependencies.size(), "Too many pair sort read dependencies");
                std::copy(reads.begin(), reads.end(), dependencies.begin());
                dependencies[reads.size()] = control;
                dispatch(*context, pipelines, stage, groups, push,
                         std::span(dependencies.data(), reads.size() + 1), writes, control);
            } else {
                dispatch(*context, pipelines, stage, groups, push, reads, writes);
            }
            if (pass_timings != nullptr && gpu_elapsed.ready())
                (void)gpu_elapsed.mark(timestamp + 1, target);
            timestamp += 2;
        };
        for (uint32_t pass = 0; pass < pass_count; ++pass) {
            const StorageRef source_keys = in_a ? keys_a : keys_b;
            const StorageRef destination_keys = in_a ? keys_b : keys_a;
            const StorageRef source_values = in_a ? values_a : values_b;
            const StorageRef destination_values = in_a ? values_b : values_a;
            const PairSortPush push{
                .source_keys = vk::address(source_keys),
                .destination_keys = vk::address(destination_keys),
                .source_values = vk::address(source_values),
                .destination_values = vk::address(destination_values),
                .partition_counts = vk::address(counts),
                .digit_offsets = vk::address(offsets),
                .element_count = element_count,
                .partition_count = partitions,
                .bit_offset = begin_bit + pass * kDigitWidth,
                .key_width = wide_keys ? 64u : 32u,
                .significant_bits = end_bit,
                .pass_count = pass_count,
                .pass_index = pass,
                .control = control.meta ? vk::address(control) : 0,
            };
            const std::array histogram_reads{source_keys};
            const std::array histogram_writes{counts};
            timed_dispatch(kHistogramStage, partitions, push,
                           histogram_reads, histogram_writes,
                           pass_timings == nullptr ? nullptr
                                                   : &(*pass_timings)[pass].histogram_ms);
            const std::array partition_scan_reads{counts};
            const std::array partition_scan_writes{counts, offsets};
            timed_dispatch(kPartitionScanStage, 256, push, partition_scan_reads, partition_scan_writes,
                           pass_timings == nullptr ? nullptr : &(*pass_timings)[pass].partition_scan_ms);
            const std::array base_scan_reads{offsets};
            const std::array base_scan_writes{offsets};
            timed_dispatch(kDigitBaseScanStage, 1, push, base_scan_reads, base_scan_writes,
                           pass_timings == nullptr ? nullptr : &(*pass_timings)[pass].digit_base_scan_ms);
            const std::array scatter_reads{source_keys, source_values, counts, offsets};
            const std::array scatter_writes{destination_keys, destination_values};
            timed_dispatch(kScatterStage, partitions, push, scatter_reads, scatter_writes,
                           pass_timings == nullptr ? nullptr
                                                   : &(*pass_timings)[pass].scatter_ms);
            in_a = !in_a;
        }
        if (pass_timings != nullptr && gpu_elapsed.ready()) {
            (void)gpu_elapsed.wait_queue(target);
            for (const auto& [first, output] : timing_records) {
                if (const auto elapsed = gpu_elapsed.milliseconds(first, first + 1))
                    *output = *elapsed;
            }
        }
        return in_a;
    }
} // namespace lfs::training
