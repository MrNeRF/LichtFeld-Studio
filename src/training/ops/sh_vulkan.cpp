/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/sh_vulkan.hpp"

#include "core/assert.hpp"
#include "core/sh_layout.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_memory.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "training_shader_table.hpp"

#include <array>
#include <cstddef>
#include <format>
#include <limits>
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

        struct Push {
            uint64_t source, destination, indices, source_bounds, destination_bounds;
            uint32_t source_rows, destination_rows, count;
            uint32_t source_offset, destination_offset, padding;
        };
        static_assert(sizeof(Push) == 64);
        static_assert(offsetof(Push, source_rows) == 40);
        static_assert(offsetof(Push, padding) == 60);

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

        using Specialization = std::array<uint32_t, 7>;

        struct ReencodeParams {
            uint64_t codes, bounds, canonical, destinations;
            uint64_t unique_blocks, offsets, run_count, canonical_order;
            uint32_t sorted_count, primitives, decode_rows, rest;
            uint32_t reserved[4]{};
        };
        static_assert(sizeof(ReencodeParams) == 96);

        struct ReencodeLifetime {
            VulkanContext* context = nullptr;
            StorageRef storage{};
            ~ReencodeLifetime() {
                if (context)
                    context->memory().deallocate(storage);
            }
        };

        std::shared_ptr<Pipeline> pipeline_for(const std::shared_ptr<VulkanContext>& context,
                                               const Specialization& constants) {
            static std::mutex mutex;
            static std::map<std::pair<uint64_t, Specialization>, std::shared_ptr<Pipeline>> cache;
            const auto key = std::pair{context->context_id(), constants};
            std::lock_guard lock(mutex);
            if (const auto found = cache.find(key); found != cache.end())
                return found->second;

            const auto modules = vulkan::embedded_training_shaders();
            const auto module = std::ranges::find(modules, std::string_view("sh_ops"),
                                                  &vulkan::EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan SH shader module is missing");
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = module->words.size_bytes();
            shader_info.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &shader_info, nullptr, &shader),
                     "vkCreateShaderModule(training.sh_ops)");

            auto pipeline = std::make_shared<Pipeline>();
            pipeline->context = context;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(context->physical_device(), &properties);
            LFS_ASSERT_MSG(sizeof(Push) <= properties.limits.maxPushConstantsSize,
                           "Vulkan SH parameters exceed device push-constant limit");
            VkPushConstantRange range{};
            range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            range.offset = 0;
            range.size = sizeof(Push);
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &layout_info, nullptr, &pipeline->layout),
                     "vkCreatePipelineLayout(training.sh_ops)");

            std::array<VkSpecializationMapEntry, Specialization{}.size()> entries{};
            for (uint32_t i = 0; i < entries.size(); ++i)
                entries[i] = {i, static_cast<uint32_t>(sizeof(uint32_t) * i), sizeof(uint32_t)};
            VkSpecializationInfo specialization{};
            specialization.mapEntryCount = static_cast<uint32_t>(entries.size());
            specialization.pMapEntries = entries.data();
            specialization.dataSize = sizeof(constants);
            specialization.pData = constants.data();
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            stage.pSpecializationInfo = &specialization;
            VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline_info.stage = stage;
            pipeline_info.layout = pipeline->layout;
            vk_check(context.get(), vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &pipeline_info, nullptr, &pipeline->handle),
                     "vkCreateComputePipelines(training.sh_ops)");
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cache.emplace(key, pipeline);
            return pipeline;
        }

        StorageRef ref(const Tensor& tensor) {
            const StorageRef storage = storage_ref(tensor);
            LFS_ASSERT_MSG(storage.backend == core::GpuBackend::Vulkan,
                           "Vulkan SH op received storage from another backend");
            return storage;
        }

        void launch(const Push& push, const Specialization& constants,
                    const std::span<const StorageRef> reads,
                    const std::span<const StorageRef> writes, const uint32_t groups) {
            if (groups == 0)
                return;
            const auto context = acquire_vulkan_context();
            const auto pipeline = pipeline_for(context, constants);
            context->recorders().record(reads, writes, [&](const VkCommandBuffer command) {
                                            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                              pipeline->handle);
                                            vkCmdPushConstants(command, pipeline->layout,
                                                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                                               sizeof(push), &push);
                                            vkCmdDispatch(command, groups, 1, 1); }, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_WHOLE_SIZE, pipeline);
        }

        uint32_t groups(const size_t work) {
            if (work == 0)
                return 0;
            const auto context = acquire_vulkan_context();
            return vk::dispatch_groups(*context, work);
        }

        struct RunScratch final : BackendState {};
        State create_run_scratch() { return std::make_unique<RunScratch>(); }

        void encode_q16(In swizzled, Out codes, Out bounds, const size_t primitives,
                        const uint32_t rest, const size_t code_offset, const size_t bounds_offset) {
            const auto src = ref(swizzled), dst = ref(codes), bnd = ref(bounds);
            LFS_ASSERT_MSG(code_offset <= UINT32_MAX / sizeof(uint16_t) &&
                               bounds_offset <= UINT32_MAX / sizeof(float),
                           "Vulkan SH encode offset exceeds 32-bit byte indexing");
            StorageRef code_storage = dst;
            StorageRef bounds_storage = bnd;
            code_storage.byte_offset += code_offset * sizeof(uint16_t);
            bounds_storage.byte_offset += bounds_offset * sizeof(float);
            Push push{vk::address(src), vk::address(code_storage), 0, 0,
                      vk::address(bounds_storage), static_cast<uint32_t>(primitives),
                      static_cast<uint32_t>(primitives), static_cast<uint32_t>(primitives), 0, 0, 0};
            const Specialization spec{1, 3, 0, 0, rest, rest, 0};
            const std::array reads{src, dst};
            const std::array writes{dst, bnd};
            launch(push, spec, reads, writes, static_cast<uint32_t>((primitives + 255) / 256));
        }

        void decode_q16(In codes, In bounds, Out swizzled, const size_t primitives,
                        const uint32_t rest) {
            const auto src = ref(codes), bnd = ref(bounds), dst = ref(swizzled);
            Push push{vk::address(src), vk::address(dst), 0, vk::address(bnd), 0,
                      static_cast<uint32_t>(primitives), static_cast<uint32_t>(primitives),
                      static_cast<uint32_t>(primitives), 0, 0, 0};
            const Specialization spec{3, 1, 0, 0, rest, rest, 1};
            const std::array reads{src, bnd};
            const std::array writes{dst};
            launch(push, spec, reads, writes, groups(swizzled.numel()));
        }

        void block_ids(In indices, Out output) {
            const auto src = ref(indices), dst = ref(output);
            Push push{vk::address(src), vk::address(dst), 0, 0, 0, 0, 0,
                      static_cast<uint32_t>(indices.numel()), 0, 0, 0};
            const std::array reads{src};
            const std::array writes{dst};
            launch(push, {0, 0, 0, 0, 0, 0, 2}, reads, writes, groups(indices.numel()));
        }

        void block_runs(BackendState&, In sorted, Out unique, Out offsets, Out count) {
            const auto src = ref(sorted), ids = ref(unique), off = ref(offsets), cnt = ref(count);
            Push push{vk::address(src), vk::address(ids), 0, vk::address(cnt), vk::address(off),
                      0, 0, static_cast<uint32_t>(sorted.numel()), 0, 0, 0};
            const std::array reads{src};
            const std::array writes{ids, off, cnt};
            launch(push, {0, 0, 0, 0, 0, 0, 3}, reads, writes, 1);
        }

        void decode_range(In values, In bounds, Out canonical, const ShRangeParams& params) {
            if (params.primitives == 0 || params.destination_rest == 0)
                throw std::invalid_argument(std::format("SH range decode needs primitives and rest, got {} and {}",
                                                        params.primitives, params.destination_rest));
            const uint64_t per_primitive = uint64_t{params.destination_rest} * core::kShChannels;
            if (params.primitives > std::numeric_limits<uint64_t>::max() / per_primitive ||
                params.canonical_float_offset > params.primitives * per_primitive ||
                params.float_count > params.primitives * per_primitive - params.canonical_float_offset)
                throw std::out_of_range(std::format("SH range [{}, +{}) exceeds the canonical {} x {} floats",
                                                    params.canonical_float_offset, params.float_count,
                                                    params.primitives, per_primitive));
            const auto src = ref(values), dst = ref(canonical);
            StorageRef bnd{};
            const bool needs_bounds = params.storage == ShStorage::Q16;
            if (needs_bounds)
                bnd = ref(bounds);
            LFS_ASSERT_MSG(params.canonical_float_offset <= UINT32_MAX && params.float_count <= UINT32_MAX,
                           "Vulkan SH range exceeds 32-bit scalar indexing");
            const uint32_t fmt = params.storage == ShStorage::Float32       ? 1u
                                 : params.storage == ShStorage::IeeeFloat16 ? 2u
                                                                            : 3u;
            Push push{vk::address(src), vk::address(dst), 0, needs_bounds ? vk::address(bnd) : 0, 0,
                      params.layout_rest, params.destination_rest, static_cast<uint32_t>(params.primitives),
                      static_cast<uint32_t>(params.canonical_float_offset),
                      static_cast<uint32_t>(params.float_count), 0};
            const Specialization spec{fmt, 1, 0, 0, params.layout_rest, params.destination_rest, 4};
            std::vector<StorageRef> reads{src};
            if (needs_bounds)
                reads.push_back(bnd);
            const std::array writes{dst};
            launch(push, spec, reads, writes, groups(params.float_count));
        }

        void zero_rows(Out values, In indices, const uint32_t rest) {
            const auto dst = ref(values), idx = ref(indices);
            Push push{0, vk::address(dst), vk::address(idx), 0, 0, 0, 0,
                      static_cast<uint32_t>(indices.numel()), 0, 0, 0};
            const std::array reads{idx};
            const std::array writes{dst};
            launch(push, {0, 1, 1, 0, rest, rest, 5}, reads, writes,
                   static_cast<uint32_t>((indices.numel() + 255) / 256));
        }

        void gather_swizzled(In source, In indices, Out destination, const ShRowsParams& params) {
            const auto src = ref(source), idx = ref(indices), dst = ref(destination);
            const uint32_t fmt = source.dtype() == core::DataType::UInt8 ? 4u : 1u;
            Push push{vk::address(src), vk::address(dst), vk::address(idx), 0, 0,
                      static_cast<uint32_t>(params.source_rows), static_cast<uint32_t>(destination.shape()[0]),
                      static_cast<uint32_t>(params.count), 0,
                      static_cast<uint32_t>(params.destination_offset), 0};
            const uint32_t index_type = indices.dtype() == core::DataType::Int64 ? 2u : 1u;
            const Specialization spec{fmt, fmt, index_type, 0,
                                      params.source_rest, params.destination_rest, 6};
            const std::array reads{src, idx};
            const std::array writes{dst};
            const size_t destination_width = (static_cast<size_t>(params.destination_rest) * 3 + 3) / 4 * 4;
            launch(push, spec, reads, writes, groups(params.count * destination_width));
        }

        void reencode_touched(Out codes, Out bounds, In canonical, In destinations,
                              In unique_blocks, In offsets, In run_count, In canonical_order,
                              const Q16TouchParams& params) {
            if (params.sorted_count == 0 || params.rest == 0 || params.primitives == 0)
                return;
            const auto context = acquire_vulkan_context();
            const auto code_ref = ref(codes), bounds_ref = ref(bounds), canonical_ref = ref(canonical);
            const auto dest_ref = ref(destinations), unique_ref = ref(unique_blocks);
            const auto offsets_ref = ref(offsets), count_ref = ref(run_count);
            const StorageRef order_ref = canonical_order.is_valid() ? ref(canonical_order) : StorageRef{};
            ReencodeParams host{
                .codes = vk::address(code_ref),
                .bounds = vk::address(bounds_ref),
                .canonical = vk::address(canonical_ref),
                .destinations = vk::address(dest_ref),
                .unique_blocks = vk::address(unique_ref),
                .offsets = vk::address(offsets_ref),
                .run_count = vk::address(count_ref),
                .canonical_order = canonical_order.is_valid() ? vk::address(order_ref) : 0,
                .sorted_count = vk::checked_u32(params.sorted_count, "SH reencode row count exceeds 32-bit indexing"),
                .primitives = vk::checked_u32(params.primitives, "SH reencode primitive count exceeds 32-bit indexing"),
                .decode_rows = vk::checked_u32(params.decode_source_rows, "SH decode row count exceeds 32-bit indexing"),
                .rest = params.rest,
            };
            auto lifetime = std::make_shared<ReencodeLifetime>();
            lifetime->context = context.get();
            lifetime->storage = context->memory().allocate(sizeof(host), 16, {});
            context->memory().copy_host_to_device(CopyRequest{
                .src = raw_storage_ref(&host),
                .dst = lifetime->storage,
                .bytes = sizeof(host),
                .synchronous = false});
            const auto pipeline = pipeline_for(context, Specialization{0, 0, 0, 0, 0, 0, 12});
            const Push push{vk::address(lifetime->storage), 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0};
            std::vector<StorageRef> reads{code_ref, bounds_ref, canonical_ref, dest_ref,
                                          unique_ref, offsets_ref, count_ref, lifetime->storage};
            if (canonical_order.is_valid())
                reads.push_back(order_ref);
            const std::array writes{code_ref, bounds_ref};
            context->recorders().record(
                reads, writes,
                [pipeline, push, count = std::min({host.sorted_count, uint32_t((uint64_t(host.primitives) + 255u) / 256u), context->caps().max_workgroup_count[0]})](VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
                    vkCmdPushConstants(command, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                       0, sizeof(push), &push);
                    vkCmdDispatch(command, count, 1, 1);
                },
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_WHOLE_SIZE, lifetime);
        }

        void gather_canonical(In source, In indices, Out canonical, const ShRowsParams& params) {
            const auto src = ref(source), idx = ref(indices), dst = ref(canonical);
            Push push{vk::address(src), vk::address(dst), vk::address(idx), 0, 0,
                      static_cast<uint32_t>(params.source_rows), static_cast<uint32_t>(params.count),
                      static_cast<uint32_t>(params.count), 0, 0, 0};
            const Specialization spec{1, 0, 2, 0, params.source_rest, params.destination_rest, 7};
            const std::array reads{src, idx};
            const std::array writes{dst};
            launch(push, spec, reads, writes, groups(params.count * params.destination_rest * 3));
        }

        void append_canonical(In canonical, Out destination, const ShRowsParams& params) {
            const auto src = ref(canonical), dst = ref(destination);
            Push push{vk::address(src), vk::address(dst), 0, 0, 0,
                      static_cast<uint32_t>(params.count), static_cast<uint32_t>(params.destination_offset + params.count),
                      static_cast<uint32_t>(params.count), 0, static_cast<uint32_t>(params.destination_offset), 0};
            const Specialization spec{0, 1, 0, 0, params.source_rest, params.destination_rest, 8};
            const std::array reads{src, dst};
            const std::array writes{dst};
            launch(push, spec, reads, writes, groups(params.count * params.source_rest * 3));
        }

        void scatter_canonical(In canonical, In indices, Out destination, const ShRowsParams& params) {
            const auto src = ref(canonical), idx = ref(indices), dst = ref(destination);
            Push push{vk::address(src), vk::address(dst), vk::address(idx), 0, 0,
                      static_cast<uint32_t>(params.count), static_cast<uint32_t>(params.count),
                      static_cast<uint32_t>(params.count), 0, 0, 0};
            const Specialization spec{0, 1, 1, 0, params.source_rest, params.destination_rest, 9};
            const std::array reads{src, idx, dst};
            const std::array writes{dst};
            launch(push, spec, reads, writes, groups(params.count * params.source_rest * 3));
        }

        void fill_bytes(Out storage, const size_t byte_count, const uint8_t value) {
            if (byte_count == 0)
                return;
            const auto dst = ref(storage);
            Push push{0, vk::address(dst), 0, 0, 0, 0, 0,
                      vk::checked_u32(byte_count, "SH byte fill exceeds 32-bit indexing"), value, 0, 0};
            const std::array writes{dst};
            launch(push, {0, 1, 0, 0, 0, 0, 10}, std::span<const StorageRef>{}, writes, groups(byte_count));
        }

        Tensor concatenate_into_arena(const std::span<const Tensor> parts, char* data,
                                      const core::TensorShape shape, const core::DataType dtype,
                                      const core::TensorExecutionTarget target) {
            Tensor result = Tensor::from_blob(data, shape, core::Device::GPU, dtype, target);
            const auto dst = ref(result);
            size_t offset = 0;
            for (const auto& part : parts) {
                LFS_ASSERT_MSG(part.is_valid() && part.is_contiguous() &&
                                   part.device() == core::Device::GPU && part.dtype() == dtype,
                               "SH arena concatenate requires contiguous Vulkan parts of one dtype");
                const auto src = ref(part);
                Push push{vk::address(src), vk::address(dst), 0, 0, 0,
                          0, 0, vk::checked_u32(part.bytes(), "SH arena copy exceeds 32-bit indexing"),
                          0, vk::checked_u32(offset, "SH arena offset exceeds 32-bit indexing"), 0};
                const std::array reads{src};
                const std::array writes{dst};
                launch(push, {0, 1, 0, 0, 0, 0, 11}, reads, writes, groups(part.bytes()));
                offset += part.bytes();
            }
            LFS_ASSERT_MSG(offset == result.bytes(), "SH arena concatenate byte count mismatch");
            return result;
        }

        const ShOps kVulkanShOps{
            .create_run_scratch = create_run_scratch,
            .encode_q16 = encode_q16,
            .decode_q16 = decode_q16,
            .block_ids = block_ids,
            .block_runs = block_runs,
            .reencode_touched = reencode_touched,
            .decode_range = decode_range,
            .zero_rows = zero_rows,
            .gather_swizzled = gather_swizzled,
            .gather_canonical = gather_canonical,
            .append_canonical = append_canonical,
            .scatter_canonical = scatter_canonical,
            .fill_bytes = fill_bytes,
            .concatenate_into_arena = concatenate_into_arena,
        };
    } // namespace

    const lfs::gpu_ops::ShOps& vulkan_sh_ops() { return kVulkanShOps; }
} // namespace lfs::training
