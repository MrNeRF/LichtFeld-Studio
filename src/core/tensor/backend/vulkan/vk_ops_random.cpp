/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../facade_trace.hpp"
#include "vk_backend_ops.hpp"

#include "../../internal/tensor_impl.hpp"
#include "core/assert.hpp"
#include "vk_context.hpp"
#include "vk_memory.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>

namespace lfs::core::internal {
    namespace {
        using vk::address;
        using vk::checked_u32;
        using vk::dispatch_groups;

        // random.slang kinds.
        constexpr uint32_t kUniform = 0;
        constexpr uint32_t kBernoulli = 1;
        constexpr uint32_t kRandint = 2;
        constexpr uint32_t kNormal = 3;
        constexpr uint32_t kMultinomialReplacement = 4;
        constexpr uint32_t kGumbelKeys = 5;
        constexpr uint32_t kWeightStatistics = 7;
        constexpr uint32_t kRunningSums = 8;
        constexpr uint32_t kBlockOffsets = 9;
        // Weights per block of running sums; random.slang's kSumBlock.
        constexpr uint32_t kSumBlock = 1024;

        // A power of two that brings the largest weight near 2^0.
        float multinomial_scale(const float maximum) {
            const uint32_t exponent = std::clamp((std::bit_cast<uint32_t>(maximum) >> 23) & 255u, 1u, 253u);
            return std::bit_cast<float>((254u - exponent) << 23);
        }

        struct RandomPush {
            uint64_t output_address;
            uint64_t weights_address;
            uint64_t keys_address;
            uint64_t seed;
            uint32_t count;
            uint32_t sample_count;
            int32_t low;
            int32_t high;
            float first;
            float second;
            float total;
            uint32_t pad0;
        };
        static_assert(sizeof(RandomPush) == 64);

        void record_random(VulkanContext& context, const uint32_t kind, const RandomPush& push,
                           const std::span<const StorageRef> reads,
                           const std::span<const StorageRef> writes, const uint32_t groups) {
            const std::array constants{kind};
            const VulkanPipeline& pipeline =
                context.pipelines().specialized("random", sizeof(RandomPush), constants);
            context.recorders().record(
                reads, writes, [&](const VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      pipeline.pipeline);
                    vkCmdPushConstants(command, pipeline.layout,
                                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(push), &push);
                    vkCmdDispatch(command, groups, 1, 1);
                });
        }

        // Elementwise draws: every element is one Philox block keyed by the seed.
        void draw_elements(const uint32_t kind, const StorageRef output,
                           const RandomProgram& program, const uint64_t seed) {
            if (program.count == 0) {
                return;
            }
            const auto context = acquire_vulkan_context();
            const RandomPush push{
                .output_address = address(output),
                .seed = seed,
                .count = checked_u32(program.count, "Vulkan random count exceeds uint32"),
                .low = program.low,
                .high = program.high,
                .first = program.first,
                .second = program.second,
            };
            const std::array writes{output};
            record_random(*context, kind, push, {}, writes, dispatch_groups(*context, program.count));
        }

        struct WeightStatistics {
            float maximum;
            uint32_t invalid;
        };

        // One workgroup finds the largest weight and flags negative or non-finite ones;
        // the result is read back so the host can reject bad inputs like the CUDA
        // path does.
        WeightStatistics weight_statistics(VulkanContext& context, const StorageRef weights,
                                           const size_t count) {
            const StorageRef scratch = context.memory().allocate(16, 16, {});
            const RandomPush push{
                .output_address = address(scratch),
                .weights_address = address(weights),
                .count = checked_u32(count, "Vulkan multinomial category count exceeds uint32"),
            };
            const std::array reads{weights};
            const std::array writes{scratch};
            record_random(context, kWeightStatistics, push, reads, writes, 1);
            WeightStatistics statistics{};
            backend_ops(GpuBackend::Vulkan).copy_device_to_host(CopyRequest{
                .src = scratch,
                .dst = raw_storage_ref(&statistics),
                .bytes = sizeof(statistics),
                .synchronous = true,
                .operation = "tensor.multinomial.weight_statistics",
            });
            context.memory().deallocate(scratch);
            return statistics;
        }
    } // namespace

    void VulkanBackendOps::uniform(
        const StorageRef output, const RandomProgram& program, ExecContext) {
        LFS_FACADE_TRACE(uniform);
        draw_elements(kUniform, output, program, program.seed);
    }

    void VulkanBackendOps::bernoulli(
        const StorageRef output, const RandomProgram& program, ExecContext) {
        LFS_FACADE_TRACE(bernoulli);
        draw_elements(kBernoulli, output, program, program.seed);
    }

    void VulkanBackendOps::randint(
        const StorageRef output, const RandomProgram& program, ExecContext) {
        LFS_FACADE_TRACE(randint);
        draw_elements(kRandint, output, program, program.seed);
    }

    void VulkanBackendOps::normal(
        const StorageRef output, StorageRef, const RandomProgram& program, ExecContext) {
        LFS_FACADE_TRACE(normal);
        // The CUDA path draws from the process generator's stream; Vulkan draws
        // every element from its own Philox block under the program seed, so odd
        // counts need no scratch.
        draw_elements(kNormal, output, program, program.seed);
    }

    void VulkanBackendOps::multinomial(
        const StorageRef weights, const StorageRef output, const RandomProgram& program,
        ExecContext) {
        LFS_FACADE_TRACE(multinomial);
        if (program.count == 0 || program.sample_count == 0) {
            return;
        }
        LFS_ASSERT_MSG(weights.dtype == DataType::Float32 && output.dtype == DataType::Int64,
                       "Vulkan multinomial requires Float32 weights and Int64 samples");
        const auto context = acquire_vulkan_context();
        const WeightStatistics statistics = weight_statistics(*context, weights, program.count);
        LFS_ASSERT_MSG(statistics.invalid == 0,
                       "multinomial weights must be finite and non-negative");
        LFS_ASSERT_MSG(statistics.maximum > 0.0f,
                       "multinomial weights must have a positive finite sum");
        const uint32_t categories = checked_u32(program.count, "Vulkan multinomial category count exceeds uint32");
        const uint32_t samples = checked_u32(program.sample_count, "Vulkan multinomial sample count exceeds uint32");
        if (program.replacement) {
            // Running sums within blocks, the blocks' offsets, then a binary
            // search per draw. The weights are scaled so their maximum sits
            // near 2^0, which keeps the sums finite.
            const uint32_t blocks = (categories + kSumBlock - 1) / kSumBlock;
            const StorageRef sums =
                context->memory().allocate((program.count + blocks + 1) * sizeof(float), 16, {});
            const RandomPush push{
                .output_address = address(output),
                .weights_address = address(weights),
                .keys_address = address(sums),
                .seed = program.seed,
                .count = categories,
                .sample_count = samples,
                .first = multinomial_scale(statistics.maximum),
            };
            const std::array sum_reads{weights};
            const std::array sum_writes{sums};
            record_random(*context, kRunningSums, push, sum_reads, sum_writes, dispatch_groups(*context, blocks));
            record_random(*context, kBlockOffsets, push, sum_writes, sum_writes, 1);
            const std::array draw_reads{sums};
            const std::array draw_writes{output};
            record_random(*context, kMultinomialReplacement, push, draw_reads, draw_writes,
                          dispatch_groups(*context, program.sample_count));
            context->memory().deallocate(sums);
            return;
        }
        LFS_ASSERT_MSG(program.sample_count <= program.count,
                       "multinomial sample count exceeds weights without replacement");
        // Gumbel-top-k: the sample_count largest perturbed log-weights. The
        // sort is stable, so ties fall back to the lower index.
        const StorageRef keys = context->memory().allocate(program.count * sizeof(float), 16, {});
        const StorageRef order = context->memory().allocate(program.count * sizeof(int64_t), 16, {});
        const RandomPush key_push{
            .weights_address = address(weights),
            .keys_address = address(keys),
            .seed = program.seed,
            .count = categories,
            .sample_count = samples,
        };
        const std::array key_reads{weights};
        const std::array key_writes{keys};
        record_random(*context, kGumbelKeys, key_push, key_reads, key_writes,
                      dispatch_groups(*context, program.count));
        StorageRef key_values = keys, key_order = order;
        key_values.dtype = DataType::Float32;
        key_order.dtype = DataType::Int64;
        sort_1d(key_values, key_order, program.count, SortProgram{.dim_size = program.count, .descending = true}, {});
        copy_device_to_device(CopyRequest{
            .src = key_order,
            .dst = output,
            .bytes = program.sample_count * sizeof(int64_t),
            .synchronous = false,
        });
        context->memory().deallocate(keys);
        context->memory().deallocate(order);
    }

} // namespace lfs::core::internal
