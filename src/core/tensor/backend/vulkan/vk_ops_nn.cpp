/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../facade_trace.hpp"
#include "vk_backend_ops.hpp"

#include "../../internal/tensor_impl.hpp"
#include "core/assert.hpp"
#include "vk_context.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace lfs::core::internal {
    namespace {
        using vk::address;
        using vk::checked_u32;
        using vk::dispatch_groups;

        // nn.slang kinds.
        constexpr uint32_t kMaxPool = 0;
        constexpr uint32_t kAdaptiveAvgPool = 1;
        constexpr uint32_t kBiasAdd = 2;
        constexpr uint32_t kBiasRelu = 3;
        constexpr uint32_t kRelu = 4;

        struct NnPush {
            uint64_t input_address;
            uint64_t bias_address;
            uint64_t output_address;
            uint32_t total;
            uint32_t batch;
            uint32_t channels;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t output_height;
            uint32_t output_width;
            uint32_t kernel;
            uint32_t stride;
            int32_t padding;
            uint32_t spatial;
            uint32_t pad0;
        };
        static_assert(sizeof(NnPush) == 72);

        uint32_t checked_dimension(const int value, const char* const description) {
            LFS_ASSERT_MSG(value >= 0, description);
            return static_cast<uint32_t>(value);
        }

        void record_nn(const uint32_t kind, const NnPush& push,
                       const std::span<const StorageRef> reads, const StorageRef output) {
            if (push.total == 0) {
                return;
            }
            const auto context = acquire_vulkan_context();
            const std::array constants{kind};
            const VulkanPipeline& pipeline =
                context->pipelines().specialized("nn", sizeof(NnPush), constants);
            const std::array writes{output};
            context->recorders().record(
                reads, writes, [&](const VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      pipeline.pipeline);
                    vkCmdPushConstants(command, pipeline.layout,
                                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(push), &push);
                    vkCmdDispatch(command, dispatch_groups(*context, push.total), 1, 1);
                });
        }

        NnPush pool_push(const StorageRef input, const StorageRef output,
                         const PoolProgram& program) {
            const size_t total = static_cast<size_t>(checked_dimension(program.batch, "pool batch")) *
                                 checked_dimension(program.channels, "pool channels") *
                                 checked_dimension(program.output_height, "pool output height") *
                                 checked_dimension(program.output_width, "pool output width");
            return NnPush{
                .input_address = address(input),
                .output_address = address(output),
                .total = checked_u32(total, "Vulkan pooling output count exceeds uint32"),
                .batch = static_cast<uint32_t>(program.batch),
                .channels = static_cast<uint32_t>(program.channels),
                .input_height = checked_dimension(program.input_height, "pool input height"),
                .input_width = checked_dimension(program.input_width, "pool input width"),
                .output_height = static_cast<uint32_t>(program.output_height),
                .output_width = static_cast<uint32_t>(program.output_width),
                .kernel = checked_dimension(program.kernel_size, "pool kernel size"),
                .stride = checked_dimension(program.stride, "pool stride"),
                .padding = program.padding,
            };
        }

        void record_bias(const uint32_t kind, const StorageRef input, const StorageRef bias,
                         const StorageRef output, const int count, const int channels,
                         const int spatial_size) {
            LFS_ASSERT_MSG(channels > 0 && spatial_size > 0, "Vulkan bias kernel requires positive layout sizes");
            const NnPush push{
                .input_address = address(input),
                .bias_address = address(bias),
                .output_address = address(output),
                .total = checked_dimension(count, "bias element count"),
                .channels = static_cast<uint32_t>(channels),
                .spatial = static_cast<uint32_t>(spatial_size),
            };
            const std::array reads{input, bias};
            record_nn(kind, push, reads, output);
        }
    } // namespace

    void VulkanBackendOps::max_pool2d(
        const StorageRef input, const StorageRef output, const PoolProgram& program,
        ExecContext) {
        LFS_FACADE_TRACE(max_pool2d);
        LFS_ASSERT_MSG(program.kernel_size > 0 && program.stride > 0,
                       "Vulkan max_pool2d requires a positive kernel and stride");
        const std::array reads{input};
        record_nn(kMaxPool, pool_push(input, output, program), reads, output);
    }

    void VulkanBackendOps::adaptive_avg_pool2d(
        const StorageRef input, const StorageRef output, const PoolProgram& program,
        ExecContext) {
        LFS_FACADE_TRACE(adaptive_avg_pool2d);
        LFS_ASSERT_MSG(program.output_height > 0 && program.output_width > 0,
                       "Vulkan adaptive_avg_pool2d requires a positive output size");
        const std::array reads{input};
        record_nn(kAdaptiveAvgPool, pool_push(input, output, program), reads, output);
    }

    void VulkanBackendOps::bias_add(
        const StorageRef input, const StorageRef bias, const StorageRef output,
        const int count, const int channels, const int spatial_size, ExecContext) {
        LFS_FACADE_TRACE(bias_add);
        record_bias(kBiasAdd, input, bias, output, count, channels, spatial_size);
    }

    void VulkanBackendOps::bias_relu(
        const StorageRef input, const StorageRef bias, const StorageRef output,
        const int count, const int channels, const int spatial_size, ExecContext) {
        LFS_FACADE_TRACE(bias_relu);
        record_bias(kBiasRelu, input, bias, output, count, channels, spatial_size);
    }

    void VulkanBackendOps::relu(
        const StorageRef input, const StorageRef output, const int count, ExecContext) {
        LFS_FACADE_TRACE(relu);
        const NnPush push{
            .input_address = address(input),
            .output_address = address(output),
            .total = checked_dimension(count, "relu element count"),
        };
        const std::array reads{input};
        record_nn(kRelu, push, reads, output);
    }

} // namespace lfs::core::internal
