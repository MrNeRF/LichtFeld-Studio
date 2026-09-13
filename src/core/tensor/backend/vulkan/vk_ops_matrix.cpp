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
#include <cstdint>
#include <span>

namespace lfs::core::internal {
    namespace {
        using vk::address;
        using vk::checked_u32;
        using vk::dispatch_groups;
        using vk::kLocalSize;

        constexpr uint32_t kGemmTile = 64;
        constexpr size_t kDotElementsPerPartial = kLocalSize * 8;
        constexpr size_t kDotMaxPartials = 1024;

        struct GemmPush {
            uint64_t a_address;
            uint64_t b_address;
            uint64_t c_address;
            uint64_t bias_address;
            uint32_t m;
            uint32_t n;
            uint32_t k;
            uint32_t batch;
            uint32_t stride_a;
            uint32_t stride_b;
            uint32_t stride_c;
            uint32_t pad0;
        };
        static_assert(sizeof(GemmPush) == 64);

        struct CdistPush {
            uint64_t a_address;
            uint64_t b_address;
            uint64_t output_address;
            uint32_t rows;
            uint32_t columns;
            uint32_t features;
            uint32_t pad0;
            float p;
            uint32_t pad1;
        };
        static_assert(sizeof(CdistPush) == 48);

        struct MiscPush {
            uint64_t output_address;
            uint64_t a_address;
            uint64_t b_address;
            uint32_t rows;
            uint32_t columns;
            uint32_t count;
            uint32_t pad0;
        };
        static_assert(sizeof(MiscPush) == 40);

        // Matrix_misc kinds.
        constexpr uint32_t kEye = 0;
        constexpr uint32_t kDiag = 1;
        constexpr uint32_t kDotPartial = 2;
        constexpr uint32_t kDotFinalize = 3;

        template <class Push>
        void record_dispatch(VulkanContext& context, const VulkanPipeline& pipeline,
                             const Push& push, const std::span<const StorageRef> reads,
                             const std::span<const StorageRef> writes,
                             const uint32_t groups_x, const uint32_t groups_y,
                             const uint32_t groups_z) {
            context.recorders().record(
                reads, writes, [&](const VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      pipeline.pipeline);
                    vkCmdPushConstants(command, pipeline.layout,
                                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(push), &push);
                    vkCmdDispatch(command, groups_x, groups_y, groups_z);
                });
        }

        // C[m][n] = A[m][k] * B, where B is [k][n] or, transposed, [n][k]; the
        // optional bias epilogue applies max(value + bias[row], 0). Rows beyond the
        // device's workgroup-count limit are dispatched in chunks like the CUDA
        // launcher.
        void record_gemm(VulkanContext& context, const StorageRef lhs, const StorageRef rhs,
                         const StorageRef* const bias, const StorageRef output,
                         const GemmProgram& program, const bool transpose_b) {
            if (program.m == 0 || program.n == 0 || program.batch == 0) {
                return;
            }
            LFS_ASSERT_MSG(lhs.dtype == DataType::Float32 && rhs.dtype == DataType::Float32 &&
                               output.dtype == DataType::Float32,
                           "Vulkan GEMM requires Float32 operands");
            LFS_ASSERT_MSG(program.batch <= context.caps().max_workgroup_count[2],
                           "Vulkan GEMM batch count exceeds the device workgroup limit");
            const std::array constants{transpose_b ? 1u : 0u, bias != nullptr ? 1u : 0u};
            const VulkanPipeline& pipeline =
                context.pipelines().specialized("gemm", sizeof(GemmPush), constants);
            const size_t rows_per_dispatch =
                static_cast<size_t>(context.caps().max_workgroup_count[1]) * kGemmTile;
            std::array<StorageRef, 3> reads{lhs, rhs, bias != nullptr ? *bias : StorageRef{}};
            const size_t read_count = bias != nullptr ? 3 : 2;
            const std::array writes{output};
            const uint32_t groups_x = static_cast<uint32_t>((program.n + kGemmTile - 1) / kGemmTile);
            LFS_ASSERT_MSG(groups_x <= context.caps().max_workgroup_count[0],
                           "Vulkan GEMM column count exceeds the device workgroup limit");
            for (size_t row_offset = 0; row_offset < program.m; row_offset += rows_per_dispatch) {
                const size_t rows = std::min(rows_per_dispatch, program.m - row_offset);
                const GemmPush push{
                    .a_address = address(lhs) + row_offset * program.k * sizeof(float),
                    .b_address = address(rhs),
                    .c_address = address(output) + row_offset * program.n * sizeof(float),
                    .bias_address = bias != nullptr ? address(*bias) + row_offset * sizeof(float) : 0,
                    .m = checked_u32(rows, "Vulkan GEMM rows exceed uint32"),
                    .n = checked_u32(program.n, "Vulkan GEMM columns exceed uint32"),
                    .k = checked_u32(program.k, "Vulkan GEMM depth exceeds uint32"),
                    .batch = checked_u32(program.batch, "Vulkan GEMM batch exceeds uint32"),
                    .stride_a = checked_u32(program.m * program.k, "Vulkan GEMM lhs batch stride exceeds uint32"),
                    .stride_b = checked_u32(program.k * program.n, "Vulkan GEMM rhs batch stride exceeds uint32"),
                    .stride_c = checked_u32(program.m * program.n, "Vulkan GEMM output batch stride exceeds uint32"),
                };
                record_dispatch(context, pipeline, push,
                                std::span<const StorageRef>(reads.data(), read_count), writes,
                                groups_x, static_cast<uint32_t>((rows + kGemmTile - 1) / kGemmTile),
                                static_cast<uint32_t>(program.batch));
            }
        }

        void record_misc(VulkanContext& context, const uint32_t kind, const MiscPush& push,
                         const std::span<const StorageRef> reads,
                         const std::span<const StorageRef> writes, const uint32_t groups) {
            const std::array constants{kind};
            const VulkanPipeline& pipeline =
                context.pipelines().specialized("matrix_misc", sizeof(MiscPush), constants);
            record_dispatch(context, pipeline, push, reads, writes, groups, 1, 1);
        }
    } // namespace

    void VulkanBackendOps::sgemm(
        const StorageRef lhs, const StorageRef rhs, const StorageRef output,
        const GemmProgram& program, ExecContext) {
        LFS_FACADE_TRACE(sgemm);
        const auto context = acquire_vulkan_context();
        record_gemm(*context, lhs, rhs, nullptr, output, program, false);
    }

    void VulkanBackendOps::sgemm_tn(
        const StorageRef lhs, const StorageRef rhs, const StorageRef output,
        const GemmProgram& program, ExecContext) {
        LFS_FACADE_TRACE(sgemm_tn);
        const auto context = acquire_vulkan_context();
        record_gemm(*context, lhs, rhs, nullptr, output, program, true);
    }

    void VulkanBackendOps::sgemm_batched(
        const StorageRef lhs, const StorageRef rhs, const StorageRef output,
        const GemmProgram& program, ExecContext) {
        LFS_FACADE_TRACE(sgemm_batched);
        const auto context = acquire_vulkan_context();
        record_gemm(*context, lhs, rhs, nullptr, output, program, false);
    }

    void VulkanBackendOps::sgemm_bias_relu(
        const StorageRef lhs, const StorageRef rhs, const StorageRef bias,
        const StorageRef output, const GemmProgram& program, ExecContext) {
        LFS_FACADE_TRACE(sgemm_bias_relu);
        const auto context = acquire_vulkan_context();
        record_gemm(*context, lhs, rhs, &bias, output, program, false);
    }

    void VulkanBackendOps::dot_product(
        const StorageRef lhs, const StorageRef rhs, const StorageRef output,
        const size_t count, ExecContext) {
        LFS_FACADE_TRACE(dot_product);
        const auto context = acquire_vulkan_context();
        if (count == 0) {
            context->memory().memset(FillRequest{
                .dst = output,
                .bytes = sizeof(float),
                .value = 0,
                .operation = "tensor.dot.zero",
            });
            return;
        }
        const size_t groups =
            std::min(kDotMaxPartials, (count + kDotElementsPerPartial - 1) / kDotElementsPerPartial);
        const std::array reads{lhs, rhs};
        if (groups == 1) {
            const MiscPush push{
                .output_address = address(output),
                .a_address = address(lhs),
                .b_address = address(rhs),
                .count = checked_u32(count, "Vulkan dot product count exceeds uint32"),
            };
            const std::array writes{output};
            record_misc(*context, kDotPartial, push, reads, writes, 1);
            return;
        }
        const StorageRef partials = context->memory().allocate(groups * sizeof(float), 16, {});
        const MiscPush first{
            .output_address = address(partials),
            .a_address = address(lhs),
            .b_address = address(rhs),
            .count = checked_u32(count, "Vulkan dot product count exceeds uint32"),
        };
        const std::array partial_writes{partials};
        record_misc(*context, kDotPartial, first, reads, partial_writes,
                    static_cast<uint32_t>(groups));
        const MiscPush second{
            .output_address = address(output),
            .a_address = address(partials),
            .count = static_cast<uint32_t>(groups),
        };
        const std::array partial_reads{partials};
        const std::array writes{output};
        record_misc(*context, kDotFinalize, second, partial_reads, writes, 1);
        context->memory().deallocate(partials);
    }

    void VulkanBackendOps::diag(
        const StorageRef diagonal, const StorageRef output, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(diag);
        if (count == 0) {
            return;
        }
        const auto context = acquire_vulkan_context();
        const size_t elements = count * count;
        const MiscPush push{
            .output_address = address(output),
            .a_address = address(diagonal),
            .rows = checked_u32(count, "Vulkan diag size exceeds uint32"),
            .columns = checked_u32(count, "Vulkan diag size exceeds uint32"),
            .count = checked_u32(elements, "Vulkan diag element count exceeds uint32"),
        };
        const std::array reads{diagonal};
        const std::array writes{output};
        record_misc(*context, kDiag, push, reads, writes, dispatch_groups(*context, elements));
    }

    void VulkanBackendOps::eye(
        const StorageRef output, const size_t rows, const size_t columns, ExecContext) {
        LFS_FACADE_TRACE(eye);
        const size_t elements = rows * columns;
        if (elements == 0) {
            return;
        }
        const auto context = acquire_vulkan_context();
        const MiscPush push{
            .output_address = address(output),
            .rows = checked_u32(rows, "Vulkan eye rows exceed uint32"),
            .columns = checked_u32(columns, "Vulkan eye columns exceed uint32"),
            .count = checked_u32(elements, "Vulkan eye element count exceeds uint32"),
        };
        const std::array writes{output};
        record_misc(*context, kEye, push, {}, writes, dispatch_groups(*context, elements));
    }

    void VulkanBackendOps::cdist(
        const StorageRef lhs, const StorageRef rhs, const StorageRef output,
        const size_t lhs_rows, const size_t rhs_rows, const size_t columns, const float p,
        ExecContext) {
        LFS_FACADE_TRACE(cdist);
        const size_t outputs = lhs_rows * rhs_rows;
        if (outputs == 0) {
            return;
        }
        const auto context = acquire_vulkan_context();
        const CdistPush push{
            .a_address = address(lhs),
            .b_address = address(rhs),
            .output_address = address(output),
            .rows = checked_u32(lhs_rows, "Vulkan cdist rows exceed uint32"),
            .columns = checked_u32(rhs_rows, "Vulkan cdist columns exceed uint32"),
            .features = checked_u32(columns, "Vulkan cdist features exceed uint32"),
            .p = p,
        };
        const VulkanPipeline& pipeline =
            context->pipelines().specialized("cdist", sizeof(CdistPush), {});
        const std::array reads{lhs, rhs};
        const std::array writes{output};
        record_dispatch(*context, pipeline, push, reads, writes,
                        dispatch_groups(*context, outputs), 1, 1);
    }

} // namespace lfs::core::internal
