/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../facade_trace.hpp"
#include "../readback_buffer.hpp"
#include "../scalar_operand.hpp"
#include "metal_backend_ops.hpp"
#include "metal_context.hpp"

#include "core/assert.hpp"
#include "core/detail/fused_pointwise.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace lfs::core::internal {

    namespace {
        using metal::acquire_context;
        using metal::Context;
        using metal::kThreadgroupWidth;
        using metal::live_context;
        using metal::param_bytes;

        MTLSize threads(const size_t count) {
            return MTLSizeMake(count, 1, 1);
        }

        uint32_t checked_u32(const size_t value, const char* const description) {
            LFS_ASSERT_MSG(value <= std::numeric_limits<uint32_t>::max(), description);
            return static_cast<uint32_t>(value);
        }

        struct PointwiseParams {
            uint64_t lhs_offset;
            uint64_t rhs_offset;
            uint64_t output_offset;
            uint64_t scalar_int64;
            float scalar_float;
            uint32_t count;
            uint32_t scalar_kind;
            uint32_t flags;
        };
        static_assert(sizeof(PointwiseParams) == 48);

        API_AVAILABLE(macos(26.0))
        void dispatch_pointwise(const PointwiseProgram& program, const StorageRef lhs,
                                const StorageRef rhs, const StorageRef output,
                                const size_t count, const uint32_t arity) {
            if (count == 0)
                return;
            LFS_ASSERT_MSG(lhs.dtype == program.in_dtype && output.dtype == program.out_dtype,
                           "Metal pointwise program dtype does not match storage");
            LFS_ASSERT_MSG(arity != 2 || rhs.dtype == program.in_dtype,
                           "Metal pointwise rhs dtype does not match program");
            const auto context = acquire_context();
            const auto lhs_at = context->locate(lhs);
            const auto rhs_at = arity == 2 ? context->locate(rhs) : lhs_at;
            const auto output_at = context->locate(output);
            const bool vectorized = program.in_dtype == DataType::Float32 &&
                                    (program.out_dtype == DataType::Float32 || program.out_dtype == DataType::Bool) &&
                                    count % 4 == 0 && lhs_at.offset % 16 == 0 && output_at.offset % 16 == 0 &&
                                    (arity != 2 || rhs_at.offset % 16 == 0) && program.scalar.scalar_on_right;
            const auto pipeline = context->pipeline(
                "pointwise", {{0, static_cast<uint32_t>(program.op)},
                              {1, static_cast<uint32_t>(program.in_dtype)},
                              {2, static_cast<uint32_t>(program.out_dtype)},
                              {3, arity},
                              {4, vectorized ? 1u : 0u}});
            const PointwiseParams params{
                .lhs_offset = lhs_at.offset,
                .rhs_offset = rhs_at.offset,
                .output_offset = output_at.offset,
                .scalar_int64 = scalar_integer(program.scalar),
                .scalar_float = scalar_float(program.scalar),
                .count = checked_u32(count, "Metal pointwise count exceeds uint32"),
                .scalar_kind = static_cast<uint32_t>(program.scalar.kind),
                .flags = program.scalar.scalar_on_right ? 1u : 0u,
            };
            const std::array uses{lhs, arity == 2 ? rhs : lhs, output};
            context->dispatch(uses, {.pipeline = pipeline,
                                     .buffers = {lhs_at.address, rhs_at.address, output_at.address},
                                     .params = param_bytes(params),
                                     .grid = threads(vectorized ? count / 4 : count)});
        }

        struct ChainOp {
            float scalar;
            uint32_t padding;
            uint64_t rhs_address;
        };

        struct ChainParams {
            uint64_t input_offset;
            uint64_t output_offset;
            uint32_t count;
            uint32_t padding;
            std::array<ChainOp, tensor_ops::FUSED_POINTWISE_MAX_OPS> ops;
        };
        static_assert(sizeof(ChainOp) == 16 && sizeof(ChainParams) == 24 + 16 * 16);

        struct FillParams {
            uint64_t output_offset;
            uint64_t pattern;
            uint64_t count;
        };

        struct CopyParams {
            uint64_t source_offset;
            uint64_t destination_offset;
            uint64_t count;
        };

        // Fills bytes with a pattern of element_size bytes, widening to 16-byte
        // stores when the range allows it.
        API_AVAILABLE(macos(26.0))
        void encode_fill(Context& context, const StorageRef output, const size_t bytes,
                         uint64_t pattern, size_t element_size) {
            if (bytes == 0)
                return;
            const auto output_at = context.locate(output);
            if (element_size == 1) {
                pattern = (pattern & 0xffu) * 0x01010101u;
                element_size = 4;
                if (bytes % 4 != 0 || output_at.offset % 4 != 0)
                    element_size = 1;
            }
            if (element_size == 4 && bytes % 16 == 0 && output_at.offset % 16 == 0)
                element_size = 16;
            const auto pipeline = context.pipeline("fill", {{5, static_cast<uint32_t>(element_size)}});
            const FillParams params{.output_offset = output_at.offset, .pattern = pattern, .count = bytes / element_size};
            const std::array uses{output};
            context.dispatch(uses, {.pipeline = pipeline,
                                    .buffers = {output_at.address},
                                    .params = param_bytes(params),
                                    .grid = threads(params.count)});
        }

        API_AVAILABLE(macos(26.0))
        void encode_copy(Context& context, const StorageRef source, const StorageRef destination,
                         const size_t bytes) {
            if (bytes == 0)
                return;
            const auto source_at = context.locate(source);
            const auto destination_at = context.locate(destination);
            const auto aligned = [&](const size_t alignment) {
                return bytes % alignment == 0 && source_at.offset % alignment == 0 &&
                       destination_at.offset % alignment == 0;
            };
            const uint32_t element_size = aligned(16) ? 16 : aligned(4) ? 4 : 1;
            const auto pipeline = context.pipeline("copy_bytes", {{5, element_size}});
            const CopyParams params{
                .source_offset = source_at.offset,
                .destination_offset = destination_at.offset,
                .count = bytes / element_size,
            };
            const std::array uses{source, destination};
            context.dispatch(uses, {.pipeline = pipeline,
                                    .buffers = {source_at.address, destination_at.address},
                                    .params = param_bytes(params),
                                    .grid = threads(params.count)});
        }

        struct StridedParams {
            uint64_t input_offset;
            uint64_t output_offset;
            std::array<uint32_t, MAX_TENSOR_RANK> dims;
            std::array<uint32_t, MAX_TENSOR_RANK> strides;
            uint32_t rank;
            uint32_t count;
        };
        static_assert(sizeof(StridedParams) == 88);

        struct TransposeParams {
            uint64_t input_offset;
            uint64_t output_offset;
            uint32_t rows;
            uint32_t columns;
            uint32_t input_stride;
            uint32_t padding;
        };

        // Gathers a strided input into contiguous output or, for a scatter, the
        // reverse; the layout describes the strided side.
        API_AVAILABLE(macos(26.0))
        void encode_strided(const StorageRef input, const StorageRef output, const StridedLayout& layout,
                            const bool scatter, const DataType input_dtype, const DataType output_dtype) {
            if (layout.element_count == 0)
                return;
            LFS_ASSERT_MSG(layout.rank <= MAX_TENSOR_RANK, "Metal strided layout rank exceeds MAX_TENSOR_RANK");
            const auto context = acquire_context();
            const auto input_at = context->locate(input);
            const auto output_at = context->locate(output);
            // A transposed 2D view (rows contiguous, columns strided) goes through tiles.
            if (!scatter && input_dtype == output_dtype && layout.rank == 2 && layout.strides[0] == 1 &&
                layout.strides[1] != 1 && layout.dims[0] > 1) {
                const TransposeParams params{
                    .input_offset = input_at.offset,
                    .output_offset = output_at.offset,
                    .rows = checked_u32(layout.dims[0], "Metal transpose rows exceed uint32"),
                    .columns = checked_u32(layout.dims[1], "Metal transpose columns exceed uint32"),
                    .input_stride = checked_u32(layout.strides[1], "Metal transpose stride exceeds uint32"),
                    .padding = 0,
                };
                const auto pipeline = context->pipeline(
                    "transpose_2d", {{5, static_cast<uint32_t>(dtype_size(output_dtype))}});
                const std::array uses{input, output};
                context->dispatch(uses, {.pipeline = pipeline,
                                         .buffers = {input_at.address, output_at.address},
                                         .params = param_bytes(params),
                                         .grid = MTLSizeMake((params.rows + 31) / 32, (params.columns + 31) / 32, 1),
                                         .group_size = MTLSizeMake(32, 8, 1)});
                return;
            }
            StridedParams params{
                .input_offset = input_at.offset,
                .output_offset = output_at.offset,
                .dims = {},
                .strides = {},
                .rank = static_cast<uint32_t>(layout.rank),
                .count = checked_u32(layout.element_count, "Metal strided count exceeds uint32"),
            };
            for (size_t axis = 0; axis < layout.rank; ++axis) {
                params.dims[axis] = checked_u32(layout.dims[axis], "Metal strided dim exceeds uint32");
                params.strides[axis] = checked_u32(layout.strides[axis], "Metal strided stride exceeds uint32");
            }
            const auto pipeline = context->pipeline(
                "strided_copy", {{1, static_cast<uint32_t>(input_dtype)},
                                 {2, static_cast<uint32_t>(output_dtype)},
                                 {5, static_cast<uint32_t>(dtype_size(output_dtype))},
                                 {7, scatter ? 1u : 0u}});
            const std::array uses{input, output};
            context->dispatch(uses, {.pipeline = pipeline,
                                     .buffers = {input_at.address, output_at.address},
                                     .params = param_bytes(params),
                                     .grid = threads(layout.element_count)});
        }

        std::array<uint32_t, MAX_TENSOR_RANK> shader_dims(const StridedLayout& layout) {
            std::array<uint32_t, MAX_TENSOR_RANK> dims{};
            for (size_t axis = 0; axis < layout.rank; ++axis)
                dims[axis] = checked_u32(layout.dims[axis], "Metal where dim exceeds uint32");
            return dims;
        }

        struct ReduceParams {
            uint64_t input_offset;
            uint32_t count;
            uint32_t padding;
        };

        // The kernel's IEEE rules for folding the per-threadgroup partials on
        // the host: first NaN wins, and equal zeros follow the sign rule.
        float ieee_extreme(const float lhs, const float rhs, const bool maximum) {
            const auto lhs_bits = std::bit_cast<uint32_t>(lhs), rhs_bits = std::bit_cast<uint32_t>(rhs);
            if ((lhs_bits & 0x7fffffffu) > 0x7f800000u)
                return lhs;
            if ((rhs_bits & 0x7fffffffu) > 0x7f800000u)
                return rhs;
            if ((lhs_bits & 0x7fffffffu) == 0 && (rhs_bits & 0x7fffffffu) == 0)
                return std::bit_cast<float>((maximum ? lhs_bits & rhs_bits : lhs_bits | rhs_bits) & 0x80000000u);
            return maximum ? (lhs < rhs ? rhs : lhs) : (rhs < lhs ? rhs : lhs);
        }

        // One dispatch folds each threadgroup's share to a (value, compensation)
        // pair; the host folds the pairs, since it waits for the result anyway.
        API_AVAILABLE(macos(26.0))
        float scalar_reduce(const uint32_t op, const StorageRef input, const size_t count) {
            LFS_ASSERT_MSG(input.dtype == DataType::Float32,
                           "Metal scalar reduction requires Float32 input");
            const bool extreme = op == metal::kReduceMax || op == metal::kReduceMin;
            if (count == 0) {
                return op == metal::kReduceMax   ? -std::numeric_limits<float>::infinity()
                       : op == metal::kReduceMin ? std::numeric_limits<float>::infinity()
                                                 : 0.0f;
            }
            const auto context = acquire_context();
            constexpr size_t kElementsPerGroup = kThreadgroupWidth * 4;
            const auto groups = static_cast<uint32_t>(
                std::clamp<size_t>((count + kElementsPerGroup - 1) / kElementsPerGroup, 1, 1024));
            const StorageRef partials = context->allocate(groups * 2 * sizeof(float));
            const auto input_at = context->locate(input);
            const auto partials_at = context->locate(partials);
            const auto pipeline = context->pipeline("reduce_partial", {{6, op}});
            const ReduceParams params{
                .input_offset = input_at.offset,
                .count = checked_u32(count, "Metal reduction count exceeds uint32"),
                .padding = 0,
            };
            const std::array uses{input, partials};
            context->dispatch(uses, {.pipeline = pipeline,
                                     .buffers = {input_at.address, partials_at.address + partials_at.offset},
                                     .params = param_bytes(params),
                                     .grid = MTLSizeMake(groups, 1, 1),
                                     .group_size = MTLSizeMake(kThreadgroupWidth, 1, 1)});
            context->wait(context->pending(partials));
            const auto* const pairs = reinterpret_cast<const float*>(context->host(partials));
            float accumulator = pairs[0], compensation = extreme ? 0.0f : pairs[1];
            for (uint32_t group = 1; group < groups; ++group) {
                const float value = pairs[2 * group];
                if (extreme) {
                    accumulator = ieee_extreme(accumulator, value, op == metal::kReduceMax);
                    continue;
                }
                const float total = accumulator + value;
                const float carried = total - accumulator;
                compensation += (accumulator - (total - carried)) + (value - carried) + pairs[2 * group + 1];
                accumulator = total;
            }
            context->release(partials);
            if (extreme)
                return accumulator;
            const float sum = std::isfinite(compensation) ? accumulator + compensation : accumulator;
            return op == metal::kReduceMean ? sum * (1.0f / static_cast<float>(count)) : sum;
        }

        bool is_contiguous(const StridedLayout& layout) {
            size_t expected = 1;
            for (size_t dimension = layout.rank; dimension-- > 0;) {
                if (layout.dims[dimension] != 1 && layout.strides[dimension] != expected)
                    return false;
                expected *= layout.dims[dimension];
            }
            return true;
        }

        std::byte* host_bytes(const StorageRef& storage) {
            return static_cast<std::byte*>(storage.data) + storage.byte_offset;
        }

        // A readback snapshots its source on the GPU timeline into a shared
        // staging block; poll() copies it out once that batch completed.
        class API_AVAILABLE(macos(26.0)) MetalReadbackBuffer final : public ReadbackBuffer {
        public:
            MetalReadbackBuffer() : context_(acquire_context()) {}

            ~MetalReadbackBuffer() override {
                if (capacity_)
                    context_->release(staging_);
            }

            void enqueue(const StorageRef source, const size_t bytes, ExecContext) override {
                LFS_FACADE_TRACE(service_enqueue_readback);
                LFS_ASSERT_MSG(!pending_, "Readback staging is already in use");
                if (bytes > capacity_) {
                    const StorageRef replacement = context_->allocate(bytes);
                    if (capacity_)
                        context_->release(staging_);
                    staging_ = replacement;
                    capacity_ = bytes;
                }
                bytes_ = bytes;
                encode_copy(*context_, source, staging_, bytes);
                serial_ = context_->pending(staging_);
                context_->flush();
                pending_ = true;
            }

            void wait() override {
                context_->wait(serial_);
            }

            bool poll(void* const destination) override {
                LFS_FACADE_TRACE(service_readback_poll);
                if (context_->completed() < serial_)
                    return false;
                if (destination)
                    std::memcpy(destination, context_->host(staging_), bytes_);
                pending_ = false;
                return true;
            }

        private:
            std::shared_ptr<Context> context_;
            StorageRef staging_{};
            size_t capacity_ = 0, bytes_ = 0;
            uint64_t serial_ = 0;
            bool pending_ = false;
        };
    } // namespace

    void MetalBackendOps::unary(const PointwiseProgram& program, const StorageRef input,
                                const StorageRef output, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(unary);
        dispatch_pointwise(program, input, {}, output, count, 1);
    }

    void MetalBackendOps::binary(const PointwiseProgram& program, const StorageRef lhs,
                                 const StorageRef rhs, const StorageRef output,
                                 const size_t count, ExecContext) {
        LFS_FACADE_TRACE(binary);
        dispatch_pointwise(program, lhs, rhs, output, count, 2);
    }

    void MetalBackendOps::scalar(const PointwiseProgram& program, const StorageRef input,
                                 const StorageRef output, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(scalar);
        dispatch_pointwise(program, input, {}, output, count, 1);
    }

    void MetalBackendOps::fused_pointwise_chain(
        const StorageRef input, const StorageRef output, const size_t count,
        const tensor_ops::FusedPointwiseOpChain& chain,
        const std::span<const StorageRef> rhs_storages, ExecContext) {
        LFS_FACADE_TRACE(fused_pointwise_chain);
        if (count == 0)
            return;
        LFS_ASSERT_MSG(input.dtype == DataType::Float32 && output.dtype == DataType::Float32 &&
                           chain.num_ops > 0 && chain.num_ops <= tensor_ops::FUSED_POINTWISE_MAX_OPS,
                       "Metal fused pointwise chain requires Float32 and 1 to 16 operations");
        const auto context = acquire_context();
        const auto input_at = context->locate(input);
        const auto output_at = context->locate(output);
        ChainParams params{
            .input_offset = input_at.offset,
            .output_offset = output_at.offset,
            .count = checked_u32(count, "Metal fused pointwise count exceeds uint32"),
            .padding = 0,
            .ops = {},
        };
        // The op kinds specialize the kernel; tensor operands are read through
        // their addresses.
        std::array<uint32_t, 3> kinds{};
        bool vectorized = input_at.offset % 16 == 0 && output_at.offset % 16 == 0;
        for (int i = 0; i < chain.num_ops; ++i) {
            const auto& op = chain.ops[i];
            kinds[i / 6] |= static_cast<uint32_t>(op.kind & 31u) << (5 * (i % 6));
            params.ops[i].scalar = op.scalar;
            if (op.kind < 4 || op.kind > 7)
                continue;
            const auto rhs_at = context->locate(op.rhs);
            params.ops[i].rhs_address = rhs_at.address + rhs_at.offset;
            vectorized = vectorized && params.ops[i].rhs_address % 16 == 0;
        }
        const auto pipeline = context->pipeline(
            "pointwise_chain", {{4, vectorized ? 1u : 0u}, {8, static_cast<uint32_t>(chain.num_ops)},
                                {9, kinds[0]}, {10, kinds[1]}, {11, kinds[2]}});
        std::vector<StorageRef> uses{input, output};
        uses.insert(uses.end(), rhs_storages.begin(), rhs_storages.end());
        context->dispatch(uses, {.pipeline = pipeline,
                                 .buffers = {input_at.address, output_at.address},
                                 .params = param_bytes(params),
                                 .grid = threads(vectorized ? (count + 3) / 4 : count)});
    }

    void MetalBackendOps::convert_type(const StorageRef input, const StorageRef output,
                                       const size_t count, ExecContext) {
        LFS_FACADE_TRACE(convert_type);
        if (count == 0)
            return;
        struct ConvertParams {
            uint64_t input_offset;
            uint64_t output_offset;
            uint32_t count;
            uint32_t padding;
        };
        const auto context = acquire_context();
        const auto input_at = context->locate(input);
        const auto output_at = context->locate(output);
        const auto pipeline = context->pipeline(
            "convert", {{1, static_cast<uint32_t>(input.dtype)}, {2, static_cast<uint32_t>(output.dtype)}});
        const ConvertParams params{
            .input_offset = input_at.offset,
            .output_offset = output_at.offset,
            .count = checked_u32(count, "Metal conversion count exceeds uint32"),
        };
        const std::array uses{input, output};
        context->dispatch(uses, {.pipeline = pipeline,
                                 .buffers = {input_at.address, output_at.address},
                                 .params = param_bytes(params),
                                 .grid = threads(count)});
    }

    void MetalBackendOps::fill_strided(const StorageRef output, const StridedLayout& layout,
                                       const ScalarOperand value, ExecContext context) {
        LFS_FACADE_TRACE(fill_strided);
        if (!is_contiguous(layout))
            throw TensorError("Metal backend: strided fill is not implemented yet");
        load_fill(output, layout.element_count, value, context);
    }

    void MetalBackendOps::load_fill(const StorageRef output, const size_t count,
                                    const ScalarOperand value, ExecContext) {
        LFS_FACADE_TRACE(load_fill);
        const size_t element_size = dtype_size(output.dtype);
        encode_fill(*acquire_context(), output, count * element_size,
                    fill_pattern(output.dtype, value), element_size);
    }

    void MetalBackendOps::load_arange(const StorageRef output, const size_t count,
                                      const ScalarOperand start, const ScalarOperand step,
                                      ExecContext) {
        LFS_FACADE_TRACE(load_arange);
        if (count == 0)
            return;
        LFS_ASSERT_MSG(output.dtype == DataType::Float32 || output.dtype == DataType::Int32,
                       "Metal load_arange supports Float32 and Int32");
        LFS_ASSERT_MSG(start.kind == ScalarKind::Float && step.kind == ScalarKind::Float,
                       "Metal load_arange start and step are float scalars");
        struct ArangeParams {
            uint64_t output_offset;
            float start;
            float step;
            uint32_t count;
            uint32_t padding;
        };
        const auto context = acquire_context();
        const auto output_at = context->locate(output);
        const auto pipeline = context->pipeline("arange", {{2, static_cast<uint32_t>(output.dtype)}});
        const ArangeParams params{
            .output_offset = output_at.offset,
            .start = start.value.float_value,
            .step = step.value.float_value,
            .count = checked_u32(count, "Metal arange count exceeds uint32"),
        };
        const std::array uses{output};
        context->dispatch(uses, {.pipeline = pipeline,
                                 .buffers = {output_at.address},
                                 .params = param_bytes(params),
                                 .grid = threads(count)});
    }

    void MetalBackendOps::where(const StorageRef condition, const StorageRef x, const StorageRef y,
                                const StorageRef output, const StridedLayout& condition_layout,
                                const StridedLayout& x_layout, const StridedLayout& y_layout,
                                const StridedLayout& output_layout, ExecContext) {
        LFS_FACADE_TRACE(where);
        LFS_ASSERT_MSG(condition.dtype == DataType::Bool && x.dtype == y.dtype && x.dtype == output.dtype,
                       "Metal where requires a Bool condition and matching value dtypes");
        if (output_layout.element_count == 0)
            return;
        struct WhereParams {
            uint64_t condition_offset, x_offset, y_offset, output_offset;
            std::array<uint32_t, MAX_TENSOR_RANK> condition_dims, x_dims, y_dims, output_dims;
            uint32_t condition_rank, x_rank, y_rank, output_rank, count, padding;
        };
        static_assert(sizeof(WhereParams) == 184);
        const auto context = acquire_context();
        const auto condition_at = context->locate(condition);
        const auto x_at = context->locate(x);
        const auto y_at = context->locate(y);
        const auto output_at = context->locate(output);
        const WhereParams params{
            .condition_offset = condition_at.offset,
            .x_offset = x_at.offset,
            .y_offset = y_at.offset,
            .output_offset = output_at.offset,
            .condition_dims = shader_dims(condition_layout),
            .x_dims = shader_dims(x_layout),
            .y_dims = shader_dims(y_layout),
            .output_dims = shader_dims(output_layout),
            .condition_rank = static_cast<uint32_t>(condition_layout.rank),
            .x_rank = static_cast<uint32_t>(x_layout.rank),
            .y_rank = static_cast<uint32_t>(y_layout.rank),
            .output_rank = static_cast<uint32_t>(output_layout.rank),
            .count = checked_u32(output_layout.element_count, "Metal where count exceeds uint32"),
            .padding = 0,
        };
        const uint32_t dtype = static_cast<uint32_t>(output.dtype);
        const auto pipeline = context->pipeline(
            "where_select", {{1, dtype}, {2, dtype}, {5, static_cast<uint32_t>(dtype_size(output.dtype))}});
        const std::array uses{condition, x, y, output};
        context->dispatch(uses, {.pipeline = pipeline,
                                 .buffers = {condition_at.address, x_at.address, y_at.address, output_at.address},
                                 .params = param_bytes(params),
                                 .grid = threads(output_layout.element_count)});
    }

    void MetalBackendOps::strided_copy(const StorageRef input, const StorageRef output,
                                       const StridedLayout& input_layout, ExecContext) {
        LFS_FACADE_TRACE(strided_copy);
        LFS_ASSERT_MSG(input.dtype == output.dtype, "Metal strided copy requires matching dtypes");
        encode_strided(input, output, input_layout, false, input.dtype, output.dtype);
    }

    void MetalBackendOps::strided_copy_immediate(const StorageRef input, const StorageRef output,
                                                 const StridedLayout& input_layout, ExecContext context) {
        LFS_FACADE_TRACE(strided_copy_immediate);
        strided_copy(input, output, input_layout, context);
    }

    // The CPU gathers a strided host tensor straight into the shared output.
    void MetalBackendOps::strided_upload(const StorageRef host_input, const StorageRef output,
                                         const StridedLayout& input_layout, ExecContext) {
        LFS_FACADE_TRACE(strided_upload);
        if (input_layout.element_count == 0)
            return;
        const auto context = acquire_context();
        context->wait(context->last_use(output));
        const size_t element_size = dtype_size(output.dtype);
        const std::byte* const source = host_bytes(host_input);
        std::byte* const destination = context->host(output);
        for (size_t index = 0; index < input_layout.element_count; ++index) {
            size_t remaining = index;
            size_t offset = 0;
            for (size_t axis = input_layout.rank; axis-- > 0;) {
                offset += remaining % input_layout.dims[axis] * input_layout.strides[axis];
                remaining /= input_layout.dims[axis];
            }
            std::memcpy(destination + index * element_size, source + offset * element_size, element_size);
        }
    }

    void MetalBackendOps::strided_scatter(const StorageRef input, const StorageRef output,
                                          const StridedLayout& output_layout, ExecContext) {
        LFS_FACADE_TRACE(strided_scatter);
        LFS_ASSERT_MSG(input.dtype == output.dtype, "Metal strided scatter requires matching dtypes");
        encode_strided(input, output, output_layout, true, input.dtype, output.dtype);
    }

    void MetalBackendOps::strided_scatter_immediate(const StorageRef input, const StorageRef output,
                                                    const StridedLayout& output_layout, ExecContext context) {
        LFS_FACADE_TRACE(strided_scatter_immediate);
        strided_scatter(input, output, output_layout, context);
    }

    void MetalBackendOps::strided_scatter_int32_to_float32(const StorageRef input, const StorageRef output,
                                                           const StridedLayout& output_layout, ExecContext) {
        LFS_FACADE_TRACE(strided_scatter_int32_to_float32);
        LFS_ASSERT_MSG(input.dtype == DataType::Int32 && output.dtype == DataType::Float32,
                       "Metal fused Int32 to Float32 scatter requires Int32 input and Float32 output");
        encode_strided(input, output, output_layout, true, DataType::Int32, DataType::Float32);
    }

    float MetalBackendOps::sum_scalar(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(sum_scalar);
        return scalar_reduce(metal::kReduceSum, input, count);
    }

    float MetalBackendOps::mean_scalar(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(mean_scalar);
        return scalar_reduce(metal::kReduceMean, input, count);
    }

    float MetalBackendOps::max_scalar(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(max_scalar);
        return scalar_reduce(metal::kReduceMax, input, count);
    }

    float MetalBackendOps::min_scalar(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(min_scalar);
        return scalar_reduce(metal::kReduceMin, input, count);
    }

    StorageRef MetalBackendOps::allocate(const size_t bytes, size_t, ExecContext) {
        LFS_FACADE_TRACE(service_allocate);
        return acquire_context()->allocate(bytes);
    }

    void MetalBackendOps::deallocate(const StorageRef storage, ExecContext) noexcept {
        if (const auto context = live_context())
            context->release(storage);
    }

    void MetalBackendOps::record_stream(StorageRef, ExecContext) {}

    void MetalBackendOps::release_stream(ExecContext) {}

    void MetalBackendOps::rehome_stream(StorageRef, ExecContext) {}

    void MetalBackendOps::trim() {
        if (const auto context = live_context())
            context->trim();
    }

    void MetalBackendOps::trim_if_reserved_unused_exceeds(const size_t threshold_bytes) {
        if (const auto context = live_context(); context && context->cached_bytes() > threshold_bytes)
            context->trim();
    }

    MemoryInfo MetalBackendOps::stats() {
        return acquire_context()->stats();
    }

    void MetalBackendOps::shutdown() {
        shutdown_metal_backend();
    }

    void MetalBackendOps::set_allocation_iteration(int) {}

    void MetalBackendOps::record_tensor_allocation(StorageRef, const StridedLayout&, size_t) {}

    // Storage is shared with the CPU, so host copies are plain memcpy once the
    // last batch that may use the memory completed.
    void MetalBackendOps::copy_host_to_device(const CopyRequest& request) {
        LFS_FACADE_TRACE(service_copy_host_to_device);
        if (request.bytes == 0)
            return;
        const auto context = acquire_context();
        context->wait(context->last_use(request.dst));
        std::memcpy(context->host(request.dst), host_bytes(request.src), request.bytes);
    }

    void MetalBackendOps::copy_device_to_host(const CopyRequest& request) {
        LFS_FACADE_TRACE(service_copy_device_to_host);
        if (request.bytes == 0)
            return;
        const auto context = acquire_context();
        context->wait(context->last_use(request.src));
        std::memcpy(host_bytes(request.dst), context->host(request.src), request.bytes);
    }

    void MetalBackendOps::copy_device_to_device(const CopyRequest& request) {
        LFS_FACADE_TRACE(service_copy_device_to_device);
        const auto context = acquire_context();
        encode_copy(*context, request.src, request.dst, request.bytes);
        if (request.synchronous)
            context->wait(context->pending(request.dst));
    }

    void MetalBackendOps::memset(const FillRequest& request) {
        LFS_FACADE_TRACE(service_memset);
        const auto context = acquire_context();
        encode_fill(*context, request.dst, request.bytes, request.value, 1);
        if (request.synchronous)
            context->wait(context->pending(request.dst));
    }

    std::unique_ptr<ReadbackBuffer> MetalBackendOps::create_readback_buffer() {
        return std::make_unique<MetalReadbackBuffer>();
    }

    void MetalBackendOps::synchronize_stream(ExecContext) {
        LFS_FACADE_TRACE(service_synchronize_stream);
        if (const auto context = live_context())
            context->wait_idle();
    }

    void MetalBackendOps::synchronize_device() {
        synchronize_stream({});
    }

    void MetalBackendOps::device_barrier() {
        synchronize_stream({});
    }

    void MetalBackendOps::wait_for(const SyncToken token) {
        LFS_ASSERT_MSG(token.backend == GpuBackend::Metal, "Metal sync service received a non-Metal token");
        if (const auto context = live_context())
            context->wait(token.value);
    }

    SyncToken MetalBackendOps::bridge(ExecContext, ExecContext) {
        const auto context = live_context();
        return SyncToken{.backend = GpuBackend::Metal, .value = context ? context->flush() : 0, .native = 0};
    }

    PointerClass MetalBackendOps::classify_pointer(const void* const pointer) {
        if (const auto context = live_context(); pointer && context && context->owns(pointer))
            return PointerClass::Device;
        return PointerClass::Unknown;
    }

    bool MetalBackendOps::stream_is_capturing(ExecContext) {
        return false;
    }

} // namespace lfs::core::internal
