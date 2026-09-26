/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../../internal/point_filter.hpp"
#include "../../internal/rad_ops.hpp"
#include "../export_pipeline.hpp"
#include "../facade_trace.hpp"
#include "../readback_buffer.hpp"
#include "../scalar_operand.hpp"
#include "../tensor_vulkan_interop.hpp"
#include "metal_backend_ops.hpp"
#include "metal_context.hpp"

#include "core/assert.hpp"
#include "core/detail/fused_pointwise.hpp"
#include "core/logger.hpp"
#include "core/tensor_environment.hpp"
#include "core/tensor_image.hpp"
#include "core/tensor_labels.hpp"
#include "core/tensor_ppisp.hpp"
#include "core/tensor_spatial.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#include <vector>

namespace lfs::core::internal {

    namespace {
        using metal::acquire_context;
        using metal::Context;
        using metal::kThreadgroupWidth;
        using metal::live_context;
        using metal::param_bytes;

        // Matrix-unit GEMMs take 64x64 tiles when those still give every GPU
        // core several threadgroups, and 32x32 tiles otherwise.
        size_t gemm_tile(const size_t m, const size_t n, const size_t batch) {
            constexpr size_t kWideTiles = 256;
            return ((m + 63) / 64) * ((n + 63) / 64) * batch >= kWideTiles ? 64 : 32;
        }

        MTLSize threads(const size_t count) {
            return MTLSizeMake(count, 1, 1);
        }

        uint32_t checked_u32(const size_t value, const char* const description) {
            LFS_ASSERT_MSG(value <= std::numeric_limits<uint32_t>::max(), description);
            return static_cast<uint32_t>(value);
        }

        // Scratch storage of one operation. It returns to the context even when
        // the operation throws; queued work that still uses it runs first.
        struct API_AVAILABLE(macos(26.0)) Scratch {
            Scratch(Context& owner, const size_t bytes) : context(owner), storage(owner.allocate(bytes)) {}
            ~Scratch() { context.release(storage); }
            Scratch(const Scratch&) = delete;
            Scratch& operator=(const Scratch&) = delete;

            Context& context;
            StorageRef storage;
        };

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

        // A fused chain specializes its kernel on the op kinds (function
        // constants 8 to 11); tensor operands are read through their addresses.
        struct FusedChain {
            uint32_t length = 0;
            std::array<uint32_t, 3> kinds{};
            std::array<ChainOp, tensor_ops::FUSED_POINTWISE_MAX_OPS> ops{};
            // Every tensor operand allows four-wide access.
            bool aligned = true;
        };

        API_AVAILABLE(macos(26.0))
        FusedChain fused_chain(Context& context, const tensor_ops::FusedPointwiseOpChain& chain) {
            LFS_ASSERT_MSG(chain.num_ops > 0 && chain.num_ops <= tensor_ops::FUSED_POINTWISE_MAX_OPS,
                           "Metal fused chains hold 1 to 16 operations");
            FusedChain result{.length = static_cast<uint32_t>(chain.num_ops)};
            for (int i = 0; i < chain.num_ops; ++i) {
                const auto& op = chain.ops[i];
                result.kinds[i / 6] |= static_cast<uint32_t>(op.kind & 31u) << (5 * (i % 6));
                result.ops[i].scalar = op.scalar;
                if (op.kind < 4 || op.kind > 7)
                    continue;
                const auto rhs_at = context.locate(op.rhs);
                result.ops[i].rhs_address = rhs_at.address + rhs_at.offset;
                result.aligned = result.aligned && result.ops[i].rhs_address % 16 == 0;
            }
            return result;
        }

        struct FillParams {
            uint64_t output_offset;
            uint64_t pattern;
            uint64_t count;
            std::array<uint32_t, MAX_TENSOR_RANK> dims{};
            std::array<uint32_t, MAX_TENSOR_RANK> strides{};
            uint32_t rank = 0;
            uint32_t padding = 0;
        };
        static_assert(sizeof(FillParams) == 96);

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
            const auto pipeline = context.pipeline("fill", {{0, 0}, {5, static_cast<uint32_t>(element_size)}});
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

        // The same walk with size-1 axes dropped and each axis merged into the
        // one outside it when the two are contiguous together.
        StridedLayout merge_axes(const StridedLayout& layout) {
            StridedLayout merged{.element_count = layout.element_count};
            for (size_t axis = 0; axis < layout.rank; ++axis) {
                if (layout.dims[axis] == 1)
                    continue;
                if (merged.rank > 0 &&
                    merged.strides[merged.rank - 1] == layout.strides[axis] * layout.dims[axis]) {
                    merged.dims[merged.rank - 1] *= layout.dims[axis];
                    merged.strides[merged.rank - 1] = layout.strides[axis];
                    continue;
                }
                merged.dims[merged.rank] = layout.dims[axis];
                merged.strides[merged.rank++] = layout.strides[axis];
            }
            if (merged.rank == 0) {
                merged.rank = 1;
                merged.dims[0] = 1;
                merged.strides[0] = 1;
            }
            return merged;
        }

        // Bytes each thread of a strided copy moves: up to 16 when the inner
        // axis is contiguous and every stride and offset keeps them aligned.
        size_t strided_chunk_bytes(const StridedLayout& layout, const size_t element, const size_t input_offset,
                                   const size_t output_offset) {
            if (layout.strides[layout.rank - 1] != 1)
                return element;
            for (const size_t bytes : {size_t{16}, size_t{8}, size_t{4}}) {
                if (bytes <= element)
                    break;
                bool aligned = layout.dims[layout.rank - 1] * element % bytes == 0 && input_offset % bytes == 0 &&
                               output_offset % bytes == 0;
                for (size_t axis = 0; axis + 1 < layout.rank; ++axis)
                    aligned = aligned && layout.strides[axis] * element % bytes == 0;
                if (aligned)
                    return bytes;
            }
            return element;
        }

        // Gathers a strided input into contiguous output or, for a scatter, the
        // reverse; the layout describes the strided side.
        API_AVAILABLE(macos(26.0))
        void encode_strided(const StorageRef input, const StorageRef output, const StridedLayout& source_layout,
                            const bool scatter, const DataType input_dtype, const DataType output_dtype) {
            if (source_layout.element_count == 0)
                return;
            LFS_ASSERT_MSG(source_layout.rank <= MAX_TENSOR_RANK,
                           "Metal strided layout rank exceeds MAX_TENSOR_RANK");
            StridedLayout layout = merge_axes(source_layout);
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
            // Same-dtype copies move whole aligned chunks of the inner axis.
            const size_t element = dtype_size(output_dtype);
            const size_t chunk = input_dtype == output_dtype
                                     ? strided_chunk_bytes(layout, element, input_at.offset, output_at.offset)
                                     : element;
            if (chunk != element) {
                const size_t per_chunk = chunk / element;
                layout.dims[layout.rank - 1] /= per_chunk;
                for (size_t axis = 0; axis + 1 < layout.rank; ++axis)
                    layout.strides[axis] /= per_chunk;
                layout.element_count /= per_chunk;
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
                                 {5, static_cast<uint32_t>(chunk)},
                                 {7, scatter ? 1u : 0u}});
            const std::array uses{input, output};
            context->dispatch(uses, {.pipeline = pipeline,
                                     .buffers = {input_at.address, output_at.address},
                                     .params = param_bytes(params),
                                     .grid = threads(layout.element_count)});
        }

        std::array<uint32_t, MAX_TENSOR_RANK> shader_values(const std::array<size_t, MAX_TENSOR_RANK>& values,
                                                            const size_t rank, const char* const description) {
            std::array<uint32_t, MAX_TENSOR_RANK> result{};
            for (size_t axis = 0; axis < rank; ++axis)
                result[axis] = checked_u32(values[axis], description);
            return result;
        }

        std::array<uint32_t, MAX_TENSOR_RANK> shader_dims(const StridedLayout& layout) {
            return shader_values(layout.dims, layout.rank, "Metal layout dimension exceeds uint32");
        }

        // Mirrors BroadcastAxes in kernels.metal.
        struct BroadcastAxes {
            std::array<uint32_t, MAX_TENSOR_RANK> dims{};
            std::array<std::array<uint32_t, 4>, MAX_TENSOR_RANK> strides{};
            uint32_t rank = 0;
            std::array<uint32_t, 3> padding{};
        };
        static_assert(sizeof(BroadcastAxes) == 176);

        // Dense operands, right-aligned to an output of fewer than 2^32 elements.
        // Adjacent axes that every operand broadcasts alike merge, so most
        // launches divide once or twice per element for all operands together.
        BroadcastAxes broadcast_axes(const StridedLayout& output,
                                     const std::initializer_list<const StridedLayout*> operands) {
            LFS_ASSERT_MSG(operands.size() <= 4, "Metal broadcast takes at most four operands");
            std::array<size_t, 4> dense{1, 1, 1, 1};
            BroadcastAxes inner_first;
            uint32_t merging = 0; // which operands broadcast along the axis being merged
            for (size_t axis = output.rank; axis-- > 0;) {
                const size_t extent = output.dims[axis];
                if (extent == 1)
                    continue;
                uint32_t broadcasting = 0;
                std::array<uint32_t, 4> strides{};
                size_t operand = 0;
                for (const StridedLayout* const layout : operands) {
                    const size_t lead = output.rank - layout->rank;
                    const size_t operand_extent = axis >= lead ? layout->dims[axis - lead] : 1;
                    LFS_ASSERT_MSG(operand_extent == 1 || operand_extent == extent,
                                   "Metal broadcast operand does not match the output");
                    if (operand_extent == 1)
                        broadcasting |= 1u << operand;
                    else
                        strides[operand] = static_cast<uint32_t>(dense[operand]);
                    dense[operand++] *= operand_extent;
                }
                if (inner_first.rank > 0 && broadcasting == merging) {
                    inner_first.dims[inner_first.rank - 1] *= static_cast<uint32_t>(extent);
                    continue;
                }
                inner_first.dims[inner_first.rank] = static_cast<uint32_t>(extent);
                inner_first.strides[inner_first.rank++] = strides;
                merging = broadcasting;
            }
            BroadcastAxes axes{.dims = {1}, .rank = std::max(inner_first.rank, 1u)};
            for (uint32_t axis = 0; axis < inner_first.rank; ++axis) {
                axes.dims[axis] = inner_first.dims[inner_first.rank - 1 - axis];
                axes.strides[axis] = inner_first.strides[inner_first.rank - 1 - axis];
            }
            return axes;
        }

        struct CatPadParams {
            uint64_t input_offset;
            uint64_t output_offset;
            std::array<uint32_t, MAX_TENSOR_RANK> input_dims;
            std::array<uint32_t, MAX_TENSOR_RANK> input_strides;
            std::array<uint32_t, MAX_TENSOR_RANK> output_strides;
            std::array<uint32_t, MAX_TENSOR_RANK> pad_before;
            uint32_t count;
            uint32_t rank;
            uint32_t input_block;
            uint32_t output_block;
            uint32_t output_column;
            uint32_t padding;
        };
        static_assert(sizeof(CatPadParams) == 16 + 4 * 32 + 24);

        // Copies a contiguous input of count elements, input_block per row, into
        // columns [output_column, output_column + input_block) of output rows
        // of output_block elements.
        API_AVAILABLE(macos(26.0))
        void encode_cat_input(const StorageRef input, const StorageRef output, const size_t input_block,
                              const size_t output_block, const size_t output_column, const size_t count) {
            if (count == 0)
                return;
            const auto context = acquire_context();
            const auto input_at = context->locate(input);
            const auto output_at = context->locate(output);
            const CatPadParams params{
                .input_offset = input_at.offset,
                .output_offset = output_at.offset,
                .count = checked_u32(count, "Metal cat count exceeds uint32"),
                .input_block = checked_u32(input_block, "Metal cat input block exceeds uint32"),
                .output_block = checked_u32(output_block, "Metal cat output block exceeds uint32"),
                .output_column = checked_u32(output_column, "Metal cat output offset exceeds uint32"),
            };
            const std::array uses{input, output};
            const uint32_t dtype = static_cast<uint32_t>(output.dtype);
            context->dispatch(uses, {.pipeline = context->pipeline(
                                         "cat_pad", {{0, 0}, {1, dtype}, {2, dtype},
                                                     {5, static_cast<uint32_t>(dtype_size(output.dtype))}}),
                                     .buffers = {input_at.address, output_at.address},
                                     .params = param_bytes(params),
                                     .grid = threads(count)});
        }

        API_AVAILABLE(macos(26.0))
        void encode_clamp(const StorageRef input, const StorageRef output, const ScalarOperand minimum,
                          const ScalarOperand maximum, const size_t count) {
            if (count == 0)
                return;
            const bool integer = input.dtype == DataType::Int32;
            LFS_ASSERT_MSG(input.dtype == output.dtype &&
                               ((integer && minimum.kind == ScalarKind::Int32 && maximum.kind == ScalarKind::Int32) ||
                                (input.dtype == DataType::Float32 && minimum.kind == ScalarKind::Float &&
                                 maximum.kind == ScalarKind::Float)),
                           "Metal clamp requires matching Float32 or Int32 operands");
            struct ClampParams {
                uint64_t input_offset;
                uint64_t output_offset;
                float float_minimum;
                float float_maximum;
                int32_t int_minimum;
                int32_t int_maximum;
                uint32_t count;
                uint32_t padding;
            };
            const auto context = acquire_context();
            const auto input_at = context->locate(input);
            const auto output_at = context->locate(output);
            const ClampParams params{
                .input_offset = input_at.offset,
                .output_offset = output_at.offset,
                .float_minimum = integer ? 0.0f : minimum.value.float_value,
                .float_maximum = integer ? 0.0f : maximum.value.float_value,
                .int_minimum = integer ? minimum.value.int32_value : 0,
                .int_maximum = integer ? maximum.value.int32_value : 0,
                .count = checked_u32(count, "Metal clamp count exceeds uint32"),
            };
            const std::array uses{input, output};
            context->dispatch(uses, {.pipeline = context->pipeline("clamp_values", {{1, static_cast<uint32_t>(input.dtype)}}),
                                     .buffers = {input_at.address, output_at.address},
                                     .params = param_bytes(params),
                                     .grid = threads(count)});
        }

        // Stages of the reduce kernel, as in vk_ops_reduce.cpp.
        constexpr uint32_t kPartialMode = 0, kSegmentedMode = 2, kStridedMode = 3;
        constexpr size_t kMaxPartials = 1024;
        constexpr size_t kElementsPerPartial = kThreadgroupWidth * 8;
        constexpr size_t kSingleGroupFullReduce = 4096;
        constexpr size_t kSegmentedThreshold = 64;
        constexpr size_t kSplitOutputLimit = 4096;
        constexpr size_t kSplitReduceThreshold = 1024;
        constexpr size_t kMaxSplits = 64;

        struct ReduceParams {
            uint64_t input_offset;
            uint64_t output_offset;
            uint32_t outer;
            uint32_t reduce;
            uint32_t inner;
            uint32_t count;
            uint32_t split_chunk;
            float mean_scale;
            std::array<ChainOp, tensor_ops::FUSED_POINTWISE_MAX_OPS> ops;
        };
        static_assert(sizeof(ReduceParams) == 40 + 16 * 16);

        bool logical_op(const ReduceOp op) {
            return op == ReduceOp::Any || op == ReduceOp::All;
        }

        uint32_t element_code(const DataType dtype) {
            LFS_ASSERT_MSG(dtype == DataType::Float32 || dtype == DataType::Int32 || dtype == DataType::Int64 ||
                               dtype == DataType::UInt8 || dtype == DataType::Bool,
                           "Metal reduction received an unsupported dtype");
            return static_cast<uint32_t>(dtype);
        }

        // Partials keep the accumulator: (sum, compensation) pairs for float
        // sums, floats for the other float ops, int64 for everything else.
        uint32_t partial_code(const DataType input, const ReduceOp op) {
            if (input != DataType::Float32 || logical_op(op))
                return element_code(DataType::Int64);
            return op == ReduceOp::Sum || op == ReduceOp::Mean ? metal::kPairDType : element_code(DataType::Float32);
        }

        float mean_scale_for(const ReduceOp op, const size_t reduce) {
            return op == ReduceOp::Mean ? 1.0f / static_cast<float>(reduce) : 1.0f;
        }

        // Operands of a reduction: its input and, for a fused transform, the
        // chain and the tensors the chain reads.
        struct ReduceSource {
            StorageRef input;
            uint32_t code;
            const FusedChain* chain = nullptr;
            std::span<const StorageRef> operands;
        };

        struct ReduceStage {
            ReduceOp op;
            uint32_t mode;
            uint32_t output_code;
            StorageRef output;
            MTLSize groups;
            ReduceParams params{};
        };

        API_AVAILABLE(macos(26.0))
        void encode_reduce_stage(Context& context, const ReduceSource& source, ReduceStage stage) {
            LFS_ASSERT_MSG(static_cast<uint8_t>(stage.op) <= static_cast<uint8_t>(ReduceOp::All),
                           "Metal reduction received an unsupported operation");
            const auto input_at = context.locate(source.input);
            const auto output_at = context.locate(stage.output);
            stage.params.input_offset = input_at.offset;
            stage.params.output_offset = output_at.offset;
            const FusedChain none{};
            const FusedChain& chain = source.chain != nullptr ? *source.chain : none;
            stage.params.ops = chain.ops;
            const auto pipeline = context.pipeline(
                "reduce", {{1, source.code}, {2, stage.output_code}, {6, static_cast<uint32_t>(stage.op)},
                           {8, chain.length}, {9, chain.kinds[0]}, {10, chain.kinds[1]}, {11, chain.kinds[2]},
                           {16, stage.mode}});
            std::vector<StorageRef> uses{source.input, stage.output};
            uses.insert(uses.end(), source.operands.begin(), source.operands.end());
            context.dispatch(uses, {.pipeline = pipeline,
                                    .buffers = {input_at.address, output_at.address},
                                    .params = param_bytes(stage.params),
                                    .grid = stage.groups,
                                    .group_size = MTLSizeMake(kThreadgroupWidth, 1, 1)});
        }

        // Reduces count contiguous elements into output[0]; large inputs go
        // through per-threadgroup partials, so the result is order-deterministic.
        API_AVAILABLE(macos(26.0))
        void reduce_full(Context& context, const ReduceOp op, const ReduceSource& source, const size_t count,
                         const StorageRef output, const DataType output_dtype) {
            if (count == 0)
                return;
            const uint32_t count32 = checked_u32(count, "Metal reduction count exceeds uint32");
            if (count <= kSingleGroupFullReduce) {
                encode_reduce_stage(context, source,
                                    {.op = op, .mode = kSegmentedMode, .output_code = element_code(output_dtype),
                                     .output = output, .groups = MTLSizeMake(1, 1, 1),
                                     .params = {.outer = 1, .reduce = count32, .inner = 1,
                                                .mean_scale = mean_scale_for(op, count)}});
                return;
            }
            const size_t groups = std::min(kMaxPartials, (count + kElementsPerPartial - 1) / kElementsPerPartial);
            const uint32_t partial = partial_code(source.input.dtype, op);
            const Scratch partials(context, groups * sizeof(int64_t));
            encode_reduce_stage(context, source,
                                {.op = op, .mode = kPartialMode, .output_code = partial, .output = partials.storage,
                                 .groups = MTLSizeMake(groups, 1, 1), .params = {.count = count32, .mean_scale = 1.0f}});
            encode_reduce_stage(context, {.input = partials.storage, .code = partial},
                                {.op = op, .mode = kSegmentedMode, .output_code = element_code(output_dtype),
                                 .output = output, .groups = MTLSizeMake(1, 1, 1),
                                 .params = {.outer = 1, .reduce = static_cast<uint32_t>(groups), .inner = 1,
                                            .mean_scale = mean_scale_for(op, count)}});
        }

        // Reduces the middle extent of an (outer, reduce, inner) view into
        // outer * inner outputs.
        API_AVAILABLE(macos(26.0))
        void reduce_axes(Context& context, const ReduceOp op, const ReduceSource& source, const StorageRef output,
                         const DataType output_dtype, const size_t outer, const size_t reduce, const size_t inner) {
            if (outer == 0 || reduce == 0 || inner == 0)
                return;
            if (outer == 1 && inner == 1) {
                reduce_full(context, op, source, reduce, output, output_dtype);
                return;
            }
            const uint32_t outer32 = checked_u32(outer, "Metal reduction outer size exceeds uint32");
            const uint32_t reduce32 = checked_u32(reduce, "Metal reduction size exceeds uint32");
            const uint32_t inner32 = checked_u32(inner, "Metal reduction inner size exceeds uint32");
            const float mean_scale = mean_scale_for(op, reduce);
            if (inner == 1 && reduce >= kSegmentedThreshold) {
                encode_reduce_stage(context, source,
                                    {.op = op, .mode = kSegmentedMode, .output_code = element_code(output_dtype),
                                     .output = output, .groups = MTLSizeMake(outer, 1, 1),
                                     .params = {.outer = outer32, .reduce = reduce32, .inner = 1, .mean_scale = mean_scale}});
                return;
            }
            const size_t outputs = outer * inner;
            const size_t output_groups = (checked_u32(outputs, "Metal reduction output count exceeds uint32") +
                                          kThreadgroupWidth - 1) / kThreadgroupWidth;
            // Few outputs over a long reduce extent starve the GPU of threads, so
            // the range splits across grid rows into partials that a second
            // strided pass folds.
            const size_t splits = outputs < kSplitOutputLimit && reduce >= kSplitReduceThreshold
                                      ? std::min(kMaxSplits, (reduce + kThreadgroupWidth - 1) / kThreadgroupWidth)
                                      : 1;
            if (splits == 1) {
                encode_reduce_stage(context, source,
                                    {.op = op, .mode = kStridedMode, .output_code = element_code(output_dtype),
                                     .output = output, .groups = MTLSizeMake(output_groups, 1, 1),
                                     .params = {.outer = outer32, .reduce = reduce32, .inner = inner32,
                                                .split_chunk = reduce32, .mean_scale = mean_scale}});
                return;
            }
            const uint32_t partial = partial_code(source.input.dtype, op);
            const Scratch partials(context, splits * outputs * sizeof(int64_t));
            encode_reduce_stage(context, source,
                                {.op = op, .mode = kStridedMode, .output_code = partial, .output = partials.storage,
                                 .groups = MTLSizeMake(output_groups, splits, 1),
                                 .params = {.outer = outer32, .reduce = reduce32, .inner = inner32,
                                            .split_chunk = static_cast<uint32_t>((reduce + splits - 1) / splits),
                                            .mean_scale = 1.0f}});
            encode_reduce_stage(context, {.input = partials.storage, .code = partial},
                                {.op = op, .mode = kStridedMode, .output_code = element_code(output_dtype),
                                 .output = output, .groups = MTLSizeMake(output_groups, 1, 1),
                                 .params = {.outer = 1, .reduce = static_cast<uint32_t>(splits),
                                            .inner = static_cast<uint32_t>(outputs),
                                            .split_chunk = static_cast<uint32_t>(splits), .mean_scale = mean_scale}});
        }

        // Axes that are not one contiguous run: the kept axes are permute-copied
        // to the front, so the contiguous path, with its splits, does the work.
        API_AVAILABLE(macos(26.0))
        void reduce_general(Context& context, const ReduceOp op, const StorageRef input, const StorageRef output,
                            const DataType output_dtype, const StridedLayout& layout, const uint32_t reduced_mask) {
            size_t outputs = 1, reduce = 1, stride = 1;
            std::array<size_t, MAX_TENSOR_RANK> strides{};
            for (size_t axis = layout.rank; axis-- > 0;) {
                strides[axis] = stride;
                stride *= layout.dims[axis];
                ((reduced_mask >> axis) & 1u ? reduce : outputs) *= layout.dims[axis];
            }
            if (outputs == 0 || reduce == 0)
                return;
            StridedLayout permuted{.rank = layout.rank, .element_count = layout.element_count};
            size_t position = 0;
            for (const bool reduced_pass : {false, true}) {
                for (size_t axis = 0; axis < layout.rank; ++axis) {
                    if ((((reduced_mask >> axis) & 1u) != 0u) == reduced_pass) {
                        permuted.dims[position] = layout.dims[axis];
                        permuted.strides[position] = strides[axis];
                        ++position;
                    }
                }
            }
            Scratch scratch(context, layout.element_count * dtype_size(input.dtype));
            scratch.storage.dtype = input.dtype;
            encode_strided(input, scratch.storage, permuted, false, input.dtype, input.dtype);
            reduce_axes(context, op, {.input = scratch.storage, .code = element_code(input.dtype)}, output, output_dtype,
                        outputs, reduce, 1);
        }

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
        float scalar_reduce(const ReduceOp op, const StorageRef input, const size_t count) {
            LFS_ASSERT_MSG(input.dtype == DataType::Float32,
                           "Metal scalar reduction requires Float32 input");
            const bool extreme = op == ReduceOp::Max || op == ReduceOp::Min;
            if (count == 0) {
                return op == ReduceOp::Max   ? -std::numeric_limits<float>::infinity()
                       : op == ReduceOp::Min ? std::numeric_limits<float>::infinity()
                                             : 0.0f;
            }
            const auto context = acquire_context();
            constexpr size_t kElementsPerGroup = kThreadgroupWidth * 4;
            const size_t groups = std::clamp<size_t>((count + kElementsPerGroup - 1) / kElementsPerGroup, 1, kMaxPartials);
            const Scratch partials(*context, groups * 2 * sizeof(float));
            encode_reduce_stage(*context, {.input = input, .code = element_code(DataType::Float32)},
                                {.op = op, .mode = kPartialMode, .output_code = metal::kPairDType, .output = partials.storage,
                                 .groups = MTLSizeMake(groups, 1, 1),
                                 .params = {.count = checked_u32(count, "Metal reduction count exceeds uint32"),
                                            .mean_scale = 1.0f}});
            context->wait(context->pending(partials.storage));
            const auto* const pairs = reinterpret_cast<const float*>(context->host(partials.storage));
            float accumulator = pairs[0], compensation = extreme ? 0.0f : pairs[1];
            for (size_t group = 1; group < groups; ++group) {
                const float value = pairs[2 * group];
                if (extreme) {
                    accumulator = ieee_extreme(accumulator, value, op == ReduceOp::Max);
                    continue;
                }
                const float total = accumulator + value;
                const float carried = total - accumulator;
                compensation += (accumulator - (total - carried)) + (value - carried) + pairs[2 * group + 1];
                accumulator = total;
            }
            if (extreme)
                return accumulator;
            const float sum = std::isfinite(compensation) ? accumulator + compensation : accumulator;
            return op == ReduceOp::Mean ? sum * (1.0f / static_cast<float>(count)) : sum;
        }

        // count_matches kinds: 0 nonzero bytes, 1 nonzero floats, 2 NaN
        // present, 3 infinity present.
        API_AVAILABLE(macos(26.0))
        uint32_t count_matches(const uint32_t kind, const StorageRef input, const size_t count) {
            if (count == 0)
                return 0;
            struct CountParams {
                uint64_t input_offset;
                uint32_t count;
                uint32_t padding;
            };
            const auto context = acquire_context();
            const Scratch result(*context, sizeof(uint32_t));
            encode_fill(*context, result.storage, sizeof(uint32_t), 0, sizeof(uint32_t));
            const auto input_at = context->locate(input);
            const auto result_at = context->locate(result.storage);
            const CountParams params{.input_offset = input_at.offset,
                                     .count = checked_u32(count, "Metal count exceeds uint32")};
            const std::array uses{input, result.storage};
            context->dispatch(uses, {.pipeline = context->pipeline("count_matches", {{0, kind}}),
                                     .buffers = {input_at.address, result_at.address + result_at.offset},
                                     .params = param_bytes(params),
                                     .grid = MTLSizeMake(std::min<size_t>(256, (count + kThreadgroupWidth - 1) / kThreadgroupWidth), 1, 1),
                                     .group_size = MTLSizeMake(kThreadgroupWidth, 1, 1)});
            context->wait(context->pending(result.storage));
            uint32_t value = 0;
            std::memcpy(&value, context->host(result.storage), sizeof(value));
            return value;
        }

        // In-place inclusive scan along the middle extent of an (outer, size,
        // inner) view. Long lines scan their blocks independently, scan the
        // block totals the same way, then add them back.
        API_AVAILABLE(macos(26.0))
        void encode_scan(Context& context, const StorageRef data, const size_t outer, const size_t size,
                         const size_t inner) {
            struct ScanParams {
                uint64_t data_offset;
                uint64_t totals_offset;
                uint32_t outer;
                uint32_t dim;
                uint32_t inner;
                uint32_t lines;
            };
            const size_t lines = outer * inner;
            const size_t blocks = (size + kThreadgroupWidth - 1) / kThreadgroupWidth;
            const uint32_t pass = size <= 32 ? 0 : blocks == 1 ? 1 : 2;
            std::optional<Scratch> block_totals;
            StorageRef totals = data;
            if (pass == 2) {
                totals = block_totals.emplace(context, lines * blocks * sizeof(uint32_t)).storage;
                totals.dtype = data.dtype;
            }
            const auto data_at = context.locate(data);
            const auto totals_at = context.locate(totals);
            const ScanParams params{
                .data_offset = data_at.offset,
                .totals_offset = totals_at.offset,
                .outer = checked_u32(outer, "Metal cumsum outer size exceeds uint32"),
                .dim = checked_u32(size, "Metal cumsum size exceeds uint32"),
                .inner = checked_u32(inner, "Metal cumsum inner size exceeds uint32"),
                .lines = checked_u32(lines, "Metal cumsum line count exceeds uint32"),
            };
            checked_u32(lines * size, "Metal cumsum element count exceeds uint32");
            const auto dispatch = [&](const uint32_t scan_pass, const MTLSize groups) {
                const std::array uses{data, totals};
                context.dispatch(uses, {.pipeline = context.pipeline("scan", {{0, scan_pass}, {1, static_cast<uint32_t>(data.dtype)}}),
                                        .buffers = {data_at.address, totals_at.address},
                                        .params = param_bytes(params),
                                        .grid = groups,
                                        .group_size = MTLSizeMake(kThreadgroupWidth, 1, 1)});
            };
            dispatch(pass, pass == 0 ? MTLSizeMake((lines + kThreadgroupWidth - 1) / kThreadgroupWidth, 1, 1)
                                     : MTLSizeMake(blocks, lines, 1));
            if (pass == 2) {
                encode_scan(context, totals, lines, blocks, 1);
                dispatch(3, MTLSizeMake((lines * size + kThreadgroupWidth - 1) / kThreadgroupWidth, 1, 1));
            }
        }

        // nn kernel kinds.
        constexpr uint32_t kMaxPool = 0, kAdaptiveAvgPool = 1, kBiasAdd = 2, kBiasRelu = 3, kRelu = 4;

        struct NnParams {
            uint64_t input_offset;
            uint64_t bias_offset;
            uint64_t output_offset;
            uint32_t total;
            uint32_t channels;
            uint32_t input_height;
            uint32_t input_width;
            uint32_t output_height;
            uint32_t output_width;
            uint32_t window;
            uint32_t stride;
            int32_t padding;
            uint32_t spatial;
        };
        static_assert(sizeof(NnParams) == 64);

        uint32_t checked_dimension(const int value, const char* const description) {
            LFS_ASSERT_MSG(value >= 0, description);
            return static_cast<uint32_t>(value);
        }

        API_AVAILABLE(macos(26.0))
        void encode_nn(const uint32_t kind, const StorageRef input, const StorageRef* const bias,
                       const StorageRef output, NnParams params) {
            if (params.total == 0)
                return;
            LFS_ASSERT_MSG(input.dtype == DataType::Float32 && output.dtype == DataType::Float32,
                           "Metal neural-network kernels require Float32");
            const auto context = acquire_context();
            const auto input_at = context->locate(input);
            const auto output_at = context->locate(output);
            const auto bias_at = bias != nullptr ? context->locate(*bias) : input_at;
            params.input_offset = input_at.offset;
            params.bias_offset = bias_at.offset;
            params.output_offset = output_at.offset;
            const std::array uses{input, output, bias != nullptr ? *bias : input};
            context->dispatch(uses, {.pipeline = context->pipeline("nn", {{0, kind}}),
                                     .buffers = {input_at.address, bias_at.address, output_at.address},
                                     .params = param_bytes(params),
                                     .grid = threads(params.total)});
        }

        NnParams pool_params(const PoolProgram& program) {
            const size_t total = static_cast<size_t>(checked_dimension(program.batch, "pool batch")) *
                                 checked_dimension(program.channels, "pool channels") *
                                 checked_dimension(program.output_height, "pool output height") *
                                 checked_dimension(program.output_width, "pool output width");
            return NnParams{
                .total = checked_u32(total, "Metal pooling output count exceeds uint32"),
                .channels = static_cast<uint32_t>(program.channels),
                .input_height = checked_dimension(program.input_height, "pool input height"),
                .input_width = checked_dimension(program.input_width, "pool input width"),
                .output_height = static_cast<uint32_t>(program.output_height),
                .output_width = static_cast<uint32_t>(program.output_width),
                .window = checked_dimension(program.kernel_size, "pool kernel size"),
                .stride = checked_dimension(program.stride, "pool stride"),
                .padding = program.padding,
            };
        }

        NnParams bias_params(const int count, const int channels, const int spatial_size) {
            LFS_ASSERT_MSG(channels > 0 && spatial_size > 0, "Metal bias kernel requires positive layout sizes");
            return NnParams{
                .total = checked_dimension(count, "bias element count"),
                .channels = static_cast<uint32_t>(channels),
                .spatial = static_cast<uint32_t>(spatial_size),
            };
        }

        struct GemmParams {
            uint64_t lhs_offset;
            uint64_t rhs_offset;
            uint64_t output_offset;
            uint64_t bias_offset;
            uint64_t lhs_stride;
            uint64_t rhs_stride;
            uint64_t output_stride;
            uint32_t m;
            uint32_t n;
            uint32_t k;
            uint32_t padding;
        };
        static_assert(sizeof(GemmParams) == 72);

        // The matmul tiles index with int32.
        uint32_t checked_extent(const size_t value, const char* const description) {
            LFS_ASSERT_MSG(value <= static_cast<size_t>(std::numeric_limits<int32_t>::max()), description);
            return static_cast<uint32_t>(value);
        }

        // C[m][n] = A[m][k] * B, with B stored as [k][n] or, transposed, as
        // [n][k]; batches are packed. A bias applies max(value + bias[row], 0).
        API_AVAILABLE(macos(26.0))
        void encode_gemm(const StorageRef lhs, const StorageRef rhs, const StorageRef* const bias,
                         const StorageRef output, const GemmProgram& program, const bool transpose_rhs) {
            if (program.m == 0 || program.n == 0 || program.batch == 0)
                return;
            LFS_ASSERT_MSG(lhs.dtype == DataType::Float32 && rhs.dtype == DataType::Float32 &&
                               output.dtype == DataType::Float32 &&
                               (bias == nullptr || bias->dtype == DataType::Float32),
                           "Metal GEMM requires Float32 operands");
            const auto context = acquire_context();
            if (program.k == 0) {
                // An empty product is zero; the bias epilogue still applies.
                const size_t count = program.batch * program.m * program.n;
                encode_fill(*context, output, count * sizeof(float), 0, sizeof(float));
                if (bias != nullptr) {
                    encode_nn(kBiasRelu, output, bias, output,
                              NnParams{.total = checked_u32(count, "Metal GEMM output count exceeds uint32"),
                                       .channels = checked_u32(program.m, "Metal GEMM rows exceed uint32"),
                                       .spatial = checked_u32(program.n, "Metal GEMM columns exceed uint32")});
                }
                return;
            }
            const auto lhs_at = context->locate(lhs);
            const auto rhs_at = context->locate(rhs);
            const auto output_at = context->locate(output);
            const auto bias_at = bias != nullptr ? context->locate(*bias) : output_at;
            const GemmParams params{
                .lhs_offset = lhs_at.offset,
                .rhs_offset = rhs_at.offset,
                .output_offset = output_at.offset,
                .bias_offset = bias_at.offset,
                .lhs_stride = program.m * program.k,
                .rhs_stride = program.k * program.n,
                .output_stride = program.m * program.n,
                .m = checked_extent(program.m, "Metal GEMM rows exceed int32"),
                .n = checked_extent(program.n, "Metal GEMM columns exceed int32"),
                .k = checked_extent(program.k, "Metal GEMM depth exceeds int32"),
                .padding = 0,
            };
            const size_t tile = gemm_tile(program.m, program.n, program.batch);
            const auto pipeline = context->pipeline(
                "gemm", {{14, transpose_rhs ? 1u : 0u}, {15, bias != nullptr ? 1u : 0u}, {27, tile == 64 ? 1u : 0u}});
            const std::array uses{lhs, rhs, output, bias != nullptr ? *bias : output};
            context->dispatch(uses, {.pipeline = pipeline,
                                     .buffers = {lhs_at.address, rhs_at.address, output_at.address, bias_at.address},
                                     .params = param_bytes(params),
                                     .grid = MTLSizeMake((program.n + tile - 1) / tile, (program.m + tile - 1) / tile,
                                                         program.batch),
                                     // The matmul runs on four SIMD groups.
                                     .group_size = MTLSizeMake(4 * pipeline.threadExecutionWidth, 1, 1)});
        }

        struct MatrixFillParams {
            uint64_t diagonal_offset;
            uint64_t output_offset;
            uint32_t columns;
            uint32_t count;
        };

        // eye (kind 0) or diag (kind 1) into a [rows][columns] Float32 matrix.
        API_AVAILABLE(macos(26.0))
        void encode_matrix_fill(const uint32_t kind, const StorageRef* const diagonal, const StorageRef output,
                                const size_t rows, const size_t columns) {
            const size_t count = rows * columns;
            if (count == 0)
                return;
            LFS_ASSERT_MSG(output.dtype == DataType::Float32 && (diagonal == nullptr || diagonal->dtype == DataType::Float32),
                           "Metal eye and diag write Float32");
            const auto context = acquire_context();
            const auto output_at = context->locate(output);
            const auto diagonal_at = diagonal != nullptr ? context->locate(*diagonal) : output_at;
            const MatrixFillParams params{
                .diagonal_offset = diagonal_at.offset,
                .output_offset = output_at.offset,
                .columns = checked_u32(columns, "Metal matrix columns exceed uint32"),
                .count = checked_u32(count, "Metal matrix element count exceeds uint32"),
            };
            const std::array uses{output, diagonal != nullptr ? *diagonal : output};
            context->dispatch(uses, {.pipeline = context->pipeline("matrix_fill", {{0, kind}}),
                                     .buffers = {diagonal_at.address, output_at.address},
                                     .params = param_bytes(params),
                                     .grid = threads(count)});
        }

        // index_op modes.
        constexpr uint32_t kGatherMode = 0, kTakeMode = 1, kIndexSelectMode = 2, kScatterAssignMode = 3,
                           kScatterAddMode = 4, kIndexPutMode = 5, kIndexFillMode = 6, kWinnerMode = 7,
                           kIndexCastMode = 8;

        struct IndexParams {
            uint64_t input_offset;
            uint64_t index_offset;
            uint64_t value_offset;
            uint64_t winner_offset;
            uint32_t outer;
            uint32_t dim_size;
            uint32_t inner;
            uint32_t index_size;
            uint32_t total;
            uint32_t rank;
            uint32_t index_rank;
            uint32_t dim;
            uint32_t fill_low;
            uint32_t fill_high;
            uint32_t input_size;
            uint32_t op_id;
            std::array<uint32_t, MAX_TENSOR_RANK> input_dims;
            std::array<uint32_t, MAX_TENSOR_RANK> index_dims;
        };
        static_assert(sizeof(IndexParams) == 32 + 12 * 4 + 2 * 32);

        // One index_op dispatch; the tensor it reads or writes is input, the
        // gathered output or scattered source is values.
        struct IndexLaunch {
            uint32_t mode;
            DataType dtype;
            size_t total;
            StorageRef input{};
            StorageRef indices{};
            StorageRef values{};
            StorageRef winners{};
            uint32_t boundary = 0;
            uint32_t unary = 0;
            IndexParams params{};
        };

        API_AVAILABLE(macos(26.0))
        void encode_index(Context& context, IndexLaunch launch) {
            if (launch.total == 0)
                return;
            std::vector<StorageRef> uses;
            std::array<uint64_t, 4> addresses{};
            const auto bind = [&](const StorageRef& storage, const size_t slot, uint64_t& offset) {
                if (storage.data == nullptr)
                    return;
                const auto at = context.locate(storage);
                addresses[slot] = at.address;
                offset = at.offset;
                uses.push_back(storage);
            };
            bind(launch.input, 0, launch.params.input_offset);
            bind(launch.indices, 1, launch.params.index_offset);
            bind(launch.values, 2, launch.params.value_offset);
            bind(launch.winners, 3, launch.params.winner_offset);
            launch.params.total = checked_u32(launch.total, "Metal index operation count exceeds uint32");
            const auto dtype = static_cast<uint32_t>(launch.dtype);
            const auto pipeline = context.pipeline(
                "index_op", {{0, launch.mode}, {1, dtype}, {2, dtype},
                             {5, static_cast<uint32_t>(dtype_size(launch.dtype))},
                             {17, launch.boundary}, {18, launch.unary}});
            context.dispatch(uses, {.pipeline = pipeline,
                                    .buffers = {addresses[0], addresses[1], addresses[2], addresses[3]},
                                    .params = param_bytes(launch.params),
                                    .grid = threads(launch.total)});
        }

        struct Geometry {
            size_t outer = 1;
            size_t dim_size = 0;
            size_t inner = 1;
        };

        Geometry geometry(const StridedLayout& layout, const int dim) {
            LFS_ASSERT_MSG(dim >= 0 && static_cast<size_t>(dim) < layout.rank, "index dimension is out of range");
            Geometry result{.dim_size = layout.dims[static_cast<size_t>(dim)]};
            for (size_t axis = 0; axis < layout.rank; ++axis) {
                if (static_cast<int>(axis) < dim)
                    result.outer *= layout.dims[axis];
                else if (static_cast<int>(axis) > dim)
                    result.inner *= layout.dims[axis];
            }
            return result;
        }

        // Scatter-family launches cover outer * index_size * inner source elements.
        IndexLaunch scatter_launch(const uint32_t mode, const StorageRef output, const StorageRef indices,
                                   const StorageRef source, const StridedLayout& output_layout, const int dim,
                                   const size_t index_size) {
            const Geometry shape = geometry(output_layout, dim);
            return IndexLaunch{
                .mode = mode,
                .dtype = output.dtype,
                .total = shape.outer * index_size * shape.inner,
                .input = output,
                .indices = indices,
                .values = source,
                .params = {.outer = checked_u32(shape.outer, "Metal scatter outer size exceeds uint32"),
                           .dim_size = checked_u32(shape.dim_size, "Metal scatter dimension exceeds uint32"),
                           .inner = checked_u32(shape.inner, "Metal scatter inner size exceeds uint32"),
                           .index_size = checked_u32(index_size, "Metal scatter index count exceeds uint32")},
            };
        }

        // Assignment with duplicate targets is deterministic: a first pass finds
        // the last position of every target, and only that position writes.
        API_AVAILABLE(macos(26.0))
        void scatter_assign(Context& context, IndexLaunch launch) {
            if (launch.total == 0)
                return;
            const Scratch last_positions(context, launch.params.dim_size * sizeof(int32_t));
            launch.winners = last_positions.storage;
            encode_fill(context, launch.winners, launch.params.dim_size * sizeof(int32_t), 0xffffffffu, sizeof(int32_t));
            IndexLaunch winners = launch;
            winners.mode = kWinnerMode;
            winners.total = launch.params.index_size;
            winners.values = {};
            encode_index(context, winners);
            launch.mode = kScatterAssignMode;
            encode_index(context, launch);
        }

        API_AVAILABLE(macos(26.0))
        void scatter_add(Context& context, IndexLaunch launch) {
            LFS_ASSERT_MSG(launch.dtype == DataType::Float32 || launch.dtype == DataType::Int32 ||
                               launch.dtype == DataType::UInt8 || launch.dtype == DataType::Bool,
                           "Metal scatter add supports Float32, Int32 and byte tensors");
            launch.mode = kScatterAddMode;
            encode_index(context, launch);
        }

        // mask_op modes and predicates.
        constexpr uint32_t kMaskFill = 0, kAndLive = 1, kCompactSelect = 2, kCompactScatter = 3, kNonzeroPositions = 4,
                           kMaskScan = 5, kWhereInto = 6;
        constexpr uint32_t kBytePredicate = 0, kFloatPredicate = 1;

        struct MaskLaunch {
            uint32_t mode;
            DataType dtype;
            size_t count;
            StorageRef data{};
            StorageRef mask{};
            StorageRef source{};
            StorageRef scan{};
            uint32_t predicate = kBytePredicate;
            std::pair<uint32_t, uint32_t> fill{};
        };

        API_AVAILABLE(macos(26.0))
        void encode_mask(Context& context, const MaskLaunch& launch) {
            if (launch.count == 0)
                return;
            struct MaskParams {
                uint64_t data_offset;
                uint64_t mask_offset;
                uint64_t source_offset;
                uint64_t scan_offset;
                uint32_t count;
                uint32_t fill_low;
                uint32_t fill_high;
                uint32_t padding;
            };
            MaskParams params{.count = checked_u32(launch.count, "Metal mask count exceeds uint32"),
                              .fill_low = launch.fill.first,
                              .fill_high = launch.fill.second};
            std::vector<StorageRef> uses;
            std::array<uint64_t, 4> addresses{};
            const auto bind = [&](const StorageRef& storage, const size_t slot, uint64_t& offset) {
                if (storage.data == nullptr)
                    return;
                const auto at = context.locate(storage);
                addresses[slot] = at.address;
                offset = at.offset;
                uses.push_back(storage);
            };
            bind(launch.data, 0, params.data_offset);
            bind(launch.mask, 1, params.mask_offset);
            bind(launch.source, 2, params.source_offset);
            bind(launch.scan, 3, params.scan_offset);
            const auto dtype = static_cast<uint32_t>(launch.dtype);
            const auto pipeline = context.pipeline(
                "mask_op", {{0, launch.mode}, {1, dtype}, {2, dtype},
                            {5, static_cast<uint32_t>(dtype_size(launch.dtype))}, {19, launch.predicate}});
            context.dispatch(uses, {.pipeline = pipeline,
                                    .buffers = {addresses[0], addresses[1], addresses[2], addresses[3]},
                                    .params = param_bytes(params),
                                    .grid = threads(launch.count)});
        }

        // Inclusive scan of the predicate: the compacted slot of i is scan[i] - 1.
        struct API_AVAILABLE(macos(26.0)) PredicateScan : Scratch {
            PredicateScan(Context& owner, const uint32_t predicate, const StorageRef mask, const size_t count)
                : Scratch(owner, count * sizeof(uint32_t)) {
                storage.dtype = DataType::Int32;
                encode_mask(owner, {.mode = kMaskScan, .dtype = DataType::UInt8, .count = count, .mask = mask,
                                    .scan = storage, .predicate = predicate});
                encode_scan(owner, storage, 1, count, 1);
            }
        };

        // Writes the Int64 positions where the predicate holds and returns
        // their count, which the scan's last element holds.
        API_AVAILABLE(macos(26.0))
        size_t compact_nonzero(const uint32_t predicate, const StorageRef input, const StorageRef output,
                               const MaskProgram& program) {
            if (program.count == 0 || program.selected_count == 0)
                return 0;
            const auto context = acquire_context();
            const PredicateScan scan(*context, predicate, input, program.count);
            encode_mask(*context, {.mode = kNonzeroPositions, .dtype = DataType::Int64, .count = program.count,
                                   .mask = input, .source = output, .scan = scan.storage, .predicate = predicate});
            context->wait(context->pending(scan.storage));
            uint32_t total = 0;
            std::memcpy(&total, context->host(scan.storage) + (program.count - 1) * sizeof(uint32_t), sizeof(total));
            return total;
        }

        struct SortParams {
            uint64_t values_offset;
            uint64_t indices_offset;
            uint64_t keys_a_offset;
            uint64_t keys_b_offset;
            uint64_t positions_a_offset;
            uint64_t positions_b_offset;
            uint64_t histogram_offset;
            uint32_t lines;
            uint32_t dim_size;
            uint32_t inner;
            uint32_t blocks_per_line;
            uint32_t shift;
            uint32_t parity;
            uint32_t descending;
            uint32_t total;
        };
        static_assert(sizeof(SortParams) == 7 * 8 + 8 * 4);

        // Lines up to kSortCapacity (kernels.metal) sort in one threadgroup;
        // longer ones take the radix sort, 4 bits per pass.
        constexpr size_t kSortCapacity = 2048;
        constexpr size_t kRadixBlock = kThreadgroupWidth * 8;
        constexpr size_t kRadixDigits = 16;
        constexpr uint32_t kRadixPasses = 8;

        // Sorts the lines of an (outer, dim_size, inner) view in place and
        // writes each element's source position along the line.
        API_AVAILABLE(macos(26.0))
        void sort_lines(const StorageRef values, const StorageRef indices, const size_t lines, const size_t dim_size,
                        const size_t inner, const bool descending) {
            LFS_ASSERT_MSG(values.dtype == DataType::Float32 && indices.dtype == DataType::Int64,
                           "Metal sort requires Float32 values and Int64 indices");
            if (lines == 0 || dim_size == 0)
                return;
            const auto context = acquire_context();
            const auto values_at = context->locate(values);
            const auto indices_at = context->locate(indices);
            SortParams params{
                .values_offset = values_at.offset,
                .indices_offset = indices_at.offset,
                .lines = checked_u32(lines, "Metal sort line count exceeds uint32"),
                .dim_size = checked_u32(dim_size, "Metal sort size exceeds uint32"),
                .inner = checked_u32(inner, "Metal sort inner size exceeds uint32"),
                .descending = descending ? 1u : 0u,
            };
            if (dim_size <= kSortCapacity) {
                const std::array uses{values, indices};
                context->dispatch(uses, {.pipeline = context->pipeline("sort_shared"),
                                         .buffers = {values_at.address, indices_at.address},
                                         .params = param_bytes(params),
                                         .grid = MTLSizeMake(lines, 1, 1),
                                         .group_size = MTLSizeMake(kThreadgroupWidth, 1, 1)});
                return;
            }
            // Keys and positions ping-pong between A and B; the digit histograms
            // follow, all in one scratch block.
            const size_t total = lines * dim_size;
            const size_t blocks_per_line = (dim_size + kRadixBlock - 1) / kRadixBlock;
            const Scratch scratch(*context, (4 * total + lines * kRadixDigits * blocks_per_line) * sizeof(uint32_t));
            const auto scratch_at = context->locate(scratch.storage);
            const uint64_t array_bytes = total * sizeof(uint32_t);
            params.keys_a_offset = scratch_at.offset;
            params.keys_b_offset = scratch_at.offset + array_bytes;
            params.positions_a_offset = scratch_at.offset + 2 * array_bytes;
            params.positions_b_offset = scratch_at.offset + 3 * array_bytes;
            params.histogram_offset = scratch_at.offset + 4 * array_bytes;
            params.blocks_per_line = checked_u32(blocks_per_line, "Metal sort block count exceeds uint32");
            params.total = checked_u32(total, "Metal sort element count exceeds uint32");
            const std::array uses{values, indices, scratch.storage};
            const auto dispatch = [&](const uint32_t phase, const size_t groups) {
                context->dispatch(uses, {.pipeline = context->pipeline("radix_sort", {{0, phase}}),
                                         .buffers = {values_at.address, indices_at.address, scratch_at.address},
                                         .params = param_bytes(params),
                                         .grid = MTLSizeMake(groups, 1, 1),
                                         .group_size = MTLSizeMake(kThreadgroupWidth, 1, 1)});
            };
            const size_t element_groups = (total + kThreadgroupWidth - 1) / kThreadgroupWidth;
            dispatch(0, element_groups);
            for (uint32_t pass = 0; pass < kRadixPasses; ++pass) {
                params.shift = pass * 4;
                params.parity = pass & 1u;
                dispatch(1, lines * blocks_per_line);
                dispatch(2, lines);
                dispatch(3, lines * blocks_per_line);
            }
            dispatch(4, element_groups);
            dispatch(5, element_groups);
        }

        // random_op kinds.
        constexpr uint32_t kUniform = 0, kBernoulli = 1, kRandint = 2, kNormal = 3, kMultinomialReplacement = 4,
                           kGumbelKeys = 5, kRankSelect = 6, kWeightStatistics = 7;

        struct RandomParams {
            uint64_t output_offset;
            uint64_t weights_offset;
            uint64_t keys_offset;
            uint64_t seed;
            uint32_t count;
            uint32_t sample_count;
            int32_t low;
            int32_t high;
            float first;
            float second;
            float total;
            uint32_t padding;
        };
        static_assert(sizeof(RandomParams) == 64);

        API_AVAILABLE(macos(26.0))
        void encode_random(Context& context, const uint32_t kind, const StorageRef output, const StorageRef weights,
                           const StorageRef keys, RandomParams params, const MTLSize grid) {
            std::vector<StorageRef> uses;
            std::array<uint64_t, 3> addresses{};
            const auto bind = [&](const StorageRef& storage, const size_t slot, uint64_t& offset) {
                if (storage.data == nullptr)
                    return;
                const auto at = context.locate(storage);
                addresses[slot] = at.address;
                offset = at.offset;
                uses.push_back(storage);
            };
            bind(output, 0, params.output_offset);
            bind(weights, 1, params.weights_offset);
            bind(keys, 2, params.keys_offset);
            context.dispatch(uses, {.pipeline = context.pipeline("random_op", {{0, kind}}),
                                    .buffers = {addresses[0], addresses[1], addresses[2]},
                                    .params = param_bytes(params),
                                    .grid = grid,
                                    .group_size = MTLSizeMake(kThreadgroupWidth, 1, 1)});
        }

        MTLSize thread_groups(const size_t count) {
            return MTLSizeMake((count + kThreadgroupWidth - 1) / kThreadgroupWidth, 1, 1);
        }

        // Elementwise draws: every element is one Philox block keyed by the seed.
        API_AVAILABLE(macos(26.0))
        void draw_elements(const uint32_t kind, const StorageRef output, const RandomProgram& program) {
            if (program.count == 0)
                return;
            encode_random(*acquire_context(), kind, output, {}, {},
                          {.seed = program.seed,
                           .count = checked_u32(program.count, "Metal random count exceeds uint32"),
                           .low = program.low,
                           .high = program.high,
                           .first = program.first,
                           .second = program.second},
                          thread_groups(program.count));
        }

        // Address of a storage's first element, for kernels that take operands
        // as GPU addresses in their parameters.
        API_AVAILABLE(macos(26.0))
        uint64_t address_of(Context& context, const StorageRef& storage) {
            const auto at = context.locate(storage);
            return at.address + at.offset;
        }

        // Dispatches a kernel whose operands are all addresses in its parameters.
        template <class Params>
        API_AVAILABLE(macos(26.0))
        void dispatch_addressed(Context& context, const std::span<const StorageRef> uses,
                                const id<MTLComputePipelineState> pipeline, const Params& params, const size_t count) {
            if (count == 0)
                return;
            context.dispatch(uses, {.pipeline = pipeline, .buffers = {}, .params = param_bytes(params), .grid = threads(count)});
        }

        struct ResampleParams {
            uint64_t input, output;
            float src_fx, src_fy, src_cx, src_cy, dst_fx, dst_fy, dst_cx, dst_cy;
            int32_t sw, sh, dw, dh;
            float distortion[12];
            int32_t model, num_distortion, channels, padding;
        };
        static_assert(sizeof(ResampleParams) == 128);

        // Resamples a contiguous Float32 image into a new [C,H,W] or [H,W] tensor.
        API_AVAILABLE(macos(26.0))
        Tensor resample_image(const Tensor& input, ResampleParams params, const bool plane, const uint32_t kind) {
            const GpuBackendScope scope(GpuBackend::Metal);
            const auto height = static_cast<size_t>(params.dh), width = static_cast<size_t>(params.dw);
            auto output = Tensor::empty(plane ? TensorShape{height, width}
                                              : TensorShape{static_cast<size_t>(params.channels), height, width},
                                        Device::GPU);
            const auto context = acquire_context();
            const std::array uses{storage_ref(input), storage_ref(output)};
            params.input = address_of(*context, uses[0]);
            params.output = address_of(*context, uses[1]);
            dispatch_addressed(*context, uses, context->pipeline("image_resample", {{0, kind}}), params, height * width);
            return output;
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
        // Metal adds floats atomically and screens the SH3 assignment with
        // SIMD-group matrices.
        class API_AVAILABLE(macos(26.0)) MetalExportKernels final : public ExportKernels {
        public:
            GpuBackend backend() const override { return GpuBackend::Metal; }

            bool float_atomics() const override { return true; }

            bool screened_assignment() const override { return true; }

            uint64_t address(const Tensor& tensor) const override { return address_of(*context_, storage_ref(tensor)); }

            void launch(const char* const module, const uint32_t phase, const std::span<const std::byte> params,
                        const std::span<const StorageRef> reads, const std::span<const StorageRef> writes,
                        const size_t work) override {
                std::vector<StorageRef> uses(reads.begin(), reads.end());
                uses.insert(uses.end(), writes.begin(), writes.end());
                context_->dispatch(uses, {.pipeline = context_->pipeline(module, {{0, phase}}),
                                          .params = params,
                                          .grid = thread_groups(work),
                                          .group_size = MTLSizeMake(kThreadgroupWidth, 1, 1)});
            }

        private:
            std::shared_ptr<Context> context_ = acquire_context();
        };

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

    // Fused expressions compile to MSL once per layout class; the launch
    // arguments are the dispatch parameters.
    void MetalBackendOps::compiled_expression(const ExpressionLaunch& launch, ExecContext) {
        LFS_FACADE_TRACE(compiled_expression);
        const auto context = acquire_context();
        const auto pipeline = context->expression_pipeline(*launch.program, launch.signature);
        const uint32_t packing = expression_packing(launch.signature);
        const uint64_t work = (uint64_t{launch.count} + packing - 1) / packing;
        if (launch.prepare_only || work == 0)
            return;
        std::vector<StorageRef> uses(launch.reads.begin(), launch.reads.begin() + launch.read_count);
        uses.insert(uses.end(), launch.writes.begin(), launch.writes.begin() + launch.write_count);
        context->dispatch(uses, {.pipeline = pipeline,
                                 .buffers = {},
                                 .params = std::as_bytes(std::span(launch.arguments.data(), launch.words)),
                                 .grid = threads(work)});
    }

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
        const FusedChain fused = fused_chain(*context, chain);
        params.ops = fused.ops;
        const bool vectorized = fused.aligned && input_at.offset % 16 == 0 && output_at.offset % 16 == 0;
        const auto pipeline = context->pipeline(
            "pointwise_chain", {{4, vectorized ? 1u : 0u}, {8, fused.length},
                                {9, fused.kinds[0]}, {10, fused.kinds[1]}, {11, fused.kinds[2]}});
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
        if (is_contiguous(layout)) {
            load_fill(output, layout.element_count, value, context);
            return;
        }
        if (layout.element_count == 0)
            return;
        LFS_ASSERT_MSG(layout.rank <= MAX_TENSOR_RANK, "Metal strided fill rank exceeds MAX_TENSOR_RANK");
        const auto metal = acquire_context();
        const auto output_at = metal->locate(output);
        FillParams params{.output_offset = output_at.offset,
                          .pattern = fill_pattern(output.dtype, value),
                          .count = checked_u32(layout.element_count, "Metal fill count exceeds uint32"),
                          .rank = static_cast<uint32_t>(layout.rank)};
        for (size_t axis = 0; axis < layout.rank; ++axis) {
            params.dims[axis] = checked_u32(layout.dims[axis], "Metal fill extent exceeds uint32");
            params.strides[axis] = checked_u32(layout.strides[axis], "Metal fill stride exceeds uint32");
        }
        const std::array uses{output};
        metal->dispatch(uses, {.pipeline = metal->pipeline("fill", {{0, 1}, {5, static_cast<uint32_t>(dtype_size(output.dtype))}}),
                               .buffers = {output_at.address},
                               .params = param_bytes(params),
                               .grid = threads(layout.element_count)});
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
            uint32_t count;
            std::array<uint32_t, 3> padding;
            BroadcastAxes axes;
        };
        static_assert(sizeof(WhereParams) == 224);
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
            .count = checked_u32(output_layout.element_count, "Metal where count exceeds uint32"),
            .padding = {},
            .axes = broadcast_axes(output_layout, {&condition_layout, &x_layout, &y_layout}),
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
        return scalar_reduce(ReduceOp::Sum, input, count);
    }

    float MetalBackendOps::mean_scalar(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(mean_scalar);
        return scalar_reduce(ReduceOp::Mean, input, count);
    }

    float MetalBackendOps::max_scalar(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(max_scalar);
        return scalar_reduce(ReduceOp::Max, input, count);
    }

    float MetalBackendOps::min_scalar(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(min_scalar);
        return scalar_reduce(ReduceOp::Min, input, count);
    }

    void MetalBackendOps::broadcast_binary(const PointwiseProgram& program, const StorageRef lhs,
                                           const StridedLayout& lhs_layout, const StorageRef rhs,
                                           const StridedLayout& rhs_layout, const StorageRef output,
                                           const StridedLayout& output_layout, ExecContext) {
        LFS_FACADE_TRACE(broadcast_binary);
        if (output_layout.element_count == 0)
            return;
        LFS_ASSERT_MSG(lhs.dtype == program.in_dtype && rhs.dtype == program.in_dtype &&
                           output.dtype == program.out_dtype,
                       "Metal broadcast dtype does not match the pointwise program");
        struct BroadcastParams {
            PointwiseParams pointwise;
            BroadcastAxes axes;
        };
        static_assert(sizeof(BroadcastParams) == 224);
        const auto context = acquire_context();
        const auto lhs_at = context->locate(lhs);
        const auto rhs_at = context->locate(rhs);
        const auto output_at = context->locate(output);
        const BroadcastParams params{
            .pointwise = {.lhs_offset = lhs_at.offset,
                          .rhs_offset = rhs_at.offset,
                          .output_offset = output_at.offset,
                          .count = checked_u32(output_layout.element_count, "Metal broadcast count exceeds uint32")},
            .axes = broadcast_axes(output_layout, {&lhs_layout, &rhs_layout}),
        };
        const auto pipeline = context->pipeline(
            "broadcast_binary", {{0, static_cast<uint32_t>(program.op)},
                                 {1, static_cast<uint32_t>(program.in_dtype)},
                                 {2, static_cast<uint32_t>(program.out_dtype)},
                                 {3, 2}});
        const std::array uses{lhs, rhs, output};
        context->dispatch(uses, {.pipeline = pipeline,
                                 .buffers = {lhs_at.address, rhs_at.address, output_at.address},
                                 .params = param_bytes(params),
                                 .grid = threads(output_layout.element_count)});
    }

    void MetalBackendOps::clamp_scalar(const StorageRef data, const ScalarOperand minimum,
                                       const ScalarOperand maximum, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(clamp_scalar);
        encode_clamp(data, data, minimum, maximum, count);
    }

    void MetalBackendOps::clamp_fused(const StorageRef input, const StorageRef output, const ScalarOperand minimum,
                                      const ScalarOperand maximum, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(clamp_fused);
        encode_clamp(input, output, minimum, maximum, count);
    }

    void MetalBackendOps::clamp_scalar_int(const StorageRef data, const ScalarOperand minimum,
                                           const ScalarOperand maximum, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(clamp_scalar_int);
        encode_clamp(data, data, minimum, maximum, count);
    }

    void MetalBackendOps::cat_last_dim(const StorageRef output, const std::span<const StorageRef> inputs,
                                       const std::span<const StridedLayout> layouts, const size_t num_rows,
                                       const size_t row_size, const size_t element_size, ExecContext) {
        LFS_FACADE_TRACE(cat_last_dim);
        LFS_ASSERT_MSG(inputs.size() == layouts.size() && !inputs.empty(),
                       "Metal cat requires matching non-empty input metadata");
        LFS_ASSERT_MSG(element_size == dtype_size(output.dtype), "Metal cat element size does not match output dtype");
        size_t column = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            LFS_ASSERT_MSG(inputs[i].dtype == output.dtype && layouts[i].rank > 0,
                           "Metal cat input dtype or rank is invalid");
            const size_t width = layouts[i].dims[layouts[i].rank - 1];
            encode_cat_input(inputs[i], output, width, row_size, column, num_rows * width);
            column += width;
        }
        LFS_ASSERT_MSG(column == row_size, "Metal last-dimension cat widths do not sum to output width");
    }

    void MetalBackendOps::cat_middle_dim(const StorageRef output, const std::span<const StorageRef> inputs,
                                         const std::span<const StridedLayout> layouts, const size_t outer_size,
                                         const size_t inner_size, const int dim, const size_t element_size,
                                         ExecContext) {
        LFS_FACADE_TRACE(cat_middle_dim);
        LFS_ASSERT_MSG(inputs.size() == layouts.size() && !inputs.empty(),
                       "Metal cat requires matching non-empty input metadata");
        LFS_ASSERT_MSG(dim >= 0 && element_size == dtype_size(output.dtype),
                       "Metal middle-dimension cat metadata is invalid");
        size_t total = 0;
        for (const StridedLayout& layout : layouts) {
            LFS_ASSERT_MSG(static_cast<size_t>(dim) < layout.rank, "Metal cat dimension is outside an input rank");
            total += layout.dims[dim];
        }
        size_t offset = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            LFS_ASSERT_MSG(inputs[i].dtype == output.dtype, "Metal cat inputs must match output dtype");
            const size_t extent = layouts[i].dims[dim];
            encode_cat_input(inputs[i], output, extent * inner_size, total * inner_size, offset * inner_size,
                             outer_size * extent * inner_size);
            offset += extent;
        }
    }

    void MetalBackendOps::pad(const StorageRef input, const StorageRef output, const StridedLayout& input_layout,
                              const StridedLayout& output_layout,
                              const std::array<size_t, MAX_TENSOR_RANK>& pad_before, ExecContext) {
        LFS_FACADE_TRACE(pad);
        if (input_layout.element_count == 0)
            return;
        LFS_ASSERT_MSG(input.dtype == output.dtype && input_layout.rank == output_layout.rank &&
                           input_layout.rank <= MAX_TENSOR_RANK,
                       "Metal pad requires matching dtypes and ranks");
        const auto context = acquire_context();
        const auto input_at = context->locate(input);
        const auto output_at = context->locate(output);
        const size_t rank = input_layout.rank;
        const CatPadParams params{
            .input_offset = input_at.offset,
            .output_offset = output_at.offset,
            .input_dims = shader_dims(input_layout),
            .input_strides = shader_values(input_layout.strides, rank, "Metal pad input stride exceeds uint32"),
            .output_strides = shader_values(output_layout.strides, rank, "Metal pad output stride exceeds uint32"),
            .pad_before = shader_values(pad_before, rank, "Metal pad width exceeds uint32"),
            .count = checked_u32(input_layout.element_count, "Metal pad count exceeds uint32"),
            .rank = static_cast<uint32_t>(rank),
        };
        const std::array uses{input, output};
        const uint32_t dtype = static_cast<uint32_t>(input.dtype);
        context->dispatch(uses, {.pipeline = context->pipeline(
                                     "cat_pad", {{0, 1}, {1, dtype}, {2, dtype},
                                                 {5, static_cast<uint32_t>(dtype_size(input.dtype))}}),
                                 .buffers = {input_at.address, output_at.address},
                                 .params = param_bytes(params),
                                 .grid = threads(input_layout.element_count)});
    }

    void MetalBackendOps::index_cast(const StorageRef input, const StorageRef output, const size_t count,
                                     const size_t extent, ExecContext) {
        LFS_FACADE_TRACE(index_cast);
        encode_index(*acquire_context(),
                     {.mode = kIndexCastMode, .dtype = DataType::Int32, .total = count, .input = input, .values = output,
                      .params = {.dim_size = checked_u32(extent, "Metal index extent exceeds uint32")}});
    }

    void MetalBackendOps::gather(const StorageRef input, const StorageRef indices, const StorageRef output,
                                 const StridedLayout& input_layout, const StridedLayout& index_layout,
                                 const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(gather);
        LFS_ASSERT_MSG(input_layout.rank > 0 && input_layout.rank <= MAX_TENSOR_RANK &&
                           index_layout.rank <= MAX_TENSOR_RANK,
                       "gather rank exceeds MAX_TENSOR_RANK");
        encode_index(*acquire_context(),
                     {.mode = kGatherMode, .dtype = input.dtype, .total = program.total_elements, .input = input,
                      .indices = indices, .values = output, .boundary = static_cast<uint32_t>(program.boundary_mode),
                      .params = {.rank = static_cast<uint32_t>(input_layout.rank),
                                 .index_rank = static_cast<uint32_t>(index_layout.rank),
                                 .dim = static_cast<uint32_t>(program.dim),
                                 .input_dims = shader_dims(input_layout),
                                 .index_dims = shader_dims(index_layout)}});
    }

    void MetalBackendOps::gather_fused_unary(const StorageRef input, const StorageRef indices, const StorageRef output,
                                             const PointwiseOp unary, const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(gather_fused_unary);
        LFS_ASSERT_MSG(input.dtype == DataType::Float32 && output.dtype == DataType::Float32,
                       "Metal fused gather supports only Float32");
        const uint32_t unary_code = unary == PointwiseOp::Abs ? 1u : unary == PointwiseOp::Sqrt ? 2u
                                                                 : unary == PointwiseOp::Neg    ? 3u
                                                                                                : 0u;
        LFS_ASSERT_MSG(unary_code != 0, "unsupported fused gather unary operation");
        encode_index(*acquire_context(),
                     {.mode = kTakeMode, .dtype = DataType::Float32, .total = program.index_size, .input = input,
                      .indices = indices, .values = output, .unary = unary_code,
                      .params = {.input_size = checked_u32(program.input_size, "Metal gather input size exceeds uint32")}});
    }

    void MetalBackendOps::take(const StorageRef input, const StorageRef indices, const StorageRef output,
                               const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(take);
        encode_index(*acquire_context(),
                     {.mode = kTakeMode, .dtype = input.dtype, .total = program.index_size, .input = input,
                      .indices = indices, .values = output,
                      .params = {.input_size = checked_u32(program.input_size, "Metal take input size exceeds uint32")}});
    }

    void MetalBackendOps::index_select(const StorageRef input, const StorageRef indices, const StorageRef output,
                                       const StridedLayout& input_layout, const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(index_select);
        const Geometry shape = geometry(input_layout, program.dim);
        encode_index(*acquire_context(),
                     {.mode = kIndexSelectMode, .dtype = input.dtype,
                      .total = shape.outer * program.index_size * shape.inner, .input = input, .indices = indices,
                      .values = output, .boundary = static_cast<uint32_t>(program.boundary_mode),
                      .params = {.outer = checked_u32(shape.outer, "Metal index_select outer size exceeds uint32"),
                                 .dim_size = checked_u32(shape.dim_size, "Metal index_select dimension exceeds uint32"),
                                 .inner = checked_u32(shape.inner, "Metal index_select inner size exceeds uint32"),
                                 .index_size = checked_u32(program.index_size, "Metal index_select index count exceeds uint32")}});
    }

    void MetalBackendOps::scatter(const StorageRef output, const StorageRef indices, const StorageRef source,
                                  const StridedLayout& output_layout, const StridedLayout& index_layout,
                                  const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(scatter);
        LFS_ASSERT_MSG(output_layout.rank > 0 && output_layout.rank <= MAX_TENSOR_RANK,
                       "scatter rank exceeds MAX_TENSOR_RANK");
        const auto context = acquire_context();
        const IndexLaunch launch = scatter_launch(kScatterAssignMode, output, indices, source, output_layout, program.dim,
                                                  index_layout.dims[static_cast<size_t>(program.dim)]);
        if (program.scatter_mode == static_cast<int>(ScatterMode::Add))
            scatter_add(*context, launch);
        else
            scatter_assign(*context, launch);
    }

    void MetalBackendOps::index_copy(const StorageRef output, const StorageRef indices, const StorageRef source,
                                     const StridedLayout& output_layout, const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(index_copy);
        scatter_assign(*acquire_context(), scatter_launch(kScatterAssignMode, output, indices, source, output_layout,
                                                          program.dim, program.index_size));
    }

    void MetalBackendOps::index_add(const StorageRef output, const StorageRef indices, const StorageRef source,
                                    const StridedLayout& output_layout, const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(index_add);
        LFS_ASSERT_MSG(output.dtype == DataType::Float32 || output.dtype == DataType::Int32,
                       "Metal index_add supports only Float32 and Int32");
        scatter_add(*acquire_context(), scatter_launch(kScatterAddMode, output, indices, source, output_layout,
                                                       program.dim, program.index_size));
    }

    void MetalBackendOps::index_fill(const StorageRef output, const StorageRef indices,
                                     const StridedLayout& output_layout, const IndexProgram& program,
                                     const ScalarOperand value, ExecContext) {
        LFS_FACADE_TRACE(index_fill);
        IndexLaunch launch = scatter_launch(kIndexFillMode, output, indices, {}, output_layout, program.dim,
                                            program.index_size);
        std::tie(launch.params.fill_low, launch.params.fill_high) = fill_bits(output.dtype, value);
        encode_index(*acquire_context(), launch);
    }

    void MetalBackendOps::index_put(const StorageRef output, const StorageRef indices, const StorageRef values,
                                    const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(index_put);
        encode_index(*acquire_context(),
                     {.mode = kIndexPutMode, .dtype = output.dtype, .total = program.index_size, .input = output,
                      .indices = indices, .values = values,
                      .params = {.input_size = checked_u32(program.input_size, "Metal index_put size exceeds uint32")}});
    }

    void metal_where_into(Tensor& output, const Tensor& condition, const float value, const Tensor& source) {
        LFS_FACADE_TRACE(where);
        pin_operands({&output, &condition, &source});
        if (@available(macOS 26.0, *)) {
            encode_mask(*acquire_context(), {.mode = kWhereInto,
                                             .dtype = output.dtype(),
                                             .count = output.numel(),
                                             .data = storage_ref(output),
                                             .mask = storage_ref(condition),
                                             .source = storage_ref(source),
                                             .fill = fill_bits(output.dtype(), scalar_operand(value))});
        }
    }

    void MetalBackendOps::masked_fill(const StorageRef output, const StorageRef mask, const MaskProgram& program,
                                      ExecContext) {
        LFS_FACADE_TRACE(masked_fill);
        encode_mask(*acquire_context(), {.mode = kMaskFill, .dtype = output.dtype, .count = program.count,
                                         .data = output, .mask = mask,
                                         .fill = fill_bits(output.dtype, program.value)});
    }

    size_t MetalBackendOps::masked_select(const StorageRef input, const StorageRef mask, const StorageRef output,
                                          const MaskProgram& program, ExecContext) {
        LFS_FACADE_TRACE(masked_select);
        if (program.count == 0 || program.selected_count == 0)
            return 0;
        const auto context = acquire_context();
        const PredicateScan scan(*context, kBytePredicate, mask, program.count);
        encode_mask(*context, {.mode = kCompactSelect, .dtype = input.dtype, .count = program.count, .data = input,
                               .mask = mask, .source = output, .scan = scan.storage});
        // The host sized the output from the same mask; like CUDA, the launch trusts it.
        return program.selected_count;
    }

    void MetalBackendOps::masked_scatter(const StorageRef output, const StorageRef mask, const StorageRef source,
                                         const MaskProgram& program, ExecContext) {
        LFS_FACADE_TRACE(masked_scatter);
        if (program.count == 0 || program.selected_count == 0)
            return;
        const auto context = acquire_context();
        const PredicateScan scan(*context, kBytePredicate, mask, program.count);
        encode_mask(*context, {.mode = kCompactScatter, .dtype = output.dtype, .count = program.count, .data = output,
                               .mask = mask, .source = source, .scan = scan.storage});
    }

    void MetalBackendOps::and_live(const StorageRef mask, const StorageRef live_mask, const MaskProgram& program,
                                   ExecContext) {
        LFS_FACADE_TRACE(and_live);
        encode_mask(*acquire_context(), {.mode = kAndLive, .dtype = DataType::UInt8, .count = program.count,
                                         .data = mask, .mask = live_mask});
    }

    size_t MetalBackendOps::nonzero(const StorageRef input, const StorageRef output, const MaskProgram& program,
                                    ExecContext) {
        LFS_FACADE_TRACE(nonzero);
        LFS_ASSERT_MSG(input.dtype == DataType::Float32, "Metal nonzero supports only Float32");
        return compact_nonzero(kFloatPredicate, input, output, program);
    }

    size_t MetalBackendOps::nonzero_bool(const StorageRef input, const StorageRef output, const MaskProgram& program,
                                         ExecContext) {
        LFS_FACADE_TRACE(nonzero_bool);
        return compact_nonzero(kBytePredicate, input, output, program);
    }

    void MetalBackendOps::sort_1d(const StorageRef values, const StorageRef indices, const size_t count,
                                  const SortProgram& program, ExecContext) {
        LFS_FACADE_TRACE(sort_1d);
        sort_lines(values, indices, 1, count, 1, program.descending);
    }

    void MetalBackendOps::sort_2d(const StorageRef values, const StorageRef indices, const SortProgram& program,
                                  ExecContext) {
        LFS_FACADE_TRACE(sort_2d);
        sort_lines(values, indices, program.outer_size * program.inner_size, program.dim_size, program.inner_size,
                   program.descending);
    }

    void MetalBackendOps::uniform(const StorageRef output, const RandomProgram& program, ExecContext) {
        LFS_FACADE_TRACE(uniform);
        draw_elements(kUniform, output, program);
    }

    void MetalBackendOps::bernoulli(const StorageRef output, const RandomProgram& program, ExecContext) {
        LFS_FACADE_TRACE(bernoulli);
        draw_elements(kBernoulli, output, program);
    }

    void MetalBackendOps::randint(const StorageRef output, const RandomProgram& program, ExecContext) {
        LFS_FACADE_TRACE(randint);
        draw_elements(kRandint, output, program);
    }

    // Every element draws its own Philox block, so odd counts need no scratch.
    void MetalBackendOps::normal(const StorageRef output, StorageRef, const RandomProgram& program, ExecContext) {
        LFS_FACADE_TRACE(normal);
        draw_elements(kNormal, output, program);
    }

    void MetalBackendOps::multinomial(const StorageRef weights, const StorageRef output, const RandomProgram& program,
                                      ExecContext) {
        LFS_FACADE_TRACE(multinomial);
        if (program.count == 0 || program.sample_count == 0)
            return;
        LFS_ASSERT_MSG(weights.dtype == DataType::Float32 && output.dtype == DataType::Int64,
                       "Metal multinomial requires Float32 weights and Int64 samples");
        const auto context = acquire_context();
        const uint32_t categories = checked_u32(program.count, "Metal multinomial category count exceeds uint32");
        const uint32_t samples = checked_u32(program.sample_count, "Metal multinomial sample count exceeds uint32");
        // The weights are validated on the host, like the CUDA path.
        struct WeightStatistics {
            float sum;
            uint32_t invalid;
            float scale;
        };
        WeightStatistics statistics{};
        {
            const Scratch scratch(*context, sizeof(WeightStatistics));
            encode_random(*context, kWeightStatistics, scratch.storage, weights, {}, {.count = categories},
                          MTLSizeMake(1, 1, 1));
            context->wait(context->pending(scratch.storage));
            std::memcpy(&statistics, context->host(scratch.storage), sizeof(statistics));
        }
        LFS_ASSERT_MSG(statistics.invalid == 0, "multinomial weights must be finite and non-negative");
        LFS_ASSERT_MSG(std::isfinite(statistics.sum) && statistics.sum > 0.0f,
                       "multinomial weights must have a positive finite sum");
        if (program.replacement) {
            encode_random(*context, kMultinomialReplacement, output, weights, {},
                          {.seed = program.seed, .count = categories, .sample_count = samples,
                           .first = statistics.scale, .total = statistics.sum},
                          thread_groups(program.sample_count));
            return;
        }
        LFS_ASSERT_MSG(program.sample_count <= program.count,
                       "multinomial sample count exceeds weights without replacement");
        // Gumbel-top-k: the sample_count largest perturbed log-weights, ranked
        // by counting; ties fall back to the lower index.
        const Scratch keys(*context, program.count * sizeof(float));
        encode_random(*context, kGumbelKeys, {}, weights, keys.storage,
                      {.seed = program.seed, .count = categories, .sample_count = samples},
                      thread_groups(program.count));
        encode_random(*context, kRankSelect, output, {}, keys.storage, {.count = categories, .sample_count = samples},
                      thread_groups(program.count));
    }

    void MetalBackendOps::radius_neighbors(const StorageRef points, const StorageRef references, const StorageRef heads,
                                           const StorageRef next, const StorageRef output, const size_t count,
                                           const size_t buckets, const float radius, ExecContext) {
        LFS_FACADE_TRACE(radius_neighbors);
        struct RadiusParams {
            uint64_t points, references, heads, next, output;
            uint32_t count, bucket_mask;
            float radius;
            uint32_t padding;
        };
        const auto context = acquire_context();
        const RadiusParams params{
            .points = address_of(*context, points),
            .references = address_of(*context, references),
            .heads = address_of(*context, heads),
            .next = address_of(*context, next),
            .output = address_of(*context, output),
            .count = checked_u32(count, "Metal radius query count exceeds uint32"),
            .bucket_mask = checked_u32(buckets - 1, "Metal radius bucket count exceeds uint32"),
            .radius = radius,
        };
        const std::array uses{points, references, heads, next, output};
        dispatch_addressed(*context, uses, context->pipeline("radius_neighbors", {{0, 0}}), params, count);
        dispatch_addressed(*context, uses, context->pipeline("radius_neighbors", {{0, 1}}), params, count);
    }

    void MetalBackendOps::project_points(const StorageRef points, const StorageRef output, const size_t count,
                                         const PointProjection& projection, const StorageRef* transforms,
                                         const size_t transform_count, const StorageRef* indices,
                                         const StorageRef* visibility, const size_t visibility_count, ExecContext) {
        LFS_FACADE_TRACE(project_points);
        struct ProjectionParams {
            std::array<float, 4> row0, row1, row2;
            uint64_t points, output, transforms, indices, visibility;
            std::array<float, 2> scale, center;
            float invalid_value, near_distance;
            uint32_t count, transform_count, visibility_count, padding;
        };
        static_assert(sizeof(ProjectionParams) == 128);
        const auto context = acquire_context();
        const auto& r = projection.rotation;
        const auto& t = projection.translation;
        std::array<float, 2> scale{projection.focal_x, projection.focal_y};
        if (projection.model == PointProjectionModel::Orthographic)
            scale = {projection.ortho_scale, projection.ortho_scale};
        else if (projection.model == PointProjectionModel::Equirectangular)
            scale = {static_cast<float>(projection.width), static_cast<float>(projection.height)};
        const ProjectionParams params{
            .row0 = {r[0], r[1], r[2], t[0]},
            .row1 = {r[3], r[4], r[5], t[1]},
            .row2 = {r[6], r[7], r[8], t[2]},
            .points = address_of(*context, points),
            .output = address_of(*context, output),
            .transforms = transforms ? address_of(*context, *transforms) : 0,
            .indices = indices ? address_of(*context, *indices) : 0,
            .visibility = visibility ? address_of(*context, *visibility) : 0,
            .scale = scale,
            .center = {projection.center_x, projection.center_y},
            .invalid_value = projection.invalid_value,
            .near_distance = projection.near_distance,
            .count = checked_u32(count, "Metal projection count exceeds uint32"),
            .transform_count = checked_u32(transform_count, "Metal transform count exceeds uint32"),
            .visibility_count = checked_u32(visibility_count, "Metal visibility count exceeds uint32"),
        };
        std::vector<StorageRef> uses{points, output};
        for (const StorageRef* const storage : {transforms, indices, visibility}) {
            if (storage)
                uses.push_back(*storage);
        }
        const uint32_t inputs = (transforms ? 1u : 0u) | (indices ? 2u : 0u) | (visibility ? 4u : 0u);
        dispatch_addressed(*context, uses,
                           context->pipeline("project_points", {{0, static_cast<uint32_t>(projection.model)}, {20, inputs}}),
                           params, count);
    }

    void MetalBackendOps::mark_points_2d(const StorageRef mask, const StorageRef points, const size_t count,
                                         const PointRegion2D& region, const StorageRef* geometry,
                                         const size_t geometry_count, ExecContext) {
        LFS_FACADE_TRACE(mark_points_2d);
        struct RegionParams {
            uint64_t mask, points, geometry;
            uint32_t count, geometry_count;
            std::array<float, 4> bounds;
            float radius_sq, minimum_coordinate;
        };
        static_assert(sizeof(RegionParams) == 56);
        const auto context = acquire_context();
        const RegionParams params{
            .mask = address_of(*context, mask),
            .points = address_of(*context, points),
            .geometry = geometry ? address_of(*context, *geometry) : 0,
            .count = checked_u32(count, "Metal region point count exceeds uint32"),
            .geometry_count = checked_u32(geometry_count, "Metal region geometry count exceeds uint32"),
            .bounds = {region.x0, region.y0, region.x1, region.y1},
            .radius_sq = region.radius * region.radius,
            .minimum_coordinate = region.minimum_coordinate,
        };
        const std::array uses{mask, points, geometry ? *geometry : points};
        dispatch_addressed(*context, uses, context->pipeline("mark_points_2d", {{0, static_cast<uint32_t>(region.kind)}}),
                           params, count);
    }

    void MetalBackendOps::filter_points(const StorageRef mask, const PointFilterProgram& program, ExecContext) {
        LFS_FACADE_TRACE(filter_points);
        struct FilterParams {
            uint64_t mask;
            std::array<uint64_t, FilterInputCount> inputs;
            std::array<float, 25> window;
            uint32_t count, transform_count, allowed_count, padding;
        };
        static_assert(sizeof(FilterParams) == 200);
        const auto context = acquire_context();
        FilterParams params{.mask = address_of(*context, mask),
                            .count = program.count,
                            .transform_count = program.transform_count,
                            .allowed_count = program.allowed_count};
        std::vector<StorageRef> uses{mask};
        for (size_t i = 0; i < program.inputs.size(); ++i) {
            if (program.inputs[i]) {
                params.inputs[i] = address_of(*context, *program.inputs[i]);
                uses.push_back(*program.inputs[i]);
            }
        }
        if (program.flags & LFS_FILTER_WINDOW)
            params.window = pointFilterWindowBlock(program.window);
        const auto model = static_cast<uint32_t>(program.window.projection.model);
        dispatch_addressed(*context, uses, context->pipeline("filter_points", {{0, model}, {21, program.flags}}), params,
                           program.count);
    }

    void MetalBackendOps::update_labels(const StorageRef output, const StorageRef selected,
                                        const LabelUpdateProgram& program, ExecContext) {
        LFS_FACADE_TRACE(update_labels);
        struct LabelParams {
            uint64_t output, selected, existing, locked, indices, categories, allowed;
            uint32_t count, output_count, allowed_count, label, mode, padding;
        };
        static_assert(sizeof(LabelParams) == 80);
        const auto context = acquire_context();
        const auto address = [&](const std::optional<StorageRef>& storage) {
            return storage ? address_of(*context, *storage) : uint64_t{0};
        };
        const LabelParams params{
            .output = address_of(*context, output),
            .selected = address_of(*context, selected),
            .existing = address(program.existing),
            .locked = address(program.locked),
            .indices = address(program.indices),
            .categories = address(program.categories),
            .allowed = address(program.allowed),
            .count = checked_u32(program.count, "Metal label row count exceeds uint32"),
            .output_count = checked_u32(program.output_count, "Metal label count exceeds uint32"),
            .allowed_count = checked_u32(program.allowed_count, "Metal label category count exceeds uint32"),
            .label = program.label,
            .mode = program.mode,
        };
        std::vector<StorageRef> uses{output, selected};
        for (const auto* input : {&program.existing, &program.locked, &program.indices, &program.categories,
                                  &program.allowed}) {
            if (*input)
                uses.push_back(**input);
        }
        const auto phase = [&](const uint32_t kind, const size_t count) {
            dispatch_addressed(*context, uses, context->pipeline("update_labels", {{0, kind}}), params, count);
        };
        if (!program.indices) {
            phase(0, program.output_count);
            return;
        }
        phase(3, program.output_count);
        if (program.mode == static_cast<uint32_t>(LabelUpdateMode::Replace))
            phase(1, program.count);
        phase(2, program.count);
    }

    void MetalBackendOps::histogram_u8(const StorageRef values, const StorageRef counts, const size_t size,
                                       ExecContext) {
        LFS_FACADE_TRACE(histogram_u8);
        struct HistogramParams {
            uint64_t input, output;
            uint32_t size, padding;
        };
        const auto context = acquire_context();
        const HistogramParams params{
            .input = address_of(*context, values),
            .output = address_of(*context, counts),
            .size = checked_u32(size, "Metal histogram size exceeds uint32"),
        };
        const std::array uses{values, counts};
        // Every thread of a threadgroup owns one of the 256 shared bins.
        context->dispatch(uses, {.pipeline = context->pipeline("histogram_u8"),
                                 .params = param_bytes(params),
                                 .grid = MTLSizeMake(std::min<NSUInteger>(thread_groups(size).width, 4096), 1, 1),
                                 .group_size = MTLSizeMake(kThreadgroupWidth, 1, 1)});
    }

    void MetalBackendOps::ppisp_apply(const StorageRef input, const StorageRef output, const int width, const int height,
                                      const PpispParams& settings, ExecContext) {
        LFS_FACADE_TRACE(ppisp_apply);
        struct PpispApplyParams {
            uint64_t input, output;
            int32_t width, height;
            PpispParams settings;
        };
        static_assert(sizeof(PpispApplyParams) == 192);
        const auto context = acquire_context();
        const PpispApplyParams params{
            .input = address_of(*context, input),
            .output = address_of(*context, output),
            .width = width,
            .height = height,
            .settings = settings,
        };
        const std::array uses{input, output};
        dispatch_addressed(*context, uses, context->pipeline("ppisp_apply"), params, size_t(width) * height);
    }

    void MetalBackendOps::environment_composite(const StorageRef rgb, const StorageRef alpha, const StorageRef environment,
                                                const StorageRef output, const EnvironmentCompositeParams& settings,
                                                ExecContext) {
        LFS_FACADE_TRACE(environment_composite);
        struct CompositeParams {
            uint64_t rgb, alpha, environment, output;
            EnvironmentCompositeParams p;
        };
        static_assert(sizeof(CompositeParams) == 128);
        const auto context = acquire_context();
        const CompositeParams params{
            .rgb = address_of(*context, rgb),
            .alpha = address_of(*context, alpha),
            .environment = address_of(*context, environment),
            .output = address_of(*context, output),
            .p = settings,
        };
        const std::array uses{rgb, alpha, environment, output};
        dispatch_addressed(*context, uses,
                           context->pipeline("environment_composite", {{0, settings.equirect_view != 0 ? 1u : 0u}}),
                           params, size_t(settings.band_width) * settings.band_height);
    }

    void MetalBackendOps::affine_splat_geometry(const StorageRef scales, const StorageRef rotations,
                                                const StorageRef out_scales, const StorageRef out_rotations,
                                                const splat_transform::LinearTransform& linear, const size_t n,
                                                ExecContext) {
        LFS_FACADE_TRACE(affine_splat_geometry);
        struct AffineSplatParams {
            uint64_t scales, rotations, out_scales, out_rotations;
            splat_transform::LinearTransform linear;
            uint32_t count;
        };
        static_assert(sizeof(AffineSplatParams) == 72);
        const auto context = acquire_context();
        const AffineSplatParams params{
            .scales = address_of(*context, scales),
            .rotations = address_of(*context, rotations),
            .out_scales = address_of(*context, out_scales),
            .out_rotations = address_of(*context, out_rotations),
            .linear = linear,
            .count = checked_u32(n, "Metal affine splat count exceeds uint32"),
        };
        const std::array uses{scales, rotations, out_scales, out_rotations};
        dispatch_addressed(*context, uses, context->pipeline("affine_splat_geometry"), params, n);
    }

    namespace {
        // The layouts of RadPageParams and RadPagePackedDesc in kernels.metal.
        struct RadPageParams {
            std::array<uint64_t, 9> regions{};
            uint64_t packed = 0, descriptor = 0;
            uint32_t page = 0, page_splats = 0, slots = 0, padding = 0;
            struct {
                uint64_t means, sh0, shN, rotation, scaling, opacity, sh_bounds;
                uint32_t offset, count, rest, half_sh, quant_sh, padding;
            } sources{};
        };
        static_assert(sizeof(RadPageParams) == 184 && sizeof(RadPagePackedDesc) == 304);

        // Runs rad_page phases over one pool page; the kernel reaches the pool
        // regions through the parameters.
        API_AVAILABLE(macos(26.0))
        void encode_rad_page(Context& context, RadPageParams params, const RadPagePool& pool,
                             std::vector<StorageRef> uses, const std::initializer_list<uint32_t> phases) {
            params.page_splats = pool.page_splats;
            params.slots = pool.sh_slots;
            for (size_t i = 0; i < pool.regions.size(); ++i) {
                if (pool.regions[i].is_valid()) {
                    uses.push_back(storage_ref(pool.regions[i]));
                    params.regions[i] = address_of(context, uses.back());
                }
            }
            for (const uint32_t phase : phases)
                dispatch_addressed(context, uses, context.pipeline("rad_page", {{0, phase}}), params, pool.page_splats);
        }
    } // namespace

    void metal_rad_page_dequant(const Tensor& packed, const RadPagePool& pool, const uint32_t page) {
        if (@available(macOS 26.0, *)) {
            const auto context = acquire_context();
            const StorageRef input = storage_ref(packed);
            const uint64_t descriptor = address_of(*context, input);
            encode_rad_page(*context, {.packed = descriptor + sizeof(RadPagePackedDesc), .descriptor = descriptor, .page = page},
                            pool, {input}, {0});
        }
    }

    void metal_rad_page_quantize(const RadPageSources& src, const RadPagePool& pool, const uint32_t page) {
        if (@available(macOS 26.0, *)) {
            const auto context = acquire_context();
            std::vector<StorageRef> uses;
            // Absent attributes, such as the SH of degree-zero splats, have no storage.
            const auto address = [&](const Tensor& tensor) {
                if (!tensor.is_valid() || tensor.bytes() == 0)
                    return uint64_t{0};
                uses.push_back(storage_ref(tensor));
                return address_of(*context, uses.back());
            };
            RadPageParams params{.page = page};
            params.sources = {.means = address(src.means),
                              .sh0 = address(src.sh0),
                              .shN = src.sh_rest ? address(src.shN) : 0,
                              .rotation = address(src.rotation),
                              .scaling = address(src.scaling),
                              .opacity = address(src.opacity),
                              .sh_bounds = address(src.shN_bounds),
                              .offset = src.offset,
                              .count = src.count,
                              .rest = src.sh_rest,
                              .half_sh = src.shN.is_valid() && src.shN.dtype() == DataType::Float16,
                              .quant_sh = src.sh_q16};
            if (params.sources.shN)
                encode_rad_page(*context, params, pool, std::move(uses), {1, 2, 3});
            else
                encode_rad_page(*context, params, pool, std::move(uses), {1, 3});
        }
    }

    Tensor MetalBackendOps::image_undistort(const Tensor& input, const UndistortParams& p, const bool mask,
                                            ExecContext) {
        LFS_FACADE_TRACE(image_undistort);
        ResampleParams params{.src_fx = p.src_fx,
                              .src_fy = p.src_fy,
                              .src_cx = p.src_cx,
                              .src_cy = p.src_cy,
                              .dst_fx = p.dst_fx,
                              .dst_fy = p.dst_fy,
                              .dst_cx = p.dst_cx,
                              .dst_cy = p.dst_cy,
                              .sw = p.src_width,
                              .sh = p.src_height,
                              .dw = p.dst_width,
                              .dh = p.dst_height,
                              .model = static_cast<int32_t>(p.model_type),
                              .num_distortion = p.num_distortion,
                              .channels = mask ? 1 : static_cast<int32_t>(input.size(0))};
        std::copy_n(p.distortion, 12, params.distortion);
        return resample_image(input, params, mask, 0);
    }

    Tensor MetalBackendOps::image_resize_prior(const Tensor& input, const int height, const int width, const bool normal,
                                               ExecContext) {
        LFS_FACADE_TRACE(image_resize_prior);
        const ResampleParams params{.sw = static_cast<int32_t>(input.size(input.ndim() - 1)),
                                    .sh = static_cast<int32_t>(input.size(input.ndim() - 2)),
                                    .dw = width,
                                    .dh = height,
                                    .channels = normal ? 3 : 1};
        return resample_image(input, params, !normal, normal ? 2 : 1);
    }

    void MetalBackendOps::sh_codec(const StorageRef source, const StorageRef destination,
                                   const ShCodecProgram& program, ExecContext) {
        LFS_FACADE_TRACE(sh_codec);
        const ShCodec& codec = program.codec;
        struct ShParams {
            uint64_t source, destination, indices, source_bounds, destination_bounds;
            uint32_t source_rows, destination_rows, count, source_offset, destination_offset, padding;
        };
        static_assert(sizeof(ShParams) == 64);
        const auto context = acquire_context();
        const auto address = [&](const std::optional<StorageRef>& storage) {
            return storage ? address_of(*context, *storage) : uint64_t{0};
        };
        const ShParams params{
            .source = address_of(*context, source),
            .destination = address_of(*context, destination),
            .indices = address(program.indices),
            .source_bounds = address(program.source_bounds),
            .destination_bounds = address(program.destination_bounds),
            .source_rows = checked_u32(codec.source_rows, "Metal SH source rows exceed uint32"),
            .destination_rows = checked_u32(codec.destination_rows, "Metal SH destination rows exceed uint32"),
            .count = checked_u32(codec.count, "Metal SH row count exceeds uint32"),
            .source_offset = checked_u32(codec.source_offset, "Metal SH source offset exceeds uint32"),
            .destination_offset = checked_u32(codec.destination_offset, "Metal SH destination offset exceeds uint32"),
        };
        std::vector<StorageRef> uses{source, destination};
        for (const auto* storage : {&program.indices, &program.source_bounds, &program.destination_bounds}) {
            if (*storage)
                uses.push_back(**storage);
        }
        const uint32_t indices = program.indices ? (program.indices->dtype == DataType::Int64 ? 2u : 1u) : 0u;
        const bool encode = codec.destination_format == ShFormat::Q16;
        const auto pipeline = context->pipeline(encode ? "sh_encode" : "sh_codec",
                                                {{7, codec.scatter ? 1u : 0u},
                                                 {22, static_cast<uint32_t>(codec.source_format)},
                                                 {23, static_cast<uint32_t>(codec.destination_format)},
                                                 {24, indices},
                                                 {25, codec.source_rest},
                                                 {26, codec.destination_rest}});
        if (encode) {
            context->dispatch(uses, {.pipeline = pipeline,
                                     .params = param_bytes(params),
                                     .grid = MTLSizeMake((codec.count + 255) / 256, 1, 1),
                                     .group_size = MTLSizeMake(256, 1, 1)});
            return;
        }
        // Tiles pad rows to 32 and cells to float4 groups.
        const size_t elements = (codec.count + 31) / 32 * 32 * ((codec.destination_rest * 3 + 3) / 4 * 4);
        dispatch_addressed(*context, uses, pipeline, params, std::min<size_t>(elements, size_t{4096} * kThreadgroupWidth));
    }

    Tensor MetalBackendOps::morton_sort(const Tensor& positions, Tensor* sorted_keys, ExecContext) {
        LFS_FACADE_TRACE(morton_sort);
        MetalExportKernels kernels;
        return export_morton_sort(kernels, positions, sorted_keys);
    }

    std::tuple<Tensor, Tensor> MetalBackendOps::kmeans_sh(const Tensor& sh, const int n_points, const int sh_coeffs,
                                                          const int k, const int iterations, bool, ExecContext) {
        LFS_FACADE_TRACE(kmeans_sh);
        MetalExportKernels kernels;
        return export_kmeans_sh(kernels, sh, n_points, sh_coeffs, k, iterations);
    }

    void MetalBackendOps::assign_sh3(const Tensor& sh, const Tensor& centroids, const Tensor& norms, Tensor& labels,
                                     const bool fast, const bool have_labels, ExecContext) {
        LFS_FACADE_TRACE(assign_sh3);
        MetalExportKernels kernels;
        export_assign_sh3(kernels, sh, centroids, norms, labels, fast, have_labels);
    }

    void MetalBackendOps::decimate_candidates(const Tensor& position, const Tensor& rotation, const Tensor& scale,
                                              const Tensor& opacity, const Tensor& dc, const Tensor& sh, const int rest,
                                              std::vector<uint32_t>& idx, std::vector<float>& cost, ExecContext) {
        LFS_FACADE_TRACE(decimate_candidates);
        MetalExportKernels kernels;
        export_decimate_candidates(kernels, position, rotation, scale, opacity, dc, sh, rest, idx, cost);
    }

    DecimateMerge MetalBackendOps::decimate_merge(const Tensor& position, const Tensor& rotation, const Tensor& scale,
                                                  const Tensor& opacity, const Tensor& dc, const Tensor& sh,
                                                  const int rest, const std::vector<int>& member_group,
                                                  const std::vector<uint32_t>& minimum,
                                                  const std::vector<uint32_t>& members,
                                                  const std::vector<uint32_t>& offsets, const size_t removed,
                                                  ExecContext) {
        LFS_FACADE_TRACE(decimate_merge);
        MetalExportKernels kernels;
        return export_decimate_merge(kernels, position, rotation, scale, opacity, dc, sh, rest, member_group, minimum,
                                     members, offsets, removed);
    }

    void MetalBackendOps::inference(const StorageRef input, const StorageRef output, const InferenceProgram& program,
                                    ExecContext) {
        LFS_FACADE_TRACE(inference);
        struct InferenceParams {
            uint64_t input, output;
            uint32_t total, step;
            InferenceGeometry geometry;
        };
        static_assert(sizeof(InferenceParams) == 112);
        const auto context = acquire_context();
        const InferenceParams params{address_of(*context, input), address_of(*context, output),
                                     checked_u32(program.count, "Metal inference output exceeds uint32"), 0,
                                     program.geometry};
        const std::array uses{input, output};
        dispatch_addressed(*context, uses, context->pipeline("inference", {{0, static_cast<uint32_t>(program.kernel)}}),
                           params, program.count);
    }

    namespace {
        // One matrix-unit GEMM of nn.metal's nn_linear. Strides and pitch are
        // in elements, and the byte offsets move a, w, bias and output within
        // their storage.
        struct LinearLaunch {
            StorageRef a, w;
            std::optional<StorageRef> bias, scale, residual;
            StorageRef output;
            size_t batch = 1, m = 0, n = 0, k = 0;
            size_t a_stride = 0, w_stride = 0, output_stride = 0, out_pitch = 0;
            size_t a_offset = 0, w_offset = 0, bias_offset = 0, output_offset = 0;
            bool trans_b = false, row_bias = false;
            int activation = 0;
        };

        API_AVAILABLE(macos(26.0))
        void dispatch_linear(Context& context, const LinearLaunch& launch) {
            struct Params {
                uint64_t a, w, bias, scale, residual, output, a_stride, w_stride, output_stride;
                uint32_t m, n, k;
                int32_t activation;
                uint32_t has_bias, has_scale, has_residual, row_bias, out_pitch, padding;
            };
            static_assert(sizeof(Params) == 112);
            const auto address = [&](const std::optional<StorageRef>& storage, const size_t offset) {
                return storage ? address_of(context, *storage) + offset : uint64_t{0};
            };
            (void)checked_extent(launch.m * launch.out_pitch, "Metal linear output exceeds int32");
            const Params params{
                .a = address_of(context, launch.a) + launch.a_offset,
                .w = address_of(context, launch.w) + launch.w_offset,
                .bias = address(launch.bias, launch.bias_offset),
                .scale = address(launch.scale, 0),
                .residual = address(launch.residual, 0),
                .output = address_of(context, launch.output) + launch.output_offset,
                .a_stride = launch.a_stride,
                .w_stride = launch.w_stride,
                .output_stride = launch.output_stride,
                .m = checked_extent(launch.m, "Metal linear rows exceed int32"),
                .n = checked_extent(launch.n, "Metal linear columns exceed int32"),
                .k = checked_extent(launch.k, "Metal linear depth exceeds int32"),
                .activation = launch.activation,
                .has_bias = launch.bias ? 1u : 0u,
                .has_scale = launch.scale ? 1u : 0u,
                .has_residual = launch.residual ? 1u : 0u,
                .row_bias = launch.row_bias ? 1u : 0u,
                .out_pitch = checked_extent(launch.out_pitch, "Metal linear pitch exceeds int32"),
                .padding = 0,
            };
            std::array<StorageRef, 6> uses{launch.a, launch.w, launch.output};
            size_t used = 3;
            for (const auto& operand : {launch.bias, launch.scale, launch.residual}) {
                if (operand)
                    uses[used++] = *operand;
            }
            const size_t tile = gemm_tile(launch.m, launch.n, launch.batch);
            // Scratch outputs carry no dtype of their own.
            const uint32_t dtype = static_cast<uint32_t>(launch.a.dtype);
            const auto pipeline = context.pipeline(
                "nn_linear", {{1, dtype}, {2, dtype}, {14, launch.trans_b ? 1u : 0u}, {27, tile == 64 ? 1u : 0u}});
            context.dispatch(std::span(uses.data(), used),
                             {.pipeline = pipeline,
                              .buffers = {},
                              .params = param_bytes(params),
                              .grid = MTLSizeMake((launch.n + tile - 1) / tile, (launch.m + tile - 1) / tile,
                                                  checked_u32(launch.batch, "Metal linear batch exceeds uint32")),
                              // The matmul runs on four SIMD groups.
                              .group_size = MTLSizeMake(4 * pipeline.threadExecutionWidth, 1, 1)});
        }
    } // namespace

    void MetalBackendOps::nn_linear(const StorageRef a, const StorageRef w, const std::optional<StorageRef> bias,
                                    const std::optional<StorageRef> scale, const std::optional<StorageRef> residual,
                                    const StorageRef output, const LinearProgram& program, ExecContext) {
        LFS_FACADE_TRACE(nn_linear);
        if (program.batch == 0 || program.m == 0 || program.n == 0)
            return;
        LFS_ASSERT_MSG(program.k > 0, "Metal linear layers need a positive inner dimension");
        const auto context = acquire_context();
        dispatch_linear(*context, {.a = a,
                                   .w = w,
                                   .bias = bias,
                                   .scale = scale,
                                   .residual = residual,
                                   .output = output,
                                   .batch = program.batch,
                                   .m = program.m,
                                   .n = program.n,
                                   .k = program.k,
                                   .a_stride = program.m * program.k,
                                   .w_stride = program.batched_b ? program.n * program.k : 0,
                                   .output_stride = program.m * program.n,
                                   .out_pitch = program.n,
                                   .trans_b = program.trans_b,
                                   .activation = program.activation});
    }

    namespace {
        // Transposed convolution: per group, GEMMs of the weights, transposed
        // to [out][ky][kx][in], with the images [in][pixels] give columns that
        // nn_col2im gathers into the output, adding the bias. The columns of
        // as many images as fit 64 MiB go at once.
        API_AVAILABLE(macos(26.0))
        void encode_conv_transpose(Context& context, const StorageRef input, const StorageRef weight,
                              const std::optional<StorageRef> bias, const StorageRef output,
                              const ConvProgram& program) {
            const InferenceGeometry& g = program.geometry;
            const size_t in_group = static_cast<size_t>(g.channels), out_group = program.out_channels / program.groups;
            const size_t plane = static_cast<size_t>(g.height) * static_cast<size_t>(g.width);
            const size_t pixels = static_cast<size_t>(g.out_height) * static_cast<size_t>(g.out_width);
            const size_t rows = out_group * static_cast<size_t>(g.kernel_h) * static_cast<size_t>(g.kernel_w);
            const size_t element = dtype_size(output.dtype);
            const size_t in_batch = in_group * program.groups * plane, out_batch = program.out_channels * pixels;
            (void)checked_u32(program.batch * std::max(in_batch, out_batch), "Metal convolution exceeds uint32");
            constexpr size_t kColumnBytes = size_t{64} << 20;
            const size_t images = std::clamp<size_t>(kColumnBytes / (rows * plane * element), 1, program.batch);
            const Scratch columns(context, images * rows * plane * element);
            struct Col2imParams {
                uint64_t columns, bias, output;
                uint32_t pixels, channel0, out_batch, columns_batch;
                int32_t activation;
                uint32_t has_bias;
                InferenceGeometry geometry;
            };
            static_assert(sizeof(Col2imParams) == 136);
            const uint32_t dtype = static_cast<uint32_t>(output.dtype);
            const auto col2im = context.pipeline("nn_col2im", {{1, dtype}, {2, dtype}});
            for (size_t first = 0; first < program.batch; first += images) {
                const size_t count = std::min(images, program.batch - first);
                for (size_t group = 0; group < program.groups; ++group) {
                    dispatch_linear(context, {.a = weight,
                                              .w = input,
                                              .output = columns.storage,
                                              .batch = count,
                                              .m = rows,
                                              .n = plane,
                                              .k = in_group,
                                              .w_stride = in_batch,
                                              .output_stride = rows * plane,
                                              .out_pitch = plane,
                                              .a_offset = group * in_group * rows * element,
                                              .w_offset = (first * in_batch + group * in_group * plane) * element});
                    const Col2imParams params{
                        .columns = address_of(context, columns.storage),
                        .bias = bias ? address_of(context, *bias) : 0,
                        .output = address_of(context, output) + first * out_batch * element,
                        .pixels = static_cast<uint32_t>(pixels),
                        .channel0 = static_cast<uint32_t>(group * out_group),
                        .out_batch = static_cast<uint32_t>(out_batch),
                        .columns_batch = static_cast<uint32_t>(rows * plane),
                        .activation = program.activation,
                        .has_bias = bias ? 1u : 0u,
                        .geometry = g};
                    const std::array uses{columns.storage, bias.value_or(output), output};
                    context.dispatch(uses, {.pipeline = col2im,
                                            .buffers = {},
                                            .params = param_bytes(params),
                                            .grid = MTLSizeMake(pixels, out_group, count)});
                }
            }
        }
    } // namespace

    // Convolution as GEMMs of each group's weights [out][taps] with the
    // image's patches [taps][pixels], written straight into NCHW rows with
    // the bias per row. 1x1 convolutions multiply the image itself; others
    // build the patches in bounded chunks of output pixels.
    void MetalBackendOps::nn_conv2d(const StorageRef input, const StorageRef weight,
                                    const std::optional<StorageRef> bias, const StorageRef output,
                                    const ConvProgram& program, ExecContext) {
        LFS_FACADE_TRACE(nn_conv2d);
        const InferenceGeometry& g = program.geometry;
        const size_t pixels = static_cast<size_t>(g.out_height) * static_cast<size_t>(g.out_width);
        if (program.batch == 0 || program.out_channels == 0 || pixels == 0)
            return;
        const size_t in_group = static_cast<size_t>(g.channels), out_group = program.out_channels / program.groups;
        const size_t taps = in_group * static_cast<size_t>(g.kernel_h) * static_cast<size_t>(g.kernel_w);
        const size_t plane = static_cast<size_t>(g.height) * static_cast<size_t>(g.width);
        const size_t in_batch = in_group * program.groups * plane, out_batch = program.out_channels * pixels;
        const size_t element = dtype_size(output.dtype);
        (void)checked_u32(program.batch * in_batch, "Metal convolution input exceeds uint32");
        const auto context = acquire_context();
        if (program.transpose) {
            encode_conv_transpose(*context, input, weight, bias, output, program);
            return;
        }
        LinearLaunch launch{.a = weight,
                            .w = input,
                            .bias = bias,
                            .output = output,
                            .batch = program.batch,
                            .m = out_group,
                            .k = taps,
                            .output_stride = out_batch,
                            .out_pitch = pixels,
                            .row_bias = true,
                            .activation = program.activation};
        const bool pointwise = g.kernel_h == 1 && g.kernel_w == 1 && g.stride_h == 1 && g.stride_w == 1 &&
                               g.pad_h == 0 && g.pad_w == 0;
        if (pointwise) {
            for (size_t group = 0; group < program.groups; ++group) {
                launch.n = pixels;
                launch.w_stride = in_batch;
                launch.a_offset = group * out_group * taps * element;
                launch.w_offset = group * in_group * plane * element;
                launch.bias_offset = group * out_group * element;
                launch.output_offset = group * out_group * pixels * element;
                dispatch_linear(*context, launch);
            }
            return;
        }
        // Patches for at most 64 MiB, in whole 32-pixel tiles.
        constexpr size_t kPatchBytes = size_t{64} << 20;
        const size_t per_pixel = program.batch * taps * element;
        const size_t chunk = std::min(pixels, std::max<size_t>(32, kPatchBytes / per_pixel / 32 * 32));
        const Scratch patches(*context, chunk * per_pixel);
        struct Im2colParams {
            uint64_t input, columns;
            uint32_t input_batch, count, taps, first, channel0, padding;
            InferenceGeometry geometry;
        };
        static_assert(sizeof(Im2colParams) == 128);
        const uint32_t dtype = static_cast<uint32_t>(output.dtype);
        const auto im2col = context->pipeline("nn_im2col", {{1, dtype}, {2, dtype}});
        launch.w = patches.storage;
        for (size_t group = 0; group < program.groups; ++group) {
            for (size_t first = 0; first < pixels; first += chunk) {
                const size_t count = std::min(chunk, pixels - first);
                (void)checked_u32(program.batch * taps * count, "Metal convolution patches exceed uint32");
                const Im2colParams params{.input = address_of(*context, input),
                                          .columns = address_of(*context, patches.storage),
                                          .input_batch = static_cast<uint32_t>(in_batch),
                                          .count = static_cast<uint32_t>(count),
                                          .taps = static_cast<uint32_t>(taps),
                                          .first = static_cast<uint32_t>(first),
                                          .channel0 = static_cast<uint32_t>(group * in_group),
                                          .padding = 0,
                                          .geometry = g};
                const std::array uses{input, patches.storage};
                context->dispatch(uses, {.pipeline = im2col,
                                         .buffers = {},
                                         .params = param_bytes(params),
                                         .grid = MTLSizeMake(count, in_group, program.batch)});
                launch.n = count;
                launch.w_stride = taps * count;
                launch.a_offset = group * out_group * taps * element;
                launch.bias_offset = group * out_group * element;
                launch.output_offset = (group * out_group * pixels + first) * element;
                dispatch_linear(*context, launch);
            }
        }
    }

    void MetalBackendOps::nn_attention(const StorageRef q, const StorageRef k, const StorageRef v,
                                       const std::optional<StorageRef> mask, const StorageRef output,
                                       const AttentionProgram& program, ExecContext) {
        LFS_FACADE_TRACE(nn_attention);
        if (program.groups == 0 || program.queries == 0)
            return;
        LFS_ASSERT_MSG(program.dim > 0 && program.dim <= 128, "Metal attention supports head dims up to 128");
        struct Params {
            uint64_t q, k, v, mask, output;
            int64_t mask_batch, mask_head, mask_query, mask_key;
            uint64_t partial;
            uint32_t heads, queries, keys, dim;
            float scale;
            uint32_t has_mask, key_split, groups;
        };
        static_assert(sizeof(Params) == 112);
        const auto context = acquire_context();
        // Four SIMD groups of 16 queries (kNnAttentionQueries, kNnAttentionSimds).
        constexpr size_t kQueries = 64, kSimdQueries = 16;
        // Too few query blocks to fill the GPU split the keys, in whole
        // 32-key blocks, until there are about kBusySimds SIMD groups.
        constexpr size_t kBusySimds = 1024, kMinSplitKeys = 128;
        const size_t simds = (program.queries + kSimdQueries - 1) / kSimdQueries * program.groups;
        const size_t splits = std::clamp<size_t>(std::min(kBusySimds / simds, program.keys / kMinSplitKeys), 1, 64);
        const size_t key_split = ((program.keys + splits - 1) / splits + 31) & ~size_t{31};
        const size_t used_splits = (program.keys + key_split - 1) / key_split;
        std::optional<Scratch> partial;
        if (used_splits > 1)
            partial.emplace(*context, used_splits * program.groups * program.queries * (program.dim + 2) * sizeof(float));
        const Params params{
            .q = address_of(*context, q),
            .k = address_of(*context, k),
            .v = address_of(*context, v),
            .mask = mask ? address_of(*context, *mask) : 0,
            .output = address_of(*context, output),
            .mask_batch = program.mask_strides[0],
            .mask_head = program.mask_strides[1],
            .mask_query = program.mask_strides[2],
            .mask_key = program.mask_strides[3],
            .partial = partial ? address_of(*context, partial->storage) : 0,
            .heads = checked_u32(program.heads, "Metal attention heads exceed uint32"),
            .queries = checked_u32(program.queries, "Metal attention queries exceed uint32"),
            .keys = checked_u32(program.keys, "Metal attention keys exceed uint32"),
            .dim = static_cast<uint32_t>(program.dim),
            .scale = program.scale,
            .has_mask = mask ? 1u : 0u,
            .key_split = static_cast<uint32_t>(key_split),
            .groups = checked_u32(program.groups, "Metal attention groups exceed uint32"),
        };
        std::array<StorageRef, 6> uses{q, k, v, output, output, output};
        if (mask)
            uses[4] = *mask;
        if (partial)
            uses[5] = partial->storage;
        const uint32_t dtype = static_cast<uint32_t>(output.dtype);
        const auto pipeline = context->pipeline(
            "nn_attention", {{1, dtype}, {2, dtype}, {12, program.dim <= 64 ? 64u : 128u}, {28, partial ? 1u : 0u}});
        context->dispatch(uses, {.pipeline = pipeline,
                                 .buffers = {},
                                 .params = param_bytes(params),
                                 .grid = MTLSizeMake((program.queries + kQueries - 1) / kQueries, params.groups,
                                                     used_splits),
                                 .group_size = MTLSizeMake(4 * pipeline.threadExecutionWidth, 1, 1)});
        if (partial) {
            context->dispatch(uses, {.pipeline = context->pipeline("nn_attention_combine", {{1, dtype}, {2, dtype}}),
                                     .buffers = {},
                                     .params = param_bytes(params),
                                     .grid = MTLSizeMake(program.dim, program.queries, program.groups)});
        }
    }

    void MetalBackendOps::nn_norm(const StorageRef input, const StorageRef weight, const std::optional<StorageRef> bias,
                                  const StorageRef output, const NormProgram& program, ExecContext) {
        LFS_FACADE_TRACE(nn_norm);
        if (program.rows == 0)
            return;
        struct Params {
            uint64_t input, weight, bias, output;
            uint32_t rows, cols;
            float eps;
            uint32_t has_bias;
        };
        static_assert(sizeof(Params) == 48);
        const auto context = acquire_context();
        const Params params{
            .input = address_of(*context, input),
            .weight = address_of(*context, weight),
            .bias = bias ? address_of(*context, *bias) : 0,
            .output = address_of(*context, output),
            .rows = checked_u32(program.rows, "Metal norm rows exceed uint32"),
            .cols = checked_u32(program.cols, "Metal norm columns exceed uint32"),
            .eps = program.eps,
            .has_bias = bias ? 1u : 0u,
        };
        (void)checked_u32(program.rows * program.cols, "Metal norm input exceeds uint32");
        const std::array uses{input, weight, bias.value_or(weight), output};
        const uint32_t dtype = static_cast<uint32_t>(output.dtype);
        // A SIMD group per row, eight rows per threadgroup.
        context->dispatch(uses, {.pipeline = context->pipeline("nn_norm", {{1, dtype}, {2, dtype}}),
                                 .buffers = {},
                                 .params = param_bytes(params),
                                 .grid = MTLSizeMake((program.rows + 7) / 8, 1, 1),
                                 .group_size = MTLSizeMake(256, 1, 1)});
    }

    void MetalBackendOps::reduce(const StorageRef input, const StorageRef output, const StridedLayout& input_layout,
                                 const ReduceProgram& program, ExecContext) {
        LFS_FACADE_TRACE(reduce);
        LFS_ASSERT_MSG(program.axis_count <= MAX_TENSOR_RANK && input_layout.rank <= MAX_TENSOR_RANK,
                       "reduction axis count exceeds MAX_TENSOR_RANK");
        LFS_ASSERT_MSG(output.dtype == program.result_dtype,
                       "Metal reduction output storage dtype does not match the program");
        if (input_layout.element_count == 0)
            return;
        const auto context = acquire_context();
        const ReduceSource source{.input = input, .code = element_code(input.dtype)};
        const size_t rank = input_layout.rank;
        if (program.axis_count == 0 || program.axis_count == rank) {
            reduce_full(*context, program.op, source, input_layout.element_count, output, program.result_dtype);
            return;
        }
        std::array<int, MAX_TENSOR_RANK> axes{};
        std::copy_n(program.axes.begin(), program.axis_count, axes.begin());
        std::sort(axes.begin(), axes.begin() + program.axis_count);
        uint32_t reduced_mask = 0;
        bool contiguous_run = true;
        for (size_t i = 0; i < program.axis_count; ++i) {
            LFS_ASSERT_MSG(axes[i] >= 0 && axes[i] < static_cast<int>(rank), "reduction axis is out of range");
            reduced_mask |= 1u << static_cast<unsigned>(axes[i]);
            contiguous_run = contiguous_run && (i == 0 || axes[i] == axes[i - 1] + 1);
        }
        if (!contiguous_run) {
            reduce_general(*context, program.op, input, output, program.result_dtype, input_layout, reduced_mask);
            return;
        }
        const auto first = static_cast<size_t>(axes[0]);
        const auto last = static_cast<size_t>(axes[program.axis_count - 1]);
        size_t outer = 1, reduce = 1, inner = 1;
        for (size_t axis = 0; axis < rank; ++axis)
            (axis < first ? outer : axis <= last ? reduce : inner) *= input_layout.dims[axis];
        reduce_axes(*context, program.op, source, output, program.result_dtype, outer, reduce, inner);
    }

    void MetalBackendOps::column_reduce(const StorageRef input, const StorageRef output, const size_t rows,
                                        const size_t columns, const ReduceProgram& program, ExecContext) {
        LFS_FACADE_TRACE(column_reduce);
        reduce_axes(*acquire_context(), program.op, {.input = input, .code = element_code(DataType::Float32)}, output,
                    DataType::Float32, 1, rows, columns);
    }

    void MetalBackendOps::strided_reduce(const StorageRef input, const StorageRef output, const size_t outer_size,
                                         const size_t reduce_size, const size_t inner_size,
                                         const ReduceProgram& program, ExecContext) {
        LFS_FACADE_TRACE(strided_reduce);
        reduce_axes(*acquire_context(), program.op, {.input = input, .code = element_code(DataType::Float32)}, output,
                    DataType::Float32, outer_size, reduce_size, inner_size);
    }

    void MetalBackendOps::fused_transform_reduce(const StorageRef input, const StorageRef output, const size_t count,
                                                 const tensor_ops::FusedPointwiseOpChain& chain,
                                                 const ReduceProgram& program,
                                                 const std::span<const StorageRef> rhs_storages, ExecContext) {
        LFS_FACADE_TRACE(fused_transform_reduce);
        if (count == 0)
            return;
        const auto context = acquire_context();
        const FusedChain fused = fused_chain(*context, chain);
        reduce_full(*context, program.op,
                    {.input = input, .code = element_code(DataType::Float32), .chain = &fused, .operands = rhs_storages},
                    count, output, DataType::Float32);
    }

    void MetalBackendOps::fused_segmented_transform_reduce(
        const StorageRef input, const StorageRef output, const size_t segment_count, const size_t segment_size,
        const tensor_ops::FusedPointwiseOpChain& chain, const ReduceProgram& program,
        const std::span<const StorageRef> rhs_storages, ExecContext) {
        LFS_FACADE_TRACE(fused_segmented_transform_reduce);
        if (segment_count == 0 || segment_size == 0)
            return;
        const auto context = acquire_context();
        const FusedChain fused = fused_chain(*context, chain);
        reduce_axes(*context, program.op,
                    {.input = input, .code = element_code(DataType::Float32), .chain = &fused, .operands = rhs_storages},
                    output, DataType::Float32, segment_count, segment_size, 1);
    }

    size_t MetalBackendOps::count_nonzero_bool(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(count_nonzero_bool);
        return count_matches(0, input, count);
    }

    size_t MetalBackendOps::count_nonzero_float(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(count_nonzero_float);
        return count_matches(1, input, count);
    }

    bool MetalBackendOps::has_nan(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(has_nan);
        return count_matches(2, input, count) != 0;
    }

    bool MetalBackendOps::has_inf(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(has_inf);
        return count_matches(3, input, count) != 0;
    }

    void MetalBackendOps::cumsum(const StorageRef data, const StridedLayout& layout, const int dim, ExecContext) {
        LFS_FACADE_TRACE(cumsum);
        LFS_ASSERT_MSG(data.dtype == DataType::Float32 || data.dtype == DataType::Int32,
                       "Metal cumsum supports Float32 and Int32");
        LFS_ASSERT_MSG(dim >= 0 && static_cast<size_t>(dim) < layout.rank, "cumsum dimension is out of range");
        size_t outer = 1, inner = 1;
        for (size_t axis = 0; axis < layout.rank; ++axis) {
            if (axis < static_cast<size_t>(dim))
                outer *= layout.dims[axis];
            else if (axis > static_cast<size_t>(dim))
                inner *= layout.dims[axis];
        }
        const size_t size = layout.dims[static_cast<size_t>(dim)];
        if (outer * inner == 0 || size == 0)
            return;
        encode_scan(*acquire_context(), data, outer, size, inner);
    }

    void MetalBackendOps::sgemm(const StorageRef lhs, const StorageRef rhs, const StorageRef output,
                                const GemmProgram& program, ExecContext) {
        LFS_FACADE_TRACE(sgemm);
        encode_gemm(lhs, rhs, nullptr, output, program, false);
    }

    void MetalBackendOps::sgemm_tn(const StorageRef lhs, const StorageRef rhs, const StorageRef output,
                                   const GemmProgram& program, ExecContext) {
        LFS_FACADE_TRACE(sgemm_tn);
        encode_gemm(lhs, rhs, nullptr, output, program, true);
    }

    void MetalBackendOps::sgemm_batched(const StorageRef lhs, const StorageRef rhs, const StorageRef output,
                                        const GemmProgram& program, ExecContext) {
        LFS_FACADE_TRACE(sgemm_batched);
        encode_gemm(lhs, rhs, nullptr, output, program, false);
    }

    void MetalBackendOps::sgemm_bias_relu(const StorageRef lhs, const StorageRef rhs, const StorageRef bias,
                                          const StorageRef output, const GemmProgram& program, ExecContext) {
        LFS_FACADE_TRACE(sgemm_bias_relu);
        encode_gemm(lhs, rhs, &bias, output, program, false);
    }

    // A sum of lhs * rhs: the product is a one-op fused chain on the reduction.
    void MetalBackendOps::dot_product(const StorageRef lhs, const StorageRef rhs, const StorageRef output,
                                      const size_t count, ExecContext) {
        LFS_FACADE_TRACE(dot_product);
        LFS_ASSERT_MSG(lhs.dtype == DataType::Float32 && rhs.dtype == DataType::Float32 &&
                           output.dtype == DataType::Float32,
                       "Metal dot product requires Float32");
        const auto context = acquire_context();
        if (count == 0) {
            encode_fill(*context, output, sizeof(float), 0, sizeof(float));
            return;
        }
        const auto rhs_at = context->locate(rhs);
        FusedChain product{.length = 1, .kinds = {6}};
        product.ops[0].rhs_address = rhs_at.address + rhs_at.offset;
        const std::array operands{rhs};
        reduce_full(*context, ReduceOp::Sum,
                    {.input = lhs, .code = element_code(DataType::Float32), .chain = &product, .operands = operands},
                    count, output, DataType::Float32);
    }

    void MetalBackendOps::diag(const StorageRef diagonal, const StorageRef output, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(diag);
        encode_matrix_fill(1, &diagonal, output, count, count);
    }

    void MetalBackendOps::eye(const StorageRef output, const size_t rows, const size_t columns, ExecContext) {
        LFS_FACADE_TRACE(eye);
        encode_matrix_fill(0, nullptr, output, rows, columns);
    }

    void MetalBackendOps::cdist(const StorageRef lhs, const StorageRef rhs, const StorageRef output,
                                const size_t lhs_rows, const size_t rhs_rows, const size_t columns, const float p,
                                ExecContext) {
        LFS_FACADE_TRACE(cdist);
        const size_t count = lhs_rows * rhs_rows;
        if (count == 0)
            return;
        LFS_ASSERT_MSG(lhs.dtype == DataType::Float32 && rhs.dtype == DataType::Float32 &&
                           output.dtype == DataType::Float32,
                       "Metal cdist requires Float32");
        struct CdistParams {
            uint64_t lhs_offset;
            uint64_t rhs_offset;
            uint64_t output_offset;
            uint32_t rows;
            uint32_t columns;
            uint32_t features;
            float p;
        };
        const auto context = acquire_context();
        const auto lhs_at = context->locate(lhs);
        const auto rhs_at = context->locate(rhs);
        const auto output_at = context->locate(output);
        const CdistParams params{
            .lhs_offset = lhs_at.offset,
            .rhs_offset = rhs_at.offset,
            .output_offset = output_at.offset,
            .rows = checked_u32(lhs_rows, "Metal cdist rows exceed uint32"),
            .columns = checked_u32(rhs_rows, "Metal cdist columns exceed uint32"),
            .features = checked_u32(columns, "Metal cdist features exceed uint32"),
            .p = p,
        };
        checked_u32(count, "Metal cdist output count exceeds uint32");
        const std::array uses{lhs, rhs, output};
        context->dispatch(uses, {.pipeline = context->pipeline("cdist"),
                                 .buffers = {lhs_at.address, rhs_at.address, output_at.address},
                                 .params = param_bytes(params),
                                 .grid = threads(count)});
    }

    void MetalBackendOps::max_pool2d(const StorageRef input, const StorageRef output,
                                     const PoolProgram& program, ExecContext) {
        LFS_FACADE_TRACE(max_pool2d);
        LFS_ASSERT_MSG(program.kernel_size > 0 && program.stride > 0,
                       "Metal max_pool2d requires a positive kernel and stride");
        encode_nn(kMaxPool, input, nullptr, output, pool_params(program));
    }

    void MetalBackendOps::adaptive_avg_pool2d(const StorageRef input, const StorageRef output,
                                              const PoolProgram& program, ExecContext) {
        LFS_FACADE_TRACE(adaptive_avg_pool2d);
        LFS_ASSERT_MSG(program.output_height > 0 && program.output_width > 0,
                       "Metal adaptive_avg_pool2d requires a positive output size");
        encode_nn(kAdaptiveAvgPool, input, nullptr, output, pool_params(program));
    }

    void MetalBackendOps::bias_add(const StorageRef input, const StorageRef bias, const StorageRef output,
                                   const int count, const int channels, const int spatial_size, ExecContext) {
        LFS_FACADE_TRACE(bias_add);
        encode_nn(kBiasAdd, input, &bias, output, bias_params(count, channels, spatial_size));
    }

    void MetalBackendOps::bias_relu(const StorageRef input, const StorageRef bias, const StorageRef output,
                                    const int count, const int channels, const int spatial_size, ExecContext) {
        LFS_FACADE_TRACE(bias_relu);
        encode_nn(kBiasRelu, input, &bias, output, bias_params(count, channels, spatial_size));
    }

    void MetalBackendOps::relu(const StorageRef input, const StorageRef output, const int count, ExecContext) {
        LFS_FACADE_TRACE(relu);
        encode_nn(kRelu, input, nullptr, output, NnParams{.total = checked_dimension(count, "relu element count")});
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
