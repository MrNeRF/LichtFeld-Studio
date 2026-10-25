/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "integer_reduce_api.h"
namespace {
    template <class Out, unsigned Op>
    struct IntegerReduction {
        using Accumulator = uint64_t;
        __device__ uint64_t identity() const {
            if constexpr (Op == LFS_HIP_REDUCE_MAX)
                return static_cast<uint64_t>(std::numeric_limits<Out>::lowest());
            if constexpr (Op == LFS_HIP_REDUCE_MIN)
                return static_cast<uint64_t>(std::numeric_limits<Out>::max());
            return Op == LFS_HIP_REDUCE_ALL ? 1 : 0;
        }
        __device__ uint64_t operator()(uint64_t a, uint64_t b) const {
            if constexpr (Op == LFS_HIP_REDUCE_MAX)
                return static_cast<uint64_t>(lfs::core::ops::maximum_op{}(static_cast<Out>(a), static_cast<Out>(b)));
            if constexpr (Op == LFS_HIP_REDUCE_MIN)
                return static_cast<uint64_t>(lfs::core::ops::minimum_op{}(static_cast<Out>(a), static_cast<Out>(b)));
            if constexpr (Op == LFS_HIP_REDUCE_ANY)
                return a | b;
            if constexpr (Op == LFS_HIP_REDUCE_ALL)
                return a & b;
            return a + b;
        }
    };
    template <class In, class Out, unsigned Op>
    struct IntegerLayoutReduction {
        const In* input;
        LfsHipLayout kept, reduced;
        __device__ uint64_t operator()(size_t row, size_t index) const {
            const In value = input[layout_index(row, kept) + layout_index(index, reduced)];
            if constexpr (Op == LFS_HIP_REDUCE_ANY || Op == LFS_HIP_REDUCE_ALL)
                return value != In(0);
            else
                return static_cast<uint64_t>(value);
        }
    };
} // namespace
extern "C" int lfs_hip_reduce_typed_layout(LfsHipContext* context, LfsHipView input, LfsHipView output, const LfsHipLayout* source, uint32_t axes, uint32_t operation, uint32_t input_dtype, uint32_t output_dtype, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && source && (operation == LFS_HIP_REDUCE_SUM || operation == LFS_HIP_REDUCE_MEAN || operation == LFS_HIP_REDUCE_MAX || operation == LFS_HIP_REDUCE_MIN || operation == LFS_HIP_REDUCE_ANY || operation == LFS_HIP_REDUCE_ALL), "Invalid HIP typed axis reduction");
        const bool logical = operation == LFS_HIP_REDUCE_ANY || operation == LFS_HIP_REDUCE_ALL;
        const bool half_numeric = !logical && input_dtype == LFS_HIP_F16;
        require(operation != LFS_HIP_REDUCE_MEAN || half_numeric, "HIP integer mean is unsupported");
        require(logical ? output_dtype == LFS_HIP_BOOL : half_numeric ? (output_dtype == LFS_HIP_F16 || output_dtype == LFS_HIP_F32)
                                                                      : ((input_dtype == LFS_HIP_I32 && (output_dtype == LFS_HIP_I32 || output_dtype == LFS_HIP_I64)) || (input_dtype == LFS_HIP_I64 && output_dtype == LFS_HIP_I64) || (input_dtype == LFS_HIP_U32 && (output_dtype == LFS_HIP_U32 || output_dtype == LFS_HIP_I64)) || (input_dtype == LFS_HIP_U8 && (output_dtype == LFS_HIP_U8 || output_dtype == LFS_HIP_I64)) || (input_dtype == LFS_HIP_BOOL && ((operation == LFS_HIP_REDUCE_SUM && output_dtype == LFS_HIP_I64) || (operation != LFS_HIP_REDUCE_SUM && output_dtype == LFS_HIP_BOOL)))),
                "Unsupported HIP typed reduction dtypes");
        const size_t iw = typed_bytes(input_dtype, 1), ow = typed_bytes(output_dtype, 1);
        const auto input_bytes = layout_span(*source, unsigned(iw), false);
        require((axes >> source->rank) == 0, "HIP typed reduction axis exceeds rank");
        if (!axes)
            axes = (1u << source->rank) - 1;
        LfsHipLayout kept{}, reduced{};
        split_reduction_layout(*source, axes, kept, reduced);
        require(!kept.elements || reduced.elements || (operation != LFS_HIP_REDUCE_MAX && operation != LFS_HIP_REDUCE_MIN), "HIP integer extrema require a nonempty axis");
        const auto output_bytes = typed_bytes(output_dtype, kept.elements);
        validate(context, input.buffer, input.offset, input_bytes);
        validate(context, output.buffer, output.offset, output_bytes);
        require(input.offset % iw == 0 && output.offset % ow == 0, "Unaligned HIP typed reduction");
        require(input.buffer != output.buffer || !input_bytes || !output_bytes || input.offset >= output.offset + output_bytes || output.offset >= input.offset + input_bytes, "HIP typed reduction input/output overlap");
        if (!kept.elements)
            return;
        DeviceScope selected(context->device);
        const void* in = static_cast<char*>(input.buffer->data) + input.offset;
        void* out = static_cast<char*>(output.buffer->data) + output.offset;
        if (half_numeric) {
            ensure_reduction_workspace(context);
            const auto dispatch = [&]<class O>() {
                switch (operation) {
                case LFS_HIP_REDUCE_SUM: launch_axis<LFS_HIP_REDUCE_SUM>(context, static_cast<const __half*>(in), static_cast<O*>(out), kept, reduced, false); break;
                case LFS_HIP_REDUCE_MEAN: launch_axis<LFS_HIP_REDUCE_SUM>(context, static_cast<const __half*>(in), static_cast<O*>(out), kept, reduced, true); break;
                case LFS_HIP_REDUCE_MAX: launch_axis<LFS_HIP_REDUCE_MAX>(context, static_cast<const __half*>(in), static_cast<O*>(out), kept, reduced, false); break;
                case LFS_HIP_REDUCE_MIN: launch_axis<LFS_HIP_REDUCE_MIN>(context, static_cast<const __half*>(in), static_cast<O*>(out), kept, reduced, false); break;
                }
            };
            if (output_dtype == LFS_HIP_F16)
                dispatch.template operator()<__half>();
            else
                dispatch.template operator()<float>();
            return;
        }
        auto launch = [&]<class I, class O, unsigned Op>() {
            using Reader = IntegerLayoutReduction<I, O, Op>;
            using Store = ReductionStore<O, false>;
            hipLaunchKernelGGL((block_reduction<Reader, IntegerReduction<O, Op>, Store>), dim3(unsigned(std::min<size_t>(kept.elements, 65535))), dim3(256), 0, context->stream,
                               (Reader{static_cast<const I*>(in), kept, reduced}), (Store{static_cast<O*>(out)}), kept.elements, reduced.elements, 1u);
        };
        auto numeric = [&]<class I, class O>() {
            switch (operation) {
            case LFS_HIP_REDUCE_SUM: launch.template operator()<I, O, LFS_HIP_REDUCE_SUM>(); break;
            case LFS_HIP_REDUCE_MAX: launch.template operator()<I, O, LFS_HIP_REDUCE_MAX>(); break;
            case LFS_HIP_REDUCE_MIN: launch.template operator()<I, O, LFS_HIP_REDUCE_MIN>(); break;
            }
        };
        if (logical) {
            auto dispatch = [&]<class I>() {if(operation==LFS_HIP_REDUCE_ANY)launch.template operator()<I,uint8_t,LFS_HIP_REDUCE_ANY>();else launch.template operator()<I,uint8_t,LFS_HIP_REDUCE_ALL>(); };
            switch (input_dtype) {
            case LFS_HIP_F32: dispatch.template operator()<float>(); break;
            case LFS_HIP_F16: dispatch.template operator()<__half>(); break;
            case LFS_HIP_I32: dispatch.template operator()<int32_t>(); break;
            case LFS_HIP_I64: dispatch.template operator()<int64_t>(); break;
            case LFS_HIP_U32: dispatch.template operator()<uint32_t>(); break;
            case LFS_HIP_U8:
            case LFS_HIP_BOOL: dispatch.template operator()<uint8_t>(); break;
            default: throw std::invalid_argument("Invalid HIP logical reduction dtype");
            }
        } else if (input_dtype == LFS_HIP_I32) {
            if (output_dtype == LFS_HIP_I32)
                numeric.template operator()<int32_t, int32_t>();
            else
                numeric.template operator()<int32_t, int64_t>();
        } else if (input_dtype == LFS_HIP_I64)
            numeric.template operator()<int64_t, int64_t>();
        else if (input_dtype == LFS_HIP_U32) {
            if (output_dtype == LFS_HIP_U32)
                numeric.template operator()<uint32_t, uint32_t>();
            else
                numeric.template operator()<uint32_t, int64_t>();
        } else if (input_dtype == LFS_HIP_U8 && output_dtype == LFS_HIP_U8)
            numeric.template operator()<uint8_t, uint8_t>();
        else if (output_dtype == LFS_HIP_BOOL)
            numeric.template operator()<uint8_t, uint8_t>();
        else
            numeric.template operator()<uint8_t, int64_t>();
        check(hipGetLastError());
    });
}
