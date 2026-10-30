/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "pointwise_catalog.hpp"
#include "runtime_common.hpp"
namespace {
    template <unsigned Mode, class In, class Out, class Op>
    __global__ void pointwise_kernel(const In* a, const In* b, Out* out, size_t count, float scalar, Op op) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            const float x = static_cast<float>(a[i]);
            float value;
            if constexpr (Mode == 0)
                value = op(x);
            else if constexpr (Mode == 1)
                value = op(x, scalar);
            else
                value = op(x, static_cast<float>(b[i]));
            if constexpr (std::is_same_v<Out, __half>)
                out[i] = __float2half_rn(value);
            else
                out[i] = static_cast<Out>(value);
        }
    }
    template <unsigned Mode, class Out, class In = float, class Op>
    void launch_pointwise(LfsHipContext* context, LfsHipView a, LfsHipView b, LfsHipView out, size_t count, float scalar, Op op) {
        require(count <= SIZE_MAX / sizeof(In) && count <= SIZE_MAX / sizeof(Out), "Pointwise element count overflows");
        const size_t input_bytes = count * sizeof(In);
        validate(context, a.buffer, a.offset, input_bytes);
        validate(context, out.buffer, out.offset, count * sizeof(Out));
        require(a.offset % alignof(In) == 0 && out.offset % alignof(Out) == 0, "Unaligned pointwise view");
        if constexpr (Mode == 2) {
            validate(context, b.buffer, b.offset, input_bytes);
            require(b.offset % alignof(In) == 0, "Unaligned pointwise rhs");
        }
        require_compatible_views(a, input_bytes, out, count * sizeof(Out), sizeof(Out) == sizeof(In), "Overlapping pointwise views require identical typed ranges");
        if constexpr (Mode == 2)
            require_compatible_views(b, input_bytes, out, count * sizeof(Out), sizeof(Out) == sizeof(In), "Overlapping pointwise views require identical typed ranges");
        if (!count)
            return;
        auto* lhs = reinterpret_cast<const In*>(static_cast<const char*>(a.buffer->data) + a.offset);
        const In* rhs = nullptr;
        if constexpr (Mode == 2)
            rhs = reinterpret_cast<const In*>(static_cast<const char*>(b.buffer->data) + b.offset);
        auto* output = reinterpret_cast<Out*>(static_cast<char*>(out.buffer->data) + out.offset);
        hipLaunchKernelGGL((pointwise_kernel<Mode, In, Out, Op>), dim3(blocks(count)), dim3(256), 0, context->stream, lhs, rhs, output, count, scalar, op);
        check(hipGetLastError());
    }

    template <unsigned Id, class F, class Storage>
    void dispatch_pointwise(LfsHipContext* context, LfsHipView lhs, LfsHipView rhs,
                            LfsHipView output, size_t count, float scalar) {
        using L = hip_pointwise::Lowering<F>;
        using Out = std::conditional_t<L::predicate, uint8_t, Storage>;
        if constexpr (std::is_same_v<Storage, float> || hip_pointwise::half_supported(Id))
            launch_pointwise<L::mode, Out, Storage>(context, lhs, rhs, output, count, scalar, L::make(scalar));
        else
            throw std::invalid_argument("Unknown HIP pointwise operation");
    }

} // namespace
