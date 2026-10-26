/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
namespace {
    __device__ float chain_value(const float* input, size_t index, const Chain& chain) {
        float value = input[index];
        for (uint32_t stage = 0; stage < chain.count; ++stage) {
            const auto& op = chain.ops[stage];
            value = evaluate_op(op.op, value, op.rhs ? op.rhs[index] : 0, op.scalar);
        }
        return value;
    }
    struct ChainReduction {
        const float* input;
        size_t size;
        Chain chain;
        __device__ float operator()(size_t row, size_t index) const { return chain_value(input, row * size + index, chain); }
    };
    template <unsigned Mode>
    void launch_chain_reduce(LfsHipContext* context, const float* input, float* output, size_t segments, size_t size, Chain chain, bool mean) {
        launch_float_reduction<Mode, float, false>(context, ChainReduction{input, size, chain}, output, segments, size, mean);
    }
} // namespace
extern "C" int lfs_hip_chain_reduce_f32(LfsHipContext* context, LfsHipView input, LfsHipView output, size_t segments, size_t size, const LfsHipChain* source, uint32_t operation, LfsHipError* error) {
    return boundary(error, [&] {
        require(source && source->count && source->count <= 16 && operation <= LFS_HIP_REDUCE_MAX, "Invalid HIP fused reduction descriptor");
        require(!segments || size <= SIZE_MAX / segments, "HIP fused reduction extent overflow");
        const size_t bytes = float_bytes(segments * size), out_bytes = float_bytes(segments);
        validate(context, input.buffer, input.offset, bytes);
        validate(context, output.buffer, output.offset, out_bytes);
        require(input.offset % 4 == 0 && output.offset % 4 == 0, "Unaligned HIP fused reduction");
        require_disjoint(input, bytes, output, out_bytes, "HIP fused reduction input and output overlap");
        Chain chain{};
        chain.count = source->count;
        for (unsigned i = 0; i < chain.count; ++i) {
            const auto& op = source->ops[i];
            require(op.op < LFS_HIP_OP_COUNT && !(pointwise_flags[op.op] & 4), "Unsupported HIP fused reduction pointwise operation");
            chain.ops[i].op = op.op;
            chain.ops[i].scalar = op.scalar;
            if ((pointwise_flags[op.op] & 3) == 2) {
                validate(context, op.rhs.buffer, op.rhs.offset, bytes);
                require(op.rhs.offset % 4 == 0, "Unaligned HIP fused reduction rhs");
                require_disjoint(op.rhs, bytes, output, out_bytes, "HIP fused reduction input and output overlap");
                chain.ops[i].rhs = reinterpret_cast<const float*>(static_cast<const char*>(op.rhs.buffer->data) + op.rhs.offset);
            }
        }
        if (!segments)
            return;
        DeviceScope device(context->device);
        if (!size) {
            require(operation == LFS_HIP_REDUCE_SUM || operation == LFS_HIP_REDUCE_MEAN, "Empty HIP fused minimum or maximum reduction");
            check(lfs_hip_fill_internal(context, static_cast<float*>(output.buffer->data) + output.offset / sizeof(float), segments, operation == LFS_HIP_REDUCE_SUM ? 0.0f : NAN));
            return;
        }
        const auto* in = reinterpret_cast<const float*>(static_cast<const char*>(input.buffer->data) + input.offset);
        auto* out = reinterpret_cast<float*>(static_cast<char*>(output.buffer->data) + output.offset);
        switch (operation) {
        case LFS_HIP_REDUCE_SUM: launch_chain_reduce<LFS_HIP_REDUCE_SUM>(context, in, out, segments, size, chain, false); break;
        case LFS_HIP_REDUCE_MEAN: launch_chain_reduce<LFS_HIP_REDUCE_SUM>(context, in, out, segments, size, chain, true); break;
        case LFS_HIP_REDUCE_MIN: launch_chain_reduce<LFS_HIP_REDUCE_MIN>(context, in, out, segments, size, chain, false); break;
        case LFS_HIP_REDUCE_MAX: launch_chain_reduce<LFS_HIP_REDUCE_MAX>(context, in, out, segments, size, chain, false); break;
        }
    });
}
