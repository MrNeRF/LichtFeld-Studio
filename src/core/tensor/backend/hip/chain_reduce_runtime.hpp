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
    template <unsigned Mode>
    __global__ void chain_short_reduce(const float* input, float* output, size_t segments, size_t size, Chain chain, float scale) {
        for (size_t segment = size_t(blockIdx.x) * blockDim.x + threadIdx.x; segment < segments; segment += size_t(gridDim.x) * blockDim.x) {
            float value = reduce_identity<Mode>();
            for (size_t i = 0; i < size; ++i)
                value = combine_reduce<Mode>(value, chain_value(input, segment * size + i, chain));
            output[segment] = value * scale;
        }
    }
    template <unsigned Mode>
    __global__ void chain_block_reduce(const float* input, float* output, size_t segments, size_t size, Chain chain, unsigned parts, float scale) {
        __shared__ float values[256];
        for (size_t group = blockIdx.x; group < segments * parts; group += gridDim.x) {
            const size_t segment = group / parts, part = group % parts;
            float value = reduce_identity<Mode>();
            for (size_t i = part * 256 + threadIdx.x; i < size; i += size_t(parts) * 256)
                value = combine_reduce<Mode>(value, chain_value(input, segment * size + i, chain));
            const float result = block_reduce_float<Mode>(value, values);
            if (!threadIdx.x)
                output[group] = result * scale;
            __syncthreads();
        }
    }
    template <unsigned Mode>
    void launch_chain_reduce(LfsHipContext* context, const float* input, float* output, size_t segments, size_t size, Chain chain, bool mean) {
        const float scale = mean ? (size ? 1.f / float(size) : NAN) : 1.f;
        if (size <= 32) {
            hipLaunchKernelGGL((chain_short_reduce<Mode>), dim3(blocks(segments)), dim3(256), 0, context->stream, input, output, segments, size, chain, scale);
        } else {
            const unsigned parts = segments < 64 && size > 4096 ? unsigned(std::min<size_t>(4096 / segments, (size - 1) / 1024 + 1)) : 1;
            if (parts > 1)
                ensure_reduction_workspace(context);
            auto* destination = parts > 1 ? context->reduction_workspace : output;
            hipLaunchKernelGGL((chain_block_reduce<Mode>), dim3(unsigned(std::min<size_t>(4096, segments * parts))), dim3(256), 0, context->stream, input, destination, segments, size, chain, parts, parts > 1 ? 1.f : scale);
            check(hipGetLastError());
            if (parts > 1) {
                // A second native reduction consumes only the bounded partials.
                // The pointwise chain is applied exactly once in the first pass.
                hipLaunchKernelGGL((chain_short_reduce<Mode>), dim3(blocks(segments)), dim3(256), 0, context->stream, destination, output, segments, size_t(parts), Chain{}, scale);
            }
        }
        check(hipGetLastError());
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
        auto separate = [&](LfsHipView view) { require(view.buffer != output.buffer || view.offset + bytes <= output.offset || output.offset + out_bytes <= view.offset, "HIP fused reduction input and output overlap"); };
        separate(input);
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
                separate(op.rhs);
                chain.ops[i].rhs = reinterpret_cast<const float*>(static_cast<const char*>(op.rhs.buffer->data) + op.rhs.offset);
            }
        }
        if (!segments)
            return;
        DeviceScope device(context->device);
        if (!size) {
            require(operation == LFS_HIP_REDUCE_SUM || operation == LFS_HIP_REDUCE_MEAN, "Empty HIP fused minimum or maximum reduction");
            hipLaunchKernelGGL(fill, dim3(blocks(segments)), dim3(256), 0, context->stream, static_cast<float*>(output.buffer->data) + output.offset / sizeof(float), segments, operation == LFS_HIP_REDUCE_SUM ? 0.0f : NAN);
            check(hipGetLastError());
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
