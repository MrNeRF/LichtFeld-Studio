/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "point_tree_api.h"
namespace {
    __global__ void multinomial_weights_native(const float* weights, double* prefix, uint32_t* invalid, size_t count) {
        for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < count; i += size_t(gridDim.x) * 256) {
            const float w = weights[i];
            const bool valid = isfinite(w) && w >= 0;
            if (!valid)
                atomicOr(invalid, 1u);
            prefix[i] = valid ? double(w) : 0.;
        }
    }
    __global__ void multinomial_draw_native(const double* prefix, int64_t* output, size_t count, size_t samples, uint64_t seed) {
        const double total = prefix[count - 1];
        for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < samples; i += size_t(gridDim.x) * 256) {
            const double u = double(unit_draw(random_words(i, seed).x)) * total;
            size_t lo = 0, hi = count - 1;
            while (lo < hi) {
                const size_t middle = lo + (hi - lo) / 2;
                if (prefix[middle] <= u)
                    lo = middle + 1;
                else
                    hi = middle;
            }
            output[i] = int64_t(lo);
        }
    }
    __global__ void multinomial_keys_native(const float* weights, float* keys, int64_t* indices, size_t count, uint64_t seed) {
        for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < count; i += size_t(gridDim.x) * 256) {
            const float u = fminf(fmaxf(unit_draw(random_words(i, seed).x), 1e-10f), 0.9999999403953552f);
            keys[i] = weights[i] > 0 ? logf(weights[i]) - logf(-logf(u)) : -INFINITY;
            indices[i] = int64_t(i);
        }
    }
} // namespace
int lfs_hip_multinomial(LfsHipContext* context, LfsHipView weights, LfsHipView output, size_t count,
                        size_t samples, uint64_t seed, uint32_t replacement, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && count && count <= INT32_MAX && samples && samples <= INT32_MAX && replacement <= 1 &&
                    (replacement || samples <= count),
                "Invalid HIP multinomial descriptor");
        const size_t weights_bytes = count * 4, output_bytes = samples * 8;
        validate(context, weights.buffer, weights.offset, weights_bytes);
        validate(context, output.buffer, output.offset, output_bytes);
        require(weights.offset % 4 == 0 && output.offset % 8 == 0, "Misaligned HIP multinomial storage");
        require(weights.buffer != output.buffer || weights.offset >= output.offset + output_bytes || output.offset >= weights.offset + weights_bytes,
                "Overlapping HIP multinomial weights/output");
        DeviceScope device(context->device);
        const auto* input = reinterpret_cast<const float*>(static_cast<const char*>(weights.buffer->data) + weights.offset);
        auto* result = reinterpret_cast<int64_t*>(static_cast<char*>(output.buffer->data) + output.offset);
        size_t scan_bytes = 0, sort_bytes = 0;
        double* dummy_prefix = nullptr;
        check(hipcub::DeviceScan::InclusiveSum(nullptr, scan_bytes, dummy_prefix, dummy_prefix, int(count), context->stream));
        float* dummy_keys = nullptr;
        int64_t* dummy_indices = nullptr;
        if (!replacement)
            check(hipcub::DeviceRadixSort::SortPairsDescending(nullptr, sort_bytes, dummy_keys, dummy_keys, dummy_indices, dummy_indices, int(count), 0, 32, context->stream));
        const auto aligned = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
        const size_t prefix_bytes = aligned(count * 8), key_bytes = aligned(count * 4), index_bytes = aligned(count * 8);
        const size_t data_bytes = replacement ? prefix_bytes : std::max(prefix_bytes, 2 * key_bytes) + 2 * index_bytes;
        const size_t buffers = data_bytes + 256, temp = std::max(scan_bytes, sort_bytes);
        require(temp <= SIZE_MAX - buffers, "HIP multinomial workspace overflow");
        const size_t required = buffers + temp;
        ensure_workspace(context, context->sort_workspace, context->sort_workspace_bytes, required);
        auto* memory = static_cast<char*>(context->sort_workspace);
        auto* prefix = reinterpret_cast<double*>(memory);
        auto* invalid = reinterpret_cast<uint32_t*>(memory + data_bytes);
        auto* temporary = memory + buffers;
        check(hipMemsetAsync(invalid, 0, 4, context->stream));
        multinomial_weights_native<<<blocks(count), 256, 0, context->stream>>>(input, prefix, invalid, count);
        check(hipGetLastError());
        check(hipcub::DeviceScan::InclusiveSum(temporary, scan_bytes, prefix, prefix, int(count), context->stream));
        double sum = 0;
        uint32_t bad = 0;
        read_device_scalar(context, &bad, invalid, sizeof(bad), &sum, prefix + count - 1, sizeof(sum));
        require(!bad, "HIP multinomial weights must be finite and non-negative");
        require(std::isfinite(sum) && sum > 0, "HIP multinomial weights must have a positive finite sum");
        if (replacement) {
            multinomial_draw_native<<<blocks(samples), 256, 0, context->stream>>>(prefix, result, count, samples, seed);
            check(hipGetLastError());
        } else {
            auto* keys = reinterpret_cast<float*>(memory);
            auto* sorted_keys = reinterpret_cast<float*>(memory + key_bytes);
            const size_t indices_offset = std::max(prefix_bytes, 2 * key_bytes);
            auto* indices = reinterpret_cast<int64_t*>(memory + indices_offset);
            auto* sorted_indices = reinterpret_cast<int64_t*>(memory + indices_offset + index_bytes);
            multinomial_keys_native<<<blocks(count), 256, 0, context->stream>>>(input, keys, indices, count, seed);
            check(hipGetLastError());
            check(hipcub::DeviceRadixSort::SortPairsDescending(temporary, sort_bytes, keys, sorted_keys, indices, sorted_indices, int(count), 0, 32, context->stream));
            check(hipMemcpyAsync(result, sorted_indices, output_bytes, hipMemcpyDeviceToDevice, context->stream));
        }
        check(hipGetLastError());
    });
}
