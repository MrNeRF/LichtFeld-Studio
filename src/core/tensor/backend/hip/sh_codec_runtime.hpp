/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "sh_codec_api.h"
namespace hip_sh_codec {
    struct Bindings {
        LfsHipShCodec codec;
        const void* source;
        void* destination;
        const void* indices;
        const float* source_bounds;
        float* destination_bounds;
    };
    // Same resident/pad-dropped layouts as core/tensor/internal/sh_codec.hpp.
    __device__ inline size_t offset(size_t row, unsigned cell, unsigned rest, bool quantized) {
        if (quantized)
            return (row / 32 * (rest * 3) + cell) * 32 + row % 32;
        const unsigned slots = (rest * 3 + 3) / 4;
        return ((row / 32 * slots + cell / 4) * 32 + row % 32) * 4 + cell % 4;
    }
    __device__ inline size_t index(const Bindings& b, size_t row) {
        return b.codec.index_type == 2 ? size_t(static_cast<const int64_t*>(b.indices)[row]) : size_t(static_cast<const int32_t*>(b.indices)[row]);
    }
    __device__ inline float read(const Bindings& b, size_t row, unsigned cell) {
        const auto& p = b.codec;
        const unsigned width = p.source_format == 1 || p.source_format == 2 ? (p.source_rest * 3 + 3) / 4 * 4 : p.source_rest * 3;
        if (row >= p.source_rows || cell >= width)
            return 0;
        const size_t at = p.source_format == 0 ? row * p.source_rest * 3 + cell : offset(row, cell, p.source_rest, p.source_format == 3);
        if (p.source_format == 3) {
            const float lo = b.source_bounds[row / 256 * 2], hi = b.source_bounds[row / 256 * 2 + 1];
            const uint16_t code = static_cast<const uint16_t*>(b.source)[at];
            volatile float range = hi - lo, normalized = float(code) * (1.f / 65535.f);
            return fmaf(range, normalized, lo);
        }
        if (p.source_format == 2)
            return __half2float(static_cast<const __half*>(b.source)[at]);
        return static_cast<const float*>(b.source)[at];
    }
    __global__ void convert(Bindings b, size_t rows, unsigned width) {
        const auto& p = b.codec;
        const size_t total = p.destination_format == 0 ? rows * width : (rows + 31) / 32 * 32 * width;
        for (size_t work = size_t(blockIdx.x) * blockDim.x + threadIdx.x; work < total; work += size_t(blockDim.x) * gridDim.x) {
            const size_t row = p.destination_format == 0 ? work / width : (work / (width * 32)) * 32 + (work / 4) % 32;
            if (row >= rows)
                continue;
            const unsigned cell = p.destination_format == 0 ? unsigned(work % width) : unsigned((work / 128) % (width / 4)) * 4 + work % 4;
            const bool tail = row >= p.count;
            const size_t sr = p.index_type && !p.scatter && !tail ? index(b, row) : p.source_offset + row;
            const size_t dr = p.scatter ? index(b, row) : p.destination_offset + row;
            if (dr >= ((p.destination_rows + 31) / 32 * 32) || (p.scatter && dr >= p.destination_rows))
                continue;
            const float value = !tail && !(cell >= p.destination_rest * 3 && p.source_format != 1) ? read(b, sr, cell) : 0;
            const size_t at = p.destination_format == 0 ? dr * width + cell : offset(dr, cell, p.destination_rest, false);
            if (p.destination_format == 2)
                static_cast<__half*>(b.destination)[at] = __float2half_rn(value);
            else
                static_cast<float*>(b.destination)[at] = value;
        }
    }
    __global__ void encode(Bindings b) {
        const auto& p = b.codec;
        const unsigned lane = threadIdx.x;
        const size_t row = size_t(blockIdx.x) * 256 + lane;
        float cells[45], lo = 1e30f, hi = -1e30f;
        if (row < p.count) {
            const size_t sr = p.index_type ? index(b, row) : p.source_offset + row;
            for (unsigned cell = 0; cell < p.destination_rest * 3; ++cell) {
                const float value = cell < p.source_rest * 3 ? read(b, sr, cell) : 0;
                cells[cell] = value;
                lo = fminf(lo, value);
                hi = fmaxf(hi, value);
            }
        }
        __shared__ float lows[256], highs[256];
        lows[lane] = lo;
        highs[lane] = hi;
        __syncthreads();
        for (unsigned stride = 128; stride; stride /= 2) {
            if (lane < stride) {
                lows[lane] = fminf(lows[lane], lows[lane + stride]);
                highs[lane] = fmaxf(highs[lane], highs[lane + stride]);
            }
            __syncthreads();
        }
        lo = lows[0];
        hi = highs[0];
        if (lo > hi)
            lo = hi = 0;
        if (!lane) {
            b.destination_bounds[blockIdx.x * 2] = lo;
            b.destination_bounds[blockIdx.x * 2 + 1] = hi;
        }
        if (row >= (p.count + 31) / 32 * 32)
            return;
        for (unsigned cell = 0; cell < p.destination_rest * 3; ++cell) {
            uint16_t code = 0;
            if (row < p.count) {
                volatile float range = hi - lo, difference = cells[cell] - lo, numerator = 65535.f * difference;
                const float scaled = numerator * (1.f / fmaxf(range, 1e-20f));
                code = uint16_t(fminf(fmaxf(roundf(scaled), 0.f), 65535.f));
            }
            static_cast<uint16_t*>(b.destination)[offset(row, cell, p.destination_rest, true)] = code;
        }
    }
} // namespace hip_sh_codec
int lfs_hip_sh_codec(LfsHipContext* context, const LfsHipShCodec* p, LfsHipView source, LfsHipView destination,
                     LfsHipView indices, LfsHipView source_bounds, LfsHipView destination_bounds, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && p && p->struct_bytes == sizeof(*p) && p->source_format <= 3 && p->destination_format <= 3 &&
                    p->source_rest <= 15 && p->destination_rest <= 15 && p->index_type <= 2 && p->scatter <= 1 &&
                    p->source_rows <= INT32_MAX && p->destination_rows <= INT32_MAX && p->count <= INT32_MAX,
                "Invalid HIP SH codec descriptor");
        require(p->source_offset <= p->source_rows && p->destination_offset <= p->destination_rows &&
                    (p->index_type && !p->scatter || p->count <= p->source_rows - p->source_offset) &&
                    (p->index_type && p->scatter || p->count <= p->destination_rows - p->destination_offset),
                "HIP SH codec row range exceeds storage");
        require(!(p->source_format == 0 && p->destination_format == 0) && (!p->scatter || (p->index_type && p->source_format == 0 && p->destination_format != 0)) &&
                    (p->source_format != 0 || !p->index_type || p->scatter),
                "Invalid HIP SH codec gather/scatter conversion");
        require(p->destination_format != 3 || (!p->scatter && !p->destination_offset && p->count == p->destination_rows), "HIP Q16 encode requires complete destination");
        auto bytes = [](size_t rows, unsigned rest, unsigned format) { return (format == 0 ? rows * rest * 3 : ((rows + 31) / 32 * 32) * (format == 3 ? rest * 3 : (rest * 3 + 3) / 4 * 4)) * (format >= 2 ? 2 : 4); };
        const size_t src_bytes = bytes(p->source_rows, p->source_rest, p->source_format), dst_bytes = bytes(p->destination_rows, p->destination_rest, p->destination_format);
        if (!p->count || !p->destination_rest)
            return;
        if (src_bytes || source.buffer)
            validate(context, source.buffer, source.offset, src_bytes);
        else
            require(source.offset == 0, "Invalid empty HIP SH source");
        validate(context, destination.buffer, destination.offset, dst_bytes);
        require(source.offset % (p->source_format >= 2 ? 2 : 4) == 0 && destination.offset % (p->destination_format >= 2 ? 2 : 4) == 0, "Misaligned HIP SH codec storage");
        const size_t index_bytes = p->index_type ? p->count * (p->index_type == 2 ? 8 : 4) : 0;
        const size_t src_bounds_bytes = p->source_format == 3 ? (p->source_rows + 255) / 256 * 8 : 0;
        const size_t dst_bounds_bytes = p->destination_format == 3 ? (p->destination_rows + 255) / 256 * 8 : 0;
        if (index_bytes) {
            validate(context, indices.buffer, indices.offset, index_bytes);
            require(indices.offset % (p->index_type == 2 ? 8 : 4) == 0, "Misaligned HIP SH indices");
        }
        if (src_bounds_bytes) {
            validate(context, source_bounds.buffer, source_bounds.offset, src_bounds_bytes);
            require(source_bounds.offset % 4 == 0, "Misaligned HIP SH source bounds");
        }
        if (dst_bounds_bytes) {
            validate(context, destination_bounds.buffer, destination_bounds.offset, dst_bounds_bytes);
            require(destination_bounds.offset % 4 == 0, "Misaligned HIP SH destination bounds");
        }
        const LfsHipView views[]{source, destination, indices, source_bounds, destination_bounds};
        const size_t sizes[]{src_bytes, dst_bytes, index_bytes, src_bounds_bytes, dst_bounds_bytes};
        for (unsigned i = 0; i < 5; ++i)
            for (unsigned j = i + 1; j < 5; ++j) {
                if (i != 1 && i != 4 && j != 1 && j != 4)
                    continue;
                require(!sizes[i] || !sizes[j] || views[i].buffer != views[j].buffer || views[i].offset >= views[j].offset + sizes[j] || views[j].offset >= views[i].offset + sizes[i], "HIP SH output aliases an input/output");
            }
        if (!p->count || !p->destination_rest)
            return;
        auto ptr = [](LfsHipView v) { return static_cast<char*>(v.buffer->data) + v.offset; };
        hip_sh_codec::Bindings b{*p, src_bytes ? ptr(source) : nullptr, ptr(destination), index_bytes ? ptr(indices) : nullptr,
                                 src_bounds_bytes ? reinterpret_cast<float*>(ptr(source_bounds)) : nullptr, dst_bounds_bytes ? reinterpret_cast<float*>(ptr(destination_bounds)) : nullptr};
        DeviceScope device(context->device);
        if (p->destination_format == 3)
            hipLaunchKernelGGL(hip_sh_codec::encode, dim3((p->count + 255) / 256), dim3(256), 0, context->stream, b);
        else {
            const unsigned width = p->destination_format == 0 ? p->destination_rest * 3 : (p->destination_rest * 3 + 3) / 4 * 4;
            const size_t rows = p->count + (!p->scatter && p->destination_format != 0 && p->destination_offset + p->count == p->destination_rows ? (32 - p->destination_rows % 32) % 32 : 0);
            // Enumeration follows physical float4 tiles, including incomplete row groups.
            const size_t work = p->destination_format == 0 ? rows * width : (rows + 31) / 32 * 32 * width;
            hipLaunchKernelGGL(hip_sh_codec::convert, dim3(std::min<size_t>((work + 255) / 256, 4096)), dim3(256), 0, context->stream, b, rows, width);
        }
        check(hipGetLastError());
    });
}
