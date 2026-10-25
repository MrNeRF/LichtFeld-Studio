/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"

namespace {
    template <unsigned Mode>
    __device__ float distance_accumulate(float sum, float a, float b, float p) {
        if constexpr (Mode == 0)
            return sum + float(a != b);
        const float difference = fabsf(a - b);
        if constexpr (Mode == 1)
            return sum + difference;
        if constexpr (Mode == 2)
            return sum + difference * difference;
        if constexpr (Mode == 3) {
            if (isnan(sum))
                return sum;
            if (isnan(difference))
                return difference;
            return fmaxf(sum, difference);
        }
        return sum + powf(difference, p);
    }
    template <unsigned Mode>
    __device__ float distance_finish(float sum, float inverse_p) {
        if constexpr (Mode == 2)
            return sqrtf(sum);
        if constexpr (Mode == 4)
            return powf(sum, inverse_p);
        return sum;
    }
    template <unsigned Mode>
    __global__ void cdist_short(const float* lhs, const float* rhs, float* output, size_t rhs_rows,
                                size_t columns, size_t count, float p, float inverse_p) {
        for (size_t pair = size_t(blockIdx.x) * blockDim.x + threadIdx.x; pair < count; pair += size_t(gridDim.x) * blockDim.x) {
            const size_t a = pair / rhs_rows, b = pair % rhs_rows;
            float sum = 0.f;
            for (size_t column = 0; column < columns; ++column)
                sum = distance_accumulate<Mode>(sum, lhs[a * columns + column], rhs[b * columns + column], p);
            output[pair] = distance_finish<Mode>(sum, inverse_p);
        }
    }
    template <unsigned Mode>
    __global__ void cdist_tiles(const float* lhs, const float* rhs, float* output, size_t lhs_rows,
                                size_t rhs_rows, size_t columns, size_t tile_columns, size_t tile_count,
                                float p, float inverse_p) {
        __shared__ float a[16][17], b[16][17];
        for (size_t tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
            const size_t row = (tile / tile_columns) * 16 + threadIdx.y;
            const size_t column = (tile % tile_columns) * 16 + threadIdx.x;
            const size_t b_row = (tile % tile_columns) * 16 + threadIdx.y;
            float sum = 0.f;
            for (size_t start = 0; start < columns; start += 16) {
                const size_t feature = start + threadIdx.x;
                a[threadIdx.y][threadIdx.x] = row < lhs_rows && feature < columns ? lhs[row * columns + feature] : 0.f;
                b[threadIdx.y][threadIdx.x] = b_row < rhs_rows && feature < columns ? rhs[b_row * columns + feature] : 0.f;
                __syncthreads();
                const unsigned valid = unsigned(min(size_t(16), columns - start));
                for (unsigned k = 0; k < valid; ++k)
                    sum = distance_accumulate<Mode>(sum, a[threadIdx.y][k], b[threadIdx.x][k], p);
                __syncthreads();
            }
            if (row < lhs_rows && column < rhs_rows)
                output[row * rhs_rows + column] = distance_finish<Mode>(sum, inverse_p);
        }
    }
    template <unsigned Mode>
    void launch_cdist(LfsHipContext* context, LfsHipView lhs, LfsHipView rhs, LfsHipView output,
                      size_t lhs_rows, size_t rhs_rows, size_t columns, float p) {
        const auto* a = reinterpret_cast<const float*>(static_cast<const char*>(lhs.buffer->data) + lhs.offset);
        const auto* b = reinterpret_cast<const float*>(static_cast<const char*>(rhs.buffer->data) + rhs.offset);
        auto* out = reinterpret_cast<float*>(static_cast<char*>(output.buffer->data) + output.offset);
        const float inverse_p = Mode == 4 ? 1.f / p : 0.f;
        if (columns <= 16) {
            const size_t count = lhs_rows * rhs_rows;
            hipLaunchKernelGGL((cdist_short<Mode>), dim3(blocks(count)), dim3(256), 0, context->stream,
                               a, b, out, rhs_rows, columns, count, p, inverse_p);
        } else {
            const size_t tile_columns = (rhs_rows - 1) / 16 + 1;
            const size_t tile_count = ((lhs_rows - 1) / 16 + 1) * tile_columns;
            hipLaunchKernelGGL((cdist_tiles<Mode>), dim3(unsigned(std::min<size_t>(tile_count, 4096))), dim3(16, 16), 0, context->stream,
                               a, b, out, lhs_rows, rhs_rows, columns, tile_columns, tile_count, p, inverse_p);
        }
        check(hipGetLastError());
    }
} // namespace

int lfs_hip_cdist_f32(LfsHipContext* context, LfsHipView lhs, LfsHipView rhs, LfsHipView output,
                      size_t lhs_rows, size_t rhs_rows, size_t columns, float p, LfsHipError* error) {
    return boundary(error, [&] {
        require(!std::isnan(p) && p >= 0.f, "HIP cdist p must be nonnegative");
        const auto bytes = [](size_t rows, size_t cols) {
            require(!cols || rows <= SIZE_MAX / cols, "HIP cdist shape overflow");
            return float_bytes(rows * cols);
        };
        const size_t a_bytes = bytes(lhs_rows, columns), b_bytes = bytes(rhs_rows, columns), out_bytes = bytes(lhs_rows, rhs_rows);
        validate(context, lhs.buffer, lhs.offset, a_bytes);
        validate(context, rhs.buffer, rhs.offset, b_bytes);
        validate(context, output.buffer, output.offset, out_bytes);
        require(lhs.offset % 4 == 0 && rhs.offset % 4 == 0 && output.offset % 4 == 0, "Unaligned HIP cdist view");
        require_disjoint(lhs, a_bytes, output, out_bytes, "HIP cdist output overlaps an input");
        require_disjoint(rhs, b_bytes, output, out_bytes, "HIP cdist output overlaps an input");
        if (!out_bytes)
            return;
        DeviceScope device(context->device);
        if (p == 0.f)
            launch_cdist<0>(context, lhs, rhs, output, lhs_rows, rhs_rows, columns, p);
        else if (p == 1.f)
            launch_cdist<1>(context, lhs, rhs, output, lhs_rows, rhs_rows, columns, p);
        else if (p == 2.f)
            launch_cdist<2>(context, lhs, rhs, output, lhs_rows, rhs_rows, columns, p);
        else if (std::isinf(p))
            launch_cdist<3>(context, lhs, rhs, output, lhs_rows, rhs_rows, columns, p);
        else
            launch_cdist<4>(context, lhs, rhs, output, lhs_rows, rhs_rows, columns, p);
    });
}
