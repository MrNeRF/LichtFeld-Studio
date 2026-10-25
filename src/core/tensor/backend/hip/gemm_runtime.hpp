/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "gemm_api.h"
namespace hip_gemm {
    // Depthwise filters and vector products have very few output rows. A full
    // matrix tile would spend most of its arithmetic on padded zero rows.
    template <bool Transpose, bool Relu>
    __global__ void skinny(const float* a, const float* b, const float* bias, float* out,
                           size_t m, size_t n, size_t k) {
        const size_t column = size_t(blockIdx.x) * blockDim.x + threadIdx.x, row = blockIdx.y;
        if (column >= n)
            return;
        float value = 0;
        const size_t ao = (size_t(blockIdx.z) * m + row) * k, bo = size_t(blockIdx.z) * k * n;
        for (size_t t = 0; t < k; ++t)
            value = fmaf(a[ao + t], b[bo + (Transpose ? column * k + t : t * n + column)], value);
        value += bias ? bias[column] : 0.f;
        out[(size_t(blockIdx.z) * m + row) * n + column] = Relu ? fmaxf(value, 0.f) : value;
    }
    template <bool Transpose, bool Relu>
    __global__ void tiled(const float* a, const float* b, const float* bias, float* out,
                          size_t m, size_t n, size_t k) {
        __shared__ float aa[16][64], bb[16][64];
        const auto tx = threadIdx.x, ty = threadIdx.y, lane = ty * 16 + tx;
        const size_t row = size_t(blockIdx.y) * 64, col = size_t(blockIdx.x) * 64;
        if (k) {
            a += size_t(blockIdx.z) * m * k;
            b += size_t(blockIdx.z) * k * n;
        }
        out += size_t(blockIdx.z) * m * n;
        float sum[4][4]{};
        for (size_t first = 0; first < k; first += 16) {
            for (unsigned i = lane; i < 1024; i += 256) {
                const size_t inner = i / 64, outer = i % 64;
                aa[inner][outer] = row + outer < m && first + inner < k ? a[(row + outer) * k + first + inner] : 0;
                bb[inner][outer] = col + outer < n && first + inner < k ? (Transpose ? b[(col + outer) * k + first + inner] : b[(first + inner) * n + col + outer]) : 0;
            }
            __syncthreads();
            for (unsigned inner = 0; inner < 16; ++inner) {
                float av[4], bv[4];
                for (unsigned i = 0; i < 4; ++i) {
                    av[i] = aa[inner][ty + i * 16];
                    bv[i] = bb[inner][tx + i * 16];
                }
                for (unsigned i = 0; i < 4; ++i)
                    for (unsigned j = 0; j < 4; ++j)
                        sum[i][j] = fmaf(av[i], bv[j], sum[i][j]);
            }
            __syncthreads();
        }
        for (unsigned i = 0; i < 4; ++i)
            for (unsigned j = 0; j < 4; ++j) {
                const size_t r = row + ty + i * 16, c = col + tx + j * 16;
                if (r < m && c < n) {
                    const float value = sum[i][j] + (bias ? bias[c] : 0.f);
                    out[r * n + c] = Relu ? fmaxf(value, 0.f) : value;
                }
            }
    }
} // namespace hip_gemm
int lfs_hip_gemm_f32(LfsHipContext* context, LfsHipView a, LfsHipView b, LfsHipView bias, LfsHipView output, const LfsHipGemm* p, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && p && p->struct_bytes == sizeof(*p) && p->transpose_b <= 1 && p->relu <= 1 &&
                    p->m <= 65535ull * 64 && p->n <= INT_MAX && p->k <= INT_MAX,
                "Invalid HIP GEMM descriptor");
        if (!p->m || !p->n || !p->batch)
            return;
        auto bytes = [](uint64_t batch, uint64_t rows, uint64_t columns) {
            require(rows == 0 || columns <= SIZE_MAX / 4 / rows, "HIP GEMM matrix byte count overflows");
            const auto one = rows * columns * 4;
            require(one == 0 || batch <= SIZE_MAX / one, "HIP GEMM batch byte count overflows");
            return size_t(batch * one);
        };
        const auto an = bytes(p->batch, p->m, p->k), bn = bytes(p->batch, p->k, p->n), cn = bytes(p->batch, p->m, p->n);
        const auto biasn = bias.buffer ? bytes(1, 1, p->n) : 0;
        if (an)
            validate(context, a.buffer, a.offset, an);
        if (bn)
            validate(context, b.buffer, b.offset, bn);
        validate(context, output.buffer, output.offset, cn);
        if (biasn)
            validate(context, bias.buffer, bias.offset, biasn);
        require(a.offset % 4 == 0 && b.offset % 4 == 0 && bias.offset % 4 == 0 && output.offset % 4 == 0, "Misaligned HIP GEMM views");
        require(!views_overlap(a, an, output, cn) && !views_overlap(b, bn, output, cn) && !views_overlap(bias, biasn, output, cn), "HIP GEMM output overlaps input");
        DeviceScope device(context->device);
        auto ptr = [](LfsHipView v) { return v.buffer ? reinterpret_cast<float*>(static_cast<char*>(v.buffer->data) + v.offset) : nullptr; };
        for (size_t batch = 0; batch < p->batch; batch += 65535) {
            const auto chunk = std::min<uint64_t>(65535, p->batch - batch);
            auto* aa = an ? ptr(a) + batch * p->m * p->k : nullptr;
            auto* bb = bn ? ptr(b) + batch * p->k * p->n : nullptr;
            auto* cc = ptr(output) + batch * p->m * p->n;
            const dim3 grid(static_cast<unsigned>((p->n + 63) / 64), static_cast<unsigned>((p->m + 63) / 64), static_cast<unsigned>(chunk)), block(16, 16);
#define LFS_GEMM_LAUNCH(T, R)                                                                                                                                                                                                                                         \
    do {                                                                                                                                                                                                                                                              \
        if (p->m <= 4)                                                                                                                                                                                                                                                \
            hipLaunchKernelGGL((hip_gemm::skinny<T, R>), dim3(static_cast<unsigned>((p->n + 255) / 256), static_cast<unsigned>(p->m), static_cast<unsigned>(chunk)), dim3(256), 0, context->stream, aa, bb, ptr(bias), cc, size_t(p->m), size_t(p->n), size_t(p->k)); \
        else                                                                                                                                                                                                                                                          \
            hipLaunchKernelGGL((hip_gemm::tiled<T, R>), grid, block, 0, context->stream, aa, bb, ptr(bias), cc, size_t(p->m), size_t(p->n), size_t(p->k));                                                                                                            \
    } while (false)
            if (p->transpose_b) {
                if (p->relu)
                    LFS_GEMM_LAUNCH(true, true);
                else
                    LFS_GEMM_LAUNCH(true, false);
            } else {
                if (p->relu)
                    LFS_GEMM_LAUNCH(false, true);
                else
                    LFS_GEMM_LAUNCH(false, false);
            }
#undef LFS_GEMM_LAUNCH
            check(hipGetLastError());
        }
    });
}
