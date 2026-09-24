/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: GPL-3.0-or-later AND Apache-2.0 */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include "core/tensor_ppisp.hpp"
#include <cmath>
#if defined(__CUDACC__)
#define LFS_PPISP_HD __host__ __device__ __forceinline__
#else
#define LFS_PPISP_HD inline
#endif
namespace lfs::core::internal {
    LFS_PPISP_HD void ppisp_pixel(const float* src, float* dst, int width, int height, const PpispParams& p, int i) {
        const int full_height = p.full_height > 0 ? p.full_height : height;
        const float resolution = float(width > full_height ? width : full_height);
        const float x = (float(i % width) + 0.5f - width * 0.5f) / resolution;
        const float y = (float(i / width + p.y_offset) + 0.5f - full_height * 0.5f) / resolution;
        const int plane = width * height;
        float rgb[3];
        for (int c = 0; c < 3; ++c) {
            const float* v = p.vignetting + c * 5;
            const float dx = x - v[0], dy = y - v[1], r2 = fmaf(dx, dx, dy * dy), r4 = r2 * r2, r6 = r4 * r2;
            const float falloff = fmaxf(0.0f, fminf(1.0f, fmaf(v[4], r6, fmaf(v[3], r4, fmaf(v[2], r2, 1.0f)))));
            rgb[c] = fmaxf((src[c * plane + i] * p.exposure_factor) * falloff, 0.0f);
        }
        const float intensity = rgb[0] + rgb[1] + rgb[2];
        float rgi[3];
        for (int c = 0; c < 3; ++c) {
            const float* m = p.color_matrix + c * 3;
            rgi[c] = fmaf(m[0], rgb[0], fmaf(m[1], rgb[1], m[2] * intensity));
        }
        const float norm = intensity / (fmaxf(rgi[2], 0.0f) + 1e-5f);
        rgb[0] = rgi[0] * norm;
        rgb[1] = rgi[1] * norm;
        rgb[2] = rgi[2] * norm - rgb[0] - rgb[1];
        for (int c = 0; c < 3; ++c) {
            const float* crf = p.crf + c * 5;
            const float value = fminf(fmaxf(rgb[c], 0.0f), 1.0f);
            const float curve = value <= crf[3] ? crf[4] * powf(value / crf[3], crf[0]) : 1.0f - (1.0f - crf[4]) * powf((1.0f - value) / (1.0f - crf[3]), crf[1]);
            dst[c * plane + i] = powf(fmaxf(0.0f, curve), crf[2]);
        }
    }
} // namespace lfs::core::internal
#undef LFS_PPISP_HD
