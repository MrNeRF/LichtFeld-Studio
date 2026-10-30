/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
namespace hip_matrix {
    template <bool Transpose, bool Masked = false, class A, class B>
    __device__ inline float dot(const A* a, const B* b, size_t column, size_t n, size_t k, const uint8_t* validity = nullptr) {
        float value = 0;
        for (size_t t = 0; t < k; ++t) {
            if constexpr (Masked) {
                if (!validity[t * n + column])
                    continue;
            }
            value = fmaf(float(a[t]), float(b[Transpose ? column * k + t : t * n + column]), value);
        }
        return value;
    }
    template <bool Transpose, bool Masked = false, class A, class B>
    __device__ inline void tile(const A* a, const B* b, size_t m, size_t n, size_t k, size_t row, size_t col,
                                float (&aa)[16][64], float (&bb)[16][64], float (&sums)[4][4],
                                uint8_t* valid = nullptr, const uint8_t* validity = nullptr) {
        const unsigned tx = threadIdx.x, ty = threadIdx.y, lane = ty * 16 + tx;
        for (size_t first = 0; first < k; first += 16) {
            for (unsigned i = lane; i < 1024; i += 256) {
                const size_t inner = i / 64, outer = i % 64;
                aa[inner][outer] = row + outer < m && first + inner < k ? float(a[(row + outer) * k + first + inner]) : 0;
                bb[inner][outer] = col + outer < n && first + inner < k ? float(b[Transpose ? (col + outer) * k + first + inner : (first + inner) * n + col + outer]) : 0;
                if constexpr (Masked)
                    valid[inner * 64 + outer] = col + outer < n && first + inner < k && validity[(first + inner) * n + col + outer];
            }
            __syncthreads();
            for (unsigned inner = 0; inner < 16; ++inner) {
                float av[4], bv[4];
                for (unsigned i = 0; i < 4; ++i) {
                    av[i] = aa[inner][ty + i * 16];
                    bv[i] = bb[inner][tx + i * 16];
                }
                for (unsigned i = 0; i < 4; ++i)
                    for (unsigned j = 0; j < 4; ++j) {
                        if constexpr (Masked) {
                            if (!valid[inner * 64 + tx + j * 16])
                                continue;
                        }
                        sums[i][j] = fmaf(av[i], bv[j], sums[i][j]);
                    }
            }
            __syncthreads();
        }
    }
} // namespace hip_matrix
