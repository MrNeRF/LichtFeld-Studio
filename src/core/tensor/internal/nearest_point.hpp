/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "point_spatial.hpp"
#ifdef __CUDACC__
#define LFS_NEAREST_HD __host__ __device__ inline
#else
#define LFS_NEAREST_HD inline
#endif
namespace lfs::core::internal {
    LFS_NEAREST_HD int nearestPoint(const float* p, const float* targets, const int* heads, const int* next,
                                    int count, unsigned mask, float width) {
        if (!finite_point(p))
            return -1;
        const int cx = cell(p[0], width), cy = cell(p[1], width), cz = cell(p[2], width);
        float best = INFINITY;
        int index = -1;
        for (int shell = 0; shell < 8; ++shell) {
            for (int z = -shell; z <= shell; ++z)
                for (int y = -shell; y <= shell; ++y)
                    for (int x = -shell; x <= shell; ++x) {
                        if (shell && abs(x) != shell && abs(y) != shell && abs(z) != shell)
                            continue;
                        const int tx = cx + x, ty = cy + y, tz = cz + z;
                        for (int j = heads[hash_cell(tx, ty, tz, mask)]; j >= 0; j = next[j]) {
                            const float* q = targets + 3 * size_t(j);
                            if (cell(q[0], width) != tx || cell(q[1], width) != ty || cell(q[2], width) != tz)
                                continue;
                            const float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
                            const float d = dx * dx + dy * dy + dz * dz;
                            if (d < best || (d == best && (index < 0 || j < index))) {
                                best = d;
                                index = j;
                            }
                        }
                    }
            float bound = INFINITY;
            const int centre[3] = {cx, cy, cz};
            for (int a = 0; a < 3; ++a)
                bound = fminf(bound, fminf(p[a] - (centre[a] - shell) * width, (centre[a] + shell + 1) * width - p[a]));
            if (index >= 0 && bound > 0 && best < bound * bound)
                return index;
        }
        // Bounds terminate ordinary grid queries; an exhaustive fallback keeps
        // the result exact even across large empty gaps and hash saturation.
        for (int j = 0; j < count; ++j) {
            const float* q = targets + 3 * size_t(j);
            const float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
            const float d = dx * dx + dy * dy + dz * dz;
            if (d < best || (d == best && (index < 0 || j < index))) {
                best = d;
                index = j;
            }
        }
        return index;
    }
    LFS_NEAREST_HD int cameraFrustumCount(const float* p, const float* cameras, int count, float maximum) {
        int result = 0;
        for (int c = 0; c < count; ++c) {
            const float* v = cameras + 16 * size_t(c);
            const float x = v[0] * p[0] + v[1] * p[1] + v[2] * p[2] + v[3];
            const float y = v[4] * p[0] + v[5] * p[1] + v[6] * p[2] + v[7];
            const float z = v[8] * p[0] + v[9] * p[1] + v[10] * p[2] + v[11];
            const float dx = p[0] - v[12], dy = p[1] - v[13], dz = p[2] - v[14];
            result += z > 1e-6f && x >= 0 && y >= 0 && x <= z && y <= z &&
                      (maximum <= 0 || dx * dx + dy * dy + dz * dz <= maximum * maximum);
        }
        return result;
    }
} // namespace lfs::core::internal
#undef LFS_NEAREST_HD
