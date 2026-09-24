/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include "point_math.hpp"
#include <cstddef>
#include <cstdint>

namespace lfs::core::internal {
    LFS_POINT_HD inline int cell(const float value, const float radius) {
        return static_cast<int>(fminf(268435456.0f, fmaxf(-268435456.0f, floorf(roundedDiv(value, radius) * 0.5f))));
    }

    LFS_POINT_HD inline uint32_t hash_cell(const int x, const int y, const int z, const uint32_t mask) {
        return ((static_cast<uint32_t>(x) * 73856093u) ^
                (static_cast<uint32_t>(y) * 19349663u) ^
                (static_cast<uint32_t>(z) * 83492791u)) &
               mask;
    }

    LFS_POINT_HD inline bool finite_point(const float* p) {
        return pointFinite(p[0]) && pointFinite(p[1]) && pointFinite(p[2]);
    }

    LFS_POINT_HD inline bool within(const float* a, const float* b, const float radius) {
        float x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
        const float radius_squared = roundedMul(radius, radius);
        float limit = radius_squared;
        if (radius_squared < 1.17549435e-38f || !pointFinite(radius_squared)) {
            x = roundedDiv(x, radius);
            y = roundedDiv(y, radius);
            z = roundedDiv(z, radius);
            limit = 1.0f;
        }
        return roundedAdd(roundedAdd(roundedMul(x, x), roundedMul(y, y)), roundedMul(z, z)) <= limit;
    }

    LFS_POINT_HD inline bool pointHasNeighbor(const float* points, const uint8_t* references,
                                              const int32_t* heads, const int32_t* next, size_t i,
                                              uint32_t bucket_mask, float radius) {
        const float* p = points + i * 3;
        if (!finite_point(p)) {
            return false;
        }
        if (references[i]) {
            return true;
        }
        const int x = cell(p[0], radius), y = cell(p[1], radius), z = cell(p[2], radius);
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const auto bucket = hash_cell(x + dx, y + dy, z + dz, bucket_mask);
                    for (int32_t j = heads[bucket]; j >= 0; j = next[j]) {
                        if (within(p, points + static_cast<size_t>(j) * 3, radius)) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
} // namespace lfs::core::internal
