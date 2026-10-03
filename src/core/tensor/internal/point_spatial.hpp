/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include "point_math.hpp"
#include <cstddef>
#include <cstdint>

namespace lfs::core::internal {
    // Radius-wide cells bound an exact query to the current cell and its 26
    // neighbours without scanning the eightfold volume of 2*radius cells.
    LFS_POINT_HD inline int cell(const float value, const float radius) {
        return static_cast<int>(fminf(268435456.0f, fmaxf(-268435456.0f, floorf(roundedDiv(value, radius)))));
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
                                              uint32_t bucket_mask, float radius, bool exclude_self) {
        const float* p = points + i * 3;
        if (!finite_point(p)) {
            return false;
        }
        if (references[i] && !exclude_self) {
            return true;
        }
        const int x = cell(p[0], radius), y = cell(p[1], radius), z = cell(p[2], radius);
        // Most neighbours share the query cell. Check it before adjacent
        // buckets, which may contain long lists of more distant candidates.
        const auto center = hash_cell(x, y, z, bucket_mask);
        for (int32_t j = heads[center]; j >= 0; j = next[j])
            if ((!exclude_self || static_cast<size_t>(j) != i) &&
                within(p, points + static_cast<size_t>(j) * 3, radius))
                return true;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;
                    const auto bucket = hash_cell(x + dx, y + dy, z + dz, bucket_mask);
                    for (int32_t j = heads[bucket]; j >= 0; j = next[j]) {
                        if ((!exclude_self || static_cast<size_t>(j) != i) &&
                            within(p, points + static_cast<size_t>(j) * 3, radius)) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
    LFS_POINT_HD inline int32_t pointNeighborCount(const float* points, const int32_t* heads,
                                                   const int32_t* next, size_t i, uint32_t bucket_mask,
                                                   float radius, int32_t max_count) {
        const float* p = points + i * 3;
        if (!finite_point(p))
            return 0;
        const int x = cell(p[0], radius);
        const int y = cell(p[1], radius);
        const int z = cell(p[2], radius);
        int32_t count = 0;
        for (int slot = 0; slot < 27; ++slot) {
            const int neighbor = slot == 0 ? 13 : (slot <= 13 ? slot - 1 : slot);
            const int dx = neighbor % 3 - 1;
            const int dy = neighbor / 3 % 3 - 1;
            const int dz = neighbor / 9 - 1;
            const auto bucket = hash_cell(x + dx, y + dy, z + dz, bucket_mask);
            for (int32_t j = heads[bucket]; j >= 0; j = next[j]) {
                const float* q = points + static_cast<size_t>(j) * 3;
                // Hash collisions must not count the same reference twice.
                if (static_cast<size_t>(j) != i && cell(q[0], radius) == x + dx &&
                    cell(q[1], radius) == y + dy && cell(q[2], radius) == z + dz && within(p, q, radius)) {
                    if (++count == max_count)
                        return count;
                }
            }
        }
        return count;
    }
    LFS_POINT_HD inline float pointNeighborSpacing(const float* points, const int32_t* heads,
                                                   const int32_t* next, size_t i, uint32_t bucket_mask,
                                                   float radius) {
        const float* p = points + i * 3;
        if (!finite_point(p))
            return 0;
        const int cx = cell(p[0], radius), cy = cell(p[1], radius), cz = cell(p[2], radius);
        float best[3] = {INFINITY, INFINITY, INFINITY};
        int found = 0;
        for (int extent = 1; extent <= 2; ++extent) {
            for (int z = -extent; z <= extent; ++z) {
                for (int y = -extent; y <= extent; ++y) {
                    for (int x = -extent; x <= extent; ++x) {
                        if (extent == 2 && x >= -1 && x <= 1 && y >= -1 && y <= 1 && z >= -1 && z <= 1)
                            continue;
                        const int tx = cx + x, ty = cy + y, tz = cz + z;
                        int visited = 0;
                        for (int j = heads[hash_cell(tx, ty, tz, bucket_mask)]; j >= 0 && visited < 128; j = next[j], ++visited) {
                            const float* q = points + static_cast<size_t>(j) * 3;
                            if (static_cast<size_t>(j) == i || cell(q[0], radius) != tx || cell(q[1], radius) != ty || cell(q[2], radius) != tz)
                                continue;
                            const float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
                            const float distance = dx * dx + dy * dy + dz * dz;
                            if (distance < best[2]) {
                                best[2] = fmaxf(best[1], distance);
                                best[1] = fmaxf(best[0], fminf(best[1], distance));
                                best[0] = fminf(best[0], distance);
                            }
                            found = found < 3 ? found + 1 : found;
                        }
                    }
                }
            }
            if (found == 3)
                break;
        }
        float sum = 0;
        for (int k = 0; k < found; ++k)
            sum += sqrtf(best[k]);
        return found ? sum / found : radius * 4;
    }
} // namespace lfs::core::internal
