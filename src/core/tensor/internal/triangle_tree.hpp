/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "point_spatial.hpp"

#include "core/tensor/backend/descriptors.hpp"

namespace lfs::core::internal {
    // Every parity ray runs along (1, 0.37139067, 0.52911311): off the axes, so rays miss the shared edges
    // of axis-aligned boxes. Tree boxes hold triangle bounds in the frame (U, V, D) with D along the ray:
    // a ray from p can only cross a triangle whose box contains p across the ray and reaches beyond it
    // along the ray. The tree build and the Vulkan shader use the same rows.
    inline constexpr float kRayFrame[9] = {0.348155307f, -0.937436869f, 0.0f,
                                           0.416551804f, 0.154703454f, -0.895852351f,
                                           0.839805023f, 0.311895750f, 0.444351848f};

    LFS_POINT_HD inline void rayFrame(const float* p, float* q) {
        q[0] = 0.348155307f * p[0] + -0.937436869f * p[1] + 0.0f * p[2];
        q[1] = 0.416551804f * p[0] + 0.154703454f * p[1] + -0.895852351f * p[2];
        q[2] = 0.839805023f * p[0] + 0.311895750f * p[1] + 0.444351848f * p[2];
    }

    // Whether the ray from p crosses the triangle stored as (a, b - a, c - a).
    LFS_POINT_HD inline bool rayCrossesTriangle(const float* p, const float* triangle) {
        const float e1x = triangle[3], e1y = triangle[4], e1z = triangle[5];
        const float e2x = triangle[6], e2y = triangle[7], e2z = triangle[8];
        const float px = 0.37139067f * e2z - 0.52911311f * e2y;
        const float py = 0.52911311f * e2x - e2z;
        const float pz = e2y - 0.37139067f * e2x;
        const float determinant = e1x * px + e1y * py + e1z * pz;
        if (!(fabsf(determinant) > 1e-8f))
            return false;
        const float inverse = 1.0f / determinant;
        const float tx = p[0] - triangle[0], ty = p[1] - triangle[1], tz = p[2] - triangle[2];
        const float u = (tx * px + ty * py + tz * pz) * inverse;
        if (!(u >= 0.0f && u <= 1.0f))
            return false;
        const float qx = ty * e1z - tz * e1y;
        const float qy = tz * e1x - tx * e1z;
        const float qz = tx * e1y - ty * e1x;
        const float v = (qx + 0.37139067f * qy + 0.52911311f * qz) * inverse;
        return v >= 0.0f && u + v <= 1.0f && (e2x * qx + e2y * qy + e2z * qz) * inverse > 1e-7f;
    }

    // 1 when the ray from p crosses an odd number of the tree's triangles, else 0. Boxes are (low, high) in
    // the ray frame, widened at build time for the triangles' rounding; slack covers the query's, so
    // rounding never hides a crossing.
    LFS_POINT_HD inline int32_t triangleTreeParity(const float* triangles, const float* boxes,
                                                   const PointTreeProgram& tree, const float* p) {
        if (tree.references == 0 || !pointFinite(p[0]) || !pointFinite(p[1]) || !pointFinite(p[2]))
            return 0;
        float q[3];
        rayFrame(p, q);
        const float slack = 1e-6f * (fabsf(p[0]) + fabsf(p[1]) + fabsf(p[2]));
        uint32_t next[kPointTreeMaxLevels];
        uint32_t end[kPointTreeMaxLevels];
        const uint32_t top = tree.levels - 1;
        uint32_t level = top;
        next[top] = 0;
        end[top] = tree.level_count[top];
        int32_t parity = 0;
        while (true) {
            if (next[level] == end[level]) {
                if (level == top)
                    break;
                ++level;
                continue;
            }
            const uint32_t node = next[level]++;
            const float* box = boxes + (static_cast<size_t>(tree.level_offset[level]) + node) * 6;
            if (q[0] + slack < box[0] || q[0] - slack > box[3] || q[1] + slack < box[1] || q[1] - slack > box[4] ||
                q[2] - slack > box[5])
                continue;
            if (level != 0) {
                --level;
                next[level] = node << kPointTreeFanoutBits;
                const uint32_t children = (node << kPointTreeFanoutBits) + kPointTreeFanout;
                end[level] = children < tree.level_count[level] ? children : tree.level_count[level];
                continue;
            }
            const uint64_t first = static_cast<uint64_t>(node) << kPointTreeFanoutBits;
            const uint64_t last = first + kPointTreeFanout < tree.references ? first + kPointTreeFanout : tree.references;
            for (uint64_t j = first; j < last; ++j)
                parity ^= rayCrossesTriangle(p, triangles + j * 9) ? 1 : 0;
        }
        return parity;
    }
} // namespace lfs::core::internal
