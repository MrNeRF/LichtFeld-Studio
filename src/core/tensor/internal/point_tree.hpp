/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include "point_spatial.hpp"

#include "core/tensor/backend/descriptors.hpp"

namespace lfs::core::internal {
    // Bounds on the squared distance from p to any point of an axis-aligned box, rounded exactly like
    // within(). Rounding is monotonic and the terms are combined in the same order, so near > limit proves
    // no point of the box is within the radius, and far <= limit proves every point is.
    struct BoxReach {
        float near;
        float far;
        float limit;
    };

    LFS_POINT_HD inline BoxReach boxReach(const float* p, const float* box, const float radius) {
        float near[3], far[3];
        for (int axis = 0; axis < 3; ++axis) {
            const float below = roundedAdd(box[axis], -p[axis]);
            const float above = roundedAdd(p[axis], -box[3 + axis]);
            near[axis] = fmaxf(0.0f, fmaxf(below, above));
            far[axis] = fmaxf(fabsf(roundedAdd(p[axis], -box[axis])), fabsf(roundedAdd(p[axis], -box[3 + axis])));
        }
        const float radius_squared = roundedMul(radius, radius);
        float limit = radius_squared;
        if (radius_squared < 1.17549435e-38f || !pointFinite(radius_squared)) {
            for (int axis = 0; axis < 3; ++axis) {
                near[axis] = roundedDiv(near[axis], radius);
                far[axis] = roundedDiv(far[axis], radius);
            }
            limit = 1.0f;
        }
        const auto squared = [](const float* v) {
            return roundedAdd(roundedAdd(roundedMul(v[0], v[0]), roundedMul(v[1], v[1])), roundedMul(v[2], v[2]));
        };
        return {squared(near), squared(far), limit};
    }

    // Counts the reference points within radius of p, other than the one at sorted position rank (-1 when p
    // is not a reference), up to max_count. Boxes beyond the radius are skipped and boxes inside it counted
    // whole, so a query costs its distance to the clusters around it rather than their size. (The Vulkan
    // shader walks the same tree without the per-level cursors, which is faster there and slower here.)
    LFS_POINT_HD inline int32_t pointTreeCount(const float* sorted, const float* boxes, const PointTreeProgram& tree,
                                               const float* p, const int64_t rank, const float radius,
                                               const int32_t max_count) {
        if (!(radius > 0.0f) || !pointFinite(radius) || tree.references == 0)
            return 0;
        uint32_t next[kPointTreeMaxLevels];
        uint32_t end[kPointTreeMaxLevels];
        const uint32_t top = tree.levels - 1;
        uint32_t level = top;
        next[top] = 0;
        end[top] = tree.level_count[top];
        int64_t found = 0;
        while (true) {
            if (next[level] == end[level]) {
                if (level == top)
                    break;
                ++level;
                continue;
            }
            const uint32_t node = next[level]++;
            const BoxReach reach = boxReach(p, boxes + (static_cast<size_t>(tree.level_offset[level]) + node) * 6, radius);
            if (reach.near > reach.limit)
                continue;
            const unsigned shift = kPointTreeFanoutBits * (level + 1);
            const uint64_t first = static_cast<uint64_t>(node) << shift;
            const uint64_t last = first + (uint64_t(1) << shift) < tree.references ? first + (uint64_t(1) << shift)
                                                                                   : tree.references;
            if (reach.far <= reach.limit) {
                found += static_cast<int64_t>(last - first) - (rank >= int64_t(first) && rank < int64_t(last) ? 1 : 0);
                if (found >= max_count)
                    return max_count;
                continue;
            }
            if (level != 0) {
                --level;
                next[level] = node << kPointTreeFanoutBits;
                const uint32_t children = (node << kPointTreeFanoutBits) + kPointTreeFanout;
                end[level] = children < tree.level_count[level] ? children : tree.level_count[level];
                continue;
            }
            for (uint64_t j = first; j < last; ++j)
                if (static_cast<int64_t>(j) != rank && within(p, sorted + j * 3, radius) && ++found == max_count)
                    return max_count;
        }
        return static_cast<int32_t>(found);
    }

    // Visits every reference after sorted position rank whose distance from p is within both radius and its
    // own radius (sorted_radii), passing its sorted position. box_radii holds the largest radius under each
    // box: a box whose near distance exceeds the smaller of that and radius holds no such point.
    template <typename Visit>
    LFS_POINT_HD inline void pointTreeMutualNeighbors(const float* sorted, const float* boxes, const float* box_radii,
                                                      const float* sorted_radii, const PointTreeProgram& tree,
                                                      const float* p, const int64_t rank, const float radius,
                                                      Visit&& visit) {
        if (!(radius > 0.0f) || !pointFinite(radius) || tree.references == 0)
            return;
        uint32_t next[kPointTreeMaxLevels];
        uint32_t end[kPointTreeMaxLevels];
        const uint32_t top = tree.levels - 1;
        uint32_t level = top;
        next[top] = 0;
        end[top] = tree.level_count[top];
        while (true) {
            if (next[level] == end[level]) {
                if (level == top)
                    return;
                ++level;
                continue;
            }
            const uint32_t node = next[level]++;
            const unsigned shift = kPointTreeFanoutBits * (level + 1);
            const uint64_t first = static_cast<uint64_t>(node) << shift;
            const uint64_t last = first + (uint64_t(1) << shift) < tree.references ? first + (uint64_t(1) << shift)
                                                                                   : tree.references;
            if (static_cast<int64_t>(last) <= rank + 1)
                continue;
            const size_t index = static_cast<size_t>(tree.level_offset[level]) + node;
            const float reach_radius = fminf(radius, box_radii[index]);
            if (!(reach_radius > 0.0f))
                continue;
            const BoxReach reach = boxReach(p, boxes + index * 6, reach_radius);
            if (reach.near > reach.limit)
                continue;
            if (level != 0) {
                --level;
                next[level] = node << kPointTreeFanoutBits;
                const uint32_t children = (node << kPointTreeFanoutBits) + kPointTreeFanout;
                end[level] = children < tree.level_count[level] ? children : tree.level_count[level];
                continue;
            }
            for (uint64_t j = first > static_cast<uint64_t>(rank + 1) ? first : static_cast<uint64_t>(rank + 1); j < last; ++j) {
                const float* q = sorted + j * 3;
                if (within(p, q, radius) && within(p, q, sorted_radii[j]))
                    visit(j);
            }
        }
    }
    // point_neighbor_spacing over a point tree of every finite point: the mean distance to the nearest three
    // others whose cells lie within one cell of p's (within two when those hold fewer than three), or four
    // cell widths with none. A box whose bounds lie in cells outside the range holds no candidate, since
    // cell() is monotonic; once three candidates are held, a box no nearer than the third is skipped too.
    // Candidates, distances and their order of summation are those of pointNeighborSpacing.
    LFS_POINT_HD inline float pointTreeSpacing(const float* sorted, const float* boxes, const PointTreeProgram& tree,
                                               const float* p, const int64_t rank, const float width) {
        if (!finite_point(p))
            return 0;
        const int centre[3] = {cell(p[0], width), cell(p[1], width), cell(p[2], width)};
        float best[3] = {INFINITY, INFINITY, INFINITY};
        int found = 0;
        for (int extent = 1; extent <= 2 && found < 3; ++extent) {
            uint32_t next[kPointTreeMaxLevels];
            uint32_t end[kPointTreeMaxLevels];
            const uint32_t top = tree.levels - 1;
            uint32_t level = top;
            next[top] = 0;
            end[top] = tree.level_count[top];
            while (true) {
                if (next[level] == end[level]) {
                    if (level == top)
                        break;
                    ++level;
                    continue;
                }
                const uint32_t node = next[level]++;
                const float* box = boxes + (static_cast<size_t>(tree.level_offset[level]) + node) * 6;
                bool outside = false;
                for (int axis = 0; axis < 3; ++axis)
                    outside = outside || cell(box[3 + axis], width) < centre[axis] - extent ||
                              cell(box[axis], width) > centre[axis] + extent;
                if (outside)
                    continue;
                if (found == 3) {
                    float near[3];
                    for (int axis = 0; axis < 3; ++axis)
                        near[axis] = fmaxf(0.0f, fmaxf(roundedAdd(box[axis], -p[axis]), roundedAdd(p[axis], -box[3 + axis])));
                    const float nearest =
                        roundedAdd(roundedAdd(roundedMul(near[0], near[0]), roundedMul(near[1], near[1])), roundedMul(near[2], near[2]));
                    if (!(nearest < best[2]))
                        continue;
                }
                if (level != 0) {
                    --level;
                    next[level] = node << kPointTreeFanoutBits;
                    const uint32_t children = (node << kPointTreeFanoutBits) + kPointTreeFanout;
                    end[level] = children < tree.level_count[level] ? children : tree.level_count[level];
                    continue;
                }
                const uint64_t first = static_cast<uint64_t>(node) << kPointTreeFanoutBits;
                const uint64_t last = first + kPointTreeFanout < tree.references ? first + kPointTreeFanout : tree.references;
                for (uint64_t j = first; j < last; ++j) {
                    if (static_cast<int64_t>(j) == rank)
                        continue;
                    const float* q = sorted + j * 3;
                    bool inner = true, in_range = true;
                    for (int axis = 0; axis < 3; ++axis) {
                        const int offset = cell(q[axis], width) - centre[axis];
                        in_range = in_range && offset >= -extent && offset <= extent;
                        inner = inner && offset >= -1 && offset <= 1;
                    }
                    // The second pass adds only the ring around the first.
                    if (!in_range || (extent == 2 && inner))
                        continue;
                    const float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
                    const float distance = roundedAdd(roundedAdd(roundedMul(dx, dx), roundedMul(dy, dy)), roundedMul(dz, dz));
                    if (distance < best[2]) {
                        best[2] = fmaxf(best[1], distance);
                        best[1] = fmaxf(best[0], fminf(best[1], distance));
                        best[0] = fminf(best[0], distance);
                    }
                    found = found < 3 ? found + 1 : found;
                }
            }
        }
        float sum = 0;
        for (int k = 0; k < found; ++k)
            sum += sqrtf(best[k]);
        return found ? sum / found : width * 4;
    }
} // namespace lfs::core::internal
