/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
__global__ void union_components(const float* points, const uint8_t* references, const int32_t* heads,
                                 const int32_t* next, int32_t* parent, const size_t count,
                                 const uint32_t bucket_mask, const float radius) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count || !references[i])
        return;
    const float* p = points + i * 3;
    if (!finite_point(p))
        return;
    int32_t mine = componentRoot(parent, static_cast<int32_t>(i));
    const int cx = cell(p[0], radius), cy = cell(p[1], radius), cz = cell(p[2], radius);
    const float x = p[0], y = p[1], z = p[2];
    const auto self = static_cast<int32_t>(i);
    // Each pair is joined from its smaller index. A candidate from another cell sharing the
    // bucket fails the distance test, and joining a pair twice changes nothing, so the cell
    // check of the counting queries is not needed here.
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                for (int32_t j = __ldg(heads + hash_cell(cx + dx, cy + dy, cz + dz, bucket_mask)); j >= 0;
                     j = __ldg(next + j)) {
                    if (j <= self)
                        continue;
                    const float* q = points + static_cast<size_t>(j) * 3;
                    const float candidate[3] = {__ldg(q), __ldg(q + 1), __ldg(q + 2)};
                    const float query[3] = {x, y, z};
                    if (!within(query, candidate, radius))
                        continue;
                    int32_t other = componentRoot(parent, j);
                    while (mine != other) {
                        if (mine < other) {
                            const int32_t seen = atomicCAS(parent + other, other, mine);
                            if (seen == other)
                                break;
                            other = seen;
                        } else {
                            const int32_t seen = atomicCAS(parent + mine, mine, other);
                            if (seen == mine)
                                break;
                            mine = seen;
                        }
                    }
                }
}

__global__ void flatten_components(int32_t* parent, const size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        lfs::core::device::component_store(parent + i, finalComponentRoot(parent, static_cast<int32_t>(i)));
}

__global__ void point_tree_union(const float* points, const float* sorted, const float* boxes,
                                 const float* box_radii, const int32_t* visit, const float* sorted_radii,
                                 const float* radii, int32_t* parent, const PointTreeProgram tree) {
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= tree.references)
        return;
    const int32_t self = visit[t];
    int32_t mine = componentRoot(parent, self);
    // Joins run between steps of the walk, never inside it: a warp that stops its walk for one lane's
    // atomics and dependent loads costs several times the walk itself.
    constexpr int kHeld = 64, kBatch = 8;
    PointTreeMutualWalk walk;
    pointTreeMutualBegin(walk, tree, radii[self]);
    uint32_t held[kHeld];
    while (!walk.done) {
        const int count = pointTreeMutualStep(walk, sorted, boxes, box_radii, sorted_radii, tree,
                                              points + static_cast<size_t>(self) * 3, static_cast<int64_t>(t),
                                              radii[self], held);
        for (int base = 0; base < count; base += kBatch) {
            // Loads for the whole batch first; a neighbour whose parent is this root needs no join.
            int32_t nodes[kBatch], ups[kBatch];
#pragma unroll
            for (int k = 0; k < kBatch; ++k)
                nodes[k] = base + k < count ? visit[held[base + k]] : mine;
#pragma unroll
            for (int k = 0; k < kBatch; ++k)
                ups[k] = base + k < count ? lfs::core::device::component_load(parent + nodes[k]) : mine;
#pragma unroll
            for (int k = 0; k < kBatch; ++k) {
                if (ups[k] == mine)
                    continue;
                int32_t other = componentRoot(parent, nodes[k]);
                while (mine != other) {
                    if (mine < other) {
                        const int32_t seen = atomicCAS(parent + other, other, mine);
                        if (seen == other)
                            break;
                        other = seen;
                    } else {
                        const int32_t seen = atomicCAS(parent + mine, mine, other);
                        if (seen == mine)
                            break;
                        mine = seen;
                    }
                }
            }
        }
    }
}
