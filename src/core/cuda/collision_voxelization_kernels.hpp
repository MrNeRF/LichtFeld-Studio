/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
namespace lfs::core::cuda {
    inline constexpr size_t max_collision_records = 64'000'000;
    LFS_CORE_API size_t collisionOutputBytes(size_t count, size_t additional = 0);
    struct CollisionBrick {
        static constexpr uint64_t key_mask = (uint64_t(1) << 57) - 1;
        uint64_t morton, mask;
    };
    inline constexpr uint32_t max_regions = 1'048'576;
    struct CollisionGaussianGpu {
        float center[3], inverse[9], extent[3], opacity;
        int min[3], max[3];
    };
    struct CollisionBvhNode {
        float min[3], max[3];
        uint32_t left, right, begin, count;
    };
    using Region = glm::ivec3;
    static_assert(sizeof(Region) == 3 * sizeof(int) && alignof(Region) == alignof(int));
    struct Grid {
        float origin[3], size;
        int dimensions[3];
        double cutoff;
    };
    __global__ void split_regions(const Region* input, uint32_t offset, uint32_t count, int size, Grid grid, const CollisionGaussianGpu* gs, const CollisionBvhNode* nodes, Region* output, uint32_t* output_count, CollisionBrick* solids, uint32_t* solid_count);
    __global__ void evaluate_bricks(const Region* regions, uint32_t offset, Grid grid, const CollisionGaussianGpu* gs, const CollisionBvhNode* nodes, CollisionBrick* output, uint32_t* count);
    __global__ void prepare(const float* means, const float* scales, const float* rotations, const float* opacity, const bool* deleted, size_t count, glm::mat4 transform, CollisionGaussianGpu* output, float* metrics, uint32_t* errors);
    __global__ void filter(CollisionGaussianGpu* gs, float* metrics, float* bounds, uint32_t n, float area_limit, float thickness, float volume, float thin_guard);
    __global__ void keys(const CollisionGaussianGpu* gs, uint64_t* morton, uint32_t* order, uint32_t n, glm::vec3 low, glm::vec3 multiplier);
    __global__ void gather(const CollisionGaussianGpu* source, CollisionGaussianGpu* dest, const uint32_t* order, uint32_t n);
    __global__ void leaf_bounds(const CollisionGaussianGpu* gs, CollisionBvhNode* nodes, uint32_t n, uint32_t leaves, float cell);
    __global__ void parent_bounds(CollisionBvhNode* nodes, uint32_t first);
    __global__ void eligible(const CollisionGaussianGpu* gs, const CollisionBvhNode* nodes, bool* keep, uint32_t n, double cutoff, float cell);
    __global__ void grid_indices(CollisionGaussianGpu* gs, float* bounds, const bool* keep, uint32_t n, Grid grid);
    size_t queryCollisionSortWorkspaceBytes(size_t count);
    void sortCollisionSources(const uint64_t* keys, uint64_t* sorted_keys, const uint32_t* indices, uint32_t* sorted_indices, size_t count, cudaStream_t stream);
    void sortCollisionBricks(CollisionBrick* records, size_t count, cudaStream_t stream);
} // namespace lfs::core::cuda
