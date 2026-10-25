/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
namespace lfs::core::cuda {
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
    struct Region {
        int x, y, z;
    };
    struct Grid {
        float origin[3], size;
        int dimensions[3];
        double cutoff;
    };
    void launch_split_regions(dim3 blocks, dim3 threads, cudaStream_t stream, const Region* input, uint32_t offset, uint32_t count, int size, Grid grid, const CollisionGaussianGpu* gs, const CollisionBvhNode* nodes, Region* output, uint32_t* output_count, CollisionBrick* solids, uint32_t* solid_count);
    void launch_evaluate_bricks(dim3 blocks, dim3 threads, cudaStream_t stream, const Region* regions, uint32_t offset, Grid grid, const CollisionGaussianGpu* gs, const CollisionBvhNode* nodes, CollisionBrick* output, uint32_t* count);
    void launch_prepare(dim3 blocks, dim3 threads, cudaStream_t stream, const float* means, const float* scales, const float* rotations, const float* opacity, const bool* deleted, size_t count, size_t offset, glm::mat4 transform, CollisionGaussianGpu* output, float* metrics, uint32_t* errors);
    void launch_filter(dim3 blocks, dim3 threads, cudaStream_t stream, CollisionGaussianGpu* gs, float* metrics, float* bounds, uint32_t n, float area_limit, float thickness, float volume, float thin_guard);
    void launch_keys(dim3 blocks, dim3 threads, cudaStream_t stream, const CollisionGaussianGpu* gs, uint64_t* morton, uint32_t* order, uint32_t n, glm::vec3 low, glm::vec3 multiplier);
    void launch_gather(dim3 blocks, dim3 threads, cudaStream_t stream, const CollisionGaussianGpu* source, CollisionGaussianGpu* dest, const uint32_t* order, uint32_t n);
    void launch_leaf_bounds(dim3 blocks, dim3 threads, cudaStream_t stream, const CollisionGaussianGpu* gs, CollisionBvhNode* nodes, uint32_t n, uint32_t leaves, float cell);
    void launch_parent_bounds(dim3 blocks, dim3 threads, cudaStream_t stream, CollisionBvhNode* nodes, uint32_t first);
    void launch_eligible(dim3 blocks, dim3 threads, cudaStream_t stream, const CollisionGaussianGpu* gs, const CollisionBvhNode* nodes, bool* keep, uint32_t n, double cutoff, float cell);
    void launch_grid_indices(dim3 blocks, dim3 threads, cudaStream_t stream, CollisionGaussianGpu* gs, float* bounds, const bool* keep, uint32_t n, Grid grid);
    size_t queryCollisionSortWorkspaceBytes(size_t count);
    void sortCollisionSources(const uint64_t* keys, uint64_t* sorted_keys, const uint32_t* indices, uint32_t* sorted_indices, size_t count, cudaStream_t stream);
    void sortCollisionBricks(CollisionBrick* records, size_t count, cudaStream_t stream);
} // namespace lfs::core::cuda
