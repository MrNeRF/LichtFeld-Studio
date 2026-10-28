/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "collision_voxelization_kernels.hpp"
#include "core/cuda/morton.cuh"
#include "core/tensor/internal/cub_workspace.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include <cfloat>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <glm/gtc/quaternion.hpp>
namespace lfs::core::cuda {
    namespace {
        struct BrickLess {
            __host__ __device__ bool operator()(CollisionBrick a, CollisionBrick b) const {
                return (a.morton & CollisionBrick::key_mask) < (b.morton & CollisionBrick::key_mask);
            }
        };
        __device__ float component(const Region p, int a) { return a == 0 ? p.x : a == 1 ? p.y
                                                                                         : p.z; }
        __device__ bool overlaps(const CollisionBvhNode& node, Region p, int n, Grid grid) {
            for (int a = 0; a < 3; ++a)
                if (node.min[a] >= grid.origin[a] + (component(p, a) + n) * grid.size ||
                    node.max[a] <= grid.origin[a] + component(p, a) * grid.size)
                    return false;
            return true;
        }
        // Both conservative proofs share BVH traversal. The lower bound requires
        // complete Gaussian support and never marks logical padding solid.
        template <bool solid>
        __device__ bool region_bound(Region p, int n, const Grid grid,
                                     const CollisionGaussianGpu* gs, const CollisionBvhNode* nodes) {
            if constexpr (solid)
                for (int a = 0; a < 3; ++a)
                    if (int(component(p, a)) + n > grid.dimensions[a])
                        return false;
            uint32_t stack[32];
            int top = 0;
            stack[top++] = 0;
            float density = 0;
            const float minimum_support = (float(n) * grid.size * .5f) * .99999f;
            while (top) {
                const auto node = nodes[stack[--top]];
                if (!overlaps(node, p, n, grid))
                    continue;
                if constexpr (solid)
                    if (__uint_as_float(node.count ? node.left : node.begin) < minimum_support)
                        continue;
                if (!node.count) {
                    stack[top++] = node.left;
                    stack[top++] = node.right;
                    continue;
                }
                for (uint32_t i = node.begin; i < node.begin + node.count; ++i) {
                    const auto& g = gs[i];
                    float d[3], h[3];
                    bool supported = true;
                    for (int a = 0; a < 3; ++a) {
                        if constexpr (solid) {
                            const float middle = grid.origin[a] + (component(p, a) + float(n) * .5f) * grid.size;
                            d[a] = middle - g.center[a];
                            h[a] = float(n) * grid.size * .5f + (fabsf(middle) + fabsf(grid.origin[a]) + fabsf(g.center[a]) + 1) * 2e-6f;
                            supported &= fabsf(d[a]) + h[a] <= g.extent[a];
                        } else {
                            const int low = int(component(p, a)), high = min(low + n, grid.dimensions[a]);
                            supported &= low < g.max[a] && high > g.min[a];
                            const float middle = grid.origin[a] + (float(low) + float(high)) * (grid.size * .5f);
                            d[a] = middle - g.center[a];
                            h[a] = (high - low) * grid.size * .5f + (fabsf(middle) + fabsf(grid.origin[a]) + fabsf(g.center[a]) + 1) * 1e-6f;
                        }
                    }
                    if (!supported)
                        continue;
                    float distance = 0;
                    for (int a = 0; a < 3; ++a) {
                        const float q = g.inverse[a] * d[0] + g.inverse[a + 3] * d[1] + g.inverse[a + 6] * d[2];
                        const float r = fabsf(g.inverse[a]) * h[0] + fabsf(g.inverse[a + 3]) * h[1] + fabsf(g.inverse[a + 6]) * h[2];
                        const float axis_distance = solid ? fabsf(q) + r + (fabsf(q) + r + 1) * 2e-6f
                                                          : fmaxf(fabsf(q) - (r + fabsf(q) * 1e-6f), 0);
                        distance += axis_distance * axis_distance;
                    }
                    // Reserve coordinate and float-summation error. A failed
                    // proof falls back to the unchanged cell-density kernel.
                    if constexpr (solid) {
                        density = __fadd_rd(density, g.opacity * expf(-.5f * distance) * .99999f);
                        if (density > grid.cutoff * 1.001 + 1e-5)
                            return true;
                    } else {
                        density += g.opacity * expf(-.5f * distance) + 1e-6f;
                        if (double(density) >= grid.cutoff)
                            return true;
                    }
                }
            }
            return false;
        }
        static_assert(morton_encode((1u << 19) - 1, (1u << 19) - 1, (1u << 19) - 1) == (uint64_t(1) << 57) - 1);
    } // namespace
    __global__ void split_regions(const Region* input, uint32_t offset, uint32_t count, int size, Grid grid,
                                  const CollisionGaussianGpu* gs, const CollisionBvhNode* nodes,
                                  Region* output, uint32_t* output_count,
                                  CollisionBrick* solids, uint32_t* solid_count) {
        const auto i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= uint64_t(count) * 8)
            return;
        const auto parent = input[offset + i / 8];
        const int child = i & 7, half = size / 2;
        const Region p{parent.x + (child & 1) * half, parent.y + ((child >> 1) & 1) * half, parent.z + (child >> 2) * half};
        // Reject empty children before they consume frontier storage.
        if (!region_bound<false>(p, half, grid, gs, nodes))
            return;
        if (half <= 128 && region_bound<true>(p, half, grid, gs, nodes)) {
            const auto row = atomicAdd(solid_count, 1u);
            if (row < max_regions) {
                int exponent = 0;
                for (int edge = half; edge > 4; edge /= 2)
                    ++exponent;
                solids[row] = {morton_encode(p.x >> 2, p.y >> 2, p.z >> 2) | (uint64_t(exponent) << 57), ~uint64_t(0)};
            }
            return;
        }
        const auto base = atomicAdd(output_count, 1u);
        if (base >= max_regions)
            return;
        output[base] = p;
    }
    __global__ void evaluate_bricks(const Region* regions, uint32_t offset, Grid grid,
                                    const CollisionGaussianGpu* gs, const CollisionBvhNode* nodes,
                                    CollisionBrick* output, uint32_t* count) {
        const auto tile = regions[offset + blockIdx.x];
        const auto lane = threadIdx.x;
        Region p{tile.x + int(lane & 3), tile.y + int((lane >> 2) & 3), tile.z + int(lane >> 4)};
        float density = 0;
        if (p.x < grid.dimensions[0] && p.y < grid.dimensions[1] && p.z < grid.dimensions[2]) {
            float center[3];
            for (int a = 0; a < 3; ++a)
                center[a] = grid.origin[a] + (component(p, a) + .5f) * grid.size;
            uint32_t stack[32];
            int top = 0;
            stack[top++] = 0;
            while (top && !(density > 0 && double(density) >= grid.cutoff)) {
                const auto node = nodes[stack[--top]];
                if (!overlaps(node, p, 1, grid))
                    continue;
                if (!node.count) {
                    stack[top++] = node.left;
                    stack[top++] = node.right;
                    continue;
                }
                for (uint32_t i = node.begin; i < node.begin + node.count; ++i) {
                    const auto& g = gs[i];
                    float d[3];
                    bool overlap = true;
                    for (int a = 0; a < 3; ++a) {
                        overlap &= fabsf(center[a] - g.center[a]) <= g.extent[a] + grid.size * .5f;
                        d[a] = fminf(fmaxf(g.center[a], center[a] - grid.size * .5f), center[a] + grid.size * .5f) - g.center[a];
                    }
                    if (!overlap)
                        continue;
                    float distance = 0;
                    for (int a = 0; a < 3; ++a) {
                        const float q = g.inverse[a] * d[0] + g.inverse[a + 3] * d[1] + g.inverse[a + 6] * d[2];
                        distance += q * q;
                    }
                    density += g.opacity * expf(-.5f * distance);
                    if (density > 0 && double(density) >= grid.cutoff)
                        break;
                }
            }
        }
        const uint32_t mask = __ballot_sync(0xffffffffu, density > 0 && double(density) >= grid.cutoff);
        __shared__ uint32_t masks[2];
        if ((lane & 31) == 0)
            masks[lane >> 5] = mask;
        __syncthreads();
        if (lane == 0 && (masks[0] | masks[1])) {
            const auto row = atomicAdd(count, 1u);
            if (masks[0] != UINT32_MAX || masks[1] != UINT32_MAX)
                atomicAdd(count + 1, 1u);
            output[row] = {morton_encode(tile.x >> 2, tile.y >> 2, tile.z >> 2), uint64_t(masks[0]) | (uint64_t(masks[1]) << 32)};
        }
    }
    __global__ void prepare(const float* means, const float* scales, const float* rotations, const float* opacity,
                            const bool* deleted, size_t count, glm::mat4 transform,
                            CollisionGaussianGpu* output, float* metrics, uint32_t* errors) {
        const auto row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (row >= count)
            return;
        auto& g = output[row];
        if (deleted && deleted[row]) {
            g = {};
            for (int a = 0; a < 3; ++a)
                metrics[3 * row + a] = NAN;
            return;
        }
        const auto q = glm::quat(rotations[4 * row], rotations[4 * row + 1], rotations[4 * row + 2], rotations[4 * row + 3]);
        auto axes = glm::mat3(transform) * glm::mat3_cast(q);
        for (int a = 0; a < 3; ++a)
            axes[a] *= scales[3 * row + a];
        const auto center = glm::vec3(transform * glm::vec4(means[3 * row], means[3 * row + 1], means[3 * row + 2], 1));
        const auto area = fmax(glm::length(glm::cross(glm::dvec3(axes[0]), glm::dvec3(axes[1]))), fmax(glm::length(glm::cross(glm::dvec3(axes[0]), glm::dvec3(axes[2]))), glm::length(glm::cross(glm::dvec3(axes[1]), glm::dvec3(axes[2])))));
        const auto determinant = fabs(glm::determinant(glm::dmat3(axes)));
        bool valid = (!deleted || !deleted[row]) && opacity[row] > 0 && determinant > 0 && area > 0;
        for (int a = 0; a < 3; ++a) {
            g.center[a] = center[a];
            g.extent[a] = 3 * glm::length(glm::vec3(axes[0][a], axes[1][a], axes[2][a]));
            if (!isfinite(center[a]) || !isfinite(g.extent[a]) || !isfinite(opacity[row]) ||
                !isfinite(scales[3 * row + a]) || scales[3 * row + a] <= 0 ||
                !isfinite(glm::dot(q, q)) || glm::dot(q, q) <= 0) {
                atomicExch(errors, 1u);
                valid = false;
            }
        }
        const auto inverse = valid ? glm::mat3(glm::inverse(glm::dmat3(axes))) : glm::mat3(0);
        for (int a = 0; a < 9; ++a) {
            g.inverse[a] = inverse[a / 3][a % 3];
            if (!isfinite(g.inverse[a]))
                atomicExch(errors, 1u);
        }
        g.opacity = valid ? opacity[row] : 0;
        metrics[3 * row] = valid ? float(log(area)) : NAN;
        metrics[3 * row + 1] = valid ? float(log(determinant / area)) : NAN;
        metrics[3 * row + 2] = valid ? float(log(determinant) / 3) : NAN;
    }
    __global__ void filter(CollisionGaussianGpu* gs, float* metrics, float* bounds, uint32_t n,
                           float area_limit, float thickness, float volume, float thin_guard) {
        const auto i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        auto& g = gs[i];
        if (metrics[3 * i] > area_limit || metrics[3 * i + 1] > thickness ||
            (metrics[3 * i + 2] > volume && metrics[3 * i + 1] > thin_guard))
            g.opacity = 0;
        if (!g.opacity)
            metrics[3 * i] = NAN;
        for (int a = 0; a < 3; ++a) {
            bounds[size_t(a) * n + i] = g.opacity ? g.center[a] - g.extent[a] : INFINITY;
            bounds[size_t(a + 3) * n + i] = g.opacity ? g.center[a] + g.extent[a] : -INFINITY;
        }
    }
    __global__ void keys(const CollisionGaussianGpu* gs, uint64_t* morton, uint32_t* order, uint32_t n, glm::vec3 low, glm::vec3 multiplier) {
        const auto i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        morton[i] = gs[i].opacity ? morton_encode(morton_coordinate(gs[i].center[0], low.x, multiplier.x), morton_coordinate(gs[i].center[1], low.y, multiplier.y), morton_coordinate(gs[i].center[2], low.z, multiplier.z)) : UINT64_MAX;
        order[i] = uint32_t(i);
    }
    __global__ void gather(const CollisionGaussianGpu* source, CollisionGaussianGpu* dest, const uint32_t* order, uint32_t n) {
        const auto i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n)
            dest[i] = source[order[i]];
    }
    __global__ void leaf_bounds(const CollisionGaussianGpu* gs, CollisionBvhNode* nodes, uint32_t n, uint32_t leaves, float cell) {
        const auto i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= leaves)
            return;
        CollisionBvhNode node{};
        node.begin = uint32_t(i * 8);
        node.count = max(1u, min(n, node.begin + 8) - min(n, node.begin));
        float support = 0;
        for (int a = 0; a < 3; ++a) {
            node.min[a] = INFINITY;
            node.max[a] = -INFINITY;
        }
        for (auto row = node.begin; row < min(n, node.begin + 8); ++row) {
            const auto& g = gs[row];
            if (!g.opacity)
                continue;
            support = fmaxf(support, fminf(g.extent[0], fminf(g.extent[1], g.extent[2])));
            for (int a = 0; a < 3; ++a) {
                node.min[a] = fminf(node.min[a], g.center[a] - g.extent[a] - cell * .5f);
                node.max[a] = fmaxf(node.max[a], g.center[a] + g.extent[a] + cell * .5f);
            }
        }
        node.left = __float_as_uint(support);
        nodes[leaves - 1 + i] = node;
    }
    __global__ void parent_bounds(CollisionBvhNode* nodes, uint32_t first) {
        const auto row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= first)
            return;
        const auto index = first - 1 + row;
        const auto a = nodes[2 * index + 1], b = nodes[2 * index + 2];
        auto& node = nodes[index];
        node.left = 2 * index + 1;
        node.right = 2 * index + 2;
        node.count = 0;
        node.begin = __float_as_uint(fmaxf(__uint_as_float(a.count ? a.left : a.begin), __uint_as_float(b.count ? b.left : b.begin)));
        for (int axis = 0; axis < 3; ++axis) {
            node.min[axis] = fminf(a.min[axis], b.min[axis]);
            node.max[axis] = fmaxf(a.max[axis], b.max[axis]);
        }
    }
    __global__ void eligible(const CollisionGaussianGpu* gs, const CollisionBvhNode* nodes, bool* keep, uint32_t n, double cutoff, float cell) {
        const auto i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        const auto& g = gs[i];
        if (g.opacity >= cutoff && g.opacity > 0) {
            keep[i] = true;
            return;
        }
        double upper = 0;
        uint32_t visited = 0, contributors = 0;
        uint32_t stack[32], top = 1;
        stack[0] = 0;
        while (g.opacity && top && upper < cutoff) {
            // Bound proof work, retaining uncertain support for exact evaluation.
            if (++visited > 256) {
                keep[i] = true;
                return;
            }
            const auto node = nodes[stack[--top]];
            bool overlap = true;
            for (int a = 0; a < 3; ++a)
                overlap &= node.min[a] < g.center[a] + g.extent[a] + cell * .5f && node.max[a] > g.center[a] - g.extent[a] - cell * .5f;
            if (!overlap)
                continue;
            if (!node.count) {
                stack[top++] = node.left;
                stack[top++] = node.right;
                continue;
            }
            for (uint32_t j = node.begin; j < node.begin + node.count; ++j) {
                bool hit = true;
                for (int a = 0; a < 3; ++a)
                    hit &= fabsf(gs[j].center[a] - g.center[a]) <= gs[j].extent[a] + g.extent[a] + cell * 1.00001f;
                if (hit) {
                    upper += gs[j].opacity;
                    ++contributors;
                }
            }
        }
        // Bound positive float summation error independently of traversal order.
        keep[i] = g.opacity > 0 && upper * (1 + 2 * contributors * double(FLT_EPSILON)) >= cutoff;
    }
    __global__ void grid_indices(CollisionGaussianGpu* gs, float* bounds, const bool* keep, uint32_t n, Grid grid) {
        const auto i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        auto& g = gs[i];
        if (keep && !keep[i])
            g.opacity = 0;
        for (int a = 0; a < 3; ++a) {
            bounds[size_t(a) * n + i] = g.opacity ? g.center[a] - g.extent[a] : INFINITY;
            bounds[size_t(a + 3) * n + i] = g.opacity ? g.center[a] + g.extent[a] : -INFINITY;
            if (!keep) {
                g.min[a] = g.opacity ? int(fmaxf(0, floorf((g.center[a] - g.extent[a] - grid.origin[a]) / grid.size - .5f))) : 0;
                g.max[a] = g.opacity ? int(fminf(float(grid.dimensions[a]), ceilf((g.center[a] + g.extent[a] - grid.origin[a]) / grid.size + .5f))) : 0;
            }
        }
    }
    size_t queryCollisionSortWorkspaceBytes(size_t count) {
        size_t bytes = 0;
        if (count)
            LFS_CUDA_CHECK(cub::DeviceMergeSort::SortKeys(nullptr, bytes, static_cast<CollisionBrick*>(nullptr), count, BrickLess{}, getCurrentCUDAStream()));
        return bytes;
    }
    void sortCollisionSources(const uint64_t* keys, uint64_t* sorted_keys, const uint32_t* indices, uint32_t* sorted_indices, size_t count, cudaStream_t stream) {
        tensor_ops::run_cub_operation("collision.sort_sources", stream, [&](void* p, size_t& bytes) { return cub::DeviceRadixSort::SortPairs(p, bytes, keys, sorted_keys, indices, sorted_indices, count, 0, 63, stream); });
    }
    void sortCollisionBricks(CollisionBrick* records, size_t count, cudaStream_t stream) {
        tensor_ops::run_cub_operation("collision.sort_bricks", stream, [&](void* p, size_t& bytes) { return cub::DeviceMergeSort::SortKeys(p, bytes, records, count, BrickLess{}, stream); });
    }
} // namespace lfs::core::cuda
