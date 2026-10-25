/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "collision_voxelization.hpp"
#include "core/cuda/morton.cuh"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "core/tensor/internal/cub_workspace.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include <bit>
#include <cfloat>
#include <climits>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <glm/gtc/quaternion.hpp>
#include <numbers>
namespace lfs::core::cuda {
    namespace {
        constexpr uint32_t max_regions = 1'048'576;
        struct CollisionGaussianGpu {
            float center[3], inverse[9], extent[3], opacity;
            int min[3], max[3];
        };
        struct CollisionBvhNode {
            float min[3], max[3];
            uint32_t left, right, begin, count;
        };
        template <class T>
        struct Buffer {
            tensor_ops::ScopedDeviceBuffer storage;
            T* data = nullptr;
            size_t capacity = 0;
            void reserve(size_t n) {
                if (n <= capacity)
                    return;
                storage = tensor_ops::ScopedDeviceBuffer(checked_product(n, sizeof(T), "Collision allocation"), getCurrentCUDAStream(), "collision");
                data = storage.template as<T>();
                capacity = n;
            }
        };
        struct BrickLess {
            __host__ __device__ bool operator()(CollisionBrick a, CollisionBrick b) const {
                return (a.morton & CollisionBrick::key_mask) < (b.morton & CollisionBrick::key_mask);
            }
        };
        struct Region {
            int x, y, z;
        };
        struct Grid {
            float origin[3], size;
            int dimensions[3];
            double cutoff;
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
        // Conservative density bound over entire voxel volumes, including float
        // coordinate error. No ellipsoid/voxel-center truncation of thin splats.
        __device__ bool possible(Region p, int n, const Grid grid,
                                 const CollisionGaussianGpu* gs, const CollisionBvhNode* nodes) {
            uint32_t stack[32];
            int top = 0;
            stack[top++] = 0;
            float upper = 0;
            while (top) {
                const auto node = nodes[stack[--top]];
                if (!overlaps(node, p, n, grid))
                    continue;
                if (!node.count) {
                    stack[top++] = node.left;
                    stack[top++] = node.right;
                    continue;
                }
                for (uint32_t i = node.begin; i < node.begin + node.count; ++i) {
                    const auto& g = gs[i];
                    float d[3], h[3];
                    bool overlap = true;
                    for (int a = 0; a < 3; ++a) {
                        int low = int(component(p, a)), high = min(low + n, grid.dimensions[a]);
                        overlap &= low < g.max[a] && high > g.min[a];
                        const float middle = grid.origin[a] + (float(low) + float(high)) * (grid.size * .5f);
                        d[a] = middle - g.center[a];
                        h[a] = (high - low) * grid.size * .5f + (fabsf(middle) + fabsf(grid.origin[a]) + fabsf(g.center[a]) + 1) * 1e-6f;
                    }
                    if (!overlap)
                        continue;
                    float distance = 0;
                    for (int a = 0; a < 3; ++a) {
                        const float q = g.inverse[a] * d[0] + g.inverse[a + 3] * d[1] + g.inverse[a + 6] * d[2];
                        const float r = fabsf(g.inverse[a]) * h[0] + fabsf(g.inverse[a + 3]) * h[1] + fabsf(g.inverse[a + 6]) * h[2] + fabsf(q) * 1e-6f;
                        const float nearest_distance = fmaxf(fabsf(q) - r, 0);
                        distance += nearest_distance * nearest_distance;
                    }
                    upper += g.opacity * expf(-.5f * distance) + 1e-6f;
                    if (double(upper) >= grid.cutoff)
                        return true;
                }
            }
            return false;
        }
        static_assert(morton_encode((1u << 19) - 1, (1u << 19) - 1, (1u << 19) - 1) == (uint64_t(1) << 57) - 1);
        __device__ bool certain(Region p, int n, const Grid grid,
                                const CollisionGaussianGpu* gs, const CollisionBvhNode* nodes) {
            // Never mark padding solid: clipped root regions keep exact bounds.
            for (int a = 0; a < 3; ++a)
                if (int(component(p, a)) + n > grid.dimensions[a])
                    return false;
            uint32_t stack[32];
            int top = 0;
            stack[top++] = 0;
            float lower = 0;
            const float minimum_support = (float(n) * grid.size * .5f) * .99999f;
            while (top) {
                const auto node = nodes[stack[--top]];
                if (!overlaps(node, p, n, grid))
                    continue;
                // Every contributing Gaussian must support the whole cube on
                // all three axes. A smaller subtree-wide maximum proves that
                // this lower-bound search cannot contribute, even before the
                // more selective centre/rotation tests. The margin only makes
                // rejection less aggressive near floating-point boundaries.
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
                        const float middle = grid.origin[a] + (component(p, a) + float(n) * .5f) * grid.size;
                        d[a] = middle - g.center[a];
                        h[a] = float(n) * grid.size * .5f + (fabsf(middle) + fabsf(grid.origin[a]) + fabsf(g.center[a]) + 1) * 2e-6f;
                        supported &= fabsf(d[a]) + h[a] <= g.extent[a];
                    }
                    if (!supported)
                        continue;
                    float far2 = 0;
                    for (int a = 0; a < 3; ++a) {
                        const float q = g.inverse[a] * d[0] + g.inverse[a + 3] * d[1] + g.inverse[a + 6] * d[2];
                        const float r = fabsf(g.inverse[a]) * h[0] + fabsf(g.inverse[a + 3]) * h[1] + fabsf(g.inverse[a + 6]) * h[2];
                        const float farthest_distance = fabsf(q) + r + (fabsf(q) + r + 1) * 2e-6f;
                        far2 += farthest_distance * farthest_distance;
                    }
                    lower = __fadd_rd(lower, g.opacity * expf(-.5f * far2) * .99999f);
                    // Reserve float summation error as well as coordinate error.
                    // A failed proof merely falls back to the exact cell kernel.
                    if (lower > grid.cutoff * 1.001 + 1e-5)
                        return true;
                }
            }
            return false;
        }
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
            // Reject children before appending. Previously eight children were
            // counted for every plausible parent, so empty padding alone could
            // exhaust the frontier budget before the next rejection pass.
            if (!possible(p, half, grid, gs, nodes))
                return;
            if (half <= 128 && certain(p, half, grid, gs, nodes)) {
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
                output[row] = {morton_encode(tile.x >> 2, tile.y >> 2, tile.z >> 2), uint64_t(masks[0]) | (uint64_t(masks[1]) << 32)};
            }
        }
        __global__ void prepare(const float* means, const float* scales, const float* rotations, const float* opacity,
                                const bool* deleted, size_t count, size_t offset, glm::mat4 transform,
                                CollisionGaussianGpu* output, float* metrics, uint32_t* errors) {
            const auto row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (row >= count)
                return;
            auto& g = output[offset + row];
            if (deleted && deleted[row]) {
                g = {};
                for (int a = 0; a < 3; ++a)
                    metrics[3 * (offset + row) + a] = NAN;
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
            metrics[3 * (offset + row)] = valid ? float(log(area)) : NAN;
            metrics[3 * (offset + row) + 1] = valid ? float(log(determinant / area)) : NAN;
            metrics[3 * (offset + row) + 2] = valid ? float(log(determinant) / 3) : NAN;
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
                bounds[6 * i + a] = g.opacity ? g.center[a] - g.extent[a] : INFINITY;
                bounds[6 * i + a + 3] = g.opacity ? g.center[a] + g.extent[a] : -INFINITY;
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
                bounds[6 * i + a] = g.opacity ? g.center[a] - g.extent[a] : INFINITY;
                bounds[6 * i + a + 3] = g.opacity ? g.center[a] + g.extent[a] : -INFINITY;
                if (!keep) {
                    g.min[a] = g.opacity ? int(fmaxf(0, floorf((g.center[a] - g.extent[a] - grid.origin[a]) / grid.size - .5f))) : 0;
                    g.max[a] = g.opacity ? int(fminf(float(grid.dimensions[a]), ceilf((g.center[a] + g.extent[a] - grid.origin[a]) / grid.size + .5f))) : 0;
                }
            }
        }
    } // namespace
    size_t queryCollisionSortWorkspaceBytes(size_t count) {
        (void)collisionOutputBytes(count);
        size_t bytes = 0;
        if (count)
            LFS_CUDA_CHECK(cub::DeviceMergeSort::SortKeys(nullptr, bytes, static_cast<CollisionBrick*>(nullptr), count, BrickLess{}, getCurrentCUDAStream()));
        return bytes;
    }
    CollisionBricks voxelizeCollision(std::span<const CollisionSource> sources, CollisionSettings settings, std::stop_token stop, ProgressCallback progress, const std::function<void(float)>& resolution) {
        reportProgress(stop, progress, 0, "collision.prepare");
        size_t total = 0;
        for (const auto& source : sources) {
            if (!source.means.is_valid() || source.means.ndim() != 2 || source.means.shape()[1] != 3)
                throw std::invalid_argument("Collision means must have shape [N,3]");
            total = saturating_add(total, source.means.shape()[0]);
        }
        if (!total || total > size_t(INT_MAX))
            throw std::invalid_argument("Collision source count exceeds the signed tensor index range or is empty");
        const auto stream = getCurrentCUDAStream();
        if (!MemoryPressureCoordinator::instance().preflight({.operation = "Collision geometry",
                                                              .temporary_device_bytes = checked_product(total, 2 * sizeof(CollisionGaussianGpu) + 72, "Collision geometry")})
                 .ok)
            throw std::runtime_error("Insufficient GPU memory for collision geometry; free memory and retry");
        const auto blocks = [](size_t n) { return unsigned((n + 255) / 256); };
        Buffer<CollisionGaussianGpu> gs;
        gs.reserve(total);
        auto metrics = Tensor::empty({total, 3}, Device::CUDA);
        auto bounds = Tensor::empty({total, 6}, Device::CUDA);
        auto errors = Tensor::empty({1}, Device::CUDA, DataType::UInt32);
        LFS_CUDA_CHECK(cudaMemsetAsync(errors.ptr<uint32_t>(), 0, sizeof(uint32_t), stream));
        size_t offset = 0;
        for (const auto& s : sources) {
            const auto n = s.means.shape()[0];
            if (!n)
                continue;
            auto host_check = [&](const Tensor& t, size_t count, DataType type) {
                if (!t.is_valid() || t.numel() != count || t.dtype() != type)
                    throw std::invalid_argument("Invalid collision geometry tensor");
            };
            host_check(s.means, n * 3, DataType::Float32);
            host_check(s.scales, n * 3, DataType::Float32);
            host_check(s.rotations, n * 4, DataType::Float32);
            host_check(s.opacity, n, DataType::Float32);
            if (s.deleted.is_valid())
                host_check(s.deleted, n, DataType::Bool);
            for (int c = 0; c < 4; ++c)
                for (int r = 0; r < 4; ++r)
                    if (!std::isfinite(s.transform[c][r]))
                        throw std::invalid_argument("Non-finite collision source transform");
            if (s.transform[0][3] != 0 || s.transform[1][3] != 0 || s.transform[2][3] != 0 || s.transform[3][3] != 1 ||
                glm::determinant(glm::dmat3(s.transform)) == 0)
                throw std::invalid_argument("Collision source transform must be invertible and affine");
            auto means = s.means.to(Device::CUDA, stream).contiguous(), scales = s.scales.to(Device::CUDA, stream).contiguous();
            auto rotation = s.rotations.to(Device::CUDA, stream).contiguous(), opacity = s.opacity.to(Device::CUDA, stream).contiguous();
            auto deleted = s.deleted.is_valid() ? s.deleted.to(Device::CUDA, stream).contiguous() : Tensor{};
            prepare_inputs_for_stream({&means, &scales, &rotation, &opacity}, stream);
            if (deleted.is_valid())
                prepare_inputs_for_stream({&deleted}, stream);
            prepare<<<blocks(n), 256, 0, stream>>>(means.ptr<float>(), scales.ptr<float>(), rotation.ptr<float>(), opacity.ptr<float>(), deleted.is_valid() ? deleted.ptr<bool>() : nullptr, n, offset, s.transform, gs.data, metrics.ptr<float>(), errors.ptr<uint32_t>());
            LFS_CUDA_LAUNCH_CHECK(stream, "collision.prepare");
            offset += n;
        }
        if (errors.to(Device::CPU).ptr<uint32_t>()[0])
            throw std::invalid_argument("Non-finite Gaussian geometry");
        reportProgress(stop, progress, .04f, "collision.filter");
        std::array<float, 3> fences{INFINITY, INFINITY, INFINITY};
        float thin = -INFINITY;
        for (int a = 0; a < 3; ++a) {
            auto column = metrics.slice(1, a, a + 1).contiguous().reshape({int(total)});
            auto samples = column.masked_select(column.isfinite());
            if (!settings.cleanup || samples.numel() < size_t(a ? 128 : 16))
                continue;
            const auto sorted = samples.sort(0).first;
            const auto q1 = sorted.slice(0, samples.numel() / 4, samples.numel() / 4 + 1).item<float>();
            const auto q3 = sorted.slice(0, 3 * samples.numel() / 4, 3 * samples.numel() / 4 + 1).item<float>();
            fences[a] = q3 + std::max(3 * (q3 - q1), std::log(a ? 8.f : 64.f));
            if (a == 1)
                thin = q1 - std::log(4.f);
        }
        if (settings.max_gaussian_area)
            fences[0] = std::log(settings.max_gaussian_area / float(9 * std::numbers::pi));
        filter<<<blocks(total), 256, 0, stream>>>(gs.data, metrics.ptr<float>(), bounds.ptr<float>(), total, fences[0], fences[1], fences[2], thin);
        LFS_CUDA_LAUNCH_CHECK(stream, "collision.filter");
        auto area = metrics.slice(1, 0, 1).contiguous().reshape({int(total)});
        auto retained = area.masked_select(area.isfinite());
        if (!retained.numel())
            throw std::runtime_error("Source cleanup excluded all splats; disable cleanup or increase maximum area");
        if (retained.numel() * 10 < total)
            LOG_WARN("Collision source filtering retained %zu of %zu Gaussians; reduce filtering if coverage is incomplete", retained.numel(), total);
        if (settings.auto_resolution) {
            const auto sorted = retained.sort(0).first;
            const auto median = sorted.slice(0, retained.numel() / 2, retained.numel() / 2 + 1).item<float>();
            settings.voxel_size = std::exp(median * .5f) * std::max(.05f, float(std::sqrt(retained.numel() * 4 * std::numbers::pi / 32'000'000)));
            settings.validate();
        }
        if (resolution)
            resolution(settings.voxel_size);
        auto bound = [&](bool high) {
            auto column = bounds.slice(1, high ? 3 : 0, high ? 6 : 3);
            auto reduced = (high ? column.max(0) : column.min(0)).to(Device::CPU).contiguous();
            return glm::vec3(reduced.ptr<float>()[0], reduced.ptr<float>()[1], reduced.ptr<float>()[2]);
        };
        auto low = bound(false), high = bound(true);
        Buffer<uint64_t> key_in, key_out;
        Buffer<uint32_t> order_in, order_out;
        key_in.reserve(total);
        key_out.reserve(total);
        order_in.reserve(total);
        order_out.reserve(total);
        keys<<<blocks(total), 256, 0, stream>>>(gs.data, key_in.data, order_in.data, total, low,
                                                {morton_multiplier(high.x - low.x), morton_multiplier(high.y - low.y), morton_multiplier(high.z - low.z)});
        LFS_CUDA_LAUNCH_CHECK(stream, "collision.keys");
        tensor_ops::run_cub_operation("collision.sort_sources", stream, [&](void* p, size_t& bytes) { return cub::DeviceRadixSort::SortPairs(p, bytes, key_in.data, key_out.data, order_in.data, order_out.data, total, 0, 63, stream); });
        Buffer<CollisionGaussianGpu> sorted;
        sorted.reserve(total);
        gather<<<blocks(total), 256, 0, stream>>>(gs.data, sorted.data, order_out.data, total);
        LFS_CUDA_LAUNCH_CHECK(stream, "collision.gather");
        gs = std::move(sorted);
        key_in = {};
        key_out = {};
        order_in = {};
        order_out = {};
        metrics = {};
        area = {};
        retained = {};
        const auto leaves = std::bit_ceil(uint32_t((total + 7) / 8));
        Buffer<CollisionBvhNode> bvh;
        bvh.reserve(2 * leaves - 1);
        auto build_bvh = [&] {
            leaf_bounds<<<blocks(leaves), 256, 0, stream>>>(gs.data, bvh.data, total, leaves, settings.voxel_size);
            LFS_CUDA_LAUNCH_CHECK(stream, "collision.leaf_bounds");
            for (auto first = leaves / 2; first; first /= 2) {
                parent_bounds<<<blocks(first), 256, 0, stream>>>(bvh.data, first);
                LFS_CUDA_LAUNCH_CHECK(stream, "collision.parent_bounds");
            }
        };
        build_bvh();
        auto keep = Tensor::empty({total}, Device::CUDA, DataType::Bool);
        const auto cutoff = -std::log1p(-double(settings.opacity));
        eligible<<<blocks(total), 256, 0, stream>>>(gs.data, bvh.data, keep.ptr<bool>(), total, cutoff, settings.voxel_size);
        LFS_CUDA_LAUNCH_CHECK(stream, "collision.support_bounds");
        Grid grid{{0, 0, 0}, settings.voxel_size, {1, 1, 1}, cutoff};
        grid_indices<<<blocks(total), 256, 0, stream>>>(gs.data, bounds.ptr<float>(), keep.ptr<bool>(), total, grid);
        LFS_CUDA_LAUNCH_CHECK(stream, "collision.support_selection");
        low = bound(false);
        high = bound(true);
        if (!std::isfinite(low.x + low.y + low.z + high.x + high.y + high.z))
            throw std::runtime_error("No occupied support; lower opacity or check the source");
        CollisionBricks result;
        auto& v = result.volume;
        v.settings = settings;
        v.origin = glm::floor((low - settings.voxel_size * .5f) / settings.voxel_size) * settings.voxel_size - 2 * settings.voxel_size;
        const auto dimensions = glm::ceil((glm::dvec3(high) + settings.voxel_size * .5 - glm::dvec3(v.origin)) / double(settings.voxel_size)) + 3.;
        if (glm::any(glm::lessThan(dimensions, glm::dvec3(1))) || glm::any(glm::greaterThan(dimensions, glm::dvec3(1 << 21))) || !std::isfinite(dimensions.x + dimensions.y + dimensions.z))
            throw std::runtime_error("Collision bounds exceed coordinate limits; increase cell size or remove remote outliers");
        v.dimensions = glm::ivec3(dimensions);
        LOG_INFO("Collision grid %d x %d x %d at %g scene units; sparse storage grows on demand", v.dimensions.x, v.dimensions.y, v.dimensions.z, settings.voxel_size);
        v.edge = std::max(4u, std::bit_ceil(unsigned(std::max({v.dimensions.x, v.dimensions.y, v.dimensions.z}))));
        for (int a = 0; a < 3; ++a) {
            grid.origin[a] = v.origin[a];
            grid.dimensions[a] = v.dimensions[a];
        }
        grid_indices<<<blocks(total), 256, 0, stream>>>(gs.data, bounds.ptr<float>(), nullptr, total, grid);
        LFS_CUDA_LAUNCH_CHECK(stream, "collision.grid_indices");
        build_bvh();
        bounds = {};
        keep = {};
        errors = {};
        struct Frame {
            Buffer<Region> regions;
            uint32_t count = 0, offset = 0;
            int edge = 0;
        };
        std::vector<Frame> frames;
        for (int n = v.edge;; n /= 2) {
            frames.emplace_back();
            frames.back().edge = n;
            if (n == 4)
                break;
        }
        frames[0].regions.reserve(1);
        frames[0].count = 1;
        LFS_CUDA_CHECK(cudaMemsetAsync(frames[0].regions.data, 0, sizeof(Region), stream));
        Buffer<uint32_t> count, solid_count;
        count.reserve(1);
        solid_count.reserve(1);
        Buffer<CollisionBrick> output;
        std::vector<Buffer<CollisionBrick>> chunks;
        size_t result_size = 0;
        auto append = [&](uint32_t n) {
            if (n > output.capacity)
                throw std::runtime_error("Collision dispatch exceeded its output buffer");
            const auto bytes = collisionOutputBytes(result_size, n);
            if (!n)
                return;
            chunks.emplace_back();
            chunks.back().reserve(n);
            LFS_CUDA_CHECK(cudaMemcpyAsync(chunks.back().data, output.data, size_t(n) * sizeof(CollisionBrick), cudaMemcpyDeviceToDevice, stream));
            result_size = bytes / sizeof(CollisionBrick);
        };
        const double volume = double(v.edge) * v.edge * v.edge;
        double completed = 0;
        int depth = 0;
        while (depth >= 0) {
            reportProgress(stop, progress, .2f + .72f * float(std::clamp(completed / volume, 0., 1.)), "collision.voxelize");
            auto& frame = frames[depth];
            if (frame.offset == frame.count) {
                --depth;
                continue;
            }
            uint32_t occupied = 0;
            if (frame.edge > 4) {
                const auto batch = std::min(frame.count - frame.offset, max_regions / 8);
                auto& child = frames[depth + 1];
                child.regions.reserve(size_t(batch) * 8);
                output.reserve(size_t(batch) * 8);
                LFS_CUDA_CHECK(cudaMemsetAsync(count.data, 0, 4, stream));
                LFS_CUDA_CHECK(cudaMemsetAsync(solid_count.data, 0, 4, stream));
                split_regions<<<(uint64_t(batch) * 8 + 127) / 128, 128, 0, stream>>>(frame.regions.data, frame.offset, batch, frame.edge, grid, gs.data, bvh.data, child.regions.data, count.data, output.data, solid_count.data);
                LFS_CUDA_LAUNCH_CHECK(stream, "collision.split_regions");
                LFS_CUDA_CHECK(cudaMemcpyAsync(&child.count, count.data, 4, cudaMemcpyDeviceToHost, stream));
                LFS_CUDA_CHECK(cudaMemcpyAsync(&occupied, solid_count.data, 4, cudaMemcpyDeviceToHost, stream));
                LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                append(occupied);
                const double half = frame.edge / 2;
                completed += (uint64_t(batch) * 8 - child.count) * half * half * half;
                frame.offset += batch;
                child.offset = 0;
                if (child.count)
                    ++depth;
            } else {
                const auto batch = std::min(frame.count - frame.offset, 65536u);
                output.reserve(batch);
                LFS_CUDA_CHECK(cudaMemsetAsync(count.data, 0, 4, stream));
                evaluate_bricks<<<batch, 64, 0, stream>>>(frame.regions.data, frame.offset, grid, gs.data, bvh.data, output.data, count.data);
                LFS_CUDA_LAUNCH_CHECK(stream, "collision.evaluate_bricks");
                LFS_CUDA_CHECK(cudaMemcpyAsync(&occupied, count.data, 4, cudaMemcpyDeviceToHost, stream));
                LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                append(occupied);
                completed += batch * double(frame.edge) * frame.edge * frame.edge;
                frame.offset += batch;
            }
        }
        reportProgress(stop, progress, .93f, "collision.sort");
        frames.clear();
        gs = {};
        bvh = {};
        output = {};
        Buffer<CollisionBrick> compact;
        LOG_INFO("Collision generated %zu sparse records", result_size);
        const auto sort_bytes = queryCollisionSortWorkspaceBytes(result_size);
        const auto compact_bytes = collisionOutputBytes(result_size);
        // Chunks coexist with the compact copy, then are released before sorting.
        // Charge the larger additional phase, including reclaimable allocator pools.
        if (!MemoryPressureCoordinator::instance().preflight({.operation = "Collision compaction",
                                                              .temporary_device_bytes = std::max(compact_bytes, sort_bytes)})
                 .ok)
            throw std::runtime_error("Insufficient GPU memory to compact collision; free memory and retry");
        compact.reserve(result_size);
        offset = 0;
        for (auto& chunk : chunks) {
            LFS_CUDA_CHECK(cudaMemcpyAsync(compact.data + offset, chunk.data, chunk.capacity * sizeof(CollisionBrick), cudaMemcpyDeviceToDevice, stream));
            offset += chunk.capacity;
            chunk = {};
        }
        if (result_size)
            tensor_ops::run_cub_operation("collision.sort_bricks", stream, [&](void* p, size_t& bytes) { return cub::DeviceMergeSort::SortKeys(p, bytes, compact.data, result_size, BrickLess{}, stream); });
        result.bricks.resize(result_size);
        if (result_size)
            LFS_CUDA_CHECK(cudaMemcpyAsync(result.bricks.data(), compact.data, result_size * sizeof(CollisionBrick), cudaMemcpyDeviceToHost, stream));
        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
        return result;
    }
} // namespace lfs::core::cuda
