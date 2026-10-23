/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "collision_voxelization.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace lfs::core::cuda {
    namespace {
        constexpr uint32_t max_regions = 32'000'000;
        void checked(cudaError_t error) {
            if (error != cudaSuccess)
                throw std::runtime_error(std::string("Collision CUDA: ") + cudaGetErrorString(error));
        }
        struct Stream {
            cudaStream_t value{};
            Stream() { checked(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
            ~Stream() {
                cudaStreamSynchronize(value);
                cudaStreamDestroy(value);
            }
        };
        template <class T>
        struct Buffer {
            T* data = nullptr;
            size_t capacity = 0;
            ~Buffer() {
                if (data)
                    cudaFree(data);
            }
            void reserve(size_t n) {
                if (n <= capacity)
                    return;
                T* replacement = nullptr;
                checked(cudaMalloc(&replacement, n * sizeof(T)));
                if (data)
                    checked(cudaFree(data));
                data = replacement;
                capacity = n;
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
        __device__ bool overlaps(const CollisionBvhNode& node, Region p, int n) {
            return node.min[0] < p.x + n && node.max[0] > p.x &&
                   node.min[1] < p.y + n && node.max[1] > p.y &&
                   node.min[2] < p.z + n && node.max[2] > p.z;
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
                if (!overlaps(node, p, n))
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
                        const float near = fmaxf(fabsf(q) - r, 0);
                        distance += near * near;
                    }
                    upper += g.opacity * expf(-.5f * distance) + 1e-6f;
                    if (double(upper) >= grid.cutoff)
                        return true;
                }
            }
            return false;
        }
        __global__ void split_regions(const Region* input, uint32_t count, int size, Grid grid,
                                      const CollisionGaussianGpu* gs, const CollisionBvhNode* nodes,
                                      Region* output, uint32_t* output_count) {
            const auto i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= count)
                return;
            const auto p = input[i];
            if (!possible(p, size, grid, gs, nodes))
                return;
            const auto base = atomicAdd(output_count, 8u);
            if (base + 8 > max_regions)
                return;
            const int half = size / 2;
            for (int child = 0; child < 8; ++child) {
                output[base + child] = {p.x + (child & 1) * half, p.y + ((child >> 1) & 1) * half, p.z + (child >> 2) * half};
            }
        }
        __device__ uint64_t morton(Region p) {
            uint64_t result = 0;
            for (int bit = 0; bit < 19; ++bit) {
                result |= uint64_t((p.x >> 2 >> bit) & 1) << (3 * bit);
                result |= uint64_t((p.y >> 2 >> bit) & 1) << (3 * bit + 1);
                result |= uint64_t((p.z >> 2 >> bit) & 1) << (3 * bit + 2);
            }
            return result;
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
                    if (!overlaps(node, p, 1))
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
                output[row] = {morton(tile), uint64_t(masks[0]) | (uint64_t(masks[1]) << 32)};
            }
        }
    } // namespace
    std::vector<CollisionBrick> voxelizeCollisionBricks(
        std::span<const CollisionGaussianGpu> gaussians, std::span<const CollisionBvhNode> nodes,
        int root_size, std::array<int, 3> dimensions, std::array<float, 3> origin,
        float voxel_size, double cutoff, std::stop_token stop, const std::function<void(float)>& progress) {
        auto cancel = [&] {
            if (stop.stop_requested())
                throw std::runtime_error("Collision generation cancelled");
        };
        cancel();
        Stream stream;
        Buffer<CollisionGaussianGpu> gs;
        gs.reserve(gaussians.size());
        Buffer<CollisionBvhNode> bvh;
        bvh.reserve(nodes.size());
        checked(cudaMemcpyAsync(gs.data, gaussians.data(), gaussians.size_bytes(), cudaMemcpyHostToDevice, stream.value));
        checked(cudaMemcpyAsync(bvh.data, nodes.data(), nodes.size_bytes(), cudaMemcpyHostToDevice, stream.value));
        Buffer<Region> a, b;
        a.reserve(1);
        Region zero{0, 0, 0};
        checked(cudaMemcpyAsync(a.data, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream.value));
        Buffer<uint32_t> count;
        count.reserve(1);
        uint32_t active = 1;
        int levels = 0;
        for (int n = root_size; n > 4; n /= 2)
            ++levels;
        int level = 0;
        Grid grid{{origin[0], origin[1], origin[2]}, voxel_size, {dimensions[0], dimensions[1], dimensions[2]}, cutoff};
        for (int n = root_size; n > 4 && active; n /= 2) {
            cancel();
            b.reserve(std::min(size_t(max_regions), size_t(active) * 8));
            checked(cudaMemsetAsync(count.data, 0, sizeof(uint32_t), stream.value));
            split_regions<<<(active + 127) / 128, 128, 0, stream.value>>>(a.data, active, n, grid, gs.data, bvh.data, b.data, count.data);
            checked(cudaGetLastError());
            checked(cudaMemcpyAsync(&active, count.data, sizeof(active), cudaMemcpyDeviceToHost, stream.value));
            checked(cudaStreamSynchronize(stream.value));
            if (active > max_regions)
                throw std::runtime_error("Collision GPU active-region budget exceeded (32 million regions)");
            std::swap(a.data, b.data);
            std::swap(a.capacity, b.capacity);
            if (progress)
                progress(.15f + .6f * float(++level) / std::max(levels, 1));
        }
        if (!active)
            return {};
        Buffer<CollisionBrick> output;
        output.reserve(active);
        checked(cudaMemsetAsync(count.data, 0, sizeof(uint32_t), stream.value));
        for (uint32_t offset = 0; offset < active;) {
            cancel();
            const auto batch = std::min(active - offset, 65536u);
            evaluate_bricks<<<batch, 64, 0, stream.value>>>(a.data, offset, grid, gs.data, bvh.data, output.data, count.data);
            checked(cudaGetLastError());
            checked(cudaStreamSynchronize(stream.value));
            offset += batch;
            if (progress)
                progress(.75f + .2f * float(offset) / active);
        }
        uint32_t occupied = 0;
        checked(cudaMemcpyAsync(&occupied, count.data, sizeof(occupied), cudaMemcpyDeviceToHost, stream.value));
        checked(cudaStreamSynchronize(stream.value));
        std::vector<CollisionBrick> result(occupied);
        if (occupied)
            checked(cudaMemcpyAsync(result.data(), output.data, size_t(occupied) * sizeof(CollisionBrick), cudaMemcpyDeviceToHost, stream.value));
        checked(cudaStreamSynchronize(stream.value));
        cancel();
        return result;
    }
} // namespace lfs::core::cuda
