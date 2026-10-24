/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "collision_voxelization.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace lfs::core::cuda {
    namespace {
        constexpr uint32_t max_regions = 1'048'576; // Bounded per-batch storage, not a scene-wide frontier cap.
        constexpr size_t max_output_bricks = 64'000'000;
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
        __device__ uint64_t morton(Region p) {
            uint64_t result = 0;
            for (int bit = 0; bit < 19; ++bit) {
                result |= uint64_t((p.x >> 2 >> bit) & 1) << (3 * bit);
                result |= uint64_t((p.y >> 2 >> bit) & 1) << (3 * bit + 1);
                result |= uint64_t((p.z >> 2 >> bit) & 1) << (3 * bit + 2);
            }
            return result;
        }
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
                if (!overlaps(node, p, n))
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
                        const float far = fabsf(q) + r + (fabsf(q) + r + 1) * 2e-6f;
                        far2 += far * far;
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
                    solids[row] = {morton(p) | (uint64_t(exponent) << 57), ~uint64_t(0)};
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
        // Depth-first GPU batches bound the live frontier independently of
        // total scene size. Parent queues remain on-device while children run;
        // only compact occupied records and counters return to the host.
        struct Frame {
            Buffer<Region> regions;
            uint32_t count = 0, offset = 0;
            int edge = 0;
        };
        std::vector<std::unique_ptr<Frame>> frames;
        for (int n = root_size;; n = std::max(4, n / 2)) {
            frames.push_back(std::make_unique<Frame>());
            frames.back()->edge = n;
            if (n <= 4)
                break;
        }
        frames[0]->regions.reserve(1);
        frames[0]->count = 1;
        Region zero{0, 0, 0};
        checked(cudaMemcpyAsync(frames[0]->regions.data, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream.value));
        Buffer<uint32_t> count, solid_count;
        count.reserve(1);
        solid_count.reserve(1);
        Buffer<CollisionBrick> output;
        std::vector<CollisionBrick> result;
        Grid grid{{origin[0], origin[1], origin[2]}, voxel_size, {dimensions[0], dimensions[1], dimensions[2]}, cutoff};
        auto append = [&](uint32_t n) {
            if (n > max_regions || result.size() + n > max_output_bricks)
                throw std::runtime_error("Collision GPU occupied-output budget exceeded (64 million compressed regions)");
            if (!n)
                return;
            const auto begin = result.size();
            result.resize(begin + n);
            checked(cudaMemcpyAsync(result.data() + begin, output.data, size_t(n) * sizeof(CollisionBrick), cudaMemcpyDeviceToHost, stream.value));
            checked(cudaStreamSynchronize(stream.value));
        };
        const double total = double(root_size) * root_size * root_size;
        double completed = 0;
        int depth = 0;
        while (depth >= 0) {
            cancel();
            auto& frame = *frames[depth];
            if (frame.offset == frame.count) {
                --depth;
                continue;
            }
            if (frame.edge > 4) {
                const auto batch = std::min(frame.count - frame.offset, max_regions / 8);
                auto& child = *frames[depth + 1];
                child.regions.reserve(size_t(batch) * 8);
                output.reserve(size_t(batch) * 8);
                checked(cudaMemsetAsync(count.data, 0, sizeof(uint32_t), stream.value));
                checked(cudaMemsetAsync(solid_count.data, 0, sizeof(uint32_t), stream.value));
                split_regions<<<(uint64_t(batch) * 8 + 127) / 128, 128, 0, stream.value>>>(
                    frame.regions.data, frame.offset, batch, frame.edge, grid, gs.data, bvh.data,
                    child.regions.data, count.data, output.data, solid_count.data);
                checked(cudaGetLastError());
                uint32_t uniform = 0;
                checked(cudaMemcpyAsync(&child.count, count.data, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.value));
                checked(cudaMemcpyAsync(&uniform, solid_count.data, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.value));
                checked(cudaStreamSynchronize(stream.value));
                append(uniform);
                const double half = frame.edge / 2;
                completed += (uint64_t(batch) * 8 - child.count) * half * half * half;
                frame.offset += batch;
                child.offset = 0;
                if (child.count)
                    ++depth;
            } else {
                const auto batch = std::min(frame.count - frame.offset, 65536u);
                output.reserve(batch);
                checked(cudaMemsetAsync(count.data, 0, sizeof(uint32_t), stream.value));
                evaluate_bricks<<<batch, 64, 0, stream.value>>>(frame.regions.data, frame.offset, grid, gs.data, bvh.data, output.data, count.data);
                checked(cudaGetLastError());
                uint32_t occupied = 0;
                checked(cudaMemcpyAsync(&occupied, count.data, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.value));
                checked(cudaStreamSynchronize(stream.value));
                append(occupied);
                completed += batch * double(frame.edge) * frame.edge * frame.edge;
                frame.offset += batch;
            }
            if (progress)
                progress(.15f + .8f * float(std::clamp(completed / total, 0., 1.)));
        }
        cancel();
        return result;
    }
} // namespace lfs::core::cuda
