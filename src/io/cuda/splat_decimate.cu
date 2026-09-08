/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "morton_encoding.hpp"
#include "splat_decimate_math.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

namespace lfs::io::decimate {
    namespace {
        using core::DataType;
        using core::Device;
        using core::Tensor;
        void check(cudaError_t error) {
            if (error != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(error));
        }
        struct Box {
            float low[3], high[3];
        };
        constexpr uint32_t leaf_size = 8;
        constexpr int threads = 128;
        __global__ void leaf_boxes(View v, const uint32_t* order, Box* boxes, uint32_t n, uint32_t leaves) {
            uint32_t leaf = blockIdx.x * blockDim.x + threadIdx.x;
            if (leaf >= leaves)
                return;
            Box b;
            for (int a = 0; a < 3; ++a) {
                b.low[a] = INFINITY;
                b.high[a] = -INFINITY;
            }
            for (uint32_t t = leaf * leaf_size; t < (leaf + 1) * leaf_size && t < n; ++t) {
                uint32_t i = order[t];
                for (int a = 0; a < 3; ++a) {
                    float x = v.pos[size_t(i) * 3 + a];
                    b.low[a] = fminf(b.low[a], x);
                    b.high[a] = fmaxf(b.high[a], x);
                }
            }
            boxes[leaves + leaf] = b;
        }
        __global__ void parent_boxes(Box* boxes, uint32_t begin) {
            uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
            if (t >= begin)
                return;
            uint32_t i = begin + t;
            Box a = boxes[i * 2], b = boxes[i * 2 + 1];
            for (int c = 0; c < 3; ++c) {
                a.low[c] = fminf(a.low[c], b.low[c]);
                a.high[c] = fmaxf(a.high[c], b.high[c]);
            }
            boxes[i] = a;
        }
        __device__ float box_distance(const float* p, const Box& b) {
            float d = 0;
            for (int a = 0; a < 3; ++a) {
                float delta = fmaxf(fmaxf(b.low[a] - p[a], p[a] - b.high[a]), 0);
                d += delta * delta;
            }
            return d;
        }
        __device__ double point_distance(const float* p, const float* q) {
            double distance = 0;
#pragma unroll
            for (int a = 0; a < 3; ++a) {
                double delta = double(p[a]) - q[a];
                distance += delta * delta;
            }
            return distance;
        }
        struct Grid {
            const uint32_t* offsets = nullptr;
            const int64_t* keys = nullptr;
            int dim = 0, shift = 0;
            float low[3], step[3], padding[3];
        };
        __global__ void grid_counts(const int64_t* keys, uint32_t* counts, uint32_t n, int shift) {
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n)
                atomicAdd(counts + (uint64_t(keys[i]) >> shift), 1u);
        }
        __global__ void neighbours(View v, const uint32_t* order, const Box* boxes, uint32_t n, uint32_t leaves, uint32_t* result, Grid grid) {
            uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
            if (t >= n)
                return;
            uint32_t i = order[t];
            const float* p = v.pos + size_t(i) * 3;
            constexpr int k = knn_k;
            float distances[knn_k];
            uint32_t ids[knn_k];
            double worst = INFINITY;
            float search_limit = INFINITY;
            uint32_t worst_id = invalid;
#pragma unroll
            for (int a = 0; a < k; ++a) {
                distances[a] = INFINITY;
                ids[a] = invalid;
            }
            const auto consider = [&](uint32_t j) {
                if (j == i)
                    return;
                float approximate = 0;
                for (int a = 0; a < 3; ++a) {
                    float delta = p[a] - v.pos[size_t(j) * 3 + a];
                    approximate += delta * delta;
                }
                if (approximate > search_limit)
                    return;
                double d2 = point_distance(p, v.pos + size_t(j) * 3);
                if (d2 > worst || (d2 == worst && j >= worst_id))
                    return;
                // Rounded keys are monotone; resolve equal keys in double.
                const auto farther = [&](float da, uint32_t ia, float db, uint32_t ib) {
                    if (da != db)
                        return da > db;
                    if (ia == invalid || ib == invalid)
                        return ia > ib;
                    const double xa = point_distance(p, v.pos + size_t(ia) * 3);
                    const double xb = point_distance(p, v.pos + size_t(ib) * 3);
                    return xa > xb || (xa == xb && ia > ib);
                };
                const float key = d2;
                int slot = 0;
                while (slot * 2 + 1 < k) {
                    int child = slot * 2 + 1;
                    if (child + 1 < k && farther(distances[child + 1], ids[child + 1], distances[child], ids[child]))
                        ++child;
                    if (!farther(distances[child], ids[child], key, j))
                        break;
                    distances[slot] = distances[child];
                    ids[slot] = ids[child];
                    slot = child;
                }
                distances[slot] = key;
                ids[slot] = j;
                worst = ids[0] == invalid ? INFINITY : point_distance(p, v.pos + size_t(ids[0]) * 3);
                // Expand float broad-phase bounds by over 16 ulps plus upward
                // conversion; the absolute allowance covers subnormal squares.
                // Accepted distances and ties are evaluated in double.
                search_limit = __double2float_ru(worst * 1.000002 + 0x1p-146);
                worst_id = ids[0];
            };
            if (grid.dim) {
                const uint32_t cell = uint64_t(grid.keys[t]) >> grid.shift;
                int center[3] = {};
#pragma unroll
                for (int bit = 0; bit < 7; ++bit)
                    for (int axis = 0; axis < 3; ++axis)
                        center[axis] |= ((cell >> (bit * 3 + axis)) & 1) << bit;
                for (int radius = 0; radius < grid.dim; ++radius) {
                    for (int z = max(0, center[2] - radius); z <= min(grid.dim - 1, center[2] + radius); ++z)
                        for (int y = max(0, center[1] - radius); y <= min(grid.dim - 1, center[1] + radius); ++y)
                            for (int x = max(0, center[0] - radius); x <= min(grid.dim - 1, center[0] + radius); ++x) {
                                if (radius && max(abs(x - center[0]), max(abs(y - center[1]), abs(z - center[2]))) != radius)
                                    continue;
                                const auto c = morton_encode(x, y, z);
                                const uint32_t begin = grid.offsets[c], end = grid.offsets[c + 1];
                                if (begin == end)
                                    continue;
                                const int coordinates[3] = {x, y, z};
                                Box bounds;
                                for (int axis = 0; axis < 3; ++axis) {
                                    bounds.low[axis] = grid.low[axis] + coordinates[axis] * grid.step[axis] - grid.padding[axis];
                                    bounds.high[axis] = grid.low[axis] + (coordinates[axis] + 1) * grid.step[axis] + grid.padding[axis];
                                }
                                if (box_distance(p, bounds) > search_limit)
                                    continue;
                                for (uint32_t row = begin; row < end; ++row)
                                    consider(order[row]);
                            }
                    // Lower bound to every unvisited cell. Expanded cell faces
                    // cover float normalization/reconstruction rounding. Exact
                    // distance and input-ID comparisons still decide all neighbours.
                    float outside = INFINITY;
                    for (int axis = 0; axis < 3; ++axis) {
                        if (center[axis] - radius > 0)
                            outside = fminf(outside, fmaxf(0, p[axis] - (grid.low[axis] + (center[axis] - radius) * grid.step[axis] + grid.padding[axis])));
                        if (center[axis] + radius + 1 < grid.dim)
                            outside = fminf(outside, fmaxf(0, grid.low[axis] + (center[axis] + radius + 1) * grid.step[axis] - grid.padding[axis] - p[axis]));
                    }
                    if (outside * outside > search_limit)
                        break;
                }
            } else {
                // At most one deferred sibling per depth; balanced heap depth <= 29.
                uint32_t stack[32];
                int sp = 0;
                stack[sp++] = 1;
                while (sp) {
                    uint32_t node = stack[--sp];
                    if (box_distance(p, boxes[node]) > search_limit)
                        continue;
                    if (node >= leaves) {
                        uint32_t begin = (node - leaves) * leaf_size;
                        for (uint32_t t2 = begin; t2 < begin + leaf_size && t2 < n; ++t2) {
                            uint32_t j = order[t2];
                            consider(j);
                        }
                    } else {
                        uint32_t left = node * 2, right = left + 1;
                        float a = box_distance(p, boxes[left]), b = box_distance(p, boxes[right]);
                        if (a < b) {
                            if (b <= search_limit)
                                stack[sp++] = right;
                            if (a <= search_limit)
                                stack[sp++] = left;
                        } else {
                            if (a <= search_limit)
                                stack[sp++] = left;
                            if (b <= search_limit)
                                stack[sp++] = right;
                        }
                    }
                }
            }
            for (int a = 0; a < k; ++a)
                result[size_t(i) * k + a] = ids[a];
        }
        __global__ void make_cache(View v, Cache* cache, uint32_t n) {
            uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n)
                cache[i] = cache_one(v, i);
        }
        __global__ void candidates(View v, const Cache* cache, const uint32_t* order, const uint32_t* nb, uint32_t n, uint32_t* idx, float* costs) {
            uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
            if (t >= n)
                return;
            uint32_t i = order[t];
            constexpr int knn = knn_k, k = candidates_k;
            float best[k];
            uint32_t ids[k];
            for (int a = 0; a < k; ++a) {
                best[a] = INFINITY;
                ids[a] = invalid;
            }
            for (int a = 0; a < knn; ++a) {
                uint32_t j = nb[size_t(i) * knn + a];
                if (j == invalid)
                    continue;
                float cost = edge(v, cache, i, j);
                if (!isfinite(cost) || cost > best[k - 1] || (cost == best[k - 1] && j >= ids[k - 1]))
                    continue;
                int at = k - 1;
                while (at > 0 && (cost < best[at - 1] || (cost == best[at - 1] && j < ids[at - 1]))) {
                    best[at] = best[at - 1];
                    ids[at] = ids[at - 1];
                    --at;
                }
                best[at] = cost;
                ids[at] = j;
            }
            for (int a = 0; a < k; ++a) {
                idx[size_t(i) * k + a] = ids[a];
                costs[size_t(i) * k + a] = best[a];
            }
        }
        __global__ void keep_flags(const int* groups, const uint32_t* minimum, uint32_t* flags, uint32_t n) {
            uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                int g = groups[i];
                flags[i] = g < 0 || i == minimum[g];
            }
        }
        __global__ void merge(View v, View out, const int* groups, const uint32_t* minimum, const uint32_t* members, const uint32_t* offsets, const uint32_t* rows, uint32_t n) {
            uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            int g = groups[i];
            if (g < 0)
                copy_one(v, i, out, rows[i]);
            else if (i == minimum[g])
                merge_one(v, members + offsets[g], offsets[g + 1] - offsets[g], out, rows[i]);
        }
        template <class T>
        Tensor upload(const std::vector<T>& values) {
            auto t = Tensor::empty({values.size()}, Device::CUDA, DataType::Int32);
            if (!values.empty())
                check(cudaMemcpy(t.template ptr<T>(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
            return t;
        }
    } // namespace
    Candidates gpu_candidates(const Data& d) {
        constexpr int knn = knn_k, k = candidates_k;
        Tensor keys;
        auto order = morton_sort_indices_for_positions(d.pos, &keys);
        if (!order.is_valid())
            throw std::runtime_error("decimation Morton sort failed");
        uint32_t leaves = 1;
        while (leaves < (d.n + leaf_size - 1) / leaf_size)
            leaves *= 2;
        auto boxes = Tensor::empty({size_t(leaves) * 2 * sizeof(Box) / 4}, Device::CUDA);
        auto bp = reinterpret_cast<Box*>(boxes.ptr<float>());
        auto ids = reinterpret_cast<const uint32_t*>(order.ptr<int32_t>());
        leaf_boxes<<<(leaves + threads - 1) / threads, threads>>>(d.view(), ids, bp, d.n, leaves);
        check(cudaGetLastError());
        for (uint32_t start = leaves / 2; start; start /= 2) {
            parent_boxes<<<(start + threads - 1) / threads, threads>>>(bp, start);
            check(cudaGetLastError());
        }
        Grid grid{};
        Tensor offsets;
        if (d.n >= 4096) {
            Box bounds;
            check(cudaMemcpy(&bounds, bp + 1, sizeof(bounds), cudaMemcpyDeviceToHost));
            float smallest = INFINITY, largest = 0;
            for (int axis = 0; axis < 3; ++axis) {
                const float span = bounds.high[axis] - bounds.low[axis];
                smallest = std::min(smallest, span);
                largest = std::max(largest, span);
            }
            if (std::isfinite(largest) && smallest > 0 && largest < smallest * 4) {
                int bits = 1;
                while (bits < 7 && (size_t{1} << (3 * (bits + 1))) <= d.n / 4)
                    ++bits;
                grid.dim = 1 << bits;
                grid.shift = 63 - 3 * bits;
                const size_t cells = size_t{1} << (bits * 3);
                offsets = Tensor::zeros({cells + 1}, Device::CUDA, DataType::Int32);
                auto counts = reinterpret_cast<uint32_t*>(offsets.ptr<int32_t>());
                grid_counts<<<(d.n + threads - 1) / threads, threads>>>(keys.ptr<int64_t>(), counts, d.n, grid.shift);
                check(cudaGetLastError());
                thrust::device_ptr<uint32_t> first(counts);
                const auto maximum = thrust::reduce(first, first + cells, 0u, thrust::maximum<uint32_t>());
                if (maximum <= 128) {
                    thrust::exclusive_scan(first, first + cells + 1, first);
                    grid.offsets = counts;
                    grid.keys = keys.ptr<int64_t>();
                    for (int axis = 0; axis < 3; ++axis) {
                        grid.low[axis] = bounds.low[axis];
                        grid.step[axis] = (bounds.high[axis] - bounds.low[axis]) / grid.dim;
                        // Over 16 float ulps for normalization and face reconstruction;
                        // the absolute allowance also covers subnormal arithmetic.
                        grid.padding[axis] = (std::abs(bounds.low[axis]) + std::abs(bounds.high[axis]) + largest) * 0.000002f + 0x1p-120f;
                    }
                } else {
                    grid.dim = 0; // Dense/clustered scenes keep exact BVH pruning.
                }
            }
        }
        auto nb = Tensor::empty({d.n * knn}, Device::CUDA, DataType::Int32);
        auto np = reinterpret_cast<uint32_t*>(nb.ptr<int32_t>());
        neighbours<<<(d.n + threads - 1) / threads, threads>>>(d.view(), ids, bp, d.n, leaves, np, grid);
        check(cudaGetLastError());
        auto cache = Tensor::empty({d.n * sizeof(Cache) / 4}, Device::CUDA);
        auto cp = reinterpret_cast<Cache*>(cache.ptr<float>());
        make_cache<<<(d.n + threads - 1) / threads, threads>>>(d.view(), cp, d.n);
        check(cudaGetLastError());
        auto idx = Tensor::empty({d.n * k}, Device::CUDA, DataType::Int32);
        auto cost = Tensor::empty({d.n * k}, Device::CUDA);
        candidates<<<(d.n + threads - 1) / threads, threads>>>(d.view(), cp, ids, np, d.n, reinterpret_cast<uint32_t*>(idx.ptr<int32_t>()), cost.ptr<float>());
        check(cudaGetLastError());
        Candidates out{std::vector<uint32_t>(d.n * k), std::vector<float>(d.n * k)};
        check(cudaMemcpy(out.idx.data(), idx.ptr<int32_t>(), out.idx.size() * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(out.cost.data(), cost.ptr<float>(), out.cost.size() * 4, cudaMemcpyDeviceToHost));
        return out;
    }
    Data gpu_merge(const Data& d, const Selection& s) {
        auto groups = upload(s.member_group), minimum = upload(s.minimum), members = upload(s.members), offsets = upload(s.offsets);
        auto rows = Tensor::empty({d.n}, Device::CUDA, DataType::Int32);
        auto rp = reinterpret_cast<uint32_t*>(rows.ptr<int32_t>());
        auto up = [](const Tensor& t) { return reinterpret_cast<const uint32_t*>(t.ptr<int32_t>()); };
        keep_flags<<<(d.n + threads - 1) / threads, threads>>>(groups.ptr<int>(), up(minimum), rp, d.n);
        check(cudaGetLastError());
        thrust::device_ptr<uint32_t> first(rp);
        thrust::exclusive_scan(first, first + d.n, first);
        auto out = allocate(d.n - s.removed, d.rest, Device::CUDA);
        merge<<<(d.n + threads - 1) / threads, threads>>>(d.view(), out.view(), groups.ptr<int>(), up(minimum), up(members), up(offsets), rp, d.n);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        return out;
    }
} // namespace lfs::io::decimate
