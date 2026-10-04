/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "core/tensor/backend/cuda/kernels/cub_workspace.hpp"
#include "internal/nearest_point.hpp"
#include "internal/point_spatial.hpp"
#include "internal/point_tree.hpp"
#include "internal/triangle_tree.hpp"
#include "tensor_spatial.hpp"

#include <algorithm>
#include <bit>
#include <cub/cub.cuh>

namespace lfs::core::tensor_ops {
    namespace {
        constexpr int kBlockSize = 256;

        using namespace internal;

        __global__ void build(const float* points, const uint8_t* references, int32_t* heads, int32_t* next,
                              const size_t count, const uint32_t bucket_mask, const float radius,
                              const size_t begin) {
            const size_t i = begin + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= count || (references && !references[i])) {
                return;
            }
            const float* p = points + i * 3;
            if (!finite_point(p)) {
                return;
            }
            const auto bucket = hash_cell(cell(p[0], radius), cell(p[1], radius), cell(p[2], radius), bucket_mask);
            next[i] = atomicExch(heads + bucket, static_cast<int32_t>(i));
        }

        __global__ void nearest_query(const float* queries, const float* targets, const int32_t* heads,
                                      const int32_t* next, int32_t* output, size_t nq, size_t nt, uint32_t mask, float width) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < nq)
                output[i] = nearestPoint(queries + 3 * i, targets, heads, next, int(nt), mask, width);
        }
        __global__ void coverage_query(const float* points, const float* cameras, int32_t* output, size_t n, size_t count, float maximum) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n)
                output[i] = cameraFrustumCount(points + 3 * i, cameras, int(count), maximum);
        }

        __global__ void query(const float* points, const uint8_t* references, const int32_t* heads,
                              const int32_t* next, bool* output, const size_t count,
                              const uint32_t bucket_mask, const float radius, const bool exclude_self,
                              const size_t begin, const uint8_t* queries) {
            const size_t i = begin + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= count) {
                return;
            }
            output[i] = (!queries || queries[i]) && pointHasNeighbor(points, references, heads, next, i, bucket_mask, radius, exclude_self);
        }
        template <class T>
        __global__ void query_min(const float* points, const T* values, const int32_t* heads,
                                  const int32_t* next, T* output, const size_t count,
                                  const uint32_t bucket_mask, const float radius, const float* radii) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < count)
                output[i] = pointNeighborMin(points, values, heads, next, i, bucket_mask, radius, radii);
        }
        // Hooks the larger root under the smaller one for every neighbour pair (lock-free union-find). A failed
        // exchange returns the root's new, smaller parent, so each retry climbs and the loop ends.
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
                parent[i] = finalComponentRoot(parent, static_cast<int32_t>(i));
        }
        __global__ void query_spacing(const float* points, const int32_t* heads, const int32_t* next,
                                      float* output, size_t count, uint32_t bucket_mask, float radius) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < count)
                output[i] = pointNeighborSpacing(points, heads, next, i, bucket_mask, radius);
        }

        // Counting walks every candidate in 27 cells. Sorting the references by bucket makes each
        // cell one contiguous run instead of a linked list scattered over all points.
        __global__ void bucket_keys(const float* points, const uint8_t* references, uint32_t* keys, int32_t* order,
                                    const size_t count, const uint32_t bucket_mask, const float cell_size) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= count)
                return;
            const float* p = points + i * 3;
            keys[i] = references[i] && finite_point(p)
                          ? hash_cell(cell(p[0], cell_size), cell(p[1], cell_size), cell(p[2], cell_size), bucket_mask)
                          : bucket_mask + 1;
            order[i] = static_cast<int32_t>(i);
        }

        __global__ void bucket_starts(const float* points, const uint32_t* keys, const int32_t* order, int32_t* starts,
                                      float* sorted, const size_t count, const uint32_t bucket_mask) {
            const size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (k >= count || keys[k] > bucket_mask)
                return;
            if (k == 0 || keys[k - 1] != keys[k])
                starts[keys[k]] = static_cast<int32_t>(k);
            const float* p = points + static_cast<size_t>(order[k]) * 3;
            sorted[k * 3] = p[0];
            sorted[k * 3 + 1] = p[1];
            sorted[k * 3 + 2] = p[2];
        }

        // Cells are half the radius wide, so the 5x5x5 block around a query, less the cells whose box
        // lies beyond the radius, holds about a third of the candidates of radius-wide cells.
        constexpr int kCellsPerRadius = 2;

        // Threads take queries in sorted order, so a warp shares the cells it scans.
        __global__ void sorted_query_counts(const float* points, const uint32_t* keys, const int32_t* order,
                                            const int32_t* starts, const float* sorted, int32_t* output,
                                            const size_t count, const uint32_t bucket_mask, const float radius,
                                            const float cell_size, const int32_t max_count, const uint8_t* queries) {
            const size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (k >= count)
                return;
            const auto i = static_cast<size_t>(order[k]);
            const float* p = points + i * 3;
            if ((queries && !queries[i]) || !finite_point(p)) {
                output[i] = 0;
                return;
            }
            const int x = cell(p[0], cell_size);
            const int y = cell(p[1], cell_size);
            const int z = cell(p[2], cell_size);
            // Prune with a margin, since cells come from rounded divisions. Cell indices clamp far from the
            // origin, where they no longer describe where a point is, so pruning stops there.
            const float reach = radius * 1.001f + cell_size * 1e-3f;
            constexpr int kUnclamped = 268435456 - 2 * kCellsPerRadius;
            const bool prune = abs(x) < kUnclamped && abs(y) < kUnclamped && abs(z) < kUnclamped;
            const auto gap = [&](const float value, const int index) {
                const float low = static_cast<float>(index) * cell_size;
                return fmaxf(0.0f, fmaxf(low - value, value - (low + cell_size)));
            };
            int32_t found = 0;
            constexpr int side = 2 * kCellsPerRadius + 1;
            constexpr int centre = (side * side * side) / 2;
            for (int slot = 0; slot < side * side * side; ++slot) {
                const int neighbor = slot == 0 ? centre : (slot <= centre ? slot - 1 : slot);
                const int cx = x + neighbor % side - kCellsPerRadius;
                const int cy = y + neighbor / side % side - kCellsPerRadius;
                const int cz = z + neighbor / (side * side) - kCellsPerRadius;
                const float gx = gap(p[0], cx), gy = gap(p[1], cy), gz = gap(p[2], cz);
                if (prune && gx * gx + gy * gy + gz * gz > reach * reach)
                    continue;
                const auto bucket = hash_cell(cx, cy, cz, bucket_mask);
                for (int32_t j = starts[bucket]; j >= 0 && static_cast<size_t>(j) < count && keys[j] == bucket; ++j) {
                    const float* q = sorted + static_cast<size_t>(j) * 3;
                    // Hash collisions must not count the same reference twice.
                    if (cell(q[0], cell_size) == cx && cell(q[1], cell_size) == cy && cell(q[2], cell_size) == cz &&
                        within(p, q, radius) && static_cast<size_t>(order[j]) != i && ++found == max_count) {
                        output[i] = found;
                        return;
                    }
                }
            }
            output[i] = found;
        }
    } // namespace

    void launch_point_neighbor_spacing(const float* points, const uint8_t* references, int32_t* heads,
                                       int32_t* next, float* output, size_t count, size_t buckets,
                                       float cell_width, cudaStream_t stream) {
        const auto blocks = static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
        const auto mask = static_cast<uint32_t>(buckets - 1);
        build<<<blocks, kBlockSize, 0, stream>>>(points, references, heads, next, count, mask, cell_width, 0);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.point_neighbor_spacing.build");
        query_spacing<<<blocks, kBlockSize, 0, stream>>>(points, heads, next, output, count, mask, cell_width);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.point_neighbor_spacing.query");
    }

    void launch_radius_neighbor_counts(const float* points, const uint8_t* references, int32_t* heads,
                                       int32_t* next, int32_t* output, const size_t count, const size_t buckets,
                                       const float radius, const int32_t max_count, const uint8_t* queries, const cudaStream_t stream) {
        const auto bucket_mask = static_cast<uint32_t>(buckets - 1);
        const auto blocks = static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
        // heads arrives filled with -1 and receives each bucket's first sorted position; next holds
        // the unsorted order.
        ScopedDeviceBuffer keys(count * sizeof(uint32_t), stream, "tensor.radius_neighbor_counts.keys");
        ScopedDeviceBuffer sorted_keys(count * sizeof(uint32_t), stream, "tensor.radius_neighbor_counts.sorted_keys");
        ScopedDeviceBuffer sorted_order(count * sizeof(int32_t), stream, "tensor.radius_neighbor_counts.order");
        ScopedDeviceBuffer sorted_points(count * 3 * sizeof(float), stream, "tensor.radius_neighbor_counts.points");
        const float cell_size = radius / static_cast<float>(kCellsPerRadius);
        bucket_keys<<<blocks, kBlockSize, 0, stream>>>(points, references, keys.as<uint32_t>(), next, count, bucket_mask, cell_size);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbor_counts.keys");
        const int end_bit = std::bit_width(static_cast<uint32_t>(buckets));
        run_cub_operation("cub::DeviceRadixSort::SortPairs", stream, [&](void* workspace, size_t& workspace_bytes) {
            return cub::DeviceRadixSort::SortPairs(workspace, workspace_bytes, keys.as<uint32_t>(),
                                                   sorted_keys.as<uint32_t>(), next, sorted_order.as<int32_t>(),
                                                   static_cast<int>(count), 0, end_bit, stream);
        });
        bucket_starts<<<blocks, kBlockSize, 0, stream>>>(points, sorted_keys.as<uint32_t>(), sorted_order.as<int32_t>(),
                                                         heads, sorted_points.as<float>(), count, bucket_mask);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbor_counts.starts");
        sorted_query_counts<<<blocks, kBlockSize, 0, stream>>>(points, sorted_keys.as<uint32_t>(), sorted_order.as<int32_t>(),
                                                               heads, sorted_points.as<float>(), output, count, bucket_mask,
                                                               radius, cell_size, max_count, queries);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbor_counts.query");
    }

    namespace {
        __global__ void point_tree_counts(const float* points, const float* sorted, const float* boxes,
                                          const int32_t* visit, const float* radii, const uint8_t* queries,
                                          int32_t* output, const PointTreeProgram tree) {
            const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (t >= tree.points)
                return;
            const auto i = static_cast<size_t>(visit[t]);
            const float* p = points + i * 3;
            output[i] = (queries && !queries[i]) || !finite_point(p)
                            ? 0
                            : pointTreeCount(sorted, boxes, tree, p, t < tree.references ? static_cast<int64_t>(t) : -1,
                                             radii[i], tree.max_count);
        }
    } // namespace

    namespace {
        __global__ void point_tree_union(const float* points, const float* sorted, const float* boxes,
                                         const float* box_radii, const int32_t* visit, const float* sorted_radii,
                                         const float* radii, int32_t* parent, const PointTreeProgram tree) {
            const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (t >= tree.references)
                return;
            const int32_t self = visit[t];
            int32_t mine = componentRoot(parent, self);
            pointTreeMutualNeighbors(sorted, boxes, box_radii, sorted_radii, tree, points + static_cast<size_t>(self) * 3,
                                     static_cast<int64_t>(t), radii[self], [&](const uint64_t j) {
                                         int32_t other = componentRoot(parent, visit[j]);
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
                                     });
        }
    } // namespace

    void launch_point_tree_components(const float* points, const float* sorted, const float* boxes, const float* box_radii,
                                      const int32_t* visit, const float* sorted_radii, const float* radii, int32_t* labels,
                                      const PointTreeProgram& tree, const cudaStream_t stream) {
        point_tree_union<<<(tree.references + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            points, sorted, boxes, box_radii, visit, sorted_radii, radii, labels, tree);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.point_tree_components.union");
        flatten_components<<<(tree.points + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(labels, tree.points);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.point_tree_components.flatten");
    }

    namespace {
        __global__ void point_tree_spacing(const float* points, const float* sorted, const float* boxes,
                                           const int32_t* visit, float* output, const PointTreeProgram tree) {
            const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (t >= tree.points)
                return;
            const auto i = static_cast<size_t>(visit[t]);
            output[i] = pointTreeSpacing(sorted, boxes, tree, points + i * 3,
                                         t < tree.references ? static_cast<int64_t>(t) : -1, tree.radius);
        }
    } // namespace

    void launch_point_tree_spacing(const float* points, const float* sorted, const float* boxes, const int32_t* visit,
                                   float* output, const PointTreeProgram& tree, const cudaStream_t stream) {
        point_tree_spacing<<<(tree.points + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            points, sorted, boxes, visit, output, tree);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.point_tree_spacing");
    }

    void launch_point_tree_counts(const float* points, const float* sorted, const float* boxes, const int32_t* visit,
                                  const float* radii, const uint8_t* queries, int32_t* output,
                                  const PointTreeProgram& tree, const cudaStream_t stream) {
        point_tree_counts<<<(tree.points + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            points, sorted, boxes, visit, radii, queries, output, tree);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.point_tree_counts");
    }

    namespace {
        __global__ void triangle_tree_parity(const float* points, const int32_t* visit, const float* triangles,
                                             const float* boxes, int32_t* output, const PointTreeProgram tree) {
            const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (t >= tree.points)
                return;
            const auto i = static_cast<size_t>(visit[t]);
            output[i] = triangleTreeParity(triangles, boxes, tree, points + i * 3);
        }
    } // namespace

    void launch_triangle_tree_parity(const float* points, const int32_t* visit, const float* triangles,
                                     const float* boxes, int32_t* output, const PointTreeProgram& tree,
                                     const cudaStream_t stream) {
        triangle_tree_parity<<<(tree.points + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            points, visit, triangles, boxes, output, tree);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.triangle_tree_parity");
    }

    void launch_nearest_point_indices(const float* queries, const float* targets, int32_t* heads, int32_t* next, int32_t* output,
                                      size_t nq, size_t nt, size_t buckets, float width, cudaStream_t stream) {
        build<<<(nt + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(targets, nullptr, heads, next, nt, uint32_t(buckets - 1), width, 0);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.nearest_point_indices.build");
        nearest_query<<<(nq + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(queries, targets, heads, next, output, nq, nt, uint32_t(buckets - 1), width);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.nearest_point_indices.query");
    }
    void launch_camera_frustum_counts(const float* points, const float* cameras, int32_t* output, size_t n, size_t count, float maximum, cudaStream_t stream) {
        coverage_query<<<(n + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(points, cameras, output, n, count, maximum);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.camera_frustum_counts");
    }

    void launch_radius_neighbor_min(const float* points, const void* values, const uint8_t value_is_float,
                                    const uint8_t* references, int32_t* heads, int32_t* next, void* output,
                                    const size_t count, const size_t buckets, const float radius,
                                    const float* radii, const cudaStream_t stream) {
        const auto bucket_mask = static_cast<uint32_t>(buckets - 1);
        const auto blocks = static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
        build<<<blocks, kBlockSize, 0, stream>>>(points, references, heads, next, count, bucket_mask, radius, 0);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbor_min.build");
        if (value_is_float) {
            query_min<<<blocks, kBlockSize, 0, stream>>>(points, static_cast<const float*>(values), heads, next,
                                                         static_cast<float*>(output), count, bucket_mask, radius, radii);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbor_min.query_float");
        } else {
            query_min<<<blocks, kBlockSize, 0, stream>>>(points, static_cast<const int32_t*>(values), heads, next,
                                                         static_cast<int32_t*>(output), count, bucket_mask, radius, radii);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbor_min.query_int");
        }
    }

    void launch_radius_connected_components(const float* points, const uint8_t* references, int32_t* heads,
                                            int32_t* next, int32_t* labels, const size_t count, const size_t buckets,
                                            const float radius, const cudaStream_t stream) {
        const auto bucket_mask = static_cast<uint32_t>(buckets - 1);
        const auto blocks = static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
        build<<<blocks, kBlockSize, 0, stream>>>(points, references, heads, next, count, bucket_mask, radius, 0);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_connected_components.build");
        union_components<<<blocks, kBlockSize, 0, stream>>>(points, references, heads, next, labels, count, bucket_mask, radius);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_connected_components.union");
        flatten_components<<<blocks, kBlockSize, 0, stream>>>(labels, count);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_connected_components.flatten");
    }

    void launch_radius_neighbors(const float* points, const uint8_t* references, int32_t* heads,
                                 int32_t* next, bool* output, const size_t count, const size_t buckets,
                                 const float radius, const bool exclude_self, const uint8_t* queries, const cudaStream_t stream) {
        const auto bucket_mask = static_cast<uint32_t>(buckets - 1);
        const size_t batch = exclude_self ? 8192 : count;
        for (size_t begin = 0; begin < count; begin += batch) {
            const auto end = std::min(begin + batch, count);
            const auto blocks = static_cast<unsigned int>((end - begin + kBlockSize - 1) / kBlockSize);
            build<<<blocks, kBlockSize, 0, stream>>>(points, references, heads, next, end, bucket_mask, radius, begin);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbors.build");
            if (exclude_self)
                LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        const size_t query_batch = exclude_self ? 8192 : count;
        for (size_t begin = 0; begin < count; begin += query_batch) {
            const auto end = std::min(begin + query_batch, count);
            const auto query_blocks = static_cast<unsigned int>((end - begin + kBlockSize - 1) / kBlockSize);
            query<<<query_blocks, kBlockSize, 0, stream>>>(points, references, heads, next, output, end, bucket_mask, radius, exclude_self, begin, queries);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbors.query");
            if (exclude_self)
                LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
} // namespace lfs::core::tensor_ops
