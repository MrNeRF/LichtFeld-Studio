/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "collision_sparse.hpp"
#include <atomic>
#include <cmath>
#include <format>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>

namespace lfs::core::collision_detail {
    namespace {
        void check(std::stop_token stop) {
            if (stop.stop_requested())
                throw std::runtime_error("Collision generation cancelled");
        }
        glm::vec3 extent(const CollisionGaussian& g) {
            glm::vec3 result;
            for (int a = 0; a < 3; ++a)
                result[a] = 3 * glm::length(glm::vec3(g.axes[0][a], g.axes[1][a], g.axes[2][a]));
            return result;
        }
        template <typename Fn>
        void partition(glm::ivec3 lo, glm::ivec3 hi, Fn&& fn) {
            if (glm::any(glm::greaterThanEqual(lo, hi)) || fn(lo, hi))
                return;
            auto delta = hi - lo;
            int axis = delta.x >= delta.y && delta.x >= delta.z ? 0 : delta.y >= delta.z ? 1
                                                                                         : 2;
            int mid = lo[axis] + delta[axis] / 2;
            auto left = hi, right = lo;
            left[axis] = right[axis] = mid;
            partition(lo, left, fn);
            partition(right, hi, fn);
        }
        Tree capsuleDilate(const Tree& source, float height, float radius, float size, std::stop_token stop) {
            Tree result(source.dimensions, stop);
            const int segment = int(std::ceil((double(height) / 2 - radius) / size));
            if (segment > source.dimensions.y * 2)
                throw std::runtime_error("Carve capsule is taller than collision bounds");
            const double r = double(radius) / size + std::sqrt(3.0) / 2;
            const double threshold = r * r;
            uint64_t work = 0;
            source.leaves([&](const Leaf& leaf) {
                if (!leaf.solid)
                    return;
                auto min = leaf.min, max = glm::min(leaf.max(), source.dimensions) - 1;
                min.y -= segment;
                max.y += segment;
                auto lo = glm::max(min - int(std::ceil(r)), glm::ivec3(0));
                auto hi = glm::min(max + int(std::ceil(r)) + 1, source.dimensions);
                partition(lo, hi, [&](auto a, auto b) {
                    if ((++work & 4095) == 0)
                        check(stop);
                    if (work > 500'000'000)
                        throw std::runtime_error("Sparse capsule dilation work budget exceeded");
                    auto near = glm::max(glm::max(min - (b - 1), a - max), glm::ivec3(0));
                    if (glm::dot(glm::dvec3(near), glm::dvec3(near)) >= threshold)
                        return true;
                    auto far = glm::max(glm::max(min - a, (b - 1) - max), glm::ivec3(0));
                    if (glm::dot(glm::dvec3(far), glm::dvec3(far)) < threshold) {
                        result.setBox(a, b);
                        return true;
                    }
                    return false;
                });
            });
            result.compact();
            return result;
        }
        void floorFill(Tree& tree, float reach_world, float voxel_size, std::stop_token stop, int maximum_height) {
            // Only columns intersecting source occupancy are materialized.
            std::unordered_map<uint64_t, int> heights;
            auto key = [](int x, int z) { return uint64_t(uint32_t(x)) | (uint64_t(uint32_t(z)) << 32); };
            tree.leaves([&](const Leaf& l) {
                if (!l.solid || l.min.y > maximum_height)
                    return;
                auto end = glm::min(l.max(), tree.dimensions);
                for (int z = l.min.z; z < end.z; ++z) {
                    check(stop);
                    for (int x = l.min.x; x < end.x; ++x) {
                        auto [it, added] = heights.try_emplace(key(x, z), l.min.y);
                        if (!added)
                            it->second = std::min(it->second, l.min.y);
                        if (heights.size() > 8'000'000)
                            throw std::runtime_error("Sparse floor active-column memory budget exceeded");
                    }
                }
            });
            const int reach = int(std::min(std::ceil(2 * double(reach_world) / voxel_size),
                                           double(std::max(tree.dimensions.x, tree.dimensions.z))));
            std::map<int, std::vector<int>> rows, columns;
            for (auto [k, h] : heights) {
                int x = uint32_t(k), z = k >> 32;
                rows[z].push_back(x);
                columns[x].push_back(z);
            }
            for (auto& [x, zs] : columns)
                std::sort(zs.begin(), zs.end());
            std::vector<std::pair<uint64_t, int>> patches;
            uint64_t work = 0;
            for (auto& [z, xs] : rows) {
                check(stop);
                std::sort(xs.begin(), xs.end());
                for (size_t i = 1; i < xs.size(); ++i) {
                    const int left = xs[i - 1], right = xs[i];
                    for (int x = std::max(left + 1, right - reach); x < right && x <= left + reach; ++x) {
                        if (++work > 500'000'000)
                            throw std::runtime_error("Sparse floor patch work budget exceeded");
                        auto col = columns.find(x);
                        if (col == columns.end())
                            continue;
                        auto upper = std::upper_bound(col->second.begin(), col->second.end(), z);
                        if (upper == col->second.end() || upper == col->second.begin())
                            continue;
                        int below = *std::prev(upper), above = *upper;
                        if (z - below > reach || above - z > reach)
                            continue;
                        patches.emplace_back(key(x, z), std::min({heights.at(key(left, z)), heights.at(key(right, z)),
                                                                  heights.at(key(x, below)), heights.at(key(x, above))}));
                        if (patches.size() + heights.size() > 8'000'000)
                            throw std::runtime_error("Sparse floor active-column memory budget exceeded");
                    }
                }
            }
            for (auto [k, h] : patches)
                heights.emplace(k, h);
            auto spread = [](uint32_t coordinate) {
                uint64_t value = coordinate;
                value = (value | (value << 16)) & 0x0000ffff0000ffffULL;
                value = (value | (value << 8)) & 0x00ff00ff00ff00ffULL;
                value = (value | (value << 4)) & 0x0f0f0f0f0f0f0f0fULL;
                value = (value | (value << 2)) & 0x3333333333333333ULL;
                return (value | (value << 1)) & 0x5555555555555555ULL;
            };
            std::vector<std::pair<uint64_t, int>> ordered;
            ordered.reserve(heights.size());
            for (auto [k, h] : heights)
                ordered.emplace_back(spread(uint32_t(k)) | (spread(uint32_t(k >> 32)) << 1), h);
            std::sort(ordered.begin(), ordered.end());
            const auto active_columns = ordered.size();
            heights.clear();
            heights.rehash(0);
            rows.clear();
            columns.clear();
            patches.clear();
            patches.shrink_to_fit();

            struct HeightNode {
                int min = 0, max = 0;
                uint32_t children = 0;
            };
            std::vector<HeightNode> height_tree(1);
            std::function<void(uint32_t, int, int, size_t, size_t)> build_heights;
            build_heights = [&](uint32_t index, int edge, int level, size_t begin, size_t end) {
                check(stop);
                if (begin == end)
                    return;
                int min = tree.dimensions.y, max = 0;
                for (auto i = begin; i < end; ++i) {
                    min = std::min(min, ordered[i].second);
                    max = std::max(max, ordered[i].second);
                }
                if (uint64_t(edge) * edge != end - begin)
                    min = 0;
                height_tree[index] = {min, max, 0};
                if (min == max)
                    return;
                if (height_tree.size() + 4 > 32'000'001)
                    throw std::runtime_error("Sparse floor height hierarchy memory budget exceeded");
                const auto base = uint32_t(height_tree.size());
                height_tree.resize(height_tree.size() + 4);
                height_tree[index].children = base;
                for (int child = 0; child < 4; ++child) {
                    auto first = begin;
                    while (begin < end && ((ordered[begin].first >> (2 * (level - 1))) & 3) == uint64_t(child))
                        ++begin;
                    build_heights(base + child, edge / 2, level - 1, first, begin);
                }
            };
            int level = 0;
            for (int n = tree.size; n > 1; n /= 2)
                ++level;
            build_heights(0, tree.size, level, 0, ordered.size());
            ordered.clear();
            ordered.shrink_to_fit();

            // Build the union directly: column-by-column mutation can exhaust the
            // node budget on temporary fragmentation even when the result is small.
            Tree filled(tree.dimensions, stop);
            std::function<void(uint32_t, uint32_t, uint32_t, int, int)> unite;
            unite = [&](uint32_t target, uint32_t source, uint32_t height_index, int y, int edge) {
                check(stop);
                const auto h = height_tree[height_index];
                if (source == 1 || y + edge <= h.min) {
                    filled.nodes[target] = 1;
                    return;
                }
                if (source == 0 && y >= h.max)
                    return;
                if (filled.nodes.size() + 8 > Tree::max_nodes)
                    throw std::runtime_error(
                        "Sparse floor canonical node memory budget exceeded: " +
                        std::to_string(filled.nodes.size()) + "/" + std::to_string(Tree::max_nodes) +
                        " words, " + std::to_string(active_columns) + " active columns, " +
                        std::to_string(height_tree.size()) + " height nodes; floor filling was not applied");
                const auto base = uint32_t(filled.nodes.size());
                filled.nodes.resize(filled.nodes.size() + 8);
                filled.nodes[target] = base + 2;
                for (int child = 0; child < 8; ++child)
                    unite(base + child, source <= 1 ? source : tree.nodes[source - 2 + child],
                          h.children ? h.children + (child & 1) + 2 * (child >> 2) : height_index,
                          y + ((child >> 1) & 1) * (edge / 2), edge / 2);
                const auto first = filled.nodes[base];
                if (first <= 1 && std::all_of(filled.nodes.begin() + base, filled.nodes.begin() + base + 8,
                                              [first](auto value) { return value == first; })) {
                    filled.nodes[target] = first;
                    filled.nodes.resize(base);
                }
            };
            unite(0, tree.nodes[0], 0, 0, tree.size);
            tree = std::move(filled);
        }
    } // namespace

    PreparedCollision prepareCollision(std::span<const CollisionGaussian> source, CollisionSettings settings,
                                       std::stop_token stop) {
        const double cutoff = -std::log1p(-double(settings.opacity));
        struct Bounds {
            glm::vec3 min, max;
            float opacity;
        };
        std::vector<Bounds> bounds;
        bounds.reserve(source.size());
        glm::vec3 lo(std::numeric_limits<float>::max()), hi(-lo);
        for (const auto& g : source) {
            check(stop);
            auto finite = [](glm::vec3 p) { return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z); };
            if (!finite(g.center) || !finite(g.axes[0]) || !finite(g.axes[1]) || !finite(g.axes[2]) ||
                !std::isfinite(g.opacity) || g.opacity < 0 || g.opacity > 1)
                throw std::invalid_argument("Non-finite or invalid Gaussian in collision source");
            if (g.opacity == 0 || std::abs(glm::determinant(g.axes)) < 1e-20f)
                continue;
            auto e = extent(g) + settings.voxel_size * 0.5f;
            if (!finite(e) || !finite(g.center - e) || !finite(g.center + e))
                throw std::invalid_argument("Gaussian collision support exceeds finite coordinate range");
            bounds.push_back({g.center - e, g.center + e, g.opacity});
            lo = glm::min(lo, bounds.back().min);
            hi = glm::max(hi, bounds.back().max);
        }
        // This pass only tightens the world bounds; it never filters the
        // Gaussians evaluated by the voxelizer. A sparse grid does not allocate
        // its empty AABB, so overlap proofs buy nothing when all supports already
        // fit the coordinate limit. Dense aerial scans otherwise spend hundreds
        // of millions of BVH visits proving optional bounds before CUDA starts.
        const auto support_origin = glm::floor(lo / settings.voxel_size) * settings.voxel_size - 2 * settings.voxel_size;
        const auto support_dimensions = glm::ceil((glm::dvec3(hi) - glm::dvec3(support_origin)) / double(settings.voxel_size)) + 3.0;
        const bool supports_fit = glm::all(glm::greaterThanEqual(support_dimensions, glm::dvec3(1))) &&
                                  glm::all(glm::lessThanEqual(support_dimensions, glm::dvec3(1 << 21)));
        if (!supports_fit) {
            lo = glm::vec3(std::numeric_limits<float>::max());
            hi = -lo;
            struct BvhNode {
                glm::vec3 min, max;
                size_t begin, end;
                uint32_t left = 0, right = 0;
            };
            std::vector<uint32_t> order(bounds.size());
            std::iota(order.begin(), order.end(), 0);
            std::vector<BvhNode> bvh;
            std::function<uint32_t(size_t, size_t)> buildBounds = [&](size_t begin, size_t end) {
                check(stop);
                const auto index = uint32_t(bvh.size());
                glm::vec3 min(std::numeric_limits<float>::max()), max(-min);
                for (size_t i = begin; i < end; ++i) {
                    min = glm::min(min, bounds[order[i]].min);
                    max = glm::max(max, bounds[order[i]].max);
                }
                bvh.push_back({min, max, begin, end});
                if (end - begin > 16) {
                    auto delta = glm::dvec3(max) - glm::dvec3(min);
                    int axis = delta.x >= delta.y && delta.x >= delta.z ? 0 : delta.y >= delta.z ? 1
                                                                                                 : 2;
                    auto middle = begin + (end - begin) / 2;
                    std::nth_element(order.begin() + begin, order.begin() + middle, order.begin() + end,
                                     [&](auto a, auto b) {
                                         return double(bounds[a].min[axis]) + bounds[a].max[axis] <
                                                double(bounds[b].min[axis]) + bounds[b].max[axis];
                                     });
                    auto left = buildBounds(begin, middle), right = buildBounds(middle, end);
                    bvh[index].left = left;
                    bvh[index].right = right;
                }
                return index;
            };
            buildBounds(0, order.size());
            // Each support proof is independent. Keep the final reduction ordered,
            // but avoid a million serial BVH queries and std::function allocations.
            std::vector<uint8_t> eligible(bounds.size());
            std::atomic<size_t> cursor{0};
            std::atomic<uint64_t> bound_work{0};
            std::atomic<bool> failed{false};
            std::exception_ptr failure;
            std::mutex failure_mutex;
            auto classify = [&] {
                try {
                    uint64_t local_work = 0;
                    for (;;) {
                        const auto begin = cursor.fetch_add(256, std::memory_order_relaxed);
                        if (begin >= bounds.size() || failed.load(std::memory_order_relaxed))
                            break;
                        for (size_t row = begin; row < std::min(begin + 256, bounds.size()); ++row) {
                            check(stop);
                            const auto& bound = bounds[row];
                            double upper = 0;
                            auto accumulate = [&](auto&& self, uint32_t i) -> void {
                                if (upper >= cutoff)
                                    return;
                                if ((++local_work & 4095) == 0) {
                                    check(stop);
                                    if (bound_work.fetch_add(4096, std::memory_order_relaxed) + 4096 > 500'000'000)
                                        throw std::runtime_error("Sparse opacity-bound work budget exceeded");
                                }
                                const auto& node = bvh[i];
                                if (glm::any(glm::greaterThan(bound.min, node.max)) || glm::any(glm::lessThan(bound.max, node.min)))
                                    return;
                                if (node.left) {
                                    self(self, node.left);
                                    self(self, node.right);
                                } else {
                                    for (size_t j = node.begin; j < node.end && upper < cutoff; ++j) {
                                        const auto& other = bounds[order[j]];
                                        if (glm::all(glm::lessThanEqual(bound.min, other.max)) && glm::all(glm::greaterThanEqual(bound.max, other.min)))
                                            upper += other.opacity + 8 * std::numeric_limits<float>::epsilon();
                                    }
                                }
                            };
                            if (bound.opacity < cutoff)
                                accumulate(accumulate, 0);
                            eligible[row] = bound.opacity >= cutoff || upper >= cutoff;
                        }
                    }
                    if (bound_work.fetch_add(local_work & 4095, std::memory_order_relaxed) + (local_work & 4095) > 500'000'000)
                        throw std::runtime_error("Sparse opacity-bound work budget exceeded");
                } catch (...) {
                    std::lock_guard lock(failure_mutex);
                    if (!failure)
                        failure = std::current_exception();
                    failed.store(true, std::memory_order_relaxed);
                }
            };
            {
                std::vector<std::jthread> workers;
                const auto count = bounds.size() >= 4096 ? std::min(8u, std::max(1u, std::thread::hardware_concurrency())) : 1u;
                for (unsigned i = 1; i < count; ++i)
                    workers.emplace_back(classify);
                classify();
            }
            if (failure)
                std::rethrow_exception(failure);
            for (size_t i = 0; i < bounds.size(); ++i)
                if (eligible[i]) {
                    lo = glm::min(lo, bounds[i].min);
                    hi = glm::max(hi, bounds[i].max);
                }
        }
        if (glm::any(glm::greaterThan(lo, hi)))
            throw std::runtime_error("No occupied voxels; lower opacity or voxel size");
        CollisionVolume volume;
        volume.settings = settings;
        const float size = settings.voxel_size;
        volume.origin = glm::floor(lo / size) * size - 2 * size;
        auto dims = glm::ceil((glm::dvec3(hi) - glm::dvec3(volume.origin)) / double(size)) + 3.0;
        if (glm::any(glm::greaterThan(dims, glm::dvec3(1 << 21))) ||
            glm::any(glm::lessThan(dims, glm::dvec3(1))) ||
            !std::isfinite(dims.x) || !std::isfinite(dims.y) || !std::isfinite(dims.z))
            throw std::runtime_error("Sparse collision coordinate range exceeded (maximum 2^21 cells per axis)");
        volume.dimensions = glm::ivec3(dims);
        std::vector<PreparedGaussian> prepared;
        prepared.reserve(source.size());
        size_t bound_index = 0;
        const auto world_max = glm::dvec3(volume.origin) + glm::dvec3(volume.dimensions) * double(size);
        for (const auto& g : source) {
            if (g.opacity == 0 || std::abs(glm::determinant(g.axes)) < 1e-20f)
                continue;
            const auto& bound = bounds[bound_index++];
            // Weak isolated supports can lie arbitrarily far outside the effective
            // grid. Clip before converting coordinates, retaining their contribution
            // wherever they overlap the grid without overflowing cell indices.
            const auto padding = (glm::abs(glm::dvec3(bound.min)) + glm::abs(glm::dvec3(bound.max)) +
                                  glm::abs(glm::dvec3(volume.origin)) + 1.0) *
                                 (8 * double(std::numeric_limits<float>::epsilon()));
            const auto clipped_min = glm::max(glm::dvec3(bound.min) - padding, glm::dvec3(volume.origin));
            const auto clipped_max = glm::min(glm::dvec3(bound.max) + padding, world_max);
            if (glm::any(glm::greaterThan(clipped_min, clipped_max)))
                continue;
            const auto cell_min = glm::clamp(glm::ivec3(glm::floor((clipped_min - glm::dvec3(volume.origin)) / double(size))),
                                             glm::ivec3(0), volume.dimensions);
            const auto cell_max = glm::clamp(glm::ivec3(glm::floor((clipped_max - glm::dvec3(volume.origin)) / double(size))) + 1,
                                             glm::ivec3(0), volume.dimensions);
            prepared.push_back({g.center, glm::inverse(g.axes), extent(g), g.opacity, cell_min, cell_max});
        }
        return {std::move(volume), std::move(prepared)};
    }

    CollisionVolume voxelizeSparse(std::span<const CollisionGaussian> source, CollisionSettings settings,
                                   std::stop_token stop, CollisionProgress progress, unsigned parallelism) {
        const double cutoff = -std::log1p(-double(settings.opacity));
        auto input = prepareCollision(source, settings, stop);
        auto volume = std::move(input.volume);
        const auto& prepared = input.gaussians;
        const float size = settings.voxel_size;
        Tree tree(volume.dimensions, stop);
        std::vector<uint32_t> all(prepared.size());
        std::iota(all.begin(), all.end(), 0);

        const auto available = parallelism ? parallelism : std::max(1u, std::thread::hardware_concurrency());
        const unsigned workers = source.size() >= 4096 && tree.size > 8 ? std::min(available, 8u) : 1;
        std::atomic<size_t> allocated{workers > 1 ? 9u : 1u};
        std::atomic<bool> failed{false};
        std::atomic<uint64_t> shared_work{0};
        auto buildBranch = [&](int octant) {
            Tree tree(volume.dimensions, stop);
            uint64_t work = 0;
            auto children = [&](uint32_t index, auto&& fn) {
                if (allocated.fetch_add(8, std::memory_order_relaxed) + 8 > Tree::max_nodes)
                    throw std::runtime_error("Sparse collision node memory budget exceeded");
                const auto base = uint32_t(tree.nodes.size());
                tree.nodes.resize(tree.nodes.size() + 8);
                tree.nodes[index] = base + 2;
                for (int child = 0; child < 8; ++child)
                    fn(base + child, child);
                const auto first = tree.nodes[base];
                if (first <= 1 && std::all_of(tree.nodes.begin() + base, tree.nodes.begin() + base + 8,
                                              [first](auto value) { return value == first; })) {
                    tree.nodes[index] = first;
                    allocated.fetch_sub(tree.nodes.size() - base, std::memory_order_relaxed);
                    tree.nodes.resize(base);
                }
            };
            std::function<void(uint32_t, glm::ivec3, int, const std::vector<uint32_t>&)> build;
            build = [&](uint32_t node, glm::ivec3 lo, int n, const std::vector<uint32_t>& candidates) {
                check(stop);
                if (failed.load(std::memory_order_relaxed))
                    throw std::runtime_error("Collision generation cancelled after worker failure");
                auto hi = glm::min(lo + n, volume.dimensions);
                if (glm::any(glm::greaterThanEqual(lo, hi)))
                    return;
                const auto middle = glm::dvec3(volume.origin) +
                                    (glm::dvec3(lo) + glm::dvec3(hi)) * (double(size) / 2);
                const auto half = glm::dvec3(hi - lo) * (double(size) / 2);
                // Bound float evaluation error as well as the whole voxel-volume box.
                const auto error = (glm::abs(middle) + glm::abs(glm::dvec3(volume.origin)) + 1.0) *
                                   (8 * double(std::numeric_limits<float>::epsilon()));
                std::vector<uint32_t> relevant;
                relevant.reserve(candidates.size());
                double upper = 0, lower = 0;
                for (auto index : candidates) {
                    if ((++work & 4095) == 0) {
                        check(stop);
                        if (shared_work.fetch_add(4096, std::memory_order_relaxed) > 4'000'000'000ULL)
                            throw std::runtime_error("Sparse Gaussian active-work budget exceeded");
                    }
                    if (work > 4'000'000'000ULL)
                        throw std::runtime_error("Sparse Gaussian active-work budget exceeded");
                    const auto& g = prepared[index];
                    if (glm::any(glm::greaterThanEqual(lo, g.max)) || glm::any(glm::lessThanEqual(hi, g.min)))
                        continue;
                    auto inv = glm::dmat3(g.inverse);
                    auto q = inv * (middle - glm::dvec3(g.center));
                    auto h = half + error + glm::abs(glm::dvec3(g.center)) * (8 * double(std::numeric_limits<float>::epsilon()));
                    auto qr = glm::abs(inv[0]) * h.x + glm::abs(inv[1]) * h.y + glm::abs(inv[2]) * h.z;
                    auto near = glm::max(glm::abs(q) - qr, glm::dvec3(0));
                    double min_distance = glm::dot(near, near);
                    relevant.push_back(index);
                    upper += g.opacity * std::exp(-0.5 * min_distance);
                    auto far = glm::abs(q) + qr;
                    double max_distance = glm::dot(far, far);
                    if (max_distance <= 9)
                        lower += g.opacity * std::exp(-0.5 * max_distance);
                }
                const double rounding = relevant.size() * (8 * double(std::numeric_limits<float>::epsilon()));
                if (relevant.empty() || upper + rounding < cutoff)
                    return;
                if (lower > 0 && lower - rounding >= cutoff &&
                    glm::all(glm::equal(hi, lo + n))) {
                    tree.nodes[node] = 1;
                    return;
                }
                if (n > 8) {
                    children(node, [&](uint32_t index, int child) {
                        build(index, lo + glm::ivec3(child & 1, (child >> 1) & 1, child >> 2) * (n / 2),
                              n / 2, relevant);
                    });
                    return;
                }
                std::array<float, 512> density{};
                const auto total = uint32_t((hi.x - lo.x) * (hi.y - lo.y) * (hi.z - lo.z));
                auto remaining = total;
                for (auto index : relevant) {
                    const auto& g = prepared[index];
                    if (remaining == 0)
                        break;
                    const auto from = glm::max(lo, g.min), to = glm::min(hi, g.max);
                    for (int z = from.z; z < to.z; ++z)
                        for (int y = from.y; y < to.y; ++y)
                            for (int x = from.x; x < to.x; ++x) {
                                if ((++work & 4095) == 0) {
                                    check(stop);
                                    if (shared_work.fetch_add(4096, std::memory_order_relaxed) > 4'000'000'000ULL)
                                        throw std::runtime_error("Sparse Gaussian active-work budget exceeded");
                                }
                                auto& value = density[(z - lo.z) * 64 + (y - lo.y) * 8 + x - lo.x];
                                if (value > 0 && double(value) >= cutoff)
                                    continue;
                                const auto center = volume.origin + (glm::vec3(x, y, z) + 0.5f) * size;
                                if (glm::any(glm::greaterThan(glm::abs(center - g.center), g.extent + size * 0.5f)))
                                    continue;
                                const auto closest = glm::clamp(g.center, center - size * 0.5f, center + size * 0.5f);
                                const auto q = g.inverse * (closest - g.center);
                                value += g.opacity * std::exp(-0.5f * glm::dot(q, q));
                                if (value > 0 && double(value) >= cutoff)
                                    --remaining;
                            }
                }
                if (remaining == total)
                    return;
                if (remaining == 0 && glm::all(glm::equal(hi, lo + n))) {
                    tree.nodes[node] = 1;
                    return;
                }
                std::function<void(uint32_t, glm::ivec3, int)> emit = [&](uint32_t index, glm::ivec3 p, int edge) {
                    if (edge == 1) {
                        if (glm::any(glm::greaterThanEqual(p, hi)))
                            return;
                        const auto q = p - lo;
                        const auto value = density[q.z * 64 + q.y * 8 + q.x];
                        tree.nodes[index] = value > 0 && double(value) >= cutoff;
                        return;
                    }
                    children(index, [&](uint32_t child_index, int child) {
                        emit(child_index, p + glm::ivec3(child & 1, (child >> 1) & 1, child >> 2) * (edge / 2),
                             edge / 2);
                    });
                };
                emit(node, lo, n);
            };
            if (octant < 0)
                build(0, {}, tree.size, all);
            else
                build(0, glm::ivec3(octant & 1, (octant >> 1) & 1, octant >> 2) * (tree.size / 2), tree.size / 2, all);
            return std::move(tree.nodes);
        };
        if (workers == 1) {
            tree.nodes = buildBranch(-1);
        } else {
            std::array<std::vector<uint32_t>, 8> branches;
            std::atomic<unsigned> next{0}, completed{0};
            std::mutex error_mutex, progress_mutex;
            std::exception_ptr error;
            {
                std::vector<std::jthread> pool;
                for (unsigned worker = 0; worker < workers; ++worker)
                    pool.emplace_back([&] {
                        try {
                            for (unsigned octant; (octant = next.fetch_add(1)) < 8 && !failed.load();) {
                                branches[octant] = buildBranch(int(octant));
                                const auto done = completed.fetch_add(1) + 1;
                                if (progress) {
                                    std::lock_guard lock(progress_mutex);
                                    progress(std::format("Voxelizing regions: {}/8", done));
                                }
                            }
                        } catch (...) {
                            std::lock_guard lock(error_mutex);
                            if (!error)
                                error = std::current_exception();
                            failed = true;
                        }
                    });
            }
            if (error)
                std::rethrow_exception(error);
            tree.nodes.assign(9, 0);
            tree.nodes[0] = 3;
            tree.nodes.reserve(allocated.load());
            for (unsigned octant = 0; octant < 8; ++octant) {
                check(stop);
                auto& branch = branches[octant];
                const uint32_t shift = uint32_t(tree.nodes.size()) - 1;
                tree.nodes[octant + 1] = branch[0] <= 1 ? branch[0] : branch[0] + shift;
                for (size_t i = 1; i < branch.size(); ++i)
                    tree.nodes.push_back(branch[i] <= 1 ? branch[i] : branch[i] + shift);
                branch.clear();
                branch.shrink_to_fit();
            }
        }
        volume.tree_size = tree.size;
        volume.tree = std::move(tree.nodes);
        return volume;
    }

    CollisionVolume processSparse(CollisionVolume volume, std::stop_token stop, CollisionProgress progress) {
        volume.validate();
        auto any_solid = [&](auto&& self, uint32_t i, glm::ivec3 lo, int edge) -> bool {
            if (volume.tree[i] == 0 || glm::any(glm::greaterThanEqual(lo, volume.dimensions)))
                return false;
            if (volume.tree[i] == 1)
                return true;
            for (int child = 0; child < 8; ++child)
                if (self(self, volume.tree[i] - 2 + child,
                         lo + glm::ivec3(child & 1, (child >> 1) & 1, child >> 2) * (edge / 2), edge / 2))
                    return true;
            return false;
        };
        if (!any_solid(any_solid, 0, {}, volume.tree_size))
            throw std::runtime_error("No occupied voxels; lower opacity or voxel size");
        auto report = [&](const char* text) { check(stop); if (progress) progress(text); };
        const auto& s = volume.settings;
        if (!s.external_fill && !s.floor_fill && s.carve_height == 0)
            return volume;
        Tree tree(volume.dimensions, stop);
        if (!volume.tree.empty())
            tree.nodes = std::move(volume.tree);
        else
            tree = volumeTree(volume, stop);
        if (s.external_fill) {
            report("Sealing sparse exterior");
            const int radius = int(std::min(std::ceil(double(s.external_size) / (2 * s.voxel_size)), double(tree.size)));
            Tree sealed(tree.dimensions, stop);
            tree.leaves([&](const Leaf& l) {
                if (l.solid)
                    sealed.setBox(l.min - radius, l.max() + radius);
            });
            auto exterior = sealed.flood({}, false, true);
            auto seed = volume.cell(s.seed);
            if (!volume.contains(seed) || sealed.occupied(seed) || exterior.occupied(seed))
                volume.warning = "External fill skipped: seed is exterior or consumed by sealing dilation";
            else
                tree.unite(exterior);
        }
        if (s.floor_fill) {
            report("Filling sparse ground columns");
            floorFill(tree, s.floor_size, s.voxel_size, stop, s.floor_below_seed ? volume.cell(s.seed - glm::vec3(0, 1.6f * s.units_per_meter, 0)).y : tree.dimensions.y);
        }
        if (s.carve_height > 0) {
            report("Flooding sparse navigable capsule space");
            auto seed = volume.cell(s.seed);
            if (!volume.contains(seed))
                throw std::runtime_error("Carve capsule does not fit at seed (seed denotes capsule center)");
            auto blocked = capsuleDilate(tree, s.carve_height, s.carve_radius, s.voxel_size, stop);
            const int horizontal = int(std::ceil(s.carve_radius / s.voxel_size + 0.5f));
            const int vertical = int(std::ceil(s.carve_height / (2 * s.voxel_size) + 0.5f));
            glm::ivec3 margin(horizontal, vertical, horizontal);
            for (int axis = 0; axis < 3; ++axis) {
                auto end = tree.dimensions, begin = glm::ivec3(0);
                end[axis] = std::min(margin[axis], tree.dimensions[axis]);
                blocked.setBox({}, end);
                begin[axis] = std::max(0, tree.dimensions[axis] - margin[axis]);
                blocked.setBox(begin, tree.dimensions);
            }
            if (blocked.occupied(seed))
                throw std::runtime_error("Carve capsule does not fit at the seed voxel; move seed or reduce capsule");
            auto reached = blocked.flood(std::span(&seed, 1), false);
            auto navigable = capsuleDilate(reached, s.carve_height, s.carve_radius, s.voxel_size, stop);
            navigable.complement();
            tree.unite(navigable);
        }
        storeTree(volume, std::move(tree));
        volume.validate();
        report("Finalizing collision occupancy");
        return volume;
    }
} // namespace lfs::core::collision_detail
