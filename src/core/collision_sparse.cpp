/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "collision_sparse.hpp"
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
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
        void floorFill(Tree& tree, float reach_world, float voxel_size, std::stop_token stop) {
            // Only columns intersecting source occupancy are materialized.
            std::unordered_map<uint64_t, int> heights;
            auto key = [](int x, int z) { return uint64_t(uint32_t(x)) | (uint64_t(uint32_t(z)) << 32); };
            tree.leaves([&](const Leaf& l) {
                if (!l.solid)
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

    CollisionVolume voxelizeSparse(std::span<const CollisionGaussian> source, CollisionSettings settings,
                                   std::stop_token stop) {
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
            auto e = extent(g);
            if (!finite(e) || !finite(g.center - e) || !finite(g.center + e))
                throw std::invalid_argument("Gaussian collision support exceeds finite coordinate range");
            bounds.push_back({g.center - e, g.center + e, g.opacity});
        }
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
        uint64_t bound_work = 0;
        for (const auto& bound : bounds) {
            check(stop);
            double upper = 0;
            std::function<void(uint32_t)> accumulate = [&](uint32_t i) {
                if (upper >= settings.opacity)
                    return;
                if ((++bound_work & 4095) == 0)
                    check(stop);
                if (bound_work > 500'000'000)
                    throw std::runtime_error("Sparse opacity-bound work budget exceeded");
                const auto& node = bvh[i];
                if (glm::any(glm::greaterThan(bound.min, node.max)) || glm::any(glm::lessThan(bound.max, node.min)))
                    return;
                if (node.left) {
                    accumulate(node.left);
                    accumulate(node.right);
                } else {
                    for (size_t j = node.begin; j < node.end && upper < settings.opacity; ++j) {
                        const auto& other = bounds[order[j]];
                        if (glm::all(glm::lessThanEqual(bound.min, other.max)) &&
                            glm::all(glm::greaterThanEqual(bound.max, other.min)))
                            upper += (1 - upper) * other.opacity + 8 * std::numeric_limits<float>::epsilon();
                    }
                }
            };
            // A weak isolated support cannot change occupancy, but remains in
            // the evaluator wherever another support can produce a solid cell.
            if (bound.opacity < settings.opacity)
                accumulate(0);
            if (bound.opacity >= settings.opacity || upper >= settings.opacity) {
                lo = glm::min(lo, bound.min);
                hi = glm::max(hi, bound.max);
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
        Tree tree(volume.dimensions, stop);
        struct Prepared {
            glm::vec3 center;
            glm::mat3 inverse;
            float opacity;
        };
        std::vector<Prepared> prepared;
        prepared.reserve(source.size());
        for (const auto& g : source) {
            if (g.opacity == 0 || std::abs(glm::determinant(g.axes)) < 1e-20f)
                continue;
            prepared.push_back({g.center, glm::inverse(g.axes), g.opacity});
        }
        uint64_t work = 0;
        std::vector<uint32_t> all(prepared.size());
        std::iota(all.begin(), all.end(), 0);
        auto children = [&](uint32_t index, auto&& fn) {
            if (tree.nodes.size() + 8 > Tree::max_nodes)
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
                tree.nodes.resize(base);
            }
        };
        std::function<void(uint32_t, glm::ivec3, int, const std::vector<uint32_t>&)> build;
        build = [&](uint32_t node, glm::ivec3 lo, int n, const std::vector<uint32_t>& candidates) {
            check(stop);
            auto hi = glm::min(lo + n, volume.dimensions);
            if (glm::any(glm::greaterThanEqual(lo, hi)))
                return;
            const auto middle = glm::dvec3(volume.origin) +
                                (glm::dvec3(lo) + glm::dvec3(hi)) * (double(size) / 2);
            const auto half = glm::dvec3(hi - lo - 1) * (double(size) / 2);
            // Bound float evaluation error as well as the whole cell-center box.
            const auto error = (glm::abs(middle) + glm::abs(glm::dvec3(volume.origin)) + 1.0) *
                               (8 * double(std::numeric_limits<float>::epsilon()));
            std::vector<uint32_t> relevant;
            relevant.reserve(candidates.size());
            double upper = 0, lower = 0;
            for (auto index : candidates) {
                if ((++work & 4095) == 0)
                    check(stop);
                if (work > 4'000'000'000ULL)
                    throw std::runtime_error("Sparse Gaussian active-work budget exceeded");
                const auto& g = prepared[index];
                auto inv = glm::dmat3(g.inverse);
                auto q = inv * (middle - glm::dvec3(g.center));
                auto h = half + error + glm::abs(glm::dvec3(g.center)) * (8 * double(std::numeric_limits<float>::epsilon()));
                auto qr = glm::abs(inv[0]) * h.x + glm::abs(inv[1]) * h.y + glm::abs(inv[2]) * h.z;
                auto near = glm::max(glm::abs(q) - qr, glm::dvec3(0));
                double min_distance = glm::dot(near, near);
                if (min_distance > 9.0001)
                    continue;
                relevant.push_back(index);
                upper += (1 - upper) * g.opacity * std::exp(-0.5 * min_distance);
                auto far = glm::abs(q) + qr;
                double max_distance = glm::dot(far, far);
                if (max_distance <= 9)
                    lower += (1 - lower) * g.opacity * std::exp(-0.5 * max_distance);
            }
            const double rounding = relevant.size() * (8 * double(std::numeric_limits<float>::epsilon()));
            if (relevant.empty() || upper + rounding < settings.opacity)
                return;
            if (lower > 0 && lower - rounding >= settings.opacity &&
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
            std::array<float, 512> alpha{};
            for (auto index : relevant) {
                const auto& g = prepared[index];
                for (int z = lo.z; z < hi.z; ++z)
                    for (int y = lo.y; y < hi.y; ++y)
                        for (int x = lo.x; x < hi.x; ++x) {
                            if (++work > 4'000'000'000ULL)
                                throw std::runtime_error("Sparse Gaussian active-work budget exceeded");
                            auto& value = alpha[(z - lo.z) * 64 + (y - lo.y) * 8 + x - lo.x];
                            if (value > 0 && value >= settings.opacity)
                                continue;
                            auto q = g.inverse * (volume.origin + (glm::vec3(x, y, z) + 0.5f) * size - g.center);
                            float distance = glm::dot(q, q);
                            if (distance <= 9)
                                value += (1 - value) * g.opacity * std::exp(-0.5f * distance);
                        }
            }
            std::function<void(uint32_t, glm::ivec3, int)> emit = [&](uint32_t index, glm::ivec3 p, int edge) {
                if (edge == 1) {
                    if (glm::any(glm::greaterThanEqual(p, hi)))
                        return;
                    const auto q = p - lo;
                    const auto value = alpha[q.z * 64 + q.y * 8 + q.x];
                    tree.nodes[index] = value > 0 && value >= settings.opacity;
                    return;
                }
                children(index, [&](uint32_t child_index, int child) {
                    emit(child_index, p + glm::ivec3(child & 1, (child >> 1) & 1, child >> 2) * (edge / 2),
                         edge / 2);
                });
            };
            emit(node, lo, n);
        };
        build(0, {}, tree.size, all);
        storeTree(volume, std::move(tree));
        return volume;
    }

    CollisionVolume processSparse(CollisionVolume volume, std::stop_token stop, CollisionProgress progress) {
        volume.validate();
        if (volume.occupiedCount() == 0)
            throw std::runtime_error("No occupied voxels; lower opacity or voxel size");
        auto report = [&](const char* text) { check(stop); if (progress) progress(text); };
        auto tree = volumeTree(volume, stop);
        const auto& s = volume.settings;
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
            floorFill(tree, s.floor_size, s.voxel_size, stop);
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
        report("Building sparse collision surface");
        return volume;
    }
} // namespace lfs::core::collision_detail
