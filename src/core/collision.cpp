/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include "collision_sparse.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace lfs::core {
    namespace {
        constexpr glm::ivec3 directions[] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        constexpr uint64_t max_work = 500'000'000;
        struct GridPointHash {
            size_t operator()(const std::array<int32_t, 3>& p) const noexcept {
                size_t h = std::hash<int32_t>{}(p[0]);
                for (int a = 1; a < 3; ++a)
                    h ^= std::hash<int32_t>{}(p[a]) + size_t(0x9e3779b9) + (h << 6) + (h >> 2);
                return h;
            }
        };
        void checkpoint(std::stop_token stop) {
            if (stop.stop_requested())
                throw std::runtime_error("Collision generation cancelled");
        }
        bool finite(glm::vec3 p) {
            return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        }
        void positive(float v, const char* name, bool zero = false) {
            if (!std::isfinite(v) || (zero ? v < 0 : v <= 0))
                throw std::invalid_argument(std::string(name) + " must be finite and positive");
        }
        void probability(float v, const char* name) {
            positive(v, name, true);
            if (v > 1)
                throw std::invalid_argument(std::string(name) + " must be in [0, 1]");
        }
        size_t cellCount(glm::ivec3 d) {
            if (glm::any(glm::lessThanEqual(d, glm::ivec3(0))))
                throw std::invalid_argument("Invalid collision grid dimensions");
            uint64_t n = 1;
            for (int a = 0; a < 3; ++a) {
                if (static_cast<uint64_t>(d[a]) > CollisionVolume::max_cells / n)
                    throw std::runtime_error("Collision grid exceeds 16 million cells; increase voxel size or isolate the scene");
                n *= d[a];
            }
            return n;
        }
        template <typename Fn>
        void eachCell(const CollisionVolume& g, std::stop_token stop, Fn fn) {
            for (int z = 0; z < g.dimensions.z; ++z) {
                checkpoint(stop);
                for (int y = 0; y < g.dimensions.y; ++y)
                    for (int x = 0; x < g.dimensions.x; ++x)
                        fn(glm::ivec3(x, y, z));
            }
        }
        std::vector<uint8_t> flood(const CollisionVolume& g, const std::vector<uint8_t>& blocked,
                                   std::span<const glm::ivec3> seeds, std::stop_token stop) {
            std::vector<uint8_t> visited(g.solid.size(), 0);
            std::vector<uint32_t> queue;
            for (auto p : seeds) {
                if (!g.contains(p))
                    continue;
                auto i = g.index(p);
                if (!blocked[i] && !visited[i]) {
                    visited[i] = 1;
                    queue.push_back(static_cast<uint32_t>(i));
                }
            }
            for (size_t head = 0; head < queue.size(); ++head) {
                if ((head & 4095) == 0)
                    checkpoint(stop);
                const auto i = queue[head];
                glm::ivec3 p;
                p.x = i % g.dimensions.x;
                p.y = (i / g.dimensions.x) % g.dimensions.y;
                p.z = i / (g.dimensions.x * g.dimensions.y);
                for (auto dir : directions) {
                    auto q = p + dir;
                    if (!g.contains(q))
                        continue;
                    auto j = g.index(q);
                    if (!visited[j] && !blocked[j]) {
                        visited[j] = 1;
                        queue.push_back(static_cast<uint32_t>(j));
                    }
                }
            }
            return visited;
        }
        std::vector<uint8_t> dilate(const CollisionVolume& g, int radius, std::stop_token stop) {
            auto data = g.solid;
            std::vector<uint8_t> next(data.size());
            // Separable sliding windows: independent of the world-space sealing radius.
            for (int axis = 0; axis < 3; ++axis) {
                int u = (axis + 1) % 3, v = (axis + 2) % 3;
                for (int a = 0; a < g.dimensions[u]; ++a) {
                    checkpoint(stop);
                    for (int b = 0; b < g.dimensions[v]; ++b) {
                        glm::ivec3 p(0);
                        p[u] = a;
                        p[v] = b;
                        int sum = 0;
                        for (int c = 0; c <= std::min(radius, g.dimensions[axis] - 1); ++c) {
                            p[axis] = c;
                            sum += data[g.index(p)] != 0;
                        }
                        for (int c = 0; c < g.dimensions[axis]; ++c) {
                            p[axis] = c;
                            next[g.index(p)] = sum > 0;
                            if (c - radius >= 0) {
                                p[axis] = c - radius;
                                sum -= data[g.index(p)] != 0;
                            }
                            if (c + radius + 1 < g.dimensions[axis]) {
                                p[axis] = c + radius + 1;
                                sum += data[g.index(p)] != 0;
                            }
                        }
                    }
                }
                data.swap(next);
            }
            return data;
        }
        std::vector<uint8_t> capsuleDilate(const CollisionVolume& g,
                                           const std::vector<uint8_t>& source,
                                           float height, float radius, std::stop_token stop) {
            const double segment_cells = std::ceil((double(height) / 2 - radius) / g.settings.voxel_size);
            if (segment_cells > g.dimensions.y * 2)
                throw std::runtime_error("Carve capsule is taller than collision bounds");
            const int segment = int(segment_cells);
            std::vector<float> distance(source.size(), 1e20f);
            for (int z = 0; z < g.dimensions.z; ++z) {
                checkpoint(stop);
                for (int x = 0; x < g.dimensions.x; ++x) {
                    int count = 0;
                    for (int y = 0; y <= std::min(segment, g.dimensions.y - 1); ++y)
                        count += source[g.index({x, y, z})] != 0;
                    for (int y = 0; y < g.dimensions.y; ++y) {
                        distance[g.index({x, y, z})] = count ? 0 : 1e20f;
                        if (y - segment >= 0)
                            count -= source[g.index({x, y - segment, z})] != 0;
                        if (y + segment + 1 < g.dimensions.y)
                            count += source[g.index({x, y + segment + 1, z})] != 0;
                    }
                }
            }
            // Squared Euclidean distance transform: lower envelopes of 1D
            // parabolas, composed along three axes. O(cells), not O(cells*capsule).
            const int longest = std::max({g.dimensions.x, g.dimensions.y, g.dimensions.z});
            std::vector<float> input(longest), output(longest);
            std::vector<int> sites(longest);
            std::vector<double> cuts(size_t(longest) + 1);
            for (int axis = 0; axis < 3; ++axis) {
                int u = (axis + 1) % 3, v = (axis + 2) % 3, n = g.dimensions[axis];
                for (int a = 0; a < g.dimensions[u]; ++a) {
                    checkpoint(stop);
                    for (int b = 0; b < g.dimensions[v]; ++b) {
                        glm::ivec3 p(0);
                        p[u] = a;
                        p[v] = b;
                        for (int q = 0; q < n; ++q) {
                            p[axis] = q;
                            input[q] = distance[g.index(p)];
                        }
                        int k = -1;
                        for (int q = 0; q < n; ++q) {
                            if (input[q] >= 1e19f)
                                continue;
                            double intersection = -std::numeric_limits<double>::infinity();
                            while (k >= 0) {
                                int previous = sites[k];
                                intersection = (double(input[q]) + double(q) * q -
                                                double(input[previous]) - double(previous) * previous) /
                                               (2.0 * (q - previous));
                                if (intersection > cuts[k])
                                    break;
                                --k;
                            }
                            ++k;
                            sites[k] = q;
                            cuts[k] = k == 0 ? -std::numeric_limits<double>::infinity() : intersection;
                            cuts[k + 1] = std::numeric_limits<double>::infinity();
                        }
                        if (k < 0)
                            continue;
                        int envelope = 0;
                        for (int q = 0; q < n; ++q) {
                            while (cuts[envelope + 1] < q)
                                ++envelope;
                            double delta = q - sites[envelope];
                            output[q] = float(delta * delta + input[sites[envelope]]);
                        }
                        for (int q = 0; q < n; ++q) {
                            p[axis] = q;
                            distance[g.index(p)] = output[q];
                        }
                    }
                }
            }
            float threshold = radius / g.settings.voxel_size + std::sqrt(3.0f) / 2;
            threshold *= threshold;
            std::vector<uint8_t> result(distance.size());
            for (size_t i = 0; i < result.size(); ++i)
                result[i] = distance[i] < threshold;
            return result;
        }
        void simplifyCoplanar(CollisionMesh& mesh, std::stop_token stop,
                              std::vector<std::array<int32_t, 3>>* keys = nullptr) {
            const size_t vertices = mesh.vertices.size() / 3, faces = mesh.indices.size() / 3;
            std::vector<std::vector<int32_t>> incident(vertices);
            std::vector<uint8_t> alive(faces, 1);
            for (size_t f = 0; f < faces; ++f)
                for (int a = 0; a < 3; ++a)
                    incident[mesh.indices[f * 3 + a]].push_back(int32_t(f));
            auto point = [&](int v) { return glm::vec3(mesh.vertices[v * 3], mesh.vertices[v * 3 + 1], mesh.vertices[v * 3 + 2]); };
            auto neighbors = [&](int v) {
                std::vector<int> result;
                for (int f : incident[v]) {
                    if (!alive[f])
                        continue;
                    bool has_vertex = false;
                    for (int a = 0; a < 3; ++a)
                        has_vertex |= mesh.indices[f * 3 + a] == v;
                    if (!has_vertex)
                        continue;
                    for (int a = 0; a < 3; ++a)
                        if (int n = mesh.indices[f * 3 + a]; n != v)
                            result.push_back(n);
                }
                std::sort(result.begin(), result.end());
                result.erase(std::unique(result.begin(), result.end()), result.end());
                return result;
            };
            // Collapse only strictly coplanar interior stars. The link condition
            // preserves manifold topology; orientation checks reject foldovers.
            for (size_t v = 0; v < vertices; ++v) {
                if ((v & 1023) == 0)
                    checkpoint(stop);
                auto ns = neighbors(int(v));
                if (ns.size() < 3)
                    continue;
                std::vector<int> star;
                glm::vec3 normal(0);
                bool planar = true;
                for (int f : incident[v]) {
                    if (!alive[f])
                        continue;
                    auto a = mesh.indices[f * 3], b = mesh.indices[f * 3 + 1], c = mesh.indices[f * 3 + 2];
                    if (a != int(v) && b != int(v) && c != int(v))
                        continue;
                    auto n = glm::cross(point(b) - point(a), point(c) - point(a));
                    float length = glm::length(n);
                    if (length < 1e-15f) {
                        planar = false;
                        break;
                    }
                    n /= length;
                    if (star.empty())
                        normal = n;
                    else if (glm::dot(normal, n) < 0.999999f) {
                        planar = false;
                        break;
                    }
                    star.push_back(f);
                }
                if (!planar || star.size() != ns.size())
                    continue;
                for (int u : ns) {
                    auto other = neighbors(u);
                    std::vector<int> shared;
                    std::set_intersection(ns.begin(), ns.end(), other.begin(), other.end(), std::back_inserter(shared));
                    if (shared.size() != 2)
                        continue;
                    bool valid = true;
                    int removed = 0;
                    for (int f : star) {
                        auto a = mesh.indices[f * 3], b = mesh.indices[f * 3 + 1], c = mesh.indices[f * 3 + 2];
                        if (a == u || b == u || c == u) {
                            ++removed;
                            continue;
                        }
                        if (a == int(v))
                            a = u;
                        if (b == int(v))
                            b = u;
                        if (c == int(v))
                            c = u;
                        if (glm::dot(glm::cross(point(b) - point(a), point(c) - point(a)), normal) <= 1e-15f) {
                            valid = false;
                            break;
                        }
                    }
                    if (!valid || removed != 2)
                        continue;
                    for (int f : star) {
                        bool has_target = false;
                        for (int a = 0; a < 3; ++a)
                            has_target |= mesh.indices[f * 3 + a] == u;
                        if (has_target) {
                            alive[f] = 0;
                            continue;
                        }
                        for (int a = 0; a < 3; ++a)
                            if (mesh.indices[f * 3 + a] == int(v))
                                mesh.indices[f * 3 + a] = u;
                        incident[u].push_back(f);
                    }
                    break;
                }
            }
            CollisionMesh compact;
            std::vector<std::array<int32_t, 3>> compact_keys;
            std::vector<int32_t> remap(vertices, -1);
            for (size_t f = 0; f < faces; ++f) {
                if (!alive[f])
                    continue;
                for (int a = 0; a < 3; ++a) {
                    auto old = mesh.indices[f * 3 + a];
                    if (remap[old] < 0) {
                        remap[old] = int32_t(compact.vertices.size() / 3);
                        auto p = point(old);
                        compact.vertices.insert(compact.vertices.end(), {p.x, p.y, p.z});
                        if (keys)
                            compact_keys.push_back((*keys)[old]);
                    }
                    compact.indices.push_back(remap[old]);
                }
            }
            mesh = std::move(compact);
            if (keys)
                *keys = std::move(compact_keys);
        }
    } // namespace

    void CollisionSettings::applyPreset(CollisionPreset value) {
        preset = value;
        external_fill = value == CollisionPreset::Indoor;
        // Surface collision is the safe default. Ground-volume extrusion is an
        // explicit authoring option, since vegetation is not a ground estimator.
        floor_fill = false;
        carve_height = value == CollisionPreset::Indoor ? 1.6f * units_per_meter : 0;
        carve_radius = 0.2f * units_per_meter;
        external_size = floor_size = 1.6f * units_per_meter;
        cluster_resolution = units_per_meter;
    }
    void CollisionSettings::validate() const {
        positive(voxel_size, "Voxel size");
        positive(units_per_meter, "Scene units per metre");
        positive(cluster_resolution, "Cluster resolution");
        probability(opacity, "Voxel opacity");
        probability(cluster_opacity, "Cluster opacity");
        positive(cluster_min_contribution, "Minimum cluster contribution", true);
        positive(external_size, "External fill size", true);
        positive(floor_size, "Floor fill size", true);
        positive(carve_height, "Carve height", true);
        positive(carve_radius, "Carve radius");
        if (carve_height > 0 && carve_height < 2 * carve_radius)
            throw std::invalid_argument("Carve height must be at least twice its radius");
        if (!finite(seed))
            throw std::invalid_argument("Seed must be finite");
        if (preset < CollisionPreset::Indoor || preset > CollisionPreset::Object ||
            shape < CollisionShape::Smooth || shape > CollisionShape::Faces)
            throw std::invalid_argument("Invalid collision preset or mesh shape");
    }
    size_t CollisionVolume::index(glm::ivec3 p) const {
        return (size_t(p.z) * dimensions.y + p.y) * dimensions.x + p.x;
    }
    bool CollisionVolume::contains(glm::ivec3 p) const {
        return glm::all(glm::greaterThanEqual(p, glm::ivec3(0))) && glm::all(glm::lessThan(p, dimensions));
    }
    bool CollisionVolume::occupied(glm::ivec3 p) const {
        if (!contains(p))
            return false;
        if (tree.empty())
            return solid[index(p)] != 0;
        uint32_t i = 0;
        int n = tree_size;
        while (tree[i] > 1) {
            n /= 2;
            auto bit = glm::ivec3(glm::greaterThanEqual(p, glm::ivec3(n)));
            i = tree[i] - 2 + bit.x + 2 * bit.y + 4 * bit.z;
            p -= bit * n;
        }
        return tree[i] != 0;
    }
    glm::ivec3 CollisionVolume::cell(glm::vec3 p) const {
        const auto q = glm::floor((p - origin) / settings.voxel_size);
        if (!finite(q) || glm::any(glm::greaterThan(glm::abs(q), glm::vec3(1e9f))))
            throw std::invalid_argument("Collision query outside supported coordinate range");
        return glm::ivec3(q);
    }
    void CollisionVolume::validate() const {
        settings.validate();
        if (!finite(origin))
            throw std::invalid_argument("Invalid collision volume payload");
        if (!tree.empty()) {
            collision_detail::Tree expected(dimensions);
            if (!solid.empty() || tree_size != expected.size || tree.size() > collision_detail::Tree::max_nodes)
                throw std::invalid_argument("Invalid sparse collision volume payload");
            std::vector<uint8_t> seen(tree.size());
            size_t visited = 0;
            std::function<void(uint32_t, int)> visit = [&](uint32_t i, int n) {
                if (i >= tree.size() || seen[i]++)
                    throw std::invalid_argument("Invalid sparse collision child reference");
                ++visited;
                if (tree[i] <= 1)
                    return;
                auto base = tree[i] - 2;
                if (n <= 1 || base <= i || uint64_t(base) + 8 > tree.size())
                    throw std::invalid_argument("Invalid sparse collision branch");
                for (int a = 0; a < 8; ++a)
                    visit(base + a, n / 2);
            };
            visit(0, tree_size);
            if (visited != tree.size())
                throw std::invalid_argument("Unreachable sparse collision nodes");
            return;
        }
        if (solid.size() != cellCount(dimensions))
            throw std::invalid_argument("Invalid collision volume payload");
        if (std::any_of(solid.begin(), solid.end(), [](auto v) { return v > 1; }))
            throw std::invalid_argument("Invalid collision occupancy value");
    }
    bool CollisionVolume::capsuleIntersects(glm::vec3 feet, glm::vec3 up, float height, float radius) const {
        const glm::vec3 a = feet + up * radius, b = feet + up * (height - radius);
        const auto delta = b - a;
        // Segment/AABB distance is piecewise quadratic. Evaluate its exact
        // minimum on intervals split at the six box-plane crossings, instead
        // of twenty ternary-search iterations for every occupied voxel.
        auto intersectsBox = [&](glm::vec3 lo, glm::vec3 hi) {
            std::array<float, 8> times{0, 1};
            int count = 2;
            for (int axis = 0; axis < 3; ++axis) {
                if (std::abs(delta[axis]) < 1e-12f)
                    continue;
                for (float plane : {lo[axis], hi[axis]}) {
                    const float t = (plane - a[axis]) / delta[axis];
                    if (t > 0 && t < 1)
                        times[count++] = t;
                }
            }
            std::sort(times.begin(), times.begin() + count);
            for (int i = 1; i < count; ++i) {
                const float left = times[i - 1], right = times[i], middle = (left + right) * 0.5f;
                const auto point = a + delta * middle;
                float numerator = 0, denominator = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    if (point[axis] >= lo[axis] && point[axis] <= hi[axis])
                        continue;
                    const float bound = point[axis] < lo[axis] ? lo[axis] : hi[axis];
                    numerator += delta[axis] * (a[axis] - bound);
                    denominator += delta[axis] * delta[axis];
                }
                const float t = denominator > 0 ? glm::clamp(-numerator / denominator, left, right) : middle;
                const auto p = a + delta * t;
                const auto distance = p - glm::clamp(p, lo, hi);
                if (glm::dot(distance, distance) < radius * radius)
                    return true;
            }
            return false;
        };
        if (!tree.empty()) {
            size_t visited = 0;
            auto visit = [&](auto&& self, uint32_t index, glm::ivec3 lo, int edge) -> bool {
                if (tree[index] == 0 || glm::any(glm::greaterThanEqual(lo, dimensions)))
                    return false;
                if (++visited > 2'000'000)
                    throw std::runtime_error("Capsule query exceeds resource limit");
                if (!intersectsBox(origin + glm::vec3(lo) * settings.voxel_size,
                                   origin + glm::vec3(glm::min(lo + edge, dimensions)) * settings.voxel_size))
                    return false;
                if (tree[index] == 1)
                    return true;
                for (int child = 0; child < 8; ++child)
                    if (self(self, tree[index] - 2 + child,
                             lo + glm::ivec3(child & 1, (child >> 1) & 1, child >> 2) * (edge / 2), edge / 2))
                        return true;
                return false;
            };
            return visit(visit, 0, {}, tree_size);
        }
        auto low = glm::max(cell(glm::min(a, b) - radius), glm::ivec3(0));
        auto high = glm::min(cell(glm::max(a, b) + radius), dimensions - 1);
        uint64_t visited = 0;
        for (int z = low.z; z <= high.z; ++z)
            for (int y = low.y; y <= high.y; ++y)
                for (int x = low.x; x <= high.x; ++x) {
                    if (++visited > 2'000'000)
                        throw std::runtime_error("Capsule query exceeds resource limit");
                    if (occupied({x, y, z}) && intersectsBox(origin + glm::vec3(x, y, z) * settings.voxel_size,
                                                             origin + glm::vec3(x + 1, y + 1, z + 1) * settings.voxel_size))
                        return true;
                }
        return false;
    }

    CollisionVolume generateCollision(std::span<const CollisionGaussian> gaussians, CollisionSettings s,
                                      std::stop_token stop, CollisionProgress progress) {
        s.validate();
        if (gaussians.empty())
            throw std::invalid_argument("No source Gaussians");
        if (gaussians.size() > 10'000'000)
            throw std::runtime_error("Collision source exceeds 10 million Gaussians");
        auto report = [&](const char* text) { checkpoint(stop); if (progress) progress(text); };
        std::vector<CollisionGaussian> filtered;
        if (s.cluster) {
            report("Clustering Gaussians");
            auto coarse_settings = s;
            coarse_settings.voxel_size = s.cluster_resolution;
            coarse_settings.opacity = s.cluster_opacity;
            auto coarse = collision_detail::voxelizeSparse(gaussians, coarse_settings, stop);
            auto seed = glm::clamp(coarse.cell(s.seed), glm::ivec3(0), coarse.dimensions - 1);
            if (!coarse.occupied(seed)) {
                const auto requested_seed = seed;
                float best = std::numeric_limits<float>::max();
                bool found = false;
                coarse.forEachSolidBox([&](auto lo, auto hi) {
                    checkpoint(stop);
                    auto p = glm::clamp(requested_seed, lo, hi - 1);
                    auto delta = glm::vec3(p - requested_seed);
                    float distance = glm::dot(delta, delta);
                    if (coarse.occupied(p) && distance < best) {
                        best = distance;
                        seed = p;
                        found = true;
                    }
                });
                if (!found)
                    throw std::runtime_error("Cluster opacity threshold produced no solid voxels");
            }
            auto region = collision_detail::volumeTree(coarse, stop).flood(std::span(&seed, 1), true);
            uint64_t work = 0;
            for (const auto& g : gaussians) {
                checkpoint(stop);
                if (g.opacity < s.cluster_min_contribution || std::abs(glm::determinant(g.axes)) < 1e-20f)
                    continue;
                auto inv = glm::inverse(g.axes);
                glm::vec3 extent;
                for (int a = 0; a < 3; ++a)
                    extent[a] = 3 * glm::length(glm::vec3(g.axes[0][a], g.axes[1][a], g.axes[2][a]));
                auto lo = glm::max(coarse.cell(g.center - extent), glm::ivec3(0));
                auto hi = glm::min(coarse.cell(g.center + extent), coarse.dimensions - 1);
                bool keep = false;
                for (int z = lo.z; z <= hi.z && !keep; ++z)
                    for (int y = lo.y; y <= hi.y && !keep; ++y)
                        for (int x = lo.x; x <= hi.x && !keep; ++x) {
                            if (++work > max_work)
                                throw std::runtime_error("Cluster work limit reached");
                            glm::ivec3 p(x, y, z);
                            if (!region.occupied(p))
                                continue;
                            auto q = inv * (coarse.origin + (glm::vec3(p) + 0.5f) * s.cluster_resolution - g.center);
                            keep = g.opacity * std::exp(-0.5f * glm::dot(q, q)) >= s.cluster_min_contribution;
                        }
                if (keep)
                    filtered.push_back(g);
            }
            if (filtered.empty())
                throw std::runtime_error("Cluster seed/contribution threshold retained no Gaussians");
            gaussians = filtered;
        }
        report("Voxelizing rotated Gaussian ellipsoids");
        auto grid = collision_detail::voxelizeSparse(gaussians, s, stop, progress);
        return processCollisionVolume(std::move(grid), stop, std::move(progress));
    }
    CollisionVolume processCollisionVolume(CollisionVolume grid, std::stop_token stop, CollisionProgress progress) {
        if (!grid.tree.empty())
            return collision_detail::processSparse(std::move(grid), stop, std::move(progress));
        grid.validate();
        auto s = grid.settings;
        auto report = [&](const char* text) { checkpoint(stop); if (progress) progress(text); };
        if (std::none_of(grid.solid.begin(), grid.solid.end(), [](auto v) { return v != 0; }))
            throw std::runtime_error("No occupied voxels; lower opacity or voxel size");
        if (s.external_fill) {
            report("Sealing exterior");
            int radius = static_cast<int>(std::min(std::ceil(s.external_size / (2 * s.voxel_size)),
                                                   float(std::max({grid.dimensions.x, grid.dimensions.y, grid.dimensions.z}))));
            auto sealed = dilate(grid, radius, stop);
            std::vector<glm::ivec3> boundary;
            eachCell(grid, stop, [&](auto p) {
                if (p.x == 0 || p.y == 0 || p.z == 0 || p.x == grid.dimensions.x - 1 ||
                    p.y == grid.dimensions.y - 1 || p.z == grid.dimensions.z - 1)
                    boundary.push_back(p);
            });
            auto exterior = flood(grid, sealed, boundary, stop);
            auto seed = grid.cell(s.seed);
            if (!grid.contains(seed) || exterior[grid.index(seed)] || sealed[grid.index(seed)]) {
                grid.warning = "External fill skipped: seed is exterior or consumed by sealing dilation";
            } else {
                for (size_t i = 0; i < grid.solid.size(); ++i)
                    grid.solid[i] = grid.solid[i] || exterior[i];
            }
        }
        if (s.floor_fill) {
            report("Filling ground columns");
            const int nx = grid.dimensions.x, nz = grid.dimensions.z;
            std::vector<int> floors(size_t(nx) * nz, -1);
            for (int z = 0; z < nz; ++z) {
                checkpoint(stop);
                for (int x = 0; x < nx; ++x)
                    for (int y = 0; y < grid.dimensions.y; ++y)
                        if (grid.occupied({x, y, z}) && (!s.floor_below_seed ||
                                                         grid.origin.y + y * s.voxel_size <= s.seed.y - 1.6f * s.units_per_meter)) {
                            floors[size_t(z) * nx + x] = y;
                            break;
                        }
            }
            auto patched = floors;
            int reach = static_cast<int>(std::min(std::ceil(2 * s.floor_size / s.voxel_size), float(std::max(nx, nz))));
            uint64_t work = 0;
            for (int z = 0; z < nz; ++z) {
                checkpoint(stop);
                for (int x = 0; x < nx; ++x) {
                    if (floors[size_t(z) * nx + x] >= 0)
                        continue;
                    int height = grid.dimensions.y, hits = 0;
                    for (auto d : {glm::ivec2(1, 0), glm::ivec2(-1, 0), glm::ivec2(0, 1), glm::ivec2(0, -1)}) {
                        for (int n = 1; n <= reach; ++n) {
                            if (++work > max_work)
                                throw std::runtime_error("Floor patch work limit reached");
                            auto p = glm::ivec2(x, z) + d * n;
                            if (p.x < 0 || p.y < 0 || p.x >= nx || p.y >= nz)
                                break;
                            int h = floors[size_t(p.y) * nx + p.x];
                            if (h >= 0) {
                                ++hits;
                                height = std::min(height, h);
                                break;
                            }
                        }
                    }
                    if (hits == 4)
                        patched[size_t(z) * nx + x] = height;
                }
            }
            for (int z = 0; z < nz; ++z) {
                checkpoint(stop);
                for (int x = 0; x < nx; ++x)
                    for (int y = 0; y < patched[size_t(z) * nx + x]; ++y)
                        grid.solid[grid.index({x, y, z})] = 1;
            }
        }
        if (s.carve_height > 0) {
            report("Flooding navigable capsule space");
            auto seed = grid.cell(s.seed);
            if (!grid.contains(seed) || grid.capsuleIntersects(s.seed - glm::vec3(0, s.carve_height / 2, 0),
                                                               {0, 1, 0}, s.carve_height, s.carve_radius))
                throw std::runtime_error("Carve capsule does not fit at seed (seed denotes capsule center)");
            auto blocked = capsuleDilate(grid, grid.solid, s.carve_height, s.carve_radius, stop);
            const float horizontal_margin = s.carve_radius / s.voxel_size + 0.5f;
            const float vertical_margin = s.carve_height / (2 * s.voxel_size) + 0.5f;
            eachCell(grid, stop, [&](auto p) {
                if (p.x < horizontal_margin || p.z < horizontal_margin ||
                    grid.dimensions.x - 1 - p.x < horizontal_margin || grid.dimensions.z - 1 - p.z < horizontal_margin ||
                    p.y < vertical_margin || grid.dimensions.y - 1 - p.y < vertical_margin)
                    blocked[grid.index(p)] = 1;
            });
            if (blocked[grid.index(seed)])
                throw std::runtime_error("Carve capsule does not fit at the seed voxel; move seed or reduce capsule");
            auto reached = flood(grid, blocked, std::span(&seed, 1), stop);
            auto navigable = capsuleDilate(grid, reached, s.carve_height, s.carve_radius, stop);
            for (size_t i = 0; i < grid.solid.size(); ++i)
                grid.solid[i] = grid.solid[i] || !navigable[i];
        }
        grid.validate();
        report("Building collision surface");
        return grid;
    }

    std::vector<uint32_t> CollisionVolume::packedOccupancy() const {
        validate();
        const auto count = cellCount(dimensions);
        std::vector<uint32_t> result((count + 31) / 32, 0);
        eachCell(*this, {}, [&](auto p) {
            if (occupied(p)) {
                const auto i = index(p);
                result[i / 32] |= uint32_t(1) << (i % 32);
            }
        });
        return result;
    }
    std::vector<uint32_t> CollisionVolume::octreeOccupancy() const {
        validate();
        return collision_detail::volumeTree(*this).nodes;
    }
    std::vector<uint32_t> CollisionVolume::gpuOccupancy() const {
        validate();
        const auto dense_tree = tree.empty() ? octreeOccupancy() : std::vector<uint32_t>{};
        const auto& source = tree.empty() ? dense_tree : tree;
        int root_size = 1;
        while (root_size < std::max({dimensions.x, dimensions.y, dimensions.z}))
            root_size *= 2;
        std::vector<uint32_t> result(1);
        std::function<void(uint32_t, uint32_t, int)> pack = [&](uint32_t src, uint32_t dst, int edge) {
            if (source[src] <= 1) {
                result[dst] = source[src];
                return;
            }
            if (edge <= 4) {
                const auto base = uint32_t(result.size());
                result.resize(result.size() + 2);
                result[dst] = 0x80000000u | base;
                for (int z = 0; z < edge; ++z)
                    for (int y = 0; y < edge; ++y)
                        for (int x = 0; x < edge; ++x) {
                            auto i = src;
                            int n = edge;
                            glm::ivec3 p(x, y, z);
                            while (source[i] > 1) {
                                n /= 2;
                                auto bit = glm::ivec3(glm::greaterThanEqual(p, glm::ivec3(n)));
                                i = source[i] - 2 + bit.x + 2 * bit.y + 4 * bit.z;
                                p -= bit * n;
                            }
                            if (source[i]) {
                                auto bit = x + 4 * y + 16 * z;
                                result[base + bit / 32] |= uint32_t(1) << (bit % 32);
                            }
                        }
                return;
            }
            auto base = uint32_t(result.size());
            result.resize(result.size() + 8);
            result[dst] = base + 2;
            for (int child = 0; child < 8; ++child)
                pack(source[src] - 2 + child, base + child, edge / 2);
        };
        pack(0, 0, root_size);
        return result;
    }
    void CollisionVolume::forEachSolidBox(const std::function<void(glm::ivec3, glm::ivec3)>& fn) const {
        if (!tree.empty()) {
            std::function<void(uint32_t, glm::ivec3, int)> visit = [&](uint32_t i, glm::ivec3 lo, int n) {
                if (glm::any(glm::greaterThanEqual(lo, dimensions)) || tree[i] == 0)
                    return;
                if (tree[i] == 1) {
                    fn(lo, glm::min(lo + n, dimensions));
                    return;
                }
                for (int a = 0; a < 8; ++a)
                    visit(tree[i] - 2 + a, lo + glm::ivec3(a & 1, (a >> 1) & 1, a >> 2) * (n / 2), n / 2);
            };
            visit(0, {}, tree_size);
        } else {
            for (int z = 0; z < dimensions.z; ++z)
                for (int y = 0; y < dimensions.y; ++y)
                    for (int x = 0; x < dimensions.x;) {
                        if (!occupied({x, y, z})) {
                            ++x;
                            continue;
                        }
                        int start = x++;
                        while (x < dimensions.x && occupied({x, y, z}))
                            ++x;
                        fn({start, y, z}, {x, y + 1, z + 1});
                    }
        }
    }
    uint64_t CollisionVolume::occupiedCount() const {
        uint64_t count = 0;
        forEachSolidBox([&](auto lo, auto hi) {
            auto d = hi - lo;
            count += uint64_t(d.x) * d.y * d.z;
        });
        return count;
    }
    void CollisionVolume::forEachBoundaryFace(const std::function<void(glm::ivec3, glm::ivec3)>& fn,
                                              std::stop_token stop) const {
        const auto occupancy = collision_detail::volumeTree(*this, stop);
        uint64_t work = 0;
        occupancy.leaves([&](const collision_detail::Leaf& l) {
            checkpoint(stop);
            if (!l.solid)
                return;
            auto end = glm::min(l.max(), dimensions);
            for (auto normal : directions) {
                int axis = normal.x ? 0 : normal.y ? 1
                                                   : 2;
                int u = (axis + 1) % 3, v = (axis + 2) % 3;
                auto emit = [&](auto lo, auto hi) {
                    for (int b = lo[v]; b < hi[v]; ++b) {
                        checkpoint(stop);
                        for (int a = lo[u]; a < hi[u]; ++a) {
                            if (++work > 32'000'000)
                                throw std::runtime_error("Collision surface work budget exceeded");
                            glm::ivec3 p = l.min;
                            p[axis] = normal[axis] > 0 ? end[axis] - 1 : l.min[axis];
                            p[u] = a;
                            p[v] = b;
                            fn(p, normal);
                        }
                    }
                };
                auto lo = l.min, hi = end;
                lo[axis] = normal[axis] > 0 ? end[axis] : l.min[axis] - 1;
                hi[axis] = lo[axis] + 1;
                if (lo[axis] < 0 || lo[axis] >= dimensions[axis]) {
                    emit(l.min, end);
                } else {
                    occupancy.visit(lo, hi, [&](const collision_detail::Leaf& neighbor) {
                        if (!neighbor.solid)
                            emit(glm::max(lo, neighbor.min), glm::min(hi, neighbor.max()));
                    });
                }
            }
        });
    }

    CollisionMesh CollisionVolume::mesh(CollisionShape shape, std::stop_token stop, CollisionProgress progress) const {
        validate();
        CollisionMesh result, output;
        std::vector<std::array<int32_t, 3>> keys;
        std::unordered_map<std::array<int32_t, 3>, int32_t, GridPointHash> seams;
        const auto occupancy = collision_detail::volumeTree(*this, stop);
        // Integer half-grid coordinates weld neighboring cubes exactly.
        std::unordered_map<std::array<int32_t, 3>, int32_t, GridPointHash> vertices;
        auto vertex = [&](glm::ivec3 p) {
            const std::array<int32_t, 3> k{p.x, p.y, p.z};
            if (auto it = vertices.find(k); it != vertices.end())
                return it->second;
            if (result.vertices.size() / 3 >= 16'000'000)
                throw std::runtime_error("Collision mesh vertex limit reached; increase voxel size");
            auto id = static_cast<int32_t>(result.vertices.size() / 3);
            auto v = origin + glm::vec3(p) * (settings.voxel_size / 2);
            result.vertices.insert(result.vertices.end(), {v.x, v.y, v.z});
            keys.push_back(k);
            vertices.emplace(k, id);
            return id;
        };
        auto triangle = [&](int a, int b, int c) {
            if (result.indices.size() >= 96'000'000)
                throw std::runtime_error("Collision mesh triangle limit reached; increase voxel size");
            result.indices.insert(result.indices.end(), {a, b, c});
        };
        uint64_t work = 0, chunks = 0;
        std::function<void(glm::ivec3, int)> build = [&](glm::ivec3 lo, int n) {
            checkpoint(stop);
            const auto halo = shape == CollisionShape::Faces ? glm::ivec3(1) : glm::ivec3(0);
            if (occupancy.uniform(lo - halo, lo + n + 1) >= 0)
                return;
            if (n > 32) {
                for (int child = 0; child < 8; ++child)
                    build(lo + glm::ivec3(child & 1, (child >> 1) & 1, child >> 2) * (n / 2), n / 2);
                return;
            }
            work += uint64_t(n) * n * n;
            if (work > 2'000'000'000ULL)
                throw std::runtime_error("Sparse mesh active-chunk work budget exceeded");
            const int stride = n + 2;
            const auto cache_lo = lo - 1;
            std::vector<uint8_t> cache(size_t(stride) * stride * stride, 0);
            occupancy.visit(cache_lo, lo + n + 1, [&](const collision_detail::Leaf& l) {
                if (!l.solid)
                    return;
                auto a = glm::max(l.min, cache_lo), b = glm::min(l.max(), glm::min(lo + n + 1, dimensions));
                for (int z = a.z; z < b.z; ++z)
                    for (int y = a.y; y < b.y; ++y)
                        std::fill_n(cache.begin() + ((z - cache_lo.z) * stride + y - cache_lo.y) * stride + a.x - cache_lo.x,
                                    b.x - a.x, uint8_t(1));
            });
            auto cached = [&](glm::ivec3 p) {
                auto q = p - cache_lo;
                return cache[(q.z * stride + q.y) * stride + q.x] != 0;
            };
            if (shape == CollisionShape::Faces) {
                for (int z = std::max(lo.z, 0); z < std::min(lo.z + n, dimensions.z); ++z)
                    for (int y = std::max(lo.y, 0); y < std::min(lo.y + n, dimensions.y); ++y)
                        for (int x = std::max(lo.x, 0); x < std::min(lo.x + n, dimensions.x); ++x) {
                            glm::ivec3 p(x, y, z);
                            if (!cached(p))
                                continue;
                            for (auto normal : directions) {
                                if (cached(p + normal))
                                    continue;
                                int axis = normal.x ? 0 : normal.y ? 1
                                                                   : 2;
                                int u = (axis + 1) % 3, v = (axis + 2) % 3;
                                glm::ivec3 q = p * 2;
                                if (normal[axis] > 0)
                                    q[axis] += 2;
                                glm::ivec3 du(0), dv(0);
                                du[u] = 2;
                                dv[v] = 2;
                                int a = vertex(q), b = vertex(q + du), c = vertex(q + du + dv), d = vertex(q + dv);
                                if (normal[axis] > 0) {
                                    triangle(a, b, c);
                                    triangle(a, c, d);
                                } else {
                                    triangle(a, c, b);
                                    triangle(a, d, c);
                                }
                            }
                        }
            } else {
                // Table-free marching cubes: face contours link crossing cube edges.
                // Four-crossing faces consistently isolate solid corners, resolving
                // ambiguous shared faces identically from either adjacent cube.
                constexpr glm::ivec3 corners[] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
                constexpr int edges[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
                constexpr int faces[6][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 9, 4, 8}, {2, 10, 6, 11}, {3, 11, 7, 8}, {1, 10, 5, 9}};
                constexpr glm::ivec3 face_normals[] = {{0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0}};
                for (int z = lo.z; z < std::min(lo.z + n, dimensions.z); ++z)
                    for (int y = lo.y; y < std::min(lo.y + n, dimensions.y); ++y)
                        for (int x = lo.x; x < std::min(lo.x + n, dimensions.x); ++x) {
                            glm::ivec3 p(x, y, z);
                            bool inside[8];
                            int count = 0;
                            for (int i = 0; i < 8; ++i) {
                                inside[i] = cached(p + corners[i]);
                                count += inside[i];
                            }
                            if (!count || count == 8)
                                continue;
                            int ids[12], next_edge[12];
                            std::fill_n(ids, 12, -1);
                            std::fill_n(next_edge, 12, -1);
                            for (int e = 0; e < 12; ++e)
                                if (inside[edges[e][0]] != inside[edges[e][1]])
                                    ids[e] = vertex(p * 2 + corners[edges[e][0]] + corners[edges[e][1]] + glm::ivec3(1));
                            for (int face_index = 0; face_index < 6; ++face_index) {
                                const auto& face = faces[face_index];
                                auto link = [&](int a, int b) {
                                    auto edge = edges[a];
                                    auto outward = glm::vec3(corners[inside[edge[0]] ? edge[1] : edge[0]] -
                                                             corners[inside[edge[0]] ? edge[0] : edge[1]]);
                                    auto delta = glm::vec3(corners[edges[b][0]] + corners[edges[b][1]] -
                                                           corners[edges[a][0]] - corners[edges[a][1]]);
                                    if (glm::dot(delta, glm::cross(outward, glm::vec3(face_normals[face_index]))) > 0)
                                        next_edge[a] = b;
                                    else
                                        next_edge[b] = a;
                                };
                                int crossing[4], n = 0;
                                for (auto e : face)
                                    if (ids[e] >= 0)
                                        crossing[n++] = e;
                                if (n == 2)
                                    link(crossing[0], crossing[1]);
                                else if (n == 4) {
                                    // Pair the two edges adjacent to each solid face corner.
                                    for (int i = 0; i < 4; ++i) {
                                        int e = face[i], f = face[(i + 1) % 4], shared = -1;
                                        for (int a : edges[e])
                                            for (int b : edges[f])
                                                if (a == b)
                                                    shared = a;
                                        if (shared >= 0 && inside[shared])
                                            link(e, f);
                                    }
                                }
                            }
                            bool used[12]{};
                            for (int e = 0; e < 12; ++e) {
                                if (ids[e] < 0 || used[e])
                                    continue;
                                std::vector<int> loop;
                                int current = e;
                                do {
                                    if (next_edge[current] < 0 || used[current])
                                        throw std::runtime_error("Invalid marching-cubes contour");
                                    used[current] = true;
                                    loop.push_back(ids[current]);
                                    current = next_edge[current];
                                } while (current != e);
                                if (loop.size() < 3)
                                    continue;
                                if (loop.size() == 3) {
                                    triangle(loop[0], loop[1], loop[2]);
                                } else {
                                    // An interior fan avoids introducing a diagonal on
                                    // a shared cube face (which could have four incident
                                    // triangles when the neighboring cube chooses it too).
                                    glm::vec3 center(0);
                                    for (auto id : loop)
                                        center += glm::vec3(result.vertices[id * 3], result.vertices[id * 3 + 1], result.vertices[id * 3 + 2]);
                                    center /= float(loop.size());
                                    if (result.vertices.size() / 3 >= 16'000'000)
                                        throw std::runtime_error("Collision mesh vertex limit reached; increase voxel size");
                                    auto id = int32_t(result.vertices.size() / 3);
                                    result.vertices.insert(result.vertices.end(), {center.x, center.y, center.z});
                                    keys.push_back({INT32_MAX, INT32_MAX, INT32_MAX});
                                    for (size_t i = 0; i < loop.size(); ++i)
                                        triangle(id, loop[i], loop[(i + 1) % loop.size()]);
                                }
                            }
                        }
            }
            if (shape == CollisionShape::Smooth)
                simplifyCoplanar(result, stop, &keys);
            std::vector<int32_t> remap(result.vertices.size() / 3);
            for (size_t i = 0; i < remap.size(); ++i) {
                const auto& key = keys[i];
                bool seam = false;
                for (int axis = 0; axis < 3; ++axis) {
                    int shift = shape == CollisionShape::Smooth ? 1 : 0;
                    seam |= key[axis] == lo[axis] * 2 + shift || key[axis] == (lo[axis] + n) * 2 + shift;
                }
                if (seam) {
                    if (auto existing = seams.find(key); existing != seams.end()) {
                        remap[i] = existing->second;
                        continue;
                    }
                }
                if (output.vertices.size() / 3 >= CollisionMesh::max_vertices)
                    throw std::runtime_error("Collision output mesh vertex memory budget exceeded");
                remap[i] = int32_t(output.vertices.size() / 3);
                output.vertices.insert(output.vertices.end(), result.vertices.begin() + i * 3, result.vertices.begin() + i * 3 + 3);
                if (seam)
                    seams.emplace(key, remap[i]);
            }
            if (output.indices.size() + result.indices.size() > CollisionMesh::max_indices)
                throw std::runtime_error("Collision output mesh index memory budget exceeded");
            for (auto index : result.indices)
                output.indices.push_back(remap[index]);
            result.vertices.clear();
            result.indices.clear();
            vertices.clear();
            keys.clear();
            if (progress && (++chunks == 1 || chunks % 32 == 0))
                progress(std::format("Surface: {} chunks, {} triangles", chunks, output.indices.size() / 3));
        };
        int root_size = 1;
        while (root_size < std::max({dimensions.x, dimensions.y, dimensions.z}) + 1)
            root_size *= 2;
        build(glm::ivec3(-1), root_size);
        return output;
    }

    CollisionMesh CollisionVolume::boundsMesh() const {
        CollisionMesh result;
        for (int c = 0; c < 8; ++c) {
            const auto p = origin + glm::vec3((c & 1) ? dimensions.x : 0,
                                              (c & 2) ? dimensions.y : 0,
                                              (c & 4) ? dimensions.z : 0) *
                                        settings.voxel_size;
            result.vertices.insert(result.vertices.end(), {p.x, p.y, p.z});
        }
        result.indices = {0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6, 0, 1, 5, 0, 5, 4,
                          2, 6, 7, 2, 7, 3, 0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5};
        return result;
    }

    CollisionVolume CollisionVolume::coarsened(int stride, std::stop_token stop) const {
        checkpoint(stop);
        if (stride < 1 || stride > 16 || (stride & (stride - 1)))
            throw std::invalid_argument("Surface stride must be a power of two from 1 to 16");
        validate();
        CollisionVolume out;
        out.settings = settings;
        out.settings.voxel_size *= stride;
        out.origin = origin;
        out.dimensions = (dimensions + stride - 1) / stride;
        collision_detail::Tree reduced(out.dimensions, stop);
        if (tree.empty()) {
            forEachSolidBox([&](auto lo, auto hi) { reduced.setBox(lo / stride, (hi + stride - 1) / stride); });
        } else {
            auto anySolid = [&](auto&& self, uint32_t i) -> bool {
                if (tree[i] <= 1)
                    return tree[i] != 0;
                for (int c = 0; c < 8; ++c)
                    if (self(self, tree[i] - 2 + c))
                        return true;
                return false;
            };
            auto visit = [&](auto&& self, uint32_t src, uint32_t dst, int edge) -> void {
                checkpoint(stop);
                if (tree[src] <= 1 || edge <= stride) {
                    reduced.nodes[dst] = anySolid(anySolid, src);
                    return;
                }
                auto base = uint32_t(reduced.nodes.size());
                reduced.nodes.resize(reduced.nodes.size() + 8);
                reduced.nodes[dst] = base + 2;
                for (int c = 0; c < 8; ++c)
                    self(self, tree[src] - 2 + c, base + c, edge / 2);
            };
            visit(visit, 0, 0, tree_size);
        }
        collision_detail::storeTree(out, std::move(reduced));
        return out;
    }

    void WalkSettings::validate() const {
        positive(units_per_meter, "Scene units per metre");
        positive(height, "Player height");
        positive(radius, "Player radius");
        positive(speed, "Walk speed");
        positive(step_height, "Step height", true);
        positive(gravity, "Gravity");
        if (height < 2 * radius || height > 10 * units_per_meter || radius > 5 * units_per_meter || speed > 50 * units_per_meter || step_height > height)
            throw std::invalid_argument("Invalid walk capsule dimensions or speed");
    }
    glm::mat3 walkTangentFrame(glm::vec3 forward, glm::vec3 right, glm::vec3 up) {
        if (!finite(forward) || !finite(right) || !finite(up) || glm::length(up) < 1e-6f)
            throw std::invalid_argument("Invalid walk navigation frame");
        up = glm::normalize(up);
        forward -= up * glm::dot(forward, up);
        if (glm::length(forward) < 1e-5f) {
            right -= up * glm::dot(right, up);
            if (glm::length(right) < 1e-5f) {
                auto axis = std::abs(up.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 0, 1);
                right = axis - up * glm::dot(axis, up);
            }
            forward = glm::cross(up, glm::normalize(right));
        }
        forward = glm::normalize(forward);
        right = glm::normalize(glm::cross(forward, up));
        return {right, up, -forward};
    }
    bool transformedCapsuleIntersects(const CollisionVolume& volume, const glm::mat4& transform,
                                      glm::vec3 feet, glm::vec3 up, float height, float radius) {
        if (std::abs(glm::determinant(glm::mat3(transform))) < 1e-10f)
            throw std::invalid_argument("Collision node has a singular transform");
        auto inverse = glm::inverse(transform);
        auto linear = glm::mat3(inverse);
        float norm1 = 0, norm_inf = 0, max_scale = 0;
        bool orthogonal = true;
        for (int a = 0; a < 3; ++a) {
            float column = 0, row = 0;
            max_scale = std::max(max_scale, glm::length(linear[a]));
            for (int b = 0; b < 3; ++b) {
                column += std::abs(linear[a][b]);
                row += std::abs(linear[b][a]);
                if (a != b && std::abs(glm::dot(linear[a], linear[b])) >
                                  1e-5f * glm::length(linear[a]) * glm::length(linear[b]))
                    orthogonal = false;
            }
            norm1 = std::max(norm1, column);
            norm_inf = std::max(norm_inf, row);
        }
        float radius_scale = orthogonal ? max_scale : std::sqrt(norm1 * norm_inf);
        auto local_up = linear * up;
        float scale = glm::length(local_up);
        auto normalized_up = local_up / scale;
        float local_radius = radius * radius_scale;
        float local_height = (height - 2 * radius) * scale + 2 * local_radius;
        auto local_feet = glm::vec3(inverse * glm::vec4(feet + up * radius, 1)) -
                          normalized_up * local_radius;
        return volume.capsuleIntersects(local_feet, normalized_up, local_height, local_radius);
    }
    void advanceWalk(WalkState& state, const WalkSettings& s, glm::vec3 up,
                     glm::vec3 velocity, float dt, const CapsuleQuery& blocked) {
        s.validate();
        if (!finite(state.eye) || !finite(up) || !finite(velocity) || !std::isfinite(dt) || dt < 0)
            throw std::invalid_argument("Invalid walk integration input");
        if (glm::length(up) < 1e-6f)
            throw std::invalid_argument("Walk up axis must be nonzero");
        up = glm::normalize(up);
        dt = std::min(dt, 0.1f);
        velocity -= up * glm::dot(velocity, up);
        auto feet = state.eye - up * s.height;
        const int steps = std::clamp(int(std::ceil((glm::length(velocity) * dt +
                                                    std::abs(state.vertical_velocity) * dt + s.gravity * dt * dt) /
                                                   std::min(0.02f * s.units_per_meter, s.radius / 3))),
                                     1, 1024);
        float tick = dt / steps;
        auto hit = [&](glm::vec3 p) { return blocked(p, up, s.height, s.radius); };
        for (int step = 0; step < steps; ++step) {
            auto displacement = velocity * tick;
            if (!hit(feet + displacement)) {
                feet += displacement;
            } else {
                bool can_step = state.grounded && s.step_height > 0;
                if (can_step) {
                    const int lift_steps = std::max(1, int(std::ceil(s.step_height / std::min(0.02f * s.units_per_meter, s.radius / 3))));
                    for (int lift = 1; lift <= lift_steps; ++lift)
                        if (hit(feet + up * (s.step_height * lift / lift_steps))) {
                            can_step = false;
                            break;
                        }
                    if (can_step && hit(feet + up * s.step_height + displacement))
                        can_step = false;
                }
                if (can_step) {
                    feet += up * s.step_height + displacement;
                } else {
                    // Projected world-axis components provide stable wall sliding for
                    // both navigation up axes without introducing a physics dependency.
                    for (int a = 0; a < 3; ++a) {
                        glm::vec3 d(0);
                        d[a] = displacement[a];
                        d -= up * glm::dot(d, up);
                        if (!hit(feet + d))
                            feet += d;
                    }
                }
            }
            state.vertical_velocity -= s.gravity * tick;
            auto vertical = up * state.vertical_velocity * tick;
            if (!hit(feet + vertical)) {
                feet += vertical;
                state.grounded = false;
            } else {
                state.grounded = state.vertical_velocity <= 0;
                state.vertical_velocity = 0;
            }
        }
        state.eye = feet + up * s.height;
    }
} // namespace lfs::core
