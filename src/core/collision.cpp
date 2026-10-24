/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include "collision_sparse.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
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
        positive(max_gaussian_area, "Maximum Gaussian area", true);
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
            if ((tree.size() - 1) % 8)
                throw std::invalid_argument("Invalid sparse collision sibling blocks");
            // Each canonical eight-child block has exactly one earlier parent.
            // Forward references rule out cycles, and block levels check depth.
            // One byte per block replaces one byte per node plus recursive calls.
            std::vector<uint8_t> levels((tree.size() - 1) / 8);
            size_t branches = 0;
            const auto root_level = std::bit_width(unsigned(tree_size));
            for (size_t i = 0; i < tree.size(); ++i) {
                const auto level = i ? levels[(i - 1) / 8] : root_level;
                if (!level)
                    throw std::invalid_argument("Unreachable sparse collision nodes");
                if (tree[i] <= 1)
                    continue;
                const auto base = tree[i] - 2;
                if (level <= 1 || base <= i || (base - 1) % 8 || uint64_t(base) + 8 > tree.size())
                    throw std::invalid_argument("Invalid sparse collision branch");
                auto& child_level = levels[(base - 1) / 8];
                if (child_level)
                    throw std::invalid_argument("Invalid sparse collision child reference");
                child_level = uint8_t(level - 1);
                ++branches;
            }
            if (branches != levels.size())
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
        const auto segment_lo = glm::min(a, b), segment_hi = glm::max(a, b);
        const bool axis_aligned = (delta.x == 0 && delta.z == 0) ||
                                  (delta.x == 0 && delta.y == 0) ||
                                  (delta.y == 0 && delta.z == 0);
        // Segment/AABB distance is piecewise quadratic. Evaluate its exact
        // minimum on intervals split at the six box-plane crossings, instead
        // of twenty ternary-search iterations for every occupied voxel.
        auto intersectsBox = [&](glm::vec3 lo, glm::vec3 hi) {
            const auto separation = glm::max(glm::max(lo - segment_hi, segment_lo - hi), glm::vec3(0));
            if (glm::dot(separation, separation) >= radius * radius)
                return false;
            // For a segment parallel to any box axis this lower bound is the
            // exact distance. The normal Walk capsule needs no plane sorting.
            if (axis_aligned)
                return true;
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

    namespace collision_detail {
        float recommendCollisionVoxelSize(std::span<const CollisionGaussian> source, float units, std::stop_token stop) {
            positive(units, "Scene units per metre");
            std::vector<double> areas;
            areas.reserve(source.size());
            for (size_t i = 0; i < source.size(); ++i) {
                if ((i & 4095) == 0)
                    checkpoint(stop);
                const auto& g = source[i];
                if (g.opacity <= 0)
                    continue;
                const auto area = std::max({glm::length(glm::cross(glm::dvec3(g.axes[0]), glm::dvec3(g.axes[1]))),
                                            glm::length(glm::cross(glm::dvec3(g.axes[0]), glm::dvec3(g.axes[2]))),
                                            glm::length(glm::cross(glm::dvec3(g.axes[1]), glm::dvec3(g.axes[2])))});
                if (area > 0 && std::isfinite(area))
                    areas.push_back(area);
            }
            if (areas.empty())
                return .05f * units;
            const auto middle = areas.begin() + areas.size() / 2;
            std::nth_element(areas.begin(), middle, areas.end());
            // Robust observed surface footprint, not empty scene AABB or a metre
            // inference. Aim for 32M surface samples of two-sigma ellipses. This
            // is an estimate, never a guarantee or a hidden retry/coarsening.
            const auto edge = std::max(.05 * double(units), std::sqrt(*middle * areas.size() * 4 * std::numbers::pi / 32'000'000));
            return float(std::ceil(edge / units * 100 - 1e-6) / 100 * units);
        }

        AdaptiveSelection selectAdaptiveSource(std::span<const CollisionGaussian> source,
                                               const CollisionSettings& settings, std::stop_token stop,
                                               CollisionProgress progress) {
            checkpoint(stop);
            AdaptiveSelection result;
            result.indices.reserve(source.size());
            std::vector<float> thickness;
            thickness.reserve(source.size());
            std::vector<float> volume;
            volume.reserve(source.size());
            std::vector<float> area_samples;
            area_samples.reserve(source.size());
            auto measure = [](const CollisionGaussian& g, double* projected_area = nullptr) {
                // Minimum parallelepiped altitude: rotation invariant, and thin
                // sheared/anisotropic sheets stay thin. A long axis alone is not
                // evidence of a volumetric reconstruction outlier.
                const double area = std::max({glm::length(glm::cross(glm::dvec3(g.axes[0]), glm::dvec3(g.axes[1]))),
                                              glm::length(glm::cross(glm::dvec3(g.axes[0]), glm::dvec3(g.axes[2]))),
                                              glm::length(glm::cross(glm::dvec3(g.axes[1]), glm::dvec3(g.axes[2])))});
                if (projected_area)
                    *projected_area = area;
                return area > 0 ? float(std::abs(glm::determinant(glm::dmat3(g.axes))) / area) : 0.f;
            };
            for (size_t i = 0; i < source.size(); ++i) {
                if ((i & 4095) == 0)
                    checkpoint(stop);
                const auto& g = source[i];
                if (!finite(g.center) || !finite(g.axes[0]) || !finite(g.axes[1]) || !finite(g.axes[2]) ||
                    !std::isfinite(g.opacity) || g.opacity < 0 || g.opacity > 1)
                    throw std::invalid_argument("Invalid Gaussian in collision source");
                double area = 0;
                const auto t = measure(g, &area);
                if (t > 0 && g.opacity > 0) {
                    area_samples.push_back(float(std::log(area)));
                    thickness.push_back(std::log(t));
                    volume.push_back(float(std::log(std::abs(glm::determinant(glm::dmat3(g.axes)))) / 3.));
                }
            }
            // Far-out fences in log thickness and equivalent-volume radius,
            // with an eightfold minimum spread. Exceptionally thin sheets are
            // exempt from volume rejection. Small fixtures bypass these volume
            // statistics; a separate area fence below also handles thin splats.
            float fence = std::numeric_limits<float>::infinity();
            float volume_fence = fence, thin_guard = fence, area_fence = fence;
            if (thickness.size() >= 128) {
                auto quantile = [&](size_t i) {
                    std::nth_element(thickness.begin(), thickness.begin() + i, thickness.end());
                    return thickness[i];
                };
                const auto lower = quantile(thickness.size() / 4);
                const auto upper = quantile(3 * thickness.size() / 4);
                fence = upper + std::max(3 * (upper - lower), float(std::log(8.0)));
                thin_guard = lower - float(std::log(4.));
                std::nth_element(volume.begin(), volume.begin() + volume.size() / 4, volume.end());
                const auto volume_lower = volume[volume.size() / 4];
                std::nth_element(volume.begin(), volume.begin() + 3 * volume.size() / 4, volume.end());
                const auto volume_upper = volume[3 * volume.size() / 4];
                volume_fence = volume_upper + std::max(3 * (volume_upper - volume_lower), float(std::log(8.)));
            }
            if (area_samples.size() >= 16) {
                std::nth_element(area_samples.begin(), area_samples.begin() + area_samples.size() / 4, area_samples.end());
                const auto area_lower = area_samples[area_samples.size() / 4];
                std::nth_element(area_samples.begin(), area_samples.begin() + 3 * area_samples.size() / 4, area_samples.end());
                const auto area_upper = area_samples[3 * area_samples.size() / 4];
                area_fence = area_upper + std::max(3 * (area_upper - area_lower), float(std::log(64.)));
            }
            result.area_limit = settings.max_gaussian_area > 0 ? settings.max_gaussian_area : std::exp(double(area_fence)) * 9 * std::numbers::pi;
            size_t volume_outliers = 0;
            for (size_t i = 0; i < source.size(); ++i) {
                if ((i & 4095) == 0)
                    checkpoint(stop);
                double area = 0;
                const auto t = measure(source[i], &area);
                const auto log_t = t > 0 ? std::log(t) : -std::numeric_limits<float>::infinity();
                const auto log_v = t > 0 ? std::log(std::abs(glm::determinant(glm::dmat3(source[i].axes)))) / 3. : -std::numeric_limits<double>::infinity();
                if (area * 9 * std::numbers::pi > result.area_limit) {
                    result.oversized_indices.push_back(uint32_t(i));
                    continue;
                }
                if (log_t <= fence && !(log_v > volume_fence && log_t > thin_guard))
                    result.indices.push_back(uint32_t(i));
                else
                    ++volume_outliers;
            }
            result.note = std::format("Surface cleanup kept {} / {} splats: {} oversized-area and {} volume outliers excluded.",
                                      result.indices.size(), source.size(), result.oversized_indices.size(), volume_outliers);
            if (std::isfinite(result.area_limit))
                result.note += std::format(" Area limit: {:.4g} m² ({}).", result.area_limit / (double(settings.units_per_meter) * settings.units_per_meter), settings.max_gaussian_area > 0 ? "custom" : "automatic");
            if (progress)
                progress("Clustering cleanup: " + result.note);
            if (!settings.cluster || result.indices.empty()) {
                if (result.indices.size() == source.size())
                    result.note.clear();
                return result;
            }

            // Connectivity comes from observed Gaussian centres, not accumulated
            // alpha. A sparse sheet need not reach 99.9% opacity to be structural.
            // The graph bridges diagonal cell neighbours but does not dilate the
            // final collision grid or connect via a giant Gaussian's support box.
            using Key = std::array<int32_t, 3>;
            struct Cell {
                Key p;
                uint32_t parent, count;
            };
            std::unordered_map<Key, uint32_t, GridPointHash> lookup;
            lookup.reserve(std::min(result.indices.size(), size_t(1'000'000)));
            std::vector<Cell> cells;
            std::vector<uint32_t> memberships;
            memberships.reserve(result.indices.size());
            const auto size = double(settings.cluster_resolution);
            auto point = [&](glm::vec3 p) {
                const auto v = glm::floor(glm::dvec3(p) / size);
                // Leave room for neighbour offsets; no signed overflow or silent
                // coordinate clipping for remote components.
                for (int a = 0; a < 3; ++a)
                    if (v[a] < double(INT32_MIN) + 1 || v[a] > double(INT32_MAX) - 1)
                        throw std::runtime_error("Cluster coordinates exceed integer range; increase cluster resolution");
                return Key{int32_t(v.x), int32_t(v.y), int32_t(v.z)};
            };
            for (auto i : result.indices) {
                if ((memberships.size() & 4095) == 0)
                    checkpoint(stop);
                if (source[i].opacity == 0 || std::abs(glm::determinant(source[i].axes)) < 1e-20f) {
                    memberships.push_back(UINT32_MAX);
                    continue;
                }
                const auto p = point(source[i].center);
                const auto [it, added] = lookup.try_emplace(p, uint32_t(cells.size()));
                if (added)
                    cells.push_back({p, it->second, 0});
                ++cells[it->second].count;
                memberships.push_back(it->second);
            }
            if (cells.empty()) {
                result.indices.clear();
                result.note += " No contributing source cells were found.";
                return result;
            }
            auto root = [&](uint32_t i) {
                while (cells[i].parent != i) {
                    cells[i].parent = cells[cells[i].parent].parent;
                    i = cells[i].parent;
                }
                return i;
            };
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if ((i & 4095) == 0) {
                    checkpoint(stop);
                    if (progress)
                        progress(std::format("Clustering source connectivity: {}/{}", i, cells.size()));
                }
                const auto p = cells[i].p;
                for (int z = -1; z <= 1; ++z)
                    for (int y = -1; y <= 1; ++y)
                        for (int x = -1; x <= 1; ++x) {
                            // Only one half of the 26 undirected neighbours.
                            if (z < 0 || (z == 0 && y < 0) || (z == 0 && y == 0 && x <= 0))
                                continue;
                            const auto it = lookup.find({p[0] + x, p[1] + y, p[2] + z});
                            if (it == lookup.end())
                                continue;
                            auto a = root(i), b = root(it->second);
                            if (a == b)
                                continue;
                            if (cells[a].count < cells[b].count)
                                std::swap(a, b);
                            cells[b].parent = a;
                            cells[a].count += cells[b].count;
                        }
            }
            // Respect an explicit seed inside observed geometry. Otherwise prefer
            // substantial nearby support over one tiny floater nearest the camera.
            const auto seed = glm::dvec3(settings.seed) / size;
            double nearest = std::numeric_limits<double>::infinity();
            uint32_t selected = 0;
            const auto exact = lookup.find(point(settings.seed));
            if (exact != lookup.end())
                selected = root(exact->second);
            else {
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if ((i & 4095) == 0)
                        checkpoint(stop);
                    const auto p = cells[i].p;
                    const auto d = glm::max(glm::max(glm::dvec3(p[0], p[1], p[2]) - seed,
                                                     seed - glm::dvec3(p[0] + 1., p[1] + 1., p[2] + 1.)),
                                            glm::dvec3(0));
                    const auto distance = glm::dot(d, d);
                    if (distance < nearest || (distance == nearest && cells[root(i)].count > cells[root(selected)].count)) {
                        nearest = distance;
                        selected = root(i);
                    }
                }
                const auto radius2 = std::pow(std::sqrt(nearest) + 2., 2);
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if ((i & 4095) == 0)
                        checkpoint(stop);
                    const auto p = cells[i].p;
                    const auto d = glm::max(glm::max(glm::dvec3(p[0], p[1], p[2]) - seed,
                                                     seed - glm::dvec3(p[0] + 1., p[1] + 1., p[2] + 1.)),
                                            glm::dvec3(0));
                    if (glm::dot(d, d) <= radius2 && cells[root(i)].count > cells[selected].count)
                        selected = root(i);
                }
            }
            size_t kept = 0;
            for (size_t i = 0; i < memberships.size(); ++i) {
                if ((i & 4095) == 0)
                    checkpoint(stop);
                if (memberships[i] != UINT32_MAX && root(memberships[i]) == selected)
                    result.indices[kept++] = result.indices[i];
            }
            result.indices.resize(kept);
            result.note += std::format(" Connected source region kept {} / {} splats at {:.4g} scene-unit spacing; final voxel size is unchanged.", kept, source.size(), size);
            if (progress)
                progress("Clustering complete: " + result.note);
            return result;
        }

        ClusterSelection selectCluster(std::span<const CollisionGaussian> source, const CollisionVolume& coarse,
                                       glm::vec3 requested_seed, float min_contribution, std::stop_token stop,
                                       CollisionProgress progress) {
            checkpoint(stop);
            if (!finite(requested_seed))
                throw std::invalid_argument("Non-finite cluster seed");
            positive(min_contribution, "Minimum cluster contribution", true);
            const auto size = coarse.settings.voxel_size;
            const auto raw_cell = glm::floor((glm::dvec3(requested_seed) - glm::dvec3(coarse.origin)) / double(size));
            const auto requested_cell = glm::clamp(raw_cell, glm::dvec3(0), glm::dvec3(coarse.dimensions - 1));
            auto seed = glm::ivec3(requested_cell);
            const bool snapped = raw_cell != requested_cell || !coarse.occupied(seed);
            if (!coarse.occupied(seed)) {
                const auto requested_cell = seed;
                double best = std::numeric_limits<double>::max();
                coarse.forEachSolidBox([&](auto lo, auto hi) {
                    checkpoint(stop);
                    const auto p = glm::clamp(requested_cell, lo, hi - 1);
                    const auto delta = glm::dvec3(p) - glm::dvec3(requested_cell);
                    const auto distance = glm::dot(delta, delta);
                    if (distance < best) {
                        best = distance;
                        seed = p;
                    }
                });
                if (best == std::numeric_limits<double>::max())
                    throw std::runtime_error("Cluster opacity threshold produced no solid voxels");
            }
            // Use a compact coarse tree rather than a dense
            // world grid. Flood visits face-connected adaptive leaves only.
            auto tree = volumeTree(coarse, stop).flood(std::span(&seed, 1), true);
            ClusterSelection result;
            result.resolved_seed = snapped ? coarse.origin + (glm::vec3(seed) + 0.5f) * size : requested_seed;
            tree.leaves([&](const Leaf& leaf) {
                if (leaf.solid) {
                    const auto d = glm::min(leaf.max(), tree.dimensions) - leaf.min;
                    result.occupied_voxels += uint64_t(d.x) * d.y * d.z;
                }
            });
            result.indices.reserve(source.size());
            uint64_t work = 0;
            for (size_t i = 0; i < source.size(); ++i) {
                if ((i & 4095) == 0) {
                    checkpoint(stop);
                    if (progress)
                        progress(std::format("Clustering splats: {}/{}", i, source.size()));
                }
                const auto& g = source[i];
                // Upstream's fast path also preserves low-opacity and sub-voxel
                // detail whose centers lie in the selected component.
                const auto center_cell = glm::floor((glm::dvec3(g.center) - glm::dvec3(coarse.origin)) / double(size));
                if (glm::all(glm::greaterThanEqual(center_cell, glm::dvec3(0))) &&
                    glm::all(glm::lessThan(center_cell, glm::dvec3(tree.dimensions))) && tree.occupied(glm::ivec3(center_cell))) {
                    result.indices.push_back(uint32_t(i));
                    continue;
                }
                if (g.opacity < min_contribution || std::abs(glm::determinant(g.axes)) < 1e-20f)
                    continue;
                glm::dvec3 extent;
                for (int a = 0; a < 3; ++a)
                    extent[a] = 3 * glm::length(glm::dvec3(g.axes[0][a], g.axes[1][a], g.axes[2][a]));
                const double aabb_voxels = std::max({extent.x, extent.y, extent.z}) > size ? (2 * extent.x / size) * (2 * extent.y / size) * (2 * extent.z / size) : 0;
                const double required = std::max(1.0, std::ceil(aabb_voxels * 0.1));
                if (required > double(result.occupied_voxels))
                    continue;
                const auto min = glm::clamp(glm::floor((glm::dvec3(g.center) - extent - glm::dvec3(coarse.origin)) / double(size)),
                                            glm::dvec3(0), glm::dvec3(tree.dimensions));
                const auto max = glm::clamp(glm::floor((glm::dvec3(g.center) + extent - glm::dvec3(coarse.origin)) / double(size)) + 1.0,
                                            glm::dvec3(0), glm::dvec3(tree.dimensions));
                const auto lo = glm::ivec3(min), hi = glm::ivec3(max);
                const auto inverse = glm::inverse(g.axes);
                const auto inv = glm::dmat3(inverse);
                uint64_t hits = 0;
                const auto needed = uint64_t(required);
                const auto limit = min_contribution == 0 ? std::numeric_limits<double>::infinity() : -2 * std::log(double(min_contribution) / g.opacity);
                // Query occupied tree regions directly. Empty world volume is
                // never scanned; uniform qualifying regions count in one step.
                auto visit = [&](auto&& self, uint32_t index, glm::ivec3 origin, int n) -> void {
                    if (hits >= needed || tree.nodes[index] == 0)
                        return;
                    if ((++work & 4095) == 0)
                        checkpoint(stop);
                    if (work > max_work)
                        throw std::runtime_error("Cluster query work budget exceeded");
                    const auto a = glm::max(lo, origin), b = glm::min(hi, origin + n);
                    if (glm::any(glm::greaterThanEqual(a, b)))
                        return;
                    const auto middle = glm::dvec3(coarse.origin) + (glm::dvec3(a) + glm::dvec3(b)) * (double(size) / 2);
                    const auto half = glm::dvec3(b - a - 1) * (double(size) / 2);
                    const auto error = (glm::abs(middle) + glm::abs(glm::dvec3(coarse.origin)) + glm::abs(glm::dvec3(g.center)) + 1.0) *
                                       (8 * double(std::numeric_limits<float>::epsilon()));
                    const auto q = inv * (middle - glm::dvec3(g.center));
                    const auto h = half + error;
                    const auto qr = glm::abs(inv[0]) * h.x + glm::abs(inv[1]) * h.y + glm::abs(inv[2]) * h.z;
                    const auto near = glm::max(glm::abs(q) - qr, glm::dvec3(0));
                    if (glm::dot(near, near) > limit)
                        return;
                    const auto value = tree.nodes[index];
                    const auto far = glm::abs(q) + qr;
                    if (value == 1 && glm::dot(far, far) < limit) {
                        const auto d = b - a;
                        hits += std::min(needed - hits, uint64_t(d.x) * d.y * d.z);
                        return;
                    }
                    if (n == 1) {
                        const auto point = coarse.origin + (glm::vec3(origin) + 0.5f) * size;
                        const auto local = inverse * (point - g.center);
                        if (g.opacity * std::exp(-0.5f * glm::dot(local, local)) >= min_contribution)
                            ++hits;
                        return;
                    }
                    const auto half_size = n / 2;
                    for (int child = 0; child < 8 && hits < needed; ++child)
                        self(self, value == 1 ? index : value - 2 + child,
                             origin + glm::ivec3(child & 1, (child >> 1) & 1, child >> 2) * half_size, half_size);
                };
                visit(visit, 0, {}, tree.size);
                if (hits >= needed)
                    result.indices.push_back(uint32_t(i));
            }
            if (result.indices.empty())
                throw std::runtime_error("Cluster seed/contribution threshold retained no Gaussians");
            if (progress)
                progress(std::format("Clustering splats: {}/{}", source.size(), source.size()));
            return result;
        }
    } // namespace collision_detail

    CollisionVolume generateCollision(std::span<const CollisionGaussian> gaussians, CollisionSettings s,
                                      std::stop_token stop, CollisionProgress progress, CollisionVoxelizer backend) {
        s.validate();
        if (gaussians.empty())
            throw std::invalid_argument("No source Gaussians");
        if (gaussians.size() > CollisionGaussian::max_count)
            throw std::runtime_error("Collision source exceeds 32 million Gaussians");
        auto report = [&](const char* text) { checkpoint(stop); if (progress) progress(text); };
        auto voxelize = [&](std::span<const CollisionGaussian> source, CollisionSettings options, CollisionProgress report) {
            return backend ? backend(source, options, stop, std::move(report)) : collision_detail::voxelizeSparse(source, options, stop, std::move(report));
        };
        std::vector<CollisionGaussian> filtered;
        std::string cluster_note;
        if (s.adaptive_filter) {
            report("Clustering source cleanup");
            auto selection = collision_detail::selectAdaptiveSource(gaussians, s, stop, progress);
            cluster_note = std::move(selection.note);
            if (selection.indices.size() != gaussians.size()) {
                filtered.reserve(selection.indices.size());
                for (auto i : selection.indices)
                    filtered.push_back(gaussians[i]);
                gaussians = filtered;
            }
            if (gaussians.empty())
                throw std::runtime_error("Source cleanup excluded all splats; increase maximum splat area or choose Keep all splats");
        }
        if (s.cluster && !s.adaptive_filter) {
            report("Clustering Gaussians");
            auto coarse_settings = s;
            coarse_settings.voxel_size = s.cluster_resolution;
            coarse_settings.opacity = s.cluster_opacity;
            auto coarse = voxelize(gaussians, coarse_settings, [&](std::string text) {
                if (progress)
                    progress("Clustering coarse " + text);
            });
            report("Clustering connected component");
            const auto selection = collision_detail::selectCluster(gaussians, coarse, s.seed, s.cluster_min_contribution, stop, progress);
            cluster_note = std::format("Cluster filter kept {} / {} splats.", selection.indices.size(), gaussians.size());
            const float seed_offset = glm::distance(selection.resolved_seed, s.seed);
            if (seed_offset > 0)
                cluster_note += std::format(" Seed snapped {:.3g} scene units to an occupied region; move the seed if the wrong region is selected.", seed_offset);
            if (progress)
                progress("Clustering complete: " + cluster_note);
            if (selection.indices.size() != gaussians.size()) {
                filtered.reserve(selection.indices.size());
                for (auto i : selection.indices)
                    filtered.push_back(gaussians[i]);
                gaussians = filtered;
            }
        }
        if (s.auto_resolution) {
            s.voxel_size = collision_detail::recommendCollisionVoxelSize(gaussians, s.units_per_meter, stop);
            const auto note = std::format("Auto cell size: {:.4g} m ({:.4g} scene units).", s.voxel_size / s.units_per_meter, s.voxel_size);
            if (!cluster_note.empty())
                cluster_note += " ";
            cluster_note += note;
            if (progress)
                progress("Voxelizing " + note);
        }
        report("Voxelizing rotated Gaussian ellipsoids");
        auto grid = voxelize(gaussians, s, progress);
        grid = processCollisionVolume(std::move(grid), stop, std::move(progress));
        if (!cluster_note.empty())
            grid.warning = cluster_note + (grid.warning.empty() ? "" : " " + grid.warning);
        return grid;
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
    namespace {
        struct WalkSweep {
            glm::vec3 position;
            bool hit = false;
        };
        // Sample at a fraction of capsule radius, then refine the first contact.
        // Used for lift, travel and settling so thin ceilings are never skipped by
        // a direct step-height teleport.
        template <class Hit>
        WalkSweep sweepWalk(glm::vec3 start, glm::vec3 delta, float spacing, float tolerance, const Hit& hit) {
            const float length = glm::length(delta);
            if (length == 0)
                return {start};
            const int count = std::max(1, int(std::ceil(length / spacing)));
            float clear = 0;
            for (int i = 1; i <= count; ++i) {
                float contact = float(i) / count;
                if (hit(start + delta * contact)) {
                    for (int j = 0; j < 16 && (contact - clear) * length > tolerance; ++j) {
                        const float middle = (clear + contact) * .5f;
                        if (hit(start + delta * middle))
                            contact = middle;
                        else
                            clear = middle;
                    }
                    return {start + delta * clear, true};
                }
                clear = contact;
            }
            return {start + delta};
        }
    } // namespace

    void advanceWalk(WalkState& state, const WalkSettings& s, glm::vec3 up,
                     glm::vec3 velocity, float dt, const CapsuleQuery& blocked) {
        s.validate();
        if (!finite(state.eye) || !finite(up) || !finite(velocity) || !std::isfinite(state.vertical_velocity) || !std::isfinite(dt) || dt < 0)
            throw std::invalid_argument("Invalid walk integration input");
        if (glm::length(up) < 1e-6f)
            throw std::invalid_argument("Walk up axis must be nonzero");
        if (dt == 0)
            return;
        up = glm::normalize(up);
        dt = std::min(dt, 0.1f);
        velocity -= up * glm::dot(velocity, up);
        auto feet = state.eye - up * s.height;
        const float spacing = std::min(0.02f * s.units_per_meter, s.radius / 3);
        const float tolerance = std::min(0.0001f * s.units_per_meter, s.radius * .001f);
        const int steps = std::clamp(int(std::ceil((glm::length(velocity) * dt +
                                                    std::abs(state.vertical_velocity) * dt + s.gravity * dt * dt) /
                                                   spacing)),
                                     1, 1024);
        const float tick = dt / steps;
        auto hit = [&](glm::vec3 p) { return blocked(p, up, s.height, s.radius); };
        auto sweep = [&](glm::vec3 p, glm::vec3 delta) { return sweepWalk(p, delta, spacing, tolerance, hit); };
        auto tryStep = [&](glm::vec3 displacement) {
            if (!state.grounded || s.step_height == 0)
                return false;
            // Find the first height where the lower cap clears the riser. Using
            // the lower cap here avoids rejecting a small step just because a
            // low ceiling prevents lifting the entire body by the maximum budget.
            auto footHit = [&](float lift) {
                return blocked(feet + displacement + up * lift, up, 2 * s.radius, s.radius);
            };
            if (!footHit(0))
                return false;
            const int count = std::max(1, int(std::ceil(s.step_height / spacing)));
            float low = 0;
            for (int i = 1; i <= count; ++i) {
                float high = s.step_height * i / count;
                if (footHit(high)) {
                    low = high;
                    continue;
                }
                for (int j = 0; j < 16 && high - low > tolerance; ++j) {
                    float middle = (low + high) * .5f;
                    if (footHit(middle))
                        low = middle;
                    else
                        high = middle;
                }
                const auto lift = sweep(feet, up * high);
                if (lift.hit)
                    return false;
                const auto travel = sweep(lift.position, displacement);
                if (travel.hit)
                    return false;
                const auto support = sweep(travel.position, -up * (high + 2 * tolerance));
                if (!support.hit)
                    return false;
                feet = support.position;
                state.vertical_velocity = 0;
                return true;
            }
            return false;
        };
        for (int step = 0; step < steps; ++step) {
            const bool was_grounded = state.grounded;
            const auto displacement = velocity * tick;
            const auto destination = feet + displacement;
            const auto travel = sweep(feet, displacement);
            if (!travel.hit) {
                feet = travel.position;
            } else if (!tryStep(displacement)) {
                feet = travel.position;
                const auto remaining = destination - feet;
                // Slide the remaining projected axis components independently.
                // Every component is swept, including partial travel to a wall.
                for (int a = 0; a < 3; ++a) {
                    glm::vec3 d(0);
                    d[a] = remaining[a];
                    d -= up * glm::dot(d, up);
                    feet = sweep(feet, d).position;
                }
            }
            // Follow small downward steps only while supported. No floor within
            // the step budget means a real ledge: retain the pose and start falling.
            if (was_grounded && state.vertical_velocity <= 0) {
                const auto support = sweep(feet, -up * std::max(s.step_height, 2 * tolerance));
                if (support.hit) {
                    feet = support.position;
                    state.grounded = true;
                    state.vertical_velocity = 0;
                    continue;
                }
            }
            state.vertical_velocity -= s.gravity * tick;
            const auto vertical = sweep(feet, up * state.vertical_velocity * tick);
            feet = vertical.position;
            state.grounded = vertical.hit && state.vertical_velocity <= 0;
            if (vertical.hit)
                state.vertical_velocity = 0;
        }
        state.eye = feet + up * s.height;
    }

    void advanceWalkCamera(WalkCameraState& camera, const WalkState& body, const WalkSettings& s,
                           glm::vec3 up, float dt, const CapsuleQuery& blocked) {
        s.validate();
        if (!finite(camera.eye) || !finite(camera.body_eye) || !finite(body.eye) || !finite(up) ||
            !std::isfinite(camera.height_offset) || !std::isfinite(camera.offset_velocity) ||
            !std::isfinite(dt) || dt < 0 || glm::length(up) < 1e-6f)
            throw std::invalid_argument("Invalid walk camera input");
        up = glm::normalize(up);
        dt = std::min(dt, .1f);
        // Cancel only grounded height changes, not ballistic falling. Then decay
        // the presentation offset with an analytic critically damped spring.
        if (body.grounded)
            camera.height_offset += glm::dot(camera.body_eye - body.eye, up);
        constexpr float omega = 20;
        const float c = camera.offset_velocity + omega * camera.height_offset;
        const float decay = std::exp(-omega * dt);
        camera.height_offset = (camera.height_offset + c * dt) * decay;
        camera.offset_velocity = (camera.offset_velocity - omega * c * dt) * decay;
        const float limit = std::min(s.step_height, s.height * .25f);
        const float offset = std::clamp(camera.height_offset, -limit, limit);
        if (offset != camera.height_offset)
            camera.offset_velocity = 0;
        camera.height_offset = offset;
        const float spacing = std::min(.02f * s.units_per_meter, s.radius / 3);
        const float tolerance = std::min(.0001f * s.units_per_meter, s.radius * .001f);
        auto headHit = [&](glm::vec3 eye) {
            return blocked(eye - up * (2 * s.radius), up, 2 * s.radius, s.radius);
        };
        // The camera may lag above the physical head on a downward step. Sweep
        // its upper-cap proxy from the safe body pose and clamp at any overhang.
        const auto allowed = sweepWalk(body.eye, up * offset, spacing, tolerance, headHit);
        if (allowed.hit) {
            camera.height_offset = glm::dot(allowed.position - body.eye, up);
            camera.offset_velocity = 0;
        }
        if (std::abs(camera.height_offset) < tolerance && std::abs(camera.offset_velocity) < tolerance * omega) {
            camera.height_offset = 0;
            camera.offset_velocity = 0;
        }
        camera.body_eye = body.eye;
        camera.eye = body.eye + up * camera.height_offset;
    }
} // namespace lfs::core
