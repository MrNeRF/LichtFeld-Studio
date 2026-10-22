/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

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
        CollisionVolume voxelize(std::span<const CollisionGaussian> gaussians, CollisionSettings s,
                                 std::stop_token stop) {
            glm::vec3 lo(std::numeric_limits<float>::max()), hi(-lo);
            for (const auto& g : gaussians) {
                if (!finite(g.center) || !finite(g.axes[0]) || !finite(g.axes[1]) ||
                    !finite(g.axes[2]) || !std::isfinite(g.opacity) || g.opacity < 0 || g.opacity > 1)
                    throw std::invalid_argument("Non-finite or invalid Gaussian in collision source");
                if (g.opacity <= 0)
                    continue;
                glm::vec3 extent;
                for (int a = 0; a < 3; ++a)
                    extent[a] = 3 * glm::length(glm::vec3(g.axes[0][a], g.axes[1][a], g.axes[2][a]));
                lo = glm::min(lo, g.center - extent);
                hi = glm::max(hi, g.center + extent);
            }
            if (!finite(lo) || !finite(hi) || glm::any(glm::greaterThan(lo, hi)))
                throw std::invalid_argument("No opaque Gaussians in source");
            CollisionVolume grid;
            grid.settings = s;
            grid.origin = glm::floor(lo / s.voxel_size) * s.voxel_size - glm::vec3(2 * s.voxel_size);
            auto dims = glm::ceil((hi - grid.origin) / s.voxel_size) + glm::vec3(3);
            if (!finite(dims) || glm::any(glm::greaterThan(dims, glm::vec3(CollisionVolume::max_cells))))
                throw std::runtime_error("Scene extent exceeds collision grid limit; increase voxel size");
            grid.dimensions = glm::ivec3(dims);
            grid.solid.resize(cellCount(grid.dimensions));
            std::vector<float> alpha(grid.solid.size(), 0);
            uint64_t work = 0;
            for (const auto& g : gaussians) {
                checkpoint(stop);
                if (g.opacity <= 0)
                    continue;
                if (std::abs(glm::determinant(g.axes)) < 1e-20f)
                    continue;
                auto inverse = glm::inverse(g.axes);
                glm::vec3 extent;
                for (int a = 0; a < 3; ++a)
                    extent[a] = 3 * glm::length(glm::vec3(g.axes[0][a], g.axes[1][a], g.axes[2][a]));
                auto start = glm::max(grid.cell(g.center - extent), glm::ivec3(0));
                auto end = glm::min(grid.cell(g.center + extent), grid.dimensions - 1);
                auto d = end - start + 1;
                work += uint64_t(d.x) * d.y * d.z;
                if (work > max_work)
                    throw std::runtime_error("Gaussian voxelization work limit reached; increase voxel size");
                for (int z = start.z; z <= end.z; ++z) {
                    checkpoint(stop);
                    for (int y = start.y; y <= end.y; ++y)
                        for (int x = start.x; x <= end.x; ++x) {
                            glm::ivec3 p(x, y, z);
                            const auto q = inverse * (grid.origin + (glm::vec3(p) + 0.5f) * s.voxel_size - g.center);
                            float d2 = glm::dot(q, q);
                            if (d2 > 9)
                                continue;
                            auto i = grid.index(p);
                            float contribution = g.opacity * std::exp(-0.5f * d2);
                            alpha[i] += (1 - alpha[i]) * contribution;
                        }
                }
            }
            for (size_t i = 0; i < alpha.size(); ++i)
                grid.solid[i] = alpha[i] > 0 && alpha[i] >= s.opacity;
            return grid;
        }
        void simplifyCoplanar(CollisionMesh& mesh, std::stop_token stop) {
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
                    }
                    compact.indices.push_back(remap[old]);
                }
            }
            mesh = std::move(compact);
        }
    } // namespace

    void CollisionSettings::applyPreset(CollisionPreset value) {
        preset = value;
        external_fill = value == CollisionPreset::Indoor;
        floor_fill = value == CollisionPreset::Outdoor;
        carve_height = value == CollisionPreset::Indoor ? 1.6f : 0;
    }
    void CollisionSettings::validate() const {
        positive(voxel_size, "Voxel size");
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
        return contains(p) && solid[index(p)] != 0;
    }
    glm::ivec3 CollisionVolume::cell(glm::vec3 p) const {
        const auto q = glm::floor((p - origin) / settings.voxel_size);
        if (!finite(q) || glm::any(glm::greaterThan(glm::abs(q), glm::vec3(1e9f))))
            throw std::invalid_argument("Collision query outside supported coordinate range");
        return glm::ivec3(q);
    }
    void CollisionVolume::validate() const {
        settings.validate();
        if (!finite(origin) || solid.size() != cellCount(dimensions))
            throw std::invalid_argument("Invalid collision volume payload");
        if (std::any_of(solid.begin(), solid.end(), [](auto v) { return v > 1; }))
            throw std::invalid_argument("Invalid collision occupancy value");
    }
    bool CollisionVolume::capsuleIntersects(glm::vec3 feet, glm::vec3 up, float height, float radius) const {
        // Segment/AABB distance minimized with ternary search; the squared distance is convex.
        const glm::vec3 a = feet + up * radius, b = feet + up * (height - radius);
        auto low = glm::max(cell(glm::min(a, b) - radius), glm::ivec3(0));
        auto high = glm::min(cell(glm::max(a, b) + radius), dimensions - 1);
        const float edge = settings.voxel_size;
        uint64_t tested = 0;
        for (int z = low.z; z <= high.z; ++z)
            for (int y = low.y; y <= high.y; ++y)
                for (int x = low.x; x <= high.x; ++x) {
                    if (++tested > 2'000'000)
                        throw std::runtime_error("Capsule query exceeds resource limit");
                    glm::ivec3 p(x, y, z);
                    if (!occupied(p))
                        continue;
                    auto min = origin + glm::vec3(p) * edge, max = min + edge;
                    auto distance = [&](float t) {
                        auto q = a + t * (b - a);
                        auto delta = q - glm::clamp(q, min, max);
                        return glm::dot(delta, delta);
                    };
                    float l = 0, r = 1;
                    for (int iteration = 0; iteration < 20; ++iteration) {
                        float u = (2 * l + r) / 3, v = (l + 2 * r) / 3;
                        if (distance(u) < distance(v))
                            r = v;
                        else
                            l = u;
                    }
                    if (distance((l + r) / 2) < radius * radius)
                        return true;
                }
        return false;
    }

    CollisionVolume generateCollision(std::span<const CollisionGaussian> gaussians, CollisionSettings s,
                                      std::stop_token stop, CollisionProgress progress) {
        s.validate();
        if (gaussians.empty())
            throw std::invalid_argument("No source Gaussians");
        auto report = [&](const char* text) { checkpoint(stop); if (progress) progress(text); };
        std::vector<CollisionGaussian> filtered;
        if (s.cluster) {
            report("Clustering Gaussians");
            auto coarse_settings = s;
            coarse_settings.voxel_size = s.cluster_resolution;
            coarse_settings.opacity = s.cluster_opacity;
            auto coarse = voxelize(gaussians, coarse_settings, stop);
            auto seed = glm::clamp(coarse.cell(s.seed), glm::ivec3(0), coarse.dimensions - 1);
            if (!coarse.occupied(seed)) {
                const auto requested_seed = seed;
                float best = std::numeric_limits<float>::max();
                bool found = false;
                eachCell(coarse, stop, [&](auto p) {
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
            auto inverted = coarse.solid;
            for (auto& value : inverted)
                value = !value;
            auto region = flood(coarse, inverted, std::span(&seed, 1), stop);
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
                            if (!region[coarse.index(p)])
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
        auto grid = voxelize(gaussians, s, stop);
        return processCollisionVolume(std::move(grid), stop, std::move(progress));
    }
    CollisionVolume processCollisionVolume(CollisionVolume grid, std::stop_token stop, CollisionProgress progress) {
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
                        if (grid.occupied({x, y, z})) {
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

    CollisionMesh CollisionVolume::mesh(CollisionShape shape, std::stop_token stop) const {
        validate();
        CollisionMesh result;
        // Integer half-grid coordinates weld neighboring cubes exactly.
        std::unordered_map<std::array<int32_t, 3>, int32_t, GridPointHash> vertices;
        auto vertex = [&](glm::ivec3 p) {
            const std::array<int32_t, 3> k{p.x, p.y, p.z};
            if (auto it = vertices.find(k); it != vertices.end())
                return it->second;
            if (result.vertices.size() / 3 >= 4'000'000)
                throw std::runtime_error("Collision mesh vertex limit reached; increase voxel size");
            auto id = static_cast<int32_t>(result.vertices.size() / 3);
            auto v = origin + glm::vec3(p) * (settings.voxel_size / 2);
            result.vertices.insert(result.vertices.end(), {v.x, v.y, v.z});
            vertices.emplace(k, id);
            return id;
        };
        auto triangle = [&](int a, int b, int c) {
            if (result.indices.size() >= 24'000'000)
                throw std::runtime_error("Collision mesh triangle limit reached; increase voxel size");
            result.indices.insert(result.indices.end(), {a, b, c});
        };
        if (shape == CollisionShape::Faces) {
            eachCell(*this, stop, [&](auto p) {
                if (!occupied(p))
                    return;
                for (auto normal : directions) {
                    if (occupied(p + normal))
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
            });
        } else {
            // Table-free marching cubes: face contours link crossing cube edges.
            // Four-crossing faces consistently isolate solid corners, resolving
            // ambiguous shared faces identically from either adjacent cube.
            constexpr glm::ivec3 corners[] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
            constexpr int edges[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
            constexpr int faces[6][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 9, 4, 8}, {2, 10, 6, 11}, {3, 11, 7, 8}, {1, 10, 5, 9}};
            constexpr glm::ivec3 face_normals[] = {{0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0}};
            for (int z = -1; z < dimensions.z; ++z) {
                checkpoint(stop);
                for (int y = -1; y < dimensions.y; ++y)
                    for (int x = -1; x < dimensions.x; ++x) {
                        glm::ivec3 p(x, y, z);
                        bool inside[8];
                        int count = 0;
                        for (int i = 0; i < 8; ++i) {
                            inside[i] = occupied(p + corners[i]);
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
                                if (result.vertices.size() / 3 >= 4'000'000)
                                    throw std::runtime_error("Collision mesh vertex limit reached; increase voxel size");
                                auto id = int32_t(result.vertices.size() / 3);
                                result.vertices.insert(result.vertices.end(), {center.x, center.y, center.z});
                                for (size_t i = 0; i < loop.size(); ++i)
                                    triangle(id, loop[i], loop[(i + 1) % loop.size()]);
                            }
                        }
                    }
            }
        }
        if (shape == CollisionShape::Smooth)
            simplifyCoplanar(result, stop);
        return result;
    }

    void WalkSettings::validate() const {
        positive(height, "Player height");
        positive(radius, "Player radius");
        positive(speed, "Walk speed");
        positive(step_height, "Step height", true);
        positive(gravity, "Gravity");
        if (height < 2 * radius || height > 10 || radius > 5 || speed > 50 || step_height > height)
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
                                                   std::min(0.02f, s.radius / 3))),
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
                    const int lift_steps = std::max(1, int(std::ceil(s.step_height / std::min(0.02f, s.radius / 3))));
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
