/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "io/collision_export.hpp"
#include "core/uuid.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace lfs::io {
    namespace {
        using Json = nlohmann::json;
        constexpr uint32_t max_nodes = 1u << 24;
        void check(std::stop_token stop) {
            if (stop.stop_requested())
                throw std::runtime_error("Collision export cancelled");
        }
        glm::mat4 playcanvas(const glm::mat4& world) {
            for (int c = 0; c < 4; ++c)
                for (int r = 0; r < 4; ++r)
                    if (!std::isfinite(world[c][r]))
                        throw std::runtime_error("Collision transform contains non-finite values");
            glm::mat4 flip(1);
            flip[0][0] = flip[1][1] = -1;
            return flip * world;
        }
        Json xyz(glm::vec3 p) { return Json::array({p.x, p.y, p.z}); }
        void word(std::ostream& out, uint32_t v) {
            const std::array<char, 4> bytes{char(v), char(v >> 8), char(v >> 16), char(v >> 24)};
            out.write(bytes.data(), bytes.size());
        }
        void words(std::ostream& out, std::span<const uint32_t> values, std::stop_token stop) {
            for (size_t i = 0; i < values.size(); i += 16384) {
                check(stop);
                const auto count = std::min(size_t(16384), values.size() - i);
                if constexpr (std::endian::native == std::endian::little)
                    out.write(reinterpret_cast<const char*>(values.data() + i), count * 4);
                else
                    for (size_t k = i; k < i + count; ++k)
                        word(out, values[k]);
            }
        }
        std::ofstream output(const std::filesystem::path& path) {
            std::ofstream out;
            out.exceptions(std::ios::badbit | std::ios::failbit);
            out.open(path, std::ios::binary | std::ios::trunc);
            return out;
        }
        // Writers prepare sibling temporary files, so cancellation never exposes
        // a partially written payload. The metadata is published last.
        struct Temporary {
            std::filesystem::path path;
            explicit Temporary(const std::filesystem::path& target) : path(target) { path += "." + core::generate_uuid_v4().to_string() + ".tmp"; }
            ~Temporary() {
                std::error_code ec;
                std::filesystem::remove(path, ec);
            }
        };
        void publish(std::initializer_list<std::pair<Temporary*, std::filesystem::path>> files) {
            struct Entry {
                Temporary* temp;
                std::filesystem::path target, backup;
                bool backed_up = false, installed = false;
            };
            std::vector<Entry> entries;
            for (const auto& [temp, target] : files) {
                if (std::filesystem::exists(target) && !std::filesystem::is_regular_file(target))
                    throw std::runtime_error("Collision export destination is not a regular file: " + target.string());
                auto backup = temp->path;
                backup += ".backup";
                entries.push_back({temp, target, backup});
            }
            try {
                for (auto& e : entries)
                    if (std::filesystem::exists(e.target)) {
                        std::filesystem::rename(e.target, e.backup);
                        e.backed_up = true;
                    }
                for (auto& e : entries) {
                    std::filesystem::rename(e.temp->path, e.target);
                    e.installed = true;
                }
            } catch (...) {
                const auto failure = std::current_exception();
                std::error_code ec;
                for (auto& e : entries) {
                    if (e.installed)
                        std::filesystem::remove(e.target, ec);
                    if (e.backed_up) {
                        std::filesystem::rename(e.backup, e.target, ec);
                        if (ec)
                            throw std::runtime_error("Collision export failed; previous file retained at " + e.backup.string() + ": " + ec.message());
                    }
                }
                std::rethrow_exception(failure);
            }
            for (auto& e : entries)
                if (e.backed_up) {
                    std::error_code ec;
                    std::filesystem::remove(e.backup, ec);
                }
        }
    } // namespace

    void exportCollisionVoxels(const core::CollisionVolume& v, const glm::mat4& data_world,
                               const std::filesystem::path& path, std::stop_token stop, core::CollisionProgress progress) {
        check(stop);
        v.validate();
        if (!path.filename().string().ends_with(".voxel.json"))
            throw std::runtime_error("SuperSplat collision metadata must end in .voxel.json");
        auto binary = path;
        binary.replace_extension(".bin");
        const auto transform = playcanvas(data_world);
        const float scale = glm::length(glm::vec3(transform[0]));
        std::array<int, 3> axis{}, sign{};
        unsigned used = 0;
        for (int src = 0; src < 3; ++src) {
            int dst = 0;
            for (int k = 1; k < 3; ++k)
                if (std::abs(transform[src][k]) > std::abs(transform[src][dst]))
                    dst = k;
            if (!(scale > 0) || (used & (1u << dst)) || std::abs(std::abs(transform[src][dst]) - scale) > scale * 1e-5f)
                throw std::runtime_error("SuperSplat voxels require an axis-aligned, uniformly scaled grid. Align the scene and regenerate collision, or export the GLB surface to preserve this transform.");
            for (int k = 0; k < 3; ++k)
                if (k != dst && std::abs(transform[src][k]) > scale * 1e-5f)
                    throw std::runtime_error("SuperSplat voxels cannot store a rotated grid. Align the scene and regenerate collision, or export the GLB surface.");
            axis[src] = dst;
            sign[src] = transform[src][dst] > 0 ? 1 : -1;
            used |= 1u << dst;
        }
        if (transform[0][3] != 0 || transform[1][3] != 0 || transform[2][3] != 0 || transform[3][3] != 1)
            throw std::runtime_error("Collision export requires an affine transform");
        int edge = 8, depth = 1;
        while (edge < std::max({v.dimensions.x, v.dimensions.y, v.dimensions.z})) {
            edge *= 2;
            ++depth;
        }
        const auto owned = v.tree.empty() ? v.octreeOccupancy() : std::vector<uint32_t>{};
        const auto& source = v.tree.empty() ? owned : v.tree;
        auto sample = [&](uint32_t index, int n, glm::ivec3 p) {
            while (source[index] > 1) {
                n /= 2;
                const auto b = glm::ivec3(glm::greaterThanEqual(p, glm::ivec3(n)));
                index = source[index] - 2 + b.x + 2 * b.y + 4 * b.z;
                p -= b * n;
            }
            return source[index] != 0;
        };
        auto remap = [&](glm::ivec3 p, int n) {
            glm::ivec3 q;
            for (int s = 0; s < 3; ++s)
                q[axis[s]] = sign[s] > 0 ? p[s] : n - 1 - p[s];
            return q;
        };
        std::vector<uint32_t> nodes(1), leaves;
        struct Pending {
            uint32_t source, dest;
            int edge;
            glm::ivec3 min;
        };
        std::vector<Pending> level{{0, 0, edge, {0, 0, 0}}}, next;
        uint32_t interiors = 0, mixed = 0;
        const bool small = std::max({v.dimensions.x, v.dimensions.y, v.dimensions.z}) <= 4;
        if (!v.occupiedCount())
            level.clear(), nodes.clear();
        while (!level.empty()) {
            if (progress)
                progress("Packing SuperSplat octree — " + std::to_string(nodes.size()) + " nodes");
            next.clear();
            for (const auto& p : level) {
                check(stop);
                const auto value = small ? 2u : source[p.source];
                if (value == 1) {
                    nodes[p.dest] = 0xff000000u;
                    continue;
                }
                if (p.edge == 4) {
                    uint64_t mask = 0;
                    for (int z = 0; z < 4; ++z)
                        for (int y = 0; y < 4; ++y)
                            for (int x = 0; x < 4; ++x) {
                                const glm::ivec3 cell(x, y, z);
                                if (small ? v.occupied(p.min + cell) : sample(p.source, 4, cell)) {
                                    auto q = remap(cell, 4);
                                    mask |= uint64_t(1) << (q.x + 4 * q.y + 16 * q.z);
                                }
                            }
                    if (mask == ~uint64_t(0)) {
                        nodes[p.dest] = 0xff000000u;
                        continue;
                    }
                    if (mixed >= max_nodes)
                        throw std::runtime_error("SuperSplat mixed-leaf limit exceeded; increase collision cell size and regenerate");
                    nodes[p.dest] = mixed++;
                    leaves.push_back(uint32_t(mask));
                    leaves.push_back(uint32_t(mask >> 32));
                    continue;
                }
                std::array<uint32_t, 8> children{};
                std::array<glm::ivec3, 8> minima{};
                uint32_t mask = 0;
                for (int oct = 0; oct < 8; ++oct) {
                    auto bits = glm::ivec3(oct & 1, (oct >> 1) & 1, oct >> 2);
                    auto q = remap(bits, 2);
                    const int dst = q.x + 2 * q.y + 4 * q.z;
                    const auto child = small ? 0u : value - 2 + oct;
                    auto min = p.min + bits * (p.edge / 2);
                    bool present = !small && source[child] != 0;
                    if (small)
                        for (int z = 0; z < 4; ++z)
                            for (int y = 0; y < 4; ++y)
                                for (int x = 0; x < 4; ++x)
                                    present |= v.occupied(min + glm::ivec3(x, y, z));
                    if (present) {
                        mask |= 1u << dst;
                        children[dst] = child;
                        minima[dst] = min;
                    }
                }
                const auto base = uint32_t(nodes.size());
                if (nodes.size() + std::popcount(mask) > max_nodes || base >= max_nodes)
                    throw std::runtime_error("SuperSplat 24-bit octree limit exceeded; increase collision cell size and regenerate");
                nodes[p.dest] = (mask << 24) | base;
                ++interiors;
                for (int oct = 0; oct < 8; ++oct)
                    if (mask & (1u << oct)) {
                        next.push_back({children[oct], uint32_t(nodes.size()), p.edge / 2, minima[oct]});
                        nodes.push_back(0);
                    }
            }
            level.swap(next);
        }
        glm::vec3 lo(std::numeric_limits<float>::max()), hi(-lo), scene_lo(lo), scene_hi(-lo);
        for (int oct = 0; oct < 8; ++oct) {
            const glm::vec3 bits(oct & 1, (oct >> 1) & 1, oct >> 2);
            const auto p = glm::vec3(transform * glm::vec4(v.origin + bits * float(edge) * v.settings.voxel_size, 1));
            lo = glm::min(lo, p);
            hi = glm::max(hi, p);
            const auto s = glm::vec3(transform * glm::vec4(v.origin + bits * glm::vec3(v.dimensions) * v.settings.voxel_size, 1));
            scene_lo = glm::min(scene_lo, s);
            scene_hi = glm::max(scene_hi, s);
        }
        Json meta{{"version", "1.1"}, {"asset", {{"generator", "LichtFeld Studio"}}}, {"gridBounds", {{"min", xyz(lo)}, {"max", xyz(hi)}}}, {"sceneBounds", {{"min", xyz(scene_lo)}, {"max", xyz(scene_hi)}}}, {"voxelResolution", scale * v.settings.voxel_size}, {"leafSize", 4}, {"treeDepth", depth}, {"numInteriorNodes", interiors}, {"numMixedLeaves", mixed}, {"nodeCount", nodes.size()}, {"leafDataCount", leaves.size()}};
        Temporary bin_tmp(binary), json_tmp(path);
        if (progress)
            progress("Writing SuperSplat voxel pair");
        {
            auto out = output(bin_tmp.path);
            words(out, nodes, stop);
            words(out, leaves, stop);
            out.close();
            auto header = output(json_tmp.path);
            header << meta.dump(2) << '\n';
            header.close();
        }
        check(stop);
        publish({{&bin_tmp, binary}, {&json_tmp, path}});
    }

    void exportCollisionGlb(std::span<const float> vertices, std::span<const int32_t> indices, const glm::mat4& data_world,
                            const std::filesystem::path& path, std::stop_token stop, core::CollisionProgress progress) {
        check(stop);
        if (vertices.empty() || indices.empty() || vertices.size() % 3 || indices.size() % 3)
            throw std::runtime_error("Build a collision surface before exporting GLB");
        const auto transform = playcanvas(data_world);
        const uint64_t vertices_bytes = vertices.size() * sizeof(float), indices_bytes = indices.size() * sizeof(uint32_t);
        glm::vec3 lo(std::numeric_limits<float>::max()), hi(-lo);
        for (size_t i = 0; i < vertices.size(); i += 3) {
            check(stop);
            const auto p = glm::vec3(transform * glm::vec4(vertices[i], vertices[i + 1], vertices[i + 2], 1));
            for (int k = 0; k < 3; ++k)
                if (!std::isfinite(p[k]))
                    throw std::runtime_error("Non-finite collision surface vertex");
            lo = glm::min(lo, p);
            hi = glm::max(hi, p);
        }
        for (auto i : indices)
            if (i < 0 || size_t(i) >= vertices.size() / 3)
                throw std::runtime_error("Invalid collision surface index");
        Json gltf{{"asset", {{"version", "2.0"}, {"generator", "LichtFeld Studio"}}}, {"scene", 0}, {"scenes", Json::array({{{"nodes", {0}}}})}, {"nodes", Json::array({{{"mesh", 0}, {"name", "Collision"}}})}, {"meshes", Json::array({{{"primitives", Json::array({{{"attributes", {{"POSITION", 0}}}, {"indices", 1}, {"mode", 4}}})}}})}, {"buffers", Json::array({{{"byteLength", vertices_bytes + indices_bytes}}})}, {"bufferViews", Json::array({{{"buffer", 0}, {"byteOffset", 0}, {"byteLength", vertices_bytes}, {"target", 34962}}, {{"buffer", 0}, {"byteOffset", vertices_bytes}, {"byteLength", indices_bytes}, {"target", 34963}}})}, {"accessors", Json::array({{{"bufferView", 0}, {"componentType", 5126}, {"count", vertices.size() / 3}, {"type", "VEC3"}, {"min", xyz(lo)}, {"max", xyz(hi)}}, {{"bufferView", 1}, {"componentType", 5125}, {"count", indices.size()}, {"type", "SCALAR"}}})}};
        auto json = gltf.dump();
        while (json.size() % 4)
            json += ' ';
        const uint64_t total = 28 + json.size() + vertices_bytes + indices_bytes;
        if (total > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("Collision surface exceeds GLB's 4 GiB size limit; build a coarser surface");
        Temporary temp(path);
        auto out = output(temp.path);
        word(out, 0x46546c67);
        word(out, 2);
        word(out, uint32_t(total));
        word(out, uint32_t(json.size()));
        word(out, 0x4e4f534a);
        out.write(json.data(), json.size());
        word(out, uint32_t(vertices_bytes + indices_bytes));
        word(out, 0x004e4942);
        if (progress)
            progress("Writing collision GLB surface");
        for (size_t i = 0; i < vertices.size(); i += 3) {
            if ((i & 4095) == 0)
                check(stop);
            const auto p = glm::vec3(transform * glm::vec4(vertices[i], vertices[i + 1], vertices[i + 2], 1));
            for (int k = 0; k < 3; ++k)
                word(out, std::bit_cast<uint32_t>(p[k]));
        }
        const bool flip = glm::determinant(glm::mat3(transform)) < 0;
        for (size_t i = 0; i < indices.size(); i += 3) {
            if ((i & 4095) == 0)
                check(stop);
            word(out, indices[i]);
            word(out, indices[i + (flip ? 2 : 1)]);
            word(out, indices[i + (flip ? 1 : 2)]);
        }
        out.close();
        check(stop);
        publish({{&temp, path}});
    }
} // namespace lfs::io
