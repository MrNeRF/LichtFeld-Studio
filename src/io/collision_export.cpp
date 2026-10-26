/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "io/collision_export.hpp"
#include "core/logger.hpp"
#include "io/atomic_output.hpp"
#include "io/loader.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
namespace lfs::io {
    namespace {
        namespace fs = core::native_fs;
        using I3 = glm::ivec3;
        struct Cursor {
            uint32_t index;
            I3 p;
            int size;
            Cursor child(uint32_t base, int c) const {
                return {base + c, p + I3(c & 1, (c >> 1) & 1, c >> 2) * (size / 2), size / 2};
            }
            void descend(const core::CollisionVolume& v, I3 lo, I3 hi) {
                // Reflected logical bounds need not align with native octants.
                while (v.words[index] > 1 && !(v.words[index] & core::CollisionVolume::brick_tag)) {
                    const auto mid = p + size / 2;
                    const auto lower = glm::greaterThanEqual(lo, mid);
                    if (lower != glm::greaterThanEqual(hi - 1, mid))
                        break;
                    *this = child(v.words[index] - 2, int(lower.x) + 2 * int(lower.y) + 4 * int(lower.z));
                }
            }
        };
        // Classify only the intersecting region; stop as soon as it is mixed.
        int classify(const core::CollisionVolume& v, Cursor cursor,
                     I3 lo, I3 hi, const std::stop_token& stop) {
            if (stop.stop_requested())
                throw core::OperationCancelled();
            lo = glm::max(lo, cursor.p);
            hi = glm::min(hi, cursor.p + cursor.size);
            if (glm::any(glm::greaterThanEqual(lo, hi)))
                return -1;
            const auto value = v.words[cursor.index];
            if (value <= 1)
                return int(value);
            int result = -1;
            auto merge = [&](int child) {
                if (result < 0)
                    result = child;
                else if (child >= 0 && result != child)
                    result = 2;
                return result == 2;
            };
            if (value & core::CollisionVolume::brick_tag) {
                const auto base = value & ~core::CollisionVolume::brick_tag;
                const uint64_t mask = v.words[base] | (uint64_t(v.words[base + 1]) << 32);
                for (int z = lo.z; z < hi.z; ++z)
                    for (int y = lo.y; y < hi.y; ++y)
                        for (int x = lo.x; x < hi.x; ++x)
                            if (merge((mask >> ((x - cursor.p.x) + 4 * (y - cursor.p.y) + 16 * (z - cursor.p.z))) & 1))
                                return 2;
            } else {
                for (int c = 0; c < 8; ++c)
                    if (merge(classify(v, cursor.child(value - 2, c), lo, hi, stop)))
                        return 2;
            }
            return result;
        }
        void write(const std::filesystem::path& path, const std::function<void(std::ostream&)>& body) {
            errno = 0;
            std::ofstream out(lfs::core::native_fs::native_path(path), std::ios::binary | std::ios::trunc);
            if (out)
                body(out);
            out.close();
            if (!out)
                throw std::runtime_error(std::format("Cannot write '{}': {}", core::path_to_utf8(path),
                                                     std::strerror(errno ? errno : EIO)));
        }
    } // namespace
    core::CollisionSource loadCollisionSource(const std::filesystem::path& path, std::stop_token stop,
                                              ExportProgressCallback progress) {
        if (path.filename() != "meta.json" && Loader::isDatasetPath(path))
            throw std::invalid_argument("Input is a dataset; train and export Gaussian splats first");
        auto loaded = Loader::create()->load(path, {.progress = [&](float p, const std::string& stage) { core::reportProgress(stop, progress, p / 100.f, stage); },
                                                    .cancel_requested = [&] { return stop.stop_requested(); }});
        if (!loaded)
            throw std::runtime_error(loaded.error().format());
        const auto* model = std::get_if<std::shared_ptr<core::SplatData>>(&loaded->data);
        if (!model || !*model || loaded->point_cloud_source)
            throw std::invalid_argument("Input must contain Gaussian splats, not a mesh, point cloud or dataset");
        // captureCollisionSource extracts original LOD leaves and rejects partial
        // out-of-core residency rather than substituting coarse resident proxies.
        auto source = core::captureCollisionSource(**model);
        core::reportProgress(stop, progress, 1, "Captured full-detail splats");
        return source;
    }
    void exportCollision(const core::CollisionVolume& v, const std::filesystem::path& path,
                         const glm::mat4& world, std::stop_token stop, ExportProgressCallback progress) {
        core::reportProgress(stop, progress, 0, "Preparing voxel export");
        if (!core::path_to_utf8(path.filename()).ends_with(".voxel.json"))
            throw std::invalid_argument("Voxel metadata output must end in .voxel.json");
        auto binary = path;
        binary.replace_extension(".bin");
        glm::mat4 flip(1);
        flip[0][0] = flip[1][1] = -1;
        const auto transform = flip * world;
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r)
                if (!std::isfinite(transform[c][r]))
                    throw std::invalid_argument("Collision transform must be finite");
        const float scale = glm::length(glm::vec3(transform[0]));
        std::array<int, 3> axis{}, sign{};
        unsigned used = 0;
        for (int s = 0; s < 3; ++s) {
            int d = 0;
            for (int k = 1; k < 3; ++k)
                if (std::abs(transform[s][k]) > std::abs(transform[s][d]))
                    d = k;
            if (!(scale > 0) || !std::isfinite(scale) || (used & (1u << d)) ||
                std::abs(std::abs(transform[s][d]) - scale) > scale * 1e-5f)
                throw std::invalid_argument("Voxel export requires uniform scale and signed axis alignment; align the source and regenerate collision");
            for (int k = 0; k < 3; ++k)
                if (k != d && std::abs(transform[s][k]) > scale * 1e-5f)
                    throw std::invalid_argument("Voxel export cannot represent rotation or shear; align the source and regenerate collision");
            axis[s] = d;
            sign[s] = transform[s][d] > 0 ? 1 : -1;
            used |= 1u << d;
        }
        if (transform[0][3] != 0 || transform[1][3] != 0 || transform[2][3] != 0 || transform[3][3] != 1)
            throw std::invalid_argument("Voxel export requires an affine grid transform");
        const auto padded = ((v.dimensions + 3) / 4) * 4;
        I3 dims{0}, logical_dims{0};
        glm::vec3 lo(transform * glm::vec4(v.origin, 1)), scene_lo = lo;
        for (int s = 0; s < 3; ++s) {
            dims[axis[s]] = padded[s];
            logical_dims[axis[s]] = v.dimensions[s];
            if (sign[s] < 0) {
                lo[axis[s]] -= scale * v.settings.voxel_size * padded[s];
                scene_lo[axis[s]] -= scale * v.settings.voxel_size * v.dimensions[s];
            }
        }
        const float resolution = scale * v.settings.voxel_size;
        const auto hi = lo + glm::vec3(dims) * resolution;
        const auto scene_hi = scene_lo + glm::vec3(logical_dims) * resolution;
        for (int k = 0; k < 3; ++k)
            if (!std::isfinite(lo[k]) || !std::isfinite(hi[k]) || !(hi[k] > lo[k]) ||
                !std::isfinite(scene_lo[k]) || !std::isfinite(scene_hi[k]) || !(scene_hi[k] > scene_lo[k]) || !(resolution > 0))
                throw std::invalid_argument("Transformed grid bounds are not representable; reduce translation or scale and regenerate");
        auto source_min = [&](I3 p, int size) {
            I3 q;
            for (int s = 0; s < 3; ++s)
                q[s] = sign[s] > 0 ? p[axis[s]] : padded[s] - p[axis[s]] - size;
            return q;
        };
        int edge = 8, depth = 1;
        while (edge < std::max({dims.x, dims.y, dims.z}))
            edge *= 2, ++depth;
        auto region = [&](I3 p, int size, Cursor& cursor) {
            const auto q = source_min(p, size);
            const auto lo = glm::max(q, I3(0)), hi = glm::min(q + size, v.dimensions);
            if (glm::any(glm::greaterThanEqual(lo, hi)))
                return 0;
            cursor.descend(v, lo, hi);
            const auto state = classify(v, cursor, lo, hi, stop);
            return state == 1 && (lo != q || hi != q + size) ? 2 : state;
        };
        struct Pending {
            uint32_t index;
            I3 p;
            int size;
            int state;
            Cursor source;
        };
        std::vector<uint32_t> nodes(v.occupied_count ? 1 : 0), leaves;
        std::vector<Pending> queue;
        if (v.occupied_count) {
            Cursor source{0, {}, v.edge};
            const auto state = region({}, edge, source);
            queue.push_back({0, {}, edge, state, source});
        }
        uint32_t interiors = 0;
        float completed = .1f;
        for (size_t i = 0; i < queue.size(); ++i) {
            const auto item = queue[i];
            completed = std::max(completed, .1f + .6f * float(i) / queue.size());
            core::reportProgress(stop, progress, completed, "Packing voxel tree");
            if (item.state == 1) {
                nodes[item.index] = 0xff000000u;
            } else if (item.size == 4) {
                uint64_t mask = 0;
                // Block-aligned reflection leaves exactly one native brick or a uniform subtree.
                const auto value = v.words[item.source.index];
                const auto base = value & ~core::CollisionVolume::brick_tag;
                const uint64_t source_mask = value <= 1 ? (value ? UINT64_MAX : 0)
                                                        : v.words[base] | (uint64_t(v.words[base + 1]) << 32);
                for (int b = 0; b < 64; ++b) {
                    const auto p = source_min(item.p + I3(b & 3, (b >> 2) & 3, b >> 4), 1);
                    if (glm::any(glm::lessThan(p, I3(0))) || glm::any(glm::greaterThanEqual(p, v.dimensions)))
                        continue;
                    if ((source_mask >> ((p.x & 3) + 4 * (p.y & 3) + 16 * (p.z & 3))) & 1)
                        mask |= uint64_t(1) << b;
                }
                if (leaves.size() / 2 >= (1u << 24))
                    throw std::runtime_error("Voxel leaf limit exceeded; increase cell size and regenerate");
                nodes[item.index] = uint32_t(leaves.size() / 2);
                leaves.push_back(uint32_t(mask));
                leaves.push_back(uint32_t(mask >> 32));
            } else {
                uint32_t mask = 0;
                const auto base = uint32_t(nodes.size());
                for (int c = 0; c < 8; ++c) {
                    const auto p = item.p + I3(c & 1, (c >> 1) & 1, c >> 2) * (item.size / 2);
                    auto source = item.source;
                    const auto state = region(p, item.size / 2, source);
                    if (state) {
                        mask |= 1u << c;
                        queue.push_back({uint32_t(nodes.size()), p, item.size / 2, state, source});
                        nodes.push_back(0);
                    }
                }
                if (nodes.size() > (1u << 24))
                    throw std::runtime_error("SuperSplat v1.1 export exceeds its 24-bit node index range");
                nodes[item.index] = (mask << 24) | base;
                ++interiors;
            }
        }
        auto xyz = [](glm::vec3 p) { return nlohmann::json::array({p.x, p.y, p.z}); };
        const nlohmann::json bounds{{"min", xyz(lo)}, {"max", xyz(hi)}};
        const nlohmann::json scene_bounds{{"min", xyz(scene_lo)}, {"max", xyz(scene_hi)}};
        const nlohmann::json metadata{{"version", "1.1"}, {"asset", {{"generator", "LichtFeld Studio"}}}, {"gridBounds", bounds}, {"sceneBounds", scene_bounds}, {"voxelResolution", resolution}, {"leafSize", 4}, {"treeDepth", depth}, {"numInteriorNodes", interiors}, {"numMixedLeaves", leaves.size() / 2}, {"nodeCount", nodes.size()}, {"leafDataCount", leaves.size()}};
        if (auto result = ensure_output_parent_directory(path); !result)
            throw std::runtime_error(result.error().format());
        ScopedAtomicOutputFile bin(binary), json(path);
        write(bin.temp_path(), [&](std::ostream& out) {
            for (const auto* values : {&nodes, &leaves})
                for (size_t i = 0; i < values->size(); ++i) {
                    if (!(i % 16384))
                        core::reportProgress(stop, progress, .8f, "Writing voxel binary");
                    const auto w = (*values)[i];
                    const std::array<char, 4> bytes{char(w), char(w >> 8), char(w >> 16), char(w >> 24)};
                    out.write(bytes.data(), bytes.size());
                }
        });
        write(json.temp_path(), [&](std::ostream& out) { out << metadata.dump(2) << '\n'; });
        std::array targets{binary, path};
        std::array<std::filesystem::path, 2> backups;
        std::array<bool, 2> installed{};
        core::reportProgress(stop, progress, .9f, "Publishing voxel pair");
        try {
            for (int i = 0; i < 2; ++i)
                if (fs::exists(targets[i])) {
                    if (!fs::is_regular_file(targets[i]))
                        throw std::runtime_error("Not a regular output file: " + core::path_to_utf8(targets[i]));
                    const auto backup = make_atomic_temp_output_path(targets[i]);
                    fs::rename(targets[i], backup);
                    backups[i] = backup;
                }
            if (auto result = bin.commit(); !result)
                throw std::runtime_error(result.error().format());
            installed[0] = true;
            core::reportProgress(stop, progress, .95f, "Publishing voxel metadata");
            if (auto result = json.commit(); !result)
                throw std::runtime_error(result.error().format());
            installed[1] = true;
        } catch (...) {
            const auto failure = std::current_exception();
            std::string errors;
            for (int i = 0; i < 2; ++i) {
                std::error_code ec;
                if (installed[i])
                    fs::remove(targets[i], ec);
                if (!ec && !backups[i].empty())
                    fs::rename(backups[i], targets[i], ec);
                if (ec)
                    errors += std::format(" Restore '{}' from '{}': {}.", core::path_to_utf8(targets[i]),
                                          core::path_to_utf8(backups[i]), ec.message());
            }
            if (!errors.empty())
                throw std::runtime_error("Voxel publication failed; recovery files retained." + errors);
            std::rethrow_exception(failure);
        }
        for (const auto& backup : backups)
            if (!backup.empty()) {
                std::error_code ec;
                fs::remove(backup, ec);
                if (ec)
                    LOG_WARN("Voxel pair saved; cannot remove backup '{}': {}", core::path_to_utf8(backup), ec.message());
            }
    }
} // namespace lfs::io
