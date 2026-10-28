/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "io/collision_export.hpp"
#include "core/collision.hpp"
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
        lfs::Exception exportError(std::string_view key, const Error& error, std::string details) {
            return lfs::Exception(to_lfs_error(error, LFS_SOURCE_SITE_CURRENT())
                                      .with_context("export collision voxels", LFS_SOURCE_SITE_CURRENT(),
                                                    lfs::SmallFields{}.add("message_key", key).add("message_details", std::move(details))));
        }
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
            core::reportProgress(stop, {}, 0, "");
            lo = glm::max(lo, cursor.p);
            hi = glm::min(hi, cursor.p + cursor.size);
            if (glm::any(glm::greaterThanEqual(lo, hi)))
                return 0;
            const auto value = v.words[cursor.index];
            if (value <= 1)
                return 1 << value;
            int result = 0; // Bit 0: empty cells; bit 1: occupied cells.
            if (value & core::CollisionVolume::brick_tag) {
                const auto base = value & ~core::CollisionVolume::brick_tag;
                const uint64_t mask = v.words[base] | (uint64_t(v.words[base + 1]) << 32);
                for (int z = lo.z; z < hi.z; ++z)
                    for (int y = lo.y; y < hi.y; ++y)
                        for (int x = lo.x; x < hi.x; ++x)
                            if ((result |= 1 << ((mask >> ((x - cursor.p.x) + 4 * (y - cursor.p.y) + 16 * (z - cursor.p.z))) & 1)) == 3)
                                return result;
            } else {
                for (int c = 0; c < 8; ++c)
                    if ((result |= classify(v, cursor.child(value - 2, c), lo, hi, stop)) == 3)
                        return result;
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
                throw exportError("collision.export_write",
                                  Error{ErrorCode::WRITE_FAILURE, std::format("Cannot write voxel output: {}", std::strerror(errno ? errno : EIO)), path},
                                  core::path_to_utf8(path));
        }
    } // namespace
    void checkCollisionExportMixedBlocks(uint64_t mixed) {
        constexpr uint64_t limit = uint64_t(1) << 24;
        if (mixed > limit || (mixed && mixed + (mixed + 5) / 7 > limit))
            throw core::collisionError("collision.output_limit",
                                       "SuperSplat v1.1 cannot represent these mixed blocks in its 24-bit node index range. Increase the export cell size; native generation uses its separate record limit",
                                       lfs::ErrorCode::ResourceExhausted, std::format("{} / {}", mixed, limit));
    }
    core::CollisionSource loadCollisionSource(const std::filesystem::path& path, std::stop_token stop,
                                              ExportProgressCallback progress) {
        if (path.filename() != "meta.json" && Loader::isDatasetPath(path))
            throw core::collisionError("collision.source_unsupported", "Input is a dataset; train and export Gaussian splats first", lfs::ErrorCode::InvalidArgument, core::path_to_utf8(path));
        auto loaded = Loader::create()->load(path, {.progress = [&](float p, const std::string& stage) { core::reportProgress(stop, progress, p / 100.f, stage); },
                                                    .cancel_requested = [&] { return stop.stop_requested(); }});
        if (!loaded)
            throw lfs::Exception(to_lfs_error(loaded.error(), LFS_SOURCE_SITE_CURRENT())
                                     .with_context("load collision source", LFS_SOURCE_SITE_CURRENT(),
                                                   lfs::SmallFields{}.add("message_key", std::string_view("collision.source_load_failed")).add("message_details", core::path_to_utf8(path))));
        const auto* model = std::get_if<std::shared_ptr<core::SplatData>>(&loaded->data);
        if (!model || !*model || loaded->point_cloud_source)
            throw core::collisionError("collision.source_unsupported", "Input must contain Gaussian splats, not a mesh, point cloud or dataset", lfs::ErrorCode::InvalidArgument, core::path_to_utf8(path));
        auto source = core::captureCollisionSource(**model);
        core::reportProgress(stop, progress, 1, "Captured full-detail splats");
        return source;
    }
    void exportCollision(const core::CollisionVolume& v, const std::filesystem::path& path,
                         const glm::mat4& world, std::stop_token stop, ExportProgressCallback progress) try {
        core::reportProgress(stop, progress, 0, core::collisionStage("runtime.export_preparing_data", "Preparing voxel export"));
        if (!core::path_to_utf8(path.filename()).ends_with(".voxel.json"))
            throw core::collisionError("collision.export_suffix", "Voxel metadata output must end in .voxel.json: " + core::path_to_utf8(path),
                                       lfs::ErrorCode::InvalidArgument, core::path_to_utf8(path));
        const auto binary = std::filesystem::path(path).replace_extension(".bin");
        glm::mat4 flip(1);
        flip[0][0] = flip[1][1] = -1;
        const auto transform = flip * world;
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r)
                if (!std::isfinite(transform[c][r]))
                    throw core::collisionError("collision.export_transform", "Collision transform must be finite", lfs::ErrorCode::InvalidArgument);
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
                throw core::collisionError("collision.export_transform", "Voxel export requires uniform scale and signed axis alignment; align the source and regenerate collision", lfs::ErrorCode::Unsupported);
            for (int k = 0; k < 3; ++k)
                if (k != d && std::abs(transform[s][k]) > scale * 1e-5f)
                    throw core::collisionError("collision.export_transform", "Voxel export cannot represent rotation or shear; align the source and regenerate collision", lfs::ErrorCode::Unsupported);
            axis[s] = d;
            sign[s] = transform[s][d] > 0 ? 1 : -1;
            used |= 1u << d;
        }
        if (transform[0][3] != 0 || transform[1][3] != 0 || transform[2][3] != 0 || transform[3][3] != 1)
            throw core::collisionError("collision.export_transform", "Voxel export requires an affine grid transform", lfs::ErrorCode::Unsupported);
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
                throw core::collisionError("collision.export_transform", "Transformed grid bounds are not representable; reduce translation or scale and regenerate", lfs::ErrorCode::BoundsViolation);
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
            return state & 2 ? (state == 2 && lo == q && hi == q + size ? 1 : 2) : 0;
        };
        struct Pending {
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
            queue.push_back({{}, edge, state, source});
        }
        uint32_t interiors = 0;
        float completed = .1f;
        for (size_t i = 0; i < queue.size(); ++i) {
            const auto item = queue[i];
            completed = std::max(completed, .1f + .6f * float(i) / queue.size());
            core::reportProgress(stop, progress, completed, core::collisionStage("export.export", "Packing voxel tree"));
            if (item.state == 1) {
                nodes[i] = 0xff000000u;
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
                    throw core::collisionError("collision.output_limit", "Voxel leaf limit exceeded; increase cell size and regenerate", lfs::ErrorCode::ResourceExhausted,
                                               std::format("{} / {}", leaves.size() / 2, 1u << 24));
                nodes[i] = uint32_t(leaves.size() / 2);
                leaves.insert(leaves.end(), {uint32_t(mask), uint32_t(mask >> 32)});
            } else {
                uint32_t mask = 0;
                const auto base = uint32_t(nodes.size());
                for (int c = 0; c < 8; ++c) {
                    const auto p = item.p + I3(c & 1, (c >> 1) & 1, c >> 2) * (item.size / 2);
                    auto source = item.source;
                    const auto state = region(p, item.size / 2, source);
                    if (state) {
                        mask |= 1u << c;
                        queue.push_back({p, item.size / 2, state, source});
                        nodes.push_back(0);
                    }
                }
                if (nodes.size() > (1u << 24))
                    throw core::collisionError("collision.output_limit", "SuperSplat v1.1 export exceeds its 24-bit node index range", lfs::ErrorCode::ResourceExhausted,
                                               std::format("{} / {}", nodes.size(), 1u << 24));
                nodes[i] = (mask << 24) | base;
                ++interiors;
            }
        }
        auto xyz = [](glm::vec3 p) { return nlohmann::json::array({p.x, p.y, p.z}); };
        const nlohmann::json bounds{{"min", xyz(lo)}, {"max", xyz(hi)}};
        const nlohmann::json scene_bounds{{"min", xyz(scene_lo)}, {"max", xyz(scene_hi)}};
        const nlohmann::json metadata{{"version", "1.1"}, {"asset", {{"generator", "LichtFeld Studio"}}}, {"gridBounds", bounds}, {"sceneBounds", scene_bounds}, {"voxelResolution", resolution}, {"leafSize", 4}, {"treeDepth", depth}, {"numInteriorNodes", interiors}, {"numMixedLeaves", leaves.size() / 2}, {"nodeCount", nodes.size()}, {"leafDataCount", leaves.size()}};
        if (auto result = ensure_output_parent_directory(path); !result)
            throw exportError("collision.export_write", result.error(), core::path_to_utf8(path));
        ScopedAtomicOutputFile bin(binary), json(path);
        write(bin.temp_path(), [&](std::ostream& out) {
            for (const auto* values : {&nodes, &leaves})
                for (size_t i = 0; i < values->size(); ++i) {
                    if (!(i % 16384))
                        core::reportProgress(stop, progress, .8f, core::collisionStage("export.export", "Writing voxel binary"));
                    const auto w = (*values)[i];
                    const std::array<char, 4> bytes{char(w), char(w >> 8), char(w >> 16), char(w >> 24)};
                    out.write(bytes.data(), bytes.size());
                }
        });
        write(json.temp_path(), [&](std::ostream& out) { out << metadata.dump(2) << '\n'; });
        const std::array outputs{&bin, &json};
        std::array<std::filesystem::path, 2> backups;
        std::array<bool, 2> installed{};
        core::reportProgress(stop, progress, .9f, core::collisionStage("export.export", "Publishing voxel pair"));
        try {
            for (int i = 0; i < 2; ++i)
                if (const auto& target = outputs[i]->output_path(); fs::exists(target)) {
                    if (!fs::is_regular_file(target))
                        throw exportError("collision.export_publish", Error{ErrorCode::NOT_A_FILE, "Not a regular voxel output file", target},
                                          core::path_to_utf8(target));
                    const auto backup = make_atomic_temp_output_path(target);
                    fs::rename(target, backup);
                    backups[i] = backup;
                }
            for (int i = 0; i < 2; ++i) {
                if (i)
                    core::reportProgress(stop, progress, .95f, core::collisionStage("export.export", "Publishing voxel metadata"));
                if (auto result = outputs[i]->commit(); !result)
                    throw exportError("collision.export_publish", result.error(), core::path_to_utf8(outputs[i]->output_path()));
                installed[i] = true;
            }
        } catch (...) {
            std::string recovery_paths;
            for (int i = 0; i < 2; ++i) {
                const auto& target = outputs[i]->output_path();
                std::error_code ec;
                if (installed[i])
                    fs::remove(target, ec);
                if (!ec && !backups[i].empty())
                    fs::rename(backups[i], target, ec);
                if (ec)
                    recovery_paths += std::format("\n{} <- {} ({}: {})", core::path_to_utf8(target), core::path_to_utf8(backups[i]), ec.value(), ec.message());
            }
            if (!recovery_paths.empty()) {
                try {
                    throw;
                } catch (const std::exception& error) {
                    LOG_ERROR("Original voxel publication failure before rollback error: {}", error.what());
                } catch (...) {
                    LOG_ERROR("Original voxel publication failure was a non-standard exception");
                }
                throw exportError("collision.export_publish",
                                  Error{ErrorCode::WRITE_FAILURE, "Voxel publication failed; recovery files retained." + recovery_paths},
                                  recovery_paths);
            }
            throw;
        }
        for (const auto& backup : backups)
            if (!backup.empty()) {
                std::error_code ec;
                fs::remove(backup, ec);
                if (ec)
                    LOG_WARN("Voxel pair saved; cannot remove backup '{}': {}", core::path_to_utf8(backup), ec.message());
            }
    } catch (const std::filesystem::filesystem_error& error) {
        throw exportError("collision.export_publish", Error{ErrorCode::WRITE_FAILURE, error.what()},
                          std::format("{}\n{} ({})", core::path_to_utf8(error.path1()), core::path_to_utf8(error.path2()), error.code().value()));
    }
} // namespace lfs::io
