/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/environment.hpp"
#include "core/splat_data.hpp"
#include "io/formats/rad.hpp"
#include "io/ply_to_rad_lod.hpp"

#include <gtest/gtest.h>

#include <set>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace {

    using lfs::core::SplatData;

    struct SyntheticSplat {
        float x, y, z;
        float nx = 0.0f, ny = 0.0f, nz = 0.0f;
        float dc0, dc1, dc2;
        float opacity;
        float s0, s1, s2;
        float r0, r1, r2, r3;
    };

    std::vector<SyntheticSplat> make_synthetic_splats(const std::size_t count) {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> cluster_dist(-100.0f, 100.0f);
        std::normal_distribution<float> offset_dist(0.0f, 2.0f);
        std::uniform_real_distribution<float> color_dist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> logit_dist(-3.0f, 3.0f);
        std::uniform_real_distribution<float> log_scale_dist(-5.0f, -2.0f);
        std::normal_distribution<float> quat_dist(0.0f, 1.0f);

        constexpr std::size_t kClusters = 64;
        std::vector<std::array<float, 3>> centers(kClusters);
        for (auto& c : centers) {
            c = {cluster_dist(rng), cluster_dist(rng), cluster_dist(rng) * 0.1f};
        }

        std::vector<SyntheticSplat> splats(count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto& c = centers[i % kClusters];
            auto& s = splats[i];
            s.x = c[0] + offset_dist(rng);
            s.y = c[1] + offset_dist(rng);
            s.z = c[2] + offset_dist(rng);
            s.dc0 = color_dist(rng);
            s.dc1 = color_dist(rng);
            s.dc2 = color_dist(rng);
            s.opacity = logit_dist(rng);
            s.s0 = log_scale_dist(rng);
            s.s1 = log_scale_dist(rng);
            s.s2 = log_scale_dist(rng);
            float q[4] = {quat_dist(rng), quat_dist(rng), quat_dist(rng), quat_dist(rng)};
            const float norm = std::max(
                std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]), 1e-6f);
            s.r0 = q[0] / norm;
            s.r1 = q[1] / norm;
            s.r2 = q[2] / norm;
            s.r3 = q[3] / norm;
        }
        return splats;
    }

    void write_synthetic_ply(const std::filesystem::path& path,
                             const std::vector<SyntheticSplat>& splats) {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.good());
        out << "ply\nformat binary_little_endian 1.0\n"
            << "element vertex " << splats.size() << "\n";
        for (const char* name : {"x", "y", "z", "nx", "ny", "nz",
                                 "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                                 "scale_0", "scale_1", "scale_2",
                                 "rot_0", "rot_1", "rot_2", "rot_3"}) {
            out << "property float " << name << "\n";
        }
        out << "end_header\n";
        static_assert(sizeof(SyntheticSplat) == 17 * sizeof(float));
        out.write(reinterpret_cast<const char*>(splats.data()),
                  static_cast<std::streamsize>(splats.size() * sizeof(SyntheticSplat)));
        ASSERT_TRUE(out.good());
    }

    struct TreeCheckResult {
        std::size_t leaf_count = 0;
        std::vector<std::array<float, 3>> leaf_positions;
    };

    void check_tree_invariants(const SplatData& data, TreeCheckResult& result) {
        const auto& tree = *data.lod_tree;
        const std::size_t n = tree.total_nodes();
        EXPECT_EQ(n, static_cast<std::size_t>(data.size()));

        const auto means_cpu = data.means_raw().cpu().contiguous();
        const float* means = means_cpu.ptr<float>();

        std::vector<std::uint32_t> parent_count(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint32_t cc = tree.child_count[i];
            const std::uint32_t cs = tree.child_start[i];
            if (cc == 0) {
                ++result.leaf_count;
                result.leaf_positions.push_back({means[i * 3], means[i * 3 + 1], means[i * 3 + 2]});
                continue;
            }
            ASSERT_GT(cs, i) << "children must follow their parent (node " << i << ")";
            ASSERT_LE(static_cast<std::size_t>(cs) + cc, n)
                << "child range out of bounds (node " << i << ")";
            for (std::uint32_t c = 0; c < cc; ++c) {
                ++parent_count[cs + c];
            }
        }
        EXPECT_EQ(parent_count[0], 0u) << "root must have no parent";
        for (std::size_t i = 1; i < n; ++i) {
            ASSERT_EQ(parent_count[i], 1u) << "node " << i << " must have exactly one parent";
        }

        // BFS level order: a coarse LOD cut must be a prefix of the file so the
        // paged viewer's chunk working set stays small.
        std::vector<std::uint8_t> level(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint32_t cc = tree.child_count[i];
            const std::uint32_t cs = tree.child_start[i];
            for (std::uint32_t c = 0; c < cc; ++c) {
                level[cs + c] = static_cast<std::uint8_t>(level[i] + 1);
            }
        }
        for (std::size_t i = 1; i < n; ++i) {
            ASSERT_GE(level[i], level[i - 1]) << "layout is not level-ordered at node " << i;
        }
    }

    void run_roundtrip(const std::size_t splat_count, const std::size_t target_bucket,
                       const lfs::io::LodBuilder builder = lfs::io::LodBuilder::kBhatt) {
        const auto temp_dir = std::filesystem::temp_directory_path() / "ply_to_rad_lod_test";
        std::filesystem::create_directories(temp_dir);
        const auto ply_path = temp_dir / "synthetic.ply";
        const auto rad_path = temp_dir / "synthetic.rad";

        const auto splats = make_synthetic_splats(splat_count);
        write_synthetic_ply(ply_path, splats);

        lfs::io::PlyToRadLodOptions options;
        options.target_bucket_splats = target_bucket;
        options.temp_dir = temp_dir / "scratch";
        options.builder = builder;
        const auto convert_result = lfs::io::convert_ply_to_rad_lod(ply_path, rad_path, options);
        ASSERT_TRUE(convert_result.has_value())
            << convert_result.error().message;

        auto loaded = lfs::io::load_rad(rad_path);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        ASSERT_TRUE(loaded->lod_tree && loaded->lod_tree->has_tree());
        EXPECT_TRUE(loaded->lod_tree->lod_opacity_encoded);
        EXPECT_GE(static_cast<std::size_t>(loaded->size()), splat_count);

        TreeCheckResult check;
        check_tree_invariants(*loaded, check);
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
        EXPECT_EQ(check.leaf_count, splat_count);

        // Every input splat must survive as a leaf with its exact position
        // (centers are stored as lossless f32).
        auto expected_positions = [&] {
            std::vector<std::array<float, 3>> positions(splat_count);
            for (std::size_t i = 0; i < splat_count; ++i) {
                positions[i] = {splats[i].x, splats[i].y, splats[i].z};
            }
            return positions;
        }();
        auto actual_positions = check.leaf_positions;
        std::sort(expected_positions.begin(), expected_positions.end());
        std::sort(actual_positions.begin(), actual_positions.end());
        EXPECT_EQ(expected_positions, actual_positions);

        std::filesystem::remove_all(temp_dir);
    }

} // namespace

TEST(PlyToRadLod, MultiBucketRoundtrip) {
    run_roundtrip(300'000, 65'536);
}

TEST(PlyToRadLod, SingleBucketRoundtrip) {
    run_roundtrip(50'000, 4'000'000);
}

TEST(PlyToRadLod, MultiBucketRoundtripOctree) {
    run_roundtrip(300'000, 65'536, lfs::io::LodBuilder::kOctree);
}

TEST(PlyToRadLod, SingleBucketRoundtripOctree) {
    run_roundtrip(50'000, 4'000'000, lfs::io::LodBuilder::kOctree);
}

TEST(PlyToRadLod, TileGridDoublesLeaves) {
    const auto temp_dir = std::filesystem::temp_directory_path() / "ply_to_rad_lod_tiles";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);
    const auto ply_path = temp_dir / "synthetic.ply";
    const auto rad_path = temp_dir / "synthetic.rad";

    constexpr std::size_t kSplats = 20'000;
    const auto splats = make_synthetic_splats(kSplats);
    write_synthetic_ply(ply_path, splats);

    lfs::io::PlyToRadLodOptions options;
    options.target_bucket_splats = 65'536;
    options.temp_dir = temp_dir / "scratch";
    options.tiles_x = 2;
    options.tiles_y = 1;
    const auto convert_result = lfs::io::convert_ply_to_rad_lod(ply_path, rad_path, options);
    ASSERT_TRUE(convert_result.has_value()) << convert_result.error().message;

    auto loaded = lfs::io::load_rad(rad_path);
    ASSERT_TRUE(loaded.has_value()) << loaded.error();
    ASSERT_TRUE(loaded->lod_tree && loaded->lod_tree->has_tree());

    TreeCheckResult check;
    check_tree_invariants(*loaded, check);
    if (::testing::Test::HasFatalFailure()) {
        return;
    }
    EXPECT_EQ(check.leaf_count, 2 * kSplats);

    // Tile 1 replicates every source splat at +X by the exact scene extent
    // plus the 1% margin, computed the same way the converter does.
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    for (const auto& s : splats) {
        min_x = std::min(min_x, s.x);
        max_x = std::max(max_x, s.x);
    }
    const float step_x = (max_x - min_x) * 1.01f;
    std::vector<std::array<float, 3>> expected_positions;
    expected_positions.reserve(2 * kSplats);
    for (const auto& s : splats) {
        expected_positions.push_back({s.x, s.y, s.z});
        expected_positions.push_back({s.x + step_x, s.y, s.z});
    }
    auto actual_positions = check.leaf_positions;
    std::sort(expected_positions.begin(), expected_positions.end());
    std::sort(actual_positions.begin(), actual_positions.end());
    EXPECT_EQ(expected_positions, actual_positions);

    std::filesystem::remove_all(temp_dir);
}

TEST(PlyToRadLod, SweepsStaleTempDirsFromCrashedRuns) {
    const auto temp_dir = std::filesystem::temp_directory_path() / "ply_to_rad_lod_stale";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);
    const auto ply_path = temp_dir / "synthetic.ply";
    const auto rad_path = temp_dir / "synthetic.rad";

    write_synthetic_ply(ply_path, make_synthetic_splats(20'000));

    // Leftovers from crashed runs: a legacy dir without an owner PID and one
    // whose owner is dead (PID above the Linux pid_max default, and Windows
    // PIDs of that magnitude do not occur in practice).
    const auto stale_legacy = temp_dir / ".synthetic.lodtmp";
    const auto stale_dead = temp_dir / ".synthetic.lodtmp.4000000000";
    std::filesystem::create_directories(stale_legacy);
    std::filesystem::create_directories(stale_dead);
    std::ofstream(stale_legacy / "bucket_00000.records").put('x');
    std::ofstream(stale_dead / "bucket_00000.nodes").put('x');
#ifndef _WIN32
    const auto live_dir =
        temp_dir / (".synthetic.lodtmp." + std::to_string(static_cast<std::uint64_t>(getppid())));
    std::filesystem::create_directories(live_dir);
#endif

    lfs::io::PlyToRadLodOptions options;
    options.target_bucket_splats = 65'536;
    const auto result = lfs::io::convert_ply_to_rad_lod(ply_path, rad_path, options);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_FALSE(std::filesystem::exists(stale_legacy));
    EXPECT_FALSE(std::filesystem::exists(stale_dead));

    // After a clean finish only the live owner's scratch may remain; the
    // conversion's own PID-suffixed dir must be gone.
    std::size_t scratch_dirs = 0;
    for (const auto& entry : std::filesystem::directory_iterator(temp_dir)) {
        if (entry.is_directory() &&
            entry.path().filename().string().starts_with(".synthetic.lodtmp")) {
            ++scratch_dirs;
        }
    }
#ifndef _WIN32
    EXPECT_TRUE(std::filesystem::exists(live_dir));
    EXPECT_EQ(scratch_dirs, 1u);
#else
    EXPECT_EQ(scratch_dirs, 0u);
#endif

    std::filesystem::remove_all(temp_dir);
}

TEST(PlyToRadLod, OutOfCoreLoadKeepsTreeAndStreamsChunks) {
    const auto temp_dir = std::filesystem::temp_directory_path() / "ply_to_rad_lod_ooc";
    std::filesystem::create_directories(temp_dir);
    const auto ply_path = temp_dir / "synthetic.ply";
    const auto rad_path = temp_dir / "synthetic.rad";

    constexpr std::size_t kSplats = 400'000;
    write_synthetic_ply(ply_path, make_synthetic_splats(kSplats));

    lfs::io::PlyToRadLodOptions options;
    options.target_bucket_splats = 65'536;
    options.temp_dir = temp_dir / "scratch";
    ASSERT_TRUE(lfs::io::convert_ply_to_rad_lod(ply_path, rad_path, options).has_value());

    auto full = lfs::io::load_rad(rad_path);
    ASSERT_TRUE(full.has_value()) << full.error();
    const auto total_nodes = full->lod_tree->total_nodes();

    auto partial = lfs::io::load_rad(
        rad_path, {.out_of_core = true, .preview_splats = 131072});
    ASSERT_TRUE(partial.has_value()) << partial.error();

    // Tree metadata covers all nodes; payload tensors hold the coarse prefix.
    // Out-of-core loads back the metadata with the mmap'd sidecar instead of
    // in-RAM vectors; the accessors must agree bit-exactly with the full load.
    ASSERT_TRUE(partial->lod_tree && partial->lod_tree->has_tree());
    EXPECT_EQ(partial->lod_tree->total_nodes(), total_nodes);
    EXPECT_EQ(static_cast<std::size_t>(partial->size()), 131072u);
    EXPECT_LT(static_cast<std::size_t>(partial->size()), total_nodes);
    EXPECT_TRUE(lfs::io::rad_paged_load_recommended(*partial));
    EXPECT_TRUE(partial->lod_tree->meta_view.valid()) << "sidecar must back OOC tree metadata";
    EXPECT_FALSE(partial->lod_tree->nodes_in_memory());
    EXPECT_TRUE(std::filesystem::exists(lfs::io::rad_meta_sidecar_path(rad_path)));

    // Links are bit-exact; bounds carry the sidecar's per-chunk u16
    // quantization, so they compare within the frame's step size.
    for (std::size_t i = 0; i < total_nodes; ++i) {
        const auto& frame = partial->lod_tree->meta_view.chunkOf(i);
        const float center_tol = std::max(
            {2.0f * frame.bbox_extent[0] / 65535.0f,
             2.0f * frame.bbox_extent[1] / 65535.0f,
             2.0f * frame.bbox_extent[2] / 65535.0f,
             1e-6f});
        EXPECT_NEAR(partial->lod_tree->center_at(i).x, full->lod_tree->center_at(i).x, center_tol);
        const float size_tol =
            full->lod_tree->size_at(i) *
                std::max(2.0f * frame.log_size_range / 65535.0f, 1e-6f) +
            1e-12f;
        EXPECT_NEAR(partial->lod_tree->size_at(i), full->lod_tree->size_at(i), size_tol);
        EXPECT_EQ(partial->lod_tree->child_start_at(i), full->lod_tree->child_start_at(i));
        EXPECT_EQ(partial->lod_tree->child_count_at(i), full->lod_tree->child_count_at(i));
        EXPECT_EQ(partial->lod_tree->level_at(i), full->lod_tree->level_at(i));
        if (HasFailure()) {
            FAIL() << "tree mismatch at node " << i;
        }
    }

    // Parent links in the sidecar must agree with a reference forward scatter
    // over the full in-RAM tree.
    {
        std::vector<std::uint32_t> reference_parent(total_nodes, 0xFFFFFFFFu);
        for (std::size_t p = 0; p < total_nodes; ++p) {
            const std::uint32_t cc = full->lod_tree->child_count_at(p);
            const std::uint32_t cs = full->lod_tree->child_start_at(p);
            for (std::uint32_t c = 0; c < cc; ++c) {
                reference_parent[cs + c] = static_cast<std::uint32_t>(p);
            }
        }
        for (std::size_t i = 0; i < total_nodes; ++i) {
            ASSERT_EQ(partial->lod_tree->meta_view.links[i].parent, reference_parent[i])
                << "sidecar parent mismatch at node " << i;
        }
    }

    // The recorded chunk ranges must stream back data identical to the full
    // in-memory decode -- this is exactly what the paged renderer does.
    const auto& source = partial->lod_tree->rad_source;
    ASSERT_TRUE(source.valid());
    const auto full_means = full->means_raw().cpu().contiguous();
    const float* full_means_ptr = full_means.ptr<float>();
    const auto full_opacity = full->opacity_raw().cpu().contiguous();
    const float* full_opacity_ptr = full_opacity.ptr<float>();
    for (const auto& range : source.chunks) {
        auto chunk = lfs::io::load_rad_chunk(rad_path, range,
                                             partial->get_max_sh_degree(),
                                             partial->lod_tree->lod_opacity_encoded);
        ASSERT_TRUE(chunk.has_value()) << chunk.error();
        ASSERT_EQ(chunk->base, range.base);
        ASSERT_EQ(chunk->count, range.count);
        for (std::size_t i = 0; i < chunk->count; ++i) {
            const std::size_t node = chunk->base + i;
            ASSERT_EQ(chunk->means[i * 3 + 0], full_means_ptr[node * 3 + 0])
                << "chunk means mismatch at node " << node;
            ASSERT_EQ(chunk->opacity_raw[i], full_opacity_ptr[node])
                << "chunk opacity mismatch at node " << node;
        }
    }

    std::filesystem::remove_all(temp_dir);
}

// Gated validation for real converted files (huge inputs):
// LFS_RAD_VALIDATE_FILE=/path/to/file.rad
TEST(PlyToRadLod, ValidateExternalRad) {
    const auto env_path = lfs::core::environment::value("LFS_RAD_VALIDATE_FILE");
    if (!env_path) {
        GTEST_SKIP() << "set LFS_RAD_VALIDATE_FILE to validate a converted RAD";
    }

    auto loaded = lfs::io::load_rad(std::filesystem::path(*env_path));
    ASSERT_TRUE(loaded.has_value()) << loaded.error();
    ASSERT_TRUE(loaded->lod_tree && loaded->lod_tree->has_tree());
    const auto& tree = *loaded->lod_tree;
    const std::size_t n = tree.total_nodes();

    std::printf("nodes=%zu resident=%lld paged=%d\n",
                n, static_cast<long long>(loaded->size()),
                lfs::io::rad_paged_load_recommended(*loaded) ? 1 : 0);

    std::vector<std::uint8_t> parent_count(n, 0);
    std::size_t leaf_count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t cc = tree.child_count_at(i);
        const std::uint32_t cs = tree.child_start_at(i);
        if (cc == 0) {
            ++leaf_count;
            continue;
        }
        ASSERT_GT(cs, i);
        ASSERT_LE(static_cast<std::size_t>(cs) + cc, n);
        for (std::uint32_t c = 0; c < cc; ++c) {
            ASSERT_LT(parent_count[cs + c], 2) << "node " << cs + c << " has multiple parents";
            ++parent_count[cs + c];
        }
    }
    ASSERT_EQ(parent_count[0], 0u);
    for (std::size_t i = 1; i < n; ++i) {
        ASSERT_EQ(parent_count[i], 1u) << "orphan node " << i;
    }
    std::printf("leaves=%zu interior=%zu\n", leaf_count, n - leaf_count);

    // Stream-sample chunks through the same path the paged renderer uses.
    const auto& source = tree.rad_source;
    ASSERT_TRUE(source.valid());
    std::mt19937 rng(7);
    std::uniform_int_distribution<std::size_t> pick(0, source.chunks.size() - 1);
    for (int s = 0; s < 50; ++s) {
        const auto& range = source.chunks[pick(rng)];
        auto chunk = lfs::io::load_rad_chunk(source.path, range,
                                             loaded->get_max_sh_degree(),
                                             tree.lod_opacity_encoded);
        ASSERT_TRUE(chunk.has_value()) << chunk.error();
        ASSERT_EQ(chunk->count, range.count);
        for (std::size_t i = 0; i < chunk->count; ++i) {
            const std::size_t node = chunk->base + i;
            if (tree.meta_view.valid()) {
                // Out-of-core centers dequantize from the sidecar's u16
                // boundsQ against the chunk AABB; exactness ends there.
                const auto& frame =
                    tree.meta_view.chunks[node / lfs::core::SplatLodTree::kChunkSplats];
                const float tol = frame.bbox_extent[0] / 65535.0f + 1e-6f;
                ASSERT_NEAR(chunk->means[i * 3 + 0], tree.center_at(node).x, tol)
                    << "chunk/tree center mismatch at node " << node;
            } else {
                ASSERT_EQ(chunk->means[i * 3 + 0], tree.center_at(node).x)
                    << "chunk/tree center mismatch at node " << node;
            }
        }
    }
}

// Layout audit: simulates view-local cuts straight from the sidecar and
// reports how many distinct chunks each layout needs. Run with
// LFS_RAD_AUDIT=<file.rad> against two files to compare layouts.
TEST(PlyToRadLod, TreeletLayoutAudit) {
    const auto target = lfs::core::environment::value("LFS_RAD_AUDIT");
    if (!target) {
        GTEST_SKIP() << "set LFS_RAD_AUDIT to run";
    }
    auto view = lfs::io::open_rad_meta_sidecar(std::filesystem::path(*target));
    ASSERT_TRUE(view.has_value()) << view.error();
    const std::size_t n = view->node_count;
    const auto level_of = [&](const std::uint32_t i) {
        return (view->links[i].packed >> 16u) & 0xffu;
    };
    const auto count_of = [&](const std::uint32_t i) {
        return view->links[i].packed & 0xffffu;
    };
    constexpr std::size_t kChunk = lfs::core::SplatLodTree::kChunkSplats;

    for (const std::uint32_t seed_level : {6u, 10u, 14u}) {
        std::vector<std::uint32_t> seeds;
        for (std::uint32_t i = 0; i < n && seeds.size() < 4096; ++i) {
            if (level_of(i) == seed_level) {
                seeds.push_back(i);
            }
            if (level_of(i) > seed_level + 1u) {
                break;
            }
        }
        if (seeds.size() < 64) {
            continue;
        }
        // Spread 64 seeds across the level.
        std::vector<std::uint32_t> picked;
        for (std::size_t k = 0; k < 64; ++k) {
            picked.push_back(seeds[k * seeds.size() / 64]);
        }
        for (const std::uint32_t descend : {4u, 8u}) {
            std::vector<std::uint32_t> cut;
            std::vector<std::uint32_t> frontier;
            std::vector<std::uint32_t> next;
            for (const std::uint32_t seed : picked) {
                frontier.assign(1, seed);
                for (std::uint32_t d = 0; d < descend && !frontier.empty(); ++d) {
                    next.clear();
                    for (const std::uint32_t node : frontier) {
                        const std::uint32_t cc = count_of(node);
                        for (std::uint32_t c = 0; c < cc; ++c) {
                            next.push_back(view->links[node].child_start + c);
                        }
                    }
                    frontier.swap(next);
                }
                cut.insert(cut.end(), frontier.begin(), frontier.end());
            }
            if (cut.empty()) {
                continue;
            }
            // Sweep hypothetical page sizes: membership is just node/size, so
            // the same file predicts utilization at any granularity.
            for (const std::size_t page : {kChunk, kChunk / 4, kChunk / 8}) {
                std::set<std::uint64_t> chunks;
                for (const std::uint32_t node : cut) {
                    chunks.insert(node / page);
                }
                std::printf(
                    "audit level=%u+%u page=%zu: cut=%zu chunks=%zu nodes/chunk=%.0f util=%.1f%%\n",
                    seed_level, descend, page, cut.size(), chunks.size(),
                    static_cast<double>(cut.size()) / static_cast<double>(chunks.size()),
                    100.0 * static_cast<double>(cut.size()) /
                        (static_cast<double>(chunks.size()) * static_cast<double>(page)));
            }
        }
    }
}

TEST(PlyToRadLod, ProbeReportsHeader) {
    const auto temp_dir = std::filesystem::temp_directory_path() / "ply_to_rad_lod_probe";
    std::filesystem::create_directories(temp_dir);
    const auto ply_path = temp_dir / "probe.ply";
    write_synthetic_ply(ply_path, make_synthetic_splats(1000));

    const auto info = lfs::io::probe_ply_gaussians(ply_path);
    ASSERT_TRUE(info.has_value()) << info.error();
    EXPECT_EQ(info->vertex_count, 1000u);
    EXPECT_EQ(info->sh_degree, 0);
    std::filesystem::remove_all(temp_dir);
}
