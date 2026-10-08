/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Streaming of streamed SOG (SSOG) through the tile streamer's source interface.
// Built on every backend; the SSOG format tests that need the converter stay in
// test_ssog_format.cpp.

#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/formats/sogs.hpp"
#include "io/formats/ssog.hpp"
#include "io/loader.hpp"
#include "io/splat_tile_source.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <vector>

namespace {
    namespace fs = std::filesystem;
    using Json = nlohmann::json;
    using namespace lfs::core;
    using namespace lfs::io;

    struct ScopedDirectory {
        fs::path path;
        ScopedDirectory() {
            static std::atomic_uint64_t sequence{0};
            path = fs::temp_directory_path() /
                   std::format("lichtfeld_ssog_tiles_{}_{}", std::chrono::steady_clock::now().time_since_epoch().count(),
                               sequence++);
            fs::create_directories(path);
        }
        ~ScopedDirectory() {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
    };

    Json read(const fs::path& p) {
        std::ifstream f(p);
        return Json::parse(f);
    }

    void write(const fs::path& p, const Json& j) {
        std::ofstream f(p);
        f << j.dump();
    }

    // Eight clusters 12 units apart, so the SSOG tree splits into several nodes.
    SplatData synthetic(const size_t n, const int degree = 1) {
        std::mt19937 rng(27);
        std::normal_distribution<float> normal(0, 1);
        const size_t k = (degree + 1) * (degree + 1) - 1;
        std::vector<float> means(n * 3), scales(n * 3), quats(n * 4), sh0(n * 3), shN(n * k * 3), opacity(n);
        for (size_t i = 0; i < n; ++i) {
            for (int a = 0; a < 3; ++a) {
                means[i * 3 + a] = normal(rng) * 1.5f + float((i % 8) >> a & 1) * 12;
                scales[i * 3 + a] = -2 + normal(rng) * 0.15f;
                sh0[i * 3 + a] = normal(rng) * 0.15f;
            }
            for (int a = 0; a < 4; ++a)
                quats[i * 4 + a] = normal(rng);
            opacity[i] = normal(rng) * 0.4f;
        }
        for (auto& v : shN)
            v = normal(rng) * 0.08f;
        return SplatData(degree, Tensor::from_vector(means, {n, 3}), Tensor::from_vector(sh0, {n, 1, 3}),
                         Tensor::from_vector(shN, {n, k, 3}), Tensor::from_vector(scales, {n, 3}),
                         Tensor::from_vector(quats, {n, 4}), Tensor::from_vector(opacity, {n, 1}), 1);
    }

    SsogSaveOptions options(const fs::path& path, const int levels = 1) {
        SsogSaveOptions o;
        o.output_path = path;
        o.lod_levels = levels;
        o.chunk_count_k = 16;
        o.chunk_extent = 2;
        o.chunk_min_k = 1;
        return o;
    }

    // The same splats in any order: tiles merge in tile order, a flat load in unit order.
    void expect_same_rows(const SplatData& actual, const SplatData& expected) {
        ASSERT_EQ(actual.size(), expected.size());
        const auto rows = [](const SplatData& s) {
            const auto means = s.means().cpu().contiguous();
            const auto scaling = s.scaling_raw().cpu().contiguous();
            const auto rotation = s.rotation_raw().cpu().contiguous();
            const auto opacity = s.opacity_raw().cpu().contiguous();
            const auto sh0 = s.sh0().cpu().contiguous();
            std::vector<std::vector<float>> out(s.size());
            for (size_t i = 0; i < out.size(); ++i) {
                auto& row = out[i];
                row.insert(row.end(), means.ptr<float>() + i * 3, means.ptr<float>() + i * 3 + 3);
                row.insert(row.end(), scaling.ptr<float>() + i * 3, scaling.ptr<float>() + i * 3 + 3);
                row.insert(row.end(), rotation.ptr<float>() + i * 4, rotation.ptr<float>() + i * 4 + 4);
                row.push_back(opacity.ptr<float>()[i]);
                row.insert(row.end(), sh0.ptr<float>() + i * 3, sh0.ptr<float>() + i * 3 + 3);
            }
            std::ranges::sort(out);
            return out;
        };
        EXPECT_TRUE(rows(actual) == rows(expected));
    }

    // Decodes and merges the tiles a view selects, like the streamer does.
    std::unique_ptr<SplatData> merge_selected(const SplatTileSource& source, const SplatTileSelection& selection) {
        std::map<std::uint32_t, SplatData> loaded;
        for (const auto tile : selection.render) {
            auto data = load_splat_tile_gpu(source, tile);
            EXPECT_TRUE(data) << (data ? "" : lfs::format_for_developer(data.error()));
            if (data)
                loaded.emplace(tile, std::move(*data));
        }
        return merge_splat_tiles(source, selection.render, [&](const std::uint32_t tile) {
            const auto it = loaded.find(tile);
            return it == loaded.end() ? nullptr : &it->second;
        });
    }
    // Distance at which a tile refines under the reference view (16 px, 1080 px, 60 deg).
    float refine_distance(const SplatTile& tile) {
        return tile.geometric_error * 935.3074f / 16.0f;
    }
} // namespace

TEST(SsogTiles, LevelsNestAtPlayCanvasDistances) {
    ScopedDirectory dir;
    // Level 0 exceeds one tile (64K splats), so finer levels split coarser regions.
    const auto s = synthetic(200000);
    ASSERT_TRUE(save_ssog(s, options(dir.path, 3)));
    const auto m = read(dir.path / "lod-meta.json");
    auto source = open_ssog_tiles(dir.path);
    ASSERT_TRUE(source) << lfs::format_for_developer(source.error());
    const auto tiles = (*source)->tiles();
    ASSERT_FALSE(tiles.empty());

    std::uint64_t total = 0;
    std::set<int> distances;
    for (std::uint32_t i = 0; i < tiles.size(); ++i) {
        const auto& tile = tiles[i];
        total += tile.splat_count;
        EXPECT_FALSE(tile.additive);
        if (tile.parent != SplatTile::kNone && tiles[tile.parent].splat_count > 0) {
            // A level refines into the next finer level of regions inside its own.
            const auto& parent = tiles[tile.parent];
            EXPECT_LT(tile.geometric_error, parent.geometric_error);
            for (int a = 0; a < 3; ++a) {
                EXPECT_GE(tile.center[a] - tile.half_axes[a][a], parent.center[a] - parent.half_axes[a][a] - 1e-3f);
                EXPECT_LE(tile.center[a] + tile.half_axes[a][a], parent.center[a] + parent.half_axes[a][a] + 1e-3f);
            }
        }
        if (tile.splat_count > 0)
            distances.insert(static_cast<int>(std::lround(refine_distance(tile))));
    }
    EXPECT_EQ(total, m["count"].get<std::uint64_t>());
    std::size_t finest = 0;
    for (const auto& tile : tiles)
        finest += tile.splat_count > 0 && tile.geometric_error == 0.0f;
    EXPECT_GT(finest, 1u);
    // Level 0 never refines; levels 1 and 2 switch at lodBaseDistance 5 and 5 * 3.
    EXPECT_EQ(distances, (std::set<int>{0, 5, 15}));

    const auto everything = [](std::uint32_t) { return true; };
    const auto full = select_splat_tiles(**source, {.sse_per_error = 1.0f, .max_sse = 0.0f}, everything);
    EXPECT_TRUE(full.complete);
    EXPECT_EQ(full.render_splats, m["counts"][0].get<std::uint64_t>());
    const auto merged = merge_selected(**source, full);
    ASSERT_TRUE(merged);
    auto flat = load_ssog(dir.path);
    ASSERT_TRUE(flat) << flat.error().message;
    expect_same_rows(*merged, *flat);

    const auto coarse = select_splat_tiles(**source, {.max_sse = INFINITY}, everything);
    EXPECT_TRUE(coarse.complete);
    EXPECT_LT(coarse.render_splats, full.render_splats);
    EXPECT_GE(coarse.render_splats, m["counts"][2].get<std::uint64_t>());

    // A larger LOD base distance keeps detail farther away: from far off, the default
    // shows the coarsest level, a base beyond the camera distance full detail.
    SplatTileView distant{.camera = glm::vec3(0, 0, 300), .sse_per_error = 935.3074f};
    const auto by_default = select_splat_tiles(**source, distant, everything);
    distant.lod_base_distance = 1000.0f;
    const auto scaled = select_splat_tiles(**source, distant, everything);
    EXPECT_EQ(by_default.render_splats, coarse.render_splats);
    EXPECT_EQ(scaled.render_splats, full.render_splats);
}

TEST(SsogTiles, EnvironmentAlwaysDraws) {
    ScopedDirectory dir;
    ASSERT_TRUE(save_ssog(synthetic(1000, 0), options(dir.path, 2)));
    auto m = read(dir.path / "lod-meta.json");
    SogEncodeOptions eo;
    eo.output_path = dir.path / "env";
    ASSERT_TRUE(encode_sog_directory(synthetic(100, 1), eo));
    m["environment"] = "env/meta.json";
    write(dir.path / "lod-meta.json", m);

    auto source = open_ssog_tiles(dir.path);
    ASSERT_TRUE(source) << lfs::format_for_developer(source.error());
    const auto tiles = (*source)->tiles();
    ASSERT_EQ(tiles[0].splat_count, 0u);
    ASSERT_EQ(tiles[0].child_count, 2u);
    const auto environment = tiles[0].first_child + 1;
    EXPECT_EQ(tiles[environment].splat_count, 100u);
    EXPECT_EQ(tiles[environment].sh_degree, 1);

    const auto everything = [](std::uint32_t) { return true; };
    // The camera far outside the scene and looking away still draws the environment.
    SplatTileView away{.camera = glm::vec3(1e6f), .sse_per_error = 935.3074f};
    away.planes[0] = glm::vec4(1, 0, 0, -2e6f);
    const auto selection = select_splat_tiles(**source, away, everything);
    EXPECT_TRUE(std::ranges::find(selection.render, environment) != selection.render.end());
    const auto coarse = select_splat_tiles(**source, {.max_sse = INFINITY}, everything);
    EXPECT_TRUE(std::ranges::find(coarse.render, environment) != coarse.render.end());
    auto tile = (*source)->load_tile(environment);
    ASSERT_TRUE(tile) << lfs::format_for_developer(tile.error());
    EXPECT_EQ(tile->size(), 100);
}

TEST(SsogTiles, OpensThroughTheGenericTileSourceCall) {
    ScopedDirectory dir;
    ASSERT_TRUE(save_ssog(synthetic(1000, 0), options(dir.path, 2)));
    // Restoring a streamed node (undo) reopens its file without knowing the format.
    auto source = open_splat_tile_source(dir.path / "lod-meta.json");
    ASSERT_TRUE(source) << lfs::format_for_developer(source.error());
    EXPECT_TRUE((*source)->distance_lod);
    EXPECT_FALSE((*source)->tiles().empty());
}

TEST(SsogTiles, SmallSsogLoadsFlat) {
    ScopedDirectory dir;
    ASSERT_TRUE(save_ssog(synthetic(3000, 0), options(dir.path, 2)));
    // Small enough to load flat: no tile source, every level-0 splat.
    auto loaded = Loader::create()->load(dir.path);
    ASSERT_TRUE(loaded) << loaded.error().format();
    EXPECT_FALSE(loaded->tile_source);
    auto* const splats = std::get_if<std::shared_ptr<SplatData>>(&loaded->data);
    ASSERT_TRUE(splats && *splats);
    EXPECT_EQ((*splats)->size(), 3000);
}

TEST(SsogTiles, MalformedSsogReportsAReadableError) {
    ScopedDirectory dir;
    ASSERT_TRUE(save_ssog(synthetic(1000, 0), options(dir.path, 2)));
    auto m = read(dir.path / "lod-meta.json");
    m["version"] = 2;
    write(dir.path / "lod-meta.json", m);
    // The open-file dialog shows this message: the reason, not the developer report.
    auto loaded = Loader::create()->load(dir.path);
    ASSERT_FALSE(loaded);
    EXPECT_TRUE(loaded.error().message.starts_with("Failed to load SSOG: ")) << loaded.error().message;
    EXPECT_EQ(loaded.error().message.find("frame["), std::string::npos) << loaded.error().message;
}
