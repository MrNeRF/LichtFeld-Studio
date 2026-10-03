/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/splat_tile_source.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {
    namespace fs = std::filesystem;
    using Json = nlohmann::json;
    using namespace lfs::io;

    struct TempDir {
        fs::path path = fs::temp_directory_path() / ("lfs_tiles3d_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
                                                     "_" + ::testing::UnitTest::GetInstance()->current_test_info()->name());
        TempDir() { fs::create_directories(path); }
        ~TempDir() {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
    };

    void write_text(const fs::path& path, const std::string& text) {
        std::ofstream(path, std::ios::binary) << text;
    }

    // A GLB whose JSON chunk declares `count` splats; the probe never reads the payload.
    void write_glb(const fs::path& path, const std::uint64_t count, const int sh_degree) {
        Json attributes = {{"POSITION", 0}};
        if (sh_degree > 0)
            attributes[std::format("KHR_gaussian_splatting:SH_DEGREE_{}_COEF_0", sh_degree)] = 1;
        const Json doc = {{"asset", {{"version", "2.0"}}},
                          {"meshes", {{{"primitives", {{{"attributes", attributes}}}}}}},
                          {"accessors", {{{"count", count}}, {{"count", count}}}}};
        std::string text = doc.dump();
        text.resize((text.size() + 3) & ~std::size_t{3}, ' ');
        const std::uint32_t header[5] = {0x46546C67, 2, static_cast<std::uint32_t>(20 + text.size()),
                                         static_cast<std::uint32_t>(text.size()), 0x4E4F534A};
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(header), sizeof(header));
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
    }

    Json box(const float x, const float half) {
        return {{"box", {x, 0, 0, half, 0, 0, 0, half, 0, 0, 0, half}}};
    }

    // Root without content -> two coarse tiles -> two leaves each, in a row along x.
    class FakeSource final : public SplatTileSource {
    public:
        std::vector<SplatTile> tiles_;
        FakeSource() {
            const auto make = [](const float x, const float half, const float error, const std::uint32_t parent,
                                 const std::uint32_t first, const std::uint32_t count, const std::uint64_t splats) {
                SplatTile tile;
                tile.center = {x, 0, 0};
                tile.half_axes = glm::mat3(half);
                tile.geometric_error = error;
                tile.parent = parent;
                tile.first_child = first;
                tile.child_count = count;
                tile.splat_count = splats;
                return tile;
            };
            tiles_ = {make(0, 4, 100, SplatTile::kNone, 1, 2, 0),
                      make(-2, 2, 1, 0, 3, 2, 10), make(2, 2, 1, 0, 5, 2, 10),
                      make(-3, 1, 0, 1, 0, 0, 20), make(-1, 1, 0, 1, 0, 0, 20),
                      make(1, 1, 0, 2, 0, 0, 20), make(3, 1, 0, 2, 0, 0, 20)};
        }
        std::span<const SplatTile> tiles() const override { return tiles_; }
        std::expected<lfs::core::SplatData, std::string> load_tile(std::uint32_t) const override {
            return std::unexpected("not used");
        }
    };

    std::vector<std::uint32_t> sorted(std::vector<std::uint32_t> v) {
        std::ranges::sort(v);
        return v;
    }

    const auto kAll = [](std::uint32_t) { return true; };
} // namespace

TEST(Tiles3d, ParsesTreeRelativeToRootFrame) {
    TempDir dir;
    write_glb(dir.path / "a.glb", 100, 3);
    write_glb(dir.path / "b.glb", 50, 0);
    write_text(dir.path / "external.json",
               Json{{"asset", {{"version", "1.1"}}}, {"geometricError", 1}, {"root", {{"boundingVolume", box(4, 1)}, {"geometricError", 0}, {"content", {{"uri", "b.glb"}}}}}}
                   .dump());
    // Root at ECEF-like offset with scale 2: local geometry must come out unscaled and near 0.
    const Json root = {{"transform", {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1e6, 2e6, 3e6, 1}},
                       {"boundingVolume", box(0, 4)},
                       {"geometricError", 10},
                       {"refine", "ADD"},
                       {"children",
                        {{{"boundingVolume", box(-2, 2)}, {"geometricError", 4}, {"content", {{"uri", "a.glb"}}}},
                         {{"boundingVolume", box(2, 2)}, {"geometricError", 4}, {"content", {{"uri", "external.json"}}}}}}};
    const auto path = dir.path / "tileset.json";
    write_text(path, Json{{"asset", {{"version", "1.1"}}}, {"geometricError", 20}, {"root", root}}.dump());

    ASSERT_TRUE(is_tiles3d_path(path));
    auto source = open_tiles3d(path);
    ASSERT_TRUE(source) << source.error();
    const auto tiles = (*source)->tiles();
    ASSERT_EQ(tiles.size(), 4u); // root, a, external link, b
    EXPECT_DOUBLE_EQ((*source)->local_to_world[3][0], 1e6);
    EXPECT_EQ(tiles[0].first_child, 1u);
    EXPECT_EQ(tiles[0].child_count, 2u);
    EXPECT_FLOAT_EQ(tiles[0].geometric_error, 5.0f); // error in world units / root scale
    EXPECT_NEAR(tiles[1].center.x, -2.0f, 1e-4f);
    EXPECT_EQ(tiles[1].splat_count, 100u);
    EXPECT_EQ(tiles[1].sh_degree, 3);
    EXPECT_TRUE(tiles[1].additive); // inherited
    EXPECT_EQ(tiles[2].splat_count, 0u);
    EXPECT_EQ(tiles[2].child_count, 1u);
    EXPECT_EQ(tiles[3].parent, 2u);
    EXPECT_EQ(tiles[3].splat_count, 50u);
    EXPECT_NEAR(tiles[3].center.x, 4.0f, 1e-4f);
}

TEST(Tiles3d, RejectsNonTilesetJson) {
    TempDir dir;
    const auto path = dir.path / "transforms.json";
    write_text(path, R"({"camera_angle_x": 0.7, "frames": []})");
    EXPECT_FALSE(is_tiles3d_path(path));
    // Required tileset properties are checked, not just their names.
    const auto no_volume = dir.path / "tileset.json";
    write_text(no_volume, R"({"asset": {"version": "1.1"}, "geometricError": 1, "root": {"geometricError": 1}})");
    EXPECT_FALSE(is_tiles3d_path(no_volume));
}

TEST(Tiles3d, SelectsByScreenSpaceError) {
    const FakeSource source;
    // Far away: the coarse level suffices.
    auto far = select_splat_tiles(source, {.camera = {0, 0, 1000}, .sse_per_error = 1000}, kAll);
    EXPECT_EQ(sorted(far.render), (std::vector<std::uint32_t>{1, 2}));
    EXPECT_TRUE(far.complete);
    // Close: coarse error projects above the threshold, so leaves replace it.
    auto near = select_splat_tiles(source, {.camera = {0, 0, 10}, .sse_per_error = 1000}, kAll);
    EXPECT_EQ(sorted(near.render), (std::vector<std::uint32_t>{3, 4, 5, 6}));
    EXPECT_EQ(near.render_splats, 80u);
    // Coarsest cut: infinite threshold still refines through the empty root.
    auto coarse = select_splat_tiles(source, {.max_sse = std::numeric_limits<float>::infinity()}, kAll);
    EXPECT_EQ(sorted(coarse.render), (std::vector<std::uint32_t>{1, 2}));
}

TEST(Tiles3d, KeepsParentUntilChildrenAreResident) {
    const FakeSource source;
    const SplatTileView view{.camera = {0, 0, 10}, .sse_per_error = 1000};
    // Leaf 4 is missing: tile 1 keeps drawing, tile 2 refines.
    auto selection = select_splat_tiles(source, view, [](const std::uint32_t tile) { return tile != 4; });
    EXPECT_EQ(sorted(selection.render), (std::vector<std::uint32_t>{1, 5, 6}));
    EXPECT_TRUE(selection.complete);
    EXPECT_NE(std::ranges::find(selection.wanted, 4u), selection.wanted.end());
    // Nothing resident: incomplete, and coarse tiles are requested first.
    auto empty = select_splat_tiles(source, view, [](std::uint32_t) { return false; });
    EXPECT_TRUE(empty.render.empty());
    EXPECT_FALSE(empty.complete);
    ASSERT_GE(empty.wanted.size(), 2u);
    EXPECT_LE(empty.wanted[0], 2u);
    EXPECT_LE(empty.wanted[1], 2u);
}

TEST(Tiles3d, HonorsMaxScreenSpaceErrorAndCullsOutsideFrustum) {
    const FakeSource source;
    // A higher maximum screen-space error keeps the coarse level.
    auto coarse = select_splat_tiles(source, {.camera = {0, 0, 10}, .sse_per_error = 1000, .max_sse = 200}, kAll);
    EXPECT_EQ(sorted(coarse.render), (std::vector<std::uint32_t>{1, 2}));
    // Keep x >= 2 only: the left subtree's bounding sphere lies outside.
    SplatTileView view{.camera = {0, 0, 10}, .sse_per_error = 1000};
    view.planes[0] = {1, 0, 0, -2};
    auto culled = select_splat_tiles(source, view, kAll);
    EXPECT_EQ(sorted(culled.render), (std::vector<std::uint32_t>{5, 6}));
}
