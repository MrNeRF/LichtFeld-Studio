/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/splat_tile_source.hpp"
#include "visualizer/scene/splat_tile_streamer.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <numbers>

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
    // Without `splats` the primitive is a plain mesh, which tilesets skip.
    void write_glb(const fs::path& path, const std::uint64_t count, const int sh_degree, const bool splats = true) {
        Json attributes = {{"POSITION", 0}};
        if (sh_degree > 0)
            attributes[std::format("KHR_gaussian_splatting:SH_DEGREE_{}_COEF_0", sh_degree)] = 1;
        Json primitive = {{"attributes", attributes}};
        if (splats)
            primitive["extensions"] = {
                {"KHR_gaussian_splatting", {{"extensions", {{"KHR_gaussian_splatting_compression_spz_2", {{"bufferView", 0}}}}}}}};
        const Json doc = {{"asset", {{"version", "2.0"}}},
                          {"meshes", {{{"primitives", {primitive}}}}},
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

    glm::dvec3 wgs84_to_ecef(const double lon, const double lat, const double height = 0.0) {
        constexpr double a = 6378137.0;
        constexpr double e2 = 6.69437999014e-3;
        const double n = a / std::sqrt(1.0 - e2 * std::sin(lat) * std::sin(lat));
        return {(n + height) * std::cos(lat) * std::cos(lon),
                (n + height) * std::cos(lat) * std::sin(lon),
                (n * (1.0 - e2) + height) * std::sin(lat)};
    }

    glm::dmat4 enu_at(const double lon, const double lat, const double height = 0.0) {
        const glm::dvec3 east(-std::sin(lon), std::cos(lon), 0);
        const glm::dvec3 north(-std::sin(lat) * std::cos(lon), -std::sin(lat) * std::sin(lon), std::cos(lat));
        const glm::dvec3 up(std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon), std::sin(lat));
        glm::dmat4 frame(1.0);
        frame[0] = glm::dvec4(east, 0);
        frame[1] = glm::dvec4(north, 0);
        frame[2] = glm::dvec4(up, 0);
        frame[3] = glm::dvec4(wgs84_to_ecef(lon, lat, height), 1);
        return frame;
    }

    Json matrix_json(const glm::dmat4& matrix) {
        Json values = Json::array();
        for (int column = 0; column < 4; ++column)
            for (int row = 0; row < 4; ++row)
                values.push_back(matrix[column][row]);
        return values;
    }

    void expect_near(const glm::dvec3& actual, const glm::dvec3& expected, const double tolerance = 1e-4) {
        EXPECT_NEAR(actual.x, expected.x, tolerance);
        EXPECT_NEAR(actual.y, expected.y, tolerance);
        EXPECT_NEAR(actual.z, expected.z, tolerance);
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
        lfs::Result<lfs::core::SplatData> load_tile(std::uint32_t) const override {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::FailedPrecondition,
                .domain = lfs::ErrorDomain::IO,
                .detail = "not used",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
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
    ASSERT_TRUE(source) << lfs::format_for_developer(source.error());
    const auto tiles = (*source)->tiles();
    ASSERT_EQ(tiles.size(), 4u); // root, a, external link, b
    EXPECT_DOUBLE_EQ((*source)->local_to_world[3][0], 1e6);
    EXPECT_EQ(tiles[0].transform, glm::mat4(1.0f));
    EXPECT_EQ(tiles[0].first_child, 1u);
    EXPECT_EQ(tiles[0].child_count, 2u);
    // 3D Tiles 1.1: the error scales with the tile transform (x2), and the local frame
    // undoes the root transform (/2).
    EXPECT_FLOAT_EQ(tiles[0].geometric_error, 10.0f);
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

namespace {
    fs::path write_tileset(const fs::path& dir, const Json& root) {
        const auto path = dir / "tileset.json";
        write_text(path, Json{{"asset", {{"version", "1.1"}}}, {"geometricError", 20}, {"root", root}}.dump());
        return path;
    }
} // namespace

TEST(Tiles3d, GeoreferencedEnuRootMapsUpToDatasetUp) {
    TempDir dir;
    write_glb(dir.path / "a.glb", 10, 0);
    constexpr double degrees = std::numbers::pi / 180.0;
    const glm::dmat4 enu = enu_at(22.5684 * degrees, 51.2465 * degrees, 180.0);
    const Json root = {
        {"transform", matrix_json(enu)},
        {"boundingVolume", {{"box", {5, 0, 20, 1, 0, 0, 0, 1, 0, 0, 0, 1}}}},
        {"geometricError", 0},
        {"content", {{"uri", "a.glb"}}},
    };
    auto source = open_tiles3d(write_tileset(dir.path, root));
    ASSERT_TRUE(source) << lfs::format_for_developer(source.error());
    const auto& tile = (*source)->tiles().front();
    expect_near(glm::dvec3(tile.transform * glm::vec4(1, 0, 0, 0)), {1, 0, 0});
    expect_near(glm::dvec3(tile.transform * glm::vec4(0, 0, 1, 0)), {0, -1, 0});
    expect_near(tile.center, {5, -20, 0});

    const glm::dmat4 round_trip = (*source)->local_to_world * glm::dmat4(tile.transform);
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            EXPECT_NEAR(round_trip[column][row], enu[column][row], 1e-4);
}

TEST(Tiles3d, TransformlessEcefBoxUsesSmallUprightLocalFrame) {
    TempDir dir;
    write_glb(dir.path / "a.glb", 10, 0);
    const glm::dvec3 center(3694433, 1535454, 4950913);
    const Json root = {
        {"boundingVolume", {{"box", {center.x, center.y, center.z, 100, 0, 0, 0, 100, 0, 0, 0, 100}}}},
        {"geometricError", 0},
        {"content", {{"uri", "a.glb"}}},
    };
    auto source = open_tiles3d(write_tileset(dir.path, root));
    ASSERT_TRUE(source) << lfs::format_for_developer(source.error());
    const auto& tile = (*source)->tiles().front();
    EXPECT_LT(glm::length(tile.center), 1e4f);
    const glm::dvec3 normal = glm::normalize(glm::dvec3(
        center.x / (6378137.0 * 6378137.0), center.y / (6378137.0 * 6378137.0),
        center.z / (6356752.314245 * 6356752.314245)));
    expect_near(glm::dvec3(tile.transform * glm::dvec4(normal, 0)), {0, -1, 0}, 1e-5);

    const glm::dmat4 identity = (*source)->local_to_world * glm::dmat4(tile.transform);
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            EXPECT_NEAR(identity[column][row], column == row ? 1.0 : 0.0, 0.25);
}

TEST(Tiles3d, TransformlessRegionUsesUprightLocalFrame) {
    TempDir dir;
    write_glb(dir.path / "a.glb", 10, 0);
    constexpr double degrees = std::numbers::pi / 180.0;
    const double lon = 22.5684 * degrees;
    const double lat = 51.2465 * degrees;
    const Json root = {
        {"boundingVolume", {{"region", {lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001, 150, 210}}}},
        {"geometricError", 0},
        {"content", {{"uri", "a.glb"}}},
    };
    auto source = open_tiles3d(write_tileset(dir.path, root));
    ASSERT_TRUE(source) << lfs::format_for_developer(source.error());
    const auto& tile = (*source)->tiles().front();
    const glm::dvec3 normal(std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon), std::sin(lat));
    expect_near(glm::dvec3(tile.transform * glm::dvec4(normal, 0)), {0, -1, 0}, 1e-5);
    EXPECT_LT(glm::length(tile.center), 1e4f);
}

TEST(Tiles3d, ScalesGeometricErrorByChildTransform) {
    TempDir dir;
    write_glb(dir.path / "a.glb", 10, 0);
    const Json child = {{"transform", {100, 0, 0, 0, 0, 100, 0, 0, 0, 0, 100, 0, 0, 0, 0, 1}},
                        {"boundingVolume", box(0, 1)},
                        {"geometricError", 1},
                        {"content", {{"uri", "a.glb"}}}};
    auto source = open_tiles3d(
        write_tileset(dir.path, {{"boundingVolume", box(0, 200)}, {"geometricError", 1000}, {"children", {child}}}));
    ASSERT_TRUE(source) << lfs::format_for_developer(source.error());
    EXPECT_FLOAT_EQ((*source)->tiles()[1].geometric_error, 100.0f);
}

TEST(Tiles3d, LoadsAllContentsAndSkipsNonSplatContent) {
    TempDir dir;
    write_glb(dir.path / "a.glb", 10, 0);
    write_glb(dir.path / "b.glb", 5, 1);
    write_glb(dir.path / "mesh.glb", 7, 0, false);
    write_text(dir.path / "c.b3dm", "b3dm");
    const Json contents = {{{"uri", "a.glb"}}, {{"uri", "b.glb"}}, {{"uri", "mesh.glb"}}, {{"uri", "c.b3dm"}}};
    auto source = open_tiles3d(write_tileset(dir.path, {{"boundingVolume", box(0, 1)}, {"geometricError", 0}, {"contents", contents}}));
    ASSERT_TRUE(source) << lfs::format_for_developer(source.error());
    EXPECT_EQ((*source)->tiles()[0].splat_count, 15u);
    EXPECT_EQ((*source)->tiles()[0].sh_degree, 1);
    EXPECT_EQ((*source)->skipped_contents, 2u);
}

TEST(Tiles3d, RejectsTilesetWithoutSplatContent) {
    TempDir dir;
    write_glb(dir.path / "mesh.glb", 7, 0, false);
    auto source = open_tiles3d(
        write_tileset(dir.path, {{"boundingVolume", box(0, 1)}, {"geometricError", 0}, {"content", {{"uri", "mesh.glb"}}}}));
    ASSERT_FALSE(source);
    EXPECT_NE(source.error().detail().find("no Gaussian splat content"), std::string_view::npos)
        << lfs::format_for_developer(source.error());
}

TEST(Tiles3d, DecodesRelativeContentUris) {
    TempDir dir;
    write_glb(dir.path / "tile one.glb", 10, 0);
    auto source = open_tiles3d(write_tileset(
        dir.path, {{"boundingVolume", box(0, 1)}, {"geometricError", 0}, {"content", {{"uri", "tile%20one.glb?v=2#x"}}}}));
    ASSERT_TRUE(source) << lfs::format_for_developer(source.error());
    EXPECT_EQ((*source)->tiles()[0].splat_count, 10u);

    auto remote = open_tiles3d(write_tileset(
        dir.path, {{"boundingVolume", box(0, 1)}, {"geometricError", 0}, {"content", {{"uri", "https://example.com/a.glb"}}}}));
    ASSERT_FALSE(remote);
    EXPECT_NE(remote.error().detail().find("scheme"), std::string_view::npos)
        << lfs::format_for_developer(remote.error());
}

TEST(Tiles3d, RejectsTooDeepTileTree) {
    TempDir dir;
    write_glb(dir.path / "a.glb", 10, 0);
    Json node = {{"boundingVolume", box(0, 1)}, {"geometricError", 0}, {"content", {{"uri", "a.glb"}}}};
    for (int level = 0; level < 200; ++level)
        node = {{"boundingVolume", box(0, 1)}, {"geometricError", 1}, {"children", {node}}};
    auto source = open_tiles3d(write_tileset(dir.path, node));
    ASSERT_FALSE(source);
    EXPECT_NE(source.error().detail().find("deeper"), std::string_view::npos)
        << lfs::format_for_developer(source.error());
}

TEST(Tiles3d, EnclosesSkewedBoxesInOrthogonalOnes) {
    TempDir dir;
    write_glb(dir.path / "a.glb", 10, 0);
    // Non-uniform scale after a 45 degree rotation shears the child's box.
    const double c = std::sqrt(0.5);
    const Json child = {{"transform", {4 * c, 4 * c, 0, 0, -c, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
                        {"boundingVolume", {{"box", {0, 0, 0, 1, 1, 0, -1, 1, 0, 0, 0, 1}}}},
                        {"geometricError", 0},
                        {"content", {{"uri", "a.glb"}}}};
    auto source = open_tiles3d(
        write_tileset(dir.path, {{"boundingVolume", box(0, 20)}, {"geometricError", 10}, {"children", {child}}}));
    ASSERT_TRUE(source) << lfs::format_for_developer(source.error());
    const auto& axes = (*source)->tiles()[1].half_axes;
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
            EXPECT_NEAR(glm::dot(axes[i], axes[j]), 0.0f, 1e-3f);
}

TEST(Tiles3d, FlatBoxDistanceUsesItsNormal) {
    SplatTile flat;
    flat.half_axes = glm::mat3(glm::vec3(5, 0, 0), glm::vec3(0, 5, 0), glm::vec3(0));
    EXPECT_FLOAT_EQ(flat.distance({0, 0, 100}), 100.0f); // above a flat box is not inside
    EXPECT_FLOAT_EQ(flat.distance({1, 2, 0}), 0.0f);
    SplatTile point;
    EXPECT_FLOAT_EQ(point.distance({3, 4, 0}), 5.0f);
}

TEST(Tiles3d, AddRefinementShowsResidentChildren) {
    FakeSource source;
    for (auto& tile : source.tiles_)
        tile.additive = true;
    const SplatTileView view{.camera = {0, 0, 10}, .sse_per_error = 1000};
    // Leaf 4 is missing: its ADD parent and sibling still draw, and the cut is complete.
    auto selection = select_splat_tiles(source, view, [](const std::uint32_t tile) { return tile != 4; });
    EXPECT_EQ(sorted(selection.render), (std::vector<std::uint32_t>{1, 2, 3, 5, 6}));
    EXPECT_TRUE(selection.complete);
}

TEST(Tiles3d, OrthographicErrorDependsOnZoomNotDistance) {
    const FakeSource source;
    // Perspective far away keeps the coarse level, but an orthographic view zoomed in to
    // 1000 pixels per unit shows every unit of error as 1000 pixels at any distance.
    auto zoomed = select_splat_tiles(source, {.camera = {0, 0, 1000}, .sse_per_error = 1000, .orthographic = true}, kAll);
    EXPECT_EQ(sorted(zoomed.render), (std::vector<std::uint32_t>{3, 4, 5, 6}));
    // Zoomed out to 10 pixels per unit, the coarse error (1) stays below max_sse at any distance.
    auto wide = select_splat_tiles(source, {.camera = {0, 0, 1}, .sse_per_error = 10, .orthographic = true}, kAll);
    EXPECT_EQ(sorted(wide.render), (std::vector<std::uint32_t>{1, 2}));
}

TEST(Tiles3d, ReplacementBudgetReclaimsDrawnModel) {
    lfs::vis::SplatTileBudgetUsage usage{
        .cached_bytes = 100,
        .drawn_bytes = 670,
        .replacement_bytes = 200,
        .limit_bytes = 603,
        .replacing_drawn = true,
    };
    // The old 670 MB model exceeds the reduced limit, but it is released by the
    // replacement. Its 200 MB model and 300 MB incoming tile fit after the swap.
    EXPECT_TRUE(lfs::vis::splat_tile_budget_allows_load(usage, 300));

    usage.replacing_drawn = false;
    EXPECT_FALSE(lfs::vis::splat_tile_budget_allows_load(usage, 300));

    // The coarsest cut is the display floor even when the user selects a limit
    // below its steady-state footprint.
    usage.replacing_drawn = true;
    usage.replacement_bytes = 700;
    EXPECT_FALSE(lfs::vis::splat_tile_budget_allows_load(usage, 1));
    usage.minimum_cut = true;
    EXPECT_TRUE(lfs::vis::splat_tile_budget_allows_load(usage, 1));
}
