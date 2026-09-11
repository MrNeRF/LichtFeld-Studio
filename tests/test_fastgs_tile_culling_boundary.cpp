/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fastgs_tile_culling_boundary_cuda.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using fastgs_tile_culling_boundary_test::AgreementSample;
using fastgs_tile_culling_boundary_test::eval_agreement;
using fastgs_tile_culling_boundary_test::eval_ellipse_overlap;
using fastgs_tile_culling_boundary_test::eval_floor_ceil;
using fastgs_tile_culling_boundary_test::eval_splat_subtile;
using fastgs_tile_culling_boundary_test::FloorCeilSample;
using fastgs_tile_culling_boundary_test::kMaxTiles;
using fastgs_tile_culling_boundary_test::OverlapSample;
using fastgs_tile_culling_boundary_test::SplatSubtileSample;

namespace {

    constexpr float kMinAlpha = 1.0f / 255.0f;
    constexpr uint32_t kTile = 16;
    constexpr uint32_t kWidth = 64;
    constexpr uint32_t kHeight = 64;

    // Axis-aligned ellipse with contribution radius E=1 at power=2:
    // sigma = 0.5 * 4 * r^2 = 2 r^2, so sigma <= power iff r <= 1.
    constexpr float kPower = 2.0f;
    constexpr float kConic = 4.0f;

    float boundary_opacity() {
        return kMinAlpha * std::exp(kPower);
    }

    AgreementSample make_agreement(float mean_x, float mean_y, float opacity,
                                   float conic_x = kConic, float conic_y = 0.0f,
                                   float conic_z = kConic) {
        AgreementSample s{};
        s.mean_x = mean_x;
        s.mean_y = mean_y;
        s.conic_x = conic_x;
        s.conic_y = conic_y;
        s.conic_z = conic_z;
        s.opacity = opacity;
        s.width = kWidth;
        s.height = kHeight;
        return s;
    }

    std::string tiles_to_string(const uint32_t* tiles, uint32_t n) {
        std::string out = "[";
        for (uint32_t i = 0; i < n && i < static_cast<uint32_t>(kMaxTiles); i++) {
            if (i > 0) {
                out += ", ";
            }
            out += std::to_string(tiles[i]);
        }
        out += "]";
        return out;
    }

} // namespace

// Closed exclusive end: coord on a tile-size integer is a pixel center of the
// next tile, so ceil must be floor+1, not IEEE ceil (which returns the integer).
TEST(FastGSTileCullingBoundary, CeilTileClampedKeepsExactTileBoundary) {
    std::vector<FloorCeilSample> samples = {
        {0.0f, 0, 8, kTile, 0, 0},
        {15.0f, 0, 8, kTile, 0, 0},
        {16.0f, 0, 8, kTile, 0, 0},
        {16.5f, 0, 8, kTile, 0, 0},
        {32.0f, 0, 8, kTile, 0, 0},
        {32.0f - 1.0f, 0, 8, kTile, 0, 0},
        {48.0f, 0, 8, kTile, 0, 0},
    };
    ASSERT_EQ(eval_floor_ceil(samples.data(), static_cast<int>(samples.size())), cudaSuccess);

    EXPECT_EQ(samples[0].floor_tile, 0u);
    EXPECT_EQ(samples[0].ceil_tile, 1u);

    EXPECT_EQ(samples[1].floor_tile, 0u);
    EXPECT_EQ(samples[1].ceil_tile, 1u);

    EXPECT_EQ(samples[2].floor_tile, 1u);
    EXPECT_EQ(samples[2].ceil_tile, 2u) << "exact 16.0 must include tile 1";

    EXPECT_EQ(samples[3].floor_tile, 1u);
    EXPECT_EQ(samples[3].ceil_tile, 2u);

    EXPECT_EQ(samples[4].floor_tile, 2u);
    EXPECT_EQ(samples[4].ceil_tile, 3u) << "exact 32.0 must include tile 2";

    EXPECT_EQ(samples[5].floor_tile, 1u);
    EXPECT_EQ(samples[5].ceil_tile, 2u);

    EXPECT_EQ(samples[6].floor_tile, 3u);
    EXPECT_EQ(samples[6].ceil_tile, 4u);
}

// Unit ellipse x^2+y^2=1 touching the box [1,2]x[-0.5,0.5] at (1,0).
// Blend includes alpha == min_alpha, so the overlap test is closed.
TEST(FastGSTileCullingBoundary, EllipseBoxOverlapIncludesExactTouch) {
    OverlapSample touch{1.0f, 0.0f, 1.0f, 1.0f, 2.0f, -0.5f, 0.5f, 0};
    OverlapSample interior{1.0f, 0.0f, 1.0f, -0.5f, 0.5f, -0.5f, 0.5f, 0};
    OverlapSample miss{1.0f, 0.0f, 1.0f, 2.0f, 3.0f, -0.5f, 0.5f, 0};
    OverlapSample samples[] = {touch, interior, miss};
    ASSERT_EQ(eval_ellipse_overlap(samples, 3), cudaSuccess);
    EXPECT_EQ(samples[0].overlaps, 1) << "unit ellipse exactly touching a box edge";
    EXPECT_EQ(samples[1].overlaps, 1);
    EXPECT_EQ(samples[2].overlaps, 0);
}

// Mean one pixel left of a tile/sub-tile edge, radius 1: the edge pixel center
// sits on the contribution boundary. Backward warp cull must keep that subtile.
TEST(FastGSTileCullingBoundary, SubtileEllipseKeepsExactEdgePixel) {
    const float opacity = boundary_opacity();
    SplatSubtileSample edge{};
    edge.mean_x = 15.5f;
    edge.mean_y = 8.5f;
    edge.conic_x = kConic;
    edge.conic_y = 0.0f;
    edge.conic_z = kConic;
    edge.opacity = opacity;
    edge.sub_x0 = 16.0f;
    edge.sub_y0 = 8.0f;
    edge.sub_w = 8.0f;
    edge.sub_h = 4.0f;

    SplatSubtileSample interior = edge;
    interior.sub_x0 = 8.0f;
    interior.sub_y0 = 8.0f;

    SplatSubtileSample miss = edge;
    miss.sub_x0 = 32.0f;
    miss.sub_y0 = 8.0f;

    SplatSubtileSample samples[] = {edge, interior, miss};
    ASSERT_EQ(eval_splat_subtile(samples, 3), cudaSuccess);
    EXPECT_NEAR(samples[0].power_threshold, kPower, 1e-5);
    EXPECT_EQ(samples[0].overlaps, 1) << "pixel (16,8) is on the contribution boundary";
    EXPECT_EQ(samples[1].overlaps, 1);
    EXPECT_EQ(samples[2].overlaps, 0);
}

// opacity == min_alpha => power == 0, a point ellipse. Blend still shades the
// coinciding pixel center; the overlap helper must not reject it.
TEST(FastGSTileCullingBoundary, PointEllipseOnPixelCenterOverlapsSubtile) {
    SplatSubtileSample on_center{};
    on_center.mean_x = 16.5f;
    on_center.mean_y = 16.5f;
    on_center.conic_x = kConic;
    on_center.conic_y = 0.0f;
    on_center.conic_z = kConic;
    on_center.opacity = kMinAlpha;
    on_center.sub_x0 = 16.0f;
    on_center.sub_y0 = 16.0f;
    on_center.sub_w = 8.0f;
    on_center.sub_h = 4.0f;

    SplatSubtileSample off_center = on_center;
    off_center.mean_x = 16.25f;
    off_center.mean_y = 16.25f;

    SplatSubtileSample samples[] = {on_center, off_center};
    ASSERT_EQ(eval_splat_subtile(samples, 2), cudaSuccess);
    EXPECT_FLOAT_EQ(samples[0].power_threshold, 0.0f);
    EXPECT_EQ(samples[0].overlaps, 1);
    EXPECT_EQ(samples[1].overlaps, 0);
}

TEST(FastGSTileCullingBoundary, CountInstancesBlendAgreeOnInteriorSplat) {
    auto s = make_agreement(8.5f, 8.5f, boundary_opacity());
    ASSERT_EQ(eval_agreement(&s, 1), cudaSuccess);
    EXPECT_NEAR(s.power_threshold, kPower, 1e-5);
    EXPECT_GT(s.n_blend_pixels, 0u);
    EXPECT_EQ(s.count, s.n_instances);
    EXPECT_EQ(s.n_missing_blend_tiles, 0u)
        << "instances=" << tiles_to_string(s.instance_tiles, s.n_instances)
        << " blend=" << tiles_to_string(s.blend_tiles, s.n_blend_tiles);
    EXPECT_EQ(s.n_bwd_drops, 0u);
    EXPECT_EQ(s.count, s.n_blend_tiles);
}

// Mean at pixel 15 center, E=1: pixel 16 center is on the contribution
// boundary and is the first pixel of tile 1. Count, instance walk, and blend
// gold must all include tile 1; backward subtile overlap must keep it.
TEST(FastGSTileCullingBoundary, RightTileEdgePixelIsBinnedAndShaded) {
    auto s = make_agreement(15.5f, 8.5f, boundary_opacity());
    ASSERT_EQ(eval_agreement(&s, 1), cudaSuccess);
    EXPECT_NEAR(s.power_threshold, kPower, 1e-5);
    EXPECT_GE(s.n_blend_pixels, 5u);
    EXPECT_EQ(s.count, s.n_instances);
    EXPECT_EQ(s.n_missing_blend_tiles, 0u)
        << "instances=" << tiles_to_string(s.instance_tiles, s.n_instances)
        << " blend=" << tiles_to_string(s.blend_tiles, s.n_blend_tiles);
    EXPECT_EQ(s.n_bwd_drops, 0u);
    EXPECT_EQ(s.count, s.n_blend_tiles);

    bool saw_tile1 = false;
    for (uint32_t i = 0; i < s.n_blend_tiles; i++) {
        if (s.blend_tiles[i] == 1u) {
            saw_tile1 = true;
        }
    }
    EXPECT_TRUE(saw_tile1) << "pixel (16,8) belongs to tile 1";
}

TEST(FastGSTileCullingBoundary, LowerTileEdgePixelIsBinnedAndShaded) {
    auto s = make_agreement(8.5f, 15.5f, boundary_opacity());
    ASSERT_EQ(eval_agreement(&s, 1), cudaSuccess);
    EXPECT_EQ(s.count, s.n_instances);
    EXPECT_EQ(s.n_missing_blend_tiles, 0u)
        << "instances=" << tiles_to_string(s.instance_tiles, s.n_instances)
        << " blend=" << tiles_to_string(s.blend_tiles, s.n_blend_tiles);
    EXPECT_EQ(s.n_bwd_drops, 0u);
    EXPECT_EQ(s.count, s.n_blend_tiles);

    const uint32_t tile_below = 4u; // tile (0,1) on a 4-wide grid
    bool saw = false;
    for (uint32_t i = 0; i < s.n_blend_tiles; i++) {
        if (s.blend_tiles[i] == tile_below) {
            saw = true;
        }
    }
    EXPECT_TRUE(saw) << "pixel (8,16) belongs to tile (0,1)";
}

TEST(FastGSTileCullingBoundary, PointEllipseOnTileOriginPixelCenter) {
    auto s = make_agreement(16.5f, 16.5f, kMinAlpha);
    ASSERT_EQ(eval_agreement(&s, 1), cudaSuccess);
    EXPECT_FLOAT_EQ(s.power_threshold, 0.0f);
    EXPECT_EQ(s.n_blend_pixels, 1u);
    EXPECT_EQ(s.count, s.n_instances);
    EXPECT_EQ(s.n_missing_blend_tiles, 0u)
        << "instances=" << tiles_to_string(s.instance_tiles, s.n_instances)
        << " blend=" << tiles_to_string(s.blend_tiles, s.n_blend_tiles);
    EXPECT_EQ(s.n_bwd_drops, 0u);
    EXPECT_EQ(s.count, 1u);
    EXPECT_EQ(s.n_blend_tiles, 1u);
    EXPECT_EQ(s.blend_tiles[0], 5u); // tile (1,1) on a 4-wide grid
}

// Same mean/conic/opacity: every subtile with a contributing pixel must be
// kept by splat_overlaps_subtile_ellipse (the backward warp-cull path).
TEST(FastGSTileCullingBoundary, ForwardBackwardTouchedSubtilesAgree) {
    AgreementSample cases[] = {
        make_agreement(8.5f, 8.5f, boundary_opacity()),
        make_agreement(15.5f, 8.5f, boundary_opacity()),
        make_agreement(8.5f, 15.5f, boundary_opacity()),
        make_agreement(15.5f, 15.5f, boundary_opacity()),
        make_agreement(16.5f, 16.5f, kMinAlpha),
        make_agreement(32.0f, 24.5f, boundary_opacity()),
    };
    ASSERT_EQ(eval_agreement(cases, 6), cudaSuccess);
    for (int i = 0; i < 6; i++) {
        EXPECT_EQ(cases[i].count, cases[i].n_instances) << "case " << i;
        EXPECT_EQ(cases[i].n_missing_blend_tiles, 0u)
            << "case " << i
            << " instances=" << tiles_to_string(cases[i].instance_tiles, cases[i].n_instances)
            << " blend=" << tiles_to_string(cases[i].blend_tiles, cases[i].n_blend_tiles);
        EXPECT_EQ(cases[i].n_bwd_drops, 0u) << "case " << i;
        EXPECT_EQ(cases[i].count, cases[i].n_blend_tiles) << "case " << i;
    }
}
