/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "screen/screen_layout.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace lfs::vis::screen {

    namespace {

        constexpr AreaId A{1};
        constexpr AreaId B{2};
        constexpr AreaId C{3};
        constexpr AreaId D{4};

        const Rect kBounds{0.0f, 0.0f, 1000.0f, 600.0f};
        const LayoutMetrics kMetrics{2.0f, 48.0f, 32.0f};

        Rect rectOf(const LayoutGeometry& g, const AreaId id) {
            const auto* found = g.find(id);
            EXPECT_NE(found, nullptr);
            return found ? found->rect : Rect{};
        }

        // Every pixel of the bounds belongs to exactly one area or divider.
        void expectTiles(const LayoutGeometry& g) {
            float covered = 0.0f;
            for (const auto& a : g.areas) {
                EXPECT_GE(a.rect.w, 0.0f);
                EXPECT_GE(a.rect.h, 0.0f);
                covered += a.rect.w * a.rect.h;
            }
            for (const auto& d : g.dividers)
                covered += d.rect.w * d.rect.h;
            EXPECT_FLOAT_EQ(covered, g.bounds.w * g.bounds.h);
            for (std::size_t i = 0; i < g.areas.size(); ++i) {
                for (std::size_t j = i + 1; j < g.areas.size(); ++j) {
                    const Rect& p = g.areas[i].rect;
                    const Rect& q = g.areas[j].rect;
                    const bool disjoint = p.right() <= q.x || q.right() <= p.x || p.bottom() <= q.y || q.bottom() <= p.y;
                    EXPECT_TRUE(disjoint) << "areas " << g.areas[i].area.value << " and " << g.areas[j].area.value;
                }
            }
        }

    } // namespace

    TEST(ScreenLayout, SingleAreaCoversBounds) {
        const ScreenLayout layout(A);
        const auto g = layout.solve(kBounds, kMetrics);
        ASSERT_EQ(g.areas.size(), 1u);
        EXPECT_EQ(g.areas[0].rect, kBounds);
        EXPECT_TRUE(g.dividers.empty());
    }

    TEST(ScreenLayout, SplitPlacesNewAreaAfterTargetWithRequestedShare) {
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Columns, 0.25f));
        const auto g = layout.solve(kBounds, kMetrics);
        expectTiles(g);
        const Rect a = rectOf(g, A);
        const Rect b = rectOf(g, B);
        EXPECT_FLOAT_EQ(a.x, 0.0f);
        EXPECT_NEAR(b.w, (1000.0f - 2.0f) * 0.25f, 1.0f);
        EXPECT_FLOAT_EQ(b.right(), 1000.0f);
        ASSERT_EQ(g.dividers.size(), 1u);
        EXPECT_FLOAT_EQ(g.dividers[0].rect.x, a.right());
        EXPECT_FLOAT_EQ(g.dividers[0].rect.right(), b.x);
    }

    TEST(ScreenLayout, SplitBeforeTarget) {
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Rows, 0.5f, true));
        const auto g = layout.solve(kBounds, kMetrics);
        EXPECT_LT(rectOf(g, B).y, rectOf(g, A).y);
    }

    TEST(ScreenLayout, SameAxisSplitsBecomeSiblings) {
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Columns, 0.5f));
        ASSERT_TRUE(layout.split(B, C, SplitAxis::Columns, 0.5f));
        const auto* root = layout.root();
        ASSERT_NE(root, nullptr);
        EXPECT_FALSE(root->isArea());
        EXPECT_EQ(root->children.size(), 3u);
        const auto g = layout.solve(kBounds, kMetrics);
        EXPECT_EQ(g.dividers.size(), 2u);
        expectTiles(g);
    }

    TEST(ScreenLayout, RejectsInvalidSplits) {
        ScreenLayout layout(A);
        EXPECT_FALSE(layout.split(A, A, SplitAxis::Columns, 0.5f));
        EXPECT_FALSE(layout.split(A, AreaId{}, SplitAxis::Columns, 0.5f));
        EXPECT_FALSE(layout.split(B, C, SplitAxis::Columns, 0.5f));
        EXPECT_FALSE(layout.split(A, B, SplitAxis::Columns, 0.0f));
        EXPECT_FALSE(layout.split(A, B, SplitAxis::Columns, 1.0f));
        EXPECT_EQ(layout.areaCount(), 1u);
    }

    TEST(ScreenLayout, RemoveGivesSpaceToNeighbourAndCollapses) {
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Columns, 0.5f));
        ASSERT_TRUE(layout.split(B, C, SplitAxis::Rows, 0.5f));
        ASSERT_TRUE(layout.remove(C));
        EXPECT_EQ(layout.areaCount(), 2u);
        // B's Rows split collapsed back into a leaf of the root Columns split.
        EXPECT_EQ(layout.root()->children.size(), 2u);
        EXPECT_TRUE(layout.root()->children[1].isArea());
        ASSERT_TRUE(layout.remove(A));
        EXPECT_TRUE(layout.root()->isArea());
        EXPECT_FALSE(layout.remove(B)) << "the last area stays";
    }

    TEST(ScreenLayout, JoinOnlyAdjacentLeavesOfOneSplit) {
        // Columns[A, Rows[B, C]]
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Columns, 0.5f));
        ASSERT_TRUE(layout.split(B, C, SplitAxis::Rows, 0.5f));
        EXPECT_TRUE(layout.canJoin(B, C));
        EXPECT_TRUE(layout.canJoin(C, B));
        EXPECT_FALSE(layout.canJoin(A, B)) << "A's edge spans B and C";
        EXPECT_FALSE(layout.canJoin(A, A));
        ASSERT_TRUE(layout.join(C, B));
        EXPECT_FALSE(layout.contains(B));
        EXPECT_TRUE(layout.canJoin(A, C)) << "now C spans the full edge";
        const auto g = layout.solve(kBounds, kMetrics);
        EXPECT_FLOAT_EQ(rectOf(g, C).h, 600.0f);
    }

    TEST(ScreenLayout, JoinableNeighbourFollowsSides) {
        // Columns[A, Rows[B, C]]
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Columns, 0.5f));
        ASSERT_TRUE(layout.split(B, C, SplitAxis::Rows, 0.5f));
        EXPECT_EQ(layout.joinableNeighbour(B, Side::Bottom), C);
        EXPECT_EQ(layout.joinableNeighbour(C, Side::Top), B);
        EXPECT_FALSE(layout.joinableNeighbour(B, Side::Left).valid()) << "A is not B's full-edge neighbour";
        EXPECT_FALSE(layout.joinableNeighbour(A, Side::Right).valid()) << "right of A is a split, not an area";
        EXPECT_FALSE(layout.joinableNeighbour(A, Side::Left).valid());
    }

    TEST(ScreenLayout, SwapExchangesPositions) {
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Columns, 0.3f));
        const auto before = layout.solve(kBounds, kMetrics);
        ASSERT_TRUE(layout.swap(A, B));
        const auto after = layout.solve(kBounds, kMetrics);
        EXPECT_EQ(rectOf(after, A), rectOf(before, B));
        EXPECT_EQ(rectOf(after, B), rectOf(before, A));
    }

    TEST(ScreenLayout, MoveDividerClampsToMinimums) {
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Columns, 0.5f));
        auto g = layout.solve(kBounds, kMetrics);
        ASSERT_EQ(g.dividers.size(), 1u);
        ASSERT_TRUE(layout.moveDivider(g.dividers[0], 300.0f));
        g = layout.solve(kBounds, kMetrics);
        EXPECT_NEAR(rectOf(g, A).w, 300.0f, 1.0f);

        ASSERT_TRUE(layout.moveDivider(g.dividers[0], -500.0f));
        g = layout.solve(kBounds, kMetrics);
        EXPECT_NEAR(rectOf(g, A).w, kMetrics.min_width, 1.0f);

        ASSERT_TRUE(layout.moveDivider(g.dividers[0], 5000.0f));
        g = layout.solve(kBounds, kMetrics);
        EXPECT_NEAR(rectOf(g, B).w, kMetrics.min_width, 1.0f);
        expectTiles(g);
    }

    TEST(ScreenLayout, MoveDividerRebasesWeightsAfterMinimumClamping) {
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Columns, 0.01f));
        ASSERT_TRUE(layout.setWeights(layout.root()->split, {0.99f, 0.01f}));
        auto g = layout.solve(Rect{0.0f, 0.0f, 1000.0f, 300.0f}, kMetrics);
        ASSERT_EQ(g.dividers.size(), 1u);
        ASSERT_NEAR(rectOf(g, A).w, 950.0f, 1.0f);
        ASSERT_TRUE(layout.moveDivider(g.dividers[0], 900.0f));
        g = layout.solve(Rect{0.0f, 0.0f, 1000.0f, 300.0f}, kMetrics);
        EXPECT_NEAR(g.dividers[0].rect.x, 900.0f, 1.0f);
    }

    TEST(ScreenLayout, NestedDividerNormalizesAgainstItsSplit) {
        // Rows[Columns[A, B], C]: the A|B divider only moves inside the top half.
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, C, SplitAxis::Rows, 0.5f));
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Columns, 0.5f));
        auto g = layout.solve(kBounds, kMetrics);
        const DividerGeometry* inner = nullptr;
        for (const auto& d : g.dividers) {
            if (d.axis == SplitAxis::Columns)
                inner = &d;
        }
        ASSERT_NE(inner, nullptr);
        const Rect c_before = rectOf(g, C);
        ASSERT_TRUE(layout.moveDivider(*inner, 700.0f));
        g = layout.solve(kBounds, kMetrics);
        EXPECT_NEAR(rectOf(g, A).w, 700.0f, 1.0f);
        EXPECT_EQ(rectOf(g, C), c_before);
    }

    TEST(ScreenLayout, SubtreeMinimumsHoldWhenSpaceShrinks) {
        // Columns[A, Rows[B, Columns[C, D]]]: the right subtree needs two
        // minimum widths plus a divider.
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Columns, 0.5f));
        ASSERT_TRUE(layout.split(B, C, SplitAxis::Rows, 0.5f));
        ASSERT_TRUE(layout.split(C, D, SplitAxis::Columns, 0.5f));
        const Rect narrow{0.0f, 0.0f, 200.0f, 400.0f};
        const auto g = layout.solve(narrow, kMetrics);
        expectTiles(g);
        EXPECT_GE(rectOf(g, C).w, kMetrics.min_width - 1.0f);
        EXPECT_GE(rectOf(g, D).w, kMetrics.min_width - 1.0f);
        EXPECT_GE(rectOf(g, A).w, kMetrics.min_width - 1.0f);
    }

    TEST(ScreenLayout, EdgesAreWholePixels) {
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Columns, 1.0f / 3.0f));
        ASSERT_TRUE(layout.split(A, C, SplitAxis::Columns, 0.5f));
        const auto g = layout.solve(Rect{0.0f, 0.0f, 997.0f, 311.0f}, kMetrics);
        for (const auto& a : g.areas) {
            EXPECT_FLOAT_EQ(a.rect.x, std::round(a.rect.x));
            EXPECT_FLOAT_EQ(a.rect.w, std::round(a.rect.w));
        }
        expectTiles(g);
    }

    TEST(ScreenLayout, TinyBoundsNeverProduceNegativeRects) {
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Columns, 0.5f));
        ASSERT_TRUE(layout.split(B, C, SplitAxis::Rows, 0.5f));
        for (const Rect bounds : {Rect{0, 0, 10, 10}, Rect{0, 0, 1, 1}, Rect{5, 5, 0, 0}}) {
            const auto g = layout.solve(bounds, kMetrics);
            for (const auto& a : g.areas) {
                EXPECT_GE(a.rect.w, 0.0f);
                EXPECT_GE(a.rect.h, 0.0f);
            }
        }
    }

    TEST(ScreenLayout, MaximizedAreaCoversBoundsAlone) {
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Columns, 0.5f));
        const auto g = layout.solve(kBounds, kMetrics, B);
        ASSERT_EQ(g.areas.size(), 1u);
        EXPECT_EQ(g.areas[0].area, B);
        EXPECT_EQ(g.areas[0].rect, kBounds);
        EXPECT_TRUE(g.dividers.empty());
        EXPECT_EQ(g.maximized, B);
        EXPECT_EQ(layout.solve(kBounds, kMetrics, C).areas.size(), 2u) << "unknown ids are ignored";
    }

    TEST(ScreenLayout, HitTestingIsHalfOpenAndDividersUseSlop) {
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Columns, 0.5f));
        const auto g = layout.solve(kBounds, kMetrics);
        const Rect a = rectOf(g, A);
        EXPECT_EQ(g.areaAt(a.x, a.y), A);
        EXPECT_FALSE(g.areaAt(a.right(), a.y).valid()) << "the divider gap belongs to no area";
        const float mid = g.dividers[0].rect.x + 1.0f;
        EXPECT_NE(g.dividerAt(mid, 300.0f, 0.0f), nullptr);
        EXPECT_NE(g.dividerAt(mid - 5.0f, 300.0f, 6.0f), nullptr);
        EXPECT_EQ(g.dividerAt(mid - 20.0f, 300.0f, 6.0f), nullptr);
    }

    TEST(ScreenLayout, JsonRoundTrip) {
        ScreenLayout layout(A);
        ASSERT_TRUE(layout.split(A, B, SplitAxis::Columns, 0.3f));
        ASSERT_TRUE(layout.split(B, C, SplitAxis::Rows, 0.4f));
        ASSERT_TRUE(layout.split(C, D, SplitAxis::Columns, 0.5f));
        const auto json = layout.toJson();
        const auto restored = ScreenLayout::fromJson(json);
        ASSERT_TRUE(restored.has_value());
        EXPECT_EQ(*restored, layout);
        EXPECT_EQ(restored->solve(kBounds, kMetrics).areas.size(), 4u);

        // New splits after a restore never reuse a split id.
        ScreenLayout grown = *restored;
        ASSERT_TRUE(grown.split(A, AreaId{9}, SplitAxis::Rows, 0.5f));
        EXPECT_GE(grown.nextSplitId(), layout.nextSplitId());
    }

    TEST(ScreenLayout, JsonRejectsMalformedInput) {
        using nlohmann::json;
        EXPECT_FALSE(ScreenLayout::fromJson(json::parse(R"({"area": 0})")));
        EXPECT_FALSE(ScreenLayout::fromJson(json::parse(R"({"split": 1, "axis": "rows", "weights": [1], "children": [{"area": 1}]})")));
        EXPECT_FALSE(ScreenLayout::fromJson(json::parse(
            R"({"split": 1, "axis": "rows", "weights": [0.5, 0.5], "children": [{"area": 1}, {"area": 1}]})")))
            << "duplicate area";
        EXPECT_FALSE(ScreenLayout::fromJson(json::parse(
            R"({"split": 1, "axis": "diagonal", "weights": [0.5, 0.5], "children": [{"area": 1}, {"area": 2}]})")));
        EXPECT_FALSE(ScreenLayout::fromJson(json::parse(
            R"({"split": 1, "axis": "rows", "weights": [0.5, -1], "children": [{"area": 1}, {"area": 2}]})")));
        EXPECT_FALSE(ScreenLayout::fromJson(json::parse(R"([1, 2, 3])")));
        EXPECT_TRUE(ScreenLayout::fromJson(json::parse(R"({"area": 7})")));
        EXPECT_FALSE(ScreenLayout::fromJson(json::parse(R"({"area": 4294967295})")));
        EXPECT_FALSE(ScreenLayout::fromJson(json::parse(
            R"({"split": 4294967295, "axis": "columns", "weights": [0.5, 0.5], "children": [{"area": 1}, {"area": 2}]})")));
    }

    TEST(ScreenLayout, JsonNormalizesLargeFiniteWeightsWithoutOverflow) {
        const auto restored = ScreenLayout::fromJson(nlohmann::json::parse(
            R"({"split": 1, "axis": "columns", "weights": [3e38, 3e38], "children": [{"area": 1}, {"area": 2}]})"));
        ASSERT_TRUE(restored);
        EXPECT_NEAR(restored->root()->weights[0], 0.5f, 1e-6f);
        EXPECT_NEAR(restored->root()->weights[1], 0.5f, 1e-6f);
    }

    TEST(ScreenLayout, JsonFlattensSameAxisNesting) {
        const auto restored = ScreenLayout::fromJson(nlohmann::json::parse(R"({
            "split": 1, "axis": "columns", "weights": [0.5, 0.5],
            "children": [{"area": 1},
                         {"split": 2, "axis": "columns", "weights": [0.5, 0.5],
                          "children": [{"area": 2}, {"area": 3}]}]})"));
        ASSERT_TRUE(restored);
        ASSERT_EQ(restored->root()->children.size(), 3u);
        EXPECT_NEAR(restored->root()->weights[0], 0.5f, 1e-6f);
        EXPECT_NEAR(restored->root()->weights[1], 0.25f, 1e-6f);
    }

} // namespace lfs::vis::screen
