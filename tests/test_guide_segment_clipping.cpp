/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/guide_segment_clipping.hpp"
#include <array>
#include <gtest/gtest.h>
#include <limits>

namespace {
    using lfs::vis::gui::clipGuideSegment;
    using lfs::vis::gui::GuideClipPlane;

    constexpr std::array<GuideClipPlane, 5> planes{{
        {{0, 0, -1}, -1e-4f},
        {{500, 0, -500}, 0},
        {{-500, 0, -500}, 0},
        {{0, -500, -500}, 0},
        {{0, 500, -500}, 0},
    }};

    TEST(GuideSegmentClipping, LongCropEdgeRetainsPositiveDepthAtCameraPlane) {
        for (const float extent : {1000.0f, 10000.0f, 100000.0f}) {
            SCOPED_TRACE(extent);
            const glm::vec3 a{-1, 0, -extent}, b{1, 1e-5f, extent};
            for (const bool reverse : {false, true}) {
                const auto clipped = clipGuideSegment(reverse ? b : a, reverse ? a : b, planes);
                ASSERT_TRUE(clipped);
                const auto endpoint = reverse ? clipped->first : clipped->second;
                EXPECT_NEAR(endpoint.z, -1e-4f, 1e-7f);
                EXPECT_GT(-endpoint.z, 0.0f);
                const float projected_y = 500.0f - endpoint.y * 500.0f / -endpoint.z;
                EXPECT_TRUE(std::isfinite(projected_y));
                EXPECT_NEAR(projected_y, 475.0f, 0.1f);
            }
        }
    }

    TEST(GuideSegmentClipping, PreservesVisibleEdge) {
        const glm::vec3 a{-1, -1, -10}, b{1, 1, -10};
        const auto clipped = clipGuideSegment(a, b, planes);
        ASSERT_TRUE(clipped);
        EXPECT_EQ(clipped->first, a);
        EXPECT_EQ(clipped->second, b);
    }

    TEST(GuideSegmentClipping, ClipsToViewportSides) {
        const auto clipped = clipGuideSegment({-20, 0, -10}, {20, 0, -10}, planes);
        ASSERT_TRUE(clipped);
        EXPECT_FLOAT_EQ(clipped->first.x, -10);
        EXPECT_FLOAT_EQ(clipped->second.x, 10);
    }

    TEST(GuideSegmentClipping, RejectsEdgeBehindCamera) {
        EXPECT_FALSE(clipGuideSegment({-1, 0, 10}, {1, 0, 10}, planes));
    }

    TEST(GuideSegmentClipping, RejectsNonFiniteCoordinates) {
        EXPECT_FALSE(clipGuideSegment({std::numeric_limits<float>::infinity(), 0, -10}, {1, 0, -10}, planes));
    }

    TEST(GuideSegmentClipping, ClipsOrthographicEdgesWithoutChangingTheirScale) {
        constexpr std::array<GuideClipPlane, 5> ortho{{
            {{0, 0, -1}, -1e-4f},
            {{10, 0, 0}, 500},
            {{-10, 0, 0}, 500},
            {{0, -10, 0}, 500},
            {{0, 10, 0}, 500},
        }};
        const auto clipped = clipGuideSegment({-100, 0, -10}, {100, 0, -10}, ortho);
        ASSERT_TRUE(clipped);
        EXPECT_FLOAT_EQ(clipped->first.x, -50);
        EXPECT_FLOAT_EQ(clipped->second.x, 50);
    }
} // namespace
