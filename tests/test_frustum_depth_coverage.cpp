/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "rendering/frustum_depth_coverage.hpp"
#include <gtest/gtest.h>

TEST(FrustumDepthCoverage, CoversClippedLineAndBilinearNeighbours) {
    constexpr glm::ivec2 size(101, 73); // Rows do not align to bitset words.
    lfs::vis::FrustumDepthCoverage coverage(size, 4.0f);
    const glm::vec2 a(-10000, -6000), b(20000, 12000);
    coverage.line(a, b);
    const auto words = coverage.take();
    for (int y = 0; y < size.y; ++y) {
        for (int x = 0; x < size.x; ++x) {
            const glm::vec2 p(x + 0.5f, y + 0.5f);
            const auto d = b - a;
            const auto closest = a + d * glm::clamp(glm::dot(p - a, d) / glm::dot(d, d), 0.0f, 1.0f);
            if (glm::distance(p, closest) <= 3.0f) {
                const size_t bit = size_t(y / 4) * ((size.x + 3) / 4) + x / 4;
                EXPECT_NE(words[bit / 32] & (1u << (bit % 32)), 0u) << x << ',' << y;
            }
        }
    }
    EXPECT_EQ(words.back() >> ((((size.x + 3) / 4) * ((size.y + 3) / 4)) % 32), 0u);
}

TEST(FrustumDepthCoverage, OffscreenGeometryDoesNotWrapIntoOtherRows) {
    lfs::vis::FrustumDepthCoverage coverage({101, 73}, 4.0f);
    coverage.line({-10000, -10000}, {-5000, -5000});
    coverage.rectangle({105, 10}, {200, 50});
    coverage.rectangle({10, 77}, {90, 200});
    for (auto word : coverage.take())
        EXPECT_EQ(word, 0u);
}

TEST(FrustumDepthCoverage, ImagePlaneIncludesSampleFootprintAtTargetEdges) {
    lfs::vis::FrustumDepthCoverage coverage({101, 73}, 4.0f);
    coverage.rectangle({98, 70}, {120, 100});
    const auto words = coverage.take();
    for (int y = 66; y < 73; ++y) {
        for (int x = 94; x < 101; ++x) {
            const size_t bit = size_t(y / 4) * 26 + x / 4;
            EXPECT_NE(words[bit / 32] & (1u << (bit % 32)), 0u);
        }
    }
    EXPECT_EQ(words.front(), 0u);
}

TEST(FrustumDepthCoverage, TiltedThumbnailIncludesEveryInteriorSample) {
    constexpr glm::ivec2 size(101, 73);
    constexpr float margin = 4.0f;
    const std::array<glm::vec2, 4> points{{{40, -20}, {110, 35}, {60, 80}, {-10, 25}}};
    lfs::vis::FrustumDepthCoverage coverage(size, margin);
    coverage.quad(points);
    const auto words = coverage.take();
    for (int y = 0; y < size.y; ++y) {
        for (int x = 0; x < size.x; ++x) {
            const glm::vec2 p(x + 0.5f, y + 0.5f);
            bool inside = true, near_edge = false;
            for (size_t i = 0; i < points.size(); ++i) {
                const auto a = points[i], d = points[(i + 1) % points.size()] - a;
                const auto offset = p - a;
                inside &= d.x * offset.y - d.y * offset.x >= 0;
                const auto closest = a + d * glm::clamp(glm::dot(offset, d) / glm::dot(d, d), 0.0f, 1.0f);
                near_edge |= glm::distance(p, closest) < margin;
            }
            if (inside || near_edge) {
                const size_t bit = size_t(y / 4) * ((size.x + 3) / 4) + x / 4;
                EXPECT_NE(words[bit / 32] & (1u << (bit % 32)), 0u) << x << ',' << y;
            }
        }
    }
}
