/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../src/rendering/rasterizer/vulkan/src/portal_compact.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace lfs::rendering;

TEST(PortalRenderProfile, CompactScaleCodesMatchPinnedPortal) {
    // Independent scale-byte fixtures from the portal's container format:
    // zero is reserved; the remaining codes span log scale [-12, 9].
    EXPECT_FLOAT_EQ(lfsPortalCompactScale(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(lfsPortalCompactScale(1e-11f), 0.0f);
    EXPECT_NEAR(lfsPortalCompactScale(1e-10f), std::exp(-12.0f + 21.0f / 255.0f), 1e-11f);
    EXPECT_NEAR(lfsPortalCompactScale(1.0f), std::exp(-12.0f + 146.0f * 21.0f / 255.0f), 2e-6f);
    EXPECT_NEAR(lfsPortalCompactScale(1e8f), std::exp(9.0f), 0.01f);
    for (int code = 1; code <= 255; ++code) {
        const float decoded = std::exp(-12.0f + code * (21.0f / 255.0f));
        EXPECT_NEAR(lfsPortalCompactScale(decoded), decoded, decoded * 2e-6f);
    }
}

TEST(PortalRenderProfile, CompactOpacityAndColorKeepTheirSourceRanges) {
    EXPECT_FLOAT_EQ(lfsPortalCompactOpacity(-1.0f), 0.0f);
    EXPECT_FLOAT_EQ(lfsPortalCompactOpacity(2.0f), 1.0f);
    EXPECT_FLOAT_EQ(lfsPortalCompactOpacity(0.5f), 128.0f / 255.0f);
    const auto limits = lfsPortalCompactColor(glm::vec3(-1.0f, 4.0f, 5.0f));
    EXPECT_FLOAT_EQ(limits.x, 0.0f);
    EXPECT_FLOAT_EQ(limits.y, 4.0f);
    EXPECT_FLOAT_EQ(limits.z, 4.0f);
    const auto mid = lfsPortalCompactColor(glm::vec3(1.0f));
    EXPECT_NEAR(mid.x, 2048.0f / 2047.0f, 1e-7f);
    EXPECT_NEAR(mid.y, 2048.0f / 2047.0f, 1e-7f);
    EXPECT_NEAR(mid.z, 1024.0f / 1023.0f, 1e-7f);
}

TEST(PortalRenderProfile, CompactRotationUsesPortalFrameAndNativeComponentOrder) {
    const auto q = lfsPortalCompactRotation(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    // Portal's Z-180 identity frame encodes quaternion vector as (1024,1024,1023).
    const float xy = 1.0f / 2047.0f;
    const float d = 1.0f + 2.0f * xy * xy;
    EXPECT_NEAR(q.x, std::sqrt(2.0f - d), 2e-7f);
    EXPECT_NEAR(q.y, std::sqrt(2.0f - d) * xy, 1e-7f);
    EXPECT_NEAR(q.z, -std::sqrt(2.0f - d) * xy, 1e-7f);
    EXPECT_NEAR(q.w, d - 1.0f, 2e-7f);
    EXPECT_NEAR(glm::dot(q, q), 1.0f, 2e-6f);
    for (const auto source : {glm::vec4(0, 1, 0, 0), glm::vec4(0, 0, 1, 0),
                              glm::vec4(0, 0, 0, 1), glm::vec4(.5f, .5f, .5f, .5f)}) {
        const auto packed = lfsPortalCompactRotation(source);
        EXPECT_NEAR(glm::dot(packed, packed), 1.0f, 2e-6f);
        EXPECT_GT(std::abs(glm::dot(source, packed)), 0.99999f);
    }
}
