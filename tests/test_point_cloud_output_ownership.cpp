/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// GPU-free ownership contract for keyed point-cloud outputs. Geometry stays
// on one renderer; color/depth records are per ViewOutputKey.

#include "rendering/point_cloud_vulkan_renderer.hpp"
#include "rendering/view_output_key.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace {

    using lfs::vis::kLegacyMainOutputKey;
    using lfs::vis::kLegacySplitLeftOutputKey;
    using lfs::vis::kLegacySplitRightOutputKey;
    using lfs::vis::PointCloudOutputOwnershipTestAccess;
    using lfs::vis::PointCloudVulkanRenderer;
    using lfs::vis::sceneOutputKey;
    using lfs::vis::ViewOutputKey;

    [[nodiscard]] std::vector<ViewOutputKey> sortedKeys(const PointCloudVulkanRenderer& renderer) {
        auto keys = PointCloudOutputOwnershipTestAccess::registeredKeys(renderer);
        std::sort(keys.begin(), keys.end(), [](const ViewOutputKey a, const ViewOutputKey b) {
            if (a.kind != b.kind) {
                return static_cast<std::uint8_t>(a.kind) < static_cast<std::uint8_t>(b.kind);
            }
            return a.view_id < b.view_id;
        });
        return keys;
    }

} // namespace

TEST(PointCloudOutputOwnership, LegacySlotsNeverAliasSceneViewIds) {
    const auto main = PointCloudOutputOwnershipTestAccess::legacyKey(
        PointCloudVulkanRenderer::OutputSlot::Main);
    const auto left = PointCloudOutputOwnershipTestAccess::legacyKey(
        PointCloudVulkanRenderer::OutputSlot::SplitLeft);
    const auto right = PointCloudOutputOwnershipTestAccess::legacyKey(
        PointCloudVulkanRenderer::OutputSlot::SplitRight);
    ASSERT_TRUE(main);
    ASSERT_TRUE(left);
    ASSERT_TRUE(right);
    EXPECT_EQ(*main, kLegacyMainOutputKey);
    EXPECT_EQ(*left, kLegacySplitLeftOutputKey);
    EXPECT_EQ(*right, kLegacySplitRightOutputKey);
    EXPECT_NE(*main, *left);
    EXPECT_NE(*main, *right);
    EXPECT_NE(*left, *right);
    EXPECT_NE(*main, sceneOutputKey(1));
    EXPECT_NE(*left, sceneOutputKey(2));
    EXPECT_NE(*right, sceneOutputKey(3));
}

TEST(PointCloudOutputOwnership, FourSceneKeysStayIndependentAfterRegister) {
    PointCloudVulkanRenderer renderer;
    constexpr std::array<lfs::vis::ViewId, 4> ids{1, 2, 3, 4};
    std::array<ViewOutputKey, 4> keys{};
    std::array<const void*, 4> identities{};
    for (std::size_t i = 0; i < ids.size(); ++i) {
        auto registered = renderer.registerViewOutput(ids[i]);
        ASSERT_TRUE(registered) << registered.error();
        keys[i] = *registered;
        EXPECT_EQ(keys[i], sceneOutputKey(ids[i]));
        identities[i] = PointCloudOutputOwnershipTestAccess::resourceIdentity(renderer, keys[i]);
        ASSERT_NE(identities[i], nullptr);
        EXPECT_EQ(PointCloudOutputOwnershipTestAccess::colorImage(renderer, keys[i]), VK_NULL_HANDLE);
        EXPECT_EQ(PointCloudOutputOwnershipTestAccess::depthImage(renderer, keys[i]), VK_NULL_HANDLE);
    }
    for (std::size_t i = 0; i < keys.size(); ++i) {
        for (std::size_t j = i + 1; j < keys.size(); ++j) {
            EXPECT_NE(keys[i], keys[j]);
            EXPECT_NE(identities[i], identities[j]);
            EXPECT_FALSE(PointCloudOutputOwnershipTestAccess::outputsAlias(
                renderer, keys[i], keys[j]));
        }
        EXPECT_NE(keys[i], kLegacyMainOutputKey);
        EXPECT_FALSE(PointCloudOutputOwnershipTestAccess::outputsAlias(
            renderer, keys[i], kLegacyMainOutputKey));
    }
    EXPECT_EQ(sortedKeys(renderer).size(), 4u);
}

TEST(PointCloudOutputOwnership, RegisterIsIdempotentAndRejectsZero) {
    PointCloudVulkanRenderer renderer;
    auto first = renderer.registerViewOutput(11);
    auto second = renderer.registerViewOutput(11);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_EQ(*first, *second);
    EXPECT_EQ(PointCloudOutputOwnershipTestAccess::resourceIdentity(renderer, *first),
              PointCloudOutputOwnershipTestAccess::resourceIdentity(renderer, *second));
    auto invalid = renderer.registerViewOutput(0);
    EXPECT_FALSE(invalid);
    EXPECT_NE(invalid.error().find("non-zero"), std::string::npos);
}

TEST(PointCloudOutputOwnership, RetiringOneViewDoesNotAliasOrClearNeighbors) {
    PointCloudVulkanRenderer renderer;
    ASSERT_TRUE(renderer.registerViewOutput(101));
    ASSERT_TRUE(renderer.registerViewOutput(202));
    ASSERT_TRUE(renderer.registerViewOutput(303));
    const auto live_a = sceneOutputKey(101);
    const auto retired = sceneOutputKey(202);
    const auto live_b = sceneOutputKey(303);
    const void* identity_a =
        PointCloudOutputOwnershipTestAccess::resourceIdentity(renderer, live_a);
    const void* identity_b =
        PointCloudOutputOwnershipTestAccess::resourceIdentity(renderer, live_b);
    ASSERT_NE(identity_a, nullptr);
    ASSERT_NE(identity_b, nullptr);

    auto released = renderer.releaseViewOutput(202);
    ASSERT_TRUE(released) << released.error();
    EXPECT_EQ(PointCloudOutputOwnershipTestAccess::resourceIdentity(renderer, retired), nullptr);
    EXPECT_EQ(PointCloudOutputOwnershipTestAccess::resourceIdentity(renderer, live_a), identity_a);
    EXPECT_EQ(PointCloudOutputOwnershipTestAccess::resourceIdentity(renderer, live_b), identity_b);
    EXPECT_FALSE(PointCloudOutputOwnershipTestAccess::outputsAlias(renderer, live_a, live_b));

    auto unknown = renderer.releaseViewOutput(202);
    EXPECT_FALSE(unknown);
    auto zero = renderer.releaseViewOutput(0);
    EXPECT_FALSE(zero);

    auto replacement = renderer.registerViewOutput(404);
    ASSERT_TRUE(replacement);
    EXPECT_NE(PointCloudOutputOwnershipTestAccess::resourceIdentity(renderer, *replacement),
              identity_a);
    EXPECT_NE(PointCloudOutputOwnershipTestAccess::resourceIdentity(renderer, *replacement),
              identity_b);
    EXPECT_EQ(PointCloudOutputOwnershipTestAccess::resourceIdentity(renderer, live_a), identity_a);
}
