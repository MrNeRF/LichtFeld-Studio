/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/vulkan_device_selection.hpp"
#include <gtest/gtest.h>

namespace {
    using namespace lfs::core;
    constexpr VulkanDeviceUuid first_uuid{0xab, 0xcd};
    constexpr VulkanDeviceUuid second_uuid{0xef, 0x12};

    std::array<VulkanDeviceCandidate, 2> devices() {
        return {{{first_uuid, true, true, true}, {second_uuid, true, true, true}}};
    }

    TEST(HeadlessVulkanDeviceSelection, HonorsConfiguredEnumerationIndex) {
        EXPECT_EQ(select_headless_vulkan_device(devices(), "1", std::nullopt), 1u);
        EXPECT_FALSE(select_headless_vulkan_device(devices(), "2", std::nullopt));
    }

    TEST(HeadlessVulkanDeviceSelection, HonorsConfiguredUuidWithGuiNormalization) {
        EXPECT_EQ(select_headless_vulkan_device(devices(), "{EF120000-0000-0000-0000-000000000000}", std::nullopt), 1u);
        EXPECT_FALSE(select_headless_vulkan_device(devices(), "not-a-device", std::nullopt));
    }

    TEST(HeadlessVulkanDeviceSelection, MatchesCudaUuidInsteadOfFirstDiscreteDevice) {
        EXPECT_EQ(select_headless_vulkan_device(devices(), {}, second_uuid), 1u);
    }

    TEST(HeadlessVulkanDeviceSelection, RejectsPreferenceConflictingWithCudaDevice) {
        EXPECT_FALSE(select_headless_vulkan_device(devices(), "0", second_uuid));
        EXPECT_EQ(select_headless_vulkan_device(devices(), "1", second_uuid), 1u);
        EXPECT_FALSE(select_headless_vulkan_device(devices(), {}, VulkanDeviceUuid{0x42}));
    }

    TEST(HeadlessVulkanDeviceSelection, TriesNextDeviceWhenRequiredExtensionsAreMissing) {
        auto candidates = devices();
        candidates[0].required_extensions = false;
        EXPECT_EQ(select_headless_vulkan_device(candidates, {}, std::nullopt), 1u);
        EXPECT_FALSE(select_headless_vulkan_device(candidates, "0", std::nullopt));
        EXPECT_FALSE(select_headless_vulkan_device(candidates, {}, first_uuid));
    }

    TEST(HeadlessVulkanDeviceSelection, TriesNextDeviceWhenRequiredFeaturesAreMissing) {
        auto candidates = devices();
        candidates[0].required_features = false;
        EXPECT_EQ(select_headless_vulkan_device(candidates, {}, std::nullopt), 1u);
        candidates[1].required_extensions = false;
        EXPECT_FALSE(select_headless_vulkan_device(candidates, {}, std::nullopt));
    }

    TEST(HeadlessVulkanDeviceSelection, PrefersDiscreteDeviceWithoutCudaInterop) {
        auto candidates = devices();
        candidates[0].discrete = false;
        EXPECT_EQ(select_headless_vulkan_device(candidates, {}, std::nullopt), 1u);
        EXPECT_EQ(select_headless_vulkan_device(candidates, "0", std::nullopt), 0u);
        candidates[1].required_extensions = false;
        EXPECT_EQ(select_headless_vulkan_device(candidates, {}, std::nullopt), 0u);
    }
} // namespace
