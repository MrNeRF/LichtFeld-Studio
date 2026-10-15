/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/vksplat_output_slot.hpp"
#include <array>
#include <gtest/gtest.h>
#include <limits>

namespace {
    using lfs::vis::kVksplatOutputSlotCount;
    using lfs::vis::VksplatOutputSlot;
    using lfs::vis::vksplatOutputSlotIndex;
} // namespace

TEST(VksplatOutputSlot, AcceptsEveryDesktopAndVrOutput) {
    constexpr std::array slots{
        VksplatOutputSlot::Main, VksplatOutputSlot::SplitLeft,
        VksplatOutputSlot::SplitRight, VksplatOutputSlot::Preview,
        VksplatOutputSlot::VrLeft, VksplatOutputSlot::VrRight};
    ASSERT_EQ(slots.size(), kVksplatOutputSlotCount);
    for (std::size_t i = 0; i < slots.size(); ++i)
        EXPECT_EQ(vksplatOutputSlotIndex(slots[i]), i);
}

TEST(VksplatOutputSlot, VrEyesAreValidDistinctArrayIndices) {
    std::array<int, kVksplatOutputSlotCount> outputs{};
    ASSERT_NO_THROW(outputs.at(vksplatOutputSlotIndex(VksplatOutputSlot::VrLeft)) = 11);
    ASSERT_NO_THROW(outputs.at(vksplatOutputSlotIndex(VksplatOutputSlot::VrRight)) = 22);
    EXPECT_EQ(outputs.at(vksplatOutputSlotIndex(VksplatOutputSlot::VrLeft)), 11);
    EXPECT_EQ(outputs.at(vksplatOutputSlotIndex(VksplatOutputSlot::VrRight)), 22);
    EXPECT_EQ(outputs.at(vksplatOutputSlotIndex(VksplatOutputSlot::Preview)), 0);
}

TEST(VksplatOutputSlot, RejectsCountSentinelAndInvalidValues) {
    EXPECT_THROW((void)vksplatOutputSlotIndex(VksplatOutputSlot::Count), std::out_of_range);
    EXPECT_THROW((void)vksplatOutputSlotIndex(static_cast<VksplatOutputSlot>(kVksplatOutputSlotCount + 1)),
                 std::out_of_range);
    EXPECT_THROW((void)vksplatOutputSlotIndex(static_cast<VksplatOutputSlot>(std::numeric_limits<std::size_t>::max())),
                 std::out_of_range);
}
