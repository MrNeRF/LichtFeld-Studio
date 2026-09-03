/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/selection_cursor.hpp"
#include "input/input_bindings.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace {

    using lfs::vis::SelectionPreviewMode;
    using lfs::vis::gui::makeSelectionCursorImage;
    using lfs::vis::gui::SelectionCursorColor;
    using lfs::vis::gui::useHardwareSelectionRing;
    using lfs::vis::input::Action;
    using lfs::vis::input::InputBindings;
    using lfs::vis::input::MODIFIER_ALT;
    using lfs::vis::input::MODIFIER_CTRL;
    using lfs::vis::input::MODIFIER_SHIFT;
    using lfs::vis::input::MouseButton;
    using lfs::vis::input::MouseDragTrigger;
    using lfs::vis::input::SelectionOp;
    using lfs::vis::input::selectionOpForModifiers;
    using lfs::vis::input::ToolMode;

    class SelectionCursorTest : public ::testing::Test {
    protected:
        void SetUp() override {
            InputBindings::setPersistenceEnabled(false);
        }

        void TearDown() override {
            InputBindings::setPersistenceEnabled(true);
        }
    };

    TEST_F(SelectionCursorTest, DefaultKeymapFollowsSelectionDragModifiers) {
        const InputBindings bindings;

        EXPECT_EQ(selectionOpForModifiers(bindings, ToolMode::SELECTION, MODIFIER_SHIFT), SelectionOp::Add);
        EXPECT_EQ(selectionOpForModifiers(bindings, ToolMode::SELECTION, MODIFIER_CTRL), SelectionOp::Remove);
        EXPECT_EQ(selectionOpForModifiers(bindings, ToolMode::SELECTION, MODIFIER_ALT), SelectionOp::Intersect);
        EXPECT_EQ(selectionOpForModifiers(bindings, ToolMode::SELECTION, 0), std::nullopt);
        EXPECT_EQ(selectionOpForModifiers(bindings, ToolMode::TRANSLATE, MODIFIER_SHIFT), std::nullopt);
    }

    TEST_F(SelectionCursorTest, RebindingAddChangesTheResolvedOperation) {
        InputBindings bindings;
        bindings.setBinding(ToolMode::SELECTION,
                            Action::SELECTION_ADD,
                            MouseDragTrigger{MouseButton::LEFT, MODIFIER_ALT});

        EXPECT_EQ(selectionOpForModifiers(bindings, ToolMode::SELECTION, MODIFIER_ALT), SelectionOp::Add);
    }

    TEST(SelectionCursorImageTest, BuildsCenteredAntiAliasedRingAndDot) {
        constexpr int radius = 24;
        const auto image = makeSelectionCursorImage(radius, {255, 32, 64, 204});

        EXPECT_TRUE(image.valid());
        EXPECT_EQ(image.size, 2 * (radius + 8));
        EXPECT_EQ(image.hotspot, radius + 8);
        EXPECT_GT(image.rgba[(static_cast<size_t>(image.hotspot) * image.size +
                              image.hotspot + radius) *
                                 4 +
                             3],
                  0);
        EXPECT_GT(image.rgba[(static_cast<size_t>(image.hotspot) * image.size +
                              image.hotspot) *
                                 4 +
                             3],
                  0);
    }

    TEST(SelectionCursorImageTest, CompositesBadgeAtLowerRight) {
        constexpr int radius = 40;
        constexpr int badge_size = 32;
        const std::vector<uint8_t> badge(static_cast<size_t>(badge_size) * badge_size * 4,
                                         255);
        const auto image = makeSelectionCursorImage(
            radius, SelectionCursorColor{10, 20, 30, 204}, badge, badge_size, badge_size);

        ASSERT_TRUE(image.valid());
        ASSERT_GE(image.badge_x, 0);
        ASSERT_GE(image.badge_y, 0);
        EXPECT_GT(image.rgba[(static_cast<size_t>(image.badge_y) * image.size + image.badge_x) * 4 + 3], 0);
        EXPECT_EQ(image.badge_x + badge_size, image.hotspot + radius);
        EXPECT_EQ(image.badge_y + badge_size, image.hotspot + radius);
    }

    TEST(SelectionCursorImageTest, HardwareRingRequiresCenteredPreviewAndFitsCursorLimit) {
        EXPECT_TRUE(useHardwareSelectionRing(true, SelectionPreviewMode::Centers, 120));
        EXPECT_FALSE(useHardwareSelectionRing(true, SelectionPreviewMode::Centers, 121));
        EXPECT_FALSE(useHardwareSelectionRing(false, SelectionPreviewMode::Centers, 20));
        EXPECT_FALSE(useHardwareSelectionRing(true, SelectionPreviewMode::Rings, 20));
    }

} // namespace
