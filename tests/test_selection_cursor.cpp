/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_bindings.hpp"
#include <gtest/gtest.h>

namespace {

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

} // namespace
