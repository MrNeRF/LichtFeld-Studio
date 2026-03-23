/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rml_text_input_handler.hpp"

#include <RmlUi/Core/Input.h>
#include <RmlUi/Core/Rectangle.h>
#include <gtest/gtest.h>
#include <limits>

namespace {

    class StubTextInputContext final : public Rml::TextInputContext {
    public:
        bool GetBoundingBox(Rml::Rectanglef& out_rectangle) const override {
            out_rectangle = {};
            return true;
        }

        void GetSelectionRange(int& start, int& end) const override {
            start = selection_start;
            end = selection_end;
        }

        void SetSelectionRange(int start, int end) override {
            selection_start = start;
            selection_end = end;
        }

        void SetCursorPosition(int position) override { cursor_position = position; }

        void SetText(Rml::StringView, int, int) override {}
        void SetCompositionRange(int start, int end) override {
            composition_start = start;
            composition_end = end;
        }
        void CommitComposition(Rml::StringView) override {}

        int selection_start = -1;
        int selection_end = -1;
        int cursor_position = -1;
        int composition_start = -1;
        int composition_end = -1;
    };

} // namespace

TEST(RmlTextInputHandlerTest, CtrlASelectsWholeInput) {
    lfs::vis::gui::RmlTextInputHandler handler;
    StubTextInputContext context;

    handler.OnActivate(&context);

    const bool handled = handler.handleKeyDown(Rml::Input::KI_A, Rml::Input::KM_CTRL);

    EXPECT_TRUE(handled);
    EXPECT_EQ(context.selection_start, 0);
    EXPECT_EQ(context.selection_end, std::numeric_limits<int>::max());
}

TEST(RmlTextInputHandlerTest, IgnoresPlainAWithoutModifier) {
    lfs::vis::gui::RmlTextInputHandler handler;
    StubTextInputContext context;

    handler.OnActivate(&context);

    const bool handled = handler.handleKeyDown(Rml::Input::KI_A, 0);

    EXPECT_FALSE(handled);
    EXPECT_EQ(context.selection_start, -1);
    EXPECT_EQ(context.selection_end, -1);
}
