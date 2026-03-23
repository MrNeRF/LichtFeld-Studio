/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rml_text_input_handler.hpp"

#include <RmlUi/Core/StringUtilities.h>
#include <RmlUi/Core/TextInputContext.h>

#include <algorithm>
#include <limits>

namespace lfs::vis::gui {

    void RmlTextInputHandler::OnActivate(Rml::TextInputContext* input_context) {
        input_context_ = input_context;
        resetState();
    }

    void RmlTextInputHandler::OnDeactivate(Rml::TextInputContext* input_context) {
        if (input_context_ != input_context)
            return;

        if (composing_)
            cancelComposition();

        input_context_ = nullptr;
        resetState();
    }

    void RmlTextInputHandler::OnDestroy(Rml::TextInputContext* input_context) {
        if (input_context_ != input_context)
            return;

        if (composing_)
            cancelComposition();

        input_context_ = nullptr;
        resetState();
    }

    bool RmlTextInputHandler::handleKeyDown(const Rml::Input::KeyIdentifier key_identifier,
                                            const int modifiers) {
        if (!input_context_ || composing_)
            return false;

        const bool ctrl_pressed = (modifiers & Rml::Input::KM_CTRL) != 0;
        const bool alt_pressed = (modifiers & Rml::Input::KM_ALT) != 0;
        if (key_identifier != Rml::Input::KI_A || !ctrl_pressed || alt_pressed)
            return false;

        input_context_->SetSelectionRange(0, std::numeric_limits<int>::max());
        return true;
    }

    bool RmlTextInputHandler::handleTextEditing(const std::string_view composition,
                                                const int cursor_start,
                                                const int selection_length) {
        if (!input_context_)
            return false;

        if (composition.empty()) {
            if (!composing_)
                return false;

            cancelComposition();
            return true;
        }

        if (!composing_)
            composing_ = true;

        cursor_start_ = cursor_start;
        selection_length_ = selection_length;
        setCompositionString(composition);
        updateSelection();
        input_context_->SetCompositionRange(composition_range_start_, composition_range_end_);
        return true;
    }

    bool RmlTextInputHandler::handleTextInput(const std::string_view text) {
        if (!input_context_ || !composing_)
            return false;

        if (text.empty()) {
            cancelComposition();
            return true;
        }

        setCompositionString(text);
        input_context_->SetCompositionRange(composition_range_start_, composition_range_end_);
        input_context_->CommitComposition(Rml::String(text));
        input_context_->SetCursorPosition(composition_range_end_);
        endComposition();
        return true;
    }

    void RmlTextInputHandler::resetState() {
        composing_ = false;
        cursor_start_ = -1;
        selection_length_ = -1;
        composition_range_start_ = 0;
        composition_range_end_ = 0;
    }

    void RmlTextInputHandler::cancelComposition() {
        if (!input_context_ || !composing_) {
            resetState();
            return;
        }

        input_context_->SetText(Rml::StringView(), composition_range_start_, composition_range_end_);
        input_context_->SetCursorPosition(composition_range_start_);
        endComposition();
    }

    void RmlTextInputHandler::endComposition() {
        if (input_context_)
            input_context_->SetCompositionRange(0, 0);
        resetState();
    }

    void RmlTextInputHandler::setCompositionString(const std::string_view composition) {
        if (!input_context_)
            return;

        if (composition_range_start_ == 0 && composition_range_end_ == 0)
            input_context_->GetSelectionRange(composition_range_start_, composition_range_end_);

        const Rml::String value(composition);
        input_context_->SetText(value, composition_range_start_, composition_range_end_);
        composition_range_end_ =
            composition_range_start_ + static_cast<int>(Rml::StringUtilities::LengthUTF8(value));
    }

    void RmlTextInputHandler::updateSelection() {
        if (!input_context_)
            return;

        if (cursor_start_ < 0) {
            input_context_->SetSelectionRange(composition_range_start_, composition_range_end_);
            return;
        }

        const int selection_start = std::clamp(composition_range_start_ + cursor_start_,
                                               composition_range_start_, composition_range_end_);
        const int selection_end = std::clamp(selection_start + std::max(selection_length_, 0),
                                             composition_range_start_, composition_range_end_);
        input_context_->SetCursorPosition(selection_start);
        input_context_->SetSelectionRange(selection_start, selection_end);
    }

} // namespace lfs::vis::gui
