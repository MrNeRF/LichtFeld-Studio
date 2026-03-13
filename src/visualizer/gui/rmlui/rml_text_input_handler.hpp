/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/Input.h>
#include <RmlUi/Core/TextInputContext.h>
#include <RmlUi/Core/TextInputHandler.h>

#include <string_view>

namespace lfs::vis::gui {

    class LFS_VIS_API RmlTextInputHandler final : public Rml::TextInputHandler {
    public:
        void OnActivate(Rml::TextInputContext* input_context) override;
        void OnDeactivate(Rml::TextInputContext* input_context) override;
        void OnDestroy(Rml::TextInputContext* input_context) override;

        bool handleKeyDown(Rml::Input::KeyIdentifier key_identifier, int modifiers);
        bool handleTextEditing(std::string_view composition, int cursor_start, int selection_length);
        bool handleTextInput(std::string_view text);
        bool isComposing() const { return composing_; }

    private:
        void resetState();
        void cancelComposition();
        void endComposition();
        void setCompositionString(std::string_view composition);
        void updateSelection();

        Rml::TextInputContext* input_context_ = nullptr;
        bool composing_ = false;
        int cursor_start_ = -1;
        int selection_length_ = -1;
        int composition_range_start_ = 0;
        int composition_range_end_ = 0;
    };

} // namespace lfs::vis::gui
