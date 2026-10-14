/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <RmlUi/Core/ElementUtilities.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <RmlUi/Core/Elements/ElementFormControlTextArea.h>
#include <RmlUi/Core/Math.h>
#include <type_traits>

namespace lfs::vis::gui {

    // Keep LichtFeld's established intrinsic control boxes across the 6.3
    // upgrade. Explicit RCSS dimensions still take precedence. Our styles
    // already scale tracks, thumbs and rows in dp, including DPI-specific
    // offsets, so scaling the default boxes again changes existing layouts.
    template <typename Control>
    class FormControlLayout : public Control {
    public:
        using Control::Control;

        bool GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) override {
            const bool intrinsic = Control::GetIntrinsicDimensions(dimensions, ratio);
            if (!intrinsic)
                return false;

            if constexpr (std::is_same_v<Control, Rml::ElementFormControlSelect>) {
                dimensions = {128.f, 16.f};
            } else if constexpr (std::is_same_v<Control, Rml::ElementFormControlTextArea>) {
                dimensions.y = this->GetNumRows() * Rml::Math::Round(this->GetLineHeight());
            } else if constexpr (std::is_same_v<Control, Rml::ElementFormControlInput>) {
                const auto type = this->template GetAttribute<Rml::String>("type", "text");
                if (type == "range" || type == "checkbox" || type == "radio") {
                    dimensions /= Rml::ElementUtilities::GetDensityIndependentPixelRatio(this);
                } else if (type == "text" || type == "password") {
                    dimensions.y = Rml::Math::Round(this->GetLineHeight());
                }
            }
            return true;
        }
    };

} // namespace lfs::vis::gui
