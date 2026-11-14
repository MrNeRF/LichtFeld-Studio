/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "form_control_layout.hpp"
#include <RmlUi/Core/ComputedValues.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <algorithm>
#include <optional>
#include <utility>

namespace lfs::vis::gui {

    class SelectViewportScope {
    public:
        SelectViewportScope(float top, float bottom)
            : previous_(bounds_) {
            bounds_ = std::pair{top, bottom};
        }
        ~SelectViewportScope() { bounds_ = previous_; }

        SelectViewportScope(const SelectViewportScope&) = delete;
        SelectViewportScope& operator=(const SelectViewportScope&) = delete;

        static const auto& bounds() { return bounds_; }

    private:
        inline static thread_local std::optional<std::pair<float, float>> bounds_;
        std::optional<std::pair<float, float>> previous_;
    };

    // Contexts include off-screen shadow padding. Native selects only know the
    // context size, so correct their popup before its children are rendered.
    class ViewportSelect : public FormControlLayout<Rml::ElementFormControlSelect> {
    public:
        explicit ViewportSelect(const Rml::String& tag) : FormControlLayout(tag) {}

    protected:
        void OnRender() override {
            if (constrained_ && (!IsSelectBoxVisible() || constrained_bounds_ != SelectViewportScope::bounds())) {
                if (auto* const option = GetOption(0)) {
                    auto* const box = option->GetParentNode();
                    if (previous_max_height_)
                        box->SetProperty(Rml::PropertyId::MaxHeight, *previous_max_height_);
                    else
                        box->RemoveProperty(Rml::PropertyId::MaxHeight);
                    GetOwnerDocument()->UpdateDocument();
                }
                constrained_ = false;
            }
            Rml::ElementFormControlSelect::OnRender();
            if (!IsSelectBoxVisible() || !SelectViewportScope::bounds() || !GetOption(0))
                return;

            const auto [top, bottom] = *SelectViewportScope::bounds();
            if (bottom <= top)
                return;
            auto* const box = GetOption(0)->GetParentNode();
            const auto position = box->GetAbsoluteOffset(Rml::BoxArea::Border);
            float height = box->GetOffsetHeight();
            if (position.y >= top && position.y + height <= bottom)
                return;

            const auto anchor = GetAbsoluteOffset(Rml::BoxArea::Border);
            const float below = anchor.y + GetOffsetHeight();
            const float space_above = std::max(0.0f, anchor.y - top);
            const float space_below = std::max(0.0f, bottom - below);
            const bool open_above = height <= space_above || space_above > space_below;
            const float available = open_above ? space_above : space_below;
            if (height > available) {
                if (!constrained_) {
                    const auto* const property = box->GetLocalProperty(Rml::PropertyId::MaxHeight);
                    previous_max_height_ = property ? std::optional(*property) : std::nullopt;
                    constrained_ = true;
                    constrained_bounds_ = SelectViewportScope::bounds();
                }
                const float edges = box->GetComputedValues().box_sizing() == Rml::Style::BoxSizing::BorderBox
                                        ? 0.0f
                                        : box->GetBox().GetSizeAcross(
                                              Rml::BoxDirection::Vertical, Rml::BoxArea::Border, Rml::BoxArea::Padding);
                box->SetProperty(Rml::PropertyId::MaxHeight,
                                 Rml::Property(std::max(0.0f, available - edges), Rml::Unit::PX));
                GetOwnerDocument()->UpdateDocument();
                Rml::ElementFormControlSelect::OnRender();
                height = box->GetOffsetHeight();
            }
            const float y = open_above ? anchor.y - height : below;
            box->SetOffset({position.x - anchor.x, y - anchor.y}, this);
        }

    private:
        bool constrained_ = false;
        std::optional<Rml::Property> previous_max_height_;
        std::optional<std::pair<float, float>> constrained_bounds_;
    };

} // namespace lfs::vis::gui
