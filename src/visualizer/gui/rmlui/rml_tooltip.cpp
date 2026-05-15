/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rml_tooltip.hpp"

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/ElementDocument.h>
#include <algorithm>
#include <cmath>
#include <format>
#include <utility>

namespace lfs::vis::gui {

    void RmlTooltipController::setHover(const std::string& text, const void* target) {
        seen_this_frame_ = true;
        if (text.empty() || !target) {
            pending_text_.clear();
            pending_target_ = nullptr;
            hover_started_at_ = {};
            return;
        }
        if (pending_target_ != target || pending_text_ != text) {
            pending_text_ = text;
            pending_target_ = target;
            hover_started_at_ = std::chrono::steady_clock::now();
        }
    }

    bool RmlTooltipController::apply(Rml::Element* body, const int mouse_x, const int mouse_y,
                                     const int doc_w, const int doc_h) {
        if (!body)
            return false;

        if (!seen_this_frame_) {
            pending_text_.clear();
            pending_target_ = nullptr;
            hover_started_at_ = {};
        }
        seen_this_frame_ = false;

        auto* doc = body->GetOwnerDocument();
        auto* tooltip_el = doc ? doc->GetElementById("frame-tooltip") : nullptr;

        const auto now = std::chrono::steady_clock::now();
        const bool should_show =
            !pending_text_.empty() && pending_target_ &&
            hover_started_at_ != std::chrono::steady_clock::time_point{} &&
            now - hover_started_at_ >= kRmlTooltipShowDelay;

        if (!should_show) {
            if (!visible_)
                return false;
            if (tooltip_el)
                tooltip_el->SetClass("visible", false);
            visible_ = false;
            applied_text_.clear();
            return true;
        }

        if (!tooltip_el) {
            if (!doc)
                return false;
            auto el_ptr = doc->CreateElement("div");
            if (!el_ptr)
                return false;
            el_ptr->SetId("frame-tooltip");
            el_ptr->SetClass("frame-tooltip", true);
            tooltip_el = body->AppendChild(std::move(el_ptr));
            if (!tooltip_el)
                return false;
        }

        const bool text_changed = pending_text_ != applied_text_;
        if (text_changed)
            tooltip_el->SetInnerRML(Rml::String(pending_text_));

        // Newly-shown or text-changed tooltips have no laid-out size yet; park
        // the element offscreen and force a layout pass so we can measure it
        // before placing it. Without this, a long tooltip near the right edge
        // shows for one frame clipped before any clamp could take effect.
        if (!visible_ || text_changed) {
            if (!visible_)
                tooltip_el->SetClass("visible", true);
            tooltip_el->SetProperty("left", "-10000px");
            tooltip_el->SetProperty("top", "-10000px");
            if (auto* ctx = doc->GetContext())
                ctx->Update();
        }

        const float tt_w = tooltip_el->GetOffsetWidth();
        const float tt_h = tooltip_el->GetOffsetHeight();
        constexpr float pad = 14.0f;
        constexpr float edge_margin = 4.0f;
        const float fmouse_x = static_cast<float>(mouse_x);
        const float fmouse_y = static_cast<float>(mouse_y);
        const float fdoc_w = static_cast<float>(doc_w);
        const float fdoc_h = static_cast<float>(doc_h);

        float x = fmouse_x + pad;
        if (x + tt_w + edge_margin > fdoc_w)
            x = std::max(edge_margin, fmouse_x - pad - tt_w);
        x = std::clamp(x, edge_margin, std::max(edge_margin, fdoc_w - tt_w - edge_margin));

        float y = fmouse_y + pad;
        if (y + tt_h + edge_margin > fdoc_h)
            y = std::max(edge_margin, fmouse_y - pad - tt_h);
        y = std::clamp(y, edge_margin, std::max(edge_margin, fdoc_h - tt_h - edge_margin));

        const bool position_changed = !visible_ || text_changed ||
                                      std::abs(x - applied_x_) > 0.5f ||
                                      std::abs(y - applied_y_) > 0.5f;
        if (!position_changed)
            return false;

        tooltip_el->SetProperty("left", std::format("{:.0f}px", x));
        tooltip_el->SetProperty("top", std::format("{:.0f}px", y));

        visible_ = true;
        applied_text_ = pending_text_;
        applied_x_ = x;
        applied_y_ = y;
        return true;
    }

} // namespace lfs::vis::gui
