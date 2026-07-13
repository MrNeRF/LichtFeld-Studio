/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rml_tooltip.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "input/input_bindings.hpp"
#include "python/python_runtime.hpp"

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/ElementDocument.h>
#include <algorithm>
#include <cmath>
#include <format>
#include <utility>

namespace lfs::vis::gui {

    namespace {
        std::string actionShortcut(std::string_view action_name) {
            if (action_name.empty())
                return {};
            const auto action = lfs::vis::input::actionFromName(action_name);
            if (!action)
                return {};
            const auto* const bindings = lfs::python::get_keymap_bindings();
            if (!bindings || !bindings->getEffectiveTriggerForAction(*action))
                return {};
            return bindings->getLocalizedTriggerDescription(*action);
        }

        std::string appendShortcut(Rml::Element* el, std::string text) {
            auto shortcut = el->GetAttribute<Rml::String>("data-shortcut", "");
            if (shortcut.empty())
                shortcut = actionShortcut(el->GetAttribute<Rml::String>("data-action", ""));
            if (!shortcut.empty())
                text.append(" (").append(shortcut).append(")");
            return text;
        }

        Rml::String trimmedTooltipKey(Rml::String key) {
            const auto first = key.find_first_not_of(" \t\r\n");
            if (first == Rml::String::npos)
                return {};
            const auto last = key.find_last_not_of(" \t\r\n");
            return key.substr(first, last - first + 1);
        }

    } // namespace

    // A native <select> marks its selectbox/value/arrow ':checked' while its
    // popup is open. Used to suppress tooltips over the popup.
    bool rmlHoverInsideOpenDropdown(Rml::Element* hover) {
        for (auto* el = hover; el; el = el->GetParentNode()) {
            const auto tag = el->GetTagName();
            if ((tag == "selectbox" || tag == "selectvalue" || tag == "selectarrow") &&
                el->IsPseudoClassSet("checked"))
                return true;
        }
        return false;
    }

    std::string resolveRmlTooltip(Rml::Element* hover) {
        if (rmlHoverInsideOpenDropdown(hover))
            return {};
        auto& loc = lfs::event::LocalizationManager::getInstance();
        for (auto* el = hover; el; el = el->GetParentNode()) {
            const auto title = el->GetAttribute<Rml::String>("title", "");
            if (auto key = trimmedTooltipKey(el->GetAttribute<Rml::String>("data-tooltip", ""));
                !key.empty()) {
                const char* const resolved = loc.get(key);
                if (resolved && resolved != key)
                    return appendShortcut(el, resolved);
            }
            if (!title.empty())
                return appendShortcut(el, title);
        }
        return {};
    }

    void RmlTooltipController::setHover(const std::string& text, const void* target) {
        if (text.empty() || !target) {
            pending_text_.clear();
            pending_target_ = nullptr;
            hover_started_at_ = {};
            return;
        }
        if (pending_target_ != target || pending_text_ != text) {
            pending_text_ = text;
            pending_target_ = target;
            const auto now = std::chrono::steady_clock::now();
            hover_started_at_ = now;
            const bool recently_active =
                visible_ ||
                (last_hidden_at_ != std::chrono::steady_clock::time_point{} &&
                 now - last_hidden_at_ < kRmlTooltipReshowGrace);
            effective_delay_ = recently_active ? kRmlTooltipReshowDelay : kRmlTooltipShowDelay;
        }
    }

    bool RmlTooltipController::apply(Rml::Element* body, const int mouse_x, const int mouse_y,
                                     const int doc_w, const int doc_h) {
        if (!body)
            return false;

        auto* doc = body->GetOwnerDocument();
        auto* tooltip_el = doc ? doc->GetElementById("frame-tooltip") : nullptr;

        const auto now = std::chrono::steady_clock::now();
        const bool should_show =
            !pending_text_.empty() && pending_target_ &&
            hover_started_at_ != std::chrono::steady_clock::time_point{} &&
            now - hover_started_at_ >= effective_delay_;

        if (!should_show) {
            if (!visible_)
                return false;
            if (tooltip_el)
                tooltip_el->SetClass("visible", false);
            visible_ = false;
            applied_text_.clear();
            last_hidden_at_ = now;
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

        // Anchor like a DCC tooltip: once shown it stays where it first
        // appeared and does not chase the cursor. Only (re)place it on the
        // reveal frame or when the text changes.
        if (visible_ && !text_changed)
            return false;

        if (text_changed)
            tooltip_el->SetInnerRML(Rml::String(pending_text_));

        // A freshly-shown or text-changed tooltip has no laid-out size yet;
        // park it offscreen and force a layout pass so we can measure it before
        // placing it. Without this, a long tooltip near the right edge shows for
        // one frame clipped before any clamp could take effect.
        if (!visible_)
            tooltip_el->SetClass("visible", true);
        tooltip_el->SetProperty("left", "-10000px");
        tooltip_el->SetProperty("top", "-10000px");
        if (auto* ctx = doc->GetContext())
            ctx->Update();

        const float tt_w = tooltip_el->GetOffsetWidth();
        const float tt_h = tooltip_el->GetOffsetHeight();
        constexpr float gap_x = 16.0f;
        constexpr float gap_below = 22.0f;
        constexpr float gap_above = 10.0f;
        constexpr float edge_margin = 4.0f;
        const float fmouse_x = static_cast<float>(mouse_x);
        const float fmouse_y = static_cast<float>(mouse_y);
        const float fdoc_w = static_cast<float>(doc_w);
        const float fdoc_h = static_cast<float>(doc_h);

        float x = fmouse_x + gap_x;
        if (x + tt_w + edge_margin > fdoc_w)
            x = std::max(edge_margin, fmouse_x - gap_x - tt_w);
        x = std::clamp(x, edge_margin, std::max(edge_margin, fdoc_w - tt_w - edge_margin));

        float y = fmouse_y + gap_below;
        if (y + tt_h + edge_margin > fdoc_h)
            y = std::max(edge_margin, fmouse_y - gap_above - tt_h);
        y = std::clamp(y, edge_margin, std::max(edge_margin, fdoc_h - tt_h - edge_margin));

        tooltip_el->SetProperty("left", std::format("{:.0f}px", x));
        tooltip_el->SetProperty("top", std::format("{:.0f}px", y));

        visible_ = true;
        applied_text_ = pending_text_;
        return true;
    }

} // namespace lfs::vis::gui
