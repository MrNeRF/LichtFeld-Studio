/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/event_bridge/localization_manager.hpp"

#include <RmlUi/Core/Element.h>
#include <chrono>
#include <string>
#include <string_view>

namespace Rml {
    class ElementDocument;
}

namespace lfs::vis::gui {

    inline constexpr auto kRmlTooltipShowDelay = std::chrono::milliseconds(500);

    inline std::string resolveRmlTooltip(Rml::Element* hover) {
        for (auto* el = hover; el; el = el->GetParentNode()) {
            const auto key = el->GetAttribute<Rml::String>("data-tooltip", "");
            if (!key.empty()) {
                auto& loc = lfs::event::LocalizationManager::getInstance();
                const char* resolved = loc.get(key.c_str());
                if (!resolved || std::string_view(resolved) == key.c_str())
                    return {};
                return resolved;
            }

            const auto title = el->GetAttribute<Rml::String>("title", "");
            if (!title.empty())
                return std::string(title.c_str(), title.size());
        }
        return {};
    }

    // Per-document tooltip state. Each renderer owns one instance and drives it
    // from its own input/render passes, so the tooltip element lives inside the
    // hovered context and is positioned in that context's local coordinates.
    class RmlTooltipController {
    public:
        // Called once per input pass per element. Pass {} / nullptr when no
        // tooltip should be shown (or skip the call — apply() will also clear
        // state if setHover() was not called this frame).
        void setHover(const std::string& text, const void* target);

        // Called once per render pass. Creates a `frame-tooltip` div under
        // `body` if it does not exist, then positions / shows / hides it.
        // mouse_x/y are document-local coordinates; doc_w/h size the clamp.
        // Returns true if the document changed and needs a fresh paint.
        bool apply(Rml::Element* body, int mouse_x, int mouse_y,
                   int doc_w, int doc_h);

    private:
        std::string pending_text_;
        const void* pending_target_ = nullptr;
        std::chrono::steady_clock::time_point hover_started_at_{};
        bool seen_this_frame_ = false;

        bool visible_ = false;
        std::string applied_text_;
        float applied_x_ = 0.0f;
        float applied_y_ = 0.0f;
    };

} // namespace lfs::vis::gui
