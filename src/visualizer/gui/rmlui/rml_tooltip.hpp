/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <RmlUi/Core/Element.h>
#include <chrono>
#include <string>
#include <string_view>

namespace Rml {
    class ElementDocument;
}

namespace lfs::vis::gui {

    inline constexpr auto kRmlTooltipShowDelay = std::chrono::milliseconds(500);

    LFS_VIS_API std::string resolveRmlTooltip(Rml::Element* hover);

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
        [[nodiscard]] bool hasActiveState() const {
            return seen_this_frame_ || visible_ || pending_target_ != nullptr || !pending_text_.empty();
        }
        [[nodiscard]] bool needsFrame() const {
            return pending_target_ != nullptr && !pending_text_.empty() && !visible_;
        }

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
