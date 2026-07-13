/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <RmlUi/Core/Element.h>
#include <chrono>
#include <optional>
#include <string>
#include <string_view>

namespace Rml {
    class ElementDocument;
}

namespace lfs::vis::gui {

    inline constexpr auto kRmlTooltipShowDelay = std::chrono::milliseconds(500);

    // Once a tooltip has been visible, hovering an adjacent control shows the
    // next one almost immediately (like a DCC toolbar) instead of re-incurring
    // the full hover delay. The grace is the window after a tooltip hides in
    // which the reduced delay still applies.
    inline constexpr auto kRmlTooltipReshowDelay = std::chrono::milliseconds(90);
    inline constexpr auto kRmlTooltipReshowGrace = std::chrono::milliseconds(600);

    LFS_VIS_API std::string resolveRmlTooltip(Rml::Element* hover);

    // True while the hovered element is (inside) a native <select> whose popup
    // is open.
    LFS_VIS_API bool rmlHoverInsideOpenDropdown(Rml::Element* hover);

    // Per-document tooltip state. Each renderer owns one instance and drives it
    // from its own input/render passes, so the tooltip element lives inside the
    // hovered context and is positioned in that context's local coordinates.
    class RmlTooltipController {
    public:
        // Called from input when the hovered tooltip target changes. Pass
        // {} / nullptr when no tooltip should be shown.
        void setHover(const std::string& text, const void* target);

        // Called once per render pass. Creates a `frame-tooltip` div under
        // `body` if it does not exist, then positions / shows / hides it.
        // mouse_x/y are document-local coordinates; doc_w/h size the clamp.
        // Returns true if the document changed and needs a fresh paint.
        bool apply(Rml::Element* body, int mouse_x, int mouse_y,
                   int doc_w, int doc_h);
        [[nodiscard]] bool hasActiveState() const {
            return visible_ || pending_target_ != nullptr || !pending_text_.empty();
        }
        [[nodiscard]] bool needsFrame() const {
            return pending_target_ != nullptr && !pending_text_.empty() && !visible_;
        }

        // Absolute time at which a pending tooltip should first appear, or empty
        // when nothing is counting down. Lets the render loop sleep through the
        // hover delay and wake exactly once to paint the tooltip.
        [[nodiscard]] std::optional<std::chrono::steady_clock::time_point> revealDeadline() const {
            if (!needsFrame() || hover_started_at_ == std::chrono::steady_clock::time_point{})
                return std::nullopt;
            return hover_started_at_ + effective_delay_;
        }

        // True only once the hover delay has elapsed and the tooltip still needs
        // to be shown; this is what drives a single reveal frame.
        [[nodiscard]] bool revealDue() const {
            const auto deadline = revealDeadline();
            return deadline && std::chrono::steady_clock::now() >= *deadline;
        }

    private:
        std::string pending_text_;
        const void* pending_target_ = nullptr;
        std::chrono::steady_clock::time_point hover_started_at_{};

        bool visible_ = false;
        std::string applied_text_;
        std::chrono::steady_clock::time_point last_hidden_at_{};
        std::chrono::milliseconds effective_delay_ = kRmlTooltipShowDelay;
    };

} // namespace lfs::vis::gui
