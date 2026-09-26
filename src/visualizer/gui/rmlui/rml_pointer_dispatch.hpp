/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "input/frame_input_buffer.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Context.h>
#include <cstddef>
#include <glm/glm.hpp>
#include <vector>

namespace lfs::vis::gui::rml_input {

    // Frame-buffered pointer delivery into RmlUi.
    // RmlUi 6.2 takes DOWN's active element from the last mouse move's hover.
    // UP clears active/drag and clicks only if active == FindFocusElement(hover).
    // Move to each event's coordinates before its DOWN/UP, in SDL order.
    //
    // Walk the canonical stream once without coalescing, reordering or truncating.
    // Skip unowned presses without capture and releases with no delivered DOWN.
    // Keep repeated or different-button presses separate, at their own points;
    // a misplaced move after DOWN can start a drag the user never began.
    //
    // down_delivered tracks each button across frames. Deliver an owned UP even
    // outside the context, or RmlUi can remain pressed after DOWN armed active/drag.
    //
    // The host owns the state and ownership predicate. This helper lets tests
    // replay the overlay's delivery loop against their own context.

    // Replay `events` into `context`.
    // `down_delivered`: host-owned per-button state, read and updated here. Keep one
    //   array per context; clear the whole array only when the context is destroyed.
    // `viewport_pos` / `viewport_size`: context rectangle in window coordinates,
    //   as are event coordinates.
    // `capture_active`: an owned drag (VRAM HUD or toolbar) accepts presses anywhere
    //   when `allow_new_presses` is true.
    // `owns_element`: host hit predicate; receives nullptr outside the rectangle.
    // `allow_new_presses`: false when the underlay is blocked. Still process every
    //   event: replacement primary DOWN cancels the old press; owned UP uses its real point.
    // Returns whether a DOWN/UP was delivered; cancellation alone returns false.
    template <typename OwnsElementFn>
    [[nodiscard]] inline bool replayButtonEvents(Rml::Context& context,
                                                 bool (&down_delivered)[3],
                                                 const std::vector<FrameMouseButtonEvent>& events,
                                                 const glm::vec2 viewport_pos,
                                                 const glm::vec2 viewport_size,
                                                 const int mods,
                                                 const bool capture_active,
                                                 OwnsElementFn owns_element,
                                                 const bool allow_new_presses = true) {
        bool replayed = false;
        for (const auto& event : events) {
            if (event.button >= 3)
                continue;
            const auto slot = static_cast<std::size_t>(event.button);
            if (event.down && down_delivered[slot]) {
                down_delivered[slot] = false;
                context.ProcessMouseButtonCancel(event.button, mods);
            }

            const float event_x = event.x - viewport_pos.x;
            const float event_y = event.y - viewport_pos.y;
            const bool event_inside = event_x >= 0.0f && event_x < viewport_size.x &&
                                      event_y >= 0.0f && event_y < viewport_size.y;
            const auto* const event_element =
                event_inside ? context.GetElementAtPoint(Rml::Vector2f(event_x, event_y))
                             : nullptr;
            // DOWN requires new presses to be allowed and either capture or a hit we own.
            // UP requires this button's delivered DOWN, regardless of release location.
            // Per-button state and refused-DOWN clearing prevent borrowing ownership from
            // another button, an earlier press or a press this host never delivered.
            const bool deliver = event.down ? (allow_new_presses && (capture_active || owns_element(event_element)))
                                            : down_delivered[slot];
            if (!deliver) {
                // A refused DOWN clears this button's ownership so its UP cannot borrow
                // delivery rights from an earlier press whose UP never reached this host.
                if (event.down)
                    down_delivered[slot] = false;
                continue;
            }
            context.ProcessMouseMove(static_cast<int>(event_x), static_cast<int>(event_y), mods);
            down_delivered[slot] = event.down;
            if (event.down)
                context.ProcessMouseButtonDown(event.button, mods);
            else
                context.ProcessMouseButtonUp(event.button, mods);
            replayed = true;
        }
        return replayed;
    }

} // namespace lfs::vis::gui::rml_input
