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

    // --------------------------------------------------------------------
    // Frame-buffered pointer delivery into an RmlUi context.
    //
    // The host buffers a whole SDL frame before it talks to RmlUi, so a press
    // and the motion queued behind it arrive together. RmlUi 6.2 decides the
    // pressed element from its LAST ProcessMouseMove -- `hover` is recomputed
    // as GetElementAtPoint(mouse_position) inside UpdateHoverChain
    // (Context.cpp:1301), ProcessMouseButtonDown takes `active = hover`
    // (Context.cpp:617-641), and ProcessMouseButtonUp fires Click only while
    // `active == FindFocusElement(hover)` (Context.cpp:721-745). So every
    // transition in the frame must be delivered at ITS OWN point: a move to
    // that event's coordinates, then that event's DOWN or UP, in the order SDL
    // recorded them.
    //
    // THE CANONICAL STREAM IS NEVER COALESCED, REORDERED OR TRUNCATED. This
    // walks `events` front to back exactly once. A PRESS this host does not
    // own -- one that landed outside the context, or on nothing of ours, while
    // no capture is in force -- is SKIPPED, and so is a release with no press
    // of this host's to end; skipping changes neither the order nor the
    // identity of the events that are delivered. Two buttons
    // going down in one frame, or the same button going down twice, therefore
    // reach RmlUi as two separate presses at two separate points, which is
    // what stops one press from being delivered at another's coordinates and
    // what stops a manufactured move after a DOWN from starting a drag the
    // user never began (Context.cpp:625 arms `drag`, Context.cpp:1277 fires
    // Dragstart).
    //
    // A DOWN THIS HOST DELIVERED OWES RMLUI ITS UP. RmlUi is stateful across
    // the pair: ProcessMouseButtonDown arms `active` and `drag`
    // (Context.cpp:617-641) and only ProcessMouseButtonUp disarms them and
    // fires Click (Context.cpp:721-745). A host that delivers a DOWN and then
    // withholds the UP because the release happened to land somewhere else
    // leaves the context pressed for good. So delivery is decided per press
    // lifecycle, not per hit test: `down_delivered` carries, per button,
    // whether THIS host delivered that button's DOWN, and it deliberately
    // OUTLIVES the frame because a button can be held down across many.
    //
    // This lives outside RmlViewportOverlay so the replay is executable
    // against a context the tests own; the host holds the state and supplies
    // the ownership predicate. The body is the overlay's own loop, moved
    // rather than rewritten, plus that lifecycle.
    // --------------------------------------------------------------------

    // Replay `events` into `context`.
    //
    //   `down_delivered`  -- the host's per-button press lifecycle, read AND
    //       written here. The host owns it (one array per context) and clears
    //       it only when the context itself goes away.
    //   `viewport_pos` / `viewport_size` -- the context's rectangle in window
    //       coordinates; event coordinates are window coordinates.
    //   `capture_active`  -- a drag this host already owns (the VRAM HUD, the
    //       toolbar) holds the pointer, so its presses are delivered wherever
    //       they land.
    //   `owns_element`    -- would this host take an event that landed on this
    //       element? Called with nullptr for an event outside the rectangle.
    //   `allow_new_presses` -- false while the underlay is blocked. Still walk
    //       every event: a refused DOWN revokes that button's prior ownership,
    //       while an UP with a still-owned DOWN is delivered at its real point.
    //
    // Returns whether anything was delivered, which is the host's cue to mark
    // a pointer-button repaint. (The overlay marked that reason once per
    // delivered event; the reason is a single bit, so marking it once for the
    // frame is the same repaint.)
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
            const float event_x = event.x - viewport_pos.x;
            const float event_y = event.y - viewport_pos.y;
            const bool event_inside = event_x >= 0.0f && event_x < viewport_size.x &&
                                      event_y >= 0.0f && event_y < viewport_size.y;
            const auto* const event_element =
                event_inside ? context.GetElementAtPoint(Rml::Vector2f(event_x, event_y))
                             : nullptr;
            // A DOWN is judged by the hit test, exactly as the loop this came
            // from judged every event. An UP is judged by ITS OWN DOWN:
            // delivered when this host delivered that DOWN -- wherever the
            // release landed -- and REJECTED when it did not, so an UP can
            // never take ownership from another button, from an earlier
            // same-button press, or from a press this host never delivered.
            const bool deliver = event.down ? (allow_new_presses && (capture_active || owns_element(event_element)))
                                            : down_delivered[slot];
            if (!deliver) {
                // A press this host refused supersedes whatever that button was
                // doing before: the flag is cleared so a stale DOWN (one whose
                // UP never reached this consumer) cannot lend its delivery right
                // to this press's release.
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
