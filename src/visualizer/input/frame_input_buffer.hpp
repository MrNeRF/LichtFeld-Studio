/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_video.h>
#include <cassert>
#include <chrono>
#include <string>
#include <vector>

namespace lfs::vis {

    struct FrameMouseButtonEvent {
        uint8_t button = 0;
        bool down = false;
        float x = 0.0f;
        float y = 0.0f;
        uint64_t timestamp = 0;
        uint8_t clicks = 0;
        // Did the GUI own THIS press when it happened, from the GUI's own hit
        // test (GuiManager::pressBelongsToGui), recorded at the SDL event by
        // the window layer through notePressOwner()?
        //
        // Rectangle containment recomputed later in the GUI frame answers
        // "where is this point now", not "would the GUI have taken this press",
        // and the layout it is compared against may have moved (DPI change,
        // programmatic resize, dock relayout) in between. Recording the verdict
        // ON the event keeps it inseparable from the coordinates it was taken
        // at: they cannot be copied apart.
        //
        // A DOWN carries the verdict for its own press -- EVERY DOWN, not just
        // the frame's first for that button. A matching UP carries the verdict
        // of the DOWN it releases, even when that DOWN was in an earlier frame
        // and even when the release lands outside the pressed control. An UP
        // with no open press of its own, and any event whose verdict was never
        // recorded, stays false: ownership is never inherited from another
        // button or from an earlier same-button press.
        bool gui_owned = false;
    };

    struct FrameInputBuffer {
        uint64_t serial = 0;
        float mouse_x = 0;
        float mouse_y = 0;
        bool mouse_down[3] = {};
        bool mouse_clicked[3] = {};
        bool mouse_released[3] = {};
        float mouse_wheel = 0;
        float mouse_wheel_x = 0;
        std::vector<FrameMouseButtonEvent> mouse_button_events;
        std::vector<SDL_Scancode> keys_pressed;
        std::vector<SDL_Scancode> keys_repeated;
        std::vector<SDL_Scancode> keys_released;
        std::vector<uint32_t> text_codepoints;
        std::vector<std::string> text_inputs;
        std::string text_editing;
        int text_editing_start = -1;
        int text_editing_length = -1;
        bool has_text_editing = false;
        bool had_event = false;
        bool mouse_moved = false;
        bool window_event = false;
        bool user_event = false;
        SDL_Keymod key_mods = SDL_KMOD_NONE;
        int window_w = 0;
        int window_h = 0;
        std::chrono::steady_clock::time_point poll_time{};

        void beginFrame() {
            ++serial;
            mouse_clicked[0] = mouse_clicked[1] = mouse_clicked[2] = false;
            mouse_released[0] = mouse_released[1] = mouse_released[2] = false;
            // The press LIFECYCLE state (press_open_ / press_owner_) is
            // deliberately NOT reset here: a press can be held across many
            // frames, and its release must still find its own DOWN's verdict.
            // Only the index into this frame's now-cleared event vector is.
            pending_owner_index_ = -1;
            mouse_wheel = 0;
            mouse_wheel_x = 0;
            mouse_button_events.clear();
            keys_pressed.clear();
            keys_repeated.clear();
            keys_released.clear();
            text_codepoints.clear();
            text_inputs.clear();
            text_editing.clear();
            text_editing_start = -1;
            text_editing_length = -1;
            has_text_editing = false;
            had_event = false;
            mouse_moved = false;
            window_event = false;
            user_event = false;
        }

        void processEvent(const SDL_Event& event, const SDL_WindowID target_window_id = 0) {
            if (!matchesWindow(event, target_window_id))
                return;

            had_event = true;
            if (event.type >= SDL_EVENT_WINDOW_FIRST && event.type <= SDL_EVENT_WINDOW_LAST)
                window_event = true;
            else if (event.type == SDL_EVENT_USER)
                user_event = true;

            switch (event.type) {
            case SDL_EVENT_MOUSE_MOTION:
                mouse_moved = true;
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
            case SDL_EVENT_MOUSE_BUTTON_UP: {
                const int idx = buttonIndex(event.button.button);
                if (idx >= 0) {
                    const bool down = event.type == SDL_EVENT_MOUSE_BUTTON_DOWN;
                    // A release carries the verdict of the press it ends, and
                    // only of that press: with no open press for this button
                    // there is nothing to inherit and the event stays unowned.
                    const bool released_owner = !down && press_open_[idx] && press_owner_[idx];
                    mouse_button_events.push_back({
                        .button = static_cast<uint8_t>(idx),
                        .down = down,
                        .x = event.button.x,
                        .y = event.button.y,
                        .timestamp = event.button.timestamp,
                        .clicks = event.button.clicks,
                        .gui_owned = released_owner,
                    });
                    if (down) {
                        // The verdict itself is the GUI's to give, so the event
                        // this DOWN just recorded is marked as the one awaiting
                        // it; the window layer answers before the next event is
                        // polled (window_manager.cpp). Every DOWN opens its own
                        // press lifecycle -- a second DOWN for the same button
                        // in the same frame replaces the first rather than
                        // being folded into it.
                        pending_owner_index_ = static_cast<int>(mouse_button_events.size()) - 1;
                        press_open_[idx] = true;
                        press_owner_[idx] = false;
                        mouse_clicked[idx] = true;
                    } else {
                        press_open_[idx] = false;
                        press_owner_[idx] = false;
                        mouse_released[idx] = true;
                    }
                }
                break;
            }
            case SDL_EVENT_MOUSE_WHEEL:
                mouse_wheel += event.wheel.y;
                mouse_wheel_x += event.wheel.x;
                break;
            case SDL_EVENT_KEY_DOWN:
                if (event.key.repeat)
                    keys_repeated.push_back(event.key.scancode);
                else
                    keys_pressed.push_back(event.key.scancode);
                break;
            case SDL_EVENT_KEY_UP:
                keys_released.push_back(event.key.scancode);
                break;
            case SDL_EVENT_TEXT_INPUT:
                if (event.text.text)
                    text_inputs.emplace_back(event.text.text);
                decodeUtf8(event.text.text, text_codepoints);
                break;
            case SDL_EVENT_TEXT_EDITING:
                text_editing = event.edit.text ? event.edit.text : "";
                text_editing_start = event.edit.start;
                text_editing_length = event.edit.length;
                has_text_editing = true;
                break;
            default:
                break;
            }
        }

        // Record who owned the DOWN that processEvent() has JUST recorded for
        // `sdl_button`, so the verdict and the coordinates it was taken at are
        // one event and cannot be copied apart.
        //
        // This answers exactly one event: the DOWN of the SDL event currently
        // being polled. It is a no-op when the buffer recorded no DOWN for that
        // event (a foreign window, an unsupported button) and when a verdict for
        // it has already been given, so a stray or duplicated call can never
        // re-own an earlier press. Deliberately synchronous: nothing between the
        // press and the GUI frame can move the layout out from under it.
        void notePressOwner(const int sdl_button, const bool gui_owned) {
            const int idx = buttonIndex(sdl_button);
            if (idx < 0 || pending_owner_index_ < 0 ||
                static_cast<size_t>(pending_owner_index_) >= mouse_button_events.size())
                return;
            auto& recorded = mouse_button_events[static_cast<size_t>(pending_owner_index_)];
            pending_owner_index_ = -1;
            if (!recorded.down || recorded.button != static_cast<uint8_t>(idx))
                return;
            recorded.gui_owned = gui_owned;
            // ...and the same verdict is what this press's release will carry,
            // however many frames later it arrives.
            press_owner_[idx] = gui_owned;
        }

        void finalize(SDL_Window* window) {
            assert(window);
            poll_time = std::chrono::steady_clock::now();
            const SDL_MouseButtonFlags buttons = SDL_GetMouseState(&mouse_x, &mouse_y);
            mouse_down[0] = (buttons & SDL_BUTTON_LMASK) != 0;
            mouse_down[1] = (buttons & SDL_BUTTON_RMASK) != 0;
            mouse_down[2] = (buttons & SDL_BUTTON_MMASK) != 0;
            key_mods = SDL_GetModState();
            int w = 0, h = 0;
            SDL_GetWindowSize(window, &w, &h);
            window_w = w;
            window_h = h;
        }

    private:
        // Index, in this frame's mouse_button_events, of the DOWN whose owner
        // has not been given yet. -1 when there is none. Cleared by beginFrame()
        // with the vector it points into, and consumed by the first
        // notePressOwner() after the DOWN was recorded.
        int pending_owner_index_ = -1;

        // The open press lifecycle per button: whether a DOWN is outstanding
        // and the verdict it was given. These OUTLIVE beginFrame() on purpose --
        // a button held across frames must still hand its own verdict to the UP
        // that eventually ends it.
        bool press_open_[3] = {};
        bool press_owner_[3] = {};

        static bool matchesWindow(const SDL_Event& event, const SDL_WindowID target_window_id) {
            if (target_window_id == 0)
                return true;

            if (event.type >= SDL_EVENT_WINDOW_FIRST && event.type <= SDL_EVENT_WINDOW_LAST)
                return event.window.windowID == target_window_id;

            switch (event.type) {
            case SDL_EVENT_MOUSE_MOTION:
                return event.motion.windowID == target_window_id;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
            case SDL_EVENT_MOUSE_BUTTON_UP:
                return event.button.windowID == target_window_id;
            case SDL_EVENT_MOUSE_WHEEL:
                return event.wheel.windowID == target_window_id;
            case SDL_EVENT_KEY_DOWN:
            case SDL_EVENT_KEY_UP:
                return event.key.windowID == target_window_id;
            case SDL_EVENT_TEXT_INPUT:
                return event.text.windowID == target_window_id;
            case SDL_EVENT_TEXT_EDITING:
                return event.edit.windowID == target_window_id;
            case SDL_EVENT_DROP_FILE:
            case SDL_EVENT_DROP_COMPLETE:
                return event.drop.windowID == target_window_id;
            default:
                return true;
            }
        }

        static bool isContinuationByte(const unsigned char c) {
            return (c & 0xC0u) == 0x80u;
        }

        static void decodeUtf8(const char* text, std::vector<uint32_t>& out) {
            if (!text)
                return;
            for (size_t i = 0; text[i] != '\0';) {
                uint32_t cp = 0;
                const auto c = static_cast<unsigned char>(text[i]);
                if (c < 0x80) {
                    cp = c;
                    i += 1;
                } else if ((c & 0xE0u) == 0xC0u) {
                    if (!text[i + 1] ||
                        !isContinuationByte(static_cast<unsigned char>(text[i + 1]))) {
                        i += 1;
                        continue;
                    }
                    cp = (c & 0x1F) << 6;
                    cp |= static_cast<unsigned char>(text[i + 1]) & 0x3F;
                    if (cp < 0x80) {
                        i += 1;
                        continue;
                    }
                    i += 2;
                } else if ((c & 0xF0u) == 0xE0u) {
                    if (!text[i + 1] || !text[i + 2] ||
                        !isContinuationByte(static_cast<unsigned char>(text[i + 1])) ||
                        !isContinuationByte(static_cast<unsigned char>(text[i + 2]))) {
                        i += 1;
                        continue;
                    }
                    cp = (c & 0x0F) << 12;
                    cp |= (static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6;
                    cp |= static_cast<unsigned char>(text[i + 2]) & 0x3F;
                    if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) {
                        i += 1;
                        continue;
                    }
                    i += 3;
                } else if ((c & 0xF8u) == 0xF0u) {
                    if (!text[i + 1] || !text[i + 2] || !text[i + 3] ||
                        !isContinuationByte(static_cast<unsigned char>(text[i + 1])) ||
                        !isContinuationByte(static_cast<unsigned char>(text[i + 2])) ||
                        !isContinuationByte(static_cast<unsigned char>(text[i + 3]))) {
                        i += 1;
                        continue;
                    }
                    cp = (c & 0x07) << 18;
                    cp |= (static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12;
                    cp |= (static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6;
                    cp |= static_cast<unsigned char>(text[i + 3]) & 0x3F;
                    if (cp < 0x10000 || cp > 0x10FFFF) {
                        i += 1;
                        continue;
                    }
                    i += 4;
                } else {
                    i += 1;
                    continue;
                }
                out.push_back(cp);
            }
        }

        static int buttonIndex(int sdl_button) {
            switch (sdl_button) {
            case SDL_BUTTON_LEFT: return 0;
            case SDL_BUTTON_RIGHT: return 1;
            case SDL_BUTTON_MIDDLE: return 2;
            default: return -1;
            }
        }
    };

} // namespace lfs::vis
