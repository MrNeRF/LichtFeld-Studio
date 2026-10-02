/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "sdl_coordinate_utils.hpp"
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_video.h>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lfs::vis {

    enum class FrameInputEventKind { KeyDown,
                                     KeyUp,
                                     Text,
                                     TextEditing,
                                     MouseButton };

    struct InputEventDispatch {
        bool consumed = false;
    };

    struct FrameInputEvent {
        FrameInputEventKind kind = FrameInputEventKind::KeyDown;
        SDL_Scancode scancode = SDL_SCANCODE_UNKNOWN;
        SDL_Keymod modifiers = SDL_KMOD_NONE;
        bool repeat = false;
        std::string text{};
        int editing_start = -1;
        int editing_length = -1;
        size_t mouse_button_index = 0;
        // Shared by transient panel adapters during one global dispatch.
        InputEventDispatch* dispatch = nullptr;
    };

    struct FrameMouseButtonEvent {
        uint8_t button = 0;
        bool down = false;
        float x = 0.0f;
        float y = 0.0f;
        uint64_t timestamp = 0;
        uint8_t clicks = 0;
        bool ctrl = false;
        // GUI ownership from GuiManager::hitTestMouseButton, recorded by the window
        // layer at the SDL event through notePressOwner(). Keep it with event coordinates:
        // later bounds checks cannot recover ownership after DPI, resize or dock changes.
        // Every DOWN carries its own verdict. Its matching UP carries the same verdict,
        // even across frames or outside the pressed control. An unmatched UP or an event
        // with no recorded verdict stays false; never borrow ownership from another
        // button or an earlier same-button press.
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
        std::vector<FrameInputEvent> input_events;
        // Aggregate key state for shortcuts; text consumers replay input_events only.
        std::vector<SDL_Scancode> keys_pressed;

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
            // Keep press_open_ / press_owner_ across frames so UP retains its DOWN's verdict.
            // Reset only the index into the event vector being cleared.
            pending_owner_index_ = -1;
            if (!event_mods_initialized_) {
                event_mods_ = SDL_GetModState();
                event_mods_initialized_ = true;
            }
            mouse_wheel = 0;
            mouse_wheel_x = 0;
            mouse_button_events.clear();
            input_events.clear();
            keys_pressed.clear();

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
                    // UP inherits only this button's open press verdict; without one, it stays unowned.
                    const bool released_owner = !down && press_open_[idx] && press_owner_[idx];
                    mouse_button_events.push_back({
                        .button = static_cast<uint8_t>(idx),
                        .down = down,
                        .x = event.button.x,
                        .y = event.button.y,
                        .timestamp = event.button.timestamp,
                        .clicks = event.button.clicks,
                        .ctrl = (event_mods_ & SDL_KMOD_CTRL) != 0,
                        .gui_owned = released_owner,
                    });
                    input_events.push_back({.kind = FrameInputEventKind::MouseButton,
                                            .mouse_button_index = mouse_button_events.size() - 1});
                    if (down) {
                        // Mark this DOWN as awaiting the GUI verdict; window_manager.cpp supplies it
                        // before polling the next event. Every DOWN starts a new press lifecycle,
                        // replacing any earlier same-button press, even within this frame.
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
                event_mods_ = event.key.mod;
                input_events.push_back({.kind = FrameInputEventKind::KeyDown, .scancode = event.key.scancode, .modifiers = event.key.mod, .repeat = event.key.repeat});
                if (!event.key.repeat)
                    keys_pressed.push_back(event.key.scancode);
                break;
            case SDL_EVENT_KEY_UP:
                event_mods_ = event.key.mod;
                input_events.push_back({.kind = FrameInputEventKind::KeyUp, .scancode = event.key.scancode, .modifiers = event.key.mod});

                break;
            case SDL_EVENT_TEXT_INPUT:
                if (event.text.text)
                    input_events.push_back({.kind = FrameInputEventKind::Text, .text = event.text.text});
                break;
            case SDL_EVENT_TEXT_EDITING:
                input_events.push_back({.kind = FrameInputEventKind::TextEditing,
                                        .text = event.edit.text ? event.edit.text : "",
                                        .editing_start = event.edit.start,
                                        .editing_length = event.edit.length});
                break;
            default:
                break;
            }
        }

        // Record the current SDL DOWN's ownership beside its coordinates.
        // Call immediately after processEvent, before another event or layout change.
        // With that ordering, foreign-window, unsupported-button and duplicate calls
        // are no-ops; only the pending matching DOWN receives a verdict.
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
            // Keep this verdict for the matching UP, even across frames.
            press_owner_[idx] = gui_owned;
        }

        void finalize(SDL_Window* window) {
            assert(window);
            poll_time = std::chrono::steady_clock::now();
            const SDL_MouseButtonFlags buttons = input::mouseStateInPixels(window, &mouse_x, &mouse_y);
            mouse_down[0] = (buttons & SDL_BUTTON_LMASK) != 0;
            mouse_down[1] = (buttons & SDL_BUTTON_RMASK) != 0;
            mouse_down[2] = (buttons & SDL_BUTTON_MMASK) != 0;
            key_mods = SDL_GetModState();
            int w = 0, h = 0;
            SDL_GetWindowSizeInPixels(window, &w, &h);
            const auto scale = input::windowPixelScale(window);
            for (auto& event : mouse_button_events) {
                event.x *= scale.x;
                event.y *= scale.y;
            }
            window_w = w;
            window_h = h;
        }

    private:
        // Index of the DOWN awaiting ownership in this frame's mouse_button_events,
        // or -1 if none. beginFrame() clears it with the vector; notePressOwner()
        // consumes it.
        int pending_owner_index_ = -1;
        SDL_Keymod event_mods_ = SDL_KMOD_NONE;
        bool event_mods_initialized_ = false;

        // Per-button open DOWN and ownership verdict. Preserve across beginFrame()
        // so the matching UP inherits its press's verdict, even in a later frame.
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
