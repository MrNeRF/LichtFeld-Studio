/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/injected_pointer.hpp"

#include <SDL3/SDL_timer.h>
#include <deque>
#include <mutex>

namespace lfs::vis::input {
    namespace {
        std::mutex state_mutex;
        InjectedPointerState state;
        bool active = false;
        bool positioned = false;
        enum class PendingKind { Move,
                                 ButtonDown,
                                 ButtonUp,
                                 Wheel };

        struct PendingState {
            PendingKind kind;
            InjectedPointerState pointer;
            Uint64 timestamp = 0;
        };

        std::deque<PendingState> pending;
        SDL_Keymod injected_modifiers = SDL_KMOD_NONE;
        SDL_Keymod previous_modifiers = SDL_KMOD_NONE;
        bool modifiers_active = false;

        SDL_MouseButtonFlags buttonMask(const int button) {
            switch (button) {
            case SDL_BUTTON_LEFT: return SDL_BUTTON_LMASK;
            case SDL_BUTTON_RIGHT: return SDL_BUTTON_RMASK;
            case SDL_BUTTON_MIDDLE: return SDL_BUTTON_MMASK;
            default: return 0;
            }
        }

        bool matches(const PendingKind kind, const Uint32 event_type) {
            if (kind == PendingKind::Move)
                return event_type == SDL_EVENT_MOUSE_MOTION;
            if (kind == PendingKind::ButtonDown)
                return event_type == SDL_EVENT_MOUSE_BUTTON_DOWN;
            if (kind == PendingKind::ButtonUp)
                return event_type == SDL_EVENT_MOUSE_BUTTON_UP;
            return event_type == SDL_EVENT_MOUSE_WHEEL || event_type == SDL_EVENT_PINCH_UPDATE;
        }
    } // namespace

    void injectPointerMove(const float x, const float y) {
        std::lock_guard lock(state_mutex);
        state.x = x;
        state.y = y;
        positioned = true;
        active = true;
        pending.push_back({PendingKind::Move, state});
    }

    void injectPointerButton(const int sdl_button, const bool down) {
        std::lock_guard lock(state_mutex);
        const auto mask = buttonMask(sdl_button);
        if (down)
            state.buttons |= mask;
        else
            state.buttons &= ~mask;
        active = true;
        pending.push_back({down ? PendingKind::ButtonDown : PendingKind::ButtonUp, state});
    }

    void injectPointerWheel() {
        std::lock_guard lock(state_mutex);
        active = true;
        pending.push_back({PendingKind::Wheel, state});
    }

    void injectPointerModifiers(const SDL_Keymod modifiers) {
        std::lock_guard lock(state_mutex);
        if (!modifiers_active)
            previous_modifiers = SDL_GetModState();
        injected_modifiers = modifiers;
        modifiers_active = true;
    }

    void pushInjectedPointerEvent(SDL_Event& event) {
        {
            std::lock_guard lock(state_mutex);
            event.common.timestamp = SDL_GetTicksNS();
            if (!pending.empty())
                pending.back().timestamp = event.common.timestamp;
        }
        SDL_PushEvent(&event);
    }

    bool prepareInjectedPointerEvent(const SDL_Event& event) {
        std::lock_guard lock(state_mutex);
        const bool pointer_event = event.type == SDL_EVENT_MOUSE_MOTION ||
                                   event.type == SDL_EVENT_MOUSE_BUTTON_DOWN ||
                                   event.type == SDL_EVENT_MOUSE_BUTTON_UP ||
                                   event.type == SDL_EVENT_MOUSE_WHEEL ||
                                   event.type == SDL_EVENT_PINCH_BEGIN ||
                                   event.type == SDL_EVENT_PINCH_UPDATE ||
                                   event.type == SDL_EVENT_PINCH_END;
        if (pending.empty() || !matches(pending.front().kind, event.type) ||
            pending.front().timestamp != event.common.timestamp)
            return !active || !pointer_event;
        state = pending.front().pointer;
        pending.pop_front();
        active = true;
        if (modifiers_active)
            SDL_SetModState(injected_modifiers);
        return true;
    }

    void finishInjectedPointerFrame() {
        std::lock_guard lock(state_mutex);
        if (pending.empty() && state.buttons == 0) {
            active = false;
            if (modifiers_active) {
                SDL_SetModState(previous_modifiers);
                modifiers_active = false;
                injected_modifiers = SDL_KMOD_NONE;
            }
        }
    }

    std::optional<InjectedPointerState> injectedPointer() {
        std::lock_guard lock(state_mutex);
        return active ? std::optional(state) : std::nullopt;
    }

    std::optional<InjectedPointerState> lastInjectedPointer() {
        std::lock_guard lock(state_mutex);
        return positioned ? std::optional(state) : std::nullopt;
    }

} // namespace lfs::vis::input
