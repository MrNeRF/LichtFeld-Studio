/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gamepad_service.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <utility>

namespace lfs::vis::input {

    struct GamepadService::Impl {
        struct HandleDeleter {
            void operator()(SDL_Gamepad* pad) const { SDL_CloseGamepad(pad); }
        };
        struct Connection {
            GamepadDevice device;
            std::unique_ptr<SDL_Gamepad, HandleDeleter> handle;
        };
        std::vector<Connection> connections;
        GamepadSnapshot state;
        uint64_t generation = 0;
        bool initialized = false;

        void add(SDL_JoystickID id) {
            const auto found = std::find_if(connections.begin(), connections.end(),
                                            [id](const auto& c) { return c.device.id == id; });
            if (found != connections.end())
                return;
            Connection connection;
            connection.device.id = id;
            connection.device.generation = ++generation;
            if (const auto* name = SDL_GetJoystickNameForID(id))
                connection.device.name = name;
            connection.device.mapped = SDL_IsGamepad(id);
            if (connection.device.mapped) {
                connection.handle.reset(SDL_OpenGamepad(id));
                if (connection.handle) {
                    connection.device.type = SDL_GetGamepadType(connection.handle.get());
                    if (const auto* name = SDL_GetGamepadName(connection.handle.get()))
                        connection.device.name = name;
                } else {
                    connection.device.error = SDL_GetError();
                }
            }
            connections.push_back(std::move(connection));
        }

        void remove(SDL_JoystickID id) {
            if (state.navigation_selected == id) {
                state.navigation_selected = 0;
                state.navigation_sample = {};
            }
            if (state.selected == id) {
                // No automatic fallback to another device, even with identical names.
                state.selected = 0;
                state.sample = {};
            }
            std::erase_if(connections, [id](const auto& c) { return c.device.id == id; });
        }
    };

    GamepadService::GamepadService() : impl_(std::make_unique<Impl>()) {}
    GamepadService::~GamepadService() { shutdown(); }

    bool GamepadService::initialize() {
        if (impl_->initialized)
            return true;
        if (!SDL_InitSubSystem(SDL_INIT_GAMEPAD)) {
            impl_->state.error = SDL_GetError();
            return false;
        }
        impl_->initialized = true;
        impl_->state.available = true;
        impl_->state.error.clear();
        int count = 0;
        SDL_JoystickID* devices = SDL_GetJoysticks(&count);
        if (!devices) {
            impl_->state.error = SDL_GetError();
            shutdown();
            return false;
        }
        for (int i = 0; i < count; ++i)
            impl_->add(devices[i]);
        SDL_free(devices);
        return true;
    }

    void GamepadService::shutdown() {
        impl_->state.navigation_selected = 0;
        impl_->state.navigation_sample = {};
        impl_->state.selected = 0;
        impl_->state.sample = {};
        impl_->connections.clear();
        if (impl_->initialized) {
            SDL_QuitSubSystem(SDL_INIT_GAMEPAD);
            impl_->initialized = false;
        }
        impl_->state.available = false;
    }

    void GamepadService::handleEvent(const SDL_Event& event) {
        if (!impl_->initialized)
            return;
        switch (event.type) {
        case SDL_EVENT_JOYSTICK_ADDED:
            impl_->add(event.jdevice.which);
            break;
        case SDL_EVENT_GAMEPAD_ADDED:
            if (const auto found = std::find_if(impl_->connections.begin(), impl_->connections.end(),
                                                [&](const auto& c) { return c.device.id == event.gdevice.which && c.handle; });
                found == impl_->connections.end()) {
                impl_->remove(event.gdevice.which);
                impl_->add(event.gdevice.which);
            }
            break;
        case SDL_EVENT_GAMEPAD_REMAPPED:
            // Reopen after a mapping change. Selection requires explicit renewal.
            impl_->remove(event.gdevice.which);
            impl_->add(event.gdevice.which);
            break;
        case SDL_EVENT_JOYSTICK_REMOVED:
            impl_->remove(event.jdevice.which);
            break;
        case SDL_EVENT_GAMEPAD_REMOVED:
            impl_->remove(event.gdevice.which);
            // Losing a gamepad mapping need not disconnect the raw joystick.
            if (SDL_GetJoystickNameForID(event.gdevice.which))
                impl_->add(event.gdevice.which);
            break;
        case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
            if (event.gbutton.which == impl_->state.navigation_selected &&
                event.gbutton.button < SDL_GAMEPAD_BUTTON_COUNT)
                impl_->state.navigation_sample.pressed[event.gbutton.button] = true;
            if (event.gbutton.which == impl_->state.selected &&
                event.gbutton.button < SDL_GAMEPAD_BUTTON_COUNT)
                impl_->state.sample.pressed[event.gbutton.button] = true;
            break;
        default:
            break;
        }
    }

    void GamepadService::beginFrame() {
        impl_->state.sample.pressed.fill(false);
        impl_->state.navigation_sample.pressed.fill(false);
    }

    bool GamepadService::select(SDL_JoystickID id) {
        if (id != 0) {
            const auto found = std::find_if(impl_->connections.begin(), impl_->connections.end(),
                                            [id](const auto& c) { return c.device.id == id && c.handle; });
            if (found == impl_->connections.end())
                return false;
        }
        impl_->state.selected = id;
        impl_->state.sample = {};
        sample();
        return true;
    }

    bool GamepadService::sample() {
        if (!impl_->initialized)
            return false;
        const bool diagnostic_changed = readSample(impl_->state.selected, impl_->state.sample);
        return readSample(impl_->state.navigation_selected, impl_->state.navigation_sample) || diagnostic_changed;
    }

    bool GamepadService::selectNavigation(SDL_JoystickID id) {
        if (id != 0 && std::none_of(impl_->connections.begin(), impl_->connections.end(),
                                    [id](const auto& c) { return c.device.id == id && c.handle; }))
            return false;
        impl_->state.navigation_selected = id;
        impl_->state.navigation_sample = {};
        sample();
        return true;
    }

    const GamepadSample& GamepadService::navigationSample() const {
        return impl_->state.navigation_sample;
    }

    SDL_JoystickID GamepadService::diagnosticSelected() const {
        return impl_->state.selected;
    }

    bool GamepadService::readSample(SDL_JoystickID id, GamepadSample& sample) {
        const auto previous = sample;
        auto found = std::find_if(impl_->connections.begin(), impl_->connections.end(),
                                  [id](const auto& c) { return c.device.id == id; });
        if (found == impl_->connections.end() || !found->handle || !SDL_GamepadConnected(found->handle.get())) {
            sample = {};
            return previous != sample;
        }
        auto* pad = found->handle.get();
        sample.id = found->device.id;
        sample.generation = found->device.generation;
        for (int i = 0; i < SDL_GAMEPAD_AXIS_COUNT; ++i) {
            const auto axis = static_cast<SDL_GamepadAxis>(i);
            sample.has_axis[i] = SDL_GamepadHasAxis(pad, axis);
            const auto raw = SDL_GetGamepadAxis(pad, axis);
            sample.axes[i] = sample.has_axis[i] ? float(raw) / (raw < 0 ? 32768.0f : 32767.0f) : 0.0f;
        }
        for (int i = 0; i < SDL_GAMEPAD_BUTTON_COUNT; ++i) {
            const auto button = static_cast<SDL_GamepadButton>(i);
            sample.has_button[i] = SDL_GamepadHasButton(pad, button);
            sample.buttons[i] = sample.has_button[i] && SDL_GetGamepadButton(pad, button);
            if (i < 4)
                sample.labels[i] = SDL_GetGamepadButtonLabel(pad, button);
        }
        return previous != sample || std::any_of(sample.pressed.begin(), sample.pressed.end(), [](bool pressed) { return pressed; });
    }

    GamepadSnapshot GamepadService::snapshot() const {
        auto state = impl_->state;
        for (const auto& connection : impl_->connections)
            state.devices.push_back(connection.device);
        return state;
    }

} // namespace lfs::vis::input
