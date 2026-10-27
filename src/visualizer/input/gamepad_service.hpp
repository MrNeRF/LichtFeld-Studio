/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <SDL3/SDL_gamepad.h>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

union SDL_Event;

namespace lfs::vis::input {

    struct GamepadDevice {
        SDL_JoystickID id = 0;
        uint64_t generation = 0;
        std::string name;
        bool mapped = false;
        SDL_GamepadType type = SDL_GAMEPAD_TYPE_UNKNOWN;
        std::string error;
        std::string profile_key; // SDL model GUID; never the connection ID
    };

    struct GamepadSample {
        SDL_JoystickID id = 0;
        uint64_t generation = 0;
        std::array<float, SDL_GAMEPAD_AXIS_COUNT> axes{};
        std::array<bool, SDL_GAMEPAD_AXIS_COUNT> has_axis{};
        std::array<bool, SDL_GAMEPAD_BUTTON_COUNT> buttons{};
        std::array<bool, SDL_GAMEPAD_BUTTON_COUNT> has_button{};
        std::array<bool, SDL_GAMEPAD_BUTTON_COUNT> pressed{};
        std::array<bool, 2> trigger_pressed{};
        std::array<SDL_GamepadButtonLabel, 4> labels{};
        bool operator==(const GamepadSample&) const = default;
    };

    struct GamepadSnapshot {
        bool available = false;
        std::string error;
        std::vector<GamepadDevice> devices;
        SDL_JoystickID selected = 0;
        GamepadSample sample;
        SDL_JoystickID navigation_selected = 0;
        GamepadSample navigation_sample;
    };

    // Viewer-thread service. SDL IDs identify connections, never saved profiles.
    // Snapshots contain values only and remain valid after a handle is closed.
    // Diagnostics and navigation have independent selections and edge buffers.
    class LFS_VIS_API GamepadService {
    public:
        GamepadService();
        ~GamepadService();
        GamepadService(const GamepadService&) = delete;
        GamepadService& operator=(const GamepadService&) = delete;

        bool initialize();
        // Acquire SDL only while navigation or a visible diagnostics page needs it.
        bool setActive(bool active);
        void shutdown();
        void beginFrame();
        void consumeNavigationPresses();
        [[nodiscard]] GamepadSnapshot takeDiagnosticSnapshot();
        void handleEvent(const SDL_Event& event);
        bool sample(bool navigation_enabled = true);
        [[nodiscard]] bool select(SDL_JoystickID id);
        [[nodiscard]] bool selectNavigation(SDL_JoystickID id);
        [[nodiscard]] const GamepadSample& navigationSample() const;
        [[nodiscard]] SDL_JoystickID diagnosticSelected() const;
        [[nodiscard]] SDL_JoystickID navigationSelected() const;
        [[nodiscard]] std::string navigationProfileKey() const;
        [[nodiscard]] GamepadSnapshot snapshot() const;

    private:
        bool readSample(SDL_JoystickID id, GamepadSample& sample);
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis::input
