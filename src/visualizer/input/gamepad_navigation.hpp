/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gamepad_service.hpp"
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <stdexcept>

namespace lfs::vis::input {

    enum class GamepadNavigationMode { Orbit,
                                       Trackball,
                                       FPV,
                                       Drone,
                                       Walk };

    struct GamepadNavigationSettings {
        bool enabled = false;
        float dead_zone = 0.15f;
        float look_speed = 1.5f; // radians/second
        float move_speed = 1.0f; // multiplier of the mode's configured speed

        void validate() const {
            if (!std::isfinite(dead_zone) || dead_zone < 0.05f || dead_zone > 0.5f ||
                !std::isfinite(look_speed) || look_speed < 0.1f || look_speed > 5.0f ||
                !std::isfinite(move_speed) || move_speed < 0.1f || move_speed > 5.0f)
                throw std::invalid_argument("Invalid gamepad navigation settings");
        }
    };

    struct GamepadNavigationContext {
        bool allowed = false;
        bool supported_projection = true;
        uintptr_t target = 0; // identity only; never dereferenced by the resolver
        uint64_t click_epoch = 0;
        GamepadNavigationMode mode = GamepadNavigationMode::Orbit;
    };

    struct GamepadNavigationIntent {
        glm::vec2 move{0}; // right/forward (pan right/up for orbit modes)
        glm::vec2 look{0}; // yaw/pitch in radians/second
        float vertical = 0;
        float speed = 1;
        bool exit_walk = false;
        bool active = false;
    };

    // Pure routing/conditioning state. It never sends keyboard or editing actions.
    class GamepadNavigationSession {
    public:
        void suspend(uint64_t click_epoch) {
            last_click_ = click_epoch;
            armed_ = pending_ = false;
            neutral_seconds_ = 0;
            status_ = "suspended";
        }

        [[nodiscard]] const char* status() const { return status_; }
        [[nodiscard]] bool awaitingNeutral() const { return pending_ && neutral_seconds_ > 0 && !armed_; }

        [[nodiscard]] static glm::vec2 filterStick(glm::vec2 value, float dead_zone) {
            if (!std::isfinite(value.x) || !std::isfinite(value.y))
                return glm::vec2(0);
            value = glm::clamp(value, glm::vec2(-1), glm::vec2(1));
            const float length = glm::length(value);
            if (length <= dead_zone)
                return glm::vec2(0);
            return value / length * ((std::min(length, 1.0f) - dead_zone) / (1.0f - dead_zone));
        }

        GamepadNavigationIntent update(const GamepadSample& sample, const GamepadNavigationSettings& settings,
                                       const GamepadNavigationContext& context, float delta_time) {
            GamepadNavigationIntent result;
            const auto blocked = [&](const char* reason) {
                suspend(context.click_epoch);
                status_ = reason;
                return result;
            };
            if (!settings.enabled)
                return blocked("disabled");
            if (!sample.id)
                return blocked("no_device");
            if (std::any_of(sample.axes.begin(), sample.axes.end(), [](float value) { return !std::isfinite(value); }))
                return blocked("suspended");
            if (!sample.has_axis[SDL_GAMEPAD_AXIS_LEFTX] || !sample.has_axis[SDL_GAMEPAD_AXIS_LEFTY] ||
                !sample.has_axis[SDL_GAMEPAD_AXIS_RIGHTX] || !sample.has_axis[SDL_GAMEPAD_AXIS_RIGHTY])
                return blocked("mapping");
            if (!context.supported_projection)
                return blocked("projection");
            if (!context.allowed || !context.target || !std::isfinite(delta_time) || delta_time <= 0)
                return blocked("suspended");
            if (sample.id != id_ || sample.generation != generation_ ||
                context.target != target_ || context.mode != mode_) {
                armed_ = pending_ = false;
                neutral_seconds_ = 0;
                id_ = sample.id;
                generation_ = sample.generation;
                target_ = context.target;
                mode_ = context.mode;
            }
            if (context.click_epoch != last_click_) {
                pending_ = true;
                armed_ = false;
                neutral_seconds_ = 0;
                last_click_ = context.click_epoch;
            }
            const auto axis = [&](int index) {
                return sample.has_axis[index] && std::isfinite(sample.axes[index]) ? sample.axes[index] : 0.0f;
            };
            const auto left = filterStick({axis(SDL_GAMEPAD_AXIS_LEFTX), axis(SDL_GAMEPAD_AXIS_LEFTY)}, settings.dead_zone);
            const auto right = filterStick({axis(SDL_GAMEPAD_AXIS_RIGHTX), axis(SDL_GAMEPAD_AXIS_RIGHTY)}, settings.dead_zone);
            const auto trigger = [&](int index) {
                return std::clamp((axis(index) - 0.05f) / 0.95f, 0.0f, 1.0f);
            };
            const float down = trigger(SDL_GAMEPAD_AXIS_LEFT_TRIGGER);
            const float up = trigger(SDL_GAMEPAD_AXIS_RIGHT_TRIGGER);
            const bool buttons_released = std::none_of(sample.buttons.begin(), sample.buttons.end(), [](bool b) { return b; }) &&
                                          std::none_of(sample.pressed.begin(), sample.pressed.end(), [](bool b) { return b; });
            // Hysteresis: entering the armed state requires more clearance than
            // remaining at zero after the regular dead zone has been applied.
            const bool neutral = buttons_released &&
                                 glm::length(glm::vec2(axis(SDL_GAMEPAD_AXIS_LEFTX), axis(SDL_GAMEPAD_AXIS_LEFTY))) <= settings.dead_zone * 0.8f &&
                                 glm::length(glm::vec2(axis(SDL_GAMEPAD_AXIS_RIGHTX), axis(SDL_GAMEPAD_AXIS_RIGHTY))) <= settings.dead_zone * 0.8f &&
                                 std::abs(axis(SDL_GAMEPAD_AXIS_LEFT_TRIGGER)) <= 0.04f &&
                                 std::abs(axis(SDL_GAMEPAD_AXIS_RIGHT_TRIGGER)) <= 0.04f;
            if (!armed_) {
                status_ = pending_ ? "neutral" : "viewport";
                neutral_seconds_ = pending_ && neutral ? neutral_seconds_ + std::min(delta_time, 0.05f) : 0;
                if (neutral_seconds_ < 0.12f)
                    return result;
                armed_ = true;
                pending_ = false;
            }
            const bool precision = sample.buttons[SDL_GAMEPAD_BUTTON_LEFT_SHOULDER];
            const bool fast = sample.buttons[SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER] && mode_ != GamepadNavigationMode::Walk;
            result.speed = settings.move_speed * (precision ? 0.35f : fast ? 1.6f
                                                                           : 1.0f);
            result.move = {left.x, -left.y};
            result.look = -right * settings.look_speed * (precision ? 0.35f : 1.0f);
            result.vertical = mode_ == GamepadNavigationMode::Walk ? 0.0f : up - down;
            result.exit_walk = mode_ == GamepadNavigationMode::Walk && sample.pressed[SDL_GAMEPAD_BUTTON_EAST];
            result.active = result.move != glm::vec2(0) || result.look != glm::vec2(0) || result.vertical != 0 || result.exit_walk;
            status_ = result.active ? "moving" : "ready";
            return result;
        }

    private:
        SDL_JoystickID id_ = 0;
        uint64_t generation_ = 0, last_click_ = 0;
        uintptr_t target_ = 0;
        GamepadNavigationMode mode_ = GamepadNavigationMode::Orbit;
        bool armed_ = false, pending_ = false;
        float neutral_seconds_ = 0;
        const char* status_ = "disabled";
    };

} // namespace lfs::vis::input
