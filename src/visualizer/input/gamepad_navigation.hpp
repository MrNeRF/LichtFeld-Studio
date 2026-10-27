/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gamepad_bindings.hpp"
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <stdexcept>

namespace lfs::vis::input {

    struct GamepadNavigationSettings {
        bool enabled = false;
        float dead_zone = 0.15f;
        float look_speed = 1.5f; // radians/second
        float move_speed = 1.0f; // multiplier of the mode's configured speed
        GamepadBindings bindings;
        uint64_t revision = 0; // session-local, invalidates arming after edits

        void validate() const {
            if (!std::isfinite(dead_zone) || dead_zone < 0.05f || dead_zone > 0.5f ||
                !std::isfinite(look_speed) || look_speed < 0.1f || look_speed > 5.0f ||
                !std::isfinite(move_speed) || move_speed < 0.1f || move_speed > 5.0f)
                throw std::invalid_argument("Invalid gamepad navigation settings");
            bindings.validate();
        }

        nlohmann::json json() const {
            return {{"schema_version", 1}, {"enabled", enabled}, {"dead_zone", dead_zone}, {"look_speed", look_speed}, {"move_speed", move_speed}, {"bindings", bindings.json()}};
        }
        static GamepadNavigationSettings fromJson(const nlohmann::json& value) {
            if (!value.is_object() || (value.contains("schema_version") && (!value["schema_version"].is_number_integer() || value["schema_version"] != 1)))
                throw std::invalid_argument("Unsupported gamepad configuration version");
            for (auto it = value.begin(); it != value.end(); ++it)
                if (it.key() != "schema_version" && it.key() != "enabled" && it.key() != "dead_zone" && it.key() != "look_speed" && it.key() != "move_speed" && it.key() != "bindings")
                    throw std::invalid_argument("Unknown gamepad configuration field");
            GamepadNavigationSettings result;
            for (const char* key : {"dead_zone", "look_speed", "move_speed"})
                if (value.contains(key) && !value[key].is_number())
                    throw std::invalid_argument("Invalid gamepad tuning value");
            result.enabled = value.value("enabled", false);
            result.dead_zone = value.value("dead_zone", 0.15f);
            result.look_speed = value.value("look_speed", 1.5f);
            result.move_speed = value.value("move_speed", 1.0f);
            if (value.contains("bindings"))
                result.bindings = GamepadBindings::fromJson(value["bindings"]);
            result.validate();
            return result;
        }
    };

    struct GamepadNavigationContext {
        bool allowed = false;
        bool supported_projection = true;
        uintptr_t target = 0; // identity only; never dereferenced by the resolver
        uint64_t click_epoch = 0;
        GamepadNavigationMode mode = GamepadNavigationMode::Orbit;
        bool sequencer_open = false;
        bool sequencer_commands_allowed = true;
    };

    struct GamepadNavigationIntent {
        glm::vec2 move{0}; // right/forward (pan right/up for orbit modes)
        glm::vec2 look{0}; // yaw/pitch in radians/second
        float vertical = 0;
        float speed = 1;
        bool exit_walk = false;
        int navigation_step = 0;
        int rendering_step = 0;
        bool add_keyframe = false;
        bool remove_selected_keyframe = false;
        bool deselect_keyframe = false;
        bool update_selected_keyframe = false;
        int keyframe_step = 0;
        bool active = false;
    };

    // Pure routing/conditioning state. It never sends keyboard or editing actions.
    class GamepadNavigationSession {
    public:
        void suspend(uint64_t click_epoch) {
            last_click_ = click_epoch;
            armed_ = pending_ = false;
            action_down_.fill(false);
            neutral_seconds_ = 0;
            status_ = "suspended";
        }

        // Temporary UI/app interruption: retain the deliberate viewport grant.
        void pause() {
            pending_ = pending_ || armed_;
            armed_ = false;
            neutral_seconds_ = 0;
            action_down_.fill(false);
            status_ = "suspended";
        }

        [[nodiscard]] const char* status() const { return status_; }
        [[nodiscard]] bool needsNeutral() const { return pending_ && !armed_; }
        [[nodiscard]] bool awaitingNeutral() const { return needsNeutral() && neutral_seconds_ > 0; }

        [[nodiscard]] static glm::vec2 filterStick(glm::vec2 value, float dead_zone) {
            if (!std::isfinite(value.x) || !std::isfinite(value.y))
                return glm::vec2(0);
            value = glm::clamp(value, glm::vec2(-1), glm::vec2(1));
            const float length = glm::length(value);
            if (length <= dead_zone)
                return glm::vec2(0);
            return value / length * ((std::min(length, 1.0f) - dead_zone) / (1.0f - dead_zone));
        }

        // An armed session only wakes for effective movement. The narrower
        // threshold matters while rearming, including entry into neutral.
        [[nodiscard]] static bool needsSampleUpdate(GamepadSample previous, GamepadSample current, float dead_zone, bool rearming = false) {
            const auto condition = [dead_zone, rearming](GamepadSample& sample) {
                for (int i : {0, 2}) {
                    const auto stick = filterStick({sample.axes[i], sample.axes[i + 1]}, rearming ? dead_zone * 0.8f : dead_zone);
                    sample.axes[i] = stick.x;
                    sample.axes[i + 1] = stick.y;
                }
                for (int i : {4, 5})
                    if (std::abs(sample.axes[i]) <= (rearming ? 0.04f : 0.05f))
                        sample.axes[i] = 0;
            };
            condition(previous);
            condition(current);
            return previous != current ||
                   std::any_of(current.pressed.begin(), current.pressed.end(), [](bool pressed) { return pressed; }) ||
                   std::any_of(current.trigger_pressed.begin(), current.trigger_pressed.end(), [](bool pressed) { return pressed; });
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
            const auto mode_index = static_cast<size_t>(context.mode);
            bool mapped_navigation = true;
            for (size_t a = 0; a <= static_cast<size_t>(GamepadAction::Vertical); ++a) {
                if (!allowedGamepadAction(mode_index, a))
                    continue;
                const auto& binding = settings.bindings.resolve(mode_index, a);
                for (int c : {binding.positive, binding.negative})
                    if (c >= 0 && c < SDL_GAMEPAD_AXIS_LEFT_TRIGGER && !sample.has_axis[c])
                        mapped_navigation = false;
            }
            if (!mapped_navigation && !context.sequencer_open)
                return blocked("mapping");
            if (!context.supported_projection && !context.sequencer_open)
                return blocked("projection");
            const bool navigation_supported = mapped_navigation && context.supported_projection;
            if (!context.target)
                return blocked("suspended");
            if (sample.id != id_ || sample.generation != generation_ ||
                context.target != target_ || settings.revision != revision_) {
                armed_ = pending_ = false;
                neutral_seconds_ = 0;
                id_ = sample.id;
                generation_ = sample.generation;
                target_ = context.target;
                mode_ = context.mode;
                sequencer_open_ = context.sequencer_open;
                revision_ = settings.revision;
                action_down_.fill(false);
            }
            if (context.mode != mode_ || context.sequencer_open != sequencer_open_) {
                pause();
                mode_ = context.mode;
                sequencer_open_ = context.sequencer_open;
            }
            if (!context.allowed || !std::isfinite(delta_time) || delta_time <= 0) {
                pause();
                last_click_ = context.click_epoch;
                return result;
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
                                          std::none_of(sample.pressed.begin(), sample.pressed.end(), [](bool b) { return b; }) &&
                                          std::none_of(sample.trigger_pressed.begin(), sample.trigger_pressed.end(), [](bool b) { return b; });
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
            const auto read_control = [&](int c) -> float {
                if (c < 0)
                    return 0;
                if (c < 2)
                    return c == 0 ? left.x : left.y;
                if (c < 4)
                    return c == 2 ? right.x : right.y;
                if (c < SDL_GAMEPAD_AXIS_COUNT)
                    return c == SDL_GAMEPAD_AXIS_LEFT_TRIGGER ? down : up;
                const auto b = c - SDL_GAMEPAD_AXIS_COUNT;
                return sample.has_button[b] && sample.buttons[b] ? 1.0f : 0.0f;
            };
            const auto reserved = [&](int control) {
                if (!context.sequencer_open || control < 0)
                    return false;
                for (size_t a = gamepad_first_sequencer_action; a < gamepad_action_count; ++a)
                    if (settings.bindings.resolve(gamepad_sequencer_scope, a).positive == control)
                        return true;
                return false;
            };
            const auto navigation_control = [&](int control) { return reserved(control) ? 0.0f : read_control(control); };
            const auto analog = [&](size_t a) {
                if (!navigation_supported || !allowedGamepadAction(mode_index, a))
                    return 0.0f;
                const auto& b = settings.bindings.resolve(mode_index, a);
                return std::clamp(navigation_control(b.positive) - navigation_control(b.negative), -1.0f, 1.0f) * (b.inverted ? -1.0f : 1.0f);
            };
            std::array<bool, gamepad_action_count> edge{};
            for (size_t a = actionIndex(GamepadAction::Precision); a < gamepad_action_count; ++a) {
                const bool sequencer_action = a >= gamepad_first_sequencer_action;
                const auto scope = sequencer_action ? gamepad_sequencer_scope : mode_index;
                const auto& b = settings.bindings.resolve(scope, a);
                const bool enabled = allowedGamepadAction(scope, a) &&
                                     (sequencer_action ? context.sequencer_open : navigation_supported && !reserved(b.positive));
                const bool trigger_control = b.positive >= SDL_GAMEPAD_AXIS_LEFT_TRIGGER && b.positive < SDL_GAMEPAD_AXIS_COUNT;
                const bool held = enabled && (trigger_control ? sample.has_axis[b.positive] && sample.trigger_down[b.positive - SDL_GAMEPAD_AXIS_LEFT_TRIGGER] : read_control(b.positive) > 0.5f);
                const bool short_press = enabled &&
                                         ((b.positive >= SDL_GAMEPAD_AXIS_COUNT &&
                                           sample.has_button[b.positive - SDL_GAMEPAD_AXIS_COUNT] && sample.pressed[b.positive - SDL_GAMEPAD_AXIS_COUNT]) ||
                                          (b.positive >= SDL_GAMEPAD_AXIS_LEFT_TRIGGER && b.positive < SDL_GAMEPAD_AXIS_COUNT &&
                                           sample.has_axis[b.positive] && sample.trigger_pressed[b.positive - SDL_GAMEPAD_AXIS_LEFT_TRIGGER]));
                // Keep held state while timeline gestures block editing, so a
                // button held through the gesture cannot fire on release.
                edge[a] = (!sequencer_action || context.sequencer_commands_allowed) &&
                          (short_press || (held && !action_down_[a]));
                action_down_[a] = held;
            }
            const bool precision = action_down_[actionIndex(GamepadAction::Precision)];
            const bool fast = action_down_[actionIndex(GamepadAction::Fast)];
            result.speed = settings.move_speed * (precision ? 0.35f : fast ? 1.6f
                                                                           : 1.0f);
            result.move = {analog(actionIndex(GamepadAction::MoveX)), analog(actionIndex(GamepadAction::MoveY))};
            result.look = glm::vec2(analog(actionIndex(GamepadAction::LookX)), analog(actionIndex(GamepadAction::LookY))) * settings.look_speed * (precision ? 0.35f : 1.0f);
            result.vertical = analog(actionIndex(GamepadAction::Vertical));
            result.exit_walk = edge[actionIndex(GamepadAction::ExitWalk)];
            result.navigation_step = static_cast<int>(edge[actionIndex(GamepadAction::NextNavigation)]) - static_cast<int>(edge[actionIndex(GamepadAction::PreviousNavigation)]);
            result.rendering_step = static_cast<int>(edge[actionIndex(GamepadAction::NextRendering)]) - static_cast<int>(edge[actionIndex(GamepadAction::PreviousRendering)]);
            // Ambiguous simultaneous editing commands are ignored, particularly
            // add + delete: never delete a key just created by the same sample.
            const auto commands = std::count(edge.begin() + gamepad_first_sequencer_action, edge.end(), true);
            if (commands == 1) {
                result.add_keyframe = edge[actionIndex(GamepadAction::AddKeyframe)];
                result.remove_selected_keyframe = edge[actionIndex(GamepadAction::RemoveSelectedKeyframe)];
                result.deselect_keyframe = edge[actionIndex(GamepadAction::DeselectKeyframe)];
                result.update_selected_keyframe = edge[actionIndex(GamepadAction::UpdateSelectedKeyframe)];
                result.keyframe_step = static_cast<int>(edge[actionIndex(GamepadAction::NextKeyframe)]) - static_cast<int>(edge[actionIndex(GamepadAction::PreviousKeyframe)]);
            }
            result.active = result.add_keyframe || result.remove_selected_keyframe || result.deselect_keyframe || result.update_selected_keyframe || result.keyframe_step || result.move != glm::vec2(0) || result.look != glm::vec2(0) || result.vertical != 0 || result.exit_walk || result.navigation_step || result.rendering_step;
            status_ = result.active ? "moving" : "ready";
            return result;
        }

    private:
        SDL_JoystickID id_ = 0;
        uint64_t generation_ = 0, last_click_ = 0;
        uintptr_t target_ = 0;
        GamepadNavigationMode mode_ = GamepadNavigationMode::Orbit;
        bool armed_ = false, pending_ = false;
        bool sequencer_open_ = false;
        float neutral_seconds_ = 0;
        uint64_t revision_ = 0;
        std::array<bool, gamepad_action_count> action_down_{};
        const char* status_ = "disabled";
    };

} // namespace lfs::vis::input
