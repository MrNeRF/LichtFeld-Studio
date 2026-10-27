/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "gamepad_service.hpp"
#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <nlohmann/json.hpp>

namespace lfs::vis::input {
    enum class GamepadNavigationMode { Orbit, Trackball, FPV, Drone, Walk };
    enum class GamepadAction { MoveX, MoveY, LookX, LookY, Vertical, Precision, Fast,
                               ExitWalk, NextNavigation, PreviousNavigation, NextRendering, PreviousRendering,
                               AddKeyframe, RemoveSelectedKeyframe, PreviousKeyframe, NextKeyframe, DeselectKeyframe, UpdateSelectedKeyframe, Count };
    inline constexpr size_t actionIndex(GamepadAction action) { return static_cast<size_t>(action); }
    inline constexpr size_t gamepad_action_count = static_cast<size_t>(GamepadAction::Count);
    inline constexpr std::array<std::string_view, 5> gamepad_modes{"orbit", "trackball", "fpv", "drone", "walk"};
    inline constexpr size_t gamepad_sequencer_scope = gamepad_modes.size();
    inline constexpr size_t gamepad_first_sequencer_action = static_cast<size_t>(GamepadAction::AddKeyframe);
    inline constexpr std::array<std::string_view, 6> gamepad_binding_contexts{"orbit", "trackball", "fpv", "drone", "walk", "sequencer"};
    inline constexpr std::array<std::string_view, gamepad_action_count> gamepad_actions{
        "move_x", "move_y", "look_x", "look_y", "vertical", "precision", "fast", "exit_walk",
        "next_navigation", "previous_navigation", "next_rendering", "previous_rendering",
        "add_keyframe", "remove_selected_keyframe", "previous_keyframe", "next_keyframe", "deselect_keyframe", "update_selected_keyframe"};
    inline constexpr std::array<std::string_view, SDL_GAMEPAD_AXIS_COUNT> gamepad_axes{
        "left_x", "left_y", "right_x", "right_y", "left_trigger", "right_trigger"};
    inline constexpr std::array<std::string_view, SDL_GAMEPAD_BUTTON_COUNT> gamepad_buttons{
        "south", "east", "west", "north", "back", "guide", "start", "left_stick", "right_stick",
        "left_shoulder", "right_shoulder", "dpad_up", "dpad_down", "dpad_left", "dpad_right",
        "misc1", "right_paddle1", "left_paddle1", "right_paddle2", "left_paddle2", "touchpad",
        "misc2", "misc3", "misc4", "misc5", "misc6"};

    // Stable SDL positions, never vendor letters or connection-specific identifiers.
    // -1 is unbound; axes precede buttons in this compact in-memory representation.
    inline std::string controlName(int control) {
        if (control == -1) return "none";
        if (control >= 0 && control < SDL_GAMEPAD_AXIS_COUNT)
            return "axis." + std::string(gamepad_axes[control]);
        const int button = control - SDL_GAMEPAD_AXIS_COUNT;
        if (button >= 0 && button < SDL_GAMEPAD_BUTTON_COUNT)
            return "button." + std::string(gamepad_buttons[button]);
        throw std::invalid_argument("Unknown gamepad control");
    }
    inline int controlFromName(const std::string& name) {
        for (int i = -1; i < int(SDL_GAMEPAD_AXIS_COUNT) + int(SDL_GAMEPAD_BUTTON_COUNT); ++i)
            if (controlName(i) == name) return i;
        throw std::invalid_argument("Unknown gamepad control: " + name);
    }
    inline bool continuousAction(size_t action) { return action <= static_cast<size_t>(GamepadAction::Vertical); }
    inline bool allowedGamepadAction(size_t mode, size_t action) {
        if (mode == gamepad_sequencer_scope) return action >= gamepad_first_sequencer_action;
        if (action >= gamepad_first_sequencer_action) return false;
        constexpr auto walk = static_cast<size_t>(GamepadNavigationMode::Walk);
        if (action == actionIndex(GamepadAction::ExitWalk)) return mode == walk;
        return mode != walk || (action != actionIndex(GamepadAction::Vertical) && action != actionIndex(GamepadAction::Fast));
    }
    // Try each other mode once. The activation callback performs the real
    // capability/spawn check without committing failed candidates.
    template <class Activate>
    bool cycleGamepadNavigation(GamepadNavigationMode current, int direction, bool orthographic, Activate&& activate) {
        if (direction != 1 && direction != -1) return false;
        const int start = static_cast<int>(current);
        for (int offset = 1; offset < int(gamepad_modes.size()); ++offset) {
            const int index = (start + direction * offset + int(gamepad_modes.size())) % int(gamepad_modes.size());
            if (orthographic && index > 1) continue;
            if (activate(static_cast<GamepadNavigationMode>(index))) return true;
        }
        return false;
    }

    struct GamepadBinding {
        int positive = -1;
        int negative = -1;
        bool inverted = false;
        bool operator==(const GamepadBinding&) const = default;
        void validate(size_t action) const {
            (void)controlName(positive);
            (void)controlName(negative);
            if (positive != -1 && positive == negative)
                throw std::invalid_argument("Opposing inputs must differ");
            if (!continuousAction(action) && (negative != -1 || inverted ||
                (positive >= 0 && positive < SDL_GAMEPAD_AXIS_LEFT_TRIGGER)))
                throw std::invalid_argument("This action requires a button or trigger");
            if (negative >= 0 && (negative < SDL_GAMEPAD_AXIS_LEFT_TRIGGER ||
                (positive >= 0 && positive < SDL_GAMEPAD_AXIS_LEFT_TRIGGER)))
                throw std::invalid_argument("A signed stick axis uses inversion, not a second input");
        }
        nlohmann::json json() const {
            return {{"positive", controlName(positive)}, {"negative", controlName(negative)}, {"inverted", inverted}};
        }
        static GamepadBinding fromJson(const nlohmann::json& value) {
            if (!value.is_object()) throw std::invalid_argument("Invalid gamepad binding");
            for (auto it = value.begin(); it != value.end(); ++it)
                if (it.key() != "positive" && it.key() != "negative" && it.key() != "inverted")
                    throw std::invalid_argument("Unknown gamepad binding field");
            return {controlFromName(value.value("positive", std::string("none"))),
                    controlFromName(value.value("negative", std::string("none"))), value.value("inverted", false)};
        }
    };
    struct GamepadBindings {
        std::array<GamepadBinding, gamepad_action_count> common{};
        std::array<std::array<std::optional<GamepadBinding>, gamepad_action_count>, gamepad_binding_contexts.size()> modes{};
        GamepadBindings() {
            common[actionIndex(GamepadAction::MoveX)] = {SDL_GAMEPAD_AXIS_LEFTX};
            common[actionIndex(GamepadAction::MoveY)] = {SDL_GAMEPAD_AXIS_LEFTY, -1, true};
            common[actionIndex(GamepadAction::LookX)] = {SDL_GAMEPAD_AXIS_RIGHTX, -1, true};
            common[actionIndex(GamepadAction::LookY)] = {SDL_GAMEPAD_AXIS_RIGHTY, -1, true};
            common[actionIndex(GamepadAction::Vertical)] = {SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, SDL_GAMEPAD_AXIS_LEFT_TRIGGER};
            common[actionIndex(GamepadAction::NextNavigation)] = {int(SDL_GAMEPAD_AXIS_COUNT) + SDL_GAMEPAD_BUTTON_LEFT_SHOULDER};
            common[actionIndex(GamepadAction::NextRendering)] = {int(SDL_GAMEPAD_AXIS_COUNT) + SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER};
            common[actionIndex(GamepadAction::AddKeyframe)] = {int(SDL_GAMEPAD_AXIS_COUNT) + SDL_GAMEPAD_BUTTON_WEST};
            common[actionIndex(GamepadAction::RemoveSelectedKeyframe)] = {int(SDL_GAMEPAD_AXIS_COUNT) + SDL_GAMEPAD_BUTTON_NORTH};
            common[actionIndex(GamepadAction::PreviousKeyframe)] = {int(SDL_GAMEPAD_AXIS_COUNT) + SDL_GAMEPAD_BUTTON_DPAD_LEFT};
            common[actionIndex(GamepadAction::NextKeyframe)] = {int(SDL_GAMEPAD_AXIS_COUNT) + SDL_GAMEPAD_BUTTON_DPAD_RIGHT};
            modes[static_cast<size_t>(GamepadNavigationMode::Walk)][actionIndex(GamepadAction::ExitWalk)] = GamepadBinding{int(SDL_GAMEPAD_AXIS_COUNT) + SDL_GAMEPAD_BUTTON_EAST};
        }
        const GamepadBinding& resolve(size_t mode, size_t action) const {
            return modes.at(mode).at(action) ? *modes[mode][action] : common.at(action);
        }
        void validate() const {
            for (size_t a = 0; a < gamepad_action_count; ++a) common[a].validate(a);
            for (size_t m = 0; m < modes.size(); ++m) {
                std::array<int, int(SDL_GAMEPAD_AXIS_COUNT) + int(SDL_GAMEPAD_BUTTON_COUNT)> owner;
                owner.fill(-1);
                for (size_t a = 0; a < gamepad_action_count; ++a) {
                    if (modes[m][a]) modes[m][a]->validate(a);
                    if (!allowedGamepadAction(m, a)) {
                        if (modes[m][a] && (modes[m][a]->positive != -1 || modes[m][a]->negative != -1))
                            throw std::invalid_argument("Action unavailable in this navigation mode");
                        continue;
                    }
                    const auto& binding = resolve(m, a);
                    for (int c : {binding.positive, binding.negative}) if (c >= 0) {
                        if (owner[c] != -1)
                            throw std::invalid_argument("Conflicting gamepad inputs in " + std::string(gamepad_binding_contexts[m]));
                        owner[c] = static_cast<int>(a);
                    }
                }
            }
        }
        nlohmann::json json() const {
            nlohmann::json result{{"common", nlohmann::json::object()}, {"modes", nlohmann::json::object()}};
            for (size_t a = 0; a < gamepad_action_count; ++a) result["common"][gamepad_actions[a]] = common[a].json();
            for (size_t m = 0; m < modes.size(); ++m)
                for (size_t a = 0; a < gamepad_action_count; ++a)
                    if (modes[m][a]) result["modes"][gamepad_binding_contexts[m]][gamepad_actions[a]] = modes[m][a]->json();
            return result;
        }
        static GamepadBindings fromJson(const nlohmann::json& value) {
            GamepadBindings result;
            if (!value.is_object()) throw std::invalid_argument("Invalid gamepad bindings");
            for (auto it = value.begin(); it != value.end(); ++it)
                if (it.key() != "common" && it.key() != "modes") throw std::invalid_argument("Unknown bindings field");
            const auto read = [&](const nlohmann::json& fields, auto& target) {
                if (!fields.is_object()) throw std::invalid_argument("Invalid binding context");
                for (auto it = fields.begin(); it != fields.end(); ++it) {
                    const auto found = std::find(gamepad_actions.begin(), gamepad_actions.end(), it.key());
                    if (found == gamepad_actions.end()) throw std::invalid_argument("Unknown gamepad action");
                    target[std::distance(gamepad_actions.begin(), found)] = GamepadBinding::fromJson(it.value());
                }
            };
            if (value.contains("common")) read(value["common"], result.common);
            if (value.contains("modes")) {
                result.modes = {};
                if (!value["modes"].is_object()) throw std::invalid_argument("Invalid navigation contexts");
                for (auto it = value["modes"].begin(); it != value["modes"].end(); ++it) {
                    const auto found = std::find(gamepad_binding_contexts.begin(), gamepad_binding_contexts.end(), it.key());
                    if (found == gamepad_binding_contexts.end()) throw std::invalid_argument("Unknown navigation context");
                    read(it.value(), result.modes[std::distance(gamepad_binding_contexts.begin(), found)]);
                }
            }
            result.validate();
            return result;
        }
    };
}
