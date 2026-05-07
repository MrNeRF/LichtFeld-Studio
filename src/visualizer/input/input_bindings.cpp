/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_bindings.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <shlobj.h>
#else
#include <pwd.h>
#include <unistd.h>
#endif

namespace lfs::vis::input {

    namespace {

        constexpr int PROFILE_VERSION = 10; // Version 10 routes polygon secondary click through key bindings.
        constexpr std::array<ToolMode, 8> ALL_MODES = {
            ToolMode::GLOBAL,
            ToolMode::SELECTION,
            ToolMode::BRUSH,
            ToolMode::ALIGN,
            ToolMode::CROP_BOX,
            ToolMode::TRANSLATE,
            ToolMode::ROTATE,
            ToolMode::SCALE,
        };
        constexpr std::array<ToolMode, 4> NODE_PICK_MODES = {
            ToolMode::GLOBAL,
            ToolMode::TRANSLATE,
            ToolMode::ROTATE,
            ToolMode::SCALE,
        };
        constexpr std::array<ToolMode, 4> DELETE_NODE_MODES = {
            ToolMode::GLOBAL,
            ToolMode::TRANSLATE,
            ToolMode::ROTATE,
            ToolMode::SCALE,
        };
        constexpr std::array<ToolMode, 4> DELETE_GAUSSIANS_MODES = {
            ToolMode::SELECTION,
            ToolMode::BRUSH,
            ToolMode::ALIGN,
            ToolMode::CROP_BOX,
        };

        [[nodiscard]] bool isSelectionDepthAction(const Action action) {
            switch (action) {
            case Action::TOGGLE_DEPTH_MODE:
            case Action::DEPTH_ADJUST_NEAR:
            case Action::DEPTH_ADJUST_FAR:
            case Action::DEPTH_ADJUST_SIDE:
            case Action::TOGGLE_SELECTION_DEPTH_FILTER:
            case Action::TOGGLE_SELECTION_CROP_FILTER:
                return true;
            default:
                return false;
            }
        }

        [[nodiscard]] bool triggerUsesDefaultRedoBinding(const std::optional<InputTrigger>& trigger) {
            const auto* key_trigger = trigger ? std::get_if<KeyTrigger>(&*trigger) : nullptr;
            return key_trigger && key_trigger->key == KEY_Y &&
                   key_trigger->modifiers == MODIFIER_CTRL;
        }

        [[nodiscard]] bool triggersOverlap(const InputTrigger& a, const InputTrigger& b) {
            if (a.index() != b.index()) {
                return false;
            }
            return std::visit([&](const auto& lhs) -> bool {
                using T = std::decay_t<decltype(lhs)>;
                const auto& rhs = std::get<T>(b);
                if constexpr (std::is_same_v<T, KeyTrigger>) {
                    return lhs.key == rhs.key && lhs.modifiers == rhs.modifiers;
                } else if constexpr (std::is_same_v<T, MouseButtonTrigger>) {
                    return lhs.button == rhs.button && lhs.modifiers == rhs.modifiers &&
                           lhs.double_click == rhs.double_click;
                } else if constexpr (std::is_same_v<T, MouseScrollTrigger>) {
                    return lhs.modifiers == rhs.modifiers && lhs.chord_key == rhs.chord_key;
                } else if constexpr (std::is_same_v<T, MouseDragTrigger>) {
                    return lhs.button == rhs.button && lhs.modifiers == rhs.modifiers &&
                           lhs.chord_key == rhs.chord_key;
                }
                return false;
            },
                              a);
        }

        [[nodiscard]] Binding normalizeLoadedBinding(Binding binding) {
            if (!isSelectionDepthAction(binding.action)) {
                return binding;
            }

            if (binding.action == Action::TOGGLE_DEPTH_MODE) {
                binding.action = Action::TOGGLE_SELECTION_DEPTH_FILTER;
            } else if (binding.action == Action::DEPTH_ADJUST_NEAR ||
                       binding.action == Action::DEPTH_ADJUST_SIDE) {
                binding.action = Action::DEPTH_ADJUST_FAR;
            }

            binding.mode = ToolMode::SELECTION;
            binding.description = getActionName(binding.action);
            return binding;
        }

        template <size_t N>
        size_t mirrorLegacyBindingToModes(std::vector<Binding>& bindings,
                                          const Binding& source,
                                          const Action target_action,
                                          const std::array<ToolMode, N>& target_modes) {
            size_t added = 0;
            for (const auto mode : target_modes) {
                const bool already_present = std::ranges::any_of(
                    bindings,
                    [&](const Binding& current) {
                        return current.mode == mode && current.action == target_action;
                    });
                if (already_present) {
                    continue;
                }

                Binding mirrored = source;
                mirrored.mode = mode;
                mirrored.action = target_action;
                mirrored.description = getActionName(target_action);
                bindings.push_back(std::move(mirrored));
                ++added;
            }
            return added;
        }

        size_t projectLegacyGlobalBindings(std::vector<Binding>& bindings, const int version) {
            if (version >= 2) {
                return 0;
            }

            const auto legacy_bindings = bindings;
            size_t added = 0;
            for (const auto& binding : legacy_bindings) {
                if (binding.mode != ToolMode::GLOBAL) {
                    continue;
                }

                switch (binding.action) {
                case Action::NONE:
                    break;
                case Action::DELETE_NODE:
                    added += mirrorLegacyBindingToModes(bindings, binding, Action::DELETE_NODE, DELETE_NODE_MODES);
                    added += mirrorLegacyBindingToModes(bindings, binding, Action::DELETE_SELECTED, DELETE_GAUSSIANS_MODES);
                    break;
                case Action::DELETE_SELECTED:
                    added += mirrorLegacyBindingToModes(bindings, binding, Action::DELETE_SELECTED, DELETE_GAUSSIANS_MODES);
                    break;
                case Action::NODE_PICK:
                    added += mirrorLegacyBindingToModes(bindings, binding, Action::NODE_PICK, NODE_PICK_MODES);
                    break;
                case Action::NODE_RECT_SELECT:
                    added += mirrorLegacyBindingToModes(bindings, binding, Action::NODE_RECT_SELECT, NODE_PICK_MODES);
                    break;
                case Action::TOGGLE_SELECTION_DEPTH_FILTER:
                case Action::TOGGLE_SELECTION_CROP_FILTER:
                case Action::DEPTH_ADJUST_FAR:
                    added += mirrorLegacyBindingToModes(bindings, binding, binding.action, std::array<ToolMode, 1>{ToolMode::SELECTION});
                    break;
                default:
                    if (!describe(binding.action).inherits_from_global) {
                        added += mirrorLegacyBindingToModes(bindings, binding, binding.action, ALL_MODES);
                    }
                    break;
                }
            }
            return added;
        }

    } // namespace

    InputBindings::InputBindings() {
        const auto config_dir = getConfigDir();
        const auto saved_path = config_dir / "Default.json";
        if (std::filesystem::exists(saved_path) && loadProfileFromFile(saved_path)) {
            return;
        }

        auto profile = createDefaultProfile();
        current_profile_name_ = profile.name;
        bindings_ = std::move(profile.bindings);
        rebuildLookupMaps();
    }

    void InputBindings::loadProfile(const std::string& name) {
        const auto config_dir = getConfigDir();
        const auto path = config_dir / (name + ".json");
        if (std::filesystem::exists(path) && loadProfileFromFile(path)) {
            notifyBindingsChanged();
            return;
        }

        if (name == "default" || name == "Default") {
            auto profile = createDefaultProfile();
            current_profile_name_ = profile.name;
            bindings_ = std::move(profile.bindings);
            rebuildLookupMaps();
            notifyBindingsChanged();
        } else {
            LOG_WARN("Unknown profile '{}', using default", name);
            loadProfile("Default");
        }
    }

    void InputBindings::saveProfile(const std::string& name) const {
        const auto config_dir = getConfigDir();
        std::filesystem::create_directories(config_dir);
        const auto path = config_dir / (name + ".json");
        saveProfileToFile(path);
    }

    std::filesystem::path InputBindings::getConfigDir() {
        std::filesystem::path config_dir;
#ifdef _WIN32
        wchar_t path[MAX_PATH];
        if (SUCCEEDED(SHGetFolderPathW(nullptr, CSIDL_APPDATA, nullptr, 0, path))) {
            config_dir = std::filesystem::path(path) / "LichtFeldStudio" / "input_profiles";
        } else {
            config_dir = std::filesystem::current_path() / "config" / "input_profiles";
        }
#else
        const char* home = getenv("HOME");
        if (!home) {
            struct passwd* pw = getpwuid(getuid());
            if (pw)
                home = pw->pw_dir;
        }
        if (home) {
            config_dir = std::filesystem::path(home) / ".config" / "LichtFeldStudio" / "input_profiles";
        } else {
            config_dir = std::filesystem::current_path() / "config" / "input_profiles";
        }
#endif
        return config_dir;
    }

    bool InputBindings::saveProfileToFile(const std::filesystem::path& path) const {
        using json = nlohmann::json;

        json j;
        j["name"] = current_profile_name_;
        j["version"] = PROFILE_VERSION;

        json bindings_array = json::array();
        for (const auto& binding : bindings_) {
            json b;
            b["mode"] = static_cast<int>(binding.mode);
            b["action"] = static_cast<int>(binding.action);
            b["description"] = binding.description;

            std::visit([&b](const auto& trigger) {
                using T = std::decay_t<decltype(trigger)>;
                if constexpr (std::is_same_v<T, KeyTrigger>) {
                    b["trigger_type"] = "key";
                    b["key"] = trigger.key;
                    b["modifiers"] = trigger.modifiers;
                    b["on_repeat"] = trigger.on_repeat;
                } else if constexpr (std::is_same_v<T, MouseButtonTrigger>) {
                    b["trigger_type"] = "mouse_button";
                    b["button"] = static_cast<int>(trigger.button);
                    b["modifiers"] = trigger.modifiers;
                    b["double_click"] = trigger.double_click;
                } else if constexpr (std::is_same_v<T, MouseScrollTrigger>) {
                    b["trigger_type"] = "scroll";
                    b["modifiers"] = trigger.modifiers;
                    if (trigger.chord_key.has_value()) {
                        b["chord_key"] = *trigger.chord_key;
                    }
                } else if constexpr (std::is_same_v<T, MouseDragTrigger>) {
                    b["trigger_type"] = "drag";
                    b["button"] = static_cast<int>(trigger.button);
                    b["modifiers"] = trigger.modifiers;
                    if (trigger.chord_key.has_value()) {
                        b["chord_key"] = *trigger.chord_key;
                    }
                }
            },
                       binding.trigger);

            bindings_array.push_back(b);
        }
        j["bindings"] = bindings_array;

        try {
            std::ofstream file;
            if (!lfs::core::open_file_for_write(path, file)) {
                LOG_ERROR("Failed to open file for writing: {}", lfs::core::path_to_utf8(path));
                return false;
            }
            file << j.dump(4);
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to save profile: {}", e.what());
            return false;
        }
    }

    bool InputBindings::loadProfileFromFile(const std::filesystem::path& path) {
        using json = nlohmann::json;

        try {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, file)) {
                LOG_ERROR("Failed to open profile file: {}", lfs::core::path_to_utf8(path));
                return false;
            }

            const json j = json::parse(file);
            const int version = j.value("version", 0);
            const std::string profile_name = j.value("name", "Custom");

            if (version < 1 || version > PROFILE_VERSION) {
                LOG_WARN("Unknown profile version: {}", version);
            }

            current_profile_name_ = profile_name;
            bindings_.clear();

            for (const auto& b : j["bindings"]) {
                Binding binding;
                // Version 1 had no mode field, default to GLOBAL
                binding.mode = static_cast<ToolMode>(b.value("mode", 0));
                binding.action = static_cast<Action>(b["action"].get<int>());
                binding.description = b.value("description", getActionName(binding.action));

                const std::string trigger_type = b["trigger_type"];
                if (trigger_type == "key") {
                    KeyTrigger trigger;
                    trigger.key = b["key"];
                    trigger.modifiers = b.value("modifiers", 0);
                    trigger.on_repeat = b.value("on_repeat", false);
                    binding.trigger = trigger;
                } else if (trigger_type == "mouse_button") {
                    MouseButtonTrigger trigger;
                    trigger.button = static_cast<MouseButton>(b["button"].get<int>());
                    trigger.modifiers = b.value("modifiers", 0);
                    trigger.double_click = b.value("double_click", false);
                    binding.trigger = trigger;
                } else if (trigger_type == "scroll") {
                    MouseScrollTrigger trigger;
                    trigger.modifiers = b.value("modifiers", 0);
                    if (b.contains("chord_key")) {
                        trigger.chord_key = b["chord_key"].get<int>();
                    }
                    binding.trigger = trigger;
                } else if (trigger_type == "drag") {
                    MouseDragTrigger trigger;
                    trigger.button = static_cast<MouseButton>(b["button"].get<int>());
                    trigger.modifiers = b.value("modifiers", 0);
                    if (b.contains("chord_key")) {
                        trigger.chord_key = b["chord_key"].get<int>();
                    }
                    binding.trigger = trigger;
                }

                binding = normalizeLoadedBinding(std::move(binding));

                if (auto existing = std::find_if(bindings_.begin(), bindings_.end(), [&](const Binding& current) {
                        return current.mode == binding.mode && current.action == binding.action;
                    });
                    existing != bindings_.end()) {
                    *existing = binding;
                } else {
                    bindings_.push_back(binding);
                }
            }

            if (const size_t added_bindings = projectLegacyGlobalBindings(bindings_, version);
                added_bindings > 0) {
                LOG_INFO("Projected {} legacy global bindings into mode-specific shortcuts for profile '{}'",
                         added_bindings, current_profile_name_);
            }

            const size_t collapsed = collapseRedundantModeBindings(version);
            const size_t migrated = collapsed + migrateLoadedProfile(version);

            rebuildLookupMaps();
            LOG_INFO("Loaded profile '{}' ({} bindings) from {}", current_profile_name_, bindings_.size(), lfs::core::path_to_utf8(path));

            // Auto-persist the canonical Default profile so disk stays current
            // after a versioned migration. User-imported files are left untouched;
            // the migration still applies in memory.
            if (migrated > 0 && version < PROFILE_VERSION) {
                std::error_code ec;
                const auto config_default = getConfigDir() / "Default.json";
                if (std::filesystem::equivalent(path, config_default, ec)) {
                    saveProfileToFile(config_default);
                }
            }
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load profile: {}", e.what());
            return false;
        }
    }

    size_t InputBindings::migrateLoadedProfile(const int version) {
        if (version >= PROFILE_VERSION) {
            return 0;
        }

        const Profile defaults = createDefaultProfile();
        size_t added = 0;
        for (const auto& def : defaults.bindings) {
            const bool should_add =
                (version < 6 && def.action == Action::CAMERA_ROLL) ||
                (version < 7 && def.action == Action::BRUSH_RESIZE) ||
                (version < 9 && def.action == Action::CONFIRM_POLYGON) ||
                (version < 10 && def.action == Action::UNDO_POLYGON_VERTEX);
            if (!should_add) {
                continue;
            }
            const bool present = std::ranges::any_of(
                bindings_, [&](const Binding& current) {
                    return current.mode == def.mode && current.action == def.action;
                });
            if (present) {
                continue;
            }
            const bool trigger_in_use = std::ranges::any_of(
                bindings_, [&](const Binding& current) {
                    return current.mode == def.mode &&
                           triggersOverlap(current.trigger, def.trigger);
                });
            if (trigger_in_use) {
                continue;
            }
            LOG_INFO("Migrating profile '{}' with default binding: action={} mode={}",
                     current_profile_name_,
                     getActionName(def.action), static_cast<int>(def.mode));
            bindings_.push_back(def);
            ++added;
        }
        return added;
    }

    size_t InputBindings::collapseRedundantModeBindings(const int version) {
        if (version >= 8) {
            return 0;
        }

        const auto collect_global_triggers = [](const std::vector<Binding>& bindings) {
            std::map<Action, InputTrigger> result;
            for (const auto& binding : bindings) {
                if (binding.mode == ToolMode::GLOBAL) {
                    result[binding.action] = binding.trigger;
                }
            }
            return result;
        };
        const Profile defaults = createDefaultProfile();
        const auto current_globals = collect_global_triggers(bindings_);
        const auto default_globals = collect_global_triggers(defaults.bindings);
        std::map<std::pair<ToolMode, Action>, InputTrigger> default_local_bindings;
        std::map<Action, InputTrigger> default_action_triggers;
        for (const auto& binding : defaults.bindings) {
            default_local_bindings[{binding.mode, binding.action}] = binding.trigger;
            default_action_triggers.try_emplace(binding.action, binding.trigger);
        }

        size_t collapsed = 0;
        std::erase_if(bindings_, [&](const Binding& binding) {
            if (binding.mode == ToolMode::GLOBAL) {
                if (default_local_bindings.find({binding.mode, binding.action}) != default_local_bindings.end()) {
                    return false;
                }
                const auto default_action = default_action_triggers.find(binding.action);
                if (default_action == default_action_triggers.end() ||
                    !triggersOverlap(binding.trigger, default_action->second)) {
                    return false;
                }
                ++collapsed;
                return true;
            }

            if (default_local_bindings.find({binding.mode, binding.action}) != default_local_bindings.end()) {
                return false;
            }

            const auto current_global = current_globals.find(binding.action);
            const auto default_global = default_globals.find(binding.action);
            const auto default_action = default_action_triggers.find(binding.action);
            const bool duplicates_current_global =
                describe(binding.action).inherits_from_global &&
                current_global != current_globals.end() &&
                triggersOverlap(binding.trigger, current_global->second);
            const bool duplicates_default_global =
                describe(binding.action).inherits_from_global &&
                default_global != default_globals.end() &&
                triggersOverlap(binding.trigger, default_global->second);
            const bool duplicates_default_action =
                default_action != default_action_triggers.end() &&
                triggersOverlap(binding.trigger, default_action->second);
            if (!duplicates_current_global &&
                !duplicates_default_global &&
                !duplicates_default_action) {
                return false;
            }

            ++collapsed;
            return true;
        });

        if (collapsed > 0) {
            LOG_INFO("Collapsed {} redundant mode bindings for profile '{}'",
                     collapsed, current_profile_name_);
        }
        return collapsed;
    }

    std::vector<std::string> InputBindings::getAvailableProfiles() const {
        std::vector<std::string> profiles = {"Default"};

        const auto config_dir = getConfigDir();
        if (std::filesystem::exists(config_dir)) {
            for (const auto& entry : std::filesystem::directory_iterator(config_dir)) {
                if (entry.path().extension() == ".json") {
                    const std::string name = lfs::core::path_to_utf8(entry.path().stem());
                    if (name != "Default") {
                        profiles.push_back(name);
                    }
                }
            }
        }

        return profiles;
    }

    Action InputBindings::getActionForKey(ToolMode mode, int key, int modifiers) const {
        const int mods = modifiers & MODIFIER_MASK;
        const auto find_in_mode = [&](ToolMode query_mode) -> Action {
            if (auto it = key_map_.find({query_mode, key, mods}); it != key_map_.end()) {
                return it->second;
            }
            if (mods != MODIFIER_NONE) {
                if (auto it = key_map_.find({query_mode, key, MODIFIER_NONE});
                    it != key_map_.end() &&
                    describe(it->second).allows_extra_modifiers) {
                    return it->second;
                }
            }
            return Action::NONE;
        };

        if (const auto local_action = find_in_mode(mode); local_action != Action::NONE) {
            return local_action;
        }

        if (mode != ToolMode::GLOBAL) {
            const auto global_action = find_in_mode(ToolMode::GLOBAL);
            if (global_action != Action::NONE &&
                describe(global_action).inherits_from_global) {
                return global_action;
            }
        }

        // Support the common redo alias when the profile still uses the default Ctrl+Y binding.
        const auto redo_trigger = [&]() -> std::optional<InputTrigger> {
            if (auto trigger = getTriggerForAction(Action::REDO, mode)) {
                return trigger;
            }
            if (mode != ToolMode::GLOBAL) {
                return getTriggerForAction(Action::REDO, ToolMode::GLOBAL);
            }
            return std::nullopt;
        }();
        if (key == KEY_Z &&
            mods == (MODIFIER_CTRL | MODIFIER_SHIFT) &&
            triggerUsesDefaultRedoBinding(redo_trigger)) {
            return Action::REDO;
        }

        return Action::NONE;
    }

    Action InputBindings::getActionForMouseButton(ToolMode mode, MouseButton button, int modifiers, bool is_double_click) const {
        const int mods = modifiers & MODIFIER_MASK;
        const auto find_in_mode = [&](ToolMode query_mode) -> Action {
            if (auto it = mouse_button_map_.find({query_mode, button, mods, is_double_click}); it != mouse_button_map_.end()) {
                return it->second;
            }
            if (mods != MODIFIER_NONE) {
                if (auto it = mouse_button_map_.find({query_mode, button, MODIFIER_NONE, is_double_click});
                    it != mouse_button_map_.end() &&
                    describe(it->second).allows_extra_modifiers) {
                    return it->second;
                }
            }
            // If double-click, also try a single-click binding in the same mode.
            if (is_double_click) {
                if (auto it = mouse_button_map_.find({query_mode, button, mods, false}); it != mouse_button_map_.end()) {
                    return it->second;
                }
                if (mods != MODIFIER_NONE) {
                    if (auto it = mouse_button_map_.find({query_mode, button, MODIFIER_NONE, false});
                        it != mouse_button_map_.end() &&
                        describe(it->second).allows_extra_modifiers) {
                        return it->second;
                    }
                }
            }
            return Action::NONE;
        };

        if (const auto local_action = find_in_mode(mode); local_action != Action::NONE) {
            return local_action;
        }
        if (mode != ToolMode::GLOBAL) {
            const auto global_action = find_in_mode(ToolMode::GLOBAL);
            if (global_action != Action::NONE &&
                describe(global_action).inherits_from_global) {
                return global_action;
            }
        }
        return Action::NONE;
    }

    Action InputBindings::getActionForScroll(ToolMode mode, int modifiers,
                                             const std::vector<int>& held_keys) const {
        const int mods = modifiers & MODIFIER_MASK;
        const auto find_in_mode = [&](ToolMode query_mode) -> Action {
            for (auto it_key = held_keys.rbegin(); it_key != held_keys.rend(); ++it_key) {
                const int chord = *it_key;
                if (auto it = scroll_chord_map_.find({query_mode, mods, chord}); it != scroll_chord_map_.end()) {
                    return it->second;
                }
            }
            if (auto it = scroll_map_.find({query_mode, mods}); it != scroll_map_.end()) {
                return it->second;
            }
            return Action::NONE;
        };

        if (const auto local_action = find_in_mode(mode); local_action != Action::NONE) {
            return local_action;
        }
        if (mode != ToolMode::GLOBAL) {
            const auto global_action = find_in_mode(ToolMode::GLOBAL);
            if (global_action != Action::NONE &&
                describe(global_action).inherits_from_global) {
                return global_action;
            }
        }
        return Action::NONE;
    }

    Action InputBindings::getActionForDrag(ToolMode mode, MouseButton button, int modifiers,
                                           const std::vector<int>& held_keys) const {
        const int mods = modifiers & MODIFIER_MASK;
        const auto find_in_mode = [&](ToolMode query_mode) -> Action {
            for (auto it_key = held_keys.rbegin(); it_key != held_keys.rend(); ++it_key) {
                const int chord = *it_key;
                if (auto it = drag_chord_map_.find({query_mode, button, mods, chord}); it != drag_chord_map_.end()) {
                    return it->second;
                }
            }
            if (auto it = drag_map_.find({query_mode, button, mods}); it != drag_map_.end()) {
                return it->second;
            }
            if (mods != MODIFIER_NONE) {
                if (auto it = drag_map_.find({query_mode, button, MODIFIER_NONE});
                    it != drag_map_.end() &&
                    describe(it->second).allows_extra_modifiers) {
                    return it->second;
                }
            }
            return Action::NONE;
        };

        if (const auto local_action = find_in_mode(mode); local_action != Action::NONE) {
            return local_action;
        }
        if (mode != ToolMode::GLOBAL) {
            const auto global_action = find_in_mode(ToolMode::GLOBAL);
            if (global_action != Action::NONE &&
                describe(global_action).inherits_from_global) {
                return global_action;
            }
        }
        return Action::NONE;
    }

    std::optional<InputTrigger> InputBindings::getTriggerForAction(Action action, ToolMode mode) const {
        for (const auto& binding : bindings_) {
            if (binding.action == action && binding.mode == mode) {
                return binding.trigger;
            }
        }
        return std::nullopt;
    }

    std::optional<InputTrigger> InputBindings::getEffectiveTriggerForAction(Action action, ToolMode mode) const {
        if (auto trigger = getTriggerForAction(action, mode)) {
            return trigger;
        }
        if (mode != ToolMode::GLOBAL && describe(action).inherits_from_global) {
            return getTriggerForAction(action, ToolMode::GLOBAL);
        }
        return std::nullopt;
    }

    std::string InputBindings::getTriggerDescription(Action action, ToolMode mode) const {
        const auto trigger = getEffectiveTriggerForAction(action, mode);
        if (!trigger) {
            return "Unbound";
        }

        return std::visit([](const auto& t) -> std::string {
            using T = std::decay_t<decltype(t)>;

            std::string result = getModifierString(t.modifiers);
            if (!result.empty())
                result += "+";

            if constexpr (std::is_same_v<T, KeyTrigger>) {
                return result + getKeyName(t.key);
            } else if constexpr (std::is_same_v<T, MouseButtonTrigger>) {
                std::string btn = getMouseButtonName(t.button);
                if (t.double_click) {
                    btn += " Double-Click";
                }
                return result + btn;
            } else if constexpr (std::is_same_v<T, MouseScrollTrigger>) {
                std::string chord = t.chord_key.has_value() ? getKeyName(*t.chord_key) + " + " : "";
                return chord + result + "Scroll";
            } else if constexpr (std::is_same_v<T, MouseDragTrigger>) {
                std::string chord = t.chord_key.has_value() ? getKeyName(*t.chord_key) + " + " : "";
                return chord + result + getMouseButtonName(t.button) + " Drag";
            }
            return "Unknown";
        },
                          *trigger);
    }

    int InputBindings::getKeyForAction(Action action, ToolMode mode) const {
        const auto trigger = getEffectiveTriggerForAction(action, mode);
        if (!trigger)
            return -1;

        if (const auto* key_trigger = std::get_if<KeyTrigger>(&*trigger)) {
            return key_trigger->key;
        }
        return -1;
    }

    void InputBindings::setBinding(ToolMode mode, Action action, const InputTrigger& trigger) {
        clearBinding(mode, action);
        bindings_.push_back({mode, trigger, action, getActionName(action)});
        rebuildLookupMaps();
        notifyBindingsChanged();
    }

    void InputBindings::clearBinding(ToolMode mode, Action action) {
        std::erase_if(bindings_, [mode, action](const Binding& b) {
            return b.mode == mode && b.action == action;
        });
        rebuildLookupMaps();
        notifyBindingsChanged();
    }

    std::optional<BindingConflict> InputBindings::findConflict(
        ToolMode mode, const InputTrigger& trigger, Action ignore_action) const {
        for (const auto& b : bindings_) {
            if (b.action == ignore_action) {
                continue;
            }
            if (b.mode != mode) {
                continue;
            }
            if (triggersOverlap(b.trigger, trigger)) {
                return BindingConflict{b.action, b.mode};
            }
        }

        if (mode == ToolMode::GLOBAL) {
            return std::nullopt;
        }

        for (const auto& b : bindings_) {
            if (b.action == ignore_action) {
                continue;
            }
            if (b.mode != ToolMode::GLOBAL || !describe(b.action).inherits_from_global) {
                continue;
            }
            if (triggersOverlap(b.trigger, trigger)) {
                return BindingConflict{b.action, b.mode};
            }
        }
        return std::nullopt;
    }

    void InputBindings::notifyBindingsChanged() {
        if (on_bindings_changed_) {
            on_bindings_changed_();
        }
    }

    void InputBindings::rebuildLookupMaps() {
        key_map_.clear();
        mouse_button_map_.clear();
        scroll_map_.clear();
        drag_map_.clear();
        scroll_chord_map_.clear();
        drag_chord_map_.clear();

        for (const auto& binding : bindings_) {
            std::visit([&](auto&& t) {
                using T = std::decay_t<decltype(t)>;

                if constexpr (std::is_same_v<T, KeyTrigger>) {
                    key_map_[{binding.mode, t.key, t.modifiers}] = binding.action;
                } else if constexpr (std::is_same_v<T, MouseButtonTrigger>) {
                    mouse_button_map_[{binding.mode, t.button, t.modifiers, t.double_click}] = binding.action;
                } else if constexpr (std::is_same_v<T, MouseScrollTrigger>) {
                    if (t.chord_key.has_value()) {
                        scroll_chord_map_[{binding.mode, t.modifiers, *t.chord_key}] = binding.action;
                    } else {
                        scroll_map_[{binding.mode, t.modifiers}] = binding.action;
                    }
                } else if constexpr (std::is_same_v<T, MouseDragTrigger>) {
                    if (t.chord_key.has_value()) {
                        drag_chord_map_[{binding.mode, t.button, t.modifiers, *t.chord_key}] = binding.action;
                    } else {
                        drag_map_[{binding.mode, t.button, t.modifiers}] = binding.action;
                    }
                }
            },
                       binding.trigger);
        }
    }

    Profile InputBindings::createDefaultProfile() {
        Profile profile;
        profile.name = "Default";
        profile.description = "Default LichtFeld Studio controls";

        // Global bindings are inherited by tool modes unless a mode provides a
        // local override for the same action.
        struct BaseBind {
            InputTrigger trigger;
            Action action;
            const char* desc;
        };
        const std::vector<BaseBind> global = {
            // Camera
            {MouseDragTrigger{MouseButton::MIDDLE, MODIFIER_NONE}, Action::CAMERA_ORBIT, "Orbit"},
            {MouseDragTrigger{MouseButton::RIGHT, MODIFIER_NONE}, Action::CAMERA_PAN, "Pan"},
            {MouseScrollTrigger{MODIFIER_NONE}, Action::CAMERA_ZOOM, "Zoom"},
            {MouseScrollTrigger{MODIFIER_NONE, KEY_R}, Action::CAMERA_ROLL, "Roll"},
            {MouseButtonTrigger{MouseButton::RIGHT, MODIFIER_NONE, true}, Action::CAMERA_SET_PIVOT, "Set pivot"},
            {KeyTrigger{KEY_W, MODIFIER_NONE, true}, Action::CAMERA_MOVE_FORWARD, "Forward"},
            {KeyTrigger{KEY_S, MODIFIER_NONE, true}, Action::CAMERA_MOVE_BACKWARD, "Backward"},
            {KeyTrigger{KEY_A, MODIFIER_NONE, true}, Action::CAMERA_MOVE_LEFT, "Left"},
            {KeyTrigger{KEY_D, MODIFIER_NONE, true}, Action::CAMERA_MOVE_RIGHT, "Right"},
            {KeyTrigger{KEY_H, MODIFIER_NONE}, Action::CAMERA_RESET_HOME, "Home"},
            {KeyTrigger{KEY_F, MODIFIER_NONE}, Action::CAMERA_FOCUS_SELECTION, "Focus selection"},
            {KeyTrigger{KEY_RIGHT, MODIFIER_NONE, true}, Action::CAMERA_NEXT_VIEW, "Next view"},
            {KeyTrigger{KEY_LEFT, MODIFIER_NONE, true}, Action::CAMERA_PREV_VIEW, "Prev view"},
            {KeyTrigger{KEY_EQUAL, MODIFIER_CTRL}, Action::CAMERA_SPEED_UP, "Speed up"},
            {KeyTrigger{KEY_MINUS, MODIFIER_CTRL}, Action::CAMERA_SPEED_DOWN, "Speed down"},
            {KeyTrigger{KEY_KP_ADD, MODIFIER_CTRL}, Action::CAMERA_SPEED_UP, "Speed up"},
            {KeyTrigger{KEY_KP_SUBTRACT, MODIFIER_CTRL}, Action::CAMERA_SPEED_DOWN, "Speed down"},
            {KeyTrigger{KEY_EQUAL, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::ZOOM_SPEED_UP, "Zoom speed up"},
            {KeyTrigger{KEY_MINUS, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::ZOOM_SPEED_DOWN, "Zoom speed down"},
            {KeyTrigger{KEY_KP_ADD, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::ZOOM_SPEED_UP, "Zoom speed up"},
            {KeyTrigger{KEY_KP_SUBTRACT, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::ZOOM_SPEED_DOWN, "Zoom speed down"},
            // View
            {KeyTrigger{KEY_V, MODIFIER_NONE}, Action::TOGGLE_SPLIT_VIEW, "Split view"},
            {KeyTrigger{KEY_V, MODIFIER_SHIFT}, Action::TOGGLE_INDEPENDENT_SPLIT_VIEW, "Independent split"},
            {KeyTrigger{KEY_G, MODIFIER_NONE}, Action::TOGGLE_GT_COMPARISON, "GT comparison"},
            {KeyTrigger{KEY_T, MODIFIER_NONE}, Action::CYCLE_PLY, "Cycle PLY"},
            // Editing (Delete is mode-specific, added below)
            {KeyTrigger{KEY_Z, MODIFIER_CTRL}, Action::UNDO, "Undo"},
            {KeyTrigger{KEY_Y, MODIFIER_CTRL}, Action::REDO, "Redo"},
            {KeyTrigger{KEY_I, MODIFIER_CTRL}, Action::INVERT_SELECTION, "Invert"},
            {KeyTrigger{KEY_D, MODIFIER_CTRL}, Action::DESELECT_ALL, "Deselect"},
            {KeyTrigger{KEY_A, MODIFIER_CTRL}, Action::SELECT_ALL, "Select all"},
            {KeyTrigger{KEY_C, MODIFIER_CTRL}, Action::COPY_SELECTION, "Copy"},
            {KeyTrigger{KEY_V, MODIFIER_CTRL}, Action::PASTE_SELECTION, "Paste"},
            // Selection mode shortcuts
            {KeyTrigger{KEY_T, MODIFIER_CTRL}, Action::CYCLE_SELECTION_VIS, "Sel vis"},
            {KeyTrigger{KEY_1, MODIFIER_CTRL}, Action::SELECT_MODE_CENTERS, "Centers"},
            {KeyTrigger{KEY_2, MODIFIER_CTRL}, Action::SELECT_MODE_RECTANGLE, "Rectangle"},
            {KeyTrigger{KEY_3, MODIFIER_CTRL}, Action::SELECT_MODE_POLYGON, "Polygon"},
            {KeyTrigger{KEY_4, MODIFIER_CTRL}, Action::SELECT_MODE_LASSO, "Lasso"},
            {KeyTrigger{KEY_5, MODIFIER_CTRL}, Action::SELECT_MODE_RINGS, "Rings"},
            {KeyTrigger{KEY_ESCAPE, MODIFIER_NONE}, Action::CANCEL_POLYGON, "Cancel"},
            // UI
            {KeyTrigger{KEY_F12, MODIFIER_NONE}, Action::TOGGLE_UI, "Hide UI"},
            {KeyTrigger{KEY_F11, MODIFIER_NONE}, Action::TOGGLE_FULLSCREEN, "Fullscreen"},
            // Sequencer
            {KeyTrigger{KEY_K, MODIFIER_NONE}, Action::SEQUENCER_ADD_KEYFRAME, "Add keyframe"},
            {KeyTrigger{KEY_U, MODIFIER_NONE}, Action::SEQUENCER_UPDATE_KEYFRAME, "Update keyframe"},
            {KeyTrigger{KEY_SPACE, MODIFIER_NONE}, Action::SEQUENCER_PLAY_PAUSE, "Play/Pause"},
            // Tool shortcuts
            {KeyTrigger{KEY_1}, Action::TOOL_SELECT, "Select"},
            {KeyTrigger{KEY_2}, Action::TOOL_TRANSLATE, "Translate"},
            {KeyTrigger{KEY_3}, Action::TOOL_ROTATE, "Rotate"},
            {KeyTrigger{KEY_4}, Action::TOOL_SCALE, "Scale"},
            {KeyTrigger{KEY_5}, Action::TOOL_MIRROR, "Mirror"},
            {KeyTrigger{KEY_6}, Action::TOOL_BRUSH, "Brush"},
            {KeyTrigger{KEY_7}, Action::TOOL_ALIGN, "Align"},
            {KeyTrigger{KEY_GRAVE_ACCENT}, Action::PIE_MENU, "Pie Menu"},
        };

        for (const auto& b : global) {
            profile.bindings.push_back({ToolMode::GLOBAL, b.trigger, b.action, b.desc});
        }

        const std::vector<BaseBind> selection_drags = {
            {MouseDragTrigger{MouseButton::LEFT, MODIFIER_NONE}, Action::SELECTION_REPLACE, "Select"},
            {MouseDragTrigger{MouseButton::LEFT, MODIFIER_SHIFT}, Action::SELECTION_ADD, "Add sel"},
            {MouseDragTrigger{MouseButton::LEFT, MODIFIER_CTRL}, Action::SELECTION_REMOVE, "Remove sel"},
        };
        for (const auto mode : std::array{ToolMode::SELECTION, ToolMode::BRUSH}) {
            for (const auto& b : selection_drags) {
                profile.bindings.push_back({mode, b.trigger, b.action, b.desc});
            }
        }

        profile.bindings.push_back({ToolMode::SELECTION,
                                    KeyTrigger{KEY_ENTER, MODIFIER_NONE},
                                    Action::CONFIRM_POLYGON,
                                    "Confirm polygon"});
        profile.bindings.push_back({ToolMode::SELECTION,
                                    MouseButtonTrigger{MouseButton::RIGHT, MODIFIER_NONE},
                                    Action::UNDO_POLYGON_VERTEX,
                                    "Undo polygon/cancel selection"});
        profile.bindings.push_back({ToolMode::SELECTION,
                                    KeyTrigger{KEY_X, MODIFIER_NONE},
                                    Action::TOGGLE_SELECTION_DEPTH_FILTER,
                                    "Depth box"});
        profile.bindings.push_back({ToolMode::SELECTION,
                                    MouseScrollTrigger{MODIFIER_ALT},
                                    Action::DEPTH_ADJUST_FAR,
                                    "Depth"});
        profile.bindings.push_back({ToolMode::SELECTION,
                                    MouseScrollTrigger{MODIFIER_CTRL},
                                    Action::BRUSH_RESIZE,
                                    "Brush size"});
        profile.bindings.push_back({ToolMode::SELECTION,
                                    KeyTrigger{KEY_C, MODIFIER_CTRL | MODIFIER_ALT},
                                    Action::TOGGLE_SELECTION_CROP_FILTER,
                                    "Crop filter"});
        profile.bindings.push_back({ToolMode::BRUSH,
                                    MouseScrollTrigger{MODIFIER_CTRL},
                                    Action::BRUSH_RESIZE,
                                    "Brush size"});
        profile.bindings.push_back({ToolMode::BRUSH,
                                    KeyTrigger{KEY_B, MODIFIER_NONE},
                                    Action::CYCLE_BRUSH_MODE,
                                    "Brush mode"});
        profile.bindings.push_back({ToolMode::CROP_BOX,
                                    KeyTrigger{KEY_ENTER, MODIFIER_NONE},
                                    Action::APPLY_CROP_BOX,
                                    "Apply/confirm"});

        // Node picking only for transform modes (not selection/cropbox/brush/align)
        for (const auto mode : NODE_PICK_MODES) {
            profile.bindings.push_back({mode, MouseButtonTrigger{MouseButton::LEFT, MODIFIER_NONE}, Action::NODE_PICK, "Pick node"});
            profile.bindings.push_back({mode, MouseDragTrigger{MouseButton::LEFT, MODIFIER_NONE}, Action::NODE_RECT_SELECT, "Rectangle select nodes"});
        }

        // Delete key: GLOBAL/transform modes delete node, SELECTION/BRUSH delete Gaussians
        for (const auto mode : DELETE_NODE_MODES) {
            profile.bindings.push_back({mode, KeyTrigger{KEY_DELETE, MODIFIER_NONE}, Action::DELETE_NODE, "Delete node"});
        }

        for (const auto mode : DELETE_GAUSSIANS_MODES) {
            profile.bindings.push_back({mode, KeyTrigger{KEY_DELETE, MODIFIER_NONE}, Action::DELETE_SELECTED, "Delete Gaussians"});
        }

        return profile;
    }

    std::string getActionName(const Action action) {
        switch (action) {
        case Action::NONE: return "None";
        case Action::CAMERA_ORBIT: return "Camera Orbit";
        case Action::CAMERA_PAN: return "Camera Pan";
        case Action::CAMERA_ZOOM: return "Camera Zoom";
        case Action::CAMERA_ROLL: return "Camera Roll";
        case Action::CAMERA_MOVE_FORWARD: return "Move Forward";
        case Action::CAMERA_MOVE_BACKWARD: return "Move Backward";
        case Action::CAMERA_MOVE_LEFT: return "Move Left";
        case Action::CAMERA_MOVE_RIGHT: return "Move Right";
        case Action::CAMERA_MOVE_UP: return "Move Up";
        case Action::CAMERA_MOVE_DOWN: return "Move Down";
        case Action::CAMERA_RESET_HOME: return "Go to Home";
        case Action::CAMERA_FOCUS_SELECTION: return "Focus Selection";
        case Action::CAMERA_SET_PIVOT: return "Set Pivot";
        case Action::CAMERA_NEXT_VIEW: return "Next Camera View";
        case Action::CAMERA_PREV_VIEW: return "Previous Camera View";
        case Action::CAMERA_SPEED_UP: return "Increase Move Speed";
        case Action::CAMERA_SPEED_DOWN: return "Decrease Move Speed";
        case Action::ZOOM_SPEED_UP: return "Increase Zoom Speed";
        case Action::ZOOM_SPEED_DOWN: return "Decrease Zoom Speed";
        case Action::TOGGLE_SPLIT_VIEW: return "Toggle Split View";
        case Action::TOGGLE_INDEPENDENT_SPLIT_VIEW: return "Toggle Independent Split View";
        case Action::TOGGLE_GT_COMPARISON: return "Toggle GT Comparison";
        case Action::TOGGLE_DEPTH_MODE: return "Toggle Depth Box";
        case Action::CYCLE_PLY: return "Cycle PLY";
        case Action::DELETE_SELECTED: return "Delete Selected Gaussians";
        case Action::DELETE_NODE: return "Delete Node";
        case Action::UNDO: return "Undo";
        case Action::REDO: return "Redo";
        case Action::INVERT_SELECTION: return "Invert Selection";
        case Action::DESELECT_ALL: return "Deselect All";
        case Action::COPY_SELECTION: return "Copy Selection";
        case Action::PASTE_SELECTION: return "Paste Selection";
        case Action::DEPTH_ADJUST_NEAR: return "Adjust Depth Box";
        case Action::DEPTH_ADJUST_FAR: return "Adjust Depth Box";
        case Action::DEPTH_ADJUST_SIDE: return "Adjust Depth Box";
        case Action::TOGGLE_SELECTION_DEPTH_FILTER: return "Toggle Depth Box";
        case Action::TOGGLE_SELECTION_CROP_FILTER: return "Toggle Selection Crop Filter";
        case Action::BRUSH_RESIZE: return "Resize Brush";
        case Action::CYCLE_BRUSH_MODE: return "Cycle Brush Mode";
        case Action::CONFIRM_POLYGON: return "Confirm Polygon";
        case Action::CANCEL_POLYGON: return "Cancel Polygon";
        case Action::UNDO_POLYGON_VERTEX: return "Undo Polygon Vertex / Cancel Selection";
        case Action::CYCLE_SELECTION_VIS: return "Cycle Selection Visualization";
        case Action::SELECTION_REPLACE: return "Selection: Replace";
        case Action::SELECTION_ADD: return "Selection: Add";
        case Action::SELECTION_REMOVE: return "Selection: Remove";
        case Action::SELECT_MODE_CENTERS: return "Selection: Centers";
        case Action::SELECT_MODE_RECTANGLE: return "Selection: Rectangle";
        case Action::SELECT_MODE_POLYGON: return "Selection: Polygon";
        case Action::SELECT_MODE_LASSO: return "Selection: Lasso";
        case Action::SELECT_MODE_RINGS: return "Selection: Rings";
        case Action::APPLY_CROP_BOX: return "Apply Crop Box";
        case Action::NODE_PICK: return "Pick Node";
        case Action::NODE_RECT_SELECT: return "Rectangle Select Nodes";
        case Action::TOGGLE_UI: return "Toggle UI";
        case Action::TOGGLE_FULLSCREEN: return "Toggle Fullscreen";
        case Action::SEQUENCER_ADD_KEYFRAME: return "Add Keyframe";
        case Action::SEQUENCER_UPDATE_KEYFRAME: return "Update Keyframe";
        case Action::SEQUENCER_PLAY_PAUSE: return "Play/Pause";
        case Action::TOOL_SELECT: return "Select Tool";
        case Action::TOOL_TRANSLATE: return "Translate Tool";
        case Action::TOOL_ROTATE: return "Rotate Tool";
        case Action::TOOL_SCALE: return "Scale Tool";
        case Action::TOOL_MIRROR: return "Mirror Tool";
        case Action::TOOL_BRUSH: return "Brush Tool";
        case Action::TOOL_ALIGN: return "Align Tool";
        case Action::PIE_MENU: return "Pie Menu";
        default: return "Unknown";
        }
    }

    std::string getKeyName(const int key) {
        switch (key) {
        case KEY_A: return "A";
        case KEY_B: return "B";
        case KEY_C: return "C";
        case KEY_D: return "D";
        case KEY_E: return "E";
        case KEY_F: return "F";
        case KEY_G: return "G";
        case KEY_H: return "H";
        case KEY_I: return "I";
        case KEY_J: return "J";
        case KEY_K: return "K";
        case KEY_L: return "L";
        case KEY_M: return "M";
        case KEY_N: return "N";
        case KEY_O: return "O";
        case KEY_P: return "P";
        case KEY_Q: return "Q";
        case KEY_R: return "R";
        case KEY_S: return "S";
        case KEY_T: return "T";
        case KEY_U: return "U";
        case KEY_V: return "V";
        case KEY_W: return "W";
        case KEY_X: return "X";
        case KEY_Y: return "Y";
        case KEY_Z: return "Z";
        case KEY_0: return "0";
        case KEY_1: return "1";
        case KEY_2: return "2";
        case KEY_3: return "3";
        case KEY_4: return "4";
        case KEY_5: return "5";
        case KEY_6: return "6";
        case KEY_7: return "7";
        case KEY_8: return "8";
        case KEY_9: return "9";
        case KEY_SPACE: return "Space";
        case KEY_ENTER: return "Enter";
        case KEY_ESCAPE: return "Escape";
        case KEY_TAB: return "Tab";
        case KEY_BACKSPACE: return "Backspace";
        case KEY_DELETE: return "Delete";
        case KEY_HOME: return "Home";
        case KEY_END: return "End";
        case KEY_PAGE_UP: return "Page Up";
        case KEY_PAGE_DOWN: return "Page Down";
        case KEY_LEFT: return "Left";
        case KEY_RIGHT: return "Right";
        case KEY_UP: return "Up";
        case KEY_DOWN: return "Down";
        case KEY_F1: return "F1";
        case KEY_F2: return "F2";
        case KEY_F3: return "F3";
        case KEY_F4: return "F4";
        case KEY_F5: return "F5";
        case KEY_F6: return "F6";
        case KEY_F7: return "F7";
        case KEY_F8: return "F8";
        case KEY_F9: return "F9";
        case KEY_F10: return "F10";
        case KEY_F11: return "F11";
        case KEY_F12: return "F12";
        case KEY_MINUS: return "-";
        case KEY_EQUAL: return "=";
        case KEY_LEFT_BRACKET: return "[";
        case KEY_RIGHT_BRACKET: return "]";
        case KEY_BACKSLASH: return "\\";
        case KEY_SEMICOLON: return ";";
        case KEY_APOSTROPHE: return "'";
        case KEY_GRAVE_ACCENT: return "`";
        case KEY_COMMA: return ",";
        case KEY_PERIOD: return ".";
        case KEY_SLASH: return "/";
        case KEY_KP_ADD: return "Num+";
        case KEY_KP_SUBTRACT: return "Num-";
        case KEY_KP_MULTIPLY: return "Num*";
        case KEY_KP_DIVIDE: return "Num/";
        case KEY_KP_ENTER: return "NumEnter";
        case KEY_KP_0: return "Num0";
        case KEY_KP_1: return "Num1";
        case KEY_KP_2: return "Num2";
        case KEY_KP_3: return "Num3";
        case KEY_KP_4: return "Num4";
        case KEY_KP_5: return "Num5";
        case KEY_KP_6: return "Num6";
        case KEY_KP_7: return "Num7";
        case KEY_KP_8: return "Num8";
        case KEY_KP_9: return "Num9";
        default: return "Key" + std::to_string(key);
        }
    }

    std::string getMouseButtonName(const MouseButton button) {
        switch (button) {
        case MouseButton::LEFT: return "LMB";
        case MouseButton::RIGHT: return "RMB";
        case MouseButton::MIDDLE: return "MMB";
        default: return "Mouse?";
        }
    }

    std::string getModifierString(const int modifiers) {
        std::string result;
        if (modifiers & MODIFIER_CTRL) {
            result += "Ctrl";
        }
        if (modifiers & MODIFIER_ALT) {
            if (!result.empty())
                result += "+";
            result += "Alt";
        }
        if (modifiers & MODIFIER_SHIFT) {
            if (!result.empty())
                result += "+";
            result += "Shift";
        }
        if (modifiers & MODIFIER_SUPER) {
            if (!result.empty())
                result += "+";
            result += "Super";
        }
        return result;
    }

    void InputBindings::startCapture(ToolMode mode, Action action) {
        capture_state_ = CaptureState{};
        capture_state_.active = true;
        capture_state_.mode = mode;
        capture_state_.action = action;
    }

    void InputBindings::cancelCapture() {
        capture_state_ = CaptureState{};
    }

    void InputBindings::captureKey(int key, int mods) {
        captureKey(key, key, mods);
    }

    void InputBindings::captureKey(const int physical_key, const int logical_key, const int mods) {
        if (!capture_state_.active)
            return;

        const auto& descriptor = describe(capture_state_.action);
        int key = descriptor.prefers_physical_key ? physical_key : logical_key;
        if (key == KEY_UNKNOWN) {
            key = (logical_key != KEY_UNKNOWN) ? logical_key : physical_key;
        }

        // ESC always exits capture, regardless of whether KEY is an allowed kind.
        if (key == KEY_ESCAPE) {
            cancelCapture();
            return;
        }

        if (key == KEY_LEFT_SHIFT || key == KEY_RIGHT_SHIFT ||
            key == KEY_LEFT_CONTROL || key == KEY_RIGHT_CONTROL ||
            key == KEY_LEFT_ALT || key == KEY_RIGHT_ALT ||
            key == KEY_LEFT_SUPER || key == KEY_RIGHT_SUPER) {
            return;
        }

        if (!(descriptor.allowed_kinds & TRIGGER_KIND_KEY)) {
            return;
        }

        // OS-reserved combos: don't let users shadow window-manager shortcuts.
        const bool alt_held = (mods & MODIFIER_ALT) != 0;
        const bool super_held = (mods & MODIFIER_SUPER) != 0;
        if (super_held) {
            return;
        }
        if (alt_held && (key == KEY_TAB || key == KEY_F4)) {
            return;
        }

        const KeyTrigger trigger{key, mods, false};
        setBinding(capture_state_.mode, capture_state_.action, trigger);
        capture_state_.captured = trigger;
        capture_state_.active = false;
    }

    void InputBindings::captureMouseButton(int button, int mods, std::optional<int> chord_key) {
        captureMouseButton(button, mods, 0.0, 0.0, chord_key);
        capture_state_.has_pending_mouse_position = false;
    }

    void InputBindings::captureMouseButton(int button, int mods, double x, double y, std::optional<int> chord_key) {
        if (!capture_state_.active)
            return;

        const uint8_t allowed = describe(capture_state_.action).allowed_kinds;
        if (!(allowed & (TRIGGER_KIND_MOUSE_BUTTON | TRIGGER_KIND_MOUSE_DRAG))) {
            return;
        }

        if (capture_state_.waiting_for_double_click) {
            if (button == capture_state_.pending_button && mods == capture_state_.pending_mods) {
                if (!(allowed & TRIGGER_KIND_MOUSE_BUTTON)) {
                    return;
                }
                const auto mouse_btn = static_cast<MouseButton>(button);
                const MouseButtonTrigger trigger{mouse_btn, mods, true};
                setBinding(capture_state_.mode, capture_state_.action, trigger);
                capture_state_.captured = trigger;
                capture_state_.active = false;
                capture_state_.waiting_for_double_click = false;
                capture_state_.pending_button = -1;
                capture_state_.pending_button_down = false;
                capture_state_.has_pending_mouse_position = false;
                capture_state_.pending_chord_key.reset();
                return;
            }
        }

        capture_state_.waiting_for_double_click = true;
        capture_state_.pending_button = button;
        capture_state_.pending_mods = mods;
        capture_state_.pending_chord_key = chord_key;
        capture_state_.pending_button_down = true;
        capture_state_.has_pending_mouse_position = true;
        capture_state_.pending_mouse_x = x;
        capture_state_.pending_mouse_y = y;
        capture_state_.first_click_time = std::chrono::steady_clock::now();
    }

    void InputBindings::captureMouseButtonRelease(int button) {
        if (!capture_state_.active || !capture_state_.waiting_for_double_click)
            return;

        if (button == capture_state_.pending_button) {
            capture_state_.pending_button_down = false;
        }
    }

    void InputBindings::captureMouseMove(double x, double y) {
        if (!capture_state_.active ||
            !capture_state_.waiting_for_double_click ||
            !capture_state_.pending_button_down ||
            !capture_state_.has_pending_mouse_position) {
            return;
        }

        const auto& descriptor = describe(capture_state_.action);
        if (!(descriptor.allowed_kinds & TRIGGER_KIND_MOUSE_DRAG)) {
            return;
        }

        const double dx = x - capture_state_.pending_mouse_x;
        const double dy = y - capture_state_.pending_mouse_y;
        const double threshold = CaptureState::DRAG_CAPTURE_THRESHOLD_PX;
        if (dx * dx + dy * dy < threshold * threshold) {
            return;
        }

        const auto mouse_btn = static_cast<MouseButton>(capture_state_.pending_button);
        const MouseDragTrigger trigger{mouse_btn, capture_state_.pending_mods,
                                       capture_state_.pending_chord_key};
        setBinding(capture_state_.mode, capture_state_.action, trigger);
        capture_state_.captured = trigger;
        capture_state_.active = false;
        capture_state_.waiting_for_double_click = false;
        capture_state_.pending_button = -1;
        capture_state_.pending_button_down = false;
        capture_state_.has_pending_mouse_position = false;
        capture_state_.pending_chord_key.reset();
    }

    void InputBindings::captureScroll(int mods, std::optional<int> chord_key) {
        if (!capture_state_.active)
            return;
        if (!(describe(capture_state_.action).allowed_kinds & TRIGGER_KIND_MOUSE_SCROLL)) {
            return;
        }
        MouseScrollTrigger trigger{mods & MODIFIER_MASK, chord_key};
        setBinding(capture_state_.mode, capture_state_.action, trigger);
        capture_state_.captured = trigger;
        capture_state_.active = false;
    }

    void InputBindings::updateCapture() {
        if (!capture_state_.active || !capture_state_.waiting_for_double_click)
            return;

        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - capture_state_.first_click_time).count();

        if (elapsed >= CaptureState::DOUBLE_CLICK_WAIT_TIME) {
            const auto mouse_btn = static_cast<MouseButton>(capture_state_.pending_button);
            const auto& descriptor = describe(capture_state_.action);
            const bool can_button = descriptor.allowed_kinds & TRIGGER_KIND_MOUSE_BUTTON;
            const bool can_drag = descriptor.allowed_kinds & TRIGGER_KIND_MOUSE_DRAG;
            if (capture_state_.pending_button_down &&
                capture_state_.has_pending_mouse_position &&
                can_drag) {
                return;
            }
            const bool produce_button = can_button && (!can_drag || descriptor.prefers_single_click);

            if (produce_button) {
                const MouseButtonTrigger trigger{mouse_btn, capture_state_.pending_mods, false};
                setBinding(capture_state_.mode, capture_state_.action, trigger);
                capture_state_.captured = trigger;
            } else if (can_drag) {
                const MouseDragTrigger trigger{mouse_btn, capture_state_.pending_mods,
                                               capture_state_.pending_chord_key};
                setBinding(capture_state_.mode, capture_state_.action, trigger);
                capture_state_.captured = trigger;
            }
            capture_state_.active = false;
            capture_state_.waiting_for_double_click = false;
            capture_state_.pending_button = -1;
            capture_state_.pending_button_down = false;
            capture_state_.has_pending_mouse_position = false;
            capture_state_.pending_chord_key.reset();
        }
    }

    std::optional<InputTrigger> InputBindings::getAndClearCaptured() {
        const auto result = capture_state_.captured;
        capture_state_.captured.reset();
        return result;
    }

    std::vector<std::pair<Action, std::string>> InputBindings::getBindingsForMode(ToolMode mode) const {
        std::vector<std::pair<Action, std::string>> result;
        for (const auto& binding : bindings_) {
            if (binding.mode == mode) {
                result.emplace_back(binding.action, getTriggerDescription(binding.action, mode));
            }
        }
        return result;
    }

    const ActionDescriptor& describe(const Action action) {
        using K = TriggerKindFlag;

        static constexpr ActionDescriptor d_none{};

        // Camera continuous-axis (scroll only).
        static constexpr ActionDescriptor d_zoom{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_SCROLL,
            .inherits_from_global = true,
            .ui_section = ActionSection::Navigation,
        };
        static constexpr ActionDescriptor d_roll{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_SCROLL,
            .inherits_from_global = true,
            .ui_section = ActionSection::Navigation,
        };
        // Orbit / pan: prefer drag, but accept a single mouse-button capture and
        // tolerate extra modifiers when nothing more specific is bound.
        static constexpr ActionDescriptor d_orbit_pan{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_DRAG | K::TRIGGER_KIND_MOUSE_BUTTON,
            .allows_extra_modifiers = true,
            .prefers_single_click = true,
            .inherits_from_global = true,
            .ui_section = ActionSection::Navigation,
        };
        static constexpr ActionDescriptor d_set_pivot{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_BUTTON,
            .allows_extra_modifiers = true,
            .prefers_single_click = true,
            .inherits_from_global = true,
            .ui_section = ActionSection::Navigation,
        };
        static constexpr ActionDescriptor d_movement{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .prefers_physical_key = true,
            .inherits_from_global = true,
            .ui_section = ActionSection::Navigation,
        };
        static constexpr ActionDescriptor d_camera_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .ui_section = ActionSection::Navigation,
        };
        static constexpr ActionDescriptor d_camera_global_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .ui_section = ActionSection::NavigationGlobal,
        };

        // Selection drags (LMB by default).
        static constexpr ActionDescriptor d_selection_drag{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_DRAG | K::TRIGGER_KIND_MOUSE_BUTTON,
            .ui_section = ActionSection::Selection,
        };
        static constexpr ActionDescriptor d_selection_mode_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .ui_section = ActionSection::SelectionGlobal,
        };

        // Depth filter actions.
        static constexpr ActionDescriptor d_depth_scroll{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_SCROLL,
            .ui_section = ActionSection::Depth,
        };
        static constexpr ActionDescriptor d_depth_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .ui_section = ActionSection::Depth,
        };

        // Editing — these inherit from GLOBAL when active mode lacks the binding.
        static constexpr ActionDescriptor d_editing_inherit{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .ui_section = ActionSection::Editing,
        };
        static constexpr ActionDescriptor d_editing_local{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .ui_section = ActionSection::Editing,
        };
        static constexpr ActionDescriptor d_selection_modal_local_control{
            .allowed_kinds = K::TRIGGER_KIND_KEY |
                             K::TRIGGER_KIND_MOUSE_BUTTON |
                             K::TRIGGER_KIND_MOUSE_DRAG,
            .allows_extra_modifiers = true,
            .ui_section = ActionSection::SelectionGlobal,
        };
        static constexpr ActionDescriptor d_selection_modal_inherit_control{
            .allowed_kinds = K::TRIGGER_KIND_KEY |
                             K::TRIGGER_KIND_MOUSE_BUTTON |
                             K::TRIGGER_KIND_MOUSE_DRAG,
            .allows_extra_modifiers = true,
            .inherits_from_global = true,
            .ui_section = ActionSection::SelectionGlobal,
        };
        static constexpr ActionDescriptor d_polygon_confirm{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .allows_extra_modifiers = true,
            .ui_section = ActionSection::SelectionGlobal,
        };

        static constexpr ActionDescriptor d_brush_scroll{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_SCROLL,
            .ui_section = ActionSection::Brush,
        };
        static constexpr ActionDescriptor d_brush_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .ui_section = ActionSection::Brush,
        };

        static constexpr ActionDescriptor d_crop_box_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .ui_section = ActionSection::CropBox,
        };

        static constexpr ActionDescriptor d_view_global_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .ui_section = ActionSection::ViewGlobal,
        };
        static constexpr ActionDescriptor d_tools_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .ui_section = ActionSection::Tools,
        };
        static constexpr ActionDescriptor d_sequencer_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .ui_section = ActionSection::Sequencer,
        };
        static constexpr ActionDescriptor d_ui_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .ui_section = ActionSection::UI,
        };

        static constexpr ActionDescriptor d_node_pick{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_BUTTON,
            .ui_section = ActionSection::NodePicking,
        };
        static constexpr ActionDescriptor d_node_rect{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_DRAG,
            .ui_section = ActionSection::NodePicking,
        };

        switch (action) {
        case Action::NONE: return d_none;

        case Action::CAMERA_ORBIT:
        case Action::CAMERA_PAN:
            return d_orbit_pan;
        case Action::CAMERA_ZOOM:
            return d_zoom;
        case Action::CAMERA_ROLL:
            return d_roll;
        case Action::CAMERA_SET_PIVOT:
            return d_set_pivot;

        case Action::CAMERA_MOVE_FORWARD:
        case Action::CAMERA_MOVE_BACKWARD:
        case Action::CAMERA_MOVE_LEFT:
        case Action::CAMERA_MOVE_RIGHT:
        case Action::CAMERA_MOVE_UP:
        case Action::CAMERA_MOVE_DOWN:
            return d_movement;

        case Action::CAMERA_RESET_HOME:
        case Action::CAMERA_FOCUS_SELECTION:
        case Action::CAMERA_NEXT_VIEW:
        case Action::CAMERA_PREV_VIEW:
            return d_camera_global_key;
        case Action::CAMERA_SPEED_UP:
        case Action::CAMERA_SPEED_DOWN:
        case Action::ZOOM_SPEED_UP:
        case Action::ZOOM_SPEED_DOWN:
            return d_camera_key;

        case Action::TOGGLE_SPLIT_VIEW:
        case Action::TOGGLE_INDEPENDENT_SPLIT_VIEW:
        case Action::TOGGLE_GT_COMPARISON:
        case Action::CYCLE_PLY:
        case Action::CYCLE_SELECTION_VIS:
            return d_view_global_key;

        case Action::DELETE_SELECTED:
        case Action::DELETE_NODE:
            return d_editing_local;
        case Action::UNDO:
        case Action::REDO:
        case Action::INVERT_SELECTION:
        case Action::DESELECT_ALL:
        case Action::SELECT_ALL:
        case Action::COPY_SELECTION:
        case Action::PASTE_SELECTION:
            return d_editing_inherit;

        case Action::TOGGLE_DEPTH_MODE:
        case Action::TOGGLE_SELECTION_DEPTH_FILTER:
        case Action::TOGGLE_SELECTION_CROP_FILTER:
            return d_depth_key;
        case Action::DEPTH_ADJUST_FAR:
        case Action::DEPTH_ADJUST_NEAR:
        case Action::DEPTH_ADJUST_SIDE:
            return d_depth_scroll;

        case Action::BRUSH_RESIZE:
            return d_brush_scroll;
        case Action::CYCLE_BRUSH_MODE:
            return d_brush_key;

        case Action::CONFIRM_POLYGON:
            return d_polygon_confirm;
        case Action::UNDO_POLYGON_VERTEX:
            return d_selection_modal_local_control;
        case Action::CANCEL_POLYGON:
            return d_selection_modal_inherit_control;

        case Action::SELECTION_REPLACE:
        case Action::SELECTION_ADD:
        case Action::SELECTION_REMOVE:
            return d_selection_drag;
        case Action::SELECT_MODE_CENTERS:
        case Action::SELECT_MODE_RECTANGLE:
        case Action::SELECT_MODE_POLYGON:
        case Action::SELECT_MODE_LASSO:
        case Action::SELECT_MODE_RINGS:
            return d_selection_mode_key;

        case Action::APPLY_CROP_BOX:
            return d_crop_box_key;

        case Action::NODE_PICK:
            return d_node_pick;
        case Action::NODE_RECT_SELECT:
            return d_node_rect;

        case Action::TOGGLE_UI:
        case Action::TOGGLE_FULLSCREEN:
            return d_ui_key;

        case Action::SEQUENCER_ADD_KEYFRAME:
        case Action::SEQUENCER_UPDATE_KEYFRAME:
        case Action::SEQUENCER_PLAY_PAUSE:
            return d_sequencer_key;

        case Action::TOOL_SELECT:
        case Action::TOOL_TRANSLATE:
        case Action::TOOL_ROTATE:
        case Action::TOOL_SCALE:
        case Action::TOOL_MIRROR:
        case Action::TOOL_BRUSH:
        case Action::TOOL_ALIGN:
            return d_tools_key;

        case Action::PIE_MENU:
            return d_ui_key;
        }
        return d_none;
    }

    ShortcutScope shortcutScopeForAction(const Action action) {
        switch (action) {
        case Action::TOOL_SELECT:
        case Action::TOOL_TRANSLATE:
        case Action::TOOL_ROTATE:
        case Action::TOOL_SCALE:
        case Action::TOOL_MIRROR:
        case Action::TOOL_BRUSH:
        case Action::TOOL_ALIGN:
        case Action::TOGGLE_UI:
        case Action::TOGGLE_FULLSCREEN:
        case Action::SELECT_MODE_CENTERS:
        case Action::SELECT_MODE_RECTANGLE:
        case Action::SELECT_MODE_POLYGON:
        case Action::SELECT_MODE_LASSO:
        case Action::SELECT_MODE_RINGS:
        case Action::UNDO:
        case Action::REDO:
        case Action::DELETE_SELECTED:
        case Action::DELETE_NODE:
        case Action::INVERT_SELECTION:
        case Action::DESELECT_ALL:
        case Action::SELECT_ALL:
        case Action::COPY_SELECTION:
        case Action::PASTE_SELECTION:
        case Action::TOGGLE_DEPTH_MODE:
        case Action::TOGGLE_SELECTION_DEPTH_FILTER:
        case Action::TOGGLE_SELECTION_CROP_FILTER:
        case Action::SEQUENCER_ADD_KEYFRAME:
        case Action::SEQUENCER_UPDATE_KEYFRAME:
        case Action::SEQUENCER_PLAY_PAUSE:
            return ShortcutScope::GlobalWhenNotTextEditing;

        case Action::CAMERA_MOVE_FORWARD:
        case Action::CAMERA_MOVE_BACKWARD:
        case Action::CAMERA_MOVE_LEFT:
        case Action::CAMERA_MOVE_RIGHT:
        case Action::CAMERA_MOVE_UP:
        case Action::CAMERA_MOVE_DOWN:
        case Action::CAMERA_RESET_HOME:
        case Action::CAMERA_FOCUS_SELECTION:
        case Action::CAMERA_SET_PIVOT:
        case Action::CAMERA_NEXT_VIEW:
        case Action::CAMERA_PREV_VIEW:
        case Action::CAMERA_SPEED_UP:
        case Action::CAMERA_SPEED_DOWN:
        case Action::ZOOM_SPEED_UP:
        case Action::ZOOM_SPEED_DOWN:
        case Action::TOGGLE_SPLIT_VIEW:
        case Action::TOGGLE_INDEPENDENT_SPLIT_VIEW:
        case Action::TOGGLE_GT_COMPARISON:
        case Action::CYCLE_SELECTION_VIS:
        case Action::PIE_MENU:
            return ShortcutScope::Viewport;

        default:
            return ShortcutScope::Global;
        }
    }

} // namespace lfs::vis::input
