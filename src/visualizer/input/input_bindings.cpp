/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_bindings.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <shlobj.h>
#else
#include <pwd.h>
#include <unistd.h>
#endif

namespace lfs::vis::input {

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

        constexpr int PROFILE_VERSION = 2; // Version 2 adds tool mode

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
                } else if constexpr (std::is_same_v<T, MouseDragTrigger>) {
                    b["trigger_type"] = "drag";
                    b["button"] = static_cast<int>(trigger.button);
                    b["modifiers"] = trigger.modifiers;
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
            if (version < 1 || version > 2) {
                LOG_WARN("Unknown profile version: {}", version);
            }

            current_profile_name_ = j.value("name", "Custom");
            bindings_.clear();

            for (const auto& b : j["bindings"]) {
                Binding binding;
                // Version 1 had no mode field, default to GLOBAL
                binding.mode = static_cast<ToolMode>(b.value("mode", 0));
                binding.action = static_cast<Action>(b["action"].get<int>());
                binding.description = b.value("description", "");

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
                    binding.trigger = trigger;
                } else if (trigger_type == "drag") {
                    MouseDragTrigger trigger;
                    trigger.button = static_cast<MouseButton>(b["button"].get<int>());
                    trigger.modifiers = b.value("modifiers", 0);
                    binding.trigger = trigger;
                }

                bindings_.push_back(binding);
            }

            rebuildLookupMaps();
            LOG_INFO("Loaded profile '{}' ({} bindings) from {}", current_profile_name_, bindings_.size(), lfs::core::path_to_utf8(path));
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load profile: {}", e.what());
            return false;
        }
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
        if (auto it = key_map_.find({mode, key, mods}); it != key_map_.end()) {
            return it->second;
        }
        return Action::NONE;
    }

    Action InputBindings::getActionForMouseButton(ToolMode mode, MouseButton button, int modifiers, bool is_double_click) const {
        const int mods = modifiers & MODIFIER_MASK;
        if (auto it = mouse_button_map_.find({mode, button, mods, is_double_click}); it != mouse_button_map_.end()) {
            return it->second;
        }
        // If double-click, also try single-click binding in same mode
        if (is_double_click) {
            if (auto it = mouse_button_map_.find({mode, button, mods, false}); it != mouse_button_map_.end()) {
                return it->second;
            }
        }
        return Action::NONE;
    }

    Action InputBindings::getActionForScroll(ToolMode mode, int modifiers) const {
        const int mods = modifiers & MODIFIER_MASK;
        if (auto it = scroll_map_.find({mode, mods}); it != scroll_map_.end()) {
            return it->second;
        }
        return Action::NONE;
    }

    Action InputBindings::getActionForDrag(ToolMode mode, MouseButton button, int modifiers) const {
        const int mods = modifiers & MODIFIER_MASK;
        if (auto it = drag_map_.find({mode, button, mods}); it != drag_map_.end()) {
            return it->second;
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

    std::string InputBindings::getTriggerDescription(Action action, ToolMode mode) const {
        const auto trigger = getTriggerForAction(action, mode);
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
                return result + "Scroll";
            } else if constexpr (std::is_same_v<T, MouseDragTrigger>) {
                return result + getMouseButtonName(t.button) + " Drag";
            }
            return "Unknown";
        },
                          *trigger);
    }

    int InputBindings::getKeyForAction(Action action, ToolMode mode) const {
        const auto trigger = getTriggerForAction(action, mode);
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

        for (const auto& binding : bindings_) {
            std::visit([&](auto&& t) {
                using T = std::decay_t<decltype(t)>;

                if constexpr (std::is_same_v<T, KeyTrigger>) {
                    key_map_[{binding.mode, t.key, t.modifiers}] = binding.action;
                } else if constexpr (std::is_same_v<T, MouseButtonTrigger>) {
                    mouse_button_map_[{binding.mode, t.button, t.modifiers, t.double_click}] = binding.action;
                } else if constexpr (std::is_same_v<T, MouseScrollTrigger>) {
                    scroll_map_[{binding.mode, t.modifiers}] = binding.action;
                } else if constexpr (std::is_same_v<T, MouseDragTrigger>) {
                    drag_map_[{binding.mode, t.button, t.modifiers}] = binding.action;
                }
            },
                       binding.trigger);
        }
    }

    Profile InputBindings::createDefaultProfile() {
        Profile profile;
        profile.name = "Default";
        profile.description = "Default LichtFeld Studio controls";

        // Base bindings - will be duplicated for each tool mode
        struct BaseBind {
            InputTrigger trigger;
            Action action;
            const char* desc;
        };
        std::vector<BaseBind> base = {
            // Camera
            {MouseDragTrigger{MouseButton::MIDDLE, MODIFIER_NONE}, Action::CAMERA_ORBIT, "Orbit"},
            {MouseDragTrigger{MouseButton::RIGHT, MODIFIER_NONE}, Action::CAMERA_PAN, "Pan"},
            {MouseScrollTrigger{MODIFIER_NONE}, Action::CAMERA_ZOOM, "Zoom"},
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
            {KeyTrigger{KEY_G, MODIFIER_NONE}, Action::TOGGLE_GT_COMPARISON, "GT comparison"},
            {KeyTrigger{KEY_T, MODIFIER_NONE}, Action::CYCLE_PLY, "Cycle PLY"},
            // Depth
            {KeyTrigger{KEY_F, MODIFIER_CTRL}, Action::TOGGLE_DEPTH_MODE, "Depth filter"},
            {MouseScrollTrigger{MODIFIER_ALT}, Action::DEPTH_ADJUST_FAR, "Depth far"},
            {MouseScrollTrigger{MODIFIER_ALT | MODIFIER_CTRL}, Action::DEPTH_ADJUST_SIDE, "Depth side"},
            // Editing (Delete is mode-specific, added below)
            {KeyTrigger{KEY_Z, MODIFIER_CTRL}, Action::UNDO, "Undo"},
            {KeyTrigger{KEY_Y, MODIFIER_CTRL}, Action::REDO, "Redo"},
            {KeyTrigger{KEY_I, MODIFIER_CTRL}, Action::INVERT_SELECTION, "Invert"},
            {KeyTrigger{KEY_D, MODIFIER_CTRL}, Action::DESELECT_ALL, "Deselect"},
            {KeyTrigger{KEY_A, MODIFIER_CTRL}, Action::SELECT_ALL, "Select all"},
            {KeyTrigger{KEY_C, MODIFIER_CTRL}, Action::COPY_SELECTION, "Copy"},
            {KeyTrigger{KEY_V, MODIFIER_CTRL}, Action::PASTE_SELECTION, "Paste"},
            // Tools
            {KeyTrigger{KEY_B, MODIFIER_NONE}, Action::CYCLE_BRUSH_MODE, "Brush mode"},
            {KeyTrigger{KEY_T, MODIFIER_CTRL}, Action::CYCLE_SELECTION_VIS, "Sel vis"},
            {KeyTrigger{KEY_ENTER, MODIFIER_NONE}, Action::APPLY_CROP_BOX, "Apply/confirm"},
            {KeyTrigger{KEY_ESCAPE, MODIFIER_NONE}, Action::CANCEL_POLYGON, "Cancel"},
            // Selection
            {MouseDragTrigger{MouseButton::LEFT, MODIFIER_NONE}, Action::SELECTION_REPLACE, "Select"},
            {MouseDragTrigger{MouseButton::LEFT, MODIFIER_SHIFT}, Action::SELECTION_ADD, "Add sel"},
            {MouseDragTrigger{MouseButton::LEFT, MODIFIER_CTRL}, Action::SELECTION_REMOVE, "Remove sel"},
            {KeyTrigger{KEY_1, MODIFIER_CTRL}, Action::SELECT_MODE_CENTERS, "Centers"},
            {KeyTrigger{KEY_2, MODIFIER_CTRL}, Action::SELECT_MODE_RECTANGLE, "Rectangle"},
            {KeyTrigger{KEY_3, MODIFIER_CTRL}, Action::SELECT_MODE_POLYGON, "Polygon"},
            {KeyTrigger{KEY_4, MODIFIER_CTRL}, Action::SELECT_MODE_LASSO, "Lasso"},
            {KeyTrigger{KEY_5, MODIFIER_CTRL}, Action::SELECT_MODE_RINGS, "Rings"},
            // UI
            {KeyTrigger{KEY_F12, MODIFIER_NONE}, Action::TOGGLE_UI, "Hide UI"},
            {KeyTrigger{KEY_F11, MODIFIER_NONE}, Action::TOGGLE_FULLSCREEN, "Fullscreen"},
            // Sequencer
            {KeyTrigger{KEY_K, MODIFIER_NONE}, Action::SEQUENCER_ADD_KEYFRAME, "Add keyframe"},
            {KeyTrigger{KEY_U, MODIFIER_NONE}, Action::SEQUENCER_UPDATE_KEYFRAME, "Update keyframe"},
            {KeyTrigger{KEY_SPACE, MODIFIER_NONE}, Action::SEQUENCER_PLAY_PAUSE, "Play/Pause"},
        };

        constexpr ToolMode ALL_MODES[] = {
            ToolMode::GLOBAL,
            ToolMode::SELECTION,
            ToolMode::BRUSH,
            ToolMode::ALIGN,
            ToolMode::CROP_BOX,
            ToolMode::TRANSLATE,
            ToolMode::ROTATE,
            ToolMode::SCALE,
        };

        for (const auto mode : ALL_MODES) {
            for (const auto& b : base) {
                profile.bindings.push_back({mode, b.trigger, b.action, b.desc});
            }
        }

        // Node picking only for transform modes (not selection/cropbox/brush/align)
        constexpr ToolMode NODE_PICK_MODES[] = {
            ToolMode::GLOBAL,
            ToolMode::TRANSLATE,
            ToolMode::ROTATE,
            ToolMode::SCALE,
        };

        for (const auto mode : NODE_PICK_MODES) {
            profile.bindings.push_back({mode, MouseButtonTrigger{MouseButton::LEFT, MODIFIER_NONE}, Action::NODE_PICK, "Pick node"});
            profile.bindings.push_back({mode, MouseDragTrigger{MouseButton::LEFT, MODIFIER_NONE}, Action::NODE_RECT_SELECT, "Rectangle select nodes"});
        }

        // Delete key: GLOBAL/transform modes delete node, SELECTION/BRUSH delete Gaussians
        constexpr ToolMode DELETE_NODE_MODES[] = {
            ToolMode::GLOBAL,
            ToolMode::TRANSLATE,
            ToolMode::ROTATE,
            ToolMode::SCALE,
        };
        for (const auto mode : DELETE_NODE_MODES) {
            profile.bindings.push_back({mode, KeyTrigger{KEY_DELETE, MODIFIER_NONE}, Action::DELETE_NODE, "Delete node"});
        }

        constexpr ToolMode DELETE_GAUSSIANS_MODES[] = {
            ToolMode::SELECTION,
            ToolMode::BRUSH,
            ToolMode::ALIGN,
            ToolMode::CROP_BOX,
        };
        for (const auto mode : DELETE_GAUSSIANS_MODES) {
            profile.bindings.push_back({mode, KeyTrigger{KEY_DELETE, MODIFIER_NONE}, Action::DELETE_SELECTED, "Delete Gaussians"});
        }

        // Tool shortcuts (GLOBAL mode only, number keys 1-7)
        profile.bindings.push_back({ToolMode::GLOBAL, KeyTrigger{KEY_1}, Action::TOOL_SELECT, "Select"});
        profile.bindings.push_back({ToolMode::GLOBAL, KeyTrigger{KEY_2}, Action::TOOL_TRANSLATE, "Translate"});
        profile.bindings.push_back({ToolMode::GLOBAL, KeyTrigger{KEY_3}, Action::TOOL_ROTATE, "Rotate"});
        profile.bindings.push_back({ToolMode::GLOBAL, KeyTrigger{KEY_4}, Action::TOOL_SCALE, "Scale"});
        profile.bindings.push_back({ToolMode::GLOBAL, KeyTrigger{KEY_5}, Action::TOOL_MIRROR, "Mirror"});
        profile.bindings.push_back({ToolMode::GLOBAL, KeyTrigger{KEY_6}, Action::TOOL_BRUSH, "Brush"});
        profile.bindings.push_back({ToolMode::GLOBAL, KeyTrigger{KEY_7}, Action::TOOL_ALIGN, "Align"});

        // Pie menu (all modes)
        for (const auto mode : ALL_MODES) {
            profile.bindings.push_back({mode, KeyTrigger{KEY_GRAVE_ACCENT}, Action::PIE_MENU, "Pie Menu"});
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
        case Action::TOGGLE_GT_COMPARISON: return "Toggle GT Comparison";
        case Action::TOGGLE_DEPTH_MODE: return "Toggle Depth Mode";
        case Action::CYCLE_PLY: return "Cycle PLY";
        case Action::DELETE_SELECTED: return "Delete Selected Gaussians";
        case Action::DELETE_NODE: return "Delete Node";
        case Action::UNDO: return "Undo";
        case Action::REDO: return "Redo";
        case Action::INVERT_SELECTION: return "Invert Selection";
        case Action::DESELECT_ALL: return "Deselect All";
        case Action::COPY_SELECTION: return "Copy Selection";
        case Action::PASTE_SELECTION: return "Paste Selection";
        case Action::DEPTH_ADJUST_FAR: return "Adjust Depth Far";
        case Action::DEPTH_ADJUST_SIDE: return "Adjust Depth Side";
        case Action::BRUSH_RESIZE: return "Resize Brush";
        case Action::CYCLE_BRUSH_MODE: return "Cycle Brush Mode";
        case Action::CONFIRM_POLYGON: return "Confirm Polygon";
        case Action::CANCEL_POLYGON: return "Cancel Polygon";
        case Action::UNDO_POLYGON_VERTEX: return "Undo Polygon Vertex";
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
        if (!capture_state_.active)
            return;

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

        const KeyTrigger trigger{key, mods, false};
        setBinding(capture_state_.mode, capture_state_.action, trigger);
        capture_state_.captured = trigger;
        capture_state_.active = false;
    }

    void InputBindings::captureMouseButton(int button, int mods) {
        if (!capture_state_.active)
            return;

        if (capture_state_.waiting_for_double_click) {
            if (button == capture_state_.pending_button && mods == capture_state_.pending_mods) {
                const auto mouse_btn = static_cast<MouseButton>(button);
                const MouseButtonTrigger trigger{mouse_btn, mods, true};
                setBinding(capture_state_.mode, capture_state_.action, trigger);
                capture_state_.captured = trigger;
                capture_state_.active = false;
                capture_state_.waiting_for_double_click = false;
                capture_state_.pending_button = -1;
                return;
            }
        }

        capture_state_.waiting_for_double_click = true;
        capture_state_.pending_button = button;
        capture_state_.pending_mods = mods;
        capture_state_.first_click_time = std::chrono::steady_clock::now();
    }

    void InputBindings::updateCapture() {
        if (!capture_state_.active || !capture_state_.waiting_for_double_click)
            return;

        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - capture_state_.first_click_time).count();

        if (elapsed >= CaptureState::DOUBLE_CLICK_WAIT_TIME) {
            const auto mouse_btn = static_cast<MouseButton>(capture_state_.pending_button);
            const MouseDragTrigger trigger{mouse_btn, capture_state_.pending_mods};
            setBinding(capture_state_.mode, capture_state_.action, trigger);
            capture_state_.captured = trigger;
            capture_state_.active = false;
            capture_state_.waiting_for_double_click = false;
            capture_state_.pending_button = -1;
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

} // namespace lfs::vis::input
