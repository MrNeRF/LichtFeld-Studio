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
            {KeyTrigger{GLFW_KEY_W, MODIFIER_NONE, true}, Action::CAMERA_MOVE_FORWARD, "Forward"},
            {KeyTrigger{GLFW_KEY_S, MODIFIER_NONE, true}, Action::CAMERA_MOVE_BACKWARD, "Backward"},
            {KeyTrigger{GLFW_KEY_A, MODIFIER_NONE, true}, Action::CAMERA_MOVE_LEFT, "Left"},
            {KeyTrigger{GLFW_KEY_D, MODIFIER_NONE, true}, Action::CAMERA_MOVE_RIGHT, "Right"},
            {KeyTrigger{GLFW_KEY_Q, MODIFIER_NONE, true}, Action::CAMERA_MOVE_DOWN, "Down"},
            {KeyTrigger{GLFW_KEY_E, MODIFIER_NONE, true}, Action::CAMERA_MOVE_UP, "Up"},
            {KeyTrigger{GLFW_KEY_H, MODIFIER_NONE}, Action::CAMERA_RESET_HOME, "Home"},
            {KeyTrigger{GLFW_KEY_F, MODIFIER_NONE}, Action::CAMERA_FOCUS_SELECTION, "Focus selection"},
            {KeyTrigger{GLFW_KEY_RIGHT, MODIFIER_NONE, true}, Action::CAMERA_NEXT_VIEW, "Next view"},
            {KeyTrigger{GLFW_KEY_LEFT, MODIFIER_NONE, true}, Action::CAMERA_PREV_VIEW, "Prev view"},
            {KeyTrigger{GLFW_KEY_EQUAL, MODIFIER_CTRL}, Action::CAMERA_SPEED_UP, "Speed up"},
            {KeyTrigger{GLFW_KEY_MINUS, MODIFIER_CTRL}, Action::CAMERA_SPEED_DOWN, "Speed down"},
            {KeyTrigger{GLFW_KEY_KP_ADD, MODIFIER_CTRL}, Action::CAMERA_SPEED_UP, "Speed up"},
            {KeyTrigger{GLFW_KEY_KP_SUBTRACT, MODIFIER_CTRL}, Action::CAMERA_SPEED_DOWN, "Speed down"},
            {KeyTrigger{GLFW_KEY_EQUAL, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::ZOOM_SPEED_UP, "Zoom speed up"},
            {KeyTrigger{GLFW_KEY_MINUS, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::ZOOM_SPEED_DOWN, "Zoom speed down"},
            {KeyTrigger{GLFW_KEY_KP_ADD, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::ZOOM_SPEED_UP, "Zoom speed up"},
            {KeyTrigger{GLFW_KEY_KP_SUBTRACT, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::ZOOM_SPEED_DOWN, "Zoom speed down"},
            // View
            {KeyTrigger{GLFW_KEY_V, MODIFIER_NONE}, Action::TOGGLE_SPLIT_VIEW, "Split view"},
            {KeyTrigger{GLFW_KEY_G, MODIFIER_NONE}, Action::TOGGLE_GT_COMPARISON, "GT comparison"},
            {KeyTrigger{GLFW_KEY_T, MODIFIER_NONE}, Action::CYCLE_PLY, "Cycle PLY"},
            // Depth
            {KeyTrigger{GLFW_KEY_F, MODIFIER_CTRL}, Action::TOGGLE_DEPTH_MODE, "Depth filter"},
            {MouseScrollTrigger{MODIFIER_ALT}, Action::DEPTH_ADJUST_FAR, "Depth far"},
            {MouseScrollTrigger{MODIFIER_ALT | MODIFIER_CTRL}, Action::DEPTH_ADJUST_SIDE, "Depth side"},
            // Editing (Delete is mode-specific, added below)
            {KeyTrigger{GLFW_KEY_Z, MODIFIER_CTRL}, Action::UNDO, "Undo"},
            {KeyTrigger{GLFW_KEY_Y, MODIFIER_CTRL}, Action::REDO, "Redo"},
            {KeyTrigger{GLFW_KEY_I, MODIFIER_CTRL}, Action::INVERT_SELECTION, "Invert"},
            {KeyTrigger{GLFW_KEY_D, MODIFIER_CTRL}, Action::DESELECT_ALL, "Deselect"},
            {KeyTrigger{GLFW_KEY_A, MODIFIER_CTRL}, Action::SELECT_ALL, "Select all"},
            {KeyTrigger{GLFW_KEY_C, MODIFIER_CTRL}, Action::COPY_SELECTION, "Copy"},
            {KeyTrigger{GLFW_KEY_V, MODIFIER_CTRL}, Action::PASTE_SELECTION, "Paste"},
            // Tools
            {KeyTrigger{GLFW_KEY_B, MODIFIER_NONE}, Action::CYCLE_BRUSH_MODE, "Brush mode"},
            {KeyTrigger{GLFW_KEY_T, MODIFIER_CTRL}, Action::CYCLE_SELECTION_VIS, "Sel vis"},
            {KeyTrigger{GLFW_KEY_ENTER, MODIFIER_NONE}, Action::APPLY_CROP_BOX, "Apply/confirm"},
            {KeyTrigger{GLFW_KEY_ESCAPE, MODIFIER_NONE}, Action::CANCEL_POLYGON, "Cancel"},
            // Selection
            {MouseDragTrigger{MouseButton::LEFT, MODIFIER_NONE}, Action::SELECTION_REPLACE, "Select"},
            {MouseDragTrigger{MouseButton::LEFT, MODIFIER_SHIFT}, Action::SELECTION_ADD, "Add sel"},
            {MouseDragTrigger{MouseButton::LEFT, MODIFIER_CTRL}, Action::SELECTION_REMOVE, "Remove sel"},
            {KeyTrigger{GLFW_KEY_1, MODIFIER_CTRL}, Action::SELECT_MODE_CENTERS, "Centers"},
            {KeyTrigger{GLFW_KEY_2, MODIFIER_CTRL}, Action::SELECT_MODE_RECTANGLE, "Rectangle"},
            {KeyTrigger{GLFW_KEY_3, MODIFIER_CTRL}, Action::SELECT_MODE_POLYGON, "Polygon"},
            {KeyTrigger{GLFW_KEY_4, MODIFIER_CTRL}, Action::SELECT_MODE_LASSO, "Lasso"},
            {KeyTrigger{GLFW_KEY_5, MODIFIER_CTRL}, Action::SELECT_MODE_RINGS, "Rings"},
            // UI
            {KeyTrigger{GLFW_KEY_F12, MODIFIER_NONE}, Action::TOGGLE_UI, "Hide UI"},
            {KeyTrigger{GLFW_KEY_F11, MODIFIER_NONE}, Action::TOGGLE_FULLSCREEN, "Fullscreen"},
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
            profile.bindings.push_back({mode, KeyTrigger{GLFW_KEY_DELETE, MODIFIER_NONE}, Action::DELETE_NODE, "Delete node"});
        }

        constexpr ToolMode DELETE_GAUSSIANS_MODES[] = {
            ToolMode::SELECTION,
            ToolMode::BRUSH,
            ToolMode::ALIGN,
            ToolMode::CROP_BOX,
        };
        for (const auto mode : DELETE_GAUSSIANS_MODES) {
            profile.bindings.push_back({mode, KeyTrigger{GLFW_KEY_DELETE, MODIFIER_NONE}, Action::DELETE_SELECTED, "Delete Gaussians"});
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
        default: return "Unknown";
        }
    }

    std::string getKeyName(const int key) {
        switch (key) {
        case GLFW_KEY_A: return "A";
        case GLFW_KEY_B: return "B";
        case GLFW_KEY_C: return "C";
        case GLFW_KEY_D: return "D";
        case GLFW_KEY_E: return "E";
        case GLFW_KEY_F: return "F";
        case GLFW_KEY_G: return "G";
        case GLFW_KEY_H: return "H";
        case GLFW_KEY_I: return "I";
        case GLFW_KEY_J: return "J";
        case GLFW_KEY_K: return "K";
        case GLFW_KEY_L: return "L";
        case GLFW_KEY_M: return "M";
        case GLFW_KEY_N: return "N";
        case GLFW_KEY_O: return "O";
        case GLFW_KEY_P: return "P";
        case GLFW_KEY_Q: return "Q";
        case GLFW_KEY_R: return "R";
        case GLFW_KEY_S: return "S";
        case GLFW_KEY_T: return "T";
        case GLFW_KEY_U: return "U";
        case GLFW_KEY_V: return "V";
        case GLFW_KEY_W: return "W";
        case GLFW_KEY_X: return "X";
        case GLFW_KEY_Y: return "Y";
        case GLFW_KEY_Z: return "Z";
        case GLFW_KEY_0: return "0";
        case GLFW_KEY_1: return "1";
        case GLFW_KEY_2: return "2";
        case GLFW_KEY_3: return "3";
        case GLFW_KEY_4: return "4";
        case GLFW_KEY_5: return "5";
        case GLFW_KEY_6: return "6";
        case GLFW_KEY_7: return "7";
        case GLFW_KEY_8: return "8";
        case GLFW_KEY_9: return "9";
        case GLFW_KEY_SPACE: return "Space";
        case GLFW_KEY_ENTER: return "Enter";
        case GLFW_KEY_ESCAPE: return "Escape";
        case GLFW_KEY_TAB: return "Tab";
        case GLFW_KEY_BACKSPACE: return "Backspace";
        case GLFW_KEY_DELETE: return "Delete";
        case GLFW_KEY_HOME: return "Home";
        case GLFW_KEY_END: return "End";
        case GLFW_KEY_PAGE_UP: return "Page Up";
        case GLFW_KEY_PAGE_DOWN: return "Page Down";
        case GLFW_KEY_LEFT: return "Left";
        case GLFW_KEY_RIGHT: return "Right";
        case GLFW_KEY_UP: return "Up";
        case GLFW_KEY_DOWN: return "Down";
        case GLFW_KEY_F1: return "F1";
        case GLFW_KEY_F2: return "F2";
        case GLFW_KEY_F3: return "F3";
        case GLFW_KEY_F4: return "F4";
        case GLFW_KEY_F5: return "F5";
        case GLFW_KEY_F6: return "F6";
        case GLFW_KEY_F7: return "F7";
        case GLFW_KEY_F8: return "F8";
        case GLFW_KEY_F9: return "F9";
        case GLFW_KEY_F10: return "F10";
        case GLFW_KEY_F11: return "F11";
        case GLFW_KEY_F12: return "F12";
        case GLFW_KEY_MINUS: return "-";
        case GLFW_KEY_EQUAL: return "=";
        case GLFW_KEY_LEFT_BRACKET: return "[";
        case GLFW_KEY_RIGHT_BRACKET: return "]";
        case GLFW_KEY_BACKSLASH: return "\\";
        case GLFW_KEY_SEMICOLON: return ";";
        case GLFW_KEY_APOSTROPHE: return "'";
        case GLFW_KEY_GRAVE_ACCENT: return "`";
        case GLFW_KEY_COMMA: return ",";
        case GLFW_KEY_PERIOD: return ".";
        case GLFW_KEY_SLASH: return "/";
        case GLFW_KEY_KP_ADD: return "Num+";
        case GLFW_KEY_KP_SUBTRACT: return "Num-";
        case GLFW_KEY_KP_MULTIPLY: return "Num*";
        case GLFW_KEY_KP_DIVIDE: return "Num/";
        case GLFW_KEY_KP_ENTER: return "NumEnter";
        case GLFW_KEY_KP_0: return "Num0";
        case GLFW_KEY_KP_1: return "Num1";
        case GLFW_KEY_KP_2: return "Num2";
        case GLFW_KEY_KP_3: return "Num3";
        case GLFW_KEY_KP_4: return "Num4";
        case GLFW_KEY_KP_5: return "Num5";
        case GLFW_KEY_KP_6: return "Num6";
        case GLFW_KEY_KP_7: return "Num7";
        case GLFW_KEY_KP_8: return "Num8";
        case GLFW_KEY_KP_9: return "Num9";
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

} // namespace lfs::vis::input
