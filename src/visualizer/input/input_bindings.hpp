/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "input/key_codes.hpp"
#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lfs::vis::input {

    enum class ToolMode {
        GLOBAL = 0,
        SELECTION,
        BRUSH,
        TRANSLATE,
        ROTATE,
        SCALE,
        ALIGN,
        CROP_BOX,
    };

    enum class Action {
        NONE = 0,
        // Camera
        CAMERA_ORBIT,
        CAMERA_PAN,
        CAMERA_ZOOM,
        CAMERA_ROLL,
        CAMERA_MOVE_FORWARD,
        CAMERA_MOVE_BACKWARD,
        CAMERA_MOVE_LEFT,
        CAMERA_MOVE_RIGHT,
        CAMERA_MOVE_UP,
        CAMERA_MOVE_DOWN,
        CAMERA_RESET_HOME,
        CAMERA_FOCUS_SELECTION,
        CAMERA_SET_PIVOT,
        CAMERA_NEXT_VIEW,
        CAMERA_PREV_VIEW,
        CAMERA_SPEED_UP,
        CAMERA_SPEED_DOWN,
        ZOOM_SPEED_UP,
        ZOOM_SPEED_DOWN,
        // View
        TOGGLE_SPLIT_VIEW,
        TOGGLE_GT_COMPARISON,
        TOGGLE_DEPTH_MODE,
        CYCLE_PLY,
        // Editing
        DELETE_SELECTED, // Delete selected Gaussians (selection tool)
        DELETE_NODE,     // Delete selected PLY node (global mode)
        UNDO,
        REDO,
        INVERT_SELECTION,
        DESELECT_ALL,
        SELECT_ALL,
        COPY_SELECTION,
        PASTE_SELECTION,
        // Depth filter
        DEPTH_ADJUST_FAR,
        DEPTH_ADJUST_SIDE,
        // Tools
        BRUSH_RESIZE,
        CYCLE_BRUSH_MODE,
        CONFIRM_POLYGON,
        CANCEL_POLYGON,
        UNDO_POLYGON_VERTEX,
        CYCLE_SELECTION_VIS,
        // Selection
        SELECTION_REPLACE,
        SELECTION_ADD,
        SELECTION_REMOVE,
        SELECT_MODE_CENTERS,
        SELECT_MODE_RECTANGLE,
        SELECT_MODE_POLYGON,
        SELECT_MODE_LASSO,
        SELECT_MODE_RINGS,
        // Misc
        APPLY_CROP_BOX,
        // Node picking
        NODE_PICK,
        NODE_RECT_SELECT,
        // UI
        TOGGLE_UI,
        TOGGLE_FULLSCREEN,
        // Sequencer
        SEQUENCER_ADD_KEYFRAME,
        SEQUENCER_UPDATE_KEYFRAME,
        SEQUENCER_PLAY_PAUSE,
        // Tool switching
        TOOL_SELECT,
        TOOL_TRANSLATE,
        TOOL_ROTATE,
        TOOL_SCALE,
        TOOL_MIRROR,
        TOOL_BRUSH,
        TOOL_ALIGN,
        // Pie menu
        PIE_MENU,
    };

    enum Modifier : int {
        MODIFIER_NONE = KEYMOD_NONE,
        MODIFIER_SHIFT = KEYMOD_SHIFT,
        MODIFIER_CTRL = KEYMOD_CTRL,
        MODIFIER_ALT = KEYMOD_ALT,
        MODIFIER_SUPER = KEYMOD_SUPER,
    };

    enum class MouseButton {
        LEFT = static_cast<int>(AppMouseButton::LEFT),
        RIGHT = static_cast<int>(AppMouseButton::RIGHT),
        MIDDLE = static_cast<int>(AppMouseButton::MIDDLE),
    };

    struct KeyTrigger {
        int key;
        int modifiers = MODIFIER_NONE;
        bool on_repeat = false;
    };

    struct MouseButtonTrigger {
        MouseButton button;
        int modifiers = MODIFIER_NONE;
        bool double_click = false;
    };

    struct MouseScrollTrigger {
        int modifiers = MODIFIER_NONE;
    };

    struct MouseDragTrigger {
        MouseButton button;
        int modifiers = MODIFIER_NONE;
    };

    using InputTrigger = std::variant<KeyTrigger, MouseButtonTrigger, MouseScrollTrigger, MouseDragTrigger>;

    struct Binding {
        ToolMode mode = ToolMode::GLOBAL;
        InputTrigger trigger;
        Action action;
        std::string description;
    };

    struct Profile {
        std::string name;
        std::string description;
        std::vector<Binding> bindings;
    };

    struct CaptureState {
        bool active = false;
        ToolMode mode = ToolMode::GLOBAL;
        Action action = Action::NONE;
        std::optional<InputTrigger> captured;
        bool waiting_for_double_click = false;
        int pending_button = -1;
        int pending_mods = 0;
        std::chrono::steady_clock::time_point first_click_time;
        static constexpr double DOUBLE_CLICK_WAIT_TIME = 0.4;
    };

    class LFS_VIS_API InputBindings {
    public:
        InputBindings();

        void loadProfile(const std::string& name);
        void saveProfile(const std::string& name) const;
        bool loadProfileFromFile(const std::filesystem::path& path);
        bool saveProfileToFile(const std::filesystem::path& path) const;
        std::vector<std::string> getAvailableProfiles() const;
        const std::string& getCurrentProfileName() const { return current_profile_name_; }

        static std::filesystem::path getConfigDir();

        // Query bindings (mode-specific only, no fallback)
        Action getActionForKey(ToolMode mode, int key, int modifiers) const;
        Action getActionForMouseButton(ToolMode mode, MouseButton button, int modifiers, bool is_double_click = false) const;
        Action getActionForScroll(ToolMode mode, int modifiers) const;
        Action getActionForDrag(ToolMode mode, MouseButton button, int modifiers) const;

        std::optional<InputTrigger> getTriggerForAction(Action action, ToolMode mode = ToolMode::GLOBAL) const;
        std::string getTriggerDescription(Action action, ToolMode mode = ToolMode::GLOBAL) const;

        // Get the key code for a continuous action (returns -1 if not a key binding)
        int getKeyForAction(Action action, ToolMode mode = ToolMode::GLOBAL) const;

        void setBinding(ToolMode mode, Action action, const InputTrigger& trigger);
        void clearBinding(ToolMode mode, Action action);

        // Callback for binding changes (e.g., to refresh cached keys)
        using BindingsChangedCallback = std::function<void()>;
        void setOnBindingsChanged(BindingsChangedCallback callback) { on_bindings_changed_ = std::move(callback); }

        // Capture mode for rebinding (called from Python)
        void startCapture(ToolMode mode, Action action);
        void cancelCapture();
        void captureKey(int key, int mods);
        void captureMouseButton(int button, int mods);
        void updateCapture();
        bool isCapturing() const { return capture_state_.active; }
        const CaptureState& getCaptureState() const { return capture_state_; }
        std::optional<InputTrigger> getAndClearCaptured();

        // Get all bindings for a mode (for displaying in UI)
        std::vector<std::pair<Action, std::string>> getBindingsForMode(ToolMode mode) const;

        static Profile createDefaultProfile();

    private:
        static constexpr int MODIFIER_MASK = MODIFIER_SHIFT | MODIFIER_CTRL | MODIFIER_ALT | MODIFIER_SUPER;

        std::string current_profile_name_;
        std::vector<Binding> bindings_;

        using KeyMapKey = std::tuple<ToolMode, int, int>;
        using MouseMapKey = std::tuple<ToolMode, MouseButton, int, bool>;
        using ScrollMapKey = std::pair<ToolMode, int>;
        using DragMapKey = std::tuple<ToolMode, MouseButton, int>;

        std::map<KeyMapKey, Action> key_map_;
        std::map<MouseMapKey, Action> mouse_button_map_;
        std::map<ScrollMapKey, Action> scroll_map_;
        std::map<DragMapKey, Action> drag_map_;

        BindingsChangedCallback on_bindings_changed_;
        CaptureState capture_state_;

        void rebuildLookupMaps();
        void notifyBindingsChanged();
    };

    LFS_VIS_API std::string getActionName(Action action);
    LFS_VIS_API std::string getKeyName(int key);
    LFS_VIS_API std::string getMouseButtonName(MouseButton button);
    LFS_VIS_API std::string getModifierString(int modifiers);

} // namespace lfs::vis::input
