/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "python/python_runtime.hpp"
#include "visualizer/input/input_bindings.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace lfs::python {

    using namespace lfs::vis::input;

    namespace {
        nb::dict triggerToDict(const InputTrigger& trigger) {
            nb::dict result;
            std::visit(
                [&result](const auto& t) {
                    using T = std::decay_t<decltype(t)>;
                    if constexpr (std::is_same_v<T, KeyTrigger>) {
                        result["type"] = "key";
                        result["key"] = t.key;
                        result["modifiers"] = t.modifiers;
                    } else if constexpr (std::is_same_v<T, MouseButtonTrigger>) {
                        result["type"] = "mouse_button";
                        result["button"] = static_cast<int>(t.button);
                        result["modifiers"] = t.modifiers;
                        result["double_click"] = t.double_click;
                    } else if constexpr (std::is_same_v<T, MouseScrollTrigger>) {
                        result["type"] = "scroll";
                        result["modifiers"] = t.modifiers;
                        if (t.chord_key.has_value()) {
                            result["chord_key"] = *t.chord_key;
                        }
                    } else if constexpr (std::is_same_v<T, MouseDragTrigger>) {
                        result["type"] = "drag";
                        result["button"] = static_cast<int>(t.button);
                        result["modifiers"] = t.modifiers;
                        if (t.chord_key.has_value()) {
                            result["chord_key"] = *t.chord_key;
                        }
                    }
                },
                trigger);
            return result;
        }

        std::optional<InputTrigger> triggerFromDict(const nb::dict& d) {
            if (!d.contains("type")) {
                return std::nullopt;
            }

            const auto type = nb::cast<std::string>(d["type"]);
            const int modifiers = d.contains("modifiers") ? nb::cast<int>(d["modifiers"]) : MODIFIER_NONE;
            const auto chord_key = [&]() -> std::optional<int> {
                if (!d.contains("chord_key")) {
                    return std::nullopt;
                }
                return nb::cast<int>(d["chord_key"]);
            }();

            if (type == "key") {
                if (!d.contains("key")) {
                    return std::nullopt;
                }
                return KeyTrigger{nb::cast<int>(d["key"]), modifiers, false};
            }
            if (type == "mouse_button") {
                if (!d.contains("button")) {
                    return std::nullopt;
                }
                const bool double_click = d.contains("double_click") && nb::cast<bool>(d["double_click"]);
                return MouseButtonTrigger{static_cast<MouseButton>(nb::cast<int>(d["button"])),
                                          modifiers,
                                          double_click};
            }
            if (type == "scroll") {
                return MouseScrollTrigger{modifiers, chord_key};
            }
            if (type == "drag") {
                if (!d.contains("button")) {
                    return std::nullopt;
                }
                return MouseDragTrigger{static_cast<MouseButton>(nb::cast<int>(d["button"])),
                                        modifiers,
                                        chord_key};
            }
            return std::nullopt;
        }
    } // namespace

    void register_keymap(nb::module_& m) {
        auto keymap = m.def_submodule("keymap", "Keymap configuration");

        // Expose Action enum
        nb::enum_<Action>(keymap, "Action")
            .value("NONE", Action::NONE)
            .value("CAMERA_ORBIT", Action::CAMERA_ORBIT)
            .value("CAMERA_PAN", Action::CAMERA_PAN)
            .value("CAMERA_ZOOM", Action::CAMERA_ZOOM)
            .value("CAMERA_ROLL", Action::CAMERA_ROLL)
            .value("CAMERA_MOVE_FORWARD", Action::CAMERA_MOVE_FORWARD)
            .value("CAMERA_MOVE_BACKWARD", Action::CAMERA_MOVE_BACKWARD)
            .value("CAMERA_MOVE_LEFT", Action::CAMERA_MOVE_LEFT)
            .value("CAMERA_MOVE_RIGHT", Action::CAMERA_MOVE_RIGHT)
            .value("CAMERA_MOVE_UP", Action::CAMERA_MOVE_UP)
            .value("CAMERA_MOVE_DOWN", Action::CAMERA_MOVE_DOWN)
            .value("CAMERA_RESET_HOME", Action::CAMERA_RESET_HOME)
            .value("CAMERA_FOCUS_SELECTION", Action::CAMERA_FOCUS_SELECTION)
            .value("CAMERA_SET_PIVOT", Action::CAMERA_SET_PIVOT)
            .value("CAMERA_NEXT_VIEW", Action::CAMERA_NEXT_VIEW)
            .value("CAMERA_PREV_VIEW", Action::CAMERA_PREV_VIEW)
            .value("CAMERA_SPEED_UP", Action::CAMERA_SPEED_UP)
            .value("CAMERA_SPEED_DOWN", Action::CAMERA_SPEED_DOWN)
            .value("ZOOM_SPEED_UP", Action::ZOOM_SPEED_UP)
            .value("ZOOM_SPEED_DOWN", Action::ZOOM_SPEED_DOWN)
            .value("TOGGLE_SPLIT_VIEW", Action::TOGGLE_SPLIT_VIEW)
            .value("TOGGLE_INDEPENDENT_SPLIT_VIEW", Action::TOGGLE_INDEPENDENT_SPLIT_VIEW)
            .value("TOGGLE_GT_COMPARISON", Action::TOGGLE_GT_COMPARISON)
            .value("TOGGLE_DEPTH_MODE", Action::TOGGLE_DEPTH_MODE)
            .value("CYCLE_PLY", Action::CYCLE_PLY)
            .value("DELETE_SELECTED", Action::DELETE_SELECTED)
            .value("DELETE_NODE", Action::DELETE_NODE)
            .value("UNDO", Action::UNDO)
            .value("REDO", Action::REDO)
            .value("INVERT_SELECTION", Action::INVERT_SELECTION)
            .value("DESELECT_ALL", Action::DESELECT_ALL)
            .value("SELECT_ALL", Action::SELECT_ALL)
            .value("COPY_SELECTION", Action::COPY_SELECTION)
            .value("PASTE_SELECTION", Action::PASTE_SELECTION)
            .value("DEPTH_ADJUST_FAR", Action::DEPTH_ADJUST_FAR)
            .value("DEPTH_ADJUST_SIDE", Action::DEPTH_ADJUST_SIDE)
            .value("TOGGLE_SELECTION_DEPTH_FILTER", Action::TOGGLE_SELECTION_DEPTH_FILTER)
            .value("TOGGLE_SELECTION_CROP_FILTER", Action::TOGGLE_SELECTION_CROP_FILTER)
            .value("BRUSH_RESIZE", Action::BRUSH_RESIZE)
            .value("CYCLE_BRUSH_MODE", Action::CYCLE_BRUSH_MODE)
            .value("CONFIRM_POLYGON", Action::CONFIRM_POLYGON)
            .value("CANCEL_POLYGON", Action::CANCEL_POLYGON)
            .value("UNDO_POLYGON_VERTEX", Action::UNDO_POLYGON_VERTEX)
            .value("CYCLE_SELECTION_VIS", Action::CYCLE_SELECTION_VIS)
            .value("SELECTION_REPLACE", Action::SELECTION_REPLACE)
            .value("SELECTION_ADD", Action::SELECTION_ADD)
            .value("SELECTION_REMOVE", Action::SELECTION_REMOVE)
            .value("SELECT_MODE_CENTERS", Action::SELECT_MODE_CENTERS)
            .value("SELECT_MODE_RECTANGLE", Action::SELECT_MODE_RECTANGLE)
            .value("SELECT_MODE_POLYGON", Action::SELECT_MODE_POLYGON)
            .value("SELECT_MODE_LASSO", Action::SELECT_MODE_LASSO)
            .value("SELECT_MODE_RINGS", Action::SELECT_MODE_RINGS)
            .value("APPLY_CROP_BOX", Action::APPLY_CROP_BOX)
            .value("NODE_PICK", Action::NODE_PICK)
            .value("NODE_RECT_SELECT", Action::NODE_RECT_SELECT)
            .value("TOGGLE_UI", Action::TOGGLE_UI)
            .value("TOGGLE_FULLSCREEN", Action::TOGGLE_FULLSCREEN)
            .value("SEQUENCER_ADD_KEYFRAME", Action::SEQUENCER_ADD_KEYFRAME)
            .value("SEQUENCER_UPDATE_KEYFRAME", Action::SEQUENCER_UPDATE_KEYFRAME)
            .value("SEQUENCER_PLAY_PAUSE", Action::SEQUENCER_PLAY_PAUSE)
            .value("TOOL_SELECT", Action::TOOL_SELECT)
            .value("TOOL_TRANSLATE", Action::TOOL_TRANSLATE)
            .value("TOOL_ROTATE", Action::TOOL_ROTATE)
            .value("TOOL_SCALE", Action::TOOL_SCALE)
            .value("TOOL_MIRROR", Action::TOOL_MIRROR)
            .value("TOOL_BRUSH", Action::TOOL_BRUSH)
            .value("TOOL_ALIGN", Action::TOOL_ALIGN)
            .value("PIE_MENU", Action::PIE_MENU)
            .value("DEPTH_ADJUST_NEAR", Action::DEPTH_ADJUST_NEAR);

        // Expose ToolMode enum
        nb::enum_<ToolMode>(keymap, "ToolMode")
            .value("GLOBAL", ToolMode::GLOBAL)
            .value("SELECTION", ToolMode::SELECTION)
            .value("BRUSH", ToolMode::BRUSH)
            .value("TRANSLATE", ToolMode::TRANSLATE)
            .value("ROTATE", ToolMode::ROTATE)
            .value("SCALE", ToolMode::SCALE)
            .value("ALIGN", ToolMode::ALIGN)
            .value("CROP_BOX", ToolMode::CROP_BOX);

        // Expose Modifier enum
        nb::enum_<Modifier>(keymap, "Modifier")
            .value("NONE", MODIFIER_NONE)
            .value("SHIFT", MODIFIER_SHIFT)
            .value("CTRL", MODIFIER_CTRL)
            .value("ALT", MODIFIER_ALT)
            .value("SUPER", MODIFIER_SUPER);

        // Expose MouseButton enum
        nb::enum_<MouseButton>(keymap, "MouseButton")
            .value("LEFT", MouseButton::LEFT)
            .value("RIGHT", MouseButton::RIGHT)
            .value("MIDDLE", MouseButton::MIDDLE);

        // KeyTrigger class
        nb::class_<KeyTrigger>(keymap, "KeyTrigger")
            .def(nb::init<int, int, bool>(),
                 nb::arg("key"), nb::arg("modifiers") = MODIFIER_NONE, nb::arg("on_repeat") = false)
            .def_rw("key", &KeyTrigger::key, "Key code")
            .def_rw("modifiers", &KeyTrigger::modifiers, "Modifier key bitmask")
            .def_rw("on_repeat", &KeyTrigger::on_repeat, "Whether to trigger on key repeat");

        // MouseButtonTrigger class
        nb::class_<MouseButtonTrigger>(keymap, "MouseButtonTrigger")
            .def(nb::init<MouseButton, int, bool>(),
                 nb::arg("button"), nb::arg("modifiers") = MODIFIER_NONE, nb::arg("double_click") = false)
            .def_rw("button", &MouseButtonTrigger::button, "Mouse button")
            .def_rw("modifiers", &MouseButtonTrigger::modifiers, "Modifier key bitmask")
            .def_rw("double_click", &MouseButtonTrigger::double_click, "Whether to require double-click");

        // Binding operations
        keymap.def(
            "get_action_for_key",
            [](ToolMode mode, int key, int modifiers) {
                if (!get_keymap_bindings())
                    return Action::NONE;
                return get_keymap_bindings()->getActionForKey(mode, key, modifiers);
            },
            nb::arg("mode"), nb::arg("key"), nb::arg("modifiers") = 0,
            "Get action bound to a key in given mode");

        keymap.def(
            "get_key_for_action",
            [](Action action, ToolMode mode) {
                if (!get_keymap_bindings())
                    return -1;
                return get_keymap_bindings()->getKeyForAction(action, mode);
            },
            nb::arg("action"), nb::arg("mode") = ToolMode::GLOBAL,
            "Get key code bound to an action");

        keymap.def(
            "get_trigger_description",
            [](Action action, ToolMode mode) {
                if (!get_keymap_bindings())
                    return std::string();
                return get_keymap_bindings()->getTriggerDescription(action, mode);
            },
            nb::arg("action"), nb::arg("mode") = ToolMode::GLOBAL,
            "Get human-readable description of action's trigger");

        keymap.def(
            "get_trigger",
            [](Action action, ToolMode mode) -> nb::object {
                if (!get_keymap_bindings())
                    return nb::none();
                const auto trigger = get_keymap_bindings()->getTriggerForAction(action, mode);
                if (!trigger)
                    return nb::none();
                return triggerToDict(*trigger);
            },
            nb::arg("action"), nb::arg("mode") = ToolMode::GLOBAL,
            "Get action's trigger as a serializable dict");

        keymap.def(
            "set_binding",
            [](ToolMode mode, Action action, int key, int modifiers) {
                if (!get_keymap_bindings())
                    return;
                KeyTrigger trigger{key, modifiers, false};
                get_keymap_bindings()->setBinding(mode, action, trigger);
            },
            nb::arg("mode"), nb::arg("action"), nb::arg("key"), nb::arg("modifiers") = 0,
            "Bind a key to an action in given mode");

        keymap.def(
            "set_trigger_binding",
            [](ToolMode mode, Action action, nb::dict trigger_dict) {
                if (!get_keymap_bindings())
                    return false;
                const auto trigger = triggerFromDict(trigger_dict);
                if (!trigger)
                    return false;
                get_keymap_bindings()->setBinding(mode, action, *trigger);
                return true;
            },
            nb::arg("mode"), nb::arg("action"), nb::arg("trigger"),
            "Bind a key, mouse button, scroll, or drag trigger dict to an action");

        keymap.def(
            "clear_binding",
            [](ToolMode mode, Action action) {
                if (!get_keymap_bindings())
                    return;
                get_keymap_bindings()->clearBinding(mode, action);
            },
            nb::arg("mode"), nb::arg("action"),
            "Remove binding for an action in given mode");

        keymap.def(
            "find_conflict_for_action",
            [](ToolMode mode, Action action) -> nb::object {
                auto* bindings = get_keymap_bindings();
                if (!bindings)
                    return nb::none();
                const auto trigger = bindings->getTriggerForAction(action, mode);
                if (!trigger)
                    return nb::none();
                const auto conflict = bindings->findConflict(mode, *trigger, action);
                if (!conflict)
                    return nb::none();
                nb::dict d;
                d["other_action"] = conflict->other_action;
                d["other_mode"] = conflict->other_mode;
                return d;
            },
            nb::arg("mode"), nb::arg("action"),
            "Return {other_action, other_mode} if another action shares this action's trigger, else None");

        keymap.def(
            "get_action_name",
            [](Action action) { return getActionName(action); },
            nb::arg("action"),
            "Get display name for an action");

        keymap.def(
            "get_key_name",
            [](int key) { return getKeyName(key); },
            nb::arg("key"),
            "Get display name for a key code");

        keymap.def(
            "get_modifier_string",
            [](int modifiers) { return getModifierString(modifiers); },
            nb::arg("modifiers"),
            "Get display string for modifier bitmask");

        keymap.def(
            "get_allowed_trigger_kinds",
            [](Action action) {
                nb::list result;
                const auto allowed = describe(action).allowed_kinds;
                if (allowed & TRIGGER_KIND_KEY)
                    result.append("key");
                if (allowed & TRIGGER_KIND_MOUSE_BUTTON)
                    result.append("mouse_button");
                if (allowed & TRIGGER_KIND_MOUSE_SCROLL)
                    result.append("mouse_scroll");
                if (allowed & TRIGGER_KIND_MOUSE_DRAG)
                    result.append("mouse_drag");
                return result;
            },
            nb::arg("action"),
            "Get allowed trigger kinds for an action");

        keymap.def(
            "get_available_profiles",
            []() {
                if (!get_keymap_bindings())
                    return std::vector<std::string>();
                return get_keymap_bindings()->getAvailableProfiles();
            },
            "Get list of available keymap profile names");

        keymap.def(
            "get_current_profile",
            []() {
                if (!get_keymap_bindings())
                    return std::string();
                return get_keymap_bindings()->getCurrentProfileName();
            },
            "Get name of active keymap profile");

        keymap.def(
            "load_profile",
            [](const std::string& name) {
                if (!get_keymap_bindings())
                    return;
                get_keymap_bindings()->loadProfile(name);
            },
            nb::arg("name"),
            "Load a keymap profile by name");

        keymap.def(
            "save_profile",
            [](const std::string& name) {
                if (!get_keymap_bindings())
                    return;
                get_keymap_bindings()->saveProfile(name);
            },
            nb::arg("name"),
            "Save current bindings as a named profile");

        keymap.def(
            "export_profile",
            [](const std::string& path) {
                const auto path_fs = lfs::core::utf8_to_path(path);
                if (!get_keymap_bindings())
                    return false;
                return get_keymap_bindings()->saveProfileToFile(path_fs);
            },
            nb::arg("path"),
            "Export current profile to file");

        keymap.def(
            "import_profile",
            [](const std::string& path) {
                const auto path_fs = lfs::core::utf8_to_path(path);
                if (!get_keymap_bindings())
                    return false;
                return get_keymap_bindings()->loadProfileFromFile(path_fs);
            },
            nb::arg("path"),
            "Import profile from file");

        keymap.def(
            "start_capture",
            [](ToolMode mode, Action action) {
                if (!get_keymap_bindings())
                    return;
                get_keymap_bindings()->startCapture(mode, action);
            },
            nb::arg("mode"), nb::arg("action"),
            "Start capturing input for rebinding");

        keymap.def(
            "cancel_capture",
            []() {
                if (get_keymap_bindings())
                    get_keymap_bindings()->cancelCapture();
            },
            "Cancel active capture");

        keymap.def(
            "capture_scroll",
            [](int modifiers, std::optional<int> chord_key) {
                if (get_keymap_bindings())
                    get_keymap_bindings()->captureScroll(modifiers, chord_key);
            },
            nb::arg("modifiers") = 0, nb::arg("chord_key") = nb::none(),
            "Forward a scroll-wheel event into the active capture");

        keymap.def(
            "is_capturing",
            []() {
                if (!get_keymap_bindings())
                    return false;
                return get_keymap_bindings()->isCapturing();
            },
            "Check if capture mode is active");

        keymap.def(
            "is_waiting_for_double_click",
            []() {
                if (!get_keymap_bindings())
                    return false;
                return get_keymap_bindings()->getCaptureState().waiting_for_double_click;
            },
            "Check if waiting for potential double-click");

        keymap.def(
            "get_captured_trigger",
            []() -> nb::object {
                if (!get_keymap_bindings())
                    return nb::none();
                auto trigger = get_keymap_bindings()->getAndClearCaptured();
                if (!trigger)
                    return nb::none();

                return triggerToDict(*trigger);
            },
            "Get captured trigger (clears it), returns None if nothing captured");

        keymap.def(
            "get_bindings_for_mode",
            [](ToolMode mode) {
                nb::list result;
                if (!get_keymap_bindings())
                    return result;

                auto bindings = get_keymap_bindings()->getBindingsForMode(mode);
                for (const auto& [action, desc] : bindings) {
                    nb::dict d;
                    d["action"] = action;
                    d["action_name"] = getActionName(action);
                    d["description"] = desc;
                    result.append(d);
                }
                return result;
            },
            nb::arg("mode"),
            "Get all bindings for a tool mode");

        keymap.def(
            "reset_to_default",
            []() {
                if (!get_keymap_bindings())
                    return;
                auto config_dir = InputBindings::getConfigDir();
                auto saved_path = config_dir / "Default.json";
                if (std::filesystem::exists(saved_path)) {
                    std::filesystem::remove(saved_path);
                }
                get_keymap_bindings()->loadProfile("Default");
                get_keymap_bindings()->saveProfile("Default");
            },
            "Reset to default bindings");

        keymap.def(
            "get_tool_mode_name",
            [](ToolMode mode) {
                switch (mode) {
                case ToolMode::GLOBAL: return "Global";
                case ToolMode::SELECTION: return "Selection";
                case ToolMode::BRUSH: return "Brush";
                case ToolMode::TRANSLATE: return "Translate";
                case ToolMode::ROTATE: return "Rotate";
                case ToolMode::SCALE: return "Scale";
                case ToolMode::ALIGN: return "Align";
                case ToolMode::CROP_BOX: return "Crop Box";
                default: return "Unknown";
                }
            },
            nb::arg("mode"),
            "Get human-readable name for tool mode");
    }

} // namespace lfs::python
