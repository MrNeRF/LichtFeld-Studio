/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_bindings.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <thread>

namespace lfs::vis::input {
    namespace {
        bool sameDispatchTrigger(const InputTrigger& lhs, const InputTrigger& rhs) {
            if (lhs.index() != rhs.index())
                return false;
            return std::visit([&](const auto& a) {
                using T = std::decay_t<decltype(a)>;
                const auto& b = std::get<T>(rhs);
                if constexpr (std::is_same_v<T, KeyTrigger>)
                    return a.key == b.key && a.modifiers == b.modifiers;
                else if constexpr (std::is_same_v<T, MouseButtonTrigger>)
                    return a.button == b.button && a.modifiers == b.modifiers &&
                           a.double_click == b.double_click;
                else if constexpr (std::is_same_v<T, MouseScrollTrigger>)
                    return a.modifiers == b.modifiers && a.chord_key == b.chord_key;
                else
                    return a.button == b.button && a.modifiers == b.modifiers &&
                           a.chord_key == b.chord_key;
            },
                              lhs);
        }

        bool activeInMode(const Binding& binding, ToolMode mode) {
            return binding.mode == mode ||
                   (binding.mode == ToolMode::GLOBAL &&
                    describe(binding.action).inherits_from_global);
        }

        struct RestorePersistence {
            bool enabled = InputBindings::isPersistenceEnabled();
            ~RestorePersistence() { InputBindings::setPersistenceEnabled(enabled); }
        };

        TEST(KeymapDefaultContractTest, NoDefaultTriggersCollideWithinAnEffectiveMode) {
            const auto profile = InputBindings::createDefaultProfile();
            for (const auto mode : kAllToolModes) {
                for (size_t i = 0; i < profile.bindings.size(); ++i) {
                    const auto& a = profile.bindings[i];
                    if (!activeInMode(a, mode))
                        continue;
                    for (size_t j = i + 1; j < profile.bindings.size(); ++j) {
                        const auto& b = profile.bindings[j];
                        if (!activeInMode(b, mode))
                            continue;
                        EXPECT_FALSE(sameDispatchTrigger(a.trigger, b.trigger))
                            << "mode " << static_cast<int>(mode) << ": "
                            << getActionName(a.action) << " overlaps " << getActionName(b.action);
                    }
                }
            }
        }

        TEST(KeymapDefaultContractTest, CoreActionsDispatchInEveryRequiredMode) {
            RestorePersistence restore;
            InputBindings::setPersistenceEnabled(false);
            const InputBindings bindings;
            for (const auto mode : kAllToolModes) {
                SCOPED_TRACE(static_cast<int>(mode));
                EXPECT_EQ(bindings.getActionForDrag(mode, MouseButton::MIDDLE, MODIFIER_NONE),
                          Action::CAMERA_ORBIT);
                EXPECT_EQ(bindings.getActionForDrag(mode, MouseButton::RIGHT, MODIFIER_NONE),
                          Action::CAMERA_PAN);
                EXPECT_EQ(bindings.getActionForScroll(mode, MODIFIER_NONE), Action::CAMERA_ZOOM);
                EXPECT_EQ(bindings.getActionForKey(mode, KEY_Z, MODIFIER_CTRL), Action::UNDO);
                EXPECT_EQ(bindings.getActionForKey(mode, KEY_Y, MODIFIER_CTRL), Action::REDO);
                EXPECT_EQ(bindings.getActionForKey(mode, KEY_ESCAPE, MODIFIER_NONE),
                          Action::CANCEL_POLYGON);
                constexpr std::array tools = {Action::TOOL_SELECT, Action::TOOL_TRANSLATE,
                                              Action::TOOL_ROTATE, Action::TOOL_SCALE,
                                              Action::TOOL_MIRROR, Action::TOOL_ALIGN};
                for (int i = 0; i < static_cast<int>(tools.size()); ++i) {
                    EXPECT_EQ(bindings.getActionForKey(mode, KEY_1 + i, MODIFIER_NONE),
                              tools[i]);
                }
                const auto delete_action =
                    mode == ToolMode::SELECTION || mode == ToolMode::ALIGN ||
                            mode == ToolMode::CROP_BOX
                        ? Action::DELETE_SELECTED
                        : Action::DELETE_NODE;
                EXPECT_EQ(bindings.getActionForKey(mode, KEY_DELETE, MODIFIER_NONE),
                          delete_action);
            }
            EXPECT_EQ(bindings.getActionForDrag(ToolMode::SELECTION, MouseButton::LEFT,
                                                MODIFIER_NONE),
                      Action::SELECTION_REPLACE);
            EXPECT_EQ(bindings.getActionForDrag(ToolMode::SELECTION, MouseButton::LEFT,
                                                MODIFIER_SHIFT),
                      Action::SELECTION_ADD);
            EXPECT_EQ(bindings.getActionForDrag(ToolMode::SELECTION, MouseButton::LEFT,
                                                MODIFIER_CTRL),
                      Action::SELECTION_REMOVE);
            for (const auto mode : {ToolMode::TRANSLATE, ToolMode::ROTATE, ToolMode::SCALE}) {
                EXPECT_EQ(bindings.getActionForMouseButton(mode, MouseButton::LEFT, 0, false),
                          Action::NODE_PICK);
                EXPECT_EQ(bindings.getActionForDrag(mode, MouseButton::LEFT, 0),
                          Action::NODE_RECT_SELECT);
            }
        }

        TEST(KeymapDefaultContractTest, EveryNamedActionHasALabelInEveryLocale) {
            const auto root = std::filesystem::path(PROJECT_ROOT_PATH) /
                              "src/visualizer/gui/resources/locales";
            size_t locale_count = 0;
            for (const auto& entry : std::filesystem::directory_iterator(root)) {
                if (entry.path().extension() != ".json")
                    continue;
                ++locale_count;
                const auto locale = nlohmann::json::parse(std::ifstream(entry.path()));
                const auto& labels = locale.at("input_settings");
                for (int i = 0; i <= static_cast<int>(Action::ASSET_REFRESH); ++i) {
                    const auto action = static_cast<Action>(i);
                    const auto name = actionNameKey(action);
                    if (name.empty())
                        continue;
                    const std::string key = "action." + std::string(name);
                    ASSERT_TRUE(labels.contains(key)) << entry.path() << ": " << key;
                    EXPECT_FALSE(labels.at(key).get<std::string>().empty())
                        << entry.path() << ": " << key;
                }
            }
            EXPECT_EQ(locale_count, 10u);
        }

        TEST(KeymapDefaultContractTest, EditorCaptureRoundTripsEveryDefaultTrigger) {
            const auto defaults = InputBindings::createDefaultProfile();
            RestorePersistence restore;
            InputBindings::setPersistenceEnabled(false);
            for (const auto& binding : defaults.bindings) {
                SCOPED_TRACE(getActionName(binding.action) + " mode " +
                             std::to_string(static_cast<int>(binding.mode)));
                InputBindings editor;
                editor.startCapture(binding.mode, binding.action);
                std::visit([&](const auto& trigger) {
                    using T = std::decay_t<decltype(trigger)>;
                    if constexpr (std::is_same_v<T, KeyTrigger>) {
                        editor.captureKey(trigger.key, trigger.key, trigger.modifiers);
                    } else if constexpr (std::is_same_v<T, MouseButtonTrigger>) {
                        const int button = static_cast<int>(trigger.button);
                        editor.captureMouseButton(button, trigger.modifiers, 0.0, 0.0);
                        if (trigger.double_click) {
                            editor.captureMouseButtonRelease(button);
                            editor.captureMouseButton(button, trigger.modifiers, 0.0, 0.0);
                        } else {
                            editor.captureMouseButtonRelease(button);
                            std::this_thread::sleep_for(std::chrono::milliseconds(410));
                            editor.updateCapture();
                        }
                    } else if constexpr (std::is_same_v<T, MouseScrollTrigger>) {
                        editor.captureScroll(trigger.modifiers, trigger.chord_key);
                    } else {
                        editor.captureMouseButton(static_cast<int>(trigger.button),
                                                  trigger.modifiers, 0.0, 0.0,
                                                  trigger.chord_key);
                        editor.captureMouseMove(10.0, 0.0);
                    }
                },
                           binding.trigger);
                const auto captured = editor.getAndClearCaptured();
                ASSERT_TRUE(captured.has_value());
                ASSERT_EQ(captured->index(), binding.trigger.index());
                EXPECT_TRUE(sameDispatchTrigger(*captured, binding.trigger));
                if (const auto* key = std::get_if<KeyTrigger>(&binding.trigger))
                    EXPECT_EQ(std::get<KeyTrigger>(*captured).on_repeat, key->on_repeat);
                const auto stored = editor.getTriggerForAction(binding.action, binding.mode);
                ASSERT_TRUE(stored);
                EXPECT_TRUE(sameDispatchTrigger(*stored, *captured));
                EXPECT_FALSE(editor.getTriggerDescription(binding.action, binding.mode).empty());
                EXPECT_NE(editor.getTriggerDescription(binding.action, binding.mode), "Unbound");
                const auto dispatched = std::visit([&](const auto& trigger) -> Action {
                    using T = std::decay_t<decltype(trigger)>;
                    if constexpr (std::is_same_v<T, KeyTrigger>)
                        return editor.getActionForKey(binding.mode, trigger.key, trigger.modifiers);
                    else if constexpr (std::is_same_v<T, MouseButtonTrigger>)
                        return editor.getActionForMouseButton(binding.mode, trigger.button,
                                                              trigger.modifiers, trigger.double_click);
                    else if constexpr (std::is_same_v<T, MouseScrollTrigger>)
                        return editor.getActionForScroll(binding.mode, trigger.modifiers,
                                                         trigger.chord_key
                                                             ? std::vector<int>{*trigger.chord_key}
                                                             : std::vector<int>{});
                    else
                        return editor.getActionForDrag(binding.mode, trigger.button,
                                                       trigger.modifiers,
                                                       trigger.chord_key
                                                           ? std::vector<int>{*trigger.chord_key}
                                                           : std::vector<int>{});
                },
                                                   *captured);
                EXPECT_EQ(dispatched, binding.action);
            }
        }
    } // namespace
} // namespace lfs::vis::input
