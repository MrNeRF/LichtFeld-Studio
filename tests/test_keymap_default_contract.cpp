#include "core/error_bus.hpp"
#include "input/input_bindings.hpp"
#include "keymap_history_helpers.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <thread>

namespace lfs::vis::input {
    namespace {
        using KeymapTest = test::KeymapTest;
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

        bool activeInMode(const Binding& binding, ToolMode mode,
                          const Profile& profile) {
            if (binding.mode == mode)
                return true;
            if (binding.mode != ToolMode::GLOBAL ||
                !describe(binding.action).inherits_from_global)
                return false;
            // A mode binding with the same gesture replaces the inherited
            // action; the global gesture is no longer effective there.
            return !std::ranges::any_of(profile.bindings, [&](const Binding& local) {
                return local.mode == mode &&
                       sameDispatchTrigger(local.trigger, binding.trigger);
            });
        }
        TEST_F(KeymapTest, NoDefaultTriggersCollideWithinAnEffectiveMode) {
            const auto profile = InputBindings::createDefaultProfile();
            for (const auto mode : kAllToolModes) {
                for (size_t i = 0; i < profile.bindings.size(); ++i) {
                    const auto& a = profile.bindings[i];
                    if (!activeInMode(a, mode, profile))
                        continue;
                    for (size_t j = i + 1; j < profile.bindings.size(); ++j) {
                        const auto& b = profile.bindings[j];
                        if (!activeInMode(b, mode, profile))
                            continue;
                        EXPECT_FALSE(sameDispatchTrigger(a.trigger, b.trigger))
                            << "mode " << static_cast<int>(mode) << ": "
                            << getActionName(a.action) << " overlaps " << getActionName(b.action);
                    }
                }
            }
        }

        TEST_F(KeymapTest, CoreActionsDispatchInEveryRequiredMode) {
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
                EXPECT_EQ(bindings.getActionForKey(mode, KEY_TAB, MODIFIER_CTRL),
                          Action::NEXT_SIDEBAR_CATEGORY);
                EXPECT_EQ(bindings.getActionForKey(mode, KEY_TAB, MODIFIER_CTRL | MODIFIER_SHIFT),
                          Action::PREVIOUS_SIDEBAR_CATEGORY);
                EXPECT_EQ(bindings.getActionForKey(mode, KEY_ESCAPE, MODIFIER_NONE),
                          mode == ToolMode::WALK ? Action::EXIT_WALK : Action::CANCEL_POLYGON);
                if (mode == ToolMode::SELECTION)
                    EXPECT_EQ(bindings.getActionForKey(mode, KEY_ENTER, MODIFIER_CTRL),
                              Action::ASSET_GALLERY_PRIMARY);
                constexpr std::array tools = {Action::TOOL_SELECT, Action::TOOL_TRANSLATE,
                                              Action::TOOL_ROTATE, Action::TOOL_SCALE,
                                              Action::TOOL_MIRROR, Action::TOOL_ALIGN,
                                              Action::TOOL_SCALE_DISTANCE};
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
                          mode == ToolMode::WALK ? Action::NONE : delete_action);
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

        TEST_F(KeymapTest, EveryNamedActionHasALabelInEveryLocale) {
            const auto root = std::filesystem::path(PROJECT_ROOT_PATH) /
                              "src/visualizer/gui/resources/locales";
            size_t locale_count = 0;
            for (const auto& entry : std::filesystem::directory_iterator(root)) {
                if (entry.path().extension() != ".json")
                    continue;
                ++locale_count;
                const auto locale = nlohmann::json::parse(std::ifstream(entry.path()));
                const auto& labels = locale.at("input_settings");
                for (int i = 0; i <= static_cast<int>(Action::PREVIOUS_SIDEBAR_CATEGORY); ++i) {
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

        TEST_F(KeymapTest, EditorCaptureRoundTripsEveryDefaultTrigger) {
            const auto defaults = InputBindings::createDefaultProfile();
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

        class CoreShortcutWarning final : public lfs::NativeErrorConsumer {
        public:
            void on_error(const lfs::ErrorNotification& notification,
                          const lfs::ErrorDeliveryInfo&) noexcept override {
                if (notification.error.code() != lfs::ErrorCode::FailedPrecondition ||
                    notification.surface != lfs::ErrorSurface::Modal)
                    return;
                try {
                    ++count;
                    message = std::string(notification.error.user_message());
                    for (const auto& candidate : notification.actions) {
                        if (candidate.kind == lfs::ErrorActionKind::Custom)
                            restore = candidate;
                    }
                } catch (...) {
                }
            }
            int count = 0;
            std::string message;
            std::optional<lfs::ErrorAction> restore;
        };

        TEST_F(KeymapTest, WarnsThenRestoresOnlyUnreachableCoreActions) {
            const std::filesystem::path path =
                std::filesystem::temp_directory_path() / "lfs_broken_keymap_core.json";
            InputBindings bindings;
            ASSERT_TRUE(bindings.saveProfileToFile(path));
            auto file = nlohmann::json::parse(std::ifstream(path));
            for (const auto& [mode, action] : {
                     std::pair{ToolMode::GLOBAL, Action::CAMERA_ORBIT},
                     std::pair{ToolMode::SELECTION, Action::SELECTION_REPLACE},
                     std::pair{ToolMode::SELECTION, Action::DELETE_SELECTED},
                 }) {
                file["overrides"]["removed"].push_back(
                    {{"mode", static_cast<int>(mode)},
                     {"action", static_cast<int>(action)},
                     {"description", getActionName(action)},
                     {"slot", 0}});
            }
            file["overrides"]["bindings"].push_back(
                {{"mode", static_cast<int>(ToolMode::SELECTION)},
                 {"action", static_cast<int>(Action::TOGGLE_UI)},
                 {"description", getActionName(Action::TOGGLE_UI)},
                 {"trigger_type", "key"},
                 {"key", KEY_F9},
                 {"modifiers", 0},
                 {"on_repeat", false}});
            {
                std::ofstream output(path);
                output << file.dump(2);
            }

            CoreShortcutWarning warning;
            auto subscription = lfs::ErrorBus::instance().subscribe(warning);
            ASSERT_TRUE(bindings.loadProfileFromFile(path));
            EXPECT_EQ(warning.count, 1);
            ASSERT_FALSE(warning.message.empty());
            ASSERT_TRUE(warning.restore.has_value());
            EXPECT_FALSE(warning.restore->label.empty());
            EXPECT_EQ(bindings.getActionForDrag(ToolMode::SELECTION, MouseButton::MIDDLE, 0),
                      Action::NONE);
            EXPECT_EQ(bindings.getActionForDrag(ToolMode::SELECTION, MouseButton::LEFT, 0),
                      Action::NONE);
            EXPECT_EQ(bindings.getActionForKey(ToolMode::SELECTION, KEY_DELETE, 0),
                      Action::NONE);
            EXPECT_EQ(bindings.getActionForKey(ToolMode::SELECTION, KEY_F9, 0),
                      Action::TOGGLE_UI);

            warning.restore->on_invoke(lfs::OperationId::generate());
            EXPECT_EQ(bindings.getActionForDrag(ToolMode::SELECTION, MouseButton::MIDDLE, 0),
                      Action::CAMERA_ORBIT);
            EXPECT_EQ(bindings.getActionForDrag(ToolMode::SELECTION, MouseButton::LEFT, 0),
                      Action::SELECTION_REPLACE);
            EXPECT_EQ(bindings.getActionForKey(ToolMode::SELECTION, KEY_DELETE, 0),
                      Action::DELETE_SELECTED);
            EXPECT_EQ(bindings.getActionForKey(ToolMode::SELECTION, KEY_F9, 0),
                      Action::TOGGLE_UI);
            ASSERT_TRUE(bindings.saveProfileToFile(path));
            const auto restored = nlohmann::json::parse(std::ifstream(path));
            EXPECT_EQ(std::ranges::count_if(restored["bindings"], [](const auto& entry) {
                          return entry.value("action", -1) ==
                                     static_cast<int>(Action::CAMERA_ORBIT) &&
                                 entry.value("mode", -1) !=
                                     static_cast<int>(ToolMode::GLOBAL);
                      }),
                      0);
            std::filesystem::remove(path);
        }
    } // namespace
} // namespace lfs::vis::input
