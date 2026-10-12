/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error_bus.hpp"
#include "input/input_bindings.hpp"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <optional>

namespace lfs::vis::input {
    namespace {
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

        TEST(KeymapSafetyNetTest, WarnsThenRestoresOnlyUnreachableCoreActions) {
            struct RestorePersistence {
                bool enabled = InputBindings::isPersistenceEnabled();
                ~RestorePersistence() { InputBindings::setPersistenceEnabled(enabled); }
            } persistence;
            InputBindings::setPersistenceEnabled(false);
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
            std::filesystem::remove(path);
        }
    } // namespace
} // namespace lfs::vis::input
