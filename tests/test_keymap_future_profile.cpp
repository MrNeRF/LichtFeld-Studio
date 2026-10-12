/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error_bus.hpp"
#include "input/input_bindings.hpp"
#include <atomic>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>

namespace lfs::vis::input {
    namespace {
        class NewerProfileWarning final : public lfs::NativeErrorConsumer {
        public:
            void on_error(const lfs::ErrorNotification& notification,
                          const lfs::ErrorDeliveryInfo&) noexcept override {
                if (notification.error.code() == lfs::ErrorCode::Unsupported &&
                    notification.error.domain() == lfs::ErrorDomain::App)
                    count.fetch_add(1, std::memory_order_relaxed);
            }
            std::atomic<int> count{0};
        };

        TEST(KeymapFutureProfileTest, UnknownNewerProfileStaysUntouched) {
            struct RestorePersistence {
                bool enabled = InputBindings::isPersistenceEnabled();
                ~RestorePersistence() { InputBindings::setPersistenceEnabled(enabled); }
            } restore;
            InputBindings::setPersistenceEnabled(false);
            NewerProfileWarning warnings;
            auto warning_subscription = lfs::ErrorBus::instance().subscribe(warnings);
            namespace fs = std::filesystem;
            const fs::path path = fs::temp_directory_path() / "lfs_future_keymap_guard.json";
            const fs::path other = fs::temp_directory_path() / "lfs_future_keymap_export.json";
            constexpr std::string_view original = R"({"name":"From a newer build","version":999,
                "future_field":{"keep":true},"bindings":[{"mode":0,"action":987654,
                "description":"Future action","trigger_type":"key","key":83,
                "modifiers":0,"on_repeat":false}]})";
            {
                std::ofstream output(path);
                output << original;
            }

            InputBindings bindings;
            EXPECT_FALSE(bindings.loadProfileFromFile(path));
            EXPECT_EQ(bindings.getActionForDrag(ToolMode::SELECTION, MouseButton::MIDDLE, 0),
                      Action::CAMERA_ORBIT);
            EXPECT_FALSE(bindings.saveProfileToFile(path));
            EXPECT_TRUE(bindings.saveProfileToFile(other));
            EXPECT_GE(warnings.count.load(std::memory_order_relaxed), 1);
            std::ifstream input(path);
            const std::string actual{std::istreambuf_iterator<char>(input),
                                     std::istreambuf_iterator<char>()};
            EXPECT_EQ(actual, original);
            fs::remove(path);
            fs::remove(other);
        }
    } // namespace
} // namespace lfs::vis::input
