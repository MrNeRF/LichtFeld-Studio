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
        namespace fs = std::filesystem;

        struct NoProfilePersistence {
            bool previous = InputBindings::isPersistenceEnabled();
            NoProfilePersistence() { InputBindings::setPersistenceEnabled(false); }
            ~NoProfilePersistence() { InputBindings::setPersistenceEnabled(previous); }
        };

        class DamagedProfileWarning final : public lfs::NativeErrorConsumer {
        public:
            void on_error(const lfs::ErrorNotification& notification,
                          const lfs::ErrorDeliveryInfo&) noexcept override {
                if (notification.error.code() == lfs::ErrorCode::InvalidArgument &&
                    notification.error.domain() == lfs::ErrorDomain::App)
                    count.fetch_add(1, std::memory_order_relaxed);
            }
            std::atomic<int> count{0};
        };

        std::string readBytes(const fs::path& path) {
            std::ifstream input(path, std::ios::binary);
            return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        }

        TEST(KeymapDamagedProfileTest, KeepsOriginalBeforeReplacingDamagedFile) {
            NoProfilePersistence isolated;
            const auto dir = fs::temp_directory_path() / "lfs_damaged_keymap_guard";
            fs::remove_all(dir);
            fs::create_directories(dir);
            DamagedProfileWarning warnings;
            auto subscription = lfs::ErrorBus::instance().subscribe(warnings);
            for (const std::string_view damaged : {
                     R"({"name":"Default","version":29,"bindings":[)",
                     R"({"name":"Default","version":29,"bindings":[{"mode":0,"trigger_type":"key","key":74}]})",
                 }) {
                SCOPED_TRACE(std::string(damaged));
                const auto path = dir / "Default.json";
                {
                    std::ofstream output(path, std::ios::binary);
                    output << damaged;
                }
                InputBindings bindings;
                EXPECT_FALSE(bindings.loadProfileFromFile(path));
                ASSERT_TRUE(bindings.saveProfileToFile(path));
                EXPECT_NE(readBytes(path), damaged);
                int copies = 0;
                for (const auto& entry : fs::directory_iterator(dir)) {
                    if (entry.path().extension() == ".bak") {
                        EXPECT_EQ(readBytes(entry.path()), damaged);
                        ++copies;
                    }
                }
                EXPECT_GE(copies, 1);
                for (const auto& entry : fs::directory_iterator(dir)) {
                    if (entry.path().extension() == ".bak")
                        fs::remove(entry.path());
                }
            }
            EXPECT_EQ(warnings.count.load(std::memory_order_relaxed), 2);
            fs::remove_all(dir);
        }

        TEST(KeymapDamagedProfileTest, RefusesToOverwriteSymbolicLink) {
            NoProfilePersistence isolated;
            const auto dir = fs::temp_directory_path() / "lfs_keymap_symlink_guard";
            fs::remove_all(dir);
            fs::create_directories(dir);
            const auto target = dir / "sentinel.txt";
            {
                std::ofstream output(target);
                output << "sentinel";
            }
            const auto link = dir / "Default.json";
            std::error_code error;
            fs::create_symlink(target, link, error);
            if (error) {
                GTEST_SKIP() << error.message();
            }
            InputBindings bindings;
            EXPECT_FALSE(bindings.saveProfileToFile(link));
            EXPECT_EQ(readBytes(target), "sentinel");
            fs::remove_all(dir);
        }
    } // namespace
} // namespace lfs::vis::input
