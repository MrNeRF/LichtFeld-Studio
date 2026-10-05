/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/logger.hpp"
#include "gui/rmlui/rmlui_vk_backend.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <mutex>
#include <string>
#include <vector>

class RenderInterfaceVKTestAccess {
public:
    static auto decode(const std::filesystem::path& path) {
        return RenderInterface_VK::DecodePreviewTexture(path, 256, true);
    }
};

namespace {

    class WarningCapture {
    public:
        WarningCapture()
            : token_(lfs::core::Logger::get().add_log_handler(
                  [this](const lfs::core::LogLevel level, const lfs::core::SourceSite&, const std::string_view message) {
                      if (level != lfs::core::LogLevel::Warn)
                          return;
                      std::scoped_lock lock(mutex_);
                      messages_.emplace_back(message);
                  })) {}
        ~WarningCapture() { lfs::core::Logger::get().remove_log_handler(token_); }

        [[nodiscard]] std::size_t count() const {
            std::scoped_lock lock(mutex_);
            return messages_.size();
        }

    private:
        lfs::core::LogHandlerToken token_;
        mutable std::mutex mutex_;
        std::vector<std::string> messages_;
    };

    class PreviewDecodeTest : public ::testing::Test {
    protected:
        void SetUp() override {
            directory_ = std::filesystem::temp_directory_path() /
                         ("lfs_preview_decode_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                          ::testing::UnitTest::GetInstance()->current_test_info()->name());
            std::filesystem::create_directories(directory_);
        }
        void TearDown() override { std::filesystem::remove_all(directory_); }

        std::filesystem::path directory_;
    };

    // Catches a catalog entry whose file was deleted flooding the log with a warning and stack on every start.
    TEST_F(PreviewDecodeTest, MissingProjectHasNoPreviewAndNoWarning) {
        WarningCapture warnings;
        const auto result = RenderInterfaceVKTestAccess::decode(directory_ / "deleted.licht");
        EXPECT_TRUE(result.pixels.empty());
        EXPECT_EQ(warnings.count(), 0u);
    }

    // Catches the missing-file case being silenced so broadly that unreadable projects go unreported.
    TEST_F(PreviewDecodeTest, DamagedProjectStillWarns) {
        const auto damaged = directory_ / "damaged.licht";
        std::ofstream(damaged, std::ios::binary) << "not a project";
        WarningCapture warnings;
        const auto result = RenderInterfaceVKTestAccess::decode(damaged);
        EXPECT_TRUE(result.pixels.empty());
        EXPECT_EQ(warnings.count(), 1u);
    }

} // namespace
